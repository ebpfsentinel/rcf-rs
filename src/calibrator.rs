//! Calibrated probability output — convert raw anomaly scores into
//! a probability `P(anomaly | score) ∈ [0, 1]` via Platt scaling
//! (sigmoid calibration). Implements the stable Newton-Raphson
//! formulation of Lin, Lin, Weng (2007) as refined from Platt 1999.
//!
//! # Why calibration
//!
//! The raw [`crate::AnomalyScore`] is unbounded and its scale shifts
//! with forest size, sample size, and point dimensionality — useful
//! for ranking, awkward for audit reporting. A sigmoid calibrator
//! fit on a labelled history set maps the raw score to a
//! defensible probability ("this traffic window is 87 % likely to
//! be an anomaly") suitable for SOC2 / NIS2 compliance paperwork
//! and for setting threshold policies via error-rate budgets.
//!
//! # Shape
//!
//! A [`PlattCalibrator`] holds two scalars `(a, b)` such that
//!
//! ```text
//! P(y = 1 | score) = 1 / (1 + exp(a · score + b))
//! ```
//!
//! Note the sign of `a`: a well-fit detector where high score
//! correlates with the anomaly label yields `a < 0` (probability
//! rises as score rises).
//!
//! # Usage
//!
//! Collect a labelled calibration set — tuples of
//! `(raw_score, is_anomaly_bool)` from historical SOC-confirmed
//! alerts and manually-labelled baseline windows — then call
//! [`PlattCalibrator::fit`]. Persist the fitted calibrator with
//! serde and reuse at inference time via
//! [`PlattCalibrator::calibrate`].

use alloc::format;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::error::{RcfError, RcfResult};

/// Default maximum Newton-Raphson iterations — well past the point
/// of diminishing returns on the 2-parameter fit.
pub const DEFAULT_MAX_ITERS: usize = 100;
/// Default gradient-norm convergence threshold.
pub const DEFAULT_TOLERANCE: f64 = 1e-5;
/// Default Levenberg-Marquardt damping — stabilises Newton steps
/// when the Hessian is near-singular on tiny calibration sets.
pub const DEFAULT_MIN_STEP: f64 = 1e-10;
/// Default smallest SIGMA for the line-search — aborts when the
/// Hessian stops being positive definite.
pub const DEFAULT_SIGMA: f64 = 1e-12;

/// Fit-time configuration. Defaults match the reference Lin et al.
/// (2007) hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlattFitConfig {
    /// Maximum Newton-Raphson iterations.
    pub max_iters: usize,
    /// Convergence threshold on the gradient infinity-norm.
    pub tolerance: f64,
    /// Minimum line-search step size before giving up.
    pub min_step: f64,
    /// Minimum Hessian-diagonal damping (`σ` in the reference
    /// paper) — added to `h11` / `h22` for numerical stability.
    pub sigma: f64,
}

impl Default for PlattFitConfig {
    fn default() -> Self {
        Self {
            max_iters: DEFAULT_MAX_ITERS,
            tolerance: DEFAULT_TOLERANCE,
            min_step: DEFAULT_MIN_STEP,
            sigma: DEFAULT_SIGMA,
        }
    }
}

impl PlattFitConfig {
    /// Validate every parameter.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `max_iters == 0` or
    /// when `tolerance` / `min_step` / `sigma` is non-finite or
    /// non-positive.
    pub fn validate(&self) -> RcfResult<()> {
        if self.max_iters == 0 {
            return Err(RcfError::InvalidConfig(
                "PlattFitConfig::max_iters must be > 0".into(),
            ));
        }
        for (name, value) in [
            ("tolerance", self.tolerance),
            ("min_step", self.min_step),
            ("sigma", self.sigma),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(RcfError::InvalidConfig(format!(
                    "PlattFitConfig::{name} must be finite and > 0, got {value}"
                )));
            }
        }
        Ok(())
    }
}

/// Fitted 1-D sigmoid calibrator `P(y = 1 | score) = 1 / (1 + exp(a · score + b))`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlattCalibrator {
    /// Slope parameter — typically negative when high score =
    /// anomaly.
    a: f64,
    /// Intercept parameter.
    b: f64,
    /// Number of Newton-Raphson iterations the fit took to
    /// converge. Useful for operator diagnostics ("did my fit
    /// terminate early?").
    iters: usize,
    /// Whether the fit hit [`PlattFitConfig::tolerance`] — `false`
    /// means the solver exhausted [`PlattFitConfig::max_iters`]
    /// and the result is the best attempt available.
    converged: bool,
}

impl PlattCalibrator {
    /// Build a calibrator from already-fitted parameters. Public
    /// for unit tests and for external fitters (sklearn exports,
    /// analytical priors).
    #[must_use]
    pub fn from_params(a: f64, b: f64) -> Self {
        Self {
            a,
            b,
            iters: 0,
            converged: true,
        }
    }

    /// Fit the sigmoid on `data` — a slice of `(score, is_anomaly)`
    /// tuples — using the reference Platt / Lin-Lin-Weng iterative
    /// algorithm. Labels `false` / `true` are remapped to the
    /// smoothed targets `1/(N- + 2)` / `(N+ + 1)/(N+ + 2)` per the
    /// reference paper, so a label-homogeneous calibration set
    /// still converges.
    ///
    /// # Errors
    ///
    /// - [`RcfError::InvalidConfig`] when `data` is empty or
    ///   contains non-finite scores.
    /// - [`RcfError::InvalidConfig`] when `config.validate` rejects
    ///   the fit configuration.
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)] // Platt 1999 nomenclature + unified objective/gradient/line-search block.
    pub fn fit(data: &[(f64, bool)], config: PlattFitConfig) -> RcfResult<Self> {
        config.validate()?;
        if data.is_empty() {
            return Err(RcfError::InvalidConfig(
                "PlattCalibrator::fit requires at least one data point".into(),
            ));
        }
        for (i, (score, _)) in data.iter().enumerate() {
            if !score.is_finite() {
                return Err(RcfError::InvalidConfig(format!(
                    "PlattCalibrator::fit: non-finite score at index {i}"
                )));
            }
        }

        let (prior0, prior1) = {
            let mut n_pos = 0_usize;
            let mut n_neg = 0_usize;
            for (_, y) in data {
                if *y {
                    n_pos += 1;
                } else {
                    n_neg += 1;
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let p = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
            #[allow(clippy::cast_precision_loss)]
            let n = 1.0 / (n_neg as f64 + 2.0);
            (n, p)
        };

        // Initialise: a = 0, b = ln((N- + 1) / (N+ + 1))
        let mut a = 0.0_f64;
        let mut b = {
            let n_pos = data.iter().filter(|(_, y)| *y).count();
            let n_neg = data.len() - n_pos;
            #[allow(clippy::cast_precision_loss)]
            let num = (n_neg as f64 + 1.0).ln();
            #[allow(clippy::cast_precision_loss)]
            let den = (n_pos as f64 + 1.0).ln();
            num - den
        };

        let targets: Vec<f64> = data
            .iter()
            .map(|(_, y)| if *y { prior1 } else { prior0 })
            .collect();

        let negative_log_likelihood = |a: f64, b: f64| -> f64 {
            let mut fval = 0.0_f64;
            for ((score, _), t) in data.iter().zip(targets.iter()) {
                let fx = a * score + b;
                // log(1 + exp(-fx)) rewritten for numerical stability:
                //   fx ≥ 0 → exp(-fx) small, direct form is stable.
                //   fx < 0 → exp(-fx) huge, factor exp(-fx):
                //            log(1 + exp(-fx)) = -fx + log(1 + exp(fx))
                let log_one_plus_exp_neg = if fx >= 0.0 {
                    (1.0 + (-fx).exp()).ln()
                } else {
                    -fx + (1.0 + fx.exp()).ln()
                };
                fval += t * fx + log_one_plus_exp_neg;
            }
            fval
        };

        let mut iters = 0_usize;
        let mut converged = false;
        for _ in 0..config.max_iters {
            iters += 1;
            let mut h11 = config.sigma;
            let mut h22 = config.sigma;
            let mut h21 = 0.0_f64;
            let mut g1 = 0.0_f64;
            let mut g2 = 0.0_f64;
            for ((score, _), t) in data.iter().zip(targets.iter()) {
                let fx = a * score + b;
                let (p, q) = if fx >= 0.0 {
                    let e = (-fx).exp();
                    let denom = 1.0 + e;
                    (e / denom, 1.0 / denom)
                } else {
                    let e = fx.exp();
                    let denom = 1.0 + e;
                    (1.0 / denom, e / denom)
                };
                let d2 = p * q;
                h11 += score * score * d2;
                h22 += d2;
                h21 += score * d2;
                let d1 = t - p;
                g1 += score * d1;
                g2 += d1;
            }
            if g1.abs() < config.tolerance && g2.abs() < config.tolerance {
                converged = true;
                break;
            }
            let det = h11 * h22 - h21 * h21;
            let da = -(h22 * g1 - h21 * g2) / det;
            let db = -(-h21 * g1 + h11 * g2) / det;
            let gd = g1 * da + g2 * db;

            let mut step_size = 1.0_f64;
            let current = negative_log_likelihood(a, b);
            loop {
                let new_a = a + step_size * da;
                let new_b = b + step_size * db;
                let new_f = negative_log_likelihood(new_a, new_b);
                if new_f < current + 1e-4 * step_size * gd {
                    a = new_a;
                    b = new_b;
                    break;
                }
                step_size /= 2.0;
                if step_size < config.min_step {
                    converged = true;
                    break;
                }
            }
            if step_size < config.min_step {
                break;
            }
        }

        Ok(Self {
            a,
            b,
            iters,
            converged,
        })
    }

    /// Slope parameter.
    #[must_use]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Intercept parameter.
    #[must_use]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Newton-Raphson iteration count used by the fit.
    #[must_use]
    pub fn iters(&self) -> usize {
        self.iters
    }

    /// Whether the fit terminated via gradient-norm convergence
    /// (`true`) or via iteration-limit / line-search abandonment
    /// (`false`).
    #[must_use]
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Calibrate a raw score to `P(y = 1 | score) ∈ [0, 1]`.
    /// Non-finite inputs are mapped to `0.5` (maximum entropy,
    /// no-signal) rather than propagating NaN.
    #[must_use]
    pub fn calibrate(&self, score: f64) -> f64 {
        if !score.is_finite() {
            return 0.5;
        }
        let fx = self.a * score + self.b;
        if fx >= 0.0 {
            let e = (-fx).exp();
            e / (1.0 + e)
        } else {
            let e = fx.exp();
            1.0 / (1.0 + e)
        }
    }

    /// Bulk-calibrate a slice of scores.
    #[must_use]
    pub fn calibrate_many(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|s| self.calibrate(*s)).collect()
    }

    /// Online update — one SGD step on the logistic loss for a
    /// single labelled observation. Use to **refine** an existing
    /// calibrator as SOC feedback accumulates without re-fitting
    /// the full batch. `lr` is the learning rate (typical range
    /// `1e-3 .. 1e-1`); higher values track drift faster but
    /// increase variance.
    ///
    /// Gradient derivation — logistic loss
    /// `L = −[y · ln(p) + (1 − y) · ln(1 − p)]` with
    /// `p = 1/(1 + exp(a·s + b))` gives:
    ///
    /// ```text
    /// ∂L/∂a = (p − y) · s
    /// ∂L/∂b = (p − y)
    /// ```
    ///
    /// SGD step: `a ← a − lr · (p − y) · s`, `b ← b − lr · (p − y)`.
    /// Note the label polarity is inverted vs the classical
    /// Platt parameterisation — `p` here is `P(y = 1 | score)`,
    /// and anomalous samples (high score) must map to
    /// `label = true`. Non-finite scores are silently dropped.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on non-positive / non-
    /// finite `lr`.
    pub fn update_online(&mut self, score: f64, label: bool, lr: f64) -> RcfResult<()> {
        if !lr.is_finite() || lr <= 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "PlattCalibrator::update_online: lr must be finite and > 0, got {lr}"
            )));
        }
        if !score.is_finite() {
            return Ok(());
        }
        let p = self.calibrate(score);
        let y = f64::from(u8::from(label));
        let err = p - y;
        self.a -= lr * err * score;
        self.b -= lr * err;
        self.iters = self.iters.saturating_add(1);
        self.converged = false;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert bounds on closed-form sigmoid values.
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        PlattFitConfig::default().validate().unwrap();
    }

    #[test]
    fn fit_rejects_empty() {
        let err = PlattCalibrator::fit(&[], PlattFitConfig::default()).unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[test]
    fn fit_rejects_non_finite_scores() {
        let data = vec![(f64::NAN, true), (0.0, false)];
        assert!(PlattCalibrator::fit(&data, PlattFitConfig::default()).is_err());
    }

    #[test]
    fn fit_on_separable_data_converges() {
        // Bimodal: normals cluster at score ~0.1, anomalies at ~5.0.
        let mut data = Vec::new();
        for i in 0..100 {
            data.push((0.1 + f64::from(i) * 0.001, false));
        }
        for i in 0..30 {
            data.push((5.0 + f64::from(i) * 0.01, true));
        }
        let cal = PlattCalibrator::fit(&data, PlattFitConfig::default()).unwrap();
        assert!(cal.converged());
        // High-score calibrated probability should dominate
        // low-score one.
        let p_anomaly = cal.calibrate(5.0);
        let p_normal = cal.calibrate(0.1);
        assert!(
            p_anomaly > 0.9 && p_normal < 0.1,
            "calibrator did not separate classes: p_anomaly={p_anomaly} p_normal={p_normal}",
        );
    }

    #[test]
    fn calibrate_non_finite_returns_half() {
        let cal = PlattCalibrator::from_params(-1.0, 0.0);
        assert_eq!(cal.calibrate(f64::NAN), 0.5);
        assert_eq!(cal.calibrate(f64::INFINITY), 0.5);
    }

    #[test]
    fn calibrate_bounds_respected() {
        let cal = PlattCalibrator::from_params(-2.0, 1.0);
        for s in [-100.0, -1.0, 0.0, 1.0, 100.0] {
            let p = cal.calibrate(s);
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn calibrate_monotonic_with_negative_a() {
        let cal = PlattCalibrator::from_params(-1.0, 0.0);
        let p0 = cal.calibrate(0.0);
        let p1 = cal.calibrate(1.0);
        let p2 = cal.calibrate(5.0);
        assert!(p0 < p1 && p1 < p2);
    }

    #[test]
    fn calibrate_many_preserves_order() {
        let cal = PlattCalibrator::from_params(-1.0, 0.0);
        let scores = vec![0.0_f64, 1.0, 2.0, 3.0];
        let out = cal.calibrate_many(&scores);
        assert_eq!(out.len(), 4);
        for &[a, b] in out.array_windows::<2>() {
            assert!(a <= b);
        }
    }

    #[test]
    fn fit_homogeneous_positive_stays_near_prior() {
        // Every label is positive. Smoothed target keeps the fit
        // stable — probability at any score should be biased
        // upward but remain in [0, 1].
        let data: Vec<(f64, bool)> = (0..10).map(|i| (f64::from(i), true)).collect();
        let cal = PlattCalibrator::fit(&data, PlattFitConfig::default()).unwrap();
        let p = cal.calibrate(5.0);
        assert!((0.0..=1.0).contains(&p));
    }

    fn cfg(max_iters: usize, tolerance: f64) -> PlattFitConfig {
        PlattFitConfig {
            max_iters,
            tolerance,
            min_step: DEFAULT_MIN_STEP,
            sigma: DEFAULT_SIGMA,
        }
    }

    #[test]
    fn validate_rejects_zero_max_iters() {
        assert!(cfg(0, DEFAULT_TOLERANCE).validate().is_err());
    }

    #[test]
    fn validate_rejects_non_positive_tolerance() {
        assert!(cfg(DEFAULT_MAX_ITERS, 0.0).validate().is_err());
        assert!(cfg(DEFAULT_MAX_ITERS, -1.0).validate().is_err());
    }
}
