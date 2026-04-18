//! End-to-end Platt calibration on a warmed thresholded forest.
//!
//! Asserts:
//!
//! 1. A calibrator fit on synthetic labelled scores separates
//!    classes — low score → low P, high score → high P.
//! 2. Calibrator round-trips through serde bit-exact.
//! 3. Forest scores piped through `calibrate_many` all land in
//!    `[0, 1]`.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{
    ForestBuilder, PlattCalibrator, PlattFitConfig, RcfError,
};

#[test]
fn fit_then_calibrate_separates_classes() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut data: Vec<(f64, bool)> = Vec::new();
    // Baseline: scores ~0.5 (false)
    for _ in 0..200 {
        data.push((0.5 + rng.random::<f64>() * 0.1, false));
    }
    // Anomalies: scores ~3.0 (true)
    for _ in 0..50 {
        data.push((3.0 + rng.random::<f64>() * 0.2, true));
    }
    let cal = PlattCalibrator::fit(&data, PlattFitConfig::default()).unwrap();
    assert!(cal.converged());
    let p_baseline = cal.calibrate(0.5);
    let p_anomaly = cal.calibrate(3.0);
    assert!(p_baseline < 0.2, "baseline P too high: {p_baseline}");
    assert!(p_anomaly > 0.8, "anomaly P too low: {p_anomaly}");
}

#[test]
fn calibrate_on_forest_scores_stays_in_unit_interval() -> Result<(), RcfError> {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(2)
        .build()?;
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut calibration_set: Vec<(f64, bool)> = Vec::new();

    for _ in 0..256 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        f.update(p)?;
        let s = f64::from(f.score(&p)?);
        calibration_set.push((s, false));
    }
    // Plant outliers as positive labels.
    for _ in 0..32 {
        let p = [50.0_f64, 50.0, 50.0, 50.0];
        f.update(p)?;
        let s = f64::from(f.score(&p)?);
        calibration_set.push((s, true));
    }

    let cal = PlattCalibrator::fit(&calibration_set, PlattFitConfig::default()).unwrap();

    // Probe batch through calibrator.
    let scores: Vec<f64> = (0..64)
        .map(|_| f64::from(f.score(&[rng.random::<f64>() * 0.1, 0.0, 0.0, 0.0]).unwrap()))
        .collect();
    let probs = cal.calibrate_many(&scores);
    for p in probs {
        assert!((0.0..=1.0).contains(&p));
    }
    Ok(())
}

#[cfg(feature = "serde_json")]
#[test]
fn calibrator_serde_roundtrips() {
    let data: Vec<(f64, bool)> = (0..32)
        .map(|i| (f64::from(i) * 0.1, i >= 16))
        .collect();
    let cal = PlattCalibrator::fit(&data, PlattFitConfig::default()).unwrap();
    let json = serde_json::to_string(&cal).unwrap();
    let back: PlattCalibrator = serde_json::from_str(&json).unwrap();
    for s in [0.0_f64, 0.5, 1.0, 2.0, 5.0] {
        assert!((cal.calibrate(s) - back.calibrate(s)).abs() < 1e-12);
    }
}
