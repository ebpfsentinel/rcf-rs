# NAB — detection-quality benchmark

The Numenta Anomaly Benchmark (Apache 2.0) is the canonical
public dataset for streaming anomaly detection.

Three runners use identical protocol so their AUCs sit side by
side:

- `tests/nab.rs` — `#[ignore]` integration test, rcf-rs side.
- `bench_rrcf_nab.py` — rrcf 0.4.4.
- `RcfBenchNab.java` — AWS `randomcutforest-java` 4.4.0 (see
  `../README.md` for the Maven Central
  jar).

## Fetch the dataset

```bash
# ~50 MB, one-shot clone.
git clone --depth 1 https://github.com/numenta/NAB.git /opt/nab
```

Layout after clone:

```
/opt/nab/
  data/realKnownCause/*.csv
  labels/combined_windows.json
```

## Running

```bash
# rcf-rs.
RCF_NAB_PATH=/opt/nab \
    cargo test --test nab --all-features -- --ignored --nocapture

# rrcf.
python3 scripts/nab/bench_rrcf_nab.py --nab /opt/nab

# AWS Java.
JAR=/tmp/aws-rcf/randomcutforest-core-4.4.0.jar
javac -cp "$JAR" scripts/nab/RcfBenchNab.java
java -cp "scripts/nab:$JAR" RcfBenchNab /opt/nab
```

## Scoring protocol

- **Feature engineering**: 32-lag temporal embedding
  (`[v_{t-31}, … v_t]`) → `D = 32`. Longer context absorbs NAB's
  wide contextual-shift anomalies. RCF on raw scalars loses most
  of its value; lag features give the tree cuts meaningful axes.
- **Per-dim z-score** against the warm-phase mean / stddev —
  NAB series mix wildly different scales (CPU %, taxi counts,
  temperatures) and RCF cut sampling is range-weighted.
- **Two-phase**: warm on the first 15 % of each series, then
  score the rest against the frozen forest — no `update` on
  eval rows (NAB anomaly windows are days-wide; folding anomaly
  points back into the reservoir drags the baseline toward
  them and tanks recall).
- **EMA smoothing** on the raw score stream (α = 0.02, half-life
  ~35 steps).
- **Labels**: timestamp comparison against
  `combined_windows.json` `[start, end]` pairs. A row is
  labelled anomalous iff its timestamp falls inside *any*
  window.
- **AUC**: trapezoidal rule on the ROC curve; per-file +
  weighted aggregate (weighted by number of anomalous rows).

## Measured numbers (i7-1370P, 100 trees × 256 samples)

Weighted aggregate AUC on the `realKnownCause` subset (7 files):

| Impl | Aggregate AUC |
|---|---|
| rcf-rs `score_codisp()` | **0.776** |
| AWS Java 4.4.0 | 0.757 |
| rrcf 0.4.4 | 0.748 |
| rcf-rs `score()` | 0.719 |

Per-file breakdown:

| File | rcf-rs `score()` | rcf-rs `score_codisp()` | rrcf | AWS Java |
|---|---|---|---|---|
| `ambient_temperature_system_failure` | 0.813 | **0.813** | 0.734 | 0.786 |
| `cpu_utilization_asg_misconfiguration` | **0.953** | 0.939 | 0.849 | 0.906 |
| `ec2_request_latency_system_failure` | 0.709 | **0.739** | 0.481 | 0.482 |
| `machine_temperature_system_failure` | 0.578 | 0.666 | 0.880 | **0.883** |
| `nyc_taxi` | 0.698 | **0.721** | 0.571 | 0.540 |
| `rogue_agent_key_hold` | 0.145 | **0.692** | 0.535 | 0.633 |
| `rogue_agent_key_updown` | 0.633 | **0.721** | 0.657 | 0.542 |

rcf-rs ships two scoring APIs: the fast `score()` path
(isolation depth, rayon-parallel, non-mutating — eBPF-hot-path
friendly) and the heavier `score_codisp()` path (probe-based,
mutating, sequential per tree — ~30× slower). rrcf + AWS Java
use probe-based scoring by default; `score_codisp()` matches
their semantic and leads the aggregate.

The rcf-rs ignored test pins an aggregate-AUC floor of `0.70`
as a regression guard.
