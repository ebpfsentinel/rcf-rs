# rcf-rs

Pure Rust Random Cut Forest for streaming anomaly detection.

Implements the RCF algorithm from Guha et al. (ICML 2016) and powers
the ML detection pipeline of the **eBPFsentinel Enterprise**
NDR agent.

## Scope

`rcf-rs` is a focused implementation of the 2016 paper, not an
attempt to match every feature of AWS's `randomcutforest-by-aws`
port. Feature additions are driven by what eBPFsentinel Enterprise
needs from its ML layer (streaming network anomaly detection, SOC
triage, multi-tenant deployments) — not by AWS parity. Features
well outside that scope (density estimation, forecasting,
shingling, GLAD variant, near-neighbour list, …) are intentionally
absent; the imputation concept is repurposed as a SOC-triage
`forensic_baseline` helper rather than a feature-completion
`impute()` call.

See [docs/features.md](docs/features.md) for the catalogue of
optional modules on top of the bare forest (TRCF, tenant pool,
bootstrap, warm reload, group scores, attribution stability,
forensic baseline, Platt calibrator, severity bands, alert
clustering, audit trail, CUSUM meta-drift, feature drift PSI/KL,
t-digest streaming quantiles, score histogram, bulk batch
scoring + cache-aware Morton reordering, timestamp retention,
early termination, probe-based codisp scoring (mutating + batched
+ stateless drift-free variant), fused score + attribution
single-walk, score confidence intervals, hot-path eBPF ingress
primitives (`UpdateSampler` + bounded MPSC channel for
classifier/updater thread split), metrics sink, …).

## Quickstart

```rust,ignore
use rcf_rs::ForestBuilder;

let mut forest = ForestBuilder::<4>::new()
    .num_trees(100)
    .sample_size(256)
    .seed(42)
    .build()?;

for point in stream_of_points {
    forest.update(point)?;
    let score = forest.score(&point)?;
    if f64::from(score) > 1.5 {
        eprintln!("anomaly: {score}");
    }
}
# Ok::<(), rcf_rs::RcfError>(())
```

## Algorithm

Guha, Mishra, Roy, Schrijvers — *Robust Random Cut Forest Based
Anomaly Detection on Streams*, ICML 2016.

Reservoir sampling without replacement: Park, Ostrouchov, Samatova,
Geist — SIAM SDM 2004.

AWS `SageMaker` hyperparameter bounds are enforced at build time
(`feature_dim`, `num_trees`, `num_samples_per_tree`, `time_decay`).
Details: [docs/conformance.md](docs/conformance.md).

## Features

| Cargo feature | Default | Role |
|---|---|---|
| `std` | ✅ | Standard library support |
| `parallel` | ✅ | Per-tree / batch parallelism via `rayon` |
| `serde` | ✅ | State serialisation |
| `postcard` | ✅ | Compact binary persistence (implies `serde`) |
| `serde_json` | ❌ | JSON persistence (implies `serde`) |

Opt out of the production profile via
`default-features = false` for embedded / mono-thread / no-persistence
scenarios.

## Performance

See [docs/performance.md](docs/performance.md) for the full
criterion bench matrix.

## License

[Apache-2.0](LICENSE). Contributions under the same licence.
