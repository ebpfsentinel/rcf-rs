# rcf-rs

Pure Rust Random Cut Forest for streaming anomaly detection.

`rcf-rs` implements the Random Cut Forest algorithm from Guha et al.
(ICML 2016) and is conformant with the
[AWS SageMaker RCF specification](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html):
reservoir sampling without replacement, random cuts weighted by per-dimension
range, anomaly score averaged across trees, hyperparameter bounds matching
the AWS reference (`feature_dim`, `num_trees`, `num_samples_per_tree`).

> **Status**: under active development — APIs are unstable until v0.1.0.
> Track the [`rcf-rs` epic](../_bmad-output/implementation-artifacts/epic-rcf-rs.md)
> for progress.

## Quickstart

```rust,ignore
use rcf_rs::{ForestBuilder, AnomalyScore};

// Build a forest with the AWS-default hyperparameters.
let mut forest = ForestBuilder::new(/* dimension */ 4)
    .num_trees(100)
    .sample_size(256)
    .seed(42)
    .build()?;

// Stream points through the forest.
for point in stream_of_points {
    forest.update(point.clone())?;
    let score: AnomalyScore = forest.score(&point)?;
    if f64::from(score) > 1.5 {
        println!("anomaly: score={score}");
    }
}
# Ok::<(), rcf_rs::RcfError>(())
```

## Algorithm

`rcf-rs` follows the original paper:

> Sudipto Guha, Nina Mishra, Gourav Roy, Okke Schrijvers.
> "Robust Random Cut Forest Based Anomaly Detection on Streams."
> *International Conference on Machine Learning*, pp. 2712–2721. 2016.

Reservoir sampling without replacement is from:

> Byung-Hoon Park, George Ostrouchov, Nagiza F. Samatova, Al Geist.
> "Reservoir-based random sampling with replacement from data stream."
> *SIAM International Conference on Data Mining*, pp. 492–496. 2004.

## AWS SageMaker conformance

| AWS specification | `rcf-rs` mapping |
|---|---|
| `feature_dim ∈ [1, 10000]` | enforced by `ForestBuilder` |
| `num_trees ∈ [50, 1000]`, default `100` | enforced by `ForestBuilder` |
| `num_samples_per_tree ∈ [1, 2048]`, default `256` | enforced by `ForestBuilder` |
| Reservoir sampling without replacement | `sampler::ReservoirSampler` |
| Score = average across trees | `forest::RandomCutForest::score` |
| Anomaly threshold `≥ 3σ` from mean | caller responsibility |

Out of scope for v0.1.0:

- Shingling for 1D time series (consumer can pre-shingle the input)
- AWS `eval_metrics` (accuracy / precision-recall) — caller owns labels

## Memory and throughput

Targets for the default configuration (100 trees × 256 samples × 16 dims):

- Memory ≤ 4 MB
- Insert throughput ≥ 100k points/sec on a typical x86_64 dev box
- Score throughput ≥ 50k points/sec on the same box

Bench numbers will be published with the v0.1.0 release.

## Cargo features

| Feature | Default | Effect |
|---|---|---|
| `std` | ✅ | Standard library support (future `no_std` planned) |
| `serde` | ❌ | Forest state serialisation |
| `serde_json` | ❌ | JSON helpers (implies `serde`) |

## Minimum Supported Rust Version

`rcf-rs` requires Rust **1.93** or later, edition 2024.

## License

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE).

Contributions submitted to this repository are licensed under the same terms,
without any additional terms or conditions.
