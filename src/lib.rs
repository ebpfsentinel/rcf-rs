//! Pure Rust Random Cut Forest for streaming anomaly detection.
//!
//! `rcf-rs` implements the Random Cut Forest (RCF) algorithm from Guha et al.
//! (ICML 2016) and is conformant with the
//! [AWS SageMaker RCF specification][aws-rcf]: reservoir sampling without
//! replacement (Park et al. 2004), random cuts weighted by per-dimension
//! range, anomaly score averaged across trees, and hyperparameter bounds
//! matching the AWS reference (`feature_dim`, `num_trees`,
//! `num_samples_per_tree`).
//!
//! # Architecture
//!
//! ```text
//! +-----------------+        +------------------+
//! |  ForestBuilder  |        |  RcfConfig       |
//! +--------+--------+        +---------+--------+
//!          |                           |
//!          v                           v
//!   +------+---------------------------+------+
//!   |          RandomCutForest                |   <- aggregate root
//!   |                                         |
//!   |   PointStore (ring buffer, repository)  |
//!   |                                         |
//!   |   trees: Vec<(RandomCutTree, Sampler)>  |
//!   +-+---------------------+-----------------+
//!     |                     |
//!     v                     v
//! +---+---------+   +-------+--------+
//! | RandomCutTr |   | ReservoirSampl |
//! +-------------+   +----------------+
//!     |
//!     v  via Visitor trait
//! +---+----------------+    +-----------------------+
//! | ScalarScoreVisitor |    | AttributionVisitor    |
//! +--------------------+    +-----------------------+
//! ```
//!
//! Modules are organised in layered fashion: `domain` (value objects),
//! `tree` (storage + cut tree), `sampler` (reservoir), `visitor` (scoring
//! strategies), and `forest` (aggregate root). The `persistence` module is
//! gated behind the `serde` feature.
//!
//! # Example
//!
//! ```ignore
//! // RandomCutForest is being implemented across stories RCF.2..RCF.7.
//! // The example below is the target API surface for v0.1.0.
//! use rcf_rs::ForestBuilder;
//!
//! let mut forest = ForestBuilder::new(4)
//!     .num_trees(100)
//!     .sample_size(256)
//!     .seed(42)
//!     .build()?;
//!
//! for point in stream_of_points {
//!     forest.update(point.clone())?;
//!     let score = forest.score(&point)?;
//!     if f64::from(score) > 1.5 {
//!         eprintln!("anomaly: {score}");
//!     }
//! }
//! # Ok::<(), rcf_rs::RcfError>(())
//! ```
//!
//! # Conformance
//!
//! `rcf-rs` enforces the AWS `SageMaker` hyperparameter bounds at build time:
//!
//! | Parameter | Range | Default |
//! |---|---|---|
//! | `feature_dim` | `[1, 10000]` | required |
//! | `num_trees` | `[50, 1000]` | `100` |
//! | `num_samples_per_tree` | `[1, 2048]` | `256` |
//!
//! See the crate's `README.md` for the full conformance matrix and the
//! comparison against `krcf` and the AWS Java port.
//!
//! # References
//!
//! 1. Sudipto Guha, Nina Mishra, Gourav Roy, Okke Schrijvers. "Robust Random
//!    Cut Forest Based Anomaly Detection on Streams." *International
//!    Conference on Machine Learning*, pp. 2712–2721. 2016.
//! 2. Byung-Hoon Park, George Ostrouchov, Nagiza F. Samatova, Al Geist.
//!    "Reservoir-based random sampling with replacement from data stream."
//!    *SIAM International Conference on Data Mining*, pp. 492–496. 2004.
//! 3. AWS `SageMaker` RCF reference.
//!
//! [aws-rcf]: https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

pub mod domain;
pub mod error;
pub mod tree;

pub use domain::{AnomalyScore, BoundingBox, Cut, DiVector, Point};
pub use error::{RcfError, RcfResult};
pub use tree::{Node, NodeRef, NodeStore};
