//! Pools of detectors keyed by tenant / stream id.
//!
//! A single shared [`crate::ThresholdedForest`] cannot safely serve
//! traffic from multiple tenants — tenant A's baseline bleeds into
//! tenant B's threshold and vice versa. [`TenantForestPool`] keeps
//! one detector per tenant with an LRU eviction policy so the total
//! memory footprint stays bounded even with a long tail of one-off
//! tenants.

pub mod tenant_pool;

pub use tenant_pool::TenantForestPool;
