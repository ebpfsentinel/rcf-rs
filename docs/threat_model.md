# anomstream threat model

Scope: adversarial inputs on the ingress path where the
`anomstream` workspace (`anomstream-core` detectors +
`anomstream-hotpath` ingress primitives + `anomstream-triage` SOC
layer) consumes untrusted feature vectors (eBPF NDR agent, MSSP
tenant pool, public-facing API). Out of scope: host compromise of
the process running the workspace, side-channel attacks on Ed25519
license verification, upstream supply-chain compromises of rustc /
dependencies.

Referenced: MITRE ATLAS (Adversarial Threat Landscape for AI
Systems) tactic / technique IDs.

## T1 — Reservoir poisoning (`AML.T0020`)

### Attack

Attacker emits traffic whose feature vectors land in the
reservoir's retained sample. Once accepted as "baseline", the
attacker's traffic shape becomes the detector's normal, and
subsequent (real) anomalous traffic of the same shape is no
longer flagged. Two vectors:

1. **Deterministic admission spray** — when `UpdateSampler` uses
   the unkeyed `accept_hash(flow_hash)` path, admission is
   `flow_hash % keep_every_n == 0`. An attacker who can probe
   the admission decision (observe whether a given flow was
   reflected in the baseline via score drift) can spray 5-tuples
   whose caller-computed hash lands on the admitted residue class.
2. **Single-source flood** — even keyed admission admits a fixed
   rate per flow; a flood from a compromised source IP that
   churns through many 5-tuples saturates the reservoir with
   points all from the attacker's network.

### Defences shipped

- **Keyed sampler**: `UpdateSampler::new_keyed(keep_every_n)`
  seeds 128 bits of per-sampler secret from `getrandom` at
  construction, applies a murmur3-style keyed finaliser to every
  `accept_hash` input before the modulo. Attacker cannot steer
  their flow hash into the admitted residue class without learning
  the sampler secret; the secret never leaves the process. The
  paired `UpdateSampler::new_keyed_with_seeds(keep, k1, k2)`
  constructor accepts caller-supplied seeds for restricted
  environments where `getrandom` is unavailable (early-boot
  embedded, chroot without `/dev/urandom`,
  `wasm32-unknown-unknown`); production deployments should still
  source the seeds from a KMS / HSM-backed entropy stream.
- **Per-prefix rate cap**:
  `PrefixRateCap::new(NonZeroU32, NonZeroU64)` bounds how many
  admissions a single `/24`-prefix hash bucket can push within a
  rolling window. Fixed 256-bucket sketch — buckets are
  **cache-line padded** (`#[repr(align(64))]`) so concurrent
  `fetch_add`s on different buckets do not bounce a shared cache
  line through MOESI/MESI; lock-free `check_and_record`,
  `O(1)`. Soft over-admission window of ≈ `4 × cap_per_window`
  per bucket per window under heavy concurrent load (the
  `fetch_add` + cap-comparison sequence is two distinct atomics);
  size the cap with that slack baked in. Collisions are soft (by
  design — trades a little cross-prefix interference for constant
  memory).
- **LSH alert clusterer keyed hash**: `LshAlertClusterer::new`
  draws a fresh per-instance 128-bit hash secret from
  `getrandom` at construction so an offline-precomputed collision
  set against one clusterer cannot be replayed against another.
  `with_seed(buckets, attr_cap, seed)` exposes a deterministic
  variant for snapshot replay; production must rotate the seed
  per instance.
- **Trimmed-mean score aggregator**:
  `RandomCutForest::score_trimmed(&point, trim_fraction)` sorts
  per-tree scores, drops the top and bottom `trim_fraction`
  fraction, averages the middle. An attacker who manages to
  poison a minority of trees sees their contribution trimmed
  from the ensemble mean. Typical value: `0.10` (10 %/10 %).

### Defences NOT shipped

- No host-level rate limit on ingress (caller concern).
- No cryptographic integrity on the feature vector itself
  (caller must sanitise upstream).
- No per-tenant separation of reservoir secrets across
  `TenantForestPool` entries — all tenants share the keyed
  sampler's secret. Rotate on tenant provisioning if higher
  isolation is needed.

## T2 — Evasion via contextual shift (`AML.T0043`)

### Attack

Attacker gradually shifts a target feature's distribution so the
detector's baseline drifts in step. When the payload arrives, it
sits inside the (drifted) baseline and is not flagged.

### Defences shipped

- `MetaDriftDetector` (CUSUM on the score stream) fires
  `DriftKind::Upward` / `DriftKind::Downward` on sustained
  baseline drift. Observability only — caller decides action.
- `FeatureDriftDetector` (PSI + KL per feature) fires
  `DriftLevel::Alert` when the production distribution diverges
  from the frozen baseline by `≥ 0.25` PSI. Pin the offending
  dim via `argmax_psi()`.
- `score_codisp_stateless` preserves the frozen baseline across
  long eval streams; the mutating `score_codisp` path drifts by
  design — the non-mutating variant is the one to use when the
  caller needs a contextual-displacement score on a trusted
  baseline.

### Defences NOT shipped

- Automatic drift *recovery* (shadow-forest swap on alert). The
  ADWIN-based swap is P1 on the roadmap.

## T3 — Model extraction (`AML.T0024`)

### Attack

Attacker queries the detector through the score-exposing API
(via exposed decisions or via a side channel), reconstructs the
isolation-depth boundary, and uses it offline to design
undetectable payloads.

### Defences shipped

- No public score API out of the process boundary. The crate
  provides in-process `score()` only; exposing it externally is
  the caller's architectural decision.
- `score` returns a clamped `AnomalyScore` newtype — its
  internal representation is not exposed beyond `f64` accessor,
  making pre-clamp score leakage unlikely under `rustc` opt.

### Defences NOT shipped

- No differentially-private score perturbation
  (DP-SGD-equivalent for isolation forests is an open research
  problem; not in scope).

## T4 — Classifier-side resource exhaustion

### Attack

Attacker emits traffic at a rate high enough to overflow the
in-process MPSC `hot_path::channel` between the classifier and
the updater thread, starving legitimate updates.

### Defences shipped

- `UpdateProducer::try_enqueue` is non-blocking — on full queue
  it increments `dropped_total` and returns `false`. The
  classifier stays hot-path-safe; the cost is visible via the
  counter so ops can alert on `dropped_total > 0`.
- `UpdateSampler` drops low-value updates before the queue. A
  1/N stride or per-flow gate is free (no allocations, no
  syscalls).
- `update_channel(capacity)` validates `capacity ∈
  1..=MAX_CHANNEL_CAPACITY` (`1 << 20` slots) at construction —
  caller cannot OOM the allocator with `usize::MAX` or silently
  drop every offer with `0`. The non-panicking
  `try_update_channel` Result-returning variant surfaces the
  bound as `RcfError::InvalidConfig` for callers who want to
  handle it gracefully. `PrefixRateCap` now uses
  `NonZeroU32` / `NonZeroU64` typed parameters so the previous
  asymmetry (`window_ms == 0` panicked, `cap == 0` silently
  disabled the cap) is impossible to express.
- `FeedbackStore::new(capacity, ...)` rejects `capacity >
  MAX_CAPACITY = 65 536` — the bounded ledger can no longer be
  driven into allocator pressure by a hostile config. The
  per-cluster `contributing_tenants` rolodex on
  `AlertClusterer` is bounded by `MAX_TENANTS_PER_CLUSTER = 32`
  with FIFO eviction, so an attacker rotating synthetic tenant
  keys against a single cluster cannot grow the membership
  vector unboundedly. Per-call `MetricsSink` dispatch is
  **batched every 64 ops** (`record_batched` helper) — line-rate
  load no longer spends ≈160 ms / s on `Arc<dyn>` vtable calls;
  call `flush_metrics()` on shutdown.

### Defences NOT shipped

- No back-pressure signal from the updater to the sampler. If
  the updater falls behind, the queue drops silently until ops
  reacts to the `dropped_total` gauge.

## T5 — Audit-trail tampering

### Attack

Attacker (or compromised storage layer / SIEM operator) edits,
deletes, reorders, or splices `AlertRecord` entries at rest. A
naïve `serde`-roundtrip of the records on disk performs **no**
integrity check beyond the 4-byte version prefix, so a downstream
consumer that decodes the bytes has no way to tell a tampered
trail from the original. Compliance regimes (SOC2 CC6 / NIS2 /
PCI-DSS 10.5) generally require the audit trail itself —
not just the storage — to be tamper-evident.

### Defences shipped

- **HMAC-SHA256 audit chain** (`audit-integrity` feature):
  `AuditChain::new(key)` wraps a stream of `AlertRecord`s in
  `AuditChainEntry { record, seq, prev_tag, tag }` with
  `tag = HMAC-SHA256(key, u64_le(seq) || prev_tag ||
  postcard(record))`. Reordering breaks the next entry's
  `prev_tag`; editing a record breaks its own `tag`; deleting a
  record breaks the next entry's `prev_tag` linkage; forging an
  appended entry without the secret key is computationally
  infeasible. `verify_chain(entries, key, genesis_prev)` walks
  the chain end-to-end with constant-time tag comparisons (via
  the `subtle` crate) so a timing oracle cannot leak partial-tag
  matches. `AuditChain::with_genesis(key, prev_tag, seq)`
  resumes appending against persisted chain state.
- **Schema-drift defence**: `AlertRecord` (and its
  `AlertRecordShadow`) carry `#[serde(deny_unknown_fields)]` so a
  newer-producer / older-consumer skew cannot silently drop
  spliced fields. Schema bumps go through
  `ALERT_RECORD_VERSION` and fail loudly at decode time.
- **Persistence size cap**: `from_bytes` / `from_json` reject
  payloads above `MAX_DESERIALIZE_BYTES = 256 MiB` /
  `MAX_JSON_BYTES = 1 GiB` before any third-party decoder runs;
  `from_*_with_max_size` opts into a larger bound on a per-call
  basis for legitimate high-`D` deployments.

### Defences NOT shipped

- HMAC key management (rotation, KMS / HSM provisioning). The
  chain accepts any `≥ 32-byte` key and treats it opaquely; the
  caller integrates with their secrets store.
- Record encryption at rest. Records remain plaintext on disk;
  pair with at-rest encryption when the audit trail itself is
  sensitive (e.g. customer PII inside the `point` field).
- Cross-chain replay protection. Two chains with the same key
  and `genesis_prev` produce the same tags for the same
  records — rotate `genesis_prev` per chain (deriving from a
  chain identifier) when separation matters.

## What is explicitly NOT in the threat model

- Kernel-side eBPF verifier compromise (kernel concern).
- Compromise of the upstream CTI feeds that drive threat
  intelligence (out of scope — `anomstream-core` does not consume CTI
  directly).
- Attacks on the persistence format (`to_bytes` / `from_bytes`)
  — the crate enforces versioned envelopes with upfront
  rejection of incompatible versions, but a compromised
  serialised state file trivially compromises the loaded
  detector. Callers must treat forest snapshots as
  integrity-sensitive (sign + verify out-of-band).
