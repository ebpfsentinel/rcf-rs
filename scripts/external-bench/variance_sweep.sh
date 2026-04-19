#!/usr/bin/env bash
# Multi-seed variance measurement. Runs each impl 5 times with
# different CSV seeds (matching gen_points.py), extracts
# inserts/s, scores/s, AUC, reports mean ± stddev.
#
# Usage:  ./variance_sweep.sh /tmp/aws-rcf-jar
set -euo pipefail

JAR="${1:-/tmp/aws-rcf/randomcutforest-core-4.4.0.jar}"
SEEDS=(2026 2027 2028 2029 2030)

cd "$(dirname "$0")/../.."

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

for seed in "${SEEDS[@]}"; do
    python3 scripts/external-bench/gen_points.py \
        --n 10000 --dim 16 --seed "$seed" > "$tmpdir/data-$seed.csv"
done

run_rcf_rs() {
    local seed=$1
    cargo run --release --example external_bench_driver -- \
        "$tmpdir/data-$seed.csv" 100 256 2>&1 | tail -8
}

run_rrcf() {
    local seed=$1
    python3 scripts/external-bench/bench_rrcf.py \
        --input "$tmpdir/data-$seed.csv" --trees 100 --sample 256 2>&1 | tail -8
}

run_sklearn() {
    local seed=$1
    python3 scripts/external-bench/bench_sklearn_iforest.py \
        --input "$tmpdir/data-$seed.csv" --trees 100 --train-frac 0.3 2>&1 | tail -8
}

run_aws_java() {
    local seed=$1
    if [[ ! -f "$JAR" ]]; then
        echo "  SKIP (jar not at $JAR)"
        return
    fi
    if [[ ! -f scripts/external-bench/java-driver/RcfBench.class ]]; then
        javac -cp "$JAR" scripts/external-bench/java-driver/RcfBench.java
    fi
    java -cp "scripts/external-bench/java-driver:$JAR" RcfBench \
        "$tmpdir/data-$seed.csv" 100 256 2>&1 | tail -8
}

for impl in rcf_rs rrcf sklearn aws_java; do
    echo "=== $impl ==="
    for seed in "${SEEDS[@]}"; do
        echo "--- seed=$seed ---"
        case "$impl" in
            rcf_rs) run_rcf_rs "$seed" ;;
            rrcf) run_rrcf "$seed" ;;
            sklearn) run_sklearn "$seed" ;;
            aws_java) run_aws_java "$seed" ;;
        esac
    done
done
