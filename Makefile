.PHONY: test test-allocations bench bench-extended fuzz fuzz-sanitize lint fmt clean build doc

all: build

build:
	cargo build

test:
	cargo test --features with_serde

test-allocations:
	cargo test test_allocations -- --show-output

bench:
	cargo criterion --bench cardinality_estimator

BENCH_RESULTS_PATH := target/bench_results_$(shell date '+%Y%m%d_%H%M%S').json

bench-extended:
	RUSTFLAGS="-C target-cpu=native" N=1048576 cargo criterion --bench cardinality_estimator --message-format json | tee $(FILENAME)
	python3 benches/analyze.py $(FILENAME)

fuzz:
	cargo +nightly fuzz run fuzz_target_estimator -- -max_len=65536

fuzz-sanitize:
	RUSTFLAGS="-Z sanitizer=address" cargo +nightly fuzz run fuzz_target_estimator -- -max_len=65536

lint:
	cargo clippy -- -D warnings

fmt:
	cargo fmt --all

clean:
	cargo clean

doc:
	cargo doc
