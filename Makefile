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

bench-extended: export RUSTFLAGS = -C target-cpu=native
bench-extended: export N = 1048576
bench-extended: export BENCH_RESULTS_PATH = target/bench_results/$(shell date '+%Y%m%d_%H%M%S')
bench-extended:
	mkdir -p $(BENCH_RESULTS_PATH)
	cargo criterion --bench cardinality_estimator --message-format json | tee $(BENCH_RESULTS_PATH)/results.json
	python3 benches/analyze.py

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
