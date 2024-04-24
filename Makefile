.PHONY: test bench fuzz fuzz-sanitize lint fmt clean build

all: build

build:
	cargo build

test:
	cargo test --features with_serde

bench:
	cargo criterion --bench cardinality_estimator

bench-extended:
	N=1048576 cargo criterion --bench cardinality_estimator --message-format json | tee benches/bench_results_$$(date '+%Y%m%d_%H%M%S').json

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
