.PHONY: test bench bench-extended fuzz-estimator fuzz-serde lint fmt clean build doc

all: build

build:
	cargo build

test:
	cargo test --features with_serde

bench: export RUSTFLAGS = -C target-cpu=native
bench: export N = 1048576
bench: export BENCH_RESULTS_PATH = target/bench_results/$(shell date '+%Y%m%d_%H%M%S')
bench:
	mkdir -p $(BENCH_RESULTS_PATH)
	cargo criterion --bench cardinality_estimator --message-format json | tee $(BENCH_RESULTS_PATH)/results.json
	python3 benches/analyze.py

fuzz-estimator:
	RUSTFLAGS="-Z sanitizer=address" cargo +nightly fuzz run estimator -- -max_len=65536

fuzz-serde:
	RUSTFLAGS="-Z sanitizer=address" cargo +nightly fuzz run serde -- -max_len=65536

fuzz-serde-json-array:
	RUSTFLAGS="-Z sanitizer=address" cargo +nightly fuzz run serde_json_array -- -max_len=65536

fuzz-serde-postcard:
	RUSTFLAGS="-Z sanitizer=address" cargo +nightly fuzz run serde_postcard -- -max_len=65536

fuzz-serde-roundtrip:
	RUSTFLAGS="-Z sanitizer=address" cargo +nightly fuzz run serde_roundtrip -- -max_len=65536

lint:
	cargo clippy --features with_serde -- -D warnings

fmt:
	cargo fmt --all

clean:
	cargo clean

doc:
	cargo doc
