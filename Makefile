.PHONY: test bench fuzz fuzz-sanitize lint fmt clean build

all: build

build:
	cargo build

test:
	cargo test

bench:
	cargo bench

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
