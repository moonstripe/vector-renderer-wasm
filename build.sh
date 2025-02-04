RUSTFLAGS=--cfg=web_sys_unstable_apis cargo build --target wasm32-unknown-unknown --release
wasm-bindgen --no-typescript --target web --out-dir ./wasm/ --out-name "vector-renderer" ./target/wasm32-unknown-unknown/release/vector_renderer_wasm.wasm
