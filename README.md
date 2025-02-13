# MNIST on Cloudflare Workers
This can be a proper "hello world" to get used to WASM + ML (with `burn`) on Cloudflare Workers.

Previously I had been trying to compile a binary to WASM and instantiate it in my JS worker. That brings a few headaches if you have quite a few imports in your code. Turns out you can use `worker-rs` to run the entire thing in Rust.
Basically, instead of bringing the Rust code to JS with WASM, we can bring the "Workers" bindigns to Rust.



