//! `kernel-node`: run a KERNEL node from the command line.
//!
//! Usage: `kernel-node [display-name]`
//!
//! Creates a fresh node identity, prints it, stores one block, reads it
//! back, and exits. A smoke test that the node runs on real hardware
//! (including an Android phone). Nothing is served on the network yet:
//! the transport in the library is in-process and test-only.

use kernel_node::Node;
use std::process::ExitCode;

fn main() -> ExitCode {
    let name = std::env::args().nth(1).unwrap_or_else(|| "tristan".to_string());
    let mut node = match Node::new(&name) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("kernel-node: cannot create node: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("identity: {}", node.public_identity().encode());
    let cid = match node.store_local(b"hello kernel") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("kernel-node: store failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("stored block: {cid}");
    match node.get_local(&cid) {
        Ok(data) => {
            println!("fetched block: {}", String::from_utf8_lossy(&data));
            println!("kernel-node: ok");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kernel-node: fetch failed: {e}");
            ExitCode::FAILURE
        }
    }
}
