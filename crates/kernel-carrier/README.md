# kernel-carrier — Phase 2 working copy (implemented by Muse)

Built 2026-11-14 ~21:36 CST on Tristan's order ("Make em", "Start
implementing all code"). The files as sent live byte-for-byte in
`../kernel-carrier-phase0/` — this copy is the only place that was edited.

## What was missing, and what was written here

| Missing piece | Implemented in |
| --- | --- |
| Phase 2 `frame.rs` — `Kind::Rekey = 5`, `Kind::Cookie = 6` | `src/frame.rs` (+ 2 tests) |
| Phase 2 `error.rs` — `BadCookie`, `RekeyPending`, `QueueFull` | `src/error.rs` (+ 1 test) |
| Phase 2 `Cargo.toml` — `hmac = "0.12"` dep | `Cargo.toml` |
| PinSet persistence (Phase 1 claim in lib.rs) | `src/pins.rs` (+ 7 tests) |

PinSet persistence format (binary, no JSON — KERNEL law):

```text
magic   "KCPIN1" (6 bytes)
pinned  count u32 LE, then that many 32-byte keys
revoked count u32 LE, then that many 32-byte keys
```

`PinSet::save(path)` writes a sibling `.tmp` file and renames it over the
target, so a crash mid-write never leaves a half-written pin file.
`PinSet::load(path)` refuses bad magic, truncation, or trailing bytes.
Disk errors surface as `CarrierError::Internal`. `encode()`/`decode()` are
the in-memory core, tested directly.

Everything else in this copy is the recovered code, unchanged: handshake
(SIGMA), sealed sessions with rekeying, TCP transport with the cookie
challenge, flood guard, gateway table, voice (MockCodec + jitter buffer).

## Verification

```text
cargo test — 106 passed, 0 failed
```

96 recovered tests + 10 new tests (frame 2, error 1, pins 7). Includes real
TCP handshakes on loopback, rekey over TCP, and a voice call over TCP.

`cargo clippy --all-targets` reports 45 warnings, all style: 43 are
`.err().expect()` in tests (the recovered code's house test style, followed
by the new tests too) and 2 are pre-existing in the recovered lib code.
Zero errors.

## Still not in this crate (per lib.rs)

Real audio codec (Opus), radio trait, multi-hop routing, discovery,
time-based rekey trigger, identity hiding, the carrier design doc.

## Ownership

Carrier is Claude's track per the KERNEL packet — this is completion of the
recovered files, not a parallel carrier. NOT merged into the from-scratch
rebuild (`~/workspace/KERNEL-staging/`) and NOT pushed to any repo.
Destination for these files is Tristan's call.
