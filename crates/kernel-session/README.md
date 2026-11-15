# kernel-session — authenticated encrypted sessions over kernel-transport

A thin integration layer, not a second crypto implementation.

All cryptography lives in `kernel-carrier`: the SIGMA handshake
(`Initiator` / `Responder`), sealed sessions (`Session`:
XChaCha20-Poly1305, 64-entry replay window, rekeying), and pinned
peer keys (`PinSet`). This crate only carries those frames over a
`kernel_transport::Connection` (length-prefixed TCP frames) and
presents a small session API on top.

## API

- `connect(conn, my_key, peer_pub, pins)` — initiator side:
  Hello, Response, Confirm.
- `accept(conn, my_key, pins)` — responder side.
- `SecureSession::send` / `recv` — sealed payloads both ways.
- `start_rekey` / `complete_rekey` / `rekey_if_due` —
  initiator-driven rekeying; payloads sent while a rekey is pending
  are queued and delivered under the new keys, in order.

One `Connection` message carries exactly one KCAR frame (at most
65,548 bytes, far under the transport's 64 MiB cap).

The integration point is the handshake: `Session::new` is
`pub(crate)` in the carrier, so sessions are created only by a
completed handshake.

## Honest limits

- **No flood protection.** There is no cookie challenge or
  proof-of-work before the responder's Diffie-Hellman, so Hello
  floods cost the responder one DH and one signature each. The
  carrier's own `transport` module adds that protection; this layer,
  riding on `kernel-transport`, does not. The handshake itself is
  still authenticated (both sides must have the other's key pinned).
- **Keys, not identities.** `connect` / `accept` take a reference
  to the caller's Ed25519 `SigningKey`, exactly as the carrier's own
  transport does. The key is used only during the handshake and is
  never stored. `kernel_identity::Identity` deliberately does not
  export its secret key, so a node holding its identity only as an
  `Identity` cannot use this layer yet. Bridging that is future
  work, not a reason to copy the handshake here.

## Verification

```sh
cargo test -p kernel-session   # 5 passed, 0 failed
```

Includes a man-in-the-middle test: a tampered Data frame on the
wire is refused with `AuthFailed`.
