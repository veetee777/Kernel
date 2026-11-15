# KERNEL — Foundation (from-scratch rebuild)

Sovereign peer-to-peer infrastructure. A protocol, not a company-dependent
hosting platform. Owner: Tristan Dene Bell.

This is a from-scratch foundation rebuild. It lives at
https://github.com/veetee777/Kernel — the owner's repo, not attached to
any assistant. It is not the canonical workspace from the other session;
that code is still locked behind that account's paywall.

## What exists

A Cargo workspace (`resolver = "2"`) with twenty-four crates. All compile
with no warnings, and all tests pass (`cargo test` from this directory:
237 tests, 0 failed).

**Foundation**
- **`kernel-crypto`** — Ed25519 signing, X25519 key exchange,
  ChaCha20-Poly1305 AEAD, SHA-256, HKDF-SHA256. Secrets zeroize on drop.
  Deps: curve25519-dalek 4, ed25519-dalek 2, getrandom 0.2, hkdf 0.12,
  chacha20poly1305 0.10, sha2 0.10, zeroize 1.
- **`kernel-identity`** — Peer identity on top of Ed25519: a human
  display name (e.g. `tristan`) plus a short fingerprint (`a1b2-c3d4`).
  Public identities serialize as `KID1 <name> <hex public key>`.
  Secret keys never leave the `Identity` that owns them.
- **`kernel-storage`** — Content-addressed store. `put(bytes) -> CID`
  (SHA-256 hex), `get`, `has`. In-memory only — no disk persistence as
  system of record. Chunked storage splits large content into 256 KiB
  chunks under a manifest root CID; every chunk is hash-verified on read,
  so corrupted chunks fail loudly.
- **`kernel-node`** — A node wires an identity and a content store
  together. In-process transport (channels) lets two nodes exchange
  blocks by CID in tests. A node serves only blocks it holds, and a
  request for content it was never given returns a clean error.

**Networking and distribution (new)**
- **`kernel-transport`** — Real TCP transport (std only): length-prefixed
  framed messages, 64 MiB frame cap, clean errors on refused
  connections and peer disconnects. Loopback-tested.
- **`kernel-dht`** — Kademlia-style DHT, in-memory. Node IDs are the
  SHA-256 of a peer's public key; k-buckets (K=20) with LRU eviction;
  `find_node` by XOR distance; values stored with the storing peer's
  signature, verified at store time and on read. A missing key returns
  the closest nodes, not an error.
- **`kernel-replication`** — Distributed storage replication over TCP.
  Replica nodes serve STORE/FETCH; `replicate` copies a block to N
  peers (bytes are hash-verified against the CID on receipt, so lying
  peers are refused); `fetch` tries known holders in order and takes
  the first that answers, so one peer going offline does not break
  reads. Holder tracking included.
- **`kernel-sync`** — Offline-first sync: versioned, signed records
  (`id`, version, payload, owner, signature). Two stores converge by
  exchanging records; highest version wins, ties broken by signature
  comparison (deterministic); bad signatures are rejected; sync is
  idempotent.

**Apps (new)**
- **`k-files` (K•Files)** — Named, signed files on chunked storage.
  A file is a `KFILE1` manifest (name, chunk root, owner, signature)
  stored as a block; `put_file`/`get_file` verify the manifest
  signature and every chunk hash; `list_files`; `delete_file` removes
  the manifest and chunks no other file references.
- **`k-talk` (K•Talk)** — Signed direct messages over TCP. A message
  carries sender, recipient, text, timestamp, a per-sender sequence
  number (orders a sender's messages across connections), and the
  sender's signature. Bad signatures and wrong-recipient messages are
  rejected before storage; signatures are verified again on read.
- **`k-photos` (K•Photos)** — Named, signed photos on chunked
  storage. A photo is a `KPHOTO1` manifest (name, chunk root, owner,
  signature, plus width, height, format, taken timestamp) stored as a
  block; `put_photo`/`get_photo` verify the manifest signature and
  every chunk hash; `list_photos`; `delete_photo` removes the
  manifest and chunks no other photo references.
- **`k-docs` (K•Docs)** — Documents with signed version history. A
  document has a stable human-readable id (slugged title plus a short
  hash), a title, and a chain of `KDOCV1` version records (id, title,
  version, chunk root, owner, signature). `create_doc` is idempotent
  for the same title and first-version content; `update_doc` adds a
  version (owner only, versions increase monotonically);
  `get_doc`/`get_version` verify signatures and chunk hashes, so old
  versions read exactly what was written. `apply_version` accepts
  signed records from peers (bad signatures rejected); content blocks
  travel separately via `export_blocks`/`import_blocks`, which refuse
  blocks that do not hash to their claimed CID.
- **`x-shield` (X•Shield)** — Node security, default-deny and
  deterministic. Allow list and block list of peers (human names
  attached; block beats allow); signed `XSHIELD1` attestations let an
  allow-listed peer vouch for another peer (bad attestations rejected
  outright); a per-peer sliding-window rate limiter over explicit
  timestamps (no wall-clock sleeps in tests). `check`/`check_node`
  return `Allowed`, `Blocked` (with a human reason), or
  `RateLimited`.
- **`k-chain` (K•Chain)** — A minimal blockchain. Transactions are
  signed (`KTX1`: sender, recipient, memo, amount); blocks carry
  index, previous hash, timestamp, transactions, and a SHA-256 block
  hash over their contents. `genesis`, `add_block` (refuses bad
  transactions before building), `validate_chain` (genesis shape,
  links, hashes, signatures — tampered transactions, broken links,
  and tampered hashes each fail with a distinct error), fork choice
  is longest valid chain wins (ties go to the first chain). Blocks
  are stored as blocks in the content store and retrievable by CID.

**Payments (new)**
- **`x-koin` (X•Koin)** — The coin: an account-based ledger on
  signed transactions (`XKOIN1`: sender, recipient, memo, amount).
  Balances start at a genesis allocation and change only through
  validated transactions, so the total supply is conserved.
  `submit` validates signature, amount > 0, sender balance, and
  replay (each transaction id is accepted exactly once, so no
  double-spend); `balance_of`, `total_supply`, `holders`.
- **`k-pay` (K•PAY)** — Signed invoices and receipts settled on
  X•Koin. An invoice (human-readable id, recipient, amount, memo,
  expiry, recipient signature) is paid by submitting a signed coin
  transaction and returning a receipt signed by the payer, which
  anyone can verify without seeing the ledger. Paying an expired
  invoice or paying twice is rejected; insufficient funds are
  refused by the ledger and nothing moves.
- **`kernel-xchange` (Kernel•XChange)** — A signed order book.
  Orders (human-readable id, owner, buy/sell, price, amount,
  owner signature) match when buy price >= sell price, executing
  at the sell price, with partial fills and deterministic
  price-then-time priority. Only the owner can cancel an order.
- **`kash-korner` (Kash•Korner)** — A simple signed store /
  marketplace settled on X•Koin. A listing (human-readable id,
  seller, item name, description, price, seller signature) is
  bought by a signed coin transaction from buyer to seller; the
  listing is then marked sold and cannot be bought again.
  Insufficient funds and bad listing signatures are refused.

**Apps (final batch — all 17 apps now present)**
- **`kernelodys` (Kernelodys)** — Signed music tracks and playlists
  on chunked storage. A track is a `KTRACK1` manifest (title, artist,
  duration, chunk root, owner, signature); a playlist is a signed
  `KPLAY1` record (name, ordered track manifest CIDs, owner,
  signature). `put_track`/`get_track` verify the manifest signature
  and every chunk hash; `create_playlist` refuses track CIDs this
  node does not hold; `list_tracks`/`list_playlists` sort by name.
  Nothing here plays audio — storage, verification, listing only.
- **`kernel-vision` (Kernel•Vision)** — Signed video on chunked
  storage. A video is a `KVIDEO1` manifest (title, duration, width,
  height, chunk root, owner, signature). `put_video`/`get_video`
  verify the manifest signature and every chunk hash;
  `get_chunk` returns one hash-verified chunk by index, so a player
  can fetch the chunk it needs without reassembling the whole video.
  Nothing here decodes or plays video.
- **`k-ai` (K•AI)** — An AI task/response **framework, not a model**.
  A task is a signed `KAITSK1` record (id, owner, prompt, signature);
  a provider is a trait (`name`, `run`); a response is a signed
  `KAIRESP` record (task id, provider name, response text, provider
  signature). `submit_task`/`run_task`/`get_response` verify
  signatures and that a response names a task this node holds.
  Ships one provider, `TestProvider`, a local deterministic test
  provider that returns the prompt uppercased with a `KAI-TEST:`
  prefix. It is clearly labeled a test provider, not intelligence.
- **`hey-ai` (HEY•AI)** — Voice. A voice note is a signed `KVOICE1`
  manifest (name, duration, chunk root, owner, signature) on chunked
  storage. A voice command is a signed `KCMDVC1` record (text, owner,
  signature) parsed by a small fixed router into an action: `Play`
  (with track name), `Pause`, `Next`, `Previous`, `ListTracks`, or
  `Unknown`. Unknown text maps to `Unknown` — never a panic, never an
  error. Nothing here records, transcribes, or plays audio.
- **`planet-k` (Planet•K)** — Social. A post is a signed `KPOST1`
  record (author, text, timestamp, optional media chunk root,
  signature); a follow is a signed `KFOL1` record (follower,
  followed, signature). `feed` returns posts from followed peers,
  newest first, verifying each post's signature on read. Following
  yourself is refused; following is idempotent.
- **`x-koms` (X•Kom's)** — The owner's personal tool. A note is a
  signed `KNOTE1` manifest (title, chunked body, owner, signature).
  A task is a signed `XKTASK1` record (id, title, done flag, owner,
  signature); `toggle_task` flips the flag and re-signs (owner only;
  the old record stays as history). A quick command is a signed
  `XKOMCMD` record (name, command string, owner, signature);
  `run_command` only appends an execution entry (name, timestamp) to
  a log — **it executes no shell command, ever**.

All 17 apps in the working inventory are now present: 16 as Rust
crates in this workspace, plus Glassforge, which is the separate
Android launcher project, not a Rust crate.

## What does NOT exist yet

- Real peer discovery and NAT traversal (the DHT is in-memory and has
  no network peer exchange yet; replication and talk use explicit
  addresses, not DHT-discovered peers).
- Persistence: no on-disk store is designed yet, by intent.
- The carrier (mobile network) work.
- Atomic swaps, smart contracts, identity beyond peer keys.
- Grants, docs, or any deployment.

## Project laws (enforced by design, not just words)

- AGPL-3.0-or-later. See `LICENSE`.
- No JSON-on-disk, no Postgres, no SQLite, no S3, no Supabase,
  no Railway for KERNEL storage.
- No facial recognition.
- Names for humans, not hashes, wherever a human sees an identifier.
- Preserve existing work; consolidate, don't scatter partial
  implementations. This foundation is one workspace, one place.

## Build and test

```sh
cargo test
```

Requires a Rust toolchain (edition 2021). First run downloads the
crates listed above from crates.io.
