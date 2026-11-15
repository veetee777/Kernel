# KERNEL — Foundation (from-scratch rebuild)

Sovereign peer-to-peer infrastructure. A protocol, not a company-dependent
hosting platform. Owner: Tristan Dene Bell.

This is a from-scratch foundation rebuild. It lives at
https://github.com/veetee777/Kernel — the owner's repo, not attached to
any assistant. It is not the canonical workspace from the other session;
that code is still locked behind that account's paywall.

## What exists

A Cargo workspace (`resolver = "2"`) with thirty-eight crates. All compile
with no warnings, and all tests pass (`cargo test` from this directory:
652 tests, 0 failed).

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
  `Node::open` persists blocks to disk (see `kernel-store-disk`) and
  the identity in `identity.kid` (see `kernel-identity-disk`), so a
  restarted node keeps its PeerId and its blocks; `net` adds real
  TCP block exchange between machines.
- **`kernel-store-disk`** — Persistent content-addressed store: one
  append-only binary log (`blocks.klog`), `KBLK1` records, fsync per
  put, hash-verified on open and on read. A torn final record from a
  crash is cut on replay; earlier blocks survive. No JSON, no SQLite.
  `DiskStore::compact()` rewrites the log to one record per CID
  (atomic temp-file + rename + fsync, crash-safe). `maybe_compact`
  and `set_auto_compact` trigger it when the log passes a size
  threshold (and only when a duplicate exists to drop).

**Networking and distribution (new)**
- **`kernel-transport`** — Real TCP transport (std only): length-prefixed
  framed messages, 64 MiB frame cap, clean errors on refused
  connections and peer disconnects. Loopback-tested.
- **`kernel-dht`** — Kademlia-style DHT, in-memory. Node IDs are the
  SHA-256 of a peer's public key; k-buckets (K=20) with LRU eviction;
  `find_node` by XOR distance; values stored with the storing peer's
  signature, verified at store time and on read. A missing key returns
  the closest nodes, not an error. `Dht::learn_discovered` feeds
  verified discovery records into the routing table.
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
- **`kernel-carrier`** — The carrier cell (mobile network layer),
  Phases 0–2, recovered from the other session and completed here.
  SIGMA handshake (Hello/Response/Confirm, pinned keys only, forward
  secrecy), sealed sessions (XChaCha20-Poly1305, 64-entry replay
  window, rekeying), KCAR binary framing, TCP transport with a cookie
  challenge + proof-of-work flood guard, a pinned-peer gateway table,
  and voice (MockCodec + jitter buffer; no real codec yet). PinSet
  persists to a binary `KCPIN1` file (atomic save). 112 tests. Not
  yet: real audio codec (Opus), radio trait, multi-hop routing,
  identity hiding.
- **`kernel-identity-disk`** — Identity persistence: save/load an
  Ed25519 `Identity` to a binary `KIDSK1` file, 0600 permissions,
  atomic save, optional passphrase wrap (HKDF + ChaCha20-Poly1305).
  Corrupt, truncated, or tampered files are refused, never silently
  loaded. A restarted node can now keep its identity instead of
  getting a fresh PeerId.
- **`kernel-discovery`** — Peer discovery: signed `KPEER1` records
  (identity, address, timestamp, signature) gossiped over TCP.
  Bad signatures are refused and never stored; stale records never
  replace newer ones. 3-node convergence verified over loopback TCP.
  Feeds into the DHT via `Dht::learn_discovered`. Records expire:
  `PeerTable::prune` drops records older than a TTL (default 1h in
  `serve`), and `insert_with_ttl` refuses already-expired records.
- **`kernel-session`** — Authenticated encrypted sessions as a thin
  integration over the carrier: runs the carrier handshake/session
  over a `kernel-transport` connection. No crypto duplicated here.
  Hello-flood protection: a per-IP `HandshakeLimiter` refuses
  excess attempts before any DH work (success refunds budget);
  the carrier's own transport has the stronger cookie challenge.
- **`kernel-nat`** — NAT traversal: minimal STUN (RFC 5389 binding
  request/response, XOR-MAPPED-ADDRESS), UDP hole punching with a
  rendezvous registry, and a relay fallback for peers that cannot
  punch through. Address learning classifies the NAT (full cone /
  restricted / symmetric). All send paths use connected UDP sockets.
  Loopback-tested only — a real NAT crossing is still unproven.
- **`kernel-swap`** — Atomic swaps: hash-time-locked contracts
  (HTLC) over a local in-memory ledger. Lock / claim with preimage /
  refund after timeout, with funds conserved exactly in every path;
  a two-party cross-ledger swap flow is tested. No real
  Bitcoin/Ethereum interop — no chain client, no on-chain finality.
- **`kernel-contracts`** — Smart contracts: a deterministic
  stack-based bytecode VM (12 opcodes) with gas metering, isolated
  per-contract storage, deploy-time validation, and atomic calls
  (writes commit only on HALT). No WASM/EVM, no cross-contract
  calls, no on-disk state.
- **`kernel-webui`** — Node web UI: std-only HTTP server. `GET /`
  status page, `GET /api/status`, `GET /api/blocks`,
  `POST /api/store`, `GET /api/fetch?cid=...`. Bearer-token auth
  on every route (caller-supplied token, constant-time compare,
  401 with an empty body). No TLS — bind to loopback or put TLS
  in front.
- **`kernel-routing`** — Multi-hop source routing: binary `KRTE1`
  envelopes (destination + remaining route + TTL + payload),
  operator-populated next-hop tables, TTL loop protection, local
  delivery without forwarding. No route discovery — routes are
  supplied, not found; forwarding is plaintext at this layer.

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
  `playback` is a real engine: decodes 16-bit PCM WAV, retunes to
  432 Hz, streams in chunks with a sample clock and underrun
  counting, and writes to an `AudioSink` — a `WavFileSink` is
  included, so playback is proven end-to-end to a file. No audio
  device sink yet (no ALSA/AAudio backend).
- **`kernel-vision` (Kernel•Vision)** — Signed video on chunked
  storage. A video is a `KVIDEO1` manifest (title, duration, width,
  height, chunk root, owner, signature). `put_video`/`get_video`
  verify the manifest signature and every chunk hash;
  `get_chunk` returns one hash-verified chunk by index, so a player
  can fetch the chunk it needs without reassembling the whole video.
  `playback` is a real frame pipeline: `KVFR` raw YUV420 streams
  decode, convert to RGB24 (BT.601), scale in software, and present
  on a frame clock to a `FrameSink` — a `PpmSink` is included, so
  playback is proven end-to-end to files. No compressed codec
  (H.264/AV1/VP9), no display sink, no audio sync.
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
  error. Nothing here records, transcribes, or plays audio. HEY•AI
  is also the render engine: a small deterministic template language
  (`{{name}}` substitution, `{{#each}}` loops, `{{#if}}`
  conditionals) over an in-crate value type — no JSON, no execution.
  Rendering is content-addressed (same template + data = same bytes
  = same address) and a `RenderQueue` runs jobs in order, recording
  each result. It renders data; it never runs code.
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
- **`kernel-sim`** — The Korn world: a deterministic tick-based grid
  simulation (64×64, plains/forest/water/mountain), agents that
  gather, eat, rest, and starve by simple rules, resources that
  regrow, an append-only event log with replay, binary save/load
  (`KSIM1`). Same seed + same ticks = the exact same world. Not
  wired to Planet-K or any other crate yet.
- **`kernel-scripture`** — Bible study and research pipeline:
  ingest `Book Chapter:Verse<TAB>text`, verse references with
  ranges, concordance, word/phrase search, signed study notes,
  related-verses research queries. Verses are content-addressed.
  No real Bible text is bundled — the caller supplies it.
- **`kernel-artifacts`** — Manuscripts and artifacts pipeline:
  records with content CIDs, signed provenance chains (verified in
  order on every read), variant links with cycle refusal, enforced
  stages intake → verify → catalog → publish. Proves integrity and
  chain order, not truth.
- **`k-store`** — KERNEL-native storefront: signed listings priced
  in X•Koin, product pages rendered through the HEY•AI engine,
  orders settled through K•Pay (seller invoices, buyer pays, stock
  decrements; failures leave balances and stock untouched). Not
  Shopify — Tristan already has a Shopify store, and that
  integration is deferred until the 13 apps have real code.

All 17 apps in the working inventory are now present: 16 as Rust
crates in this workspace, plus Glassforge, which is the separate
Android launcher project, not a Rust crate.

## What does NOT exist yet

- NAT traversal exists (`kernel-nat`: STUN, hole punching, relay)
  but is loopback-tested only — outbound UDP is blocked in this
  VM (a real STUN query to Google's servers times out), so no
  public STUN server, no public rendezvous, no real-network
  crossing is proven. The DHT is in-memory; replication and talk
  use explicit addresses, not DHT-discovered peers, except through
  `Dht::learn_discovered`.
- Persistence: `kernel-store-disk` exists (append-only log,
  `compact()`, size-triggered auto-compact); identity persistence
  exists and `Node::open` loads it.
- The carrier's remaining pieces: real audio codec (Opus), radio
  trait, multi-hop routing (the routing crate exists; the carrier
  itself has no multi-hop yet), identity hiding.
- Atomic swaps are HTLC over a local ledger — no real chain
  interop. Smart contracts are a small custom VM — no WASM/EVM,
  no cross-contract calls, no on-disk state. Playback is proven to
  file sinks — no device audio sink, no compressed video codec,
  no display sink. The web UI has bearer-token auth but no TLS.
  Routing has no route discovery.
- Grants, docs, or any deployment.

## Project laws (enforced by design, not just words)

- AGPL-3.0-or-later. See `LICENSE`.
- No JSON-on-disk, no Postgres, no SQLite, no S3, no Supabase,
  no Railway for KERNEL storage.
- Storage is 100% free: local files only, no paid service, no
  per-read or per-write charges, ever. Nothing in this workspace
  phones home to a billing meter.
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
