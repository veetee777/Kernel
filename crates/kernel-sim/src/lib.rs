//! The Korn world — a deterministic tick-based world simulation for KERNEL.
//!
//! A 64x64 grid with four terrain types, named agents that move, gather,
//! eat and rest, resources that deplete and regrow, and an append-only
//! event log. The world is a **pure function of (seed, tick count)**:
//!
//! * No wall clock anywhere. Time is a tick counter.
//! * No system RNG. All randomness comes from a small xorshift PRNG
//!   ([`rng::XorShift64Star`]) seeded from the world seed, and its state
//!   is saved with the world so a loaded world continues exactly.
//! * Every decision is made in a fixed order (agents in index order,
//!   cells row-major), so the same seed + the same ticks always gives
//!   the exact same world, byte for byte.
//!
//! ## Rules (simple, explicit, documented)
//!
//! Terrain:
//! * **Plains** — passable, movement cost 1. May hold food.
//! * **Forest** — passable, movement cost 2. May hold wood.
//! * **Water** — impassable, holds nothing.
//! * **Mountain** — impassable. May hold stone.
//!
//! Each tick, every living agent takes exactly one action, in this
//! priority order:
//!
//! 1. **Rest or starve** — if energy < 15: with food, rest (consume 1
//!    food, +30 energy). Without food, starve (-10 health; at 0 health
//!    the agent dies).
//! 2. **Eat** — else if energy < 50 and the agent carries food: consume
//!    1 food, +20 energy.
//! 3. **Gather** — else if standing on a cell with a resource: take 1
//!    unit of it (food, wood or stone) into the agent's inventory, -1
//!    energy. The cell's amount drops by 1; a depleted cell regrows
//!    slowly (see below).
//! 4. **Move** — else step one cell toward the nearest visible resource
//!    (within 8 cells, Chebyshev distance), paying the destination
//!    terrain's movement cost. If no resource is visible, wander one
//!    step in a PRNG-chosen direction. Impassable cells are never
//!    entered; a blocked agent stays put.
//!
//! Resources regrow after each tick: a depleted cell regains 1 unit
//! every [`REGROW_TICKS`] ticks, up to the amount it started with.
//!
//! Every action is recorded in an append-only in-memory event log with
//! its tick number. [`World::replay`] re-simulates from the seed and
//! checks the log against the re-simulation, so a log can be verified
//! and the world state reconstructed from it.
//!
//! ## Save format (binary, no JSON)
//!
//! `KSIM1` magic + version + seed + tick + PRNG state + terrain grid +
//! resource state + agents + event log + FNV-1a checksum over all
//! preceding bytes. [`World::load`] refuses bad magic, wrong version,
//! truncation, trailing bytes, a bad checksum, or any out-of-range
//! field — always as a clean [`SimError`], never a panic.
//!
//! ## What this is not
//!
//! A grid simulation with simple rules. No graphics, no networking,
//! no real-time clock. Agents are rule-followers, not AI. The world is
//! not connected to Planet-K or any other crate yet — wiring is a later
//! step.

mod rng;

use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{Cid, ContentStore};
use std::fmt;

pub use rng::XorShift64Star;

/// World width in cells.
pub const WORLD_WIDTH: usize = 64;
/// World height in cells.
pub const WORLD_HEIGHT: usize = 64;
/// How far an agent can see resources (Chebyshev distance).
pub const SIGHT_RADIUS: i32 = 8;
/// Ticks between one unit of regrowth on a depleted cell.
pub const REGROW_TICKS: u16 = 40;
/// Maximum energy and health.
pub const MAX_VITALS: u32 = 100;
/// Magic bytes at the start of every save.
pub const SAVE_MAGIC: &[u8; 5] = b"KSIM1";
/// Save format version.
pub const SAVE_VERSION: u8 = 1;

/// Errors returned by simulation operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimError {
    /// An agent name is empty or uses characters the identity layer
    /// does not allow (letters, digits, '-', '_' and '.').
    InvalidAgentName(String),
    /// Two agents were given the same name.
    DuplicateAgentName(String),
    /// More than 255 agents were requested (agent indices are u8).
    TooManyAgents(usize),
    /// No passable cell exists to spawn an agent on.
    NoSpawnCell,
    /// A save failed validation: bad magic, wrong version, truncation,
    /// trailing bytes, bad checksum, or an out-of-range field.
    CorruptSave(String),
    /// A replayed event log does not match a fresh re-simulation.
    ReplayMismatch(String),
    /// An identity operation failed (key derivation or name validation).
    Identity(IdentityError),
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimError::InvalidAgentName(n) => write!(f, "invalid agent name: {n}"),
            SimError::DuplicateAgentName(n) => write!(f, "duplicate agent name: {n}"),
            SimError::TooManyAgents(n) => write!(f, "too many agents: {n} (max 255)"),
            SimError::NoSpawnCell => write!(f, "no passable cell to spawn an agent on"),
            SimError::CorruptSave(m) => write!(f, "corrupt save: {m}"),
            SimError::ReplayMismatch(m) => write!(f, "replay mismatch: {m}"),
            SimError::Identity(e) => write!(f, "identity error: {e}"),
        }
    }
}

impl std::error::Error for SimError {}

impl From<IdentityError> for SimError {
    fn from(e: IdentityError) -> Self {
        SimError::Identity(e)
    }
}

/// Terrain of one grid cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Terrain {
    Plains,
    Forest,
    Water,
    Mountain,
}

impl Terrain {
    /// Water and mountain cannot be entered.
    pub fn passable(self) -> bool {
        matches!(self, Terrain::Plains | Terrain::Forest)
    }

    /// Energy cost of stepping onto a cell of this terrain.
    pub fn move_cost(self) -> u32 {
        match self {
            Terrain::Plains => 1,
            Terrain::Forest => 2,
            Terrain::Water | Terrain::Mountain => u32::MAX,
        }
    }

    fn encode(self) -> u8 {
        match self {
            Terrain::Plains => 0,
            Terrain::Forest => 1,
            Terrain::Water => 2,
            Terrain::Mountain => 3,
        }
    }

    fn decode(v: u8) -> Option<Self> {
        match v {
            0 => Some(Terrain::Plains),
            1 => Some(Terrain::Forest),
            2 => Some(Terrain::Water),
            3 => Some(Terrain::Mountain),
            _ => None,
        }
    }
}

/// A kind of gatherable resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Food,
    Wood,
    Stone,
}

impl ResourceKind {
    fn encode(self) -> u8 {
        match self {
            ResourceKind::Food => 1,
            ResourceKind::Wood => 2,
            ResourceKind::Stone => 3,
        }
    }

    fn decode(v: u8) -> Option<Self> {
        match v {
            1 => Some(ResourceKind::Food),
            2 => Some(ResourceKind::Wood),
            3 => Some(ResourceKind::Stone),
            _ => None,
        }
    }
}

/// One grid cell: terrain plus its resource state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cell {
    /// Terrain of the cell.
    pub terrain: Terrain,
    /// Resource present on the cell, if any.
    pub resource: Option<ResourceKind>,
    /// Units of the resource left.
    pub amount: u8,
    /// Units the cell started with; regrowth never exceeds this.
    pub max_amount: u8,
    /// Ticks until the next unit regrows (0 when nothing is regrowing).
    pub regrow_in: u16,
}

/// A named agent living in the world.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Agent {
    /// Public identity, derived deterministically from (seed, index,
    /// name) — the same world always gives the same agent the same id.
    pub id: PeerId,
    /// Human-readable name.
    pub name: String,
    /// Grid column.
    pub x: u8,
    /// Grid row.
    pub y: u8,
    /// Carried food units (eating and resting consume these).
    pub food: u32,
    /// Carried wood units.
    pub wood: u32,
    /// Carried stone units.
    pub stone: u32,
    /// 0..=100. Rest and eating restore it; acting spends it.
    pub energy: u32,
    /// 0..=100. Starving reduces it; at 0 the agent dies.
    pub health: u32,
    /// False once the agent has died; dead agents take no actions.
    pub alive: bool,
}

/// What kind of action an event records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Spawn,
    Move,
    Gather,
    Eat,
    Rest,
    Starve,
    Died,
    Regrow,
}

impl EventKind {
    fn encode(self) -> u8 {
        match self {
            EventKind::Spawn => 0,
            EventKind::Move => 1,
            EventKind::Gather => 2,
            EventKind::Eat => 3,
            EventKind::Rest => 4,
            EventKind::Starve => 5,
            EventKind::Died => 6,
            EventKind::Regrow => 7,
        }
    }

    fn decode(v: u8) -> Option<Self> {
        match v {
            0 => Some(EventKind::Spawn),
            1 => Some(EventKind::Move),
            2 => Some(EventKind::Gather),
            3 => Some(EventKind::Eat),
            4 => Some(EventKind::Rest),
            5 => Some(EventKind::Starve),
            6 => Some(EventKind::Died),
            7 => Some(EventKind::Regrow),
            _ => None,
        }
    }
}

/// One recorded action. `a` and `b` carry kind-specific values:
/// * Spawn: `a` = starting energy, `b` = starting health.
/// * Move: `a` = energy after the move.
/// * Gather: `a` = resource kind (1 food, 2 wood, 3 stone),
///   `b` = cell amount after gathering.
/// * Eat / Rest: `a` = food after, `b` = energy after.
/// * Starve: `a` = health after.
/// * Died: unused.
/// * Regrow: `a` = resource kind, `b` = cell amount after regrowth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    /// Tick the action happened on (0 for spawns).
    pub tick: u64,
    /// What happened.
    pub kind: EventKind,
    /// Acting agent's index, or `None` for world events (regrowth).
    pub agent: Option<u8>,
    /// Grid column the event concerns.
    pub x: u8,
    /// Grid row the event concerns.
    pub y: u8,
    /// Kind-specific value (see above).
    pub a: u32,
    /// Kind-specific value (see above).
    pub b: u32,
}

/// The Korn world: grid, agents, event log, and the PRNG state that
/// makes the whole simulation a pure function of (seed, tick count).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct World {
    seed: u64,
    tick: u64,
    rng: XorShift64Star,
    cells: Vec<Cell>,
    agents: Vec<Agent>,
    events: Vec<Event>,
}

/// Derive an agent's secret key bytes deterministically from
/// (world seed, agent index). The public identity is then derived
/// from those bytes, so the same seed + index + name always yields
/// the same [`PeerId`] — and no secret ever needs to be stored in a
/// save, because it can be re-derived on load.
fn agent_secret(seed: u64, index: usize) -> [u8; 32] {
    let mut r =
        XorShift64Star::new(seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index as u64 + 1)));
    let mut out = [0u8; 32];
    for chunk in out.chunks_mut(8) {
        chunk.copy_from_slice(&r.next_u64().to_le_bytes());
    }
    out
}

fn cell_index(x: usize, y: usize) -> usize {
    y * WORLD_WIDTH + x
}

/// FNV-1a 64-bit checksum, used to detect corrupted saves. This is a
/// checksum, not a security hash — integrity, not authentication.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Read `n` bytes at the cursor during save parsing, never past the
/// checksum. A free function (not a closure) so the returned slice can
/// borrow the input with an explicit lifetime.
fn take_bytes<'a>(cur: &mut usize, n: usize, bytes: &'a [u8]) -> Result<&'a [u8], SimError> {
    let end = cur
        .checked_add(n)
        .ok_or_else(|| SimError::CorruptSave("truncated".to_string()))?;
    if end > bytes.len() - 8 {
        return Err(SimError::CorruptSave("truncated".to_string()));
    }
    let slice = &bytes[*cur..end];
    *cur = end;
    Ok(slice)
}

impl World {
    /// Create a fresh world from a seed and agent names. Terrain,
    /// resources and spawn positions all come from the seed, so the
    /// same seed + names always builds the same world.
    pub fn new(seed: u64, agent_names: &[&str]) -> Result<Self, SimError> {
        if agent_names.len() > 255 {
            return Err(SimError::TooManyAgents(agent_names.len()));
        }
        let mut seen: Vec<&str> = Vec::with_capacity(agent_names.len());
        for name in agent_names {
            if seen.contains(name) {
                return Err(SimError::DuplicateAgentName((*name).to_string()));
            }
            seen.push(name);
        }

        let mut rng = XorShift64Star::new(seed);

        // Terrain: per-cell random value, then one 3x3 smoothing pass so
        // terrain forms blobs instead of speckle, then thresholds.
        let mut raw = vec![0.0f64; WORLD_WIDTH * WORLD_HEIGHT];
        for v in raw.iter_mut() {
            *v = rng.next_f64();
        }
        let mut cells: Vec<Cell> = Vec::with_capacity(WORLD_WIDTH * WORLD_HEIGHT);
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let mut sum = 0.0f64;
                let mut n = 0.0f64;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && nx < WORLD_WIDTH as i32 && ny >= 0 && ny < WORLD_HEIGHT as i32
                        {
                            sum += raw[cell_index(nx as usize, ny as usize)];
                            n += 1.0;
                        }
                    }
                }
                let v = sum / n;
                let terrain = if v < 0.40 {
                    Terrain::Water
                } else if v < 0.47 {
                    Terrain::Mountain
                } else if v < 0.60 {
                    Terrain::Forest
                } else {
                    Terrain::Plains
                };
                // Resources follow terrain: food on plains, wood in
                // forest, stone on mountain, nothing on water.
                let (resource, amount) = match terrain {
                    Terrain::Plains if rng.next_f64() < 0.25 => {
                        (Some(ResourceKind::Food), 3 + rng.below(6) as u8)
                    }
                    Terrain::Forest if rng.next_f64() < 0.35 => {
                        (Some(ResourceKind::Wood), 3 + rng.below(6) as u8)
                    }
                    Terrain::Mountain if rng.next_f64() < 0.30 => {
                        (Some(ResourceKind::Stone), 3 + rng.below(6) as u8)
                    }
                    _ => (None, 0),
                };
                cells.push(Cell {
                    terrain,
                    resource,
                    amount,
                    max_amount: amount,
                    regrow_in: 0,
                });
            }
        }

        // Agents: deterministic identities, spawn on a random passable cell.
        let mut agents: Vec<Agent> = Vec::with_capacity(agent_names.len());
        let mut events: Vec<Event> = Vec::new();
        for (i, name) in agent_names.iter().enumerate() {
            let secret = agent_secret(seed, i);
            let identity = match Identity::from_secret_bytes(&secret, name) {
                Ok(id) => id,
                Err(IdentityError::InvalidName(msg)) => {
                    return Err(SimError::InvalidAgentName(msg));
                }
                Err(e) => return Err(SimError::Identity(e)),
            };
            let mut spawned = None;
            for _ in 0..1000 {
                let x = rng.below(WORLD_WIDTH as u64) as u8;
                let y = rng.below(WORLD_HEIGHT as u64) as u8;
                if cells[cell_index(x as usize, y as usize)].terrain.passable() {
                    spawned = Some((x, y));
                    break;
                }
            }
            let (x, y) = spawned.ok_or(SimError::NoSpawnCell)?;
            agents.push(Agent {
                id: identity.peer_id().clone(),
                name: (*name).to_string(),
                x,
                y,
                food: 5,
                wood: 0,
                stone: 0,
                energy: MAX_VITALS,
                health: MAX_VITALS,
                alive: true,
            });
            events.push(Event {
                tick: 0,
                kind: EventKind::Spawn,
                agent: Some(i as u8),
                x,
                y,
                a: MAX_VITALS,
                b: MAX_VITALS,
            });
        }

        Ok(Self {
            seed,
            tick: 0,
            rng,
            cells,
            agents,
            events,
        })
    }

    /// World seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Current tick (starts at 0; each [`World::tick`] adds one).
    pub fn tick_count(&self) -> u64 {
        self.tick
    }

    /// All agents, in index order.
    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }

    /// The append-only event log.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// The cell at (x, y), or `None` when out of bounds.
    pub fn cell(&self, x: usize, y: usize) -> Option<&Cell> {
        if x < WORLD_WIDTH && y < WORLD_HEIGHT {
            Some(&self.cells[cell_index(x, y)])
        } else {
            None
        }
    }

    /// Advance the world by one tick: every living agent takes one
    /// action (in index order), then depleted resources regrow.
    pub fn tick(&mut self) {
        self.tick += 1;
        let t = self.tick;
        for i in 0..self.agents.len() {
            self.act(i, t);
        }
        self.regrow(t);
    }

    /// Advance the world by `n` ticks.
    pub fn tick_n(&mut self, n: u64) {
        for _ in 0..n {
            self.tick();
        }
    }

    /// One agent's action for this tick (see the crate docs for the
    /// priority order). Dead agents do nothing.
    fn act(&mut self, idx: usize, tick: u64) {
        if !self.agents[idx].alive {
            return;
        }
        let (x, y, energy, food) = {
            let a = &self.agents[idx];
            (a.x, a.y, a.energy, a.food)
        };

        // 1. Rest or starve when exhausted.
        if energy < 15 {
            if food > 0 {
                let a = &mut self.agents[idx];
                a.food -= 1;
                a.energy = (a.energy + 30).min(MAX_VITALS);
                let (food_after, energy_after) = (a.food, a.energy);
                self.events.push(Event {
                    tick,
                    kind: EventKind::Rest,
                    agent: Some(idx as u8),
                    x,
                    y,
                    a: food_after,
                    b: energy_after,
                });
            } else {
                let a = &mut self.agents[idx];
                a.health = a.health.saturating_sub(10);
                let health_after = a.health;
                self.events.push(Event {
                    tick,
                    kind: EventKind::Starve,
                    agent: Some(idx as u8),
                    x,
                    y,
                    a: health_after,
                    b: 0,
                });
                if health_after == 0 {
                    self.agents[idx].alive = false;
                    self.events.push(Event {
                        tick,
                        kind: EventKind::Died,
                        agent: Some(idx as u8),
                        x,
                        y,
                        a: 0,
                        b: 0,
                    });
                }
            }
            return;
        }

        // 2. Eat when getting tired.
        if energy < 50 && food > 0 {
            let a = &mut self.agents[idx];
            a.food -= 1;
            a.energy = (a.energy + 20).min(MAX_VITALS);
            let (food_after, energy_after) = (a.food, a.energy);
            self.events.push(Event {
                tick,
                kind: EventKind::Eat,
                agent: Some(idx as u8),
                x,
                y,
                a: food_after,
                b: energy_after,
            });
            return;
        }

        // 3. Gather when standing on a resource.
        let ci = cell_index(x as usize, y as usize);
        if let Some(kind) = self.cells[ci].resource {
            if self.cells[ci].amount > 0 {
                self.cells[ci].amount -= 1;
                if self.cells[ci].amount < self.cells[ci].max_amount
                    && self.cells[ci].regrow_in == 0
                {
                    self.cells[ci].regrow_in = REGROW_TICKS;
                }
                let amount_after = self.cells[ci].amount;
                let a = &mut self.agents[idx];
                match kind {
                    ResourceKind::Food => a.food += 1,
                    ResourceKind::Wood => a.wood += 1,
                    ResourceKind::Stone => a.stone += 1,
                }
                a.energy = a.energy.saturating_sub(1);
                self.events.push(Event {
                    tick,
                    kind: EventKind::Gather,
                    agent: Some(idx as u8),
                    x,
                    y,
                    a: kind.encode() as u32,
                    b: amount_after as u32,
                });
                return;
            }
        }

        // 4. Move toward the nearest visible resource, else wander.
        let target = self.nearest_visible_resource(x, y);
        let (dx, dy) = match target {
            Some((tx, ty)) => (
                (tx as i32 - x as i32).signum(),
                (ty as i32 - y as i32).signum(),
            ),
            None => {
                // Wander: one PRNG-chosen cardinal direction.
                match self.rng.below(4) {
                    0 => (0, -1),
                    1 => (0, 1),
                    2 => (-1, 0),
                    _ => (1, 0),
                }
            }
        };
        // Try the x step first, then the y step; never enter an
        // impassable cell or leave the grid.
        let mut nx = x as i32;
        let mut ny = y as i32;
        let mut moved = false;
        for (sx, sy) in [(dx, 0), (0, dy)] {
            if sx == 0 && sy == 0 {
                continue;
            }
            let cx = x as i32 + sx;
            let cy = y as i32 + sy;
            if cx >= 0
                && cx < WORLD_WIDTH as i32
                && cy >= 0
                && cy < WORLD_HEIGHT as i32
                && self.cells[cell_index(cx as usize, cy as usize)]
                    .terrain
                    .passable()
            {
                nx = cx;
                ny = cy;
                moved = true;
                break;
            }
        }
        if moved {
            let cost = self.cells[cell_index(nx as usize, ny as usize)]
                .terrain
                .move_cost();
            let a = &mut self.agents[idx];
            a.x = nx as u8;
            a.y = ny as u8;
            a.energy = a.energy.saturating_sub(cost);
            let energy_after = a.energy;
            self.events.push(Event {
                tick,
                kind: EventKind::Move,
                agent: Some(idx as u8),
                x: nx as u8,
                y: ny as u8,
                a: energy_after,
                b: 0,
            });
        }
    }

    /// Nearest cell with a resource, within sight, by Manhattan
    /// distance. Row-major scan order breaks ties deterministically.
    fn nearest_visible_resource(&self, x: u8, y: u8) -> Option<(u8, u8)> {
        let mut best: Option<(u8, u8)> = None;
        let mut best_dist = i32::MAX;
        for cy in (y as i32 - SIGHT_RADIUS).max(0)
            ..=(y as i32 + SIGHT_RADIUS).min(WORLD_HEIGHT as i32 - 1)
        {
            for cx in (x as i32 - SIGHT_RADIUS).max(0)
                ..=(x as i32 + SIGHT_RADIUS).min(WORLD_WIDTH as i32 - 1)
            {
                let cell = &self.cells[cell_index(cx as usize, cy as usize)];
                if cell.resource.is_some() && cell.amount > 0 && cell.terrain.passable() {
                    let dist = (cx - x as i32).abs() + (cy - y as i32).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best = Some((cx as u8, cy as u8));
                    }
                }
            }
        }
        best
    }

    /// Regrow depleted resources: each cell below its starting amount
    /// regains 1 unit every [`REGROW_TICKS`] ticks.
    fn regrow(&mut self, tick: u64) {
        let mut regrown: Vec<(u8, u8, ResourceKind, u8)> = Vec::new();
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let ci = cell_index(x, y);
                let cell = &mut self.cells[ci];
                if cell.amount < cell.max_amount && cell.regrow_in > 0 {
                    cell.regrow_in -= 1;
                    if cell.regrow_in == 0 {
                        cell.amount += 1;
                        if cell.amount < cell.max_amount {
                            cell.regrow_in = REGROW_TICKS;
                        }
                        if let Some(kind) = cell.resource {
                            regrown.push((x as u8, y as u8, kind, cell.amount));
                        }
                    }
                }
            }
        }
        for (x, y, kind, amount) in regrown {
            self.events.push(Event {
                tick,
                kind: EventKind::Regrow,
                agent: None,
                x,
                y,
                a: kind.encode() as u32,
                b: amount as u32,
            });
        }
    }

    /// Serialize the whole world to binary. Layout: `KSIM1` magic,
    /// version, seed, tick, PRNG state, terrain + resource grid,
    /// agents, event log, FNV-1a checksum over everything before it.
    pub fn save(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            6 + 8 * 3 + 2 + 6 * 4096 + 64 * self.agents.len() + 21 * self.events.len() + 8,
        );
        out.extend_from_slice(SAVE_MAGIC);
        out.push(SAVE_VERSION);
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&self.tick.to_le_bytes());
        out.extend_from_slice(&self.rng.state().to_le_bytes());
        out.push(WORLD_WIDTH as u8);
        out.push(WORLD_HEIGHT as u8);
        out.extend_from_slice(&(self.agents.len() as u16).to_le_bytes());
        for cell in &self.cells {
            out.push(cell.terrain.encode());
            out.push(cell.resource.map(|r| r.encode()).unwrap_or(0));
            out.push(cell.amount);
            out.push(cell.max_amount);
            out.extend_from_slice(&cell.regrow_in.to_le_bytes());
        }
        for agent in &self.agents {
            out.push(agent.name.len() as u8);
            out.extend_from_slice(agent.name.as_bytes());
            out.push(agent.x);
            out.push(agent.y);
            out.extend_from_slice(&agent.food.to_le_bytes());
            out.extend_from_slice(&agent.wood.to_le_bytes());
            out.extend_from_slice(&agent.stone.to_le_bytes());
            out.extend_from_slice(&agent.energy.to_le_bytes());
            out.extend_from_slice(&agent.health.to_le_bytes());
            out.push(agent.alive as u8);
        }
        out.extend_from_slice(&(self.events.len() as u32).to_le_bytes());
        for e in &self.events {
            out.extend_from_slice(&e.tick.to_le_bytes());
            out.push(e.kind.encode());
            out.push(e.agent.unwrap_or(255));
            out.push(e.x);
            out.push(e.y);
            out.extend_from_slice(&e.a.to_le_bytes());
            out.extend_from_slice(&e.b.to_le_bytes());
        }
        let checksum = fnv1a(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    /// Load a world from bytes produced by [`World::save`]. Any
    /// corruption is a clean [`SimError::CorruptSave`], never a panic.
    pub fn load(bytes: &[u8]) -> Result<World, SimError> {
        let corrupt = |msg: &str| SimError::CorruptSave(msg.to_string());
        // Header (6) + seed/tick/prng (24) + dims (2) + agent count (2)
        // + grid (6 * 4096) + event count (4) + checksum (8) is the
        // minimum; agents and events add more.
        if bytes.len() < 6 + 24 + 2 + 2 + 6 * WORLD_WIDTH * WORLD_HEIGHT + 4 + 8 {
            return Err(corrupt("save too short"));
        }
        if &bytes[..5] != SAVE_MAGIC {
            return Err(corrupt("bad magic"));
        }
        if bytes[5] != SAVE_VERSION {
            return Err(corrupt("unsupported version"));
        }
        let stored_checksum = u64::from_le_bytes(
            bytes[bytes.len() - 8..]
                .try_into()
                .map_err(|_| corrupt("missing checksum"))?,
        );
        if fnv1a(&bytes[..bytes.len() - 8]) != stored_checksum {
            return Err(corrupt("checksum mismatch"));
        }

        let mut cur = 6usize;
        let u64_at = |cur: &mut usize, bytes: &[u8]| -> Result<u64, SimError> {
            Ok(u64::from_le_bytes(
                take_bytes(cur, 8, bytes)?.try_into().unwrap(),
            ))
        };
        let u32_at = |cur: &mut usize, bytes: &[u8]| -> Result<u32, SimError> {
            Ok(u32::from_le_bytes(
                take_bytes(cur, 4, bytes)?.try_into().unwrap(),
            ))
        };
        let u16_at = |cur: &mut usize, bytes: &[u8]| -> Result<u16, SimError> {
            Ok(u16::from_le_bytes(
                take_bytes(cur, 2, bytes)?.try_into().unwrap(),
            ))
        };
        let u8_at = |cur: &mut usize, bytes: &[u8]| -> Result<u8, SimError> {
            Ok(take_bytes(cur, 1, bytes)?[0])
        };

        let seed = u64_at(&mut cur, bytes)?;
        let tick = u64_at(&mut cur, bytes)?;
        let prng_state = u64_at(&mut cur, bytes)?;
        let width = u8_at(&mut cur, bytes)? as usize;
        let height = u8_at(&mut cur, bytes)? as usize;
        if width != WORLD_WIDTH || height != WORLD_HEIGHT {
            return Err(corrupt("unexpected world dimensions"));
        }
        let agent_count = u16_at(&mut cur, bytes)? as usize;
        if agent_count > 255 {
            return Err(corrupt("too many agents"));
        }

        let mut cells = Vec::with_capacity(WORLD_WIDTH * WORLD_HEIGHT);
        for _ in 0..WORLD_WIDTH * WORLD_HEIGHT {
            let terrain = Terrain::decode(u8_at(&mut cur, bytes)?)
                .ok_or_else(|| corrupt("bad terrain value"))?;
            let res_v = u8_at(&mut cur, bytes)?;
            let resource = if res_v == 0 {
                None
            } else {
                Some(ResourceKind::decode(res_v).ok_or_else(|| corrupt("bad resource kind"))?)
            };
            let amount = u8_at(&mut cur, bytes)?;
            let max_amount = u8_at(&mut cur, bytes)?;
            let regrow_in = u16_at(&mut cur, bytes)?;
            if max_amount > 8 || amount > max_amount || regrow_in > REGROW_TICKS {
                return Err(corrupt("resource state out of range"));
            }
            if resource.is_none() && (amount != 0 || max_amount != 0 || regrow_in != 0) {
                return Err(corrupt("resourceless cell holds a resource"));
            }
            if resource.is_some() && max_amount == 0 {
                return Err(corrupt("resource cell with zero max"));
            }
            if !terrain.passable() && resource.is_some() && terrain != Terrain::Mountain {
                return Err(corrupt("resource on impassable terrain"));
            }
            cells.push(Cell {
                terrain,
                resource,
                amount,
                max_amount,
                regrow_in,
            });
        }

        let mut agents = Vec::with_capacity(agent_count);
        for i in 0..agent_count {
            let name_len = u8_at(&mut cur, bytes)? as usize;
            let name_bytes = take_bytes(&mut cur, name_len, bytes)?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| corrupt("agent name is not UTF-8"))?
                .to_string();
            let x = u8_at(&mut cur, bytes)?;
            let y = u8_at(&mut cur, bytes)?;
            let food = u32_at(&mut cur, bytes)?;
            let wood = u32_at(&mut cur, bytes)?;
            let stone = u32_at(&mut cur, bytes)?;
            let energy = u32_at(&mut cur, bytes)?;
            let health = u32_at(&mut cur, bytes)?;
            let alive_b = u8_at(&mut cur, bytes)?;
            if (x as usize) >= WORLD_WIDTH || (y as usize) >= WORLD_HEIGHT {
                return Err(corrupt("agent out of bounds"));
            }
            if energy > MAX_VITALS || health > MAX_VITALS || alive_b > 1 {
                return Err(corrupt("agent vitals out of range"));
            }
            // Re-derive the identity exactly as construction does, so a
            // loaded world has the same agent ids as the original.
            let secret = agent_secret(seed, i);
            let identity = match Identity::from_secret_bytes(&secret, &name) {
                Ok(id) => id,
                Err(IdentityError::InvalidName(msg)) => {
                    return Err(SimError::InvalidAgentName(msg));
                }
                Err(e) => return Err(SimError::Identity(e)),
            };
            agents.push(Agent {
                id: identity.peer_id().clone(),
                name,
                x,
                y,
                food,
                wood,
                stone,
                energy,
                health,
                alive: alive_b == 1,
            });
        }

        let event_count = u32_at(&mut cur, bytes)? as usize;
        let mut events = Vec::with_capacity(event_count);
        let mut prev_tick = 0u64;
        for _ in 0..event_count {
            let etick = u64_at(&mut cur, bytes)?;
            let kind = EventKind::decode(u8_at(&mut cur, bytes)?)
                .ok_or_else(|| corrupt("bad event kind"))?;
            let agent_b = u8_at(&mut cur, bytes)?;
            let agent = if agent_b == 255 {
                None
            } else if (agent_b as usize) < agent_count {
                Some(agent_b)
            } else {
                return Err(corrupt("event names unknown agent"));
            };
            let ex = u8_at(&mut cur, bytes)?;
            let ey = u8_at(&mut cur, bytes)?;
            let a = u32_at(&mut cur, bytes)?;
            let b = u32_at(&mut cur, bytes)?;
            if etick > tick || etick < prev_tick {
                return Err(corrupt("event ticks out of order"));
            }
            prev_tick = etick;
            if (ex as usize) >= WORLD_WIDTH || (ey as usize) >= WORLD_HEIGHT {
                return Err(corrupt("event out of bounds"));
            }
            match kind {
                EventKind::Regrow if agent.is_some() => {
                    return Err(corrupt("regrow event names an agent"));
                }
                EventKind::Regrow => {}
                _ if agent.is_none() => {
                    return Err(corrupt("agent event names no agent"));
                }
                _ => {}
            }
            events.push(Event {
                tick: etick,
                kind,
                agent,
                x: ex,
                y: ey,
                a,
                b,
            });
        }

        // Nothing may follow the event log except the checksum.
        if cur != bytes.len() - 8 {
            return Err(corrupt("trailing bytes"));
        }

        Ok(World {
            seed,
            tick,
            rng: XorShift64Star::from_state(prng_state),
            cells,
            agents,
            events,
        })
    }

    /// Content address of the current save bytes: the SHA-256 CID of
    /// [`World::save`], via the content store. Two identical worlds
    /// have the same snapshot CID.
    pub fn snapshot_cid(&self) -> Cid {
        let mut store = ContentStore::new();
        store.put(&self.save())
    }

    /// Reconstruct a world by re-simulating from the seed and checking
    /// the supplied event log against the re-simulation, event for
    /// event. `target_tick` is how many ticks the original world ran:
    /// the log alone cannot record ticks in which nothing happened, so
    /// replay needs the count. Returns the re-simulated world when the
    /// log matches exactly; any mismatch (or a log whose ticks exceed
    /// the target) is a [`SimError::ReplayMismatch`].
    pub fn replay(
        seed: u64,
        agent_names: &[&str],
        events: &[Event],
        target_tick: u64,
    ) -> Result<World, SimError> {
        let mut world = World::new(seed, agent_names)?;
        // The fresh world already logged the spawn events at tick 0.
        if events.len() < world.events.len() || events[..world.events.len()] != world.events[..] {
            return Err(SimError::ReplayMismatch(
                "log does not start with the world's spawn events".to_string(),
            ));
        }
        if events.iter().map(|e| e.tick).max().unwrap_or(0) > target_tick {
            return Err(SimError::ReplayMismatch(
                "log has events beyond the target tick".to_string(),
            ));
        }
        let mut checked = world.events.len();
        for _ in 0..target_tick {
            world.tick();
            let fresh = &world.events[checked..];
            let expected = &events[checked..checked + fresh.len().min(events.len() - checked)];
            if fresh != expected {
                return Err(SimError::ReplayMismatch(format!(
                    "events diverge at tick {}",
                    world.tick
                )));
            }
            checked += fresh.len();
            if checked > events.len() {
                return Err(SimError::ReplayMismatch(
                    "re-simulation produced more events than the log".to_string(),
                ));
            }
        }
        if checked != events.len() {
            return Err(SimError::ReplayMismatch(
                "log has events beyond the re-simulation".to_string(),
            ));
        }
        Ok(world)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NAMES: [&str; 4] = ["Korn", "Mira", "Juno", "Ash"];

    #[test]
    fn same_seed_same_ticks_identical_worlds() {
        let mut a = World::new(1234, &NAMES).unwrap();
        let mut b = World::new(1234, &NAMES).unwrap();
        a.tick_n(50);
        b.tick_n(50);
        assert_eq!(a, b);
        assert_eq!(a.save(), b.save());
        assert_eq!(a.snapshot_cid(), b.snapshot_cid());
    }

    #[test]
    fn different_seeds_different_worlds() {
        let a = World::new(1, &NAMES).unwrap();
        let b = World::new(2, &NAMES).unwrap();
        assert_ne!(a, b);
        assert_ne!(a.save(), b.save());
    }

    #[test]
    fn all_four_terrains_present() {
        for seed in [1u64, 7, 42, 1234] {
            let w = World::new(seed, &NAMES).unwrap();
            let mut seen = [false; 4];
            for y in 0..WORLD_HEIGHT {
                for x in 0..WORLD_WIDTH {
                    seen[w.cell(x, y).unwrap().terrain.encode() as usize] = true;
                }
            }
            assert!(seen.iter().all(|s| *s), "seed {seed} lacks a terrain");
        }
    }

    #[test]
    fn agents_gather_and_food_increases() {
        let mut w = World::new(1234, &NAMES).unwrap();
        w.tick_n(200);
        // At least one food gather happened.
        let food_gathers = w
            .events()
            .iter()
            .filter(|e| e.kind == EventKind::Gather && e.a == ResourceKind::Food.encode() as u32)
            .count();
        assert!(food_gathers > 0, "no food was ever gathered");
        // Wood and stone are never consumed, so any gather shows up.
        let total_stock: u32 = w.agents().iter().map(|a| a.wood + a.stone).sum();
        assert!(total_stock > 0, "no wood or stone was ever gathered");
        // Agents only ever stand on passable terrain.
        for a in w.agents() {
            assert!(w
                .cell(a.x as usize, a.y as usize)
                .unwrap()
                .terrain
                .passable());
        }
    }

    #[test]
    fn starving_agent_loses_health_and_dies_without_panic() {
        let mut w = World::new(7, &NAMES).unwrap();
        // Remove every resource so nothing can be gathered, and strip
        // the first agent of food and energy.
        for cell in w.cells.iter_mut() {
            cell.resource = None;
            cell.amount = 0;
            cell.max_amount = 0;
            cell.regrow_in = 0;
        }
        w.agents[0].food = 0;
        w.agents[0].energy = 0;
        w.agents[0].health = 100;
        // 10 starving ticks: health 100 -> 0, then the agent dies.
        w.tick_n(10);
        let a = &w.agents[0];
        assert_eq!(a.health, 0);
        assert!(!a.alive);
        assert!(w
            .events()
            .iter()
            .any(|e| e.kind == EventKind::Starve && e.agent == Some(0)));
        assert!(w
            .events()
            .iter()
            .any(|e| e.kind == EventKind::Died && e.agent == Some(0)));
        // The world keeps ticking with a dead agent in it.
        w.tick_n(20);
        assert_eq!(w.tick_count(), 30);
        assert!(!w.agents[0].alive);
    }

    #[test]
    fn resources_regrow_after_depletion() {
        // Agentless world, so nothing gathers the cell under test.
        let mut w = World::new(42, &[]).unwrap();
        // Find a resource cell and deplete it by hand.
        let mut target = None;
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let c = w.cell(x, y).unwrap();
                if c.resource.is_some() && c.amount > 0 {
                    target = Some((x, y));
                    break;
                }
            }
            if target.is_some() {
                break;
            }
        }
        let (x, y) = target.expect("world has no resources");
        let ci = cell_index(x, y);
        let max = w.cells[ci].max_amount;
        w.cells[ci].amount = 0;
        w.cells[ci].regrow_in = 1;
        w.tick();
        assert_eq!(w.cells[ci].amount, 1);
        assert!(w
            .events()
            .iter()
            .any(|e| e.kind == EventKind::Regrow && e.x == x as u8 && e.y == y as u8));
        // Keep ticking: it regrows back to its starting amount.
        w.tick_n((REGROW_TICKS as u64) * (max as u64));
        assert_eq!(w.cells[ci].amount, max);
    }

    #[test]
    fn event_log_replays_to_the_same_world() {
        let mut w = World::new(7, &NAMES).unwrap();
        w.tick_n(25);
        let events = w.events().to_vec();
        let replayed = World::replay(7, &NAMES, &events, 25).unwrap();
        assert_eq!(replayed, w);
        assert_eq!(replayed.save(), w.save());
    }

    #[test]
    fn tampered_log_fails_replay() {
        let mut w = World::new(7, &NAMES).unwrap();
        w.tick_n(25);
        let mut events = w.events().to_vec();
        // Flip one payload value in the middle of the log.
        let mid = events.len() / 2;
        events[mid].a = events[mid].a.wrapping_add(1);
        assert!(matches!(
            World::replay(7, &NAMES, &events, 25),
            Err(SimError::ReplayMismatch(_))
        ));
        // A truncated log also fails.
        events.truncate(events.len() - 1);
        assert!(matches!(
            World::replay(7, &NAMES, &events, 25),
            Err(SimError::ReplayMismatch(_))
        ));
    }

    #[test]
    fn save_load_round_trip_is_exact() {
        let mut w = World::new(99, &NAMES).unwrap();
        w.tick_n(30);
        let bytes = w.save();
        let loaded = World::load(&bytes).unwrap();
        assert_eq!(loaded, w);
        assert_eq!(loaded.save(), bytes);
        assert_eq!(loaded.snapshot_cid(), w.snapshot_cid());
        // A loaded world continues identically to the original.
        let mut a = w.clone();
        let mut b = loaded;
        a.tick_n(20);
        b.tick_n(20);
        assert_eq!(a, b);
    }

    #[test]
    fn corrupted_saves_are_refused() {
        let mut w = World::new(99, &NAMES).unwrap();
        w.tick_n(10);
        let bytes = w.save();

        // Flipped byte in the middle (checksum catches it).
        let mut bad = bytes.clone();
        bad[100] ^= 0xFF;
        assert!(matches!(World::load(&bad), Err(SimError::CorruptSave(_))));

        // Flipped byte in the checksum itself.
        let mut bad = bytes.clone();
        let n = bad.len();
        bad[n - 1] ^= 0xFF;
        assert!(matches!(World::load(&bad), Err(SimError::CorruptSave(_))));

        // Truncated.
        assert!(matches!(
            World::load(&bytes[..bytes.len() - 20]),
            Err(SimError::CorruptSave(_))
        ));
        // Trailing garbage.
        let mut bad = bytes.clone();
        bad.push(0);
        assert!(matches!(World::load(&bad), Err(SimError::CorruptSave(_))));
        // Bad magic.
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(matches!(World::load(&bad), Err(SimError::CorruptSave(_))));
        // Empty and tiny inputs.
        assert!(matches!(World::load(&[]), Err(SimError::CorruptSave(_))));
        assert!(matches!(
            World::load(&[1, 2, 3]),
            Err(SimError::CorruptSave(_))
        ));
    }

    #[test]
    fn bad_agent_names_are_refused() {
        assert!(matches!(
            World::new(1, &["Korn", "Korn"]),
            Err(SimError::DuplicateAgentName(_))
        ));
        assert!(matches!(
            World::new(1, &["bad name!"]),
            Err(SimError::InvalidAgentName(_))
        ));
        assert!(matches!(
            World::new(1, &[""]),
            Err(SimError::InvalidAgentName(_))
        ));
    }

    #[test]
    fn empty_world_ticks_fine() {
        let mut w = World::new(5, &[]).unwrap();
        w.tick_n(10);
        assert_eq!(w.tick_count(), 10);
        assert_eq!(w.agents().len(), 0);
        // Only regrow events can exist; replay still works.
        let events = w.events().to_vec();
        let replayed = World::replay(5, &[], &events, 10).unwrap();
        assert_eq!(replayed, w);
    }
}
