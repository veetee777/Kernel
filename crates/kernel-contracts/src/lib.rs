//! KERNEL smart contracts: a deterministic stack-based bytecode VM.
//!
//! A contract is a small program over 64-bit words. Execution is fully
//! deterministic: the same bytecode, the same input stack, and the same
//! gas limit always produce the same result. A contract can observe
//! nothing outside its inputs and its own storage — no randomness, no
//! time, no I/O, no syscalls.
//!
//! # Opcode set
//!
//! | Op | Encoding | Effect |
//! |----|----------|--------|
//! | `HALT` | `0x00` | Stop. The call succeeds with the current stack. |
//! | `PUSH` | `0x01` + 8-byte LE `u64` | Push the immediate. |
//! | `ADD` | `0x02` | Pop `b`, pop `a`, push `a + b` (wrapping). |
//! | `SUB` | `0x03` | Pop `b`, pop `a`, push `a - b` (wrapping). |
//! | `MUL` | `0x04` | Pop `b`, pop `a`, push `a * b` (wrapping). |
//! | `DIV` | `0x05` | Pop `b`, pop `a`, push `a / b`. `b == 0` is a clean `DivideByZero` error. |
//! | `EQ` | `0x06` | Pop `b`, pop `a`, push `1` if `a == b`, else `0`. |
//! | `LT` | `0x07` | Pop `b`, pop `a`, push `1` if `a < b`, else `0`. |
//! | `JUMP` | `0x08` + 2-byte LE target | Unconditional jump to a byte offset. |
//! | `JUMPIF` | `0x09` + 2-byte LE target | Pop `cond`; jump if `cond != 0`. |
//! | `STORE` | `0x0A` + 1-byte key | Pop a value, store it under `key`. |
//! | `LOAD` | `0x0B` + 1-byte key | Push the value stored under `key` (`0` if unset). |
//!
//! There are no cross-contract calls in this version: one call runs one
//! contract. Storage is per-contract and isolated — a contract cannot
//! read or write another contract's keys.
//!
//! # Gas
//!
//! Every opcode costs a fixed amount of gas (see [`gas_cost`]). A call
//! runs until `HALT`, an error, or the gas budget is exhausted. Because
//! every opcode costs at least 1 gas, the gas budget is also the step
//! limit: no program can loop forever. A call that runs out of gas
//! fails with [`ContractError::OutOfGas`], which carries the gas used
//! (always the full limit).
//!
//! # Atomicity
//!
//! A call's storage writes are committed only when the call halts
//! successfully. If a call errors — out of gas, divide by zero, stack
//! underflow, anything — its writes are discarded, as if the call never
//! ran.
//!
//! # Validation
//!
//! [`Engine::deploy`] validates the bytecode before storing it: unknown
//! opcodes, truncated immediates, jump targets that do not land on an
//! instruction boundary, and programs with no `HALT` are all refused at
//! deploy time with a clean error. A deployed contract therefore cannot
//! fail for structural reasons; only runtime conditions can stop it.
//!
//! # Example: a counter
//!
//! ```rust
//! use kernel_contracts::{Engine, OP_HALT, OP_PUSH, OP_ADD, OP_LOAD, OP_STORE};
//!
//! let mut engine = Engine::new();
//! // LOAD 0; PUSH 1; ADD; STORE 0; LOAD 0; HALT — read the counter, add
//! // one, store it, then push the new value so the call returns it.
//! let code = vec![
//!     OP_LOAD, 0,
//!     OP_PUSH, 1, 0, 0, 0, 0, 0, 0, 0,
//!     OP_ADD,
//!     OP_STORE, 0,
//!     OP_LOAD, 0,
//!     OP_HALT,
//! ];
//! let id = engine.deploy(code).unwrap();
//! let first = engine.call(&id, &[], 1_000).unwrap();
//! assert_eq!(first.stack, vec![1]);
//! let second = engine.call(&id, &[], 1_000).unwrap();
//! assert_eq!(second.stack, vec![2]);
//! assert_eq!(engine.storage(&id, 0), Some(2));
//! ```
//!
//! # Honest limits
//!
//! This is a small custom VM, not WASM and not the EVM. It has no
//! cross-contract calls, no events, no persistent on-disk state
//! (storage lives in memory — restart the `Engine` and it is gone),
//! and no compiler: programs are assembled as raw bytes. A real chain
//! would need all of those, plus a proper compiler, plus a way to
//! charge for storage. What this gives KERNEL today is a deterministic,
//! gas-metered execution core that other crates can build on.

#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::fmt;

use kernel_crypto::{sha256, to_hex};

// ---------------------------------------------------------------------------
// Opcodes and gas
// ---------------------------------------------------------------------------

/// Stop. The call succeeds with the current stack.
pub const OP_HALT: u8 = 0x00;
/// Push a `u64` immediate (8 bytes, little-endian).
pub const OP_PUSH: u8 = 0x01;
/// Pop `b`, pop `a`, push `a + b` (wrapping).
pub const OP_ADD: u8 = 0x02;
/// Pop `b`, pop `a`, push `a - b` (wrapping).
pub const OP_SUB: u8 = 0x03;
/// Pop `b`, pop `a`, push `a * b` (wrapping).
pub const OP_MUL: u8 = 0x04;
/// Pop `b`, pop `a`, push `a / b`. `b == 0` is a clean error.
pub const OP_DIV: u8 = 0x05;
/// Pop `b`, pop `a`, push `1` if `a == b`, else `0`.
pub const OP_EQ: u8 = 0x06;
/// Pop `b`, pop `a`, push `1` if `a < b`, else `0`.
pub const OP_LT: u8 = 0x07;
/// Unconditional jump to a byte offset (2-byte LE target).
pub const OP_JUMP: u8 = 0x08;
/// Pop `cond`; jump to a byte offset if `cond != 0` (2-byte LE target).
pub const OP_JUMPIF: u8 = 0x09;
/// Pop a value, store it under the 1-byte key immediate.
pub const OP_STORE: u8 = 0x0A;
/// Push the value stored under the 1-byte key immediate (`0` if unset).
pub const OP_LOAD: u8 = 0x0B;

/// Maximum stack depth, including the caller's input stack. Pushing
/// past this is a clean [`ContractError::StackOverflow`].
pub const MAX_STACK: usize = 1_024;

/// Number of storage slots per contract (one `u64` per `u8` key).
pub const STORAGE_SLOTS: usize = 256;

/// Fixed gas cost of an opcode. State writes cost the most; a halt is
/// nearly free. Every opcode costs at least 1, so the gas budget is
/// also the step limit — no program can loop forever.
pub fn gas_cost(op: u8) -> u64 {
    match op {
        OP_HALT => 1,
        OP_PUSH => 3,
        OP_ADD | OP_SUB | OP_MUL | OP_EQ | OP_LT => 5,
        OP_DIV => 8,
        OP_JUMP => 8,
        OP_JUMPIF => 10,
        OP_LOAD => 10,
        OP_STORE => 20,
        _ => 0, // unknown opcodes are refused at deploy; never executed
    }
}

/// Byte length of an instruction starting with `op`, including its
/// immediate. `None` for an unknown opcode.
fn instruction_len(op: u8) -> Option<usize> {
    match op {
        OP_HALT | OP_ADD | OP_SUB | OP_MUL | OP_DIV | OP_EQ | OP_LT => Some(1),
        OP_PUSH => Some(9),
        OP_JUMP | OP_JUMPIF => Some(3),
        OP_STORE | OP_LOAD => Some(2),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from deploying or calling a contract. Every failure is a
/// plain value — the VM never panics on contract input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    /// Deploying an empty program.
    EmptyProgram,
    /// A byte that is not a known opcode, at deploy time.
    UnknownOpcode(u8),
    /// An instruction's immediate runs past the end of the program.
    TruncatedImmediate {
        /// The opcode whose immediate was cut short.
        op: u8,
        /// Byte offset of the instruction.
        at: usize,
    },
    /// A jump target that does not land on an instruction boundary.
    InvalidJumpTarget(usize),
    /// The program contains no `HALT` at all.
    MissingHalt,
    /// Execution ran past the last byte without halting (the program's
    /// final instruction was reached and was not `HALT`).
    RanOffEnd,
    /// The gas budget ran out. Carries the gas used (always the full
    /// limit — gas is only deducted when an opcode is affordable).
    OutOfGas(u64),
    /// An opcode needed more stack values than were present.
    StackUnderflow {
        /// The opcode that needed the missing values.
        op: u8,
    },
    /// Division by zero.
    DivideByZero,
    /// The stack (input + pushes) exceeded [`MAX_STACK`].
    StackOverflow,
    /// No contract with this id is deployed in this engine.
    ContractNotFound(ContractId),
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContractError::EmptyProgram => f.write_str("contract program is empty"),
            ContractError::UnknownOpcode(op) => write!(f, "unknown opcode 0x{op:02x}"),
            ContractError::TruncatedImmediate { op, at } => {
                write!(f, "opcode 0x{op:02x} at offset {at} is truncated")
            }
            ContractError::InvalidJumpTarget(t) => {
                write!(f, "jump target {t} is not an instruction boundary")
            }
            ContractError::MissingHalt => f.write_str("contract has no HALT"),
            ContractError::RanOffEnd => {
                f.write_str("execution ran past the end of the program without halting")
            }
            ContractError::OutOfGas(used) => write!(f, "out of gas (used {used})"),
            ContractError::StackUnderflow { op } => {
                write!(f, "stack underflow at opcode 0x{op:02x}")
            }
            ContractError::DivideByZero => f.write_str("division by zero"),
            ContractError::StackOverflow => {
                write!(f, "stack exceeded {MAX_STACK} values")
            }
            ContractError::ContractNotFound(id) => write!(f, "no contract {id} deployed"),
        }
    }
}

impl std::error::Error for ContractError {}

// ---------------------------------------------------------------------------
// Contract identity
// ---------------------------------------------------------------------------

/// A deployed contract's id: the SHA-256 of its validated bytecode.
/// The same bytecode always deploys to the same id, in any engine.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContractId(pub [u8; 32]);

impl ContractId {
    /// The raw 32 bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for ContractId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContractId({})", to_hex(&self.0))
    }
}

impl fmt::Display for ContractId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&to_hex(&self.0))
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// The result of a successful call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallResult {
    /// The stack when the contract halted, bottom-to-top. The caller's
    /// input values are still there underneath anything the contract
    /// pushed (unless the contract popped them).
    pub stack: Vec<u64>,
    /// Total gas spent by the call.
    pub gas_used: u64,
    /// How many `STORE` operations the call executed.
    pub storage_writes: usize,
}

/// One deployed contract: its bytecode and its private storage.
struct Contract {
    bytecode: Vec<u8>,
    storage: [u64; STORAGE_SLOTS],
}

/// The contract engine: deploys contracts and runs calls against them.
///
/// Storage lives in memory only — dropping the engine drops every
/// contract. There is no persistence here by design; a future crate
/// can snapshot an engine's state through [`Engine::storage`] and
/// re-deploy the same bytecode to restore it.
pub struct Engine {
    contracts: HashMap<ContractId, Contract>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// An empty engine with no contracts deployed.
    pub fn new() -> Self {
        Self {
            contracts: HashMap::new(),
        }
    }

    /// Number of contracts deployed in this engine.
    pub fn contract_count(&self) -> usize {
        self.contracts.len()
    }

    /// Validate `bytecode` and deploy it, returning its id (the SHA-256
    /// of the bytecode). Deploying the same bytecode twice returns the
    /// same id and does **not** reset the contract's storage — the
    /// first deployment's state survives.
    pub fn deploy(&mut self, bytecode: Vec<u8>) -> Result<ContractId, ContractError> {
        validate(&bytecode)?;
        let id = ContractId(sha256(&bytecode));
        self.contracts.entry(id).or_insert(Contract {
            bytecode,
            storage: [0u64; STORAGE_SLOTS],
        });
        Ok(id)
    }

    /// Read one storage slot of a deployed contract. Returns `None` if
    /// the contract is not deployed. Unset slots read as `0`.
    pub fn storage(&self, id: &ContractId, key: u8) -> Option<u64> {
        self.contracts.get(id).map(|c| c.storage[key as usize])
    }

    /// Call a deployed contract with an input stack and a gas budget.
    ///
    /// On success the call's storage writes are committed and the
    /// result is returned. On any error the call is atomic: its writes
    /// are discarded and the contract's storage is exactly as before.
    pub fn call(
        &mut self,
        id: &ContractId,
        input: &[u64],
        gas_limit: u64,
    ) -> Result<CallResult, ContractError> {
        let contract = self
            .contracts
            .get(id)
            .ok_or(ContractError::ContractNotFound(*id))?;
        if input.len() > MAX_STACK {
            return Err(ContractError::StackOverflow);
        }
        let code = contract.bytecode.clone();
        let mut storage = contract.storage;
        let mut stack: Vec<u64> = input.to_vec();
        let mut gas_remaining = gas_limit;
        let mut gas_used = 0u64;
        let mut writes = 0usize;
        let mut pc = 0usize;

        loop {
            let op = *code.get(pc).ok_or(ContractError::RanOffEnd)?;
            let cost = gas_cost(op);
            if gas_remaining < cost {
                return Err(ContractError::OutOfGas(gas_limit));
            }
            gas_remaining -= cost;
            gas_used += cost;

            match op {
                OP_HALT => {
                    // Commit the call's writes; nothing else to do.
                    if let Some(c) = self.contracts.get_mut(id) {
                        c.storage = storage;
                    }
                    return Ok(CallResult {
                        stack,
                        gas_used,
                        storage_writes: writes,
                    });
                }
                OP_PUSH => {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&code[pc + 1..pc + 9]);
                    push(&mut stack, u64::from_le_bytes(bytes))?;
                    pc += 9;
                }
                OP_ADD | OP_SUB | OP_MUL | OP_DIV | OP_EQ | OP_LT => {
                    let b = pop(&mut stack, op)?;
                    let a = pop(&mut stack, op)?;
                    let value = match op {
                        OP_ADD => a.wrapping_add(b),
                        OP_SUB => a.wrapping_sub(b),
                        OP_MUL => a.wrapping_mul(b),
                        OP_DIV => {
                            if b == 0 {
                                return Err(ContractError::DivideByZero);
                            }
                            a / b
                        }
                        OP_EQ => u64::from(a == b),
                        _ => u64::from(a < b),
                    };
                    // The two pops freed space, so this push cannot overflow.
                    stack.push(value);
                    pc += 1;
                }
                OP_JUMP => {
                    pc = jump_target(&code, pc)?;
                }
                OP_JUMPIF => {
                    let cond = pop(&mut stack, op)?;
                    if cond != 0 {
                        pc = jump_target(&code, pc)?;
                    } else {
                        pc += 3;
                    }
                }
                OP_STORE => {
                    let key = code[pc + 1] as usize;
                    let value = pop(&mut stack, op)?;
                    storage[key] = value;
                    writes += 1;
                    pc += 2;
                }
                OP_LOAD => {
                    let key = code[pc + 1] as usize;
                    push(&mut stack, storage[key])?;
                    pc += 2;
                }
                _ => {
                    // Unreachable: deploy validated every opcode.
                    debug_assert!(false, "validated bytecode contained opcode {op:#x}");
                    return Err(ContractError::UnknownOpcode(op));
                }
            }
        }
    }
}

/// Push a value, refusing when the stack is full.
fn push(stack: &mut Vec<u64>, value: u64) -> Result<(), ContractError> {
    if stack.len() >= MAX_STACK {
        return Err(ContractError::StackOverflow);
    }
    stack.push(value);
    Ok(())
}

/// Pop a value, refusing when the stack is empty.
fn pop(stack: &mut Vec<u64>, op: u8) -> Result<u64, ContractError> {
    stack.pop().ok_or(ContractError::StackUnderflow { op })
}

/// Read the 2-byte little-endian jump target after a JUMP/JUMPIF at `pc`.
fn jump_target(code: &[u8], pc: usize) -> Result<usize, ContractError> {
    let lo = code[pc + 1] as usize;
    let hi = code[pc + 2] as usize;
    Ok(lo | (hi << 8))
}

/// Validate bytecode at deploy time. Walks the program once, decoding
/// every instruction: unknown opcodes, truncated immediates, and jump
/// targets that land mid-instruction are refused, and a program with
/// no `HALT` is refused.
fn validate(code: &[u8]) -> Result<(), ContractError> {
    if code.is_empty() {
        return Err(ContractError::EmptyProgram);
    }
    let mut starts: HashSet<usize> = HashSet::new();
    let mut jumps: Vec<usize> = Vec::new();
    let mut saw_halt = false;
    let mut pc = 0usize;
    while pc < code.len() {
        let op = code[pc];
        let len = instruction_len(op).ok_or(ContractError::UnknownOpcode(op))?;
        if pc + len > code.len() {
            return Err(ContractError::TruncatedImmediate { op, at: pc });
        }
        starts.insert(pc);
        match op {
            OP_HALT => saw_halt = true,
            OP_JUMP | OP_JUMPIF => {
                let target = (code[pc + 1] as usize) | ((code[pc + 2] as usize) << 8);
                jumps.push(target);
            }
            _ => {}
        }
        pc += len;
    }
    if !saw_halt {
        return Err(ContractError::MissingHalt);
    }
    for target in jumps {
        if !starts.contains(&target) {
            return Err(ContractError::InvalidJumpTarget(target));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny assembler for tests: builds programs without hand-counting
    /// byte offsets. `jump`/`jumpif` take a placeholder; call `patch`
    /// with the real target once its offset is known.
    struct B {
        code: Vec<u8>,
    }

    impl B {
        fn new() -> Self {
            Self { code: Vec::new() }
        }

        fn push(&mut self, v: u64) -> &mut Self {
            self.code.push(OP_PUSH);
            self.code.extend_from_slice(&v.to_le_bytes());
            self
        }

        fn op(&mut self, op: u8) -> &mut Self {
            self.code.push(op);
            self
        }

        fn add(&mut self) -> &mut Self {
            self.op(OP_ADD)
        }

        fn sub(&mut self) -> &mut Self {
            self.op(OP_SUB)
        }

        fn mul(&mut self) -> &mut Self {
            self.op(OP_MUL)
        }

        fn div(&mut self) -> &mut Self {
            self.op(OP_DIV)
        }

        fn eq(&mut self) -> &mut Self {
            self.op(OP_EQ)
        }

        fn lt(&mut self) -> &mut Self {
            self.op(OP_LT)
        }

        fn jump(&mut self, target: usize) -> &mut Self {
            self.code.push(OP_JUMP);
            self.code.extend_from_slice(&(target as u16).to_le_bytes());
            self
        }

        fn jump_placeholder(&mut self) -> usize {
            self.code.push(OP_JUMP);
            let at = self.code.len();
            self.code.extend_from_slice(&[0, 0]);
            at
        }

        fn jumpif_placeholder(&mut self) -> usize {
            self.code.push(OP_JUMPIF);
            let at = self.code.len();
            self.code.extend_from_slice(&[0, 0]);
            at
        }

        fn patch(&mut self, at: usize, target: usize) {
            let bytes = (target as u16).to_le_bytes();
            self.code[at] = bytes[0];
            self.code[at + 1] = bytes[1];
        }

        fn store(&mut self, key: u8) -> &mut Self {
            self.code.push(OP_STORE);
            self.code.push(key);
            self
        }

        fn load(&mut self, key: u8) -> &mut Self {
            self.code.push(OP_LOAD);
            self.code.push(key);
            self
        }

        fn halt(&mut self) -> &mut Self {
            self.op(OP_HALT)
        }

        fn pos(&self) -> usize {
            self.code.len()
        }

        fn bytes(self) -> Vec<u8> {
            self.code
        }
    }

    fn counter_code() -> Vec<u8> {
        let mut b = B::new();
        b.load(0).push(1).add().store(0).load(0).halt();
        b.bytes()
    }

    #[test]
    fn arithmetic_add_sub_mul_div() {
        let mut engine = Engine::new();
        let mut b = B::new();
        b.push(20).push(22).add(); // 42
        b.push(100).push(58).sub(); // 42
        b.push(6).push(7).mul(); // 42
        b.push(84).push(2).div(); // 42
        b.halt();
        let id = engine.deploy(b.bytes()).unwrap();
        let result = engine.call(&id, &[], 1_000).unwrap();
        assert_eq!(result.stack, vec![42, 42, 42, 42]);
    }

    #[test]
    fn arithmetic_wraps_on_overflow() {
        let mut engine = Engine::new();
        let mut b = B::new();
        b.push(u64::MAX).push(1).add(); // wraps to 0
        b.push(0).push(1).sub(); // wraps to u64::MAX
        b.push(u64::MAX).push(2).mul(); // wraps to u64::MAX - 1
        b.halt();
        let id = engine.deploy(b.bytes()).unwrap();
        let result = engine.call(&id, &[], 1_000).unwrap();
        assert_eq!(result.stack, vec![0, u64::MAX, u64::MAX - 1]);
    }

    #[test]
    fn divide_by_zero_is_a_clean_error() {
        let mut engine = Engine::new();
        let mut b = B::new();
        b.push(1).push(0).div().halt();
        let id = engine.deploy(b.bytes()).unwrap();
        assert_eq!(
            engine.call(&id, &[], 1_000),
            Err(ContractError::DivideByZero)
        );
    }

    #[test]
    fn failed_call_writes_nothing() {
        let mut engine = Engine::new();
        // Store 99 under key 5, then divide by zero: the write must roll back.
        let mut b = B::new();
        b.push(99).store(5).push(1).push(0).div().halt();
        let id = engine.deploy(b.bytes()).unwrap();
        assert_eq!(
            engine.call(&id, &[], 1_000),
            Err(ContractError::DivideByZero)
        );
        assert_eq!(engine.storage(&id, 5), Some(0));
        // A later good call still works.
        let mut good = B::new();
        good.push(7).store(5).halt();
        let id2 = engine.deploy(good.bytes()).unwrap();
        assert_ne!(id, id2);
        engine.call(&id2, &[], 1_000).unwrap();
        assert_eq!(engine.storage(&id2, 5), Some(7));
    }

    #[test]
    fn eq_and_lt_compare_top_two() {
        let mut engine = Engine::new();
        let mut b = B::new();
        b.push(3).push(3).eq(); // 1
        b.push(3).push(4).eq(); // 0
        b.push(3).push(4).lt(); // 1 (3 < 4)
        b.push(4).push(3).lt(); // 0 (4 < 3 is false)
        b.halt();
        let id = engine.deploy(b.bytes()).unwrap();
        let result = engine.call(&id, &[], 1_000).unwrap();
        assert_eq!(result.stack, vec![1, 0, 1, 0]);
    }

    #[test]
    fn jump_skips_code() {
        let mut engine = Engine::new();
        let mut b = B::new();
        let skip = b.jump_placeholder();
        b.push(111);
        let target = b.pos();
        b.patch(skip, target);
        b.push(222).halt();
        let id = engine.deploy(b.bytes()).unwrap();
        let result = engine.call(&id, &[], 1_000).unwrap();
        assert_eq!(result.stack, vec![222]);
    }

    #[test]
    fn jumpif_takes_both_branches() {
        // cond on the input stack; 0 -> 111, nonzero -> 222.
        let mut engine = Engine::new();
        let mut b = B::new();
        let jif = b.jumpif_placeholder();
        b.push(111);
        let end = b.jump_placeholder();
        let then = b.pos();
        b.patch(jif, then);
        b.push(222);
        let after = b.pos();
        b.patch(end, after);
        b.halt();
        let id = engine.deploy(b.bytes()).unwrap();
        assert_eq!(
            engine.call(&id, &[0], 1_000).unwrap().stack,
            vec![111]
        );
        assert_eq!(
            engine.call(&id, &[1], 1_000).unwrap().stack,
            vec![222]
        );
        assert_eq!(
            engine.call(&id, &[999], 1_000).unwrap().stack,
            vec![222]
        );
    }

    #[test]
    fn counter_increments_and_persists() {
        let mut engine = Engine::new();
        let id = engine.deploy(counter_code()).unwrap();
        assert_eq!(engine.storage(&id, 0), Some(0));
        let first = engine.call(&id, &[], 1_000).unwrap();
        assert_eq!(first.stack, vec![1]);
        assert_eq!(first.storage_writes, 1);
        // LOAD 10 + PUSH 3 + ADD 5 + STORE 20 + LOAD 10 + HALT 1 = 49 gas.
        assert_eq!(first.gas_used, 49);
        let second = engine.call(&id, &[], 1_000).unwrap();
        assert_eq!(second.stack, vec![2]);
        let third = engine.call(&id, &[], 1_000).unwrap();
        assert_eq!(third.stack, vec![3]);
        assert_eq!(engine.storage(&id, 0), Some(3));
    }

    #[test]
    fn gas_exhaustion_aborts_and_reports() {
        // An infinite loop: JUMP back to itself. Gas must stop it.
        let mut engine = Engine::new();
        let mut b = B::new();
        let start = b.pos();
        b.jump(start).halt();
        let id = engine.deploy(b.bytes()).unwrap();
        let limit = 100u64;
        assert_eq!(
            engine.call(&id, &[], limit),
            Err(ContractError::OutOfGas(limit))
        );
        // The loop wrote nothing and the contract is still callable.
        assert_eq!(engine.storage(&id, 0), Some(0));
        assert_eq!(
            engine.call(&id, &[], limit),
            Err(ContractError::OutOfGas(limit))
        );
    }

    #[test]
    fn unknown_opcode_refused_at_deploy() {
        let mut engine = Engine::new();
        assert_eq!(
            engine.deploy(vec![0xFF, OP_HALT]),
            Err(ContractError::UnknownOpcode(0xFF))
        );
        assert_eq!(engine.contract_count(), 0);
    }

    #[test]
    fn empty_program_and_missing_halt_refused() {
        let mut engine = Engine::new();
        assert_eq!(engine.deploy(vec![]), Err(ContractError::EmptyProgram));
        let mut b = B::new();
        b.push(1).add();
        assert_eq!(engine.deploy(b.bytes()), Err(ContractError::MissingHalt));
    }

    #[test]
    fn truncated_immediates_refused() {
        let mut engine = Engine::new();
        // PUSH with only 2 of its 8 immediate bytes.
        assert!(matches!(
            engine.deploy(vec![OP_PUSH, 1, 2]),
            Err(ContractError::TruncatedImmediate { op: OP_PUSH, at: 0 })
        ));
        // JUMP with only 1 of its 2 target bytes.
        assert!(matches!(
            engine.deploy(vec![OP_JUMP, 0x01]),
            Err(ContractError::TruncatedImmediate { op: OP_JUMP, at: 0 })
        ));
        // STORE with no key byte.
        assert!(matches!(
            engine.deploy(vec![OP_STORE]),
            Err(ContractError::TruncatedImmediate { op: OP_STORE, at: 0 })
        ));
    }

    #[test]
    fn invalid_jump_targets_refused() {
        let mut engine = Engine::new();
        // JUMP 1: offset 1 is not an instruction boundary (starts are 0 and 3).
        let mut b = B::new();
        b.jump(1).halt();
        assert_eq!(
            engine.deploy(b.bytes()),
            Err(ContractError::InvalidJumpTarget(1))
        );
        // JUMP into the middle of a PUSH immediate.
        let mut b = B::new();
        b.push(0xDEAD_BEEF).jump(2).halt();
        assert_eq!(
            engine.deploy(b.bytes()),
            Err(ContractError::InvalidJumpTarget(2))
        );
        // JUMP past the end of the program.
        let mut b = B::new();
        b.jump(99).halt();
        assert_eq!(
            engine.deploy(b.bytes()),
            Err(ContractError::InvalidJumpTarget(99))
        );
    }

    #[test]
    fn running_off_the_end_is_an_error() {
        // PUSH 1; PUSH 2; JUMP to the final ADD; HALT is dead code after it.
        // The ADD runs, then execution falls past the last byte.
        let mut engine = Engine::new();
        let mut b = B::new();
        b.push(1).push(2);
        let j = b.jump_placeholder();
        b.halt();
        let add_at = b.pos();
        b.patch(j, add_at);
        b.add();
        let id = engine.deploy(b.bytes()).unwrap();
        assert_eq!(engine.call(&id, &[], 1_000), Err(ContractError::RanOffEnd));
    }

    #[test]
    fn two_contracts_storage_is_isolated() {
        let mut engine = Engine::new();
        let a = engine.deploy(counter_code()).unwrap();
        let b = engine.deploy(counter_code()).unwrap();
        // Same bytecode deploys to the same id — so use a different
        // program for the second contract (counter on key 1 instead).
        assert_eq!(a, b);
        let mut other = B::new();
        other.load(1).push(1).add().store(1).halt();
        let b = engine.deploy(other.bytes()).unwrap();
        assert_ne!(a, b);

        engine.call(&a, &[], 1_000).unwrap();
        engine.call(&a, &[], 1_000).unwrap();
        assert_eq!(engine.storage(&a, 0), Some(2));
        assert_eq!(engine.storage(&a, 1), Some(0));
        assert_eq!(engine.storage(&b, 0), Some(0));
        assert_eq!(engine.storage(&b, 1), Some(0));

        engine.call(&b, &[], 1_000).unwrap();
        assert_eq!(engine.storage(&b, 1), Some(1));
        assert_eq!(engine.storage(&a, 0), Some(2));
    }

    #[test]
    fn redeploying_the_same_bytecode_keeps_storage() {
        let mut engine = Engine::new();
        let id = engine.deploy(counter_code()).unwrap();
        engine.call(&id, &[], 1_000).unwrap();
        assert_eq!(engine.storage(&id, 0), Some(1));
        let again = engine.deploy(counter_code()).unwrap();
        assert_eq!(again, id);
        assert_eq!(engine.storage(&id, 0), Some(1));
        assert_eq!(engine.contract_count(), 1);
    }

    #[test]
    fn determinism_same_call_twice_is_identical() {
        let mut engine = Engine::new();
        let id = engine.deploy(counter_code()).unwrap();
        // Two fresh engines, same bytecode, same call: identical results.
        let mut other = Engine::new();
        let other_id = other.deploy(counter_code()).unwrap();
        assert_eq!(id, other_id);
        let r1 = engine.call(&id, &[5], 1_000).unwrap();
        let r2 = other.call(&other_id, &[5], 1_000).unwrap();
        assert_eq!(r1, r2);
        // Same call again in the same engine: same shape, storage advanced.
        // (The counter's own storage changes, so the stack differs — but
        // a pure program must give byte-identical results.)
        let mut pure = B::new();
        pure.push(6).push(7).mul().halt();
        let pure_id = engine.deploy(pure.bytes()).unwrap();
        let p1 = engine.call(&pure_id, &[], 500).unwrap();
        let p2 = engine.call(&pure_id, &[], 500).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p1.stack, vec![42]);
    }

    #[test]
    fn input_stack_feeds_the_program() {
        let mut engine = Engine::new();
        let mut b = B::new();
        b.push(3).add().halt();
        let id = engine.deploy(b.bytes()).unwrap();
        let result = engine.call(&id, &[7], 1_000).unwrap();
        assert_eq!(result.stack, vec![10]);
    }

    #[test]
    fn stack_underflow_is_a_clean_error() {
        let mut engine = Engine::new();
        let mut b = B::new();
        b.add().halt();
        let id = engine.deploy(b.bytes()).unwrap();
        assert_eq!(
            engine.call(&id, &[], 1_000),
            Err(ContractError::StackUnderflow { op: OP_ADD })
        );
        // Popping the last value of a one-element stack also underflows.
        let mut b = B::new();
        b.push(1).add().halt();
        let id = engine.deploy(b.bytes()).unwrap();
        assert_eq!(
            engine.call(&id, &[], 1_000),
            Err(ContractError::StackUnderflow { op: OP_ADD })
        );
    }

    #[test]
    fn stack_overflow_is_a_clean_error() {
        let mut engine = Engine::new();
        let mut b = B::new();
        b.halt();
        let id = engine.deploy(b.bytes()).unwrap();
        let huge = vec![0u64; MAX_STACK + 1];
        assert_eq!(
            engine.call(&id, &huge, 1_000),
            Err(ContractError::StackOverflow)
        );
        // Filling the stack with pushes also overflows.
        let mut fill = B::new();
        for _ in 0..=MAX_STACK {
            fill.push(1);
        }
        fill.halt();
        let id = engine.deploy(fill.bytes()).unwrap();
        assert_eq!(
            engine.call(&id, &[], 1_000_000),
            Err(ContractError::StackOverflow)
        );
    }

    #[test]
    fn calling_an_unknown_contract_fails_cleanly() {
        let mut engine = Engine::new();
        let ghost = ContractId(sha256(b"nothing deployed with this bytecode"));
        assert_eq!(
            engine.call(&ghost, &[], 1_000),
            Err(ContractError::ContractNotFound(ghost))
        );
        assert_eq!(engine.storage(&ghost, 0), None);
    }

    #[test]
    fn gas_costs_are_documented_and_charged() {
        // PUSH 3 + PUSH 3 + ADD 5 + HALT 1 = 12.
        let mut b = B::new();
        b.push(1).push(2).add().halt();
        let code = b.bytes();

        let mut engine = Engine::new();
        let id = engine.deploy(code.clone()).unwrap();
        let result = engine.call(&id, &[], 1_000).unwrap();
        assert_eq!(result.gas_used, 12);
        // Exactly enough gas succeeds; one less fails.
        let mut engine = Engine::new();
        let id = engine.deploy(code).unwrap();
        assert!(engine.call(&id, &[], 12).is_ok());
        assert_eq!(
            engine.call(&id, &[], 11),
            Err(ContractError::OutOfGas(11))
        );
    }

    #[test]
    fn unset_keys_read_as_zero() {
        let mut engine = Engine::new();
        let mut b = B::new();
        b.load(200).halt();
        let id = engine.deploy(b.bytes()).unwrap();
        let result = engine.call(&id, &[], 1_000).unwrap();
        assert_eq!(result.stack, vec![0]);
    }
}
