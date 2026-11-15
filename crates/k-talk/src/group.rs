//! Groups for K•Talk.
//!
//! A group is a signed record: a human-readable id, a name, and a
//! creator. The creator is the only peer who can change membership,
//! and every membership change is a signed record the creator issues.
//! A group message is signed by its author and addressed to the
//! group; only current members may send to or receive from a group.
//!
//! Wire formats (explicit binary, no JSON):
//! - Group record: `KGROUP1` + creator key (32) + creator name length
//!   (u16 LE) + creator name + id length (u16 LE) + id + name length
//!   (u16 LE) + name + signature (64). The signature covers
//!   `KGROUP1 || creator key || id || name`.
//! - Membership change: `KMEMB1` + group id length (u16 LE) + group id
//!   + peer key (32) + peer name length (u16 LE) + peer name + action
//!     (u8: 1 = add, 2 = remove) + timestamp (u64 LE) + signature (64).
//!     The signature covers `KMEMB1 || group id || peer key || action
//!     || timestamp` and is made by the group's creator.
//! - Group message: `KTALKG` + group id length (u16 LE) + group id +
//!   sender key (32) + sender name length (u16 LE) + sender name +
//!   timestamp (u64 LE) + sender sequence (u64 LE) + text length
//!   (u32 LE) + text + signature (64). The signature covers
//!   `KTALKG || group id || sender key || timestamp || seq || text`.

use std::collections::HashSet;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_crypto::{sha256, Ed25519Signature};
use kernel_identity::{Identity, PeerId};

use crate::TalkError;

/// Magic bytes at the start of every encoded group record.
const GROUP_MAGIC: &[u8; 7] = b"KGROUP1";
/// Domain separator for group-record signatures.
const GROUP_SIGN_DOMAIN: &[u8; 7] = b"KGROUP1";
/// Magic bytes at the start of every encoded membership change.
const MEMBERSHIP_MAGIC: &[u8; 6] = b"KMEMB1";
/// Domain separator for membership-change signatures.
const MEMBERSHIP_SIGN_DOMAIN: &[u8; 6] = b"KMEMB1";
/// Magic bytes at the start of every encoded group message.
const GROUP_MESSAGE_MAGIC: &[u8; 6] = b"KTALKG";
/// Domain separator for group-message signatures.
const GROUP_MESSAGE_SIGN_DOMAIN: &[u8; 6] = b"KTALKG";

/// Maximum bytes for a group id or a group name.
pub const MAX_GROUP_ID_LEN: usize = 128;
pub const MAX_GROUP_NAME_LEN: usize = 256;

/// Unix time in seconds.
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn write_peer(out: &mut Vec<u8>, peer: &PeerId) {
    let name = peer.display_name().as_bytes();
    out.extend_from_slice(peer.public_key().as_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(name);
}

/// Read a peer id (key + name) starting at `off`. Returns the peer and
/// the offset just past it, or an error if the bytes run out or the
/// name is not valid UTF-8.
fn read_peer(bytes: &[u8], off: usize) -> Result<(PeerId, usize), TalkError> {
    if bytes.len() < off + 32 + 2 {
        return Err(TalkError::InvalidGroup(
            "record truncates a peer id".to_string(),
        ));
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&bytes[off..off + 32]);
    let key = kernel_crypto::Ed25519PublicKey::from_bytes(&key)
        .map_err(|e| TalkError::InvalidGroup(e.to_string()))?;
    let name_len =
        u16::from_le_bytes(bytes[off + 32..off + 34].try_into().expect("2-byte slice")) as usize;
    if bytes.len() < off + 34 + name_len {
        return Err(TalkError::InvalidGroup(
            "record truncates a peer name".to_string(),
        ));
    }
    let name = String::from_utf8(bytes[off + 34..off + 34 + name_len].to_vec())
        .map_err(|_| TalkError::InvalidGroup("peer name is not UTF-8".to_string()))?;
    let peer = PeerId::new(key, &name)?;
    Ok((peer, off + 34 + name_len))
}

fn write_len_prefixed(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u16).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn read_len_prefixed(bytes: &[u8], off: usize, what: &str) -> Result<(String, usize), TalkError> {
    if bytes.len() < off + 2 {
        return Err(TalkError::InvalidGroup(format!(
            "record truncates the {what} length"
        )));
    }
    let len = u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
    if bytes.len() < off + 2 + len {
        return Err(TalkError::InvalidGroup(format!(
            "record truncates the {what}"
        )));
    }
    let s = String::from_utf8(bytes[off + 2..off + 2 + len].to_vec())
        .map_err(|_| TalkError::InvalidGroup(format!("{what} is not UTF-8")))?;
    Ok((s, off + 2 + len))
}

fn validate_group_id(id: &str) -> Result<(), TalkError> {
    if id.is_empty() {
        return Err(TalkError::InvalidGroup(
            "group id must not be empty".to_string(),
        ));
    }
    if id.len() > MAX_GROUP_ID_LEN {
        return Err(TalkError::InvalidGroup(format!(
            "group id is {} bytes, max is {MAX_GROUP_ID_LEN}",
            id.len()
        )));
    }
    Ok(())
}

fn validate_group_name(name: &str) -> Result<(), TalkError> {
    if name.is_empty() {
        return Err(TalkError::InvalidGroup(
            "group name must not be empty".to_string(),
        ));
    }
    if name.len() > MAX_GROUP_NAME_LEN {
        return Err(TalkError::InvalidGroup(format!(
            "group name is {} bytes, max is {MAX_GROUP_NAME_LEN}",
            name.len()
        )));
    }
    Ok(())
}

/// A signed group record. The creator is the only peer who can change
/// membership; the creator is always a member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    /// Human-readable id, e.g. `korn-builders`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Who created the group (and who may change membership).
    pub creator: PeerId,
    /// Current members, always including the creator.
    members: HashSet<PeerId>,
    /// The creator's signature over id, name, and creator key.
    pub signature: Ed25519Signature,
}

fn group_sign_bytes(creator: &PeerId, id: &str, name: &str) -> Vec<u8> {
    let mut msg = Vec::with_capacity(GROUP_SIGN_DOMAIN.len() + 32 + 2 + id.len() + 2 + name.len());
    msg.extend_from_slice(GROUP_SIGN_DOMAIN);
    msg.extend_from_slice(creator.public_key().as_bytes());
    msg.extend_from_slice(&(id.len() as u16).to_le_bytes());
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(&(name.len() as u16).to_le_bytes());
    msg.extend_from_slice(name.as_bytes());
    msg
}

impl Group {
    /// Create a group. The creator is its first member.
    pub fn create(creator: &Identity, id: &str, name: &str) -> Result<Self, TalkError> {
        validate_group_id(id)?;
        validate_group_name(name)?;
        let peer_id = creator.peer_id().clone();
        let signature = creator.sign(&group_sign_bytes(&peer_id, id, name));
        let mut members = HashSet::new();
        members.insert(peer_id.clone());
        Ok(Self {
            id: id.to_string(),
            name: name.to_string(),
            creator: peer_id,
            members,
            signature,
        })
    }

    /// True if the creator's signature verifies for this id, name, and
    /// creator key. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = group_sign_bytes(&self.creator, &self.id, &self.name);
        self.creator.verify(&msg, &self.signature).is_ok()
    }

    /// True if `peer` is a current member.
    pub fn is_member(&self, peer: &PeerId) -> bool {
        self.members.contains(peer)
    }

    /// Current members, in no particular order.
    pub fn members(&self) -> Vec<PeerId> {
        self.members.iter().cloned().collect()
    }

    /// Number of current members.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Apply a signed membership change. Only a change signed by the
    /// creator, for this group, is accepted. Adding a member who is
    /// already in is a no-op; removing a non-member is a no-op.
    /// Removing the creator is refused — the creator is always a
    /// member.
    pub fn apply_change(&mut self, change: &MembershipChange) -> Result<(), TalkError> {
        if change.group_id != self.id {
            return Err(TalkError::InvalidGroup(format!(
                "membership change is for group {}, not {}",
                change.group_id, self.id
            )));
        }
        if !change.verify(&self.creator) {
            return Err(TalkError::BadSignature);
        }
        match change.action {
            MembershipAction::Add => {
                self.members.insert(change.peer.clone());
            }
            MembershipAction::Remove => {
                if change.peer == self.creator {
                    return Err(TalkError::InvalidGroup(
                        "the creator cannot be removed from their own group".to_string(),
                    ));
                }
                self.members.remove(&change.peer);
            }
        }
        Ok(())
    }

    /// Encode to the explicit binary wire format.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            7 + 32
                + 2
                + self.creator.display_name().len()
                + 2
                + self.id.len()
                + 2
                + self.name.len()
                + 64,
        );
        out.extend_from_slice(GROUP_MAGIC);
        write_peer(&mut out, &self.creator);
        write_len_prefixed(&mut out, &self.id);
        write_len_prefixed(&mut out, &self.name);
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a group record produced by [`Group::encode`]. Decoding
    /// does not verify the signature — call [`Group::verify`] for that.
    pub fn decode(bytes: &[u8]) -> Result<Self, TalkError> {
        if bytes.len() < 7 + 32 + 2 + 2 + 2 + 64 || &bytes[..7] != GROUP_MAGIC {
            return Err(TalkError::InvalidGroup("missing KGROUP1 magic".to_string()));
        }
        let (creator, off) = read_peer(bytes, 7)?;
        let (id, off) = read_len_prefixed(bytes, off, "group id")?;
        let (name, off) = read_len_prefixed(bytes, off, "group name")?;
        if bytes.len() != off + 64 {
            return Err(TalkError::InvalidGroup(format!(
                "group record is {} bytes, expected {}",
                bytes.len(),
                off + 64
            )));
        }
        validate_group_id(&id)?;
        validate_group_name(&name)?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let mut members = HashSet::new();
        members.insert(creator.clone());
        Ok(Self {
            id,
            name,
            creator,
            members,
            signature,
        })
    }
}

/// Adding or removing a member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MembershipAction {
    Add = 1,
    Remove = 2,
}

impl MembershipAction {
    fn from_byte(b: u8) -> Result<Self, TalkError> {
        match b {
            1 => Ok(MembershipAction::Add),
            2 => Ok(MembershipAction::Remove),
            other => Err(TalkError::InvalidGroup(format!(
                "unknown membership action byte {other}"
            ))),
        }
    }
}

/// A signed membership change, issued by the group's creator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MembershipChange {
    /// The group this change applies to.
    pub group_id: String,
    /// The peer being added or removed.
    pub peer: PeerId,
    /// What happens to them.
    pub action: MembershipAction,
    /// Unix seconds when the change was issued.
    pub timestamp: u64,
    /// The creator's signature over group id, peer, action, timestamp.
    pub signature: Ed25519Signature,
}

fn membership_sign_bytes(
    group_id: &str,
    peer: &PeerId,
    action: MembershipAction,
    timestamp: u64,
) -> Vec<u8> {
    let mut msg =
        Vec::with_capacity(MEMBERSHIP_SIGN_DOMAIN.len() + 2 + group_id.len() + 32 + 1 + 8);
    msg.extend_from_slice(MEMBERSHIP_SIGN_DOMAIN);
    msg.extend_from_slice(&(group_id.len() as u16).to_le_bytes());
    msg.extend_from_slice(group_id.as_bytes());
    msg.extend_from_slice(peer.public_key().as_bytes());
    msg.push(action as u8);
    msg.extend_from_slice(&timestamp.to_le_bytes());
    msg
}

impl MembershipChange {
    /// Issue a membership change. Only the creator should call this;
    /// verification against the group record's creator catches anyone
    /// else.
    pub fn new(
        creator: &Identity,
        group_id: &str,
        peer: &PeerId,
        action: MembershipAction,
    ) -> Result<Self, TalkError> {
        validate_group_id(group_id)?;
        let timestamp = now_unix_secs();
        let signature = creator.sign(&membership_sign_bytes(group_id, peer, action, timestamp));
        Ok(Self {
            group_id: group_id.to_string(),
            peer: peer.clone(),
            action,
            timestamp,
            signature,
        })
    }

    /// True if the signature verifies against `creator` for this exact
    /// group, peer, action, and timestamp.
    pub fn verify(&self, creator: &PeerId) -> bool {
        let msg = membership_sign_bytes(&self.group_id, &self.peer, self.action, self.timestamp);
        creator.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary wire format.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            6 + 2 + self.group_id.len() + 32 + 2 + self.peer.display_name().len() + 1 + 8 + 64,
        );
        out.extend_from_slice(MEMBERSHIP_MAGIC);
        write_len_prefixed(&mut out, &self.group_id);
        write_peer(&mut out, &self.peer);
        out.push(self.action as u8);
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a membership change produced by
    /// [`MembershipChange::encode`]. Decoding does not verify the
    /// signature — call [`MembershipChange::verify`] for that.
    pub fn decode(bytes: &[u8]) -> Result<Self, TalkError> {
        if bytes.len() < 6 + 2 + 32 + 2 + 1 + 8 + 64 || &bytes[..6] != MEMBERSHIP_MAGIC {
            return Err(TalkError::InvalidGroup("missing KMEMB1 magic".to_string()));
        }
        let (group_id, off) = read_len_prefixed(bytes, 6, "group id")?;
        let (peer, off) = read_peer(bytes, off)?;
        if bytes.len() < off + 1 + 8 + 64 {
            return Err(TalkError::InvalidGroup(
                "membership change truncates the action or timestamp".to_string(),
            ));
        }
        let action = MembershipAction::from_byte(bytes[off])?;
        let timestamp =
            u64::from_le_bytes(bytes[off + 1..off + 9].try_into().expect("8-byte slice"));
        let off = off + 9;
        if bytes.len() != off + 64 {
            return Err(TalkError::InvalidGroup(format!(
                "membership change is {} bytes, expected {}",
                bytes.len(),
                off + 64
            )));
        }
        validate_group_id(&group_id)?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        Ok(Self {
            group_id,
            peer,
            action,
            timestamp,
            signature,
        })
    }
}

/// A signed message addressed to a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupMessage {
    /// The group this message is for.
    pub group_id: String,
    /// Who sent it.
    pub sender: PeerId,
    /// Human-readable text.
    pub text: String,
    /// Unix seconds when it was sent.
    pub timestamp: u64,
    /// The sender's per-sender sequence number (shared with direct
    /// messages: it orders this sender's messages).
    pub seq: u64,
    /// The sender's signature over group id, sender, timestamp, seq,
    /// text.
    pub signature: Ed25519Signature,
}

impl fmt::Display for GroupMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "group {} message from {}: {}",
            self.group_id,
            self.sender.display_name(),
            self.text
        )
    }
}

fn group_message_sign_bytes(
    group_id: &str,
    sender: &PeerId,
    timestamp: u64,
    seq: u64,
    text: &str,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(
        GROUP_MESSAGE_SIGN_DOMAIN.len() + 2 + group_id.len() + 32 + 8 + 8 + text.len(),
    );
    msg.extend_from_slice(GROUP_MESSAGE_SIGN_DOMAIN);
    msg.extend_from_slice(&(group_id.len() as u16).to_le_bytes());
    msg.extend_from_slice(group_id.as_bytes());
    msg.extend_from_slice(sender.public_key().as_bytes());
    msg.extend_from_slice(&timestamp.to_le_bytes());
    msg.extend_from_slice(&seq.to_le_bytes());
    msg.extend_from_slice(text.as_bytes());
    msg
}

impl GroupMessage {
    /// Create and sign a group message from `sender`.
    pub fn new(sender: &Identity, group_id: &str, text: &str, seq: u64) -> Result<Self, TalkError> {
        validate_group_id(group_id)?;
        let timestamp = now_unix_secs();
        let peer_id = sender.peer_id().clone();
        let msg = group_message_sign_bytes(group_id, &peer_id, timestamp, seq, text);
        let signature = sender.sign(&msg);
        Ok(Self {
            group_id: group_id.to_string(),
            sender: peer_id,
            text: text.to_string(),
            timestamp,
            seq,
            signature,
        })
    }

    /// True if the signature verifies against the sender for this exact
    /// group, timestamp, sequence, and text.
    pub fn verify(&self) -> bool {
        let msg = group_message_sign_bytes(
            &self.group_id,
            &self.sender,
            self.timestamp,
            self.seq,
            &self.text,
        );
        self.sender.verify(&msg, &self.signature).is_ok()
    }

    /// The message's id: the SHA-256 of its exact encoding. Two nodes
    /// holding the same message compute the same id.
    pub fn id(&self) -> [u8; 32] {
        sha256(&self.encode())
    }

    /// Encode to the explicit binary wire format.
    pub fn encode(&self) -> Vec<u8> {
        let text_bytes = self.text.as_bytes();
        let mut out = Vec::with_capacity(
            6 + 2
                + self.group_id.len()
                + 32
                + 2
                + self.sender.display_name().len()
                + 8
                + 8
                + 4
                + text_bytes.len()
                + 64,
        );
        out.extend_from_slice(GROUP_MESSAGE_MAGIC);
        write_len_prefixed(&mut out, &self.group_id);
        write_peer(&mut out, &self.sender);
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(text_bytes);
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a group message produced by [`GroupMessage::encode`].
    /// Decoding does not verify the signature — call
    /// [`GroupMessage::verify`] for that.
    pub fn decode(bytes: &[u8]) -> Result<Self, TalkError> {
        if bytes.len() < 6 + 2 + 32 + 2 + 8 + 8 + 4 + 64 || &bytes[..6] != GROUP_MESSAGE_MAGIC {
            return Err(TalkError::InvalidGroup("missing KTALKG magic".to_string()));
        }
        let (group_id, off) = read_len_prefixed(bytes, 6, "group id")?;
        let (sender, off) = read_peer(bytes, off)?;
        if bytes.len() < off + 8 + 8 + 4 + 64 {
            return Err(TalkError::InvalidGroup(
                "group message truncates the timestamp, sequence, or text".to_string(),
            ));
        }
        let timestamp = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        let seq = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().expect("8-byte slice"));
        let text_len =
            u32::from_le_bytes(bytes[off + 16..off + 20].try_into().expect("4-byte slice"))
                as usize;
        let off = off + 20;
        if bytes.len() != off + text_len + 64 {
            return Err(TalkError::InvalidGroup(format!(
                "group message is {} bytes, expected {} for a {text_len}-byte text",
                bytes.len(),
                off + text_len + 64
            )));
        }
        let text = String::from_utf8(bytes[off..off + text_len].to_vec())
            .map_err(|_| TalkError::InvalidGroup("text is not valid UTF-8".to_string()))?;
        let off = off + text_len;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        validate_group_id(&group_id)?;
        Ok(Self {
            group_id,
            sender,
            text,
            timestamp,
            seq,
            signature,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_group() -> (Identity, Group) {
        let alice = Identity::generate("alice").unwrap();
        let group = Group::create(&alice, "korn-builders", "Korn Builders").unwrap();
        (alice, group)
    }

    #[test]
    fn group_create_encode_decode_and_verify() {
        let (alice, group) = make_group();
        assert!(group.verify());
        assert!(group.is_member(alice.peer_id()));
        assert_eq!(group.member_count(), 1);
        let decoded = Group::decode(&group.encode()).unwrap();
        assert_eq!(decoded.id, group.id);
        assert_eq!(decoded.name, group.name);
        assert_eq!(decoded.creator, group.creator);
        assert!(decoded.verify());
        assert!(Group::decode(b"garbage").is_err());
        // Empty id / name refused.
        assert!(Group::create(&alice, "", "name").is_err());
        assert!(Group::create(&alice, "id", "").is_err());
    }

    #[test]
    fn membership_add_and_remove_by_creator() {
        let (alice, mut group) = make_group();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let add_bob =
            MembershipChange::new(&alice, &group.id, bob.peer_id(), MembershipAction::Add).unwrap();
        assert!(add_bob.verify(&group.creator));
        group.apply_change(&add_bob).unwrap();
        assert!(group.is_member(bob.peer_id()));
        assert_eq!(group.member_count(), 2);
        // Adding the same member again is a no-op, not an error.
        group.apply_change(&add_bob).unwrap();
        assert_eq!(group.member_count(), 2);
        // A change for the wrong group is refused.
        let wrong_group = MembershipChange::new(
            &alice,
            "some-other-group",
            carol.peer_id(),
            MembershipAction::Add,
        )
        .unwrap();
        assert!(group.apply_change(&wrong_group).is_err());
        // A change signed by someone who is not the creator is refused.
        let impostor =
            MembershipChange::new(&bob, &group.id, carol.peer_id(), MembershipAction::Add).unwrap();
        assert!(!impostor.verify(&group.creator));
        assert!(group.apply_change(&impostor).is_err());
        assert!(!group.is_member(carol.peer_id()));
        // Remove works, and removing a non-member is a no-op.
        let remove_bob =
            MembershipChange::new(&alice, &group.id, bob.peer_id(), MembershipAction::Remove)
                .unwrap();
        group.apply_change(&remove_bob).unwrap();
        assert!(!group.is_member(bob.peer_id()));
        group.apply_change(&remove_bob).unwrap();
        // The creator cannot be removed.
        let remove_creator =
            MembershipChange::new(&alice, &group.id, alice.peer_id(), MembershipAction::Remove)
                .unwrap();
        assert!(group.apply_change(&remove_creator).is_err());
        assert!(group.is_member(alice.peer_id()));
    }

    #[test]
    fn membership_change_encode_decode_round_trip() {
        let (alice, group) = make_group();
        let bob = Identity::generate("bob").unwrap();
        let change =
            MembershipChange::new(&alice, &group.id, bob.peer_id(), MembershipAction::Add).unwrap();
        let decoded = MembershipChange::decode(&change.encode()).unwrap();
        assert_eq!(decoded, change);
        assert!(decoded.verify(&group.creator));
        assert!(MembershipChange::decode(b"garbage").is_err());
        assert!(MembershipChange::decode(&change.encode()[..10]).is_err());
    }

    #[test]
    fn group_message_encode_decode_verify_and_id() {
        let alice = Identity::generate("alice").unwrap();
        let msg = GroupMessage::new(&alice, "korn-builders", "hello group", 0).unwrap();
        assert!(msg.verify());
        let decoded = GroupMessage::decode(&msg.encode()).unwrap();
        assert_eq!(decoded, msg);
        assert!(decoded.verify());
        assert_eq!(decoded.id(), msg.id());
        assert!(GroupMessage::decode(b"garbage").is_err());
        // A tampered text fails verification.
        let mut tampered = msg.clone();
        tampered.text = "forged".to_string();
        assert!(!tampered.verify());
        assert_ne!(tampered.id(), msg.id());
    }
}
