//! Offline store-and-forward for K•Talk.
//!
//! When a peer is offline, messages for them wait in an outbox
//! instead of being lost. The caller drives everything — nothing
//! runs in the background:
//!
//! 1. Queue a message for an offline peer with [`Outbox::queue`].
//! 2. When the peer comes back, the caller signals
//!    [`Outbox::peer_online`], which returns the queued messages for
//!    that peer in order. The queue entries are NOT cleared yet.
//! 3. The caller delivers each message (for example with
//!    [`crate::TalkNode::send_message`]) and acknowledges each one
//!    with [`Outbox::acknowledge`]. Only acknowledged entries are
//!    removed, so a message that fails to deliver stays queued and
//!    can be retried.
//!
//! The queue is capped: at most [`MAX_QUEUED_PER_PEER`] messages per
//! peer. Queuing past the cap is a clean
//! [`crate::TalkError::OutboxFull`], never silent growth.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_identity::PeerId;

use crate::{Message, TalkError};

/// Maximum messages queued for one peer. Past this, queuing is
/// refused with [`TalkError::OutboxFull`]. A cap keeps one offline
/// peer from growing the outbox forever.
pub const MAX_QUEUED_PER_PEER: usize = 100;

/// One queued message.
#[derive(Clone, Debug)]
pub struct QueuedMessage {
    /// The message waiting for delivery.
    pub message: Message,
    /// Unix seconds when it was queued.
    pub enqueued_at: u64,
}

/// The outbox: queued messages per peer, plus which peers are
/// currently online.
#[derive(Debug, Default)]
pub struct Outbox {
    queue: HashMap<PeerId, VecDeque<QueuedMessage>>,
    online: HashSet<PeerId>,
}

impl Outbox {
    /// An empty outbox.
    pub fn new() -> Self {
        Self::default()
    }

    /// Unix time in seconds.
    fn now_unix_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Queue a message for `peer`. Refused with
    /// [`TalkError::OutboxFull`] when that peer already has
    /// [`MAX_QUEUED_PER_PEER`] messages queued.
    pub fn queue(&mut self, peer: &PeerId, message: Message) -> Result<(), TalkError> {
        let entry = self.queue.entry(peer.clone()).or_default();
        if entry.len() >= MAX_QUEUED_PER_PEER {
            return Err(TalkError::OutboxFull {
                peer: peer.clone(),
                max: MAX_QUEUED_PER_PEER,
            });
        }
        entry.push_back(QueuedMessage {
            message,
            enqueued_at: Self::now_unix_secs(),
        });
        Ok(())
    }

    /// Mark `peer` online and return their queued messages in order.
    /// The entries stay queued until the caller acknowledges each one
    /// with [`Outbox::acknowledge`] — returning them does not clear
    /// them.
    pub fn peer_online(&mut self, peer: &PeerId) -> Vec<Message> {
        self.online.insert(peer.clone());
        self.pending(peer)
    }

    /// Messages still queued for `peer`, in order. Does not change
    /// anything.
    pub fn pending(&self, peer: &PeerId) -> Vec<Message> {
        self.queue
            .get(peer)
            .map(|q| q.iter().map(|item| item.message.clone()).collect())
            .unwrap_or_default()
    }

    /// Number of messages queued for `peer`.
    pub fn pending_count(&self, peer: &PeerId) -> usize {
        self.queue.get(peer).map(|q| q.len()).unwrap_or(0)
    }

    /// Acknowledge that the message with `message_id` was delivered
    /// to `peer`: its queue entry is removed. Returns true if an
    /// entry was removed, false if there was none. A message id is
    /// the SHA-256 of the message's encoding (see
    /// [`Message::id`]).
    pub fn acknowledge(&mut self, peer: &PeerId, message_id: &[u8; 32]) -> bool {
        let Some(q) = self.queue.get_mut(peer) else {
            return false;
        };
        let before = q.len();
        q.retain(|item| item.message.id() != *message_id);
        let removed = q.len() < before;
        let empty = q.is_empty();
        // The borrow of `q` ends here (last use above), so removing
        // the peer's now-empty queue is allowed.
        if empty {
            self.queue.remove(peer);
        }
        removed
    }

    /// True if `peer` was marked online (and not since marked
    /// offline).
    pub fn is_online(&self, peer: &PeerId) -> bool {
        self.online.contains(peer)
    }

    /// Mark `peer` offline. Queued messages are kept — they will be
    /// offered again the next time the peer comes online.
    pub fn mark_offline(&mut self, peer: &PeerId) {
        self.online.remove(peer);
    }

    /// Peers with queued messages.
    pub fn queued_peers(&self) -> Vec<PeerId> {
        self.queue.keys().cloned().collect()
    }

    /// Total messages queued across all peers.
    pub fn total_queued(&self) -> usize {
        self.queue.values().map(|q| q.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_identity::Identity;

    fn make_message(sender: &Identity, recipient: &PeerId, text: &str, seq: u64) -> Message {
        Message::new(sender, recipient, text, seq)
    }

    #[test]
    fn queue_deliver_ack_in_order() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut outbox = Outbox::new();
        assert!(!outbox.is_online(bob.peer_id()));
        let m1 = make_message(&alice, bob.peer_id(), "one", 0);
        let m2 = make_message(&alice, bob.peer_id(), "two", 1);
        let m3 = make_message(&alice, bob.peer_id(), "three", 2);
        outbox.queue(bob.peer_id(), m1.clone()).unwrap();
        outbox.queue(bob.peer_id(), m2.clone()).unwrap();
        outbox.queue(bob.peer_id(), m3.clone()).unwrap();
        assert_eq!(outbox.pending_count(bob.peer_id()), 3);
        assert_eq!(outbox.total_queued(), 3);

        // Peer comes online: messages are offered in order, but the
        // queue is NOT cleared yet.
        let offered = outbox.peer_online(bob.peer_id());
        assert!(outbox.is_online(bob.peer_id()));
        assert_eq!(offered.len(), 3);
        assert_eq!(offered[0].text, "one");
        assert_eq!(offered[1].text, "two");
        assert_eq!(offered[2].text, "three");
        assert_eq!(outbox.pending_count(bob.peer_id()), 3);

        // Acknowledge the first two; the third stays queued.
        assert!(outbox.acknowledge(bob.peer_id(), &m1.id()));
        assert!(outbox.acknowledge(bob.peer_id(), &m2.id()));
        assert_eq!(outbox.pending_count(bob.peer_id()), 1);
        assert_eq!(outbox.pending(bob.peer_id())[0].text, "three");
        // Acknowledging an unknown id is a clean false, not an error.
        assert!(!outbox.acknowledge(bob.peer_id(), &[7u8; 32]));
        assert!(!outbox.acknowledge(bob.peer_id(), &m1.id()));
        // Acknowledge the last one: the peer's queue is gone.
        assert!(outbox.acknowledge(bob.peer_id(), &m3.id()));
        assert_eq!(outbox.pending_count(bob.peer_id()), 0);
        assert!(outbox.queued_peers().is_empty());
    }

    #[test]
    fn queue_cap_refuses_past_the_cap() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut outbox = Outbox::new();
        for i in 0..MAX_QUEUED_PER_PEER {
            let m = make_message(&alice, bob.peer_id(), &format!("msg {i}"), i as u64);
            outbox.queue(bob.peer_id(), m).unwrap();
        }
        assert_eq!(outbox.pending_count(bob.peer_id()), MAX_QUEUED_PER_PEER);
        let extra = make_message(&alice, bob.peer_id(), "one too many", 999);
        assert!(matches!(
            outbox.queue(bob.peer_id(), extra),
            Err(TalkError::OutboxFull { .. })
        ));
        // The cap is per peer: another peer still has room.
        let carol = Identity::generate("carol").unwrap();
        let m = make_message(&alice, carol.peer_id(), "for carol", 0);
        outbox.queue(carol.peer_id(), m).unwrap();
        // Acknowledging one frees a slot.
        let first = outbox.pending(bob.peer_id())[0].clone();
        assert!(outbox.acknowledge(bob.peer_id(), &first.id()));
        let m = make_message(&alice, bob.peer_id(), "fits now", 1000);
        outbox.queue(bob.peer_id(), m).unwrap();
    }

    #[test]
    fn offline_marking_keeps_the_queue() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut outbox = Outbox::new();
        let m = make_message(&alice, bob.peer_id(), "waiting", 0);
        outbox.queue(bob.peer_id(), m.clone()).unwrap();
        outbox.peer_online(bob.peer_id());
        assert!(outbox.is_online(bob.peer_id()));
        outbox.mark_offline(bob.peer_id());
        assert!(!outbox.is_online(bob.peer_id()));
        // The queue survives going offline.
        assert_eq!(outbox.pending_count(bob.peer_id()), 1);
        // And is offered again on the next online signal.
        let offered = outbox.peer_online(bob.peer_id());
        assert_eq!(offered.len(), 1);
        assert_eq!(offered[0].text, "waiting");
    }
}
