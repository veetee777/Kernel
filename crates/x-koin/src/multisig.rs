//! Multi-signature transactions for X•Koin.
//!
//! A multi-signature transfer is an ordinary signed [`CoinTransaction`]
//! (the sender signs it as usual) plus a gate: it only reaches the
//! ledger once M distinct approvers from a named set of N have signed
//! their approval. Approvals are signed records: an approver signs the
//! proposal id, so an approval cannot be replayed onto a different
//! proposal, and a forged approval does not verify.
//!
//! The gate is local approval counting. It proves M named keys
//! approved this exact proposal; it does not prove anything about
//! hardware keys, and it does not change the transfer rules — once
//! the threshold is reached the transaction submits exactly like a
//! normal transfer, with the same replay protection and balance
//! checks.

use kernel_crypto::{sha256, to_hex, Ed25519Signature};
use kernel_identity::{Identity, PeerId};

use crate::{CoinTransaction, KoinError};

/// Domain separator for multisig approvals: a signature covers
/// `b"XKOIN-MS-APPR1" || proposal id bytes || approver public key`.
const APPROVAL_SIGN_DOMAIN: &[u8] = b"XKOIN-MS-APPR1";

/// A transfer proposal gated by multi-signature approval: the
/// sender-signed transaction, the threshold M, and the N approvers
/// whose signatures count toward it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultisigTransaction {
    /// The sender-signed transfer this proposal gates.
    pub tx: CoinTransaction,
    /// Approvals needed before the transfer may submit (1..=N).
    pub threshold: u8,
    /// The approvers whose signatures count. Order is part of the
    /// proposal id; duplicates are refused at construction.
    pub approvers: Vec<PeerId>,
}

impl MultisigTransaction {
    /// Build a proposal. Refuses a zero threshold, a threshold above
    /// the approver count, an empty approver set, and duplicate
    /// approvers. The underlying transaction's signature and amount
    /// are checked when the proposal is submitted to the ledger, not
    /// here.
    pub fn new(
        tx: CoinTransaction,
        threshold: u8,
        approvers: Vec<PeerId>,
    ) -> Result<Self, KoinError> {
        if approvers.is_empty() {
            return Err(KoinError::InvalidMultisig(
                "multisig proposal needs at least one approver".to_string(),
            ));
        }
        if threshold == 0 {
            return Err(KoinError::InvalidMultisig(
                "multisig threshold must be at least 1".to_string(),
            ));
        }
        if threshold as usize > approvers.len() {
            return Err(KoinError::InvalidMultisig(format!(
                "threshold {} exceeds approver count {}",
                threshold,
                approvers.len()
            )));
        }
        let mut seen = std::collections::HashSet::new();
        for approver in &approvers {
            if !seen.insert(*approver.public_key().as_bytes()) {
                return Err(KoinError::InvalidMultisig(
                    "duplicate approver in multisig proposal".to_string(),
                ));
            }
        }
        Ok(Self {
            tx,
            threshold,
            approvers,
        })
    }

    /// The proposal's id: the SHA-256 of the transaction encoding
    /// plus the threshold plus the approver keys, rendered as hex.
    /// Two proposals for the same transaction, threshold, and
    /// approver set have the same id.
    pub fn id(&self) -> String {
        to_hex(&self.id_bytes())
    }

    /// The raw 32-byte proposal id.
    pub fn id_bytes(&self) -> [u8; 32] {
        let mut msg = self.tx.encode();
        msg.push(self.threshold);
        for approver in &self.approvers {
            msg.extend_from_slice(approver.public_key().as_bytes());
        }
        sha256(&msg)
    }

    /// True if this peer is one of the named approvers.
    pub fn is_approver(&self, peer: &PeerId) -> bool {
        self.approvers
            .iter()
            .any(|a| a.public_key() == peer.public_key())
    }
}

/// A signed approval of one multisig proposal by one approver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultisigApproval {
    /// The proposal id this approval is for.
    pub proposal_id: String,
    /// Who approved.
    pub approver: PeerId,
    /// The approver's signature over domain + proposal id + key.
    pub signature: Ed25519Signature,
}

impl MultisigApproval {
    /// Sign an approval for a proposal id as `approver`.
    pub fn new(approver: &Identity, proposal_id: &str) -> Self {
        let approver_id = approver.peer_id().clone();
        let msg = approval_sign_message(&approver_id, proposal_id);
        let signature = approver.sign(&msg);
        Self {
            proposal_id: proposal_id.to_string(),
            approver: approver_id,
            signature,
        }
    }

    /// True if the signature verifies against the approver for this
    /// exact proposal id. Any tampering fails this.
    pub fn verify(&self) -> bool {
        let msg = approval_sign_message(&self.approver, &self.proposal_id);
        self.approver.verify(&msg, &self.signature).is_ok()
    }
}

/// The message an approver signs: domain + proposal id + approver key.
fn approval_sign_message(approver: &PeerId, proposal_id: &str) -> Vec<u8> {
    let mut msg = Vec::with_capacity(APPROVAL_SIGN_DOMAIN.len() + proposal_id.len() + 32);
    msg.extend_from_slice(APPROVAL_SIGN_DOMAIN);
    msg.extend_from_slice(proposal_id.as_bytes());
    msg.extend_from_slice(approver.public_key().as_bytes());
    msg
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identities() -> (Identity, Identity, Identity, Identity) {
        (
            Identity::generate("alice").unwrap(),
            Identity::generate("bob").unwrap(),
            Identity::generate("carol").unwrap(),
            Identity::generate("dave").unwrap(),
        )
    }

    fn proposal() -> (MultisigTransaction, Identity, Identity, Identity, Identity) {
        let (alice, bob, carol, dave) = identities();
        let tx = CoinTransaction::new(&alice, bob.peer_id(), "shared pot", 100).unwrap();
        let proposal = MultisigTransaction::new(
            tx,
            2,
            vec![
                bob.peer_id().clone(),
                carol.peer_id().clone(),
                dave.peer_id().clone(),
            ],
        )
        .unwrap();
        (proposal, alice, bob, carol, dave)
    }

    #[test]
    fn proposal_id_is_stable_and_threshold_rules() {
        let (proposal, _, bob, carol, dave) = proposal();
        let again = MultisigTransaction::new(
            proposal.tx.clone(),
            2,
            vec![
                bob.peer_id().clone(),
                carol.peer_id().clone(),
                dave.peer_id().clone(),
            ],
        )
        .unwrap();
        assert_eq!(proposal.id(), again.id());
        // A different threshold is a different proposal.
        let other = MultisigTransaction::new(
            proposal.tx.clone(),
            3,
            vec![
                bob.peer_id().clone(),
                carol.peer_id().clone(),
                dave.peer_id().clone(),
            ],
        )
        .unwrap();
        assert_ne!(proposal.id(), other.id());
        // Zero threshold, threshold above N, empty approvers, and
        // duplicate approvers are all refused.
        assert!(matches!(
            MultisigTransaction::new(proposal.tx.clone(), 0, vec![bob.peer_id().clone()]),
            Err(KoinError::InvalidMultisig(_))
        ));
        assert!(matches!(
            MultisigTransaction::new(
                proposal.tx.clone(),
                4,
                vec![bob.peer_id().clone(), carol.peer_id().clone()]
            ),
            Err(KoinError::InvalidMultisig(_))
        ));
        assert!(matches!(
            MultisigTransaction::new(proposal.tx.clone(), 1, vec![]),
            Err(KoinError::InvalidMultisig(_))
        ));
        assert!(matches!(
            MultisigTransaction::new(
                proposal.tx.clone(),
                1,
                vec![bob.peer_id().clone(), bob.peer_id().clone()]
            ),
            Err(KoinError::InvalidMultisig(_))
        ));
    }

    #[test]
    fn approval_verifies_and_tampering_fails() {
        let (proposal, _, bob, _, _) = proposal();
        let approval = MultisigApproval::new(&bob, &proposal.id());
        assert!(approval.verify());
        assert_eq!(approval.approver.public_key(), bob.peer_id().public_key());
        // Tamper with the proposal id after signing.
        let mut forged = approval.clone();
        forged.proposal_id = "00".repeat(32);
        assert!(!forged.verify());
        // An approval for another proposal does not verify here.
        let other = MultisigApproval::new(&bob, "deadbeef");
        assert!(other.verify());
        assert_ne!(other.proposal_id, proposal.id());
    }

    #[test]
    fn is_approver_checks_by_key_not_name() {
        let (proposal, alice, bob, _, _) = proposal();
        assert!(proposal.is_approver(bob.peer_id()));
        assert!(!proposal.is_approver(alice.peer_id()));
        // Same display name, different key, is not an approver.
        let bob_twin = Identity::generate("bob").unwrap();
        assert_ne!(bob_twin.peer_id().public_key(), bob.peer_id().public_key());
        assert!(!proposal.is_approver(bob_twin.peer_id()));
    }
}
