//! HEY•AI, the KERNEL-native personal AI assistant.
//!
//! Like Gemini is Google's AI, HEY•AI is KERNEL's AI: the assistant
//! itself, living on KERNEL. It takes a user message (text), routes
//! it to the right app or tool with the fixed keyword router, runs
//! the tool against caller-supplied local data, and answers in plain
//! words. Everything it says comes from local data — the content
//! store, the caller's tracks, bundles, and orders — or it says it
//! does not know how to do that. It never invents an answer, and it
//! never reaches the network: there is no model here, only
//! deterministic keyword routing over data the caller provides.
//!
//! The conversation log records every turn in order: the message,
//! the tool that was used, the answer, and the local data the answer
//! cites. The log is the assistant's memory of the conversation;
//! it is in-memory only.
//!
//! Voice notes, voice commands, and the render engine are things
//! the assistant uses, not what it is. They live in the same crate
//! because the assistant speaks through them.
//!
//! Accepted message forms (the fixed router — see
//! [`crate::parse_command`]):
//! * `search <words>` — search the content store.
//! * `play <name>` — play a track by name.
//! * `bundle <id>` — show a creator bundle.
//! * `orders` — list the caller's orders.
//! * `list` — list the caller's tracks.
//! * `pause`, `next`, `previous` — playback control.
//! Anything else is a clean refusal, never a made-up answer.

use kernel_storage::Cid;

use crate::{dispatch, parse_command, VoiceContext, VoiceOutcome};

/// Which tool a turn used, for the conversation log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolUsed {
    /// Searched the content store.
    Search,
    /// Played a track.
    PlayTrack,
    /// Paused, skipped, or went back.
    PlaybackControl,
    /// Listed the caller's tracks.
    ListTracks,
    /// Showed a creator bundle.
    ShowBundle,
    /// Listed the caller's orders.
    ListOrders,
    /// No tool: the message matched no intent, so the turn is a
    /// refusal.
    None,
}

/// A piece of local data an answer came from. Every answer that
/// reports local data cites it; a refusal or a control response
/// cites nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Citation {
    /// A content-store block, by CID.
    Block(Cid),
    /// A track, by name.
    Track(String),
    /// A creator bundle, by id.
    Bundle(String),
    /// An order, by its summary text.
    Order(String),
}

/// One recorded turn of the conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssistantTurn {
    /// Turn number, starting at 1, in order.
    pub id: u64,
    /// The user's message, as given.
    pub user_message: String,
    /// The tool this turn used.
    pub tool: ToolUsed,
    /// The answer, in plain words.
    pub answer: String,
    /// The local data the answer came from. Empty when the answer
    /// is a refusal or a control response (pause, next, previous).
    pub citations: Vec<Citation>,
}

/// The refusal answer for a message that matches no intent. Fixed
/// text: the assistant never invents an answer.
pub const REFUSAL: &str = "I don't know how to do that. I can search the store, play a track, show a bundle, list your orders, or list your tracks.";

fn answer_for(outcome: &VoiceOutcome) -> (ToolUsed, String, Vec<Citation>) {
    match outcome {
        VoiceOutcome::Playing { track_name } => (
            ToolUsed::PlayTrack,
            format!("Playing {track_name}."),
            vec![Citation::Track(track_name.clone())],
        ),
        VoiceOutcome::TrackNotFound { track_name } => (
            ToolUsed::PlayTrack,
            format!("I couldn't find a track called '{track_name}'."),
            vec![],
        ),
        VoiceOutcome::Paused => (ToolUsed::PlaybackControl, "Paused.".to_string(), vec![]),
        VoiceOutcome::NextTrack => (
            ToolUsed::PlaybackControl,
            "Skipped to the next track.".to_string(),
            vec![],
        ),
        VoiceOutcome::PreviousTrack => (
            ToolUsed::PlaybackControl,
            "Back to the previous track.".to_string(),
            vec![],
        ),
        VoiceOutcome::TrackList(tracks) => {
            if tracks.is_empty() {
                (ToolUsed::ListTracks, "No tracks.".to_string(), vec![])
            } else {
                (
                    ToolUsed::ListTracks,
                    format!("Tracks: {}.", tracks.join(", ")),
                    tracks.iter().map(|t| Citation::Track(t.clone())).collect(),
                )
            }
        }
        VoiceOutcome::SearchResults(hits) => {
            if hits.is_empty() {
                (ToolUsed::Search, "No results.".to_string(), vec![])
            } else {
                let mut text = format!(
                    "Found {} result{}:",
                    hits.len(),
                    if hits.len() == 1 { "" } else { "s" }
                );
                for (cid, excerpt) in hits {
                    text.push_str(&format!("\n- {}: {excerpt}", cid.to_hex()));
                }
                (
                    ToolUsed::Search,
                    text,
                    hits.iter().map(|(cid, _)| Citation::Block(*cid)).collect(),
                )
            }
        }
        VoiceOutcome::BundleFound { bundle_id } => (
            ToolUsed::ShowBundle,
            format!("Bundle '{bundle_id}' found."),
            vec![Citation::Bundle(bundle_id.clone())],
        ),
        VoiceOutcome::BundleNotFound { bundle_id } => (
            ToolUsed::ShowBundle,
            format!("I couldn't find a bundle called '{bundle_id}'."),
            vec![],
        ),
        VoiceOutcome::Orders(orders) => {
            if orders.is_empty() {
                (
                    ToolUsed::ListOrders,
                    "You have no orders.".to_string(),
                    vec![],
                )
            } else {
                (
                    ToolUsed::ListOrders,
                    format!("Your orders:\n- {}", orders.join("\n- ")),
                    orders.iter().map(|o| Citation::Order(o.clone())).collect(),
                )
            }
        }
        VoiceOutcome::Unknown => (ToolUsed::None, REFUSAL.to_string(), vec![]),
    }
}

/// The assistant: a conversation log plus the ability to answer one
/// message at a time against caller-supplied local data.
pub struct Assistant {
    turns: Vec<AssistantTurn>,
    next_id: u64,
}

impl Default for Assistant {
    /// A new assistant with an empty conversation log.
    fn default() -> Self {
        Self::new()
    }
}

impl Assistant {
    /// A new assistant with an empty conversation log.
    pub fn new() -> Self {
        Self {
            turns: Vec::new(),
            next_id: 1,
        }
    }

    /// Number of turns in the conversation log.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// True if no turns have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Every recorded turn, in order.
    pub fn turns(&self) -> &[AssistantTurn] {
        &self.turns
    }

    /// Answer a user message against caller-supplied local data.
    /// The message is routed with the fixed keyword router, the
    /// tool runs against `ctx`, and the turn — message, tool,
    /// answer, citations — is recorded in the conversation log and
    /// returned. A message that matches no intent is a clean
    /// refusal, never a made-up answer.
    pub fn answer(&mut self, message: &str, ctx: &VoiceContext) -> AssistantTurn {
        let action = parse_command(message);
        let outcome = dispatch(&action, ctx);
        let (tool, answer, citations) = answer_for(&outcome);
        let turn = AssistantTurn {
            id: self.next_id,
            user_message: message.to_string(),
            tool,
            answer,
            citations,
        };
        self.next_id += 1;
        self.turns.push(turn.clone());
        turn
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_storage::ContentStore;

    fn test_context<'a>(
        store: Option<&'a ContentStore>,
        tracks: &'a [String],
        bundles: &'a [String],
        orders: &'a [String],
    ) -> VoiceContext<'a> {
        VoiceContext {
            store,
            tracks,
            bundles,
            orders,
        }
    }

    #[test]
    fn routes_each_intent_to_the_right_tool() {
        let mut store = ContentStore::new();
        let hit = store.put(b"the kernel storage notes are here");
        let tracks = vec!["Midnight".to_string(), "Dawn".to_string()];
        let bundles = vec!["my-first-bundle".to_string()];
        let orders = vec!["order-1: lamp, paid".to_string()];
        let ctx = test_context(Some(&store), &tracks, &bundles, &orders);
        let mut assistant = Assistant::new();

        let turn = assistant.answer("search kernel storage", &ctx);
        assert_eq!(turn.tool, ToolUsed::Search);
        assert!(turn.answer.starts_with("Found 1 result:"));
        assert!(turn.answer.contains(&hit.to_hex()));
        assert_eq!(turn.citations, vec![Citation::Block(hit)]);

        let turn = assistant.answer("play midnight", &ctx);
        assert_eq!(turn.tool, ToolUsed::PlayTrack);
        assert_eq!(turn.answer, "Playing Midnight.");
        assert_eq!(
            turn.citations,
            vec![Citation::Track("Midnight".to_string())]
        );

        let turn = assistant.answer("bundle my-first-bundle", &ctx);
        assert_eq!(turn.tool, ToolUsed::ShowBundle);
        assert_eq!(turn.answer, "Bundle 'my-first-bundle' found.");
        assert_eq!(
            turn.citations,
            vec![Citation::Bundle("my-first-bundle".to_string())]
        );

        let turn = assistant.answer("orders", &ctx);
        assert_eq!(turn.tool, ToolUsed::ListOrders);
        assert!(turn.answer.contains("order-1: lamp, paid"));
        assert_eq!(
            turn.citations,
            vec![Citation::Order("order-1: lamp, paid".to_string())]
        );

        let turn = assistant.answer("list", &ctx);
        assert_eq!(turn.tool, ToolUsed::ListTracks);
        assert_eq!(turn.answer, "Tracks: Midnight, Dawn.");
        assert_eq!(
            turn.citations,
            vec![
                Citation::Track("Midnight".to_string()),
                Citation::Track("Dawn".to_string()),
            ]
        );

        let turn = assistant.answer("pause", &ctx);
        assert_eq!(turn.tool, ToolUsed::PlaybackControl);
        assert_eq!(turn.answer, "Paused.");
        assert!(turn.citations.is_empty());

        // A known intent with nothing to find still uses its tool.
        let turn = assistant.answer("play Nope", &ctx);
        assert_eq!(turn.tool, ToolUsed::PlayTrack);
        assert_eq!(turn.answer, "I couldn't find a track called 'Nope'.");
        assert!(turn.citations.is_empty());
        let turn = assistant.answer("search nothing here", &ctx);
        assert_eq!(turn.tool, ToolUsed::Search);
        assert_eq!(turn.answer, "No results.");
        assert!(turn.citations.is_empty());
    }

    #[test]
    fn unknown_input_is_a_clean_refusal() {
        let ctx = VoiceContext::default();
        let mut assistant = Assistant::new();
        for message in [
            "tell me a joke",
            "what's the weather",
            "",
            "   ",
            "play",
            "search",
            "bundle",
            "open the pod bay doors",
        ] {
            let turn = assistant.answer(message, &ctx);
            assert_eq!(turn.tool, ToolUsed::None, "message: {message:?}");
            assert_eq!(turn.answer, REFUSAL, "message: {message:?}");
            assert!(turn.citations.is_empty(), "message: {message:?}");
            // The refusal names what the assistant can do and invents
            // nothing: it contains no local data.
            assert!(turn.answer.contains("search the store"));
        }
        assert_eq!(assistant.len(), 8);
    }

    #[test]
    fn conversation_log_records_turns_in_order() {
        let mut store = ContentStore::new();
        store.put(b"kernel notes");
        let tracks = vec!["Midnight".to_string()];
        let ctx = test_context(Some(&store), &tracks, &[], &[]);
        let mut assistant = Assistant::new();
        assert!(assistant.is_empty());

        let first = assistant.answer("search kernel", &ctx);
        let second = assistant.answer("play Midnight", &ctx);
        let third = assistant.answer("tell me a joke", &ctx);
        assert_eq!((first.id, second.id, third.id), (1, 2, 3));
        assert_eq!(first.user_message, "search kernel");
        assert_eq!(second.user_message, "play Midnight");
        assert_eq!(third.user_message, "tell me a joke");
        assert_eq!(
            assistant.turns().iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            assistant.turns().iter().map(|t| t.tool).collect::<Vec<_>>(),
            vec![ToolUsed::Search, ToolUsed::PlayTrack, ToolUsed::None]
        );
        assert_eq!(assistant.len(), 3);
        // The returned turn and the logged turn are the same record.
        assert_eq!(assistant.turns()[0], first);
        assert_eq!(assistant.turns()[2], third);
    }

    #[test]
    fn answers_cite_their_local_data() {
        let mut store = ContentStore::new();
        let hit = store.put(b"kernel storage notes");
        let tracks = vec!["Midnight".to_string()];
        let bundles = vec!["my-first-bundle".to_string()];
        let orders = vec!["order-1: lamp, paid".to_string()];
        let ctx = test_context(Some(&store), &tracks, &bundles, &orders);
        let mut assistant = Assistant::new();

        let turn = assistant.answer("search kernel", &ctx);
        assert_eq!(turn.citations, vec![Citation::Block(hit)]);
        assert!(turn.answer.contains(&hit.to_hex()));

        let turn = assistant.answer("bundle my-first-bundle", &ctx);
        assert_eq!(
            turn.citations,
            vec![Citation::Bundle("my-first-bundle".to_string())]
        );

        let turn = assistant.answer("orders", &ctx);
        assert_eq!(
            turn.citations,
            vec![Citation::Order("order-1: lamp, paid".to_string())]
        );

        // A refusal cites nothing, and a not-found answer cites nothing.
        let turn = assistant.answer("what is the meaning of life", &ctx);
        assert!(turn.citations.is_empty());
        let turn = assistant.answer("bundle nope", &ctx);
        assert!(turn.citations.is_empty());
    }

    #[test]
    fn assistant_is_deterministic() {
        let mut store = ContentStore::new();
        store.put(b"kernel storage notes");
        let tracks = vec!["Midnight".to_string()];
        let ctx = test_context(Some(&store), &tracks, &[], &[]);
        let mut first = Assistant::new();
        let mut second = Assistant::new();
        for message in ["search kernel", "play Midnight", "tell me a joke", "orders"] {
            let a = first.answer(message, &ctx);
            let b = second.answer(message, &ctx);
            assert_eq!(a.answer, b.answer, "message: {message:?}");
            assert_eq!(a.tool, b.tool, "message: {message:?}");
            assert_eq!(a.citations, b.citations, "message: {message:?}");
        }
    }

    #[test]
    fn empty_local_data_gives_empty_answers_not_errors() {
        let ctx = VoiceContext::default();
        let mut assistant = Assistant::new();
        let turn = assistant.answer("search kernel", &ctx);
        assert_eq!(turn.tool, ToolUsed::Search);
        assert_eq!(turn.answer, "No results.");
        let turn = assistant.answer("list", &ctx);
        assert_eq!(turn.answer, "No tracks.");
        let turn = assistant.answer("orders", &ctx);
        assert_eq!(turn.answer, "You have no orders.");
        let turn = assistant.answer("bundle anything", &ctx);
        assert_eq!(turn.tool, ToolUsed::ShowBundle);
        assert_eq!(turn.answer, "I couldn't find a bundle called 'anything'.");
    }
}
