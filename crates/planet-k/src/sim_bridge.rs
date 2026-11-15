//! Bridge between the Korn world ([`kernel_sim::World`]) and Planet-K.
//!
//! Sim events become ordinary signed Planet-K posts, so everything
//! Planet-K already does applies to them unchanged:
//!
//! * A sim event is rendered to one line of human-readable text and
//!   published as a [`Post`] signed by the acting sim agent's
//!   identity (re-derived deterministically from the world seed —
//!   see [`World::agent_identity`](kernel_sim::World::agent_identity)).
//!   World events (regrowth), which no agent acts, are published
//!   under the world's own identity
//!   ([`World::world_identity`](kernel_sim::World::world_identity)).
//! * The post's timestamp **is the sim tick** — never the wall clock.
//!   All time in the Korn world comes from the tick counter, and the
//!   bridge keeps it that way. A feed orders sim posts among normal
//!   posts by that timestamp, newest first.
//! * Because sim posts are plain posts, the feed, blocking, likes,
//!   comments, and reposts all work on them exactly as on any other
//!   post: a viewer must follow the sim agent (or the world) to see
//!   its posts; blocking the sim agent hides its posts and any
//!   reposts of them; reposting a sim post resolves to the original
//!   sim post.
//!
//! Publishing is idempotent and deterministic: Ed25519 signing is
//! deterministic, so publishing the same event twice yields the same
//! post record CID, and two identical worlds publish identical posts.
//! The bridge never touches the sim — it takes `&World`, so the
//! event log, the save format, and replay verification are
//! unaffected.

use kernel_identity::Identity;
use kernel_sim::{Event, EventKind, ResourceKind, World};
use kernel_storage::Cid;

use crate::{SocialError, SocialStore};

/// Render one sim event as the text of a Planet-K post. Returns
/// `None` when the event cannot be rendered — an agent event naming
/// an agent index the world does not have, or a gather/regrow event
/// naming an unknown resource kind. Rendered text always passes
/// Planet-K's text rules (non-empty, no control characters).
pub fn render_event(event: &Event, world: &World) -> Option<String> {
    let tick = event.tick;
    match event.kind {
        EventKind::Spawn => {
            let agent = world.agents().get(event.agent? as usize)?;
            Some(format!(
                "{} spawned at ({}, {}) with {} energy and {} health. [Korn tick {}]",
                agent.name, event.x, event.y, event.a, event.b, tick
            ))
        }
        EventKind::Move => {
            let agent = world.agents().get(event.agent? as usize)?;
            Some(format!(
                "{} moved to ({}, {}); energy {}. [Korn tick {}]",
                agent.name, event.x, event.y, event.a, tick
            ))
        }
        EventKind::Gather => {
            let agent = world.agents().get(event.agent? as usize)?;
            let kind = resource_name(event.a)?;
            Some(format!(
                "{} gathered {} at ({}, {}); {} left on the cell. [Korn tick {}]",
                agent.name, kind, event.x, event.y, event.b, tick
            ))
        }
        EventKind::Eat => {
            let agent = world.agents().get(event.agent? as usize)?;
            Some(format!(
                "{} ate food; food {}, energy {}. [Korn tick {}]",
                agent.name, event.a, event.b, tick
            ))
        }
        EventKind::Rest => {
            let agent = world.agents().get(event.agent? as usize)?;
            Some(format!(
                "{} rested; food {}, energy {}. [Korn tick {}]",
                agent.name, event.a, event.b, tick
            ))
        }
        EventKind::Starve => {
            let agent = world.agents().get(event.agent? as usize)?;
            Some(format!(
                "{} starved at ({}, {}); health {}. [Korn tick {}]",
                agent.name, event.x, event.y, event.a, tick
            ))
        }
        EventKind::Died => {
            let agent = world.agents().get(event.agent? as usize)?;
            Some(format!(
                "{} died at ({}, {}). [Korn tick {}]",
                agent.name, event.x, event.y, tick
            ))
        }
        EventKind::Regrow => {
            let kind = resource_name(event.a)?;
            Some(format!(
                "A {} regrew at ({}, {}); {} now on the cell. [Korn tick {}]",
                kind, event.x, event.y, event.b, tick
            ))
        }
    }
}

/// The resource name for an event payload code (1 food, 2 wood,
/// 3 stone), or `None` for an unknown code.
fn resource_name(code: u32) -> Option<&'static str> {
    let byte = u8::try_from(code).ok()?;
    ResourceKind::decode(byte).map(|kind| kind.name())
}

/// The signing identity that authors a post for `event`: the acting
/// agent's identity for agent events, the world's own identity for
/// regrowth. Returns `None` when the event names an agent the world
/// does not have.
fn author_identity(event: &Event, world: &World) -> Option<Identity> {
    match event.kind {
        EventKind::Regrow => world.world_identity().ok(),
        _ => {
            let index = event.agent? as usize;
            world.agent_identity(index)
        }
    }
}

/// Publish one sim event as a signed Planet-K post in `store` and
/// return the post record CID. The post is authored by the acting
/// sim agent (or the world, for regrowth) and timestamped with the
/// event's tick — never the wall clock. Publishing the same event
/// twice returns the same CID: the post bytes are identical, so the
/// content-addressed store dedupes them.
///
/// The bridge does not verify the event against the world log; it
/// publishes what it is given. A caller that wants a verified log
/// checks it with [`World::replay`](kernel_sim::World::replay) first.
pub fn publish_event(
    store: &mut SocialStore,
    world: &World,
    event: &Event,
) -> Result<Cid, SocialError> {
    let text = render_event(event, world).ok_or_else(|| {
        SocialError::InvalidRecord("sim event cannot be rendered as a post".to_string())
    })?;
    let identity = author_identity(event, world).ok_or_else(|| {
        SocialError::InvalidRecord("sim event names an unknown agent".to_string())
    })?;
    store.create_post(&text, event.tick, None, &identity)
}

/// Publish every event in the world's log, in log order, and return
/// the post record CIDs in the same order. Idempotent: publishing
/// the same log again returns the same CIDs and stores nothing new.
pub fn publish_log(store: &mut SocialStore, world: &World) -> Result<Vec<Cid>, SocialError> {
    let mut cids = Vec::with_capacity(world.events().len());
    for event in world.events() {
        cids.push(publish_event(store, world, event)?);
    }
    Ok(cids)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeedItem, Post};
    use kernel_identity::Identity;

    const NAMES: [&str; 3] = ["Korn", "Mira", "Juno"];

    fn world_ticked(seed: u64, ticks: u64) -> World {
        let mut w = World::new(seed, &NAMES).unwrap();
        w.tick_n(ticks);
        w
    }

    #[test]
    fn event_publishes_as_signed_post() {
        let world = world_ticked(1234, 200);
        let mut store = SocialStore::new();
        // A gather event: authored by the gathering agent.
        let event = world
            .events()
            .iter()
            .find(|e| e.kind == EventKind::Gather)
            .expect("world gathered something in 200 ticks")
            .clone();
        let cid = publish_event(&mut store, &world, &event).unwrap();
        let post = store.get_post(&cid).unwrap();
        assert!(post.verify(), "sim post signature must verify");
        let agent = &world.agents()[event.agent.unwrap() as usize];
        assert_eq!(post.author.public_key(), agent.id.public_key());
        assert_eq!(post.author.display_name(), agent.name);
        assert_eq!(post.timestamp, event.tick);
        assert!(post.text.contains(&agent.name));
        assert!(post.text.contains(&format!("[Korn tick {}]", event.tick)));
        // The post round-trips through the unchanged KPOST1 format.
        let decoded = Post::decode(&post.encode()).unwrap();
        assert!(decoded.verify());
        assert_eq!(decoded.text, post.text);
        assert_eq!(decoded.timestamp, post.timestamp);
        // A spawn event publishes too, at tick 0.
        let spawn = world
            .events()
            .iter()
            .find(|e| e.kind == EventKind::Spawn)
            .unwrap()
            .clone();
        let spawn_cid = publish_event(&mut store, &world, &spawn).unwrap();
        assert_eq!(store.get_post(&spawn_cid).unwrap().timestamp, 0);
    }

    #[test]
    fn regrow_publishes_under_world_identity() {
        let world = world_ticked(42, 1);
        let mut store = SocialStore::new();
        // A hand-built regrow event: publishing does not need the
        // event to come from this world's log, only the world identity.
        let event = Event {
            tick: 5,
            kind: EventKind::Regrow,
            agent: None,
            x: 3,
            y: 4,
            a: 1,
            b: 2,
        };
        let cid = publish_event(&mut store, &world, &event).unwrap();
        let post = store.get_post(&cid).unwrap();
        assert!(post.verify());
        assert_eq!(
            post.author.public_key(),
            world.world_identity().unwrap().peer_id().public_key()
        );
        assert_eq!(post.author.display_name(), "korn-world");
        assert!(post.text.contains("food regrew"));
        assert_eq!(post.timestamp, 5);
    }

    #[test]
    fn feed_includes_sim_posts_in_order_alongside_normal_posts() {
        let world = world_ticked(1234, 30);
        let mut store = SocialStore::new();
        let viewer = Identity::generate("viewer").unwrap();
        let bob = Identity::generate("bob").unwrap();
        // Publish the whole log, then add one normal post.
        let cids = publish_log(&mut store, &world).unwrap();
        assert_eq!(cids.len(), world.events().len());
        let bob_post = store.create_post("a normal post", 100, None, &bob).unwrap();
        let _ = bob_post;
        // The viewer follows the first agent and Bob.
        let agent0 = world.agents()[0].id.clone();
        store.follow(&viewer, &agent0).unwrap();
        store.follow(&viewer, bob.peer_id()).unwrap();
        let feed = store.feed(viewer.peer_id());
        // Newest first: Bob's post (timestamp 100) first, then the
        // agent's sim posts by tick descending.
        assert!(!feed.is_empty());
        assert_eq!(feed[0].text(), "a normal post");
        let sim_items: Vec<&FeedItem> = feed[1..].iter().filter(|item| !item.is_repost()).collect();
        assert!(!sim_items.is_empty(), "feed must include sim posts");
        assert!(sim_items
            .iter()
            .all(|item| item.author().public_key() == agent0.public_key()));
        assert!(sim_items
            .iter()
            .all(|item| item.text().contains("[Korn tick")));
        let ticks: Vec<u64> = sim_items.iter().map(|item| item.timestamp()).collect();
        let mut sorted = ticks.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(ticks, sorted, "sim posts must be newest first");
        // A viewer who follows nobody sees nothing.
        let stranger = Identity::generate("stranger").unwrap();
        assert!(store.feed(stranger.peer_id()).is_empty());
    }

    #[test]
    fn blocked_authors_sim_posts_hidden() {
        let world = world_ticked(1234, 30);
        let mut store = SocialStore::new();
        let viewer = Identity::generate("viewer").unwrap();
        let carol = Identity::generate("carol").unwrap();
        publish_log(&mut store, &world).unwrap();
        let agent0 = world.agents()[0].id.clone();
        let agent1 = world.agents()[1].id.clone();
        store.follow(&viewer, &agent0).unwrap();
        store.follow(&viewer, &agent1).unwrap();
        // Carol reposts one of agent 0's sim posts.
        let sim_cid = store
            .feed(viewer.peer_id())
            .iter()
            .find(|item| item.author().public_key() == agent0.public_key())
            .map(|item| match item {
                FeedItem::Post(post) => Cid::from_bytes(kernel_crypto::sha256(&post.encode())),
                FeedItem::Repost { .. } => unreachable!(),
            })
            .expect("agent 0 has a sim post in the feed");
        store.follow(&viewer, carol.peer_id()).unwrap();
        store.repost(&carol, &sim_cid, None, 200).unwrap();
        let before = store.feed(viewer.peer_id());
        assert!(
            before.iter().any(|item| item.is_repost()),
            "repost visible before blocking"
        );
        // Blocking agent 0 hides its sim posts and the repost of one.
        store.block(&viewer, &agent0).unwrap();
        let after = store.feed(viewer.peer_id());
        assert!(!after
            .iter()
            .any(|item| item.author().public_key() == agent0.public_key()));
        assert!(!after.iter().any(|item| item.is_repost()));
        // Agent 1's sim posts are untouched.
        assert!(after
            .iter()
            .any(|item| item.author().public_key() == agent1.public_key()));
        // Unblocking restores everything.
        store.unblock(&viewer, &agent0).unwrap();
        assert_eq!(store.feed(viewer.peer_id()).len(), before.len());
    }

    #[test]
    fn repost_of_sim_post_resolves_to_original() {
        let world = world_ticked(1234, 30);
        let mut store = SocialStore::new();
        let alice = Identity::generate("alice").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let dave = Identity::generate("dave").unwrap();
        // Publish one sim event and have Carol repost it.
        let event = world.events()[0].clone();
        let sim_cid = publish_event(&mut store, &world, &event).unwrap();
        let sim_post = store.get_post(&sim_cid).unwrap();
        let carol_repost = store
            .repost(&carol, &sim_cid, Some("look at this"), 50)
            .unwrap();
        assert_eq!(store.resolve_original(&carol_repost), Some(sim_cid));
        // Alice follows Carol, not the sim agent: she sees the repost
        // marked as a repost, with the sim post's words and author.
        store.follow(&alice, carol.peer_id()).unwrap();
        let feed = store.feed(alice.peer_id());
        assert_eq!(feed.len(), 1);
        assert!(feed[0].is_repost());
        assert_eq!(feed[0].text(), sim_post.text);
        assert_eq!(*feed[0].author(), *carol.peer_id());
        assert_eq!(
            feed[0].original_post().author.public_key(),
            sim_post.author.public_key()
        );
        assert_eq!(
            feed[0].repost().unwrap().comment.as_deref(),
            Some("look at this")
        );
        // Dave reposts Carol's repost: the chain still resolves to
        // the original sim post.
        let dave_repost = store.repost(&dave, &carol_repost, None, 60).unwrap();
        assert_eq!(store.resolve_original(&dave_repost), Some(sim_cid));
    }

    #[test]
    fn bridging_does_not_break_sim_replay() {
        let mut world = World::new(7, &NAMES).unwrap();
        world.tick_n(25);
        let events = world.events().to_vec();
        let before = world.save();
        let mut store = SocialStore::new();
        let cids = publish_log(&mut store, &world).unwrap();
        assert_eq!(cids.len(), events.len());
        // The bridge takes &World: the world is byte-identical after.
        assert_eq!(world.save(), before);
        let replayed = World::replay(7, &NAMES, &events, 25).unwrap();
        assert_eq!(replayed, world);
        assert_eq!(replayed.save(), world.save());
        // And the published posts still verify after the replay.
        for cid in cids {
            assert!(store.get_post(&cid).unwrap().verify());
        }
    }

    #[test]
    fn publishing_is_idempotent_and_deterministic() {
        let world_a = world_ticked(1234, 30);
        let world_b = world_ticked(1234, 30);
        assert_eq!(world_a, world_b);
        let mut store_a = SocialStore::new();
        let mut store_b = SocialStore::new();
        let cids_a = publish_log(&mut store_a, &world_a).unwrap();
        let cids_b = publish_log(&mut store_b, &world_b).unwrap();
        assert_eq!(cids_a, cids_b, "identical worlds publish identical posts");
        // Publishing the same log again changes nothing.
        let again = publish_log(&mut store_a, &world_a).unwrap();
        assert_eq!(again, cids_a);
        assert_eq!(store_a.len(), world_a.events().len());
    }

    #[test]
    fn unrenderable_events_are_refused_cleanly() {
        let world = world_ticked(42, 1);
        let mut store = SocialStore::new();
        // An event naming an agent the world does not have.
        let ghost = Event {
            tick: 1,
            kind: EventKind::Move,
            agent: Some(99),
            x: 1,
            y: 1,
            a: 90,
            b: 0,
        };
        assert!(render_event(&ghost, &world).is_none());
        assert!(matches!(
            publish_event(&mut store, &world, &ghost),
            Err(SocialError::InvalidRecord(_))
        ));
        // A gather naming an unknown resource kind.
        let weird = Event {
            tick: 1,
            kind: EventKind::Gather,
            agent: Some(0),
            x: 1,
            y: 1,
            a: 9,
            b: 0,
        };
        assert!(render_event(&weird, &world).is_none());
        assert!(publish_event(&mut store, &world, &weird).is_err());
        assert!(store.is_empty());
    }
}
