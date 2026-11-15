//! Ad and tracker blocklists.
//!
//! A [`Blocklist`] is two lists of domains: blocked domains (ads and
//! trackers) and an allowlist. The allowlist wins: a domain on both
//! lists is allowed. Matching is exact or by suffix — a subdomain of
//! a blocked domain is blocked too (`ads.doubleclick.net` is blocked
//! by `doubleclick.net`), but a domain that merely ends in the same
//! letters is not (`notdoubleclick.net` is not blocked by
//! `doubleclick.net`).
//!
//! The crate ships a small default set of well-known ad and tracker
//! domains ([`DEFAULT_BLOCKED_DOMAINS`]). It is a starting list the
//! owner can replace, not a live feed: nothing here downloads
//! updates, and no list here is guaranteed complete or current. An
//! ad network or tracker that is not on the list is not blocked.
//! Adding a domain is the owner's explicit act; the shield never
//! fetches a list from anywhere.

use std::collections::HashSet;

use crate::ShieldError;

/// Well-known ad and tracker domains shipped as the default
/// blocklist. These are the obvious, long-lived ones — ad servers,
/// analytics endpoints, and mobile attribution trackers. The owner
/// can replace this list wholesale; it is a starting point, not a
/// claim that these are the only trackers in the world.
pub const DEFAULT_BLOCKED_DOMAINS: &[&str] = &[
    // Ad serving.
    "doubleclick.net",
    "googleadservices.com",
    "googlesyndication.com",
    "adnxs.com",
    "adsrvr.org",
    "criteo.com",
    "taboola.com",
    "outbrain.com",
    "moatads.com",
    "ads.yahoo.com",
    // Analytics and tracking.
    "google-analytics.com",
    "googletagmanager.com",
    "connect.facebook.net",
    "analytics.twitter.com",
    "scorecardresearch.com",
    "hotjar.com",
    "mixpanel.com",
    "segment.io",
    "amplitude.com",
    // Mobile attribution (the trackers inside apps).
    "branch.io",
    "appsflyer.com",
    "adjust.com",
];

/// Normalize a host for matching: lowercase, strip a trailing dot
/// (the fully-qualified form), and strip a numeric port
/// (`example.com:443` becomes `example.com`). Bracketed IPv6
/// literals are left alone — they never match a domain.
pub fn normalize_host(host: &str) -> String {
    let mut h = host.trim().to_ascii_lowercase();
    while h.ends_with('.') {
        h.pop();
    }
    if !h.starts_with('[') {
        if let Some(idx) = h.rfind(':') {
            let port = &h[idx + 1..];
            if !port.is_empty() && port.chars().all(|c| c.is_ascii_digit()) {
                h.truncate(idx);
            }
        }
    }
    h
}

/// Rules for domains: 1-253 characters after normalization, labels
/// of 1-63 characters, lowercase letters, digits, and `-` only, no
/// label starting or ending with `-`.
pub fn validate_domain(domain: &str) -> Result<(), ShieldError> {
    let d = normalize_host(domain);
    if d.is_empty() {
        return Err(ShieldError::InvalidDomain(
            "domain must not be empty".to_string(),
        ));
    }
    if d.len() > 253 {
        return Err(ShieldError::InvalidDomain(
            "domain must be at most 253 characters".to_string(),
        ));
    }
    for label in d.split('.') {
        if label.is_empty() {
            return Err(ShieldError::InvalidDomain(
                "domain must not contain an empty label".to_string(),
            ));
        }
        if label.len() > 63 {
            return Err(ShieldError::InvalidDomain(
                "domain label must be at most 63 characters".to_string(),
            ));
        }
        if label.starts_with('-') || label.ends_with('-') {
            return Err(ShieldError::InvalidDomain(
                "domain label must not start or end with '-'".to_string(),
            ));
        }
        if !label
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(ShieldError::InvalidDomain(
                "domain may only use lowercase letters, digits, and '-'".to_string(),
            ));
        }
    }
    Ok(())
}

/// Two domain lists: blocked (ads and trackers) and allowed (the
/// allowlist, which overrides the blocklist). Matching is exact or
/// by suffix, on normalized hosts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Blocklist {
    blocked: HashSet<String>,
    allowed: HashSet<String>,
}

impl Blocklist {
    /// An empty blocklist: nothing blocked, nothing allowlisted.
    pub fn new() -> Self {
        Self::default()
    }

    /// A blocklist seeded with [`DEFAULT_BLOCKED_DOMAINS`].
    pub fn with_defaults() -> Self {
        let mut list = Self::new();
        for domain in DEFAULT_BLOCKED_DOMAINS {
            list.blocked.insert(domain.to_string());
        }
        list
    }

    /// Add a domain to the blocklist. Returns true if it was new.
    pub fn block_domain(&mut self, domain: &str) -> Result<bool, ShieldError> {
        validate_domain(domain)?;
        Ok(self.blocked.insert(normalize_host(domain)))
    }

    /// Remove a domain from the blocklist. Returns true if it was
    /// listed.
    pub fn unblock_domain(&mut self, domain: &str) -> bool {
        self.blocked.remove(&normalize_host(domain))
    }

    /// Add a domain to the allowlist. Returns true if it was new.
    /// An allowlisted domain is never blocked, even if it is also
    /// on the blocklist.
    pub fn allow_domain(&mut self, domain: &str) -> Result<bool, ShieldError> {
        validate_domain(domain)?;
        Ok(self.allowed.insert(normalize_host(domain)))
    }

    /// Remove a domain from the allowlist. Returns true if it was
    /// listed.
    pub fn unallow_domain(&mut self, domain: &str) -> bool {
        self.allowed.remove(&normalize_host(domain))
    }

    /// True if `host` matches a blocked domain, exactly or as a
    /// subdomain.
    pub fn is_blocked(&self, host: &str) -> bool {
        let host = normalize_host(host);
        self.blocked
            .iter()
            .any(|d| host == *d || host.ends_with(&format!(".{d}")))
    }

    /// True if `host` matches an allowlisted domain, exactly or as
    /// a subdomain.
    pub fn is_allowed(&self, host: &str) -> bool {
        let host = normalize_host(host);
        self.allowed
            .iter()
            .any(|d| host == *d || host.ends_with(&format!(".{d}")))
    }

    /// The blocklist decision for a host: blocked unless allowlisted.
    /// The allowlist wins over the blocklist.
    pub fn should_block(&self, host: &str) -> bool {
        !self.is_allowed(host) && self.is_blocked(host)
    }

    /// Number of domains on the blocklist.
    pub fn blocked_count(&self) -> usize {
        self.blocked.len()
    }

    /// Number of domains on the allowlist.
    pub fn allowed_count(&self) -> usize {
        self.allowed.len()
    }

    /// Blocked domains, sorted, for inspection.
    pub fn blocked_domains(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self.blocked.iter().map(String::as_str).collect();
        out.sort_unstable();
        out
    }

    /// Allowlisted domains, sorted, for inspection.
    pub fn allowed_domains(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self.allowed.iter().map(String::as_str).collect();
        out.sort_unstable();
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_list_blocks_well_known_ads_and_trackers() {
        let list = Blocklist::with_defaults();
        assert_eq!(list.blocked_count(), DEFAULT_BLOCKED_DOMAINS.len());
        assert!(list.should_block("doubleclick.net"));
        assert!(list.should_block("google-analytics.com"));
        assert!(list.should_block("appsflyer.com"));
        assert!(!list.should_block("example.com"));
        assert!(!list.should_block("kernel.local"));
    }

    #[test]
    fn subdomain_of_blocked_domain_is_blocked() {
        let mut list = Blocklist::new();
        list.block_domain("doubleclick.net").unwrap();
        assert!(list.should_block("ads.doubleclick.net"));
        assert!(list.should_block("a.b.doubleclick.net"));
        // A domain that merely ends in the same letters is NOT a
        // subdomain: the match needs the dot boundary.
        assert!(!list.should_block("notdoubleclick.net"));
        assert!(!list.should_block("doubleclick.net.evil.com"));
    }

    #[test]
    fn allowlist_overrides_the_blocklist() {
        let mut list = Blocklist::with_defaults();
        assert!(list.should_block("google-analytics.com"));
        assert!(list.allow_domain("google-analytics.com").unwrap());
        assert!(!list.should_block("google-analytics.com"));
        // The allowlist also covers subdomains.
        assert!(!list.should_block("ssl.google-analytics.com"));
        // Removing the allow entry restores the block.
        assert!(list.unallow_domain("google-analytics.com"));
        assert!(list.should_block("google-analytics.com"));
        assert!(!list.unallow_domain("google-analytics.com"));
    }

    #[test]
    fn hosts_are_normalized_before_matching() {
        let mut list = Blocklist::new();
        list.block_domain("DoubleClick.NET").unwrap();
        assert!(list.should_block("DOUBLECLICK.net"));
        assert!(list.should_block("doubleclick.net."));
        assert!(list.should_block("doubleclick.net:443"));
        assert!(list.should_block("ads.doubleclick.net:8080"));
        assert!(list.is_blocked("  doubleclick.net  "));
    }

    #[test]
    fn invalid_domains_are_refused() {
        let mut list = Blocklist::new();
        assert!(list.block_domain("").is_err());
        assert!(list.block_domain("exa mple.com").is_err());
        assert!(list.block_domain("exa_mple.com").is_err());
        assert!(list.block_domain("-example.com").is_err());
        assert!(list.block_domain("example-.com").is_err());
        assert!(list.block_domain("example..com").is_err());
        assert!(list.block_domain(&"a".repeat(64)).is_err());
        assert!(list.block_domain("example.com").is_ok());
        // Blocking the same domain twice is a no-op (not an error).
        assert!(!list.block_domain("example.com").unwrap());
        assert!(list.unblock_domain("example.com"));
        assert!(!list.unblock_domain("example.com"));
    }

    #[test]
    fn blocked_and_allowed_domains_list_sorted() {
        let mut list = Blocklist::new();
        list.block_domain("zebra.example").unwrap();
        list.block_domain("apple.example").unwrap();
        list.allow_domain("mango.example").unwrap();
        assert_eq!(
            list.blocked_domains(),
            vec!["apple.example", "zebra.example"]
        );
        assert_eq!(list.allowed_domains(), vec!["mango.example"]);
    }
}
