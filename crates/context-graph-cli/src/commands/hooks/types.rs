//! HookEventType enum for Claude Code native hooks
//!
//! # Performance Budget (per TECH-HOOKS.md)
//! - PreToolUse: 100ms (FAST PATH - NO DB ACCESS)
//! - UserPromptSubmit: 2000ms
//! - PostToolUse: 3000ms
//! - SessionStart: 5000ms
//! - SessionEnd: 30000ms
//!
//! # Constitution References
//! - IDENTITY-002: IC thresholds and timeout requirements
//! - AP-26: Exit codes (0=success, 1=error, 2=corruption)
//!
//! # NO BACKWARDS COMPATIBILITY - FAIL FAST

use serde::{Deserialize, Serialize};

/// Hook event types matching Claude Code native hooks
/// Implements REQ-HOOKS-01 through REQ-HOOKS-05
///
/// # Timeout Values (Claude Code enforced)
/// | Event | Timeout | Description |
/// |-------|---------|-------------|
/// | SessionStart | 5000ms | Session initialization |
/// | PreToolUse | 100ms | FAST PATH - cache only |
/// | PostToolUse | 3000ms | IC verification |
/// | UserPromptSubmit | 2000ms | Context injection |
/// | SessionEnd | 30000ms | Final persistence |
///
/// # JSON Serialization
/// Uses snake_case: `session_start`, `pre_tool_use`, etc.
///
/// # Example
/// ```
/// use context_graph_cli::commands::hooks::HookEventType;
///
/// let hook = HookEventType::PreToolUse;
/// assert_eq!(hook.timeout_ms(), 100);
/// assert!(hook.is_fast_path());
///
/// let json = serde_json::to_string(&hook).expect("serialization must succeed");
/// assert_eq!(json, "\"pre_tool_use\"");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEventType {
    /// Session initialization (timeout: 5000ms)
    /// Triggered: startup, resume, /clear
    /// CLI: `session restore-identity`
    SessionStart,

    /// Before tool execution (timeout: 100ms) - FAST PATH
    /// CRITICAL: Must not access database, uses IdentityCache only
    /// CLI: `consciousness brief`
    PreToolUse,

    /// After tool execution (timeout: 3000ms)
    /// Updates IC and trajectory based on tool result
    /// CLI: `consciousness check-identity --auto-dream`
    PostToolUse,

    /// User prompt submitted (timeout: 2000ms)
    /// Injects relevant context from session memory
    /// CLI: `consciousness inject-context --format standard`
    UserPromptSubmit,

    /// Session termination (timeout: 30000ms)
    /// Persists final snapshot and optional consolidation
    /// CLI: `session persist-identity`
    SessionEnd,
}

impl HookEventType {
    /// Get timeout in milliseconds for this hook type
    /// Constitution Reference: IDENTITY-002
    ///
    /// # Returns
    /// Timeout value in milliseconds as enforced by Claude Code
    ///
    /// # Performance Note
    /// PreToolUse has the strictest timeout (100ms) and MUST use
    /// cached state only - NO database access allowed.
    #[inline]
    pub const fn timeout_ms(&self) -> u64 {
        match self {
            Self::PreToolUse => 100,        // Fast path - NO DB access
            Self::UserPromptSubmit => 2000, // Context injection
            Self::PostToolUse => 3000,      // IC update + trajectory
            Self::SessionStart => 5000,     // Load/create snapshot
            Self::SessionEnd => 30000,      // Final persist + consolidation
        }
    }

    /// Check if this hook type is time-critical (fast path)
    ///
    /// # Returns
    /// `true` if timeout is under 500ms (only PreToolUse)
    ///
    /// # Performance Implications
    /// Fast path hooks MUST NOT:
    /// - Access disk/database
    /// - Perform network calls
    /// - Block on locks for more than microseconds
    #[inline]
    pub const fn is_fast_path(&self) -> bool {
        self.timeout_ms() < 500
    }

    /// Get human-readable description of this hook type
    ///
    /// # Returns
    /// Static string describing the hook's purpose
    pub const fn description(&self) -> &'static str {
        match self {
            Self::SessionStart => "Session initialization and identity restoration",
            Self::PreToolUse => "Pre-tool consciousness brief injection (FAST PATH)",
            Self::PostToolUse => "Post-tool identity continuity verification",
            Self::UserPromptSubmit => "User prompt context injection",
            Self::SessionEnd => "Session persistence and consolidation",
        }
    }

    /// Get the corresponding CLI command for this hook type
    ///
    /// # Returns
    /// CLI command string that implements this hook
    pub const fn cli_command(&self) -> &'static str {
        match self {
            Self::SessionStart => "session restore-identity",
            Self::PreToolUse => "consciousness brief",
            Self::PostToolUse => "consciousness check-identity --auto-dream",
            Self::UserPromptSubmit => "consciousness inject-context --format standard",
            Self::SessionEnd => "session persist-identity",
        }
    }

    /// Get all hook event types as an array
    ///
    /// # Returns
    /// Array of all 5 hook event types in execution order
    pub const fn all() -> [Self; 5] {
        [
            Self::SessionStart,
            Self::PreToolUse,
            Self::PostToolUse,
            Self::UserPromptSubmit,
            Self::SessionEnd,
        ]
    }
}

impl std::fmt::Display for HookEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// =============================================================================
// TESTS - NO MOCK DATA - REAL VALUES ONLY
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SOURCE OF TRUTH: TECH-HOOKS.md Section 2.2 + .claude/settings.json
    // =========================================================================

    // =========================================================================
    // TC-HOOKS-001: Timeout Values Match Specification
    // SOURCE: TECH-HOOKS.md Section 2.2, constitution.yaml claude_code.hooks
    // =========================================================================
    #[test]
    fn tc_hooks_001_timeout_values_match_spec() {
        println!("\n=== TC-HOOKS-001: Timeout Values Match Specification ===");
        println!("SOURCE OF TRUTH: TECH-HOOKS.md Section 2.2");
        println!("CONSTITUTION: claude_code.hooks.timeouts");

        // These values are from TECH-HOOKS.md and constitution.yaml
        // DO NOT CHANGE without updating both sources
        let expected_timeouts = [
            (HookEventType::SessionStart, 5000_u64, "session_start"),
            (HookEventType::PreToolUse, 100_u64, "pre_tool_use"),
            (HookEventType::PostToolUse, 3000_u64, "post_tool_use"),
            (HookEventType::UserPromptSubmit, 2000_u64, "user_prompt_submit"),
            (HookEventType::SessionEnd, 30000_u64, "session_end"),
        ];

        for (hook, expected_timeout, name) in expected_timeouts {
            let actual = hook.timeout_ms();
            println!("  {}: expected={}ms, actual={}ms", name, expected_timeout, actual);
            assert_eq!(
                actual, expected_timeout,
                "FAIL: {} timeout must be {}ms, got {}ms",
                name, expected_timeout, actual
            );
        }

        println!("RESULT: PASS - All timeout values match specification");
    }

    // =========================================================================
    // TC-HOOKS-002: Serialization Produces snake_case
    // SOURCE: Claude Code hooks JSON format requirement
    // =========================================================================
    #[test]
    fn tc_hooks_002_serialization_snake_case() {
        println!("\n=== TC-HOOKS-002: Serialization Produces snake_case ===");
        println!("SOURCE OF TRUTH: Claude Code hook JSON format");

        let test_cases = [
            (HookEventType::SessionStart, r#""session_start""#),
            (HookEventType::PreToolUse, r#""pre_tool_use""#),
            (HookEventType::PostToolUse, r#""post_tool_use""#),
            (HookEventType::UserPromptSubmit, r#""user_prompt_submit""#),
            (HookEventType::SessionEnd, r#""session_end""#),
        ];

        for (hook, expected_json) in test_cases {
            let actual_json = serde_json::to_string(&hook)
                .expect("serialization MUST succeed - fail fast");
            println!("  {:?} -> {}", hook, actual_json);
            assert_eq!(
                actual_json, expected_json,
                "FAIL: {:?} must serialize to {}, got {}",
                hook, expected_json, actual_json
            );
        }

        println!("RESULT: PASS - All variants serialize to snake_case");
    }

    // =========================================================================
    // TC-HOOKS-003: Deserialization from snake_case
    // SOURCE: Claude Code hook JSON format requirement
    // =========================================================================
    #[test]
    fn tc_hooks_003_deserialization_snake_case() {
        println!("\n=== TC-HOOKS-003: Deserialization from snake_case ===");
        println!("SOURCE OF TRUTH: Claude Code hook JSON format");

        let test_cases = [
            (r#""session_start""#, HookEventType::SessionStart),
            (r#""pre_tool_use""#, HookEventType::PreToolUse),
            (r#""post_tool_use""#, HookEventType::PostToolUse),
            (r#""user_prompt_submit""#, HookEventType::UserPromptSubmit),
            (r#""session_end""#, HookEventType::SessionEnd),
        ];

        for (json, expected_hook) in test_cases {
            let actual_hook: HookEventType = serde_json::from_str(json)
                .expect("deserialization MUST succeed - fail fast");
            println!("  {} -> {:?}", json, actual_hook);
            assert_eq!(
                actual_hook, expected_hook,
                "FAIL: {} must deserialize to {:?}, got {:?}",
                json, expected_hook, actual_hook
            );
        }

        println!("RESULT: PASS - All snake_case strings deserialize correctly");
    }

    // =========================================================================
    // TC-HOOKS-004: Exactly 5 Variants Exist
    // SOURCE: Claude Code native hook specification
    // =========================================================================
    #[test]
    fn tc_hooks_004_exactly_five_variants() {
        println!("\n=== TC-HOOKS-004: Exactly 5 Variants Exist ===");
        println!("SOURCE OF TRUTH: Claude Code native hook specification");

        let all_variants = HookEventType::all();
        println!("  Variant count: {}", all_variants.len());

        assert_eq!(
            all_variants.len(), 5,
            "FAIL: Must have exactly 5 variants, got {}",
            all_variants.len()
        );

        // Verify all variants are unique
        let mut seen = std::collections::HashSet::new();
        for variant in all_variants {
            assert!(
                seen.insert(variant),
                "FAIL: Duplicate variant detected: {:?}",
                variant
            );
        }

        println!("  All variants unique: true");
        println!("RESULT: PASS - Exactly 5 unique variants exist");
    }

    // =========================================================================
    // TC-HOOKS-005: Fast Path Detection
    // SOURCE: TECH-HOOKS.md fast path requirement (<500ms)
    // =========================================================================
    #[test]
    fn tc_hooks_005_fast_path_detection() {
        println!("\n=== TC-HOOKS-005: Fast Path Detection ===");
        println!("SOURCE OF TRUTH: TECH-HOOKS.md fast path requirement");
        println!("THRESHOLD: timeout < 500ms");

        let fast_path_expected = [
            (HookEventType::SessionStart, false),
            (HookEventType::PreToolUse, true),  // ONLY fast path
            (HookEventType::PostToolUse, false),
            (HookEventType::UserPromptSubmit, false),
            (HookEventType::SessionEnd, false),
        ];

        for (hook, expected_fast) in fast_path_expected {
            let actual_fast = hook.is_fast_path();
            println!(
                "  {:?}: timeout={}ms, is_fast_path={} (expected={})",
                hook, hook.timeout_ms(), actual_fast, expected_fast
            );
            assert_eq!(
                actual_fast, expected_fast,
                "FAIL: {:?}.is_fast_path() must be {}, got {}",
                hook, expected_fast, actual_fast
            );
        }

        println!("RESULT: PASS - Only PreToolUse is fast path");
    }

    // =========================================================================
    // TC-HOOKS-006: Copy and Clone Traits
    // SOURCE: Rust type safety requirement
    // =========================================================================
    #[test]
    fn tc_hooks_006_copy_clone_traits() {
        println!("\n=== TC-HOOKS-006: Copy and Clone Traits ===");
        println!("SOURCE OF TRUTH: Rust type system requirements");

        let original = HookEventType::PreToolUse;
        let copied = original;  // Copy
        let cloned = original.clone();  // Clone

        assert_eq!(original, copied, "FAIL: Copy must preserve value");
        assert_eq!(original, cloned, "FAIL: Clone must preserve value");

        // Verify we can use original after copy (proves Copy, not Move)
        assert_eq!(original.timeout_ms(), 100);

        println!("  Original after copy: {:?}", original);
        println!("  Copied: {:?}", copied);
        println!("  Cloned: {:?}", cloned);
        println!("RESULT: PASS - Copy and Clone work correctly");
    }

    // =========================================================================
    // TC-HOOKS-007: Hash Trait for HashMap Usage
    // SOURCE: Rust HashMap requirement
    // =========================================================================
    #[test]
    fn tc_hooks_007_hash_trait() {
        println!("\n=== TC-HOOKS-007: Hash Trait for HashMap Usage ===");
        println!("SOURCE OF TRUTH: Rust HashMap requirement");

        use std::collections::HashMap;

        let mut map: HashMap<HookEventType, u64> = HashMap::new();
        for hook in HookEventType::all() {
            map.insert(hook, hook.timeout_ms());
        }

        assert_eq!(map.len(), 5, "FAIL: HashMap must contain all 5 variants");
        assert_eq!(
            map.get(&HookEventType::PreToolUse),
            Some(&100),
            "FAIL: PreToolUse must map to 100"
        );

        println!("  HashMap size: {}", map.len());
        println!("  PreToolUse lookup: {:?}", map.get(&HookEventType::PreToolUse));
        println!("RESULT: PASS - Hash trait works for HashMap");
    }

    // =========================================================================
    // TC-HOOKS-008: CLI Command Mapping
    // SOURCE: .claude/settings.json hook configuration
    // =========================================================================
    #[test]
    fn tc_hooks_008_cli_command_mapping() {
        println!("\n=== TC-HOOKS-008: CLI Command Mapping ===");
        println!("SOURCE OF TRUTH: .claude/settings.json");

        let expected_commands = [
            (HookEventType::SessionStart, "session restore-identity"),
            (HookEventType::PreToolUse, "consciousness brief"),
            (HookEventType::PostToolUse, "consciousness check-identity --auto-dream"),
            (HookEventType::UserPromptSubmit, "consciousness inject-context --format standard"),
            (HookEventType::SessionEnd, "session persist-identity"),
        ];

        for (hook, expected_cmd) in expected_commands {
            let actual_cmd = hook.cli_command();
            println!("  {:?} -> \"{}\"", hook, actual_cmd);
            assert_eq!(
                actual_cmd, expected_cmd,
                "FAIL: {:?}.cli_command() must be \"{}\", got \"{}\"",
                hook, expected_cmd, actual_cmd
            );
        }

        println!("RESULT: PASS - All CLI commands match .claude/settings.json");
    }

    // =========================================================================
    // TC-HOOKS-009: Display Trait Implementation
    // SOURCE: Rust Display trait requirement
    // =========================================================================
    #[test]
    fn tc_hooks_009_display_trait() {
        println!("\n=== TC-HOOKS-009: Display Trait Implementation ===");

        for hook in HookEventType::all() {
            let display = format!("{}", hook);
            let description = hook.description();
            println!("  {:?} displays as: \"{}\"", hook, display);
            assert_eq!(
                display, description,
                "FAIL: Display must equal description"
            );
            assert!(!display.is_empty(), "FAIL: Display must not be empty");
        }

        println!("RESULT: PASS - Display trait works correctly");
    }

    // =========================================================================
    // TC-HOOKS-010: Invalid Deserialization Fails Fast
    // SOURCE: NO BACKWARDS COMPATIBILITY requirement
    // =========================================================================
    #[test]
    fn tc_hooks_010_invalid_deserialization_fails() {
        println!("\n=== TC-HOOKS-010: Invalid Deserialization Fails Fast ===");
        println!("SOURCE OF TRUTH: NO BACKWARDS COMPATIBILITY requirement");

        let invalid_inputs = [
            r#""SessionStart""#,      // PascalCase - INVALID
            r#""sessionstart""#,      // lowercase no underscore - INVALID
            r#""SESSIONSTART""#,      // UPPERCASE - INVALID
            r#""session-start""#,     // kebab-case - INVALID
            r#""unknown_hook""#,      // non-existent variant - INVALID
            r#"0"#,                   // numeric - INVALID
            r#"null"#,                // null - INVALID
        ];

        for invalid in invalid_inputs {
            let result: Result<HookEventType, _> = serde_json::from_str(invalid);
            println!("  {} -> {:?}", invalid, result.is_err());
            assert!(
                result.is_err(),
                "FAIL: Invalid input {} must fail deserialization",
                invalid
            );
        }

        println!("RESULT: PASS - All invalid inputs fail fast");
    }
}
