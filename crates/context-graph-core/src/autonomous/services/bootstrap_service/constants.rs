//! Constants used in goal extraction and analysis
//!
//! Contains keyword lists for identifying goal-related content
//! and purpose statement patterns in documents.

/// Keywords that indicate goal-related content
pub const GOAL_KEYWORDS: &[&str] = &[
    "goal",
    "mission",
    "purpose",
    "objective",
    "vision",
    "aim",
    "target",
    "north star",
    "achieve",
    "accomplish",
    "deliver",
    "provide",
    "enable",
    "empower",
    "transform",
    "create",
    "build",
    "implement",
    "system",
    "architecture",
    "framework",
    "platform",
];

/// Sentence starters that often indicate purpose statements
pub const PURPOSE_STARTERS: &[&str] = &[
    "the goal",
    "our mission",
    "the purpose",
    "this project",
    "we aim",
    "designed to",
    "intended to",
    "built to",
    "created to",
    "enables",
    "provides",
    "delivers",
];
