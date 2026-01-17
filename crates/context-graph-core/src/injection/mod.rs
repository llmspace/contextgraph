//! Injection pipeline types for context injection.
//!
//! This module provides types for the injection pipeline:
//! - [`InjectionCandidate`] - Memory candidate for injection with scores
//! - [`InjectionCategory`] - Priority category determining budget
//!
//! # Constitution Compliance
//! - ARCH-09: Topic threshold = weighted_agreement >= 2.5
//! - AP-60: Temporal embedders NEVER count toward topics

pub mod candidate;

pub use candidate::{
    InjectionCandidate, InjectionCategory, MAX_DIVERSITY_BONUS, MAX_RECENCY_FACTOR,
    MAX_WEIGHTED_AGREEMENT, MIN_DIVERSITY_BONUS, MIN_RECENCY_FACTOR, TOKEN_MULTIPLIER,
};
