//! Types for PruningService
//!
//! Contains all data structures used by the pruning service including
//! candidates, metadata, reports, and configuration.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::autonomous::curation::{MemoryId, PruningConfig};

// ============================================================================
// PruneReason
// ============================================================================

/// Extended prune reasons beyond base curation types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PruneReason {
    /// Alignment score below threshold
    LowAlignment,
    /// Memory not accessed for extended period
    Stale,
    /// No connections to other memories (isolated)
    Orphaned,
    /// Duplicate or near-duplicate of another memory
    Redundant,
    /// Quality metrics below acceptable threshold
    LowQuality,
}

impl PruneReason {
    /// Get a human-readable description of the prune reason
    pub fn description(&self) -> &'static str {
        match self {
            PruneReason::LowAlignment => "Alignment score below threshold",
            PruneReason::Stale => "Not accessed for extended period",
            PruneReason::Orphaned => "No connections to other memories",
            PruneReason::Redundant => "Duplicate or near-duplicate content",
            PruneReason::LowQuality => "Quality metrics below threshold",
        }
    }
}

// ============================================================================
// MemoryMetadata
// ============================================================================

/// Metadata about a memory for pruning evaluation
#[derive(Clone, Debug)]
pub struct MemoryMetadata {
    /// Unique memory identifier
    pub id: MemoryId,
    /// When the memory was created
    pub created_at: DateTime<Utc>,
    /// Topic alignment score [0.0, 1.0]
    pub alignment: f32,
    /// Number of connections to other memories
    pub connection_count: u32,
    /// Size in bytes
    pub byte_size: u64,
    /// Last time memory was retrieved/accessed
    pub last_accessed: Option<DateTime<Utc>>,
    /// Quality score from content analysis [0.0, 1.0]
    pub quality_score: Option<f32>,
    /// Content hash for redundancy detection
    pub content_hash: Option<u64>,
}

impl MemoryMetadata {
    /// Create new memory metadata
    pub fn new(id: MemoryId, created_at: DateTime<Utc>, alignment: f32) -> Self {
        Self {
            id,
            created_at,
            alignment,
            connection_count: 0,
            byte_size: 0,
            last_accessed: None,
            quality_score: None,
            content_hash: None,
        }
    }

    /// Calculate age in days from creation time
    pub fn age_days(&self) -> u32 {
        let now = Utc::now();
        let duration = now.signed_duration_since(self.created_at);
        duration.num_days().max(0) as u32
    }

    /// Calculate days since last access (None if never accessed)
    pub fn days_since_access(&self) -> Option<u32> {
        self.last_accessed.map(|accessed| {
            let now = Utc::now();
            let duration = now.signed_duration_since(accessed);
            duration.num_days().max(0) as u32
        })
    }
}

// ============================================================================
// PruningCandidate
// ============================================================================

/// A candidate identified for potential pruning
#[derive(Clone, Debug)]
pub struct PruningCandidate {
    /// Memory identifier
    pub memory_id: MemoryId,
    /// Age of the memory in days
    pub age_days: u32,
    /// Alignment score
    pub alignment: f32,
    /// Number of connections
    pub connections: u32,
    /// Why this memory is a candidate for pruning
    pub reason: PruneReason,
    /// Size in bytes (for freed space calculation)
    pub byte_size: u64,
    /// Pruning priority score (higher = more urgent to prune)
    pub priority_score: f32,
}

impl PruningCandidate {
    /// Create a new pruning candidate
    pub fn new(
        memory_id: MemoryId,
        age_days: u32,
        alignment: f32,
        connections: u32,
        reason: PruneReason,
        byte_size: u64,
    ) -> Self {
        // Calculate priority score: lower alignment + older age = higher priority
        let alignment_factor = 1.0 - alignment;
        let age_factor = (age_days as f32 / 365.0).min(1.0);
        let connection_factor = 1.0 / (connections as f32 + 1.0);
        let priority_score = alignment_factor * 0.5 + age_factor * 0.3 + connection_factor * 0.2;

        Self {
            memory_id,
            age_days,
            alignment,
            connections,
            reason,
            byte_size,
            priority_score,
        }
    }
}

// ============================================================================
// PruningReport
// ============================================================================

/// Report from a pruning operation
#[derive(Clone, Debug, Default)]
pub struct PruningReport {
    /// Number of memories evaluated for pruning
    pub candidates_evaluated: usize,
    /// Number of memories actually pruned
    pub pruned_count: usize,
    /// Total bytes freed by pruning
    pub bytes_freed: u64,
    /// Breakdown of prune reasons
    pub reasons_breakdown: HashMap<PruneReason, usize>,
    /// Number of candidates preserved (met protection criteria)
    pub preserved_count: usize,
    /// Whether daily limit was reached
    pub daily_limit_reached: bool,
}

impl PruningReport {
    /// Create an empty report
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pruned candidate to the report
    pub fn record_prune(&mut self, candidate: &PruningCandidate) {
        self.pruned_count += 1;
        self.bytes_freed += candidate.byte_size;
        *self
            .reasons_breakdown
            .entry(candidate.reason.clone())
            .or_insert(0) += 1;
    }

    /// Record a preserved candidate
    pub fn record_preserved(&mut self) {
        self.preserved_count += 1;
    }
}

// ============================================================================
// ExtendedPruningConfig
// ============================================================================

/// Extended pruning configuration with daily limits
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtendedPruningConfig {
    /// Base pruning configuration
    #[serde(flatten)]
    pub base: PruningConfig,
    /// Maximum prunes per day
    pub max_daily_prunes: u32,
    /// Stale threshold in days (no access)
    pub stale_days: u32,
    /// Quality score threshold
    pub min_quality: f32,
}

impl Default for ExtendedPruningConfig {
    fn default() -> Self {
        Self {
            base: PruningConfig::default(),
            max_daily_prunes: 100,
            stale_days: 90,
            min_quality: 0.30,
        }
    }
}
