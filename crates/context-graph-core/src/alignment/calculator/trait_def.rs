//! Goal alignment calculator trait definition.

use async_trait::async_trait;

use super::super::config::AlignmentConfig;
use super::super::error::AlignmentError;
use super::super::misalignment::MisalignmentFlags;
use super::super::pattern::AlignmentPattern;
use super::super::score::GoalAlignmentScore;
use super::result::AlignmentResult;
use crate::types::fingerprint::TeleologicalFingerprint;

/// Trait for computing goal alignment.
///
/// Implementations must be thread-safe (Send + Sync) and should
/// complete within the configured timeout (default 5ms).
#[async_trait]
pub trait GoalAlignmentCalculator: Send + Sync {
    /// Compute alignment for a single fingerprint.
    ///
    /// # Arguments
    /// * `fingerprint` - The teleological fingerprint to evaluate
    /// * `config` - Configuration for the computation
    ///
    /// # Errors
    /// Returns error if:
    /// - No top-level goals in hierarchy
    /// - Fingerprint is empty
    /// - Computation times out
    async fn compute_alignment(
        &self,
        fingerprint: &TeleologicalFingerprint,
        config: &AlignmentConfig,
    ) -> Result<AlignmentResult, AlignmentError>;

    /// Compute alignment for multiple fingerprints.
    ///
    /// More efficient than calling `compute_alignment` in a loop.
    /// Implementations may parallelize internally.
    ///
    /// # Arguments
    /// * `fingerprints` - Slice of fingerprints to evaluate
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// Vec of results in same order as input. Each element is
    /// either Ok(result) or Err(error) for that fingerprint.
    async fn compute_alignment_batch(
        &self,
        fingerprints: &[&TeleologicalFingerprint],
        config: &AlignmentConfig,
    ) -> Vec<Result<AlignmentResult, AlignmentError>>;

    /// Detect misalignment patterns from a result.
    ///
    /// Called automatically if `config.detect_patterns` is true.
    fn detect_patterns(
        &self,
        score: &GoalAlignmentScore,
        flags: &MisalignmentFlags,
        config: &AlignmentConfig,
    ) -> Vec<AlignmentPattern>;
}
