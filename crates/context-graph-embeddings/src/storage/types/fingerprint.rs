//! Stored quantized fingerprint type for primary storage.

use crate::quantization::{QuantizationMethod, QuantizedEmbedding};
use crate::types::ModelId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::constants::{NUM_EMBEDDERS, STORAGE_VERSION};

/// Complete stored fingerprint with quantized embeddings.
///
/// This struct is used for STORAGE in layer1_primary (RocksDB/ScyllaDB).
/// The actual 13× HNSW indexes (layer2c) use `IndexEntry` for dequantized vectors.
///
/// # Storage Layout
/// Each embedder's quantized embedding is stored separately for:
/// 1. Per-embedder HNSW indexing (requires dequantization)
/// 2. Lazy loading (only fetch needed embedders)
/// 3. Independent quantization per embedder
///
/// # Size Target
/// ~17KB per fingerprint (Constitution requirement)
///
/// # Difference from TeleologicalFingerprint
/// - `TeleologicalFingerprint` (in context-graph-core): ~63KB UNQUANTIZED, includes:
///   - `SemanticFingerprint` with raw f32 arrays
///   - `JohariFingerprint` per-embedder classification
///   - `PurposeSnapshot` evolution history
///
/// - `StoredQuantizedFingerprint` (this type): ~17KB QUANTIZED for storage
///   - Uses `QuantizedEmbedding` (compressed bytes)
///   - Johari summarized to 4 quadrant weights
///   - No evolution history (kept in TimescaleDB temporal store)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredQuantizedFingerprint {
    /// UUID of the fingerprint (primary key).
    pub id: Uuid,

    /// Storage format version (for future migration detection).
    pub version: u8,

    /// Per-embedder quantized embeddings.
    /// Key: embedder index (0-12)
    /// Value: Quantized embedding with method-specific metadata
    ///
    /// # Invariant
    /// All 13 embedders MUST be present. Missing embedder = panic on load.
    pub embeddings: HashMap<u8, QuantizedEmbedding>,

    /// 13D purpose vector (NOT quantized - only 52 bytes).
    /// Each dimension = alignment of that embedder's output to emergent goals.
    /// From Constitution: "PV = [A(E1,V), A(E2,V), ..., A(E13,V)]"
    pub purpose_vector: [f32; 13],

    // REMOVED: alignment_score per TASK-P0-001 (ARCH-03)
    // North Star alignment was a manual goal metric - goals now emerge autonomously
    /// Johari quadrant weights [Open, Hidden, Blind, Unknown].
    /// Aggregated from per-embedder JohariFingerprint.
    /// Used for fast filtering without loading full fingerprint.
    pub johari_quadrants: [f32; 4],

    /// Dominant Johari quadrant index (0=Open, 1=Hidden, 2=Blind, 3=Unknown).
    /// Pre-computed for fast classification queries.
    pub dominant_quadrant: u8,

    /// Johari confidence score [0.0, 1.0].
    /// Higher = more certain about quadrant classification.
    pub johari_confidence: f32,

    /// SHA-256 content hash (32 bytes).
    /// Used for deduplication and integrity verification.
    pub content_hash: [u8; 32],

    /// Creation timestamp (Unix millis since epoch).
    pub created_at_ms: i64,

    /// Last update timestamp (Unix millis since epoch).
    pub last_updated_ms: i64,

    /// Access count for LRU/importance scoring.
    pub access_count: u64,

    /// Soft-delete flag.
    /// True = marked for deletion but recoverable (30-day window per Constitution).
    pub deleted: bool,
}

impl StoredQuantizedFingerprint {
    /// Create a new StoredQuantizedFingerprint.
    ///
    /// # Arguments
    /// * `id` - UUID for this fingerprint
    /// * `embeddings` - HashMap of quantized embeddings (must have all 13)
    /// * `purpose_vector` - 13D alignment signature
    /// * `johari_quadrants` - Aggregated Johari weights
    /// * `content_hash` - SHA-256 of source content
    ///
    /// # Panics
    /// Panics if `embeddings` doesn't contain exactly 13 entries.
    #[must_use]
    pub fn new(
        id: Uuid,
        embeddings: HashMap<u8, QuantizedEmbedding>,
        purpose_vector: [f32; 13],
        johari_quadrants: [f32; 4],
        content_hash: [u8; 32],
    ) -> Self {
        // FAIL FAST: All 13 embedders required
        if embeddings.len() != NUM_EMBEDDERS {
            panic!(
                "CONSTRUCTION ERROR: StoredQuantizedFingerprint requires exactly {} embeddings, got {}. \
                 Missing embedder indices: {:?}. \
                 This indicates incomplete fingerprint generation.",
                NUM_EMBEDDERS,
                embeddings.len(),
                (0..13).filter(|i| !embeddings.contains_key(&(*i as u8))).collect::<Vec<_>>()
            );
        }

        // Verify all indices are valid (0-12)
        for idx in embeddings.keys() {
            if *idx >= NUM_EMBEDDERS as u8 {
                panic!(
                    "CONSTRUCTION ERROR: Invalid embedder index {}. Valid range: 0-12. \
                     This indicates embedding pipeline bug.",
                    idx
                );
            }
        }

        // REMOVED: alignment_score computation per TASK-P0-001 (ARCH-03)
        let (dominant_quadrant, johari_confidence) =
            Self::compute_dominant_quadrant(&johari_quadrants);
        let now = chrono::Utc::now().timestamp_millis();

        Self {
            id,
            version: STORAGE_VERSION,
            embeddings,
            purpose_vector,
            johari_quadrants,
            dominant_quadrant,
            johari_confidence,
            content_hash,
            created_at_ms: now,
            last_updated_ms: now,
            access_count: 0,
            deleted: false,
        }
    }

    /// Compute dominant quadrant and confidence from weights.
    fn compute_dominant_quadrant(quadrants: &[f32; 4]) -> (u8, f32) {
        let total: f32 = quadrants.iter().sum();
        if total < f32::EPSILON {
            return (0, 0.0); // Default to Open with zero confidence
        }

        let mut max_idx = 0u8;
        let mut max_val = quadrants[0];
        for (i, &v) in quadrants.iter().enumerate().skip(1) {
            if v > max_val {
                max_val = v;
                max_idx = i as u8;
            }
        }

        let confidence = max_val / total;
        (max_idx, confidence)
    }

    /// Compute total storage size in bytes (serialized).
    ///
    /// # Returns
    /// Estimated serialized size. Actual size may vary slightly due to encoding.
    #[must_use]
    pub fn estimated_size_bytes(&self) -> usize {
        let mut size = 0usize;

        // Fixed fields
        size += 16; // id (UUID)
        size += 1; // version
        size += 52; // purpose_vector (13 × 4 bytes)
                    // REMOVED: alignment_score (4 bytes) per TASK-P0-001
        size += 16; // johari_quadrants (4 × 4 bytes)
        size += 1; // dominant_quadrant
        size += 4; // johari_confidence
        size += 32; // content_hash
        size += 8; // created_at_ms
        size += 8; // last_updated_ms
        size += 8; // access_count
        size += 1; // deleted

        // Variable fields: embeddings
        for qe in self.embeddings.values() {
            size += 1; // method (enum variant)
            size += 8; // original_dim
            size += qe.data.len(); // compressed data
            size += 32; // metadata (approximate)
        }

        size
    }

    /// Get quantized embedding for a specific embedder.
    ///
    /// # Arguments
    /// * `embedder_idx` - Embedder index (0-12)
    ///
    /// # Panics
    /// Panics if embedder_idx is out of range or embedding is missing.
    #[must_use]
    pub fn get_embedding(&self, embedder_idx: u8) -> &QuantizedEmbedding {
        self.embeddings.get(&embedder_idx).unwrap_or_else(|| {
            panic!(
                "STORAGE ERROR: Missing embedding for embedder {}. \
                 Fingerprint ID: {}. Available embedders: {:?}. \
                 This indicates corrupted fingerprint or storage bug.",
                embedder_idx,
                self.id,
                self.embeddings.keys().collect::<Vec<_>>()
            );
        })
    }

    /// Check if all embeddings use correct quantization methods.
    ///
    /// # Returns
    /// `true` if all embeddings match their Constitution-assigned methods.
    #[must_use]
    pub fn validate_quantization_methods(&self) -> bool {
        for (idx, qe) in &self.embeddings {
            if let Ok(model_id) = ModelId::try_from(*idx) {
                let expected = QuantizationMethod::for_model_id(model_id);
                if qe.method != expected {
                    return false;
                }
            }
        }
        true
    }
}
