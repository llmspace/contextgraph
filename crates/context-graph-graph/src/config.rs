//! Configuration types for Knowledge Graph components.
//!
//! This module provides configuration structures for:
//! - FAISS IVF-PQ vector index (IndexConfig)
//! - Hyperbolic/Poincare ball geometry (HyperbolicConfig)
//! - Entailment cones for IS-A queries (ConeConfig)
//!
//! # Constitution Reference
//!
//! - perf.latency.faiss_1M_k100: <2ms (drives nlist/nprobe defaults)
//! - embeddings.models.E7_Code: 1536D (default dimension)
//!
//! TODO: Full implementation in M04-T01, M04-T02, M04-T03

use serde::{Deserialize, Serialize};

/// FAISS IVF-PQ index configuration.
///
/// Configures the FAISS GPU index for 1M+ vector search with <2ms latency.
///
/// # Performance Targets
/// - 1M vectors, k=100: <2ms latency
/// - Memory: ~8GB for 1M 1536D vectors with PQ64x8
///
/// TODO: M04-T01 - Add validation and builder pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Embedding dimension (default: 1536 per constitution E7_Code)
    pub dimension: usize,
    /// Number of inverted lists for IVF (default: 16384 = 4 * sqrt(1M))
    pub nlist: usize,
    /// Number of lists to probe during search (default: 128)
    pub nprobe: usize,
    /// Number of PQ subquantizers (default: 64, must divide dimension)
    pub pq_segments: usize,
    /// Bits per PQ code (default: 8)
    pub pq_bits: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 1536,
            nlist: 16384,
            nprobe: 128,
            pq_segments: 64,
            pq_bits: 8,
        }
    }
}

/// Hyperbolic (Poincare ball) configuration.
///
/// Configures the Poincare ball model for representing hierarchical
/// relationships in hyperbolic space.
///
/// # Mathematics
/// - d(x,y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
/// - Curvature must be negative (typically -1.0)
/// - All points must have norm < 1.0
///
/// TODO: M04-T02 - Add validation for curvature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicConfig {
    /// Dimension of hyperbolic space (default: 64)
    pub dimension: usize,
    /// Curvature parameter (must be negative, default: -1.0)
    pub curvature: f32,
    /// Maximum norm for points (default: 0.999, must be < 1.0)
    pub max_norm: f32,
}

impl Default for HyperbolicConfig {
    fn default() -> Self {
        Self {
            dimension: 64,
            curvature: -1.0,
            max_norm: 0.999,
        }
    }
}

/// Entailment cone configuration.
///
/// Configures entailment cones for O(1) IS-A hierarchy queries.
/// Cones narrow as depth increases (children have smaller apertures).
///
/// # Constitution Reference
/// - perf.latency.entailment_check: <1ms
///
/// TODO: M04-T03 - Add aperture calculation helpers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConeConfig {
    /// Base aperture angle in radians (default: PI/4 = 45 degrees)
    pub base_aperture: f32,
    /// Aperture decay factor per depth level (default: 0.9)
    pub aperture_decay: f32,
    /// Minimum aperture angle (default: 0.1 radians)
    pub min_aperture: f32,
}

impl Default for ConeConfig {
    fn default() -> Self {
        Self {
            base_aperture: std::f32::consts::FRAC_PI_4,
            aperture_decay: 0.9,
            min_aperture: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert_eq!(config.dimension, 1536);
        assert_eq!(config.nlist, 16384);
        assert_eq!(config.nprobe, 128);
        assert_eq!(config.pq_segments, 64);
        assert_eq!(config.pq_bits, 8);
    }

    #[test]
    fn test_index_config_pq_segments_divides_dimension() {
        let config = IndexConfig::default();
        assert_eq!(
            config.dimension % config.pq_segments,
            0,
            "PQ segments must divide dimension evenly"
        );
    }

    #[test]
    fn test_hyperbolic_config_default() {
        let config = HyperbolicConfig::default();
        assert_eq!(config.dimension, 64);
        assert_eq!(config.curvature, -1.0);
        assert!(config.curvature < 0.0, "Curvature must be negative");
        assert!(config.max_norm < 1.0, "Max norm must be < 1.0");
        assert!(config.max_norm > 0.0, "Max norm must be positive");
    }

    #[test]
    fn test_cone_config_default() {
        let config = ConeConfig::default();
        assert!(config.base_aperture > 0.0);
        assert!(config.base_aperture < std::f32::consts::PI);
        assert!(config.aperture_decay > 0.0 && config.aperture_decay <= 1.0);
        assert!(config.min_aperture > 0.0);
        assert!(config.min_aperture < config.base_aperture);
    }

    #[test]
    fn test_index_config_serialization() {
        let config = IndexConfig::default();
        let json = serde_json::to_string(&config).expect("Failed to serialize");
        let deserialized: IndexConfig =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(config.dimension, deserialized.dimension);
    }

    #[test]
    fn test_hyperbolic_config_serialization() {
        let config = HyperbolicConfig::default();
        let json = serde_json::to_string(&config).expect("Failed to serialize");
        let deserialized: HyperbolicConfig =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(config.curvature, deserialized.curvature);
    }

    #[test]
    fn test_cone_config_serialization() {
        let config = ConeConfig::default();
        let json = serde_json::to_string(&config).expect("Failed to serialize");
        let deserialized: ConeConfig =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(config.base_aperture, deserialized.base_aperture);
    }
}
