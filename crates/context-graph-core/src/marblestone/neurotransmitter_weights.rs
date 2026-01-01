//! Neurotransmitter-inspired weight modulation for graph edges.
//!
//! # Constitution Reference
//! - edge_model.nt_weights section
//! - Formula: w_eff = base × (1 + excitatory - inhibitory + 0.5×modulatory)

use serde::{Deserialize, Serialize};

use super::domain::Domain;

/// Neurotransmitter-inspired weight modulation for graph edges.
///
/// Based on the Marblestone architecture, edges are modulated by three signals:
/// - **Excitatory**: Strengthens connections (analogous to glutamate)
/// - **Inhibitory**: Weakens connections (analogous to GABA)
/// - **Modulatory**: Context-dependent adjustment (analogous to dopamine/serotonin)
///
/// # Constitution Reference
/// - edge_model.nt_weights section
/// - All weights must be in [0.0, 1.0] per AP-009
///
/// # Example
/// ```rust
/// use context_graph_core::marblestone::{NeurotransmitterWeights, Domain};
///
/// let weights = NeurotransmitterWeights::for_domain(Domain::Code);
/// let effective = weights.compute_effective_weight(0.8);
/// assert!(effective >= 0.0 && effective <= 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NeurotransmitterWeights {
    /// Excitatory signal strength [0.0, 1.0]. Higher = stronger connection.
    pub excitatory: f32,
    /// Inhibitory signal strength [0.0, 1.0]. Higher = weaker connection.
    pub inhibitory: f32,
    /// Modulatory signal strength [0.0, 1.0]. Context-dependent adjustment.
    pub modulatory: f32,
}

impl NeurotransmitterWeights {
    /// Create new weights with explicit values.
    ///
    /// # Arguments
    /// * `excitatory` - Strengthening signal [0.0, 1.0]
    /// * `inhibitory` - Weakening signal [0.0, 1.0]
    /// * `modulatory` - Domain-adjustment signal [0.0, 1.0]
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::marblestone::NeurotransmitterWeights;
    ///
    /// let weights = NeurotransmitterWeights::new(0.7, 0.2, 0.5);
    /// assert_eq!(weights.excitatory, 0.7);
    /// ```
    #[inline]
    pub fn new(excitatory: f32, inhibitory: f32, modulatory: f32) -> Self {
        Self {
            excitatory,
            inhibitory,
            modulatory,
        }
    }

    /// Get domain-specific neurotransmitter profile.
    ///
    /// Each domain has optimized NT weights for its retrieval characteristics:
    /// - **Code**: excitatory=0.6, inhibitory=0.3, modulatory=0.4 (precise)
    /// - **Legal**: excitatory=0.4, inhibitory=0.4, modulatory=0.2 (conservative)
    /// - **Medical**: excitatory=0.5, inhibitory=0.3, modulatory=0.5 (causal)
    /// - **Creative**: excitatory=0.8, inhibitory=0.1, modulatory=0.6 (exploratory)
    /// - **Research**: excitatory=0.6, inhibitory=0.2, modulatory=0.5 (balanced)
    /// - **General**: excitatory=0.5, inhibitory=0.2, modulatory=0.3 (default)
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::marblestone::{NeurotransmitterWeights, Domain};
    ///
    /// let creative = NeurotransmitterWeights::for_domain(Domain::Creative);
    /// assert_eq!(creative.excitatory, 0.8);
    /// assert_eq!(creative.inhibitory, 0.1);
    /// ```
    #[inline]
    pub fn for_domain(domain: Domain) -> Self {
        match domain {
            Domain::Code => Self::new(0.6, 0.3, 0.4),
            Domain::Legal => Self::new(0.4, 0.4, 0.2),
            Domain::Medical => Self::new(0.5, 0.3, 0.5),
            Domain::Creative => Self::new(0.8, 0.1, 0.6),
            Domain::Research => Self::new(0.6, 0.2, 0.5),
            Domain::General => Self::new(0.5, 0.2, 0.3),
        }
    }

    /// Compute effective weight given a base weight.
    ///
    /// # Formula
    /// ```text
    /// w_eff = ((base * excitatory - base * inhibitory) * (1 + (modulatory - 0.5) * 0.4)).clamp(0.0, 1.0)
    /// ```
    ///
    /// This applies:
    /// 1. Excitatory amplification: `base * excitatory`
    /// 2. Inhibitory dampening: `base * inhibitory`
    /// 3. Modulatory context adjustment: centered at 0.5, ±20% range
    /// 4. Final clamp to [0.0, 1.0] per AP-009
    ///
    /// # Arguments
    /// * `base_weight` - Original edge weight [0.0, 1.0]
    ///
    /// # Returns
    /// Effective weight always in [0.0, 1.0]
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::marblestone::{NeurotransmitterWeights, Domain};
    ///
    /// let weights = NeurotransmitterWeights::for_domain(Domain::General);
    /// let effective = weights.compute_effective_weight(1.0);
    /// // General: (1.0*0.5 - 1.0*0.2) * (1 + (0.3-0.5)*0.4) = 0.3 * 0.92 = 0.276
    /// assert!((effective - 0.276).abs() < 0.001);
    /// ```
    #[inline]
    pub fn compute_effective_weight(&self, base_weight: f32) -> f32 {
        // Step 1: Apply excitatory and inhibitory
        let signal = base_weight * self.excitatory - base_weight * self.inhibitory;
        // Step 2: Apply modulatory adjustment (centered at 0.5)
        let mod_factor = 1.0 + (self.modulatory - 0.5) * 0.4;
        // Step 3: Clamp to valid range per AP-009
        (signal * mod_factor).clamp(0.0, 1.0)
    }

    /// Validate that all weights are in valid range [0.0, 1.0].
    ///
    /// # Returns
    /// `true` if all weights are in [0.0, 1.0] and not NaN/Infinity
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::marblestone::NeurotransmitterWeights;
    ///
    /// let valid = NeurotransmitterWeights::new(0.5, 0.3, 0.4);
    /// assert!(valid.validate());
    ///
    /// let invalid = NeurotransmitterWeights::new(1.5, 0.0, 0.0);
    /// assert!(!invalid.validate());
    /// ```
    #[inline]
    pub fn validate(&self) -> bool {
        // Check for NaN/Infinity per AP-009
        if self.excitatory.is_nan() || self.excitatory.is_infinite() {
            return false;
        }
        if self.inhibitory.is_nan() || self.inhibitory.is_infinite() {
            return false;
        }
        if self.modulatory.is_nan() || self.modulatory.is_infinite() {
            return false;
        }
        // Check valid range [0.0, 1.0]
        self.excitatory >= 0.0
            && self.excitatory <= 1.0
            && self.inhibitory >= 0.0
            && self.inhibitory <= 1.0
            && self.modulatory >= 0.0
            && self.modulatory <= 1.0
    }
}

impl Default for NeurotransmitterWeights {
    /// Returns General domain profile: excitatory=0.5, inhibitory=0.2, modulatory=0.3
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::marblestone::NeurotransmitterWeights;
    ///
    /// let weights = NeurotransmitterWeights::default();
    /// assert_eq!(weights.excitatory, 0.5);
    /// assert_eq!(weights.inhibitory, 0.2);
    /// assert_eq!(weights.modulatory, 0.3);
    /// ```
    #[inline]
    fn default() -> Self {
        Self::for_domain(Domain::General)
    }
}
