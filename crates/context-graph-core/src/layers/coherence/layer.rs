//! L5 Coherence Layer - Per-space clustering coordination and Global Workspace broadcast.
//!
//! This layer integrates information from all previous layers using
//! per-space clustering coordination to achieve coherent states.
//!
//! # Constitution Compliance (v6.0.0)
//!
//! - Latency: <10ms (CRITICAL)
//! - Throughput: 100/s
//! - Components: Per-space clustering, GW broadcast, workspace update
//! - UTL: R(t) measurement (coherence)
//!
//! # Topic-Based Coherence Scoring
//!
//! Per Constitution v6.0.0 Section 14, topic-based coherence scoring is used:
//!
//! coherence_score = I(t) x R(t) x D(t)
//!
//! Where:
//! - I(t) = Information (from pulse entropy, normalized)
//! - R(t) = Resonance (coherence measure from clustering)
//! - D(t) = Differentiation (inversely related to coherence clustering)

use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::atc::{AdaptiveThresholdCalibration, Domain};
use crate::error::{CoreError, CoreResult};
use crate::traits::NervousLayer;
use crate::types::{LayerId, LayerInput, LayerOutput, LayerResult};

use super::constants::INTEGRATION_STEPS;
use super::thresholds::GwtThresholds;
use super::workspace::{CoherenceState, GlobalWorkspace};

/// L5 Coherence Layer - Per-space clustering coordination and Global Workspace broadcast.
///
/// This layer integrates information from all previous layers using
/// per-space clustering coordination to achieve coherent states.
///
/// # Constitution Compliance
///
/// - Latency: <10ms (CRITICAL)
/// - Throughput: 100/s
/// - Components: Per-space clustering, GW broadcast, workspace update
/// - UTL: R(t) measurement (coherence)
///
/// # Topic-Based Coherence Scoring
///
/// coherence_score = I(t) × R(t) × D(t)
///
/// Where:
/// - I(t) = Information (from pulse entropy, normalized)
/// - R(t) = Resonance (coherence measure from clustering)
/// - D(t) = Differentiation (inversely related to coherence clustering)
///
/// # Domain-Aware Thresholds
///
/// The layer now supports domain-aware thresholds via the ATC system.
/// Use [`with_atc`](Self::with_atc) to create a layer with domain-specific thresholds.
#[derive(Debug)]
pub struct CoherenceLayer {
    /// GWT thresholds (domain-aware)
    thresholds: GwtThresholds,
    /// Number of integration steps per process
    pub(crate) integration_steps: usize,
    /// Total processing time in microseconds
    total_processing_us: AtomicU64,
    /// Total invocation count
    invocation_count: AtomicU64,
    /// Global Workspace ignition count
    ignition_count: AtomicU64,
    /// Current coherence/resonance value
    current_resonance: std::sync::atomic::AtomicU32,
}

impl CoherenceLayer {
    /// Create a new CoherenceLayer with default configuration.
    ///
    /// Uses legacy General domain thresholds (gate=0.70, hypersync=0.95, fragmentation=0.50).
    /// For domain-aware behavior, use [`with_atc`](Self::with_atc) instead.
    pub fn new() -> Self {
        Self {
            thresholds: GwtThresholds::default_general(),
            integration_steps: INTEGRATION_STEPS,
            total_processing_us: AtomicU64::new(0),
            invocation_count: AtomicU64::new(0),
            ignition_count: AtomicU64::new(0),
            current_resonance: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Create with ATC-managed thresholds for a specific domain.
    ///
    /// Domain strictness affects thresholds:
    /// - Stricter domains (Medical, Code) have higher gates
    /// - Looser domains (Creative) have lower gates
    ///
    /// # Errors
    ///
    /// Returns error if ATC doesn't have the domain or thresholds are invalid.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use context_graph_core::atc::{AdaptiveThresholdCalibration, Domain};
    /// use context_graph_core::layers::CoherenceLayer;
    ///
    /// let atc = AdaptiveThresholdCalibration::new();
    /// let layer = CoherenceLayer::with_atc(&atc, Domain::Code)?;
    /// ```
    pub fn with_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> CoreResult<Self> {
        let thresholds = GwtThresholds::from_atc(atc, domain)?;
        Ok(Self {
            thresholds,
            integration_steps: INTEGRATION_STEPS,
            total_processing_us: AtomicU64::new(0),
            invocation_count: AtomicU64::new(0),
            ignition_count: AtomicU64::new(0),
            current_resonance: std::sync::atomic::AtomicU32::new(0),
        })
    }

    /// Create with explicit GwtThresholds.
    ///
    /// Use this when you need custom thresholds that don't come from ATC.
    ///
    /// # Errors
    ///
    /// Returns error if thresholds fail validation.
    pub fn with_thresholds(thresholds: GwtThresholds) -> CoreResult<Self> {
        if !thresholds.is_valid() {
            return Err(CoreError::ValidationError {
                field: "GwtThresholds".to_string(),
                message: format!(
                    "Invalid thresholds: gate={}, hypersync={}, fragmentation={}. \
                    Check ranges and monotonicity.",
                    thresholds.gate, thresholds.hypersync, thresholds.fragmentation
                ),
            });
        }
        Ok(Self {
            thresholds,
            integration_steps: INTEGRATION_STEPS,
            total_processing_us: AtomicU64::new(0),
            invocation_count: AtomicU64::new(0),
            ignition_count: AtomicU64::new(0),
            current_resonance: std::sync::atomic::AtomicU32::new(0),
        })
    }

    /// Create with custom GW threshold (gate).
    ///
    /// # Note
    ///
    /// This only sets the gate threshold. For full control over all thresholds,
    /// use [`with_thresholds`](Self::with_thresholds).
    pub fn with_gw_threshold(mut self, threshold: f32) -> Self {
        // Create new thresholds with custom gate, keeping hypersync and fragmentation
        let clamped = threshold.clamp(0.65, 0.95);
        self.thresholds = GwtThresholds {
            gate: clamped,
            hypersync: self.thresholds.hypersync,
            fragmentation: self.thresholds.fragmentation,
        };
        self
    }

    /// Create with custom integration steps.
    pub fn with_integration_steps(mut self, steps: usize) -> Self {
        self.integration_steps = steps.max(1);
        self
    }

    /// Get the current GWT thresholds.
    pub fn thresholds(&self) -> &GwtThresholds {
        &self.thresholds
    }

    /// Get the current GW threshold (gate).
    pub fn gw_threshold(&self) -> f32 {
        self.thresholds.gate
    }

    /// Get ignition count.
    pub fn ignition_count(&self) -> u64 {
        self.ignition_count.load(Ordering::Relaxed)
    }

    /// Get average processing time in microseconds.
    pub fn avg_processing_us(&self) -> f64 {
        let count = self.invocation_count.load(Ordering::Relaxed);
        let total = self.total_processing_us.load(Ordering::Relaxed);
        if count > 0 {
            total as f64 / count as f64
        } else {
            0.0
        }
    }

    /// Compute topic-based coherence score: C(t) = I(t) × R(t) × D(t)
    ///
    /// - I(t) = Information (normalized entropy)
    /// - R(t) = Resonance (coherence measure)
    /// - D(t) = Differentiation (diversity measure)
    pub(crate) fn compute_coherence_score(
        &self,
        info: f32,
        resonance: f32,
        differentiation: f32,
    ) -> f32 {
        // Validate inputs per AP-009
        if info.is_nan() || info.is_infinite() {
            return 0.0;
        }
        if resonance.is_nan() || resonance.is_infinite() {
            return 0.0;
        }
        if differentiation.is_nan() || differentiation.is_infinite() {
            return 0.0;
        }

        // C(t) = I(t) × R(t) × D(t)
        let c = info * resonance * differentiation;
        c.clamp(0.0, 1.0)
    }

    /// Extract learning signal from L4 layer results.
    fn extract_learning_signal(&self, input: &LayerInput) -> f32 {
        input
            .context
            .layer_results
            .iter()
            .find(|r| r.layer == LayerId::Learning)
            .and_then(|r| r.data.get("weight_delta"))
            .and_then(|v| v.as_f64())
            .map(|v| (v as f32).clamp(-1.0, 1.0))
            .unwrap_or(0.0)
    }

    /// Compute differentiation D(t) as inverse of coherence clustering.
    ///
    /// Higher differentiation = more diverse/spread out information.
    fn compute_differentiation(&self, pulse: &crate::types::CognitivePulse) -> f32 {
        // D(t) measures how differentiated the information is
        // High coherence = low differentiation (clustered)
        // Low coherence = high differentiation (diverse)
        let base_differentiation = 1.0 - pulse.coherence.abs();

        // Add entropy influence - high entropy increases differentiation
        let entropy_factor = pulse.entropy * 0.3;

        (base_differentiation + entropy_factor).clamp(0.0, 1.0)
    }
}

impl Default for CoherenceLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NervousLayer for CoherenceLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        let start = Instant::now();

        // Extract learning signal from L4 for context
        let learning_signal = self.extract_learning_signal(&input);

        // Compute resonance from input pulse coherence and learning signal
        // Resonance is derived from per-space clustering coordination
        let base_resonance = input.context.pulse.coherence.abs().clamp(0.0, 1.0);

        // Learning signal modulates resonance slightly
        let resonance = (base_resonance + learning_signal * 0.1).clamp(0.0, 1.0);

        // Validate resonance - NO silent failures per AP-009
        if resonance.is_nan() || resonance.is_infinite() {
            return Err(CoreError::LayerError {
                layer: "Coherence".to_string(),
                message: "Resonance computation produced NaN/Infinity".to_string(),
            });
        }

        // Store current resonance for health checks
        self.current_resonance.store(
            resonance.to_bits(),
            Ordering::Relaxed
        );

        // Get information I(t) from pulse entropy (normalized)
        let info = input.context.pulse.entropy.clamp(0.01, 1.0);

        // Compute differentiation D(t)
        let differentiation = self.compute_differentiation(&input.context.pulse);

        // Compute coherence score C(t) = I(t) × R(t) × D(t)
        let coherence_score = self.compute_coherence_score(info, resonance, differentiation);

        // Determine coherence state from resonance using domain-aware thresholds
        let state =
            CoherenceState::from_order_parameter_with_thresholds(resonance, &self.thresholds);

        // Check for Global Workspace ignition (using gate threshold)
        let gw_ignited = resonance >= self.thresholds.gate;

        if gw_ignited {
            self.ignition_count.fetch_add(1, Ordering::Relaxed);
        }

        // Prepare broadcast content if ignited
        let broadcast = if gw_ignited {
            Some(serde_json::json!({
                "source_layers": ["sensing", "reflex", "memory", "learning"],
                "resonance": resonance,
                "coherence_score": coherence_score,
                "state": format!("{:?}", state),
            }))
        } else {
            None
        };

        // Build Global Workspace state (included in result data)
        let _workspace = GlobalWorkspace {
            active: gw_ignited,
            ignition_level: resonance,
            broadcast_content: broadcast.clone(),
            state,
        };

        let duration = start.elapsed();
        let duration_us = duration.as_micros() as u64;

        // Record metrics
        self.total_processing_us
            .fetch_add(duration_us, Ordering::Relaxed);
        self.invocation_count.fetch_add(1, Ordering::Relaxed);

        // Check latency budget
        let budget = self.latency_budget();
        if duration > budget {
            tracing::warn!(
                "CoherenceLayer exceeded latency budget: {:?} > {:?}",
                duration,
                budget
            );
        }

        // Update pulse with coherence metrics
        let mut updated_pulse = input.context.pulse.clone();
        // Set coherence to resonance (R(t) is the coherence measure)
        updated_pulse.coherence = resonance;
        // coherence_delta reflects change toward synchronized state
        updated_pulse.coherence_delta = resonance - input.context.pulse.coherence;
        // Update source layer
        updated_pulse.source_layer = Some(LayerId::Coherence);

        // Build result data
        let result_data = serde_json::json!({
            "resonance": resonance,
            "coherence_score": coherence_score,
            "differentiation": differentiation,
            "information": info,
            "gw_ignited": gw_ignited,
            "gw_threshold": self.gw_threshold(),
            "state": format!("{:?}", state),
            "broadcast": broadcast,
            "learning_signal": learning_signal,
            "duration_us": duration_us,
            "within_budget": duration <= budget,
        });

        Ok(LayerOutput {
            layer: LayerId::Coherence,
            result: LayerResult::success(LayerId::Coherence, result_data),
            pulse: updated_pulse,
            duration_us,
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_millis(10) // 10ms budget per constitution
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Coherence
    }

    fn layer_name(&self) -> &'static str {
        "Coherence Layer"
    }

    async fn health_check(&self) -> CoreResult<bool> {
        // Verify stored resonance is valid
        let r_bits = self.current_resonance.load(Ordering::Relaxed);
        let r = f32::from_bits(r_bits);

        // Allow uninitialized state (0.0 is valid)
        if r.is_nan() || r.is_infinite() {
            return Ok(false);
        }
        if !(0.0..=1.0).contains(&r) {
            return Ok(false);
        }

        // Verify coherence score computation works
        let c = self.compute_coherence_score(0.5, 0.5, 0.5);
        if c.is_nan() || c.is_infinite() {
            return Ok(false);
        }

        Ok(true)
    }
}
