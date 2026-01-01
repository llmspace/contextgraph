//! Stub implementation of the Memory layer.
//!
//! The Memory layer handles Modern Hopfield associative storage.
//! This stub simulates memory retrieval operations.
//!
//! # Latency Budget
//! Real implementation: 1ms max
//! Stub implementation: <1us (instant return)

use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;

use crate::error::CoreResult;
use crate::traits::NervousLayer;
use crate::types::{CognitivePulse, LayerId, LayerInput, LayerOutput, LayerResult, SuggestedAction};

use super::helpers::{compute_input_hash, StubLayerConfig};

/// Stub implementation of the Memory layer.
#[derive(Debug, Clone, Default)]
pub struct StubMemoryLayer {
    _config: StubLayerConfig,
}

impl StubMemoryLayer {
    /// Create a new stub memory layer.
    pub fn new() -> Self {
        Self {
            _config: StubLayerConfig::default(),
        }
    }
}

#[async_trait]
impl NervousLayer for StubMemoryLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        let input_hash = compute_input_hash(&input.content);

        // Memory layer has moderate entropy (some uncertainty in retrieval)
        let entropy = 0.3 + (input_hash % 40) as f32 / 200.0; // Range: 0.3-0.5
        let coherence = 0.65 + (input_hash % 35) as f32 / 200.0; // Range: 0.65-0.825

        // Deterministic memory retrieval simulation
        let memories_found = (input_hash % 5) + 1; // 1-5 memories

        let result = LayerResult::success(
            LayerId::Memory,
            json!({
                "memories_retrieved": memories_found,
                "retrieval_scores": vec![0.95, 0.87, 0.76, 0.68, 0.55][..memories_found as usize].to_vec(),
                "hopfield_energy": -0.85,
                "cache_hit": input_hash % 3 == 0
            }),
        );

        Ok(LayerOutput {
            layer: LayerId::Memory,
            result,
            pulse: CognitivePulse::new(
                entropy,
                coherence,
                0.0,
                1.0,
                SuggestedAction::Continue,
                Some(LayerId::Memory),
            ),
            duration_us: 200, // 200us stub value, well under 1ms budget
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_millis(1)
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Memory
    }

    fn layer_name(&self) -> &'static str {
        "Memory Layer (Stub)"
    }

    async fn health_check(&self) -> CoreResult<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stubs::layers::helpers::test_input;

    #[tokio::test]
    async fn test_memory_layer_output_within_budget() {
        let layer = StubMemoryLayer::new();
        let input = test_input("memory retrieval query");

        let output = layer.process(input).await.expect("process should succeed");

        let budget_us = layer.latency_budget().as_micros() as u64;
        assert!(
            output.duration_us < budget_us,
            "duration {}us should be < budget {}us",
            output.duration_us,
            budget_us
        );

        assert_eq!(output.layer, LayerId::Memory);
        assert!(output.result.success);
    }

    #[tokio::test]
    async fn test_memory_layer_determinism() {
        let layer = StubMemoryLayer::new();
        let input1 = test_input("memory test");
        let input2 = test_input("memory test");

        let output1 = layer.process(input1).await.unwrap();
        let output2 = layer.process(input2).await.unwrap();

        assert_eq!(output1.pulse.entropy, output2.pulse.entropy);
        assert_eq!(output1.pulse.coherence, output2.pulse.coherence);
    }

    #[tokio::test]
    async fn test_memory_layer_health_check() {
        let layer = StubMemoryLayer::new();
        assert!(layer.health_check().await.unwrap());
    }

    #[test]
    fn test_memory_layer_properties() {
        let layer = StubMemoryLayer::new();
        assert_eq!(layer.layer_id(), LayerId::Memory);
        assert_eq!(layer.latency_budget(), Duration::from_millis(1));
        assert!(layer.layer_name().contains("Memory"));
    }
}
