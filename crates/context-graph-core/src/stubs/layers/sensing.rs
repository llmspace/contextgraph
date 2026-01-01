//! Stub implementation of the Sensing layer.
//!
//! The Sensing layer handles multi-modal input processing.
//! This stub immediately returns a processed result with sensible defaults.
//!
//! # Latency Budget
//! Real implementation: 5ms max
//! Stub implementation: <1us (instant return)

use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;

use crate::error::CoreResult;
use crate::traits::NervousLayer;
use crate::types::{CognitivePulse, LayerId, LayerInput, LayerOutput, LayerResult, SuggestedAction};

use super::helpers::{compute_input_hash, StubLayerConfig};

/// Stub implementation of the Sensing layer.
#[derive(Debug, Clone, Default)]
pub struct StubSensingLayer {
    /// Configuration flag (unused in stub, for future compatibility)
    _config: StubLayerConfig,
}

impl StubSensingLayer {
    /// Create a new stub sensing layer.
    pub fn new() -> Self {
        Self {
            _config: StubLayerConfig::default(),
        }
    }
}

#[async_trait]
impl NervousLayer for StubSensingLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        // Compute deterministic hash from input for reproducibility
        let input_hash = compute_input_hash(&input.content);

        // Generate deterministic entropy/coherence based on input
        let entropy = (input_hash % 100) as f32 / 200.0 + 0.2; // Range: 0.2-0.7
        let coherence = 0.6 + (input_hash % 50) as f32 / 200.0; // Range: 0.6-0.85

        let result = LayerResult::success(
            LayerId::Sensing,
            json!({
                "input_processed": true,
                "content_length": input.content.len(),
                "request_id": input.request_id,
                "modality": "text",
                "tokenized": true
            }),
        );

        Ok(LayerOutput {
            layer: LayerId::Sensing,
            result,
            pulse: CognitivePulse::new(
                entropy,
                coherence,
                0.0,
                1.0,
                SuggestedAction::Continue,
                Some(LayerId::Sensing),
            ),
            duration_us: 500, // 500us stub value, well under 5ms budget
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_millis(5)
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Sensing
    }

    fn layer_name(&self) -> &'static str {
        "Sensing Layer (Stub)"
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
    async fn test_sensing_layer_output_within_budget() {
        let layer = StubSensingLayer::new();
        let input = test_input("test sensing input");

        let output = layer.process(input).await.expect("process should succeed");

        let budget_us = layer.latency_budget().as_micros() as u64;
        assert!(
            output.duration_us < budget_us,
            "duration {}us should be < budget {}us",
            output.duration_us,
            budget_us
        );

        assert_eq!(output.layer, LayerId::Sensing);
        assert!(output.result.success);
    }

    #[tokio::test]
    async fn test_sensing_layer_determinism() {
        let layer = StubSensingLayer::new();
        let input1 = test_input("same input");
        let input2 = test_input("same input");

        let output1 = layer.process(input1).await.unwrap();
        let output2 = layer.process(input2).await.unwrap();

        assert_eq!(output1.pulse.entropy, output2.pulse.entropy);
        assert_eq!(output1.pulse.coherence, output2.pulse.coherence);
    }

    #[tokio::test]
    async fn test_sensing_layer_health_check() {
        let layer = StubSensingLayer::new();
        assert!(layer.health_check().await.unwrap());
    }

    #[test]
    fn test_sensing_layer_properties() {
        let layer = StubSensingLayer::new();
        assert_eq!(layer.layer_id(), LayerId::Sensing);
        assert_eq!(layer.latency_budget(), Duration::from_millis(5));
        assert!(layer.layer_name().contains("Sensing"));
    }
}
