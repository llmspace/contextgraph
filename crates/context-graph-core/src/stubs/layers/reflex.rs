//! Stub implementation of the Reflex layer.
//!
//! The Reflex layer provides pattern-matched fast responses.
//! This stub immediately returns a pattern match result.
//!
//! # Latency Budget
//! Real implementation: 100us max (very fast)
//! Stub implementation: <1us (instant return)

use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;

use crate::error::CoreResult;
use crate::traits::NervousLayer;
use crate::types::{CognitivePulse, LayerId, LayerInput, LayerOutput, LayerResult, SuggestedAction};

use super::helpers::{compute_input_hash, StubLayerConfig};

/// Stub implementation of the Reflex layer.
#[derive(Debug, Clone, Default)]
pub struct StubReflexLayer {
    _config: StubLayerConfig,
}

impl StubReflexLayer {
    /// Create a new stub reflex layer.
    pub fn new() -> Self {
        Self {
            _config: StubLayerConfig::default(),
        }
    }
}

#[async_trait]
impl NervousLayer for StubReflexLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        let input_hash = compute_input_hash(&input.content);

        // Reflex layer has very low latency, so entropy should be low (fast decisions)
        let entropy = 0.2 + (input_hash % 30) as f32 / 200.0; // Range: 0.2-0.35
        let coherence = 0.7 + (input_hash % 40) as f32 / 200.0; // Range: 0.7-0.9

        // Deterministic pattern match simulation
        let pattern_found = input.content.len() > 10;

        let result = LayerResult::success(
            LayerId::Reflex,
            json!({
                "pattern_matched": pattern_found,
                "match_confidence": if pattern_found { 0.85 } else { 0.0 },
                "reflex_triggered": pattern_found,
                "patterns_checked": 5
            }),
        );

        Ok(LayerOutput {
            layer: LayerId::Reflex,
            result,
            pulse: CognitivePulse::new(
                entropy,
                coherence,
                0.0,
                1.0,
                SuggestedAction::Continue,
                Some(LayerId::Reflex),
            ),
            duration_us: 50, // 50us stub value, well under 100us budget
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_micros(100)
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Reflex
    }

    fn layer_name(&self) -> &'static str {
        "Reflex Layer (Stub)"
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
    async fn test_reflex_layer_output_within_budget() {
        let layer = StubReflexLayer::new();
        let input = test_input("reflex pattern input");

        let output = layer.process(input).await.expect("process should succeed");

        let budget_us = layer.latency_budget().as_micros() as u64;
        assert!(
            output.duration_us < budget_us,
            "duration {}us should be < budget {}us",
            output.duration_us,
            budget_us
        );

        assert_eq!(output.layer, LayerId::Reflex);
        assert!(output.result.success);
    }

    #[tokio::test]
    async fn test_reflex_layer_determinism() {
        let layer = StubReflexLayer::new();
        let input1 = test_input("reflex test");
        let input2 = test_input("reflex test");

        let output1 = layer.process(input1).await.unwrap();
        let output2 = layer.process(input2).await.unwrap();

        assert_eq!(output1.pulse.entropy, output2.pulse.entropy);
        assert_eq!(output1.pulse.coherence, output2.pulse.coherence);
    }

    #[tokio::test]
    async fn test_reflex_layer_health_check() {
        let layer = StubReflexLayer::new();
        assert!(layer.health_check().await.unwrap());
    }

    #[test]
    fn test_reflex_layer_properties() {
        let layer = StubReflexLayer::new();
        assert_eq!(layer.layer_id(), LayerId::Reflex);
        assert_eq!(layer.latency_budget(), Duration::from_micros(100));
        assert!(layer.layer_name().contains("Reflex"));
    }
}
