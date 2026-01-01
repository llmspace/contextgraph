//! Stub implementation of the Learning layer.
//!
//! The Learning layer handles UTL-driven weight optimization.
//! This stub simulates learning computations.
//!
//! # Latency Budget
//! Real implementation: 10ms max
//! Stub implementation: <1us (instant return)

use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;

use crate::error::CoreResult;
use crate::traits::NervousLayer;
use crate::types::{CognitivePulse, LayerId, LayerInput, LayerOutput, LayerResult, SuggestedAction};

use super::helpers::{compute_input_hash, StubLayerConfig};

/// Stub implementation of the Learning layer.
#[derive(Debug, Clone, Default)]
pub struct StubLearningLayer {
    _config: StubLayerConfig,
}

impl StubLearningLayer {
    /// Create a new stub learning layer.
    pub fn new() -> Self {
        Self {
            _config: StubLayerConfig::default(),
        }
    }
}

#[async_trait]
impl NervousLayer for StubLearningLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        let input_hash = compute_input_hash(&input.content);

        // Learning layer can have higher entropy (exploring new patterns)
        let entropy = 0.4 + (input_hash % 50) as f32 / 200.0; // Range: 0.4-0.65
        let coherence = 0.55 + (input_hash % 45) as f32 / 200.0; // Range: 0.55-0.775

        // Deterministic UTL score simulation
        let utl_score = 0.6 + (input_hash % 40) as f32 / 100.0; // Range: 0.6-1.0 (capped)
        let utl_score = utl_score.min(1.0);

        let result = LayerResult::success(
            LayerId::Learning,
            json!({
                "utl_score": utl_score,
                "learning_applied": true,
                "weights_updated": (input_hash % 10) + 1,
                "gradient_norm": 0.05,
                "convergence_metric": 0.92
            }),
        );

        // Learning might suggest exploration if UTL score is high
        let action = if utl_score > 0.8 {
            SuggestedAction::Explore
        } else {
            SuggestedAction::Continue
        };

        Ok(LayerOutput {
            layer: LayerId::Learning,
            result,
            pulse: CognitivePulse::new(
                entropy,
                coherence,
                0.0,
                1.0,
                action,
                Some(LayerId::Learning),
            ),
            duration_us: 2000, // 2ms stub value, well under 10ms budget
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_millis(10)
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Learning
    }

    fn layer_name(&self) -> &'static str {
        "Learning Layer (Stub)"
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
    async fn test_learning_layer_output_within_budget() {
        let layer = StubLearningLayer::new();
        let input = test_input("learning optimization task");

        let output = layer.process(input).await.expect("process should succeed");

        let budget_us = layer.latency_budget().as_micros() as u64;
        assert!(
            output.duration_us < budget_us,
            "duration {}us should be < budget {}us",
            output.duration_us,
            budget_us
        );

        assert_eq!(output.layer, LayerId::Learning);
        assert!(output.result.success);
    }

    #[tokio::test]
    async fn test_learning_layer_determinism() {
        let layer = StubLearningLayer::new();
        let input1 = test_input("learning test");
        let input2 = test_input("learning test");

        let output1 = layer.process(input1).await.unwrap();
        let output2 = layer.process(input2).await.unwrap();

        assert_eq!(output1.pulse.entropy, output2.pulse.entropy);
        assert_eq!(output1.pulse.coherence, output2.pulse.coherence);
    }

    #[tokio::test]
    async fn test_learning_layer_health_check() {
        let layer = StubLearningLayer::new();
        assert!(layer.health_check().await.unwrap());
    }

    #[test]
    fn test_learning_layer_properties() {
        let layer = StubLearningLayer::new();
        assert_eq!(layer.layer_id(), LayerId::Learning);
        assert_eq!(layer.latency_budget(), Duration::from_millis(10));
        assert!(layer.layer_name().contains("Learning"));
    }
}
