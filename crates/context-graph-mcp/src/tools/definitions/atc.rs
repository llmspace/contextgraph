//! ATC (Adaptive Threshold Calibration) tool definitions.
//! TASK-ATC-001: Threshold status, calibration metrics, recalibration.

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns ATC tool definitions (3 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // get_threshold_status - Get current threshold status (TASK-ATC-001)
        ToolDefinition::new(
            "get_threshold_status",
            "Get current ATC threshold status including all thresholds, calibration state, \
             and adaptation metrics. Returns per-embedder temperatures, drift scores, and \
             bandit exploration stats. Requires ATC provider to be initialized.",
            json!({
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "enum": ["Code", "Medical", "Legal", "Creative", "Research", "General"],
                        "default": "General",
                        "description": "Domain for threshold context (affects priors)"
                    },
                    "embedder_id": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 13,
                        "description": "Optional: specific embedder (1-13) for detailed temperature info"
                    }
                },
                "required": []
            }),
        ),
        // get_calibration_metrics - Get calibration quality metrics (TASK-ATC-001)
        ToolDefinition::new(
            "get_calibration_metrics",
            "Get calibration quality metrics: ECE (Expected Calibration Error), \
             MCE (Maximum Calibration Error), Brier Score, drift scores per threshold, \
             and calibration status. Targets: ECE < 0.05 (excellent), < 0.10 (good).",
            json!({
                "type": "object",
                "properties": {
                    "timeframe": {
                        "type": "string",
                        "enum": ["1h", "24h", "7d", "30d"],
                        "default": "24h",
                        "description": "Timeframe for metrics aggregation"
                    }
                },
                "required": []
            }),
        ),
        // trigger_recalibration - Manually trigger recalibration (TASK-ATC-001)
        ToolDefinition::new(
            "trigger_recalibration",
            "Manually trigger recalibration at a specific ATC level. \
             Level 1: EWMA drift adjustment. Level 2: Temperature scaling. \
             Level 3: Thompson Sampling exploration. Level 4: Bayesian meta-optimization. \
             Returns new thresholds and number of observations used.",
            json!({
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4,
                        "description": "ATC level to trigger (1=EWMA, 2=Temperature, 3=Bandit, 4=Bayesian)"
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["Code", "Medical", "Legal", "Creative", "Research", "General"],
                        "default": "General",
                        "description": "Domain context for recalibration"
                    }
                },
                "required": ["level"]
            }),
        ),
    ]
}
