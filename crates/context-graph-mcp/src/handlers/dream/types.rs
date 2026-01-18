//! Types for TriggerManager MCP tools.
//!
//! Per SPEC-TRIGGER-MCP-001 and TECH-TRIGGER-MCP-001.

use context_graph_core::dream::types::ExtendedTriggerReason;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// DTO for ExtendedTriggerReason for MCP serialization.
///
/// # Constitution Compliance (v6.0.0)
///
/// Per Constitution v6.0.0, dreams are triggered by entropy and churn conditions,
/// NOT by identity coherence (IC). The IdentityCritical variant has been removed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TriggerReasonDto {
    /// Entropy above threshold.
    HighEntropy,
    /// Entropy exactly at threshold.
    ThresholdMet,
    /// Entropy below threshold (no trigger).
    None,
    /// Manual trigger.
    Manual { phase: String },
    /// GPU overload.
    GpuOverload,
    /// Memory pressure.
    MemoryPressure,
    /// Scheduled.
    Scheduled,
    /// Idle timeout.
    IdleTimeout,
}

impl Default for TriggerReasonDto {
    fn default() -> Self {
        Self::None
    }
}

impl std::fmt::Display for TriggerReasonDto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HighEntropy => write!(f, "high_entropy"),
            Self::ThresholdMet => write!(f, "threshold_met"),
            Self::None => write!(f, "none"),
            Self::Manual { phase } => write!(f, "manual:{}", phase),
            Self::GpuOverload => write!(f, "gpu_overload"),
            Self::MemoryPressure => write!(f, "memory_pressure"),
            Self::Scheduled => write!(f, "scheduled"),
            Self::IdleTimeout => write!(f, "idle_timeout"),
        }
    }
}

/// Convert core ExtendedTriggerReason to DTO for MCP serialization.
///
/// # Constitution Compliance (v6.0.0)
///
/// Per Constitution v6.0.0, dreams are triggered by entropy and churn,
/// NOT identity coherence. IdentityCritical conversion removed.
impl From<ExtendedTriggerReason> for TriggerReasonDto {
    fn from(reason: ExtendedTriggerReason) -> Self {
        match reason {
            ExtendedTriggerReason::Manual { phase } => TriggerReasonDto::Manual {
                phase: format!("{}", phase),
            },
            ExtendedTriggerReason::IdleTimeout => TriggerReasonDto::IdleTimeout,
            ExtendedTriggerReason::HighEntropy => TriggerReasonDto::HighEntropy,
            ExtendedTriggerReason::GpuOverload => TriggerReasonDto::GpuOverload,
            ExtendedTriggerReason::MemoryPressure => TriggerReasonDto::MemoryPressure,
            ExtendedTriggerReason::Scheduled => TriggerReasonDto::Scheduled,
        }
    }
}

/// A single entry in the trigger history.
///
/// Records when and why a trigger fired for observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerHistoryEntry {
    /// When the trigger fired (ISO 8601 timestamp).
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// The entropy value that caused the trigger.
    /// Range: [0.0, 1.0]
    pub entropy_value: f32,

    /// The reason the trigger fired.
    pub trigger_reason: TriggerReasonDto,

    /// Whether a mental_check workflow was initiated.
    pub workflow_initiated: bool,

    /// Optional: workflow ID if initiated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workflow_id: Option<String>,
}

impl TriggerHistoryEntry {
    /// Create a new trigger history entry with current timestamp.
    pub fn new(
        entropy_value: f32,
        trigger_reason: TriggerReasonDto,
        workflow_initiated: bool,
        workflow_id: Option<String>,
    ) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            entropy_value,
            trigger_reason,
            workflow_initiated,
            workflow_id,
        }
    }

    /// Create a "no trigger" entry for entropy below threshold.
    pub fn no_trigger(entropy_value: f32) -> Self {
        Self::new(entropy_value, TriggerReasonDto::None, false, None)
    }
}

/// DTO for TriggerManager configuration (MCP response).
///
/// # Constitution Compliance (v6.0.0)
///
/// Per Constitution v6.0.0, dreams are triggered by entropy and churn,
/// NOT identity coherence (IC). The ic_threshold field has been removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerConfigDto {
    /// Current entropy threshold.
    pub entropy_threshold: f32,

    /// Current churn threshold (Constitution v6.0.0: churn > 0.5).
    pub churn_threshold: f32,

    /// Cooldown duration in milliseconds.
    pub cooldown_ms: u64,

    /// Last trigger timestamp (ISO 8601 or null).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_trigger_timestamp: Option<String>,

    /// Total triggers in current session.
    pub trigger_count: u32,

    /// Whether triggers are enabled.
    pub enabled: bool,
}

impl Default for TriggerConfigDto {
    fn default() -> Self {
        Self {
            entropy_threshold: 0.7,
            churn_threshold: 0.5, // Constitution v6.0.0: churn > 0.5
            cooldown_ms: 60_000,  // 1 minute
            last_trigger_timestamp: None,
            trigger_count: 0,
            enabled: true,
        }
    }
}

/// Errors from trigger MCP tools.
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
#[serde(tag = "code", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TriggerMcpError {
    /// Entropy value is NaN.
    #[error("E_TRIGGER_MCP_001: Entropy value is NaN")]
    #[serde(rename = "E_TRIGGER_MCP_001")]
    EntropyNaN { field: String, reason: String },

    /// Entropy value is negative.
    #[error("E_TRIGGER_MCP_002: Entropy value {value} is negative")]
    #[serde(rename = "E_TRIGGER_MCP_002")]
    EntropyNegative { value: f32, valid_range: [f32; 2] },

    /// Entropy value exceeds 1.0.
    #[error("E_TRIGGER_MCP_003: Entropy value {value} exceeds 1.0")]
    #[serde(rename = "E_TRIGGER_MCP_003")]
    EntropyOverRange { value: f32, valid_range: [f32; 2] },

    /// TriggerManager not initialized.
    #[error("E_TRIGGER_MCP_004: TriggerManager not initialized")]
    #[serde(rename = "E_TRIGGER_MCP_004")]
    NotInitialized { guidance: String },

    /// Rate limit exceeded.
    #[error("E_TRIGGER_MCP_005: Rate limit exceeded")]
    #[serde(rename = "E_TRIGGER_MCP_005")]
    RateLimitExceeded { retry_after_ms: u64 },

    /// mental_check workflow failed.
    #[error("E_TRIGGER_MCP_006: mental_check workflow failed: {details}")]
    #[serde(rename = "E_TRIGGER_MCP_006")]
    WorkflowFailed { details: String },
}

impl TriggerMcpError {
    /// Create entropy NaN error.
    pub fn entropy_nan() -> Self {
        Self::EntropyNaN {
            field: "entropy".to_string(),
            reason: "NaN not allowed".to_string(),
        }
    }

    /// Create entropy negative error.
    pub fn entropy_negative(value: f32) -> Self {
        Self::EntropyNegative {
            value,
            valid_range: [0.0, 1.0],
        }
    }

    /// Create entropy over-range error.
    pub fn entropy_over_range(value: f32) -> Self {
        Self::EntropyOverRange {
            value,
            valid_range: [0.0, 1.0],
        }
    }

    /// Create not initialized error.
    pub fn not_initialized() -> Self {
        Self::NotInitialized {
            guidance: "Call initialize_trigger_manager() first".to_string(),
        }
    }

    /// Create rate limit error.
    pub fn rate_limit(retry_after_ms: u64) -> Self {
        Self::RateLimitExceeded { retry_after_ms }
    }

    /// Create workflow failed error.
    pub fn workflow_failed(details: impl Into<String>) -> Self {
        Self::WorkflowFailed {
            details: details.into(),
        }
    }

    /// Get the error code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::EntropyNaN { .. } => "E_TRIGGER_MCP_001",
            Self::EntropyNegative { .. } => "E_TRIGGER_MCP_002",
            Self::EntropyOverRange { .. } => "E_TRIGGER_MCP_003",
            Self::NotInitialized { .. } => "E_TRIGGER_MCP_004",
            Self::RateLimitExceeded { .. } => "E_TRIGGER_MCP_005",
            Self::WorkflowFailed { .. } => "E_TRIGGER_MCP_006",
        }
    }
}

/// Status of the trigger operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerStatus {
    /// Trigger fired, workflow initiated.
    Initiated,
    /// Trigger fired, workflow queued (another in progress).
    Queued,
    /// Entropy below threshold, no trigger.
    Skipped,
}

impl std::fmt::Display for TriggerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initiated => write!(f, "initiated"),
            Self::Queued => write!(f, "queued"),
            Self::Skipped => write!(f, "skipped"),
        }
    }
}

/// Request for trigger_mental_check MCP tool.
///
/// TASK-S01: Per SPEC-TRIGGER-MCP-001 REQ-TRIGGER-01.
#[derive(Debug, Clone, Deserialize)]
pub struct TriggerMentalCheckRequest {
    /// Entropy value [0.0, 1.0].
    /// Triggers fire when entropy > threshold (default 0.7).
    pub entropy: f32,

    /// Force trigger even if entropy below threshold.
    #[serde(default)]
    pub force: bool,

    /// Optional phase to execute (nrem/rem/full_cycle).
    #[serde(default)]
    pub phase: Option<String>,
}

impl TriggerMentalCheckRequest {
    /// Validate entropy is in valid range [0.0, 1.0] and not NaN.
    pub fn validate(&self) -> Result<(), TriggerMcpError> {
        if self.entropy.is_nan() {
            return Err(TriggerMcpError::entropy_nan());
        }
        if self.entropy < 0.0 {
            return Err(TriggerMcpError::entropy_negative(self.entropy));
        }
        if self.entropy > 1.0 {
            return Err(TriggerMcpError::entropy_over_range(self.entropy));
        }
        Ok(())
    }
}

/// Response from trigger_mental_check MCP tool.
///
/// TASK-S01: Per SPEC-TRIGGER-MCP-001 REQ-TRIGGER-02.
#[derive(Debug, Clone, Serialize)]
pub struct TriggerMentalCheckResponse {
    /// Status of the trigger operation.
    pub status: TriggerStatus,

    /// Reason the trigger fired (or None if skipped).
    pub trigger_reason: TriggerReasonDto,

    /// Workflow ID if initiated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workflow_id: Option<String>,

    /// Human-readable message.
    pub message: String,

    /// Input entropy value.
    pub entropy: f32,

    /// Whether entropy exceeded the threshold.
    pub threshold_exceeded: bool,

    /// Current threshold value.
    pub threshold: f32,
}

impl TriggerMentalCheckResponse {
    /// Create a "skipped" response (entropy below threshold).
    pub fn skipped(entropy: f32, threshold: f32) -> Self {
        Self {
            status: TriggerStatus::Skipped,
            trigger_reason: TriggerReasonDto::None,
            workflow_id: None,
            message: format!(
                "Entropy {:.3} below threshold {:.3} - no trigger",
                entropy, threshold
            ),
            entropy,
            threshold_exceeded: false,
            threshold,
        }
    }

    /// Create an "initiated" response (trigger fired, workflow started).
    pub fn initiated(
        entropy: f32,
        threshold: f32,
        reason: TriggerReasonDto,
        workflow_id: Option<String>,
    ) -> Self {
        Self {
            status: TriggerStatus::Initiated,
            trigger_reason: reason,
            workflow_id,
            message: format!(
                "Trigger fired: entropy {:.3} exceeded threshold {:.3}",
                entropy, threshold
            ),
            entropy,
            threshold_exceeded: true,
            threshold,
        }
    }

    /// Create a "queued" response (trigger accepted but another workflow in progress).
    pub fn queued(entropy: f32, threshold: f32, reason: TriggerReasonDto) -> Self {
        Self {
            status: TriggerStatus::Queued,
            trigger_reason: reason,
            workflow_id: None,
            message: format!(
                "Trigger queued: entropy {:.3} exceeded threshold {:.3}, but another workflow is in progress",
                entropy, threshold
            ),
            entropy,
            threshold_exceeded: true,
            threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_reason_dto_display() {
        assert_eq!(format!("{}", TriggerReasonDto::HighEntropy), "high_entropy");
        assert_eq!(
            format!(
                "{}",
                TriggerReasonDto::Manual {
                    phase: "NREM".to_string()
                }
            ),
            "manual:NREM"
        );
        println!("[PASS] TriggerReasonDto::Display works correctly");
    }

    #[test]
    fn test_trigger_reason_dto_serde() {
        let reason = TriggerReasonDto::HighEntropy;
        let json = serde_json::to_string(&reason).unwrap();
        assert!(json.contains("high_entropy"));

        let parsed: TriggerReasonDto = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, reason);
        println!("[PASS] TriggerReasonDto serde roundtrip works");
    }

    #[test]
    fn test_trigger_history_entry_new() {
        let entry = TriggerHistoryEntry::new(
            0.85,
            TriggerReasonDto::HighEntropy,
            true,
            Some("wf-123".to_string()),
        );

        assert!((entry.entropy_value - 0.85).abs() < f32::EPSILON);
        assert!(entry.workflow_initiated);
        assert_eq!(entry.workflow_id, Some("wf-123".to_string()));
        println!("[PASS] TriggerHistoryEntry::new() works correctly");
    }

    #[test]
    fn test_trigger_history_entry_no_trigger() {
        let entry = TriggerHistoryEntry::no_trigger(0.5);

        assert!((entry.entropy_value - 0.5).abs() < f32::EPSILON);
        assert_eq!(entry.trigger_reason, TriggerReasonDto::None);
        assert!(!entry.workflow_initiated);
        assert!(entry.workflow_id.is_none());
        println!("[PASS] TriggerHistoryEntry::no_trigger() works correctly");
    }

    #[test]
    fn test_trigger_config_dto_default() {
        let config = TriggerConfigDto::default();

        // Constitution v6.0.0: entropy > 0.7 AND churn > 0.5
        assert!((config.entropy_threshold - 0.7).abs() < f32::EPSILON);
        assert!((config.churn_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.cooldown_ms, 60_000);
        assert!(config.enabled);
        println!("[PASS] TriggerConfigDto::default() has correct values (Constitution v6.0.0)");
    }

    #[test]
    fn test_trigger_mcp_error_codes() {
        assert_eq!(TriggerMcpError::entropy_nan().code(), "E_TRIGGER_MCP_001");
        assert_eq!(
            TriggerMcpError::entropy_negative(-0.1).code(),
            "E_TRIGGER_MCP_002"
        );
        assert_eq!(
            TriggerMcpError::entropy_over_range(1.5).code(),
            "E_TRIGGER_MCP_003"
        );
        assert_eq!(
            TriggerMcpError::not_initialized().code(),
            "E_TRIGGER_MCP_004"
        );
        assert_eq!(
            TriggerMcpError::rate_limit(5000).code(),
            "E_TRIGGER_MCP_005"
        );
        assert_eq!(
            TriggerMcpError::workflow_failed("test").code(),
            "E_TRIGGER_MCP_006"
        );
        println!("[PASS] TriggerMcpError codes are correct");
    }

    #[test]
    fn test_trigger_mcp_error_display() {
        let err = TriggerMcpError::entropy_negative(-0.5);
        let msg = format!("{}", err);
        assert!(msg.contains("E_TRIGGER_MCP_002"));
        assert!(msg.contains("-0.5"));
        println!("[PASS] TriggerMcpError::Display works correctly");
    }

    #[test]
    fn test_trigger_mcp_error_serde() {
        let err = TriggerMcpError::entropy_nan();
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("E_TRIGGER_MCP_001"));
        println!("[PASS] TriggerMcpError serde works correctly");
    }

    #[test]
    fn test_trigger_status_display() {
        assert_eq!(format!("{}", TriggerStatus::Initiated), "initiated");
        assert_eq!(format!("{}", TriggerStatus::Queued), "queued");
        assert_eq!(format!("{}", TriggerStatus::Skipped), "skipped");
        println!("[PASS] TriggerStatus::Display works correctly");
    }

    // ========== TASK-S01: TriggerMentalCheckRequest tests ==========

    #[test]
    fn test_trigger_mental_check_request_validate_valid() {
        let req = TriggerMentalCheckRequest {
            entropy: 0.5,
            force: false,
            phase: None,
        };
        assert!(req.validate().is_ok());

        let req_min = TriggerMentalCheckRequest {
            entropy: 0.0,
            force: false,
            phase: None,
        };
        assert!(req_min.validate().is_ok());

        let req_max = TriggerMentalCheckRequest {
            entropy: 1.0,
            force: false,
            phase: None,
        };
        assert!(req_max.validate().is_ok());
        println!("[PASS] TriggerMentalCheckRequest::validate() accepts valid entropy [0.0, 1.0]");
    }

    #[test]
    fn test_trigger_mental_check_request_validate_nan() {
        let req = TriggerMentalCheckRequest {
            entropy: f32::NAN,
            force: false,
            phase: None,
        };
        let err = req.validate().unwrap_err();
        assert_eq!(err.code(), "E_TRIGGER_MCP_001");
        println!("[PASS] TriggerMentalCheckRequest::validate() rejects NaN");
    }

    #[test]
    fn test_trigger_mental_check_request_validate_negative() {
        let req = TriggerMentalCheckRequest {
            entropy: -0.1,
            force: false,
            phase: None,
        };
        let err = req.validate().unwrap_err();
        assert_eq!(err.code(), "E_TRIGGER_MCP_002");
        println!("[PASS] TriggerMentalCheckRequest::validate() rejects negative entropy");
    }

    #[test]
    fn test_trigger_mental_check_request_validate_over_range() {
        let req = TriggerMentalCheckRequest {
            entropy: 1.5,
            force: false,
            phase: None,
        };
        let err = req.validate().unwrap_err();
        assert_eq!(err.code(), "E_TRIGGER_MCP_003");
        println!("[PASS] TriggerMentalCheckRequest::validate() rejects entropy > 1.0");
    }

    #[test]
    fn test_trigger_mental_check_request_deserialize() {
        let json = r#"{"entropy": 0.85, "force": true, "phase": "nrem"}"#;
        let req: TriggerMentalCheckRequest = serde_json::from_str(json).unwrap();
        assert!((req.entropy - 0.85).abs() < f32::EPSILON);
        assert!(req.force);
        assert_eq!(req.phase, Some("nrem".to_string()));
        println!("[PASS] TriggerMentalCheckRequest deserializes correctly");
    }

    #[test]
    fn test_trigger_mental_check_request_deserialize_minimal() {
        // Only entropy is required; force defaults to false, phase to None
        let json = r#"{"entropy": 0.7}"#;
        let req: TriggerMentalCheckRequest = serde_json::from_str(json).unwrap();
        assert!((req.entropy - 0.7).abs() < f32::EPSILON);
        assert!(!req.force);
        assert!(req.phase.is_none());
        println!("[PASS] TriggerMentalCheckRequest deserializes with defaults");
    }

    // ========== TASK-S01: TriggerMentalCheckResponse tests ==========

    #[test]
    fn test_trigger_mental_check_response_skipped() {
        let resp = TriggerMentalCheckResponse::skipped(0.5, 0.7);
        assert_eq!(resp.status, TriggerStatus::Skipped);
        assert_eq!(resp.trigger_reason, TriggerReasonDto::None);
        assert!(resp.workflow_id.is_none());
        assert!(!resp.threshold_exceeded);
        assert!((resp.entropy - 0.5).abs() < f32::EPSILON);
        assert!((resp.threshold - 0.7).abs() < f32::EPSILON);
        assert!(resp.message.contains("below threshold"));
        println!("[PASS] TriggerMentalCheckResponse::skipped() works correctly");
    }

    #[test]
    fn test_trigger_mental_check_response_initiated() {
        let resp = TriggerMentalCheckResponse::initiated(
            0.85,
            0.7,
            TriggerReasonDto::HighEntropy,
            Some("wf-abc123".to_string()),
        );
        assert_eq!(resp.status, TriggerStatus::Initiated);
        assert_eq!(resp.trigger_reason, TriggerReasonDto::HighEntropy);
        assert_eq!(resp.workflow_id, Some("wf-abc123".to_string()));
        assert!(resp.threshold_exceeded);
        assert!((resp.entropy - 0.85).abs() < f32::EPSILON);
        assert!((resp.threshold - 0.7).abs() < f32::EPSILON);
        assert!(resp.message.contains("exceeded threshold"));
        println!("[PASS] TriggerMentalCheckResponse::initiated() works correctly");
    }

    #[test]
    fn test_trigger_mental_check_response_queued() {
        let resp = TriggerMentalCheckResponse::queued(0.9, 0.7, TriggerReasonDto::HighEntropy);
        assert_eq!(resp.status, TriggerStatus::Queued);
        assert_eq!(resp.trigger_reason, TriggerReasonDto::HighEntropy);
        assert!(resp.workflow_id.is_none());
        assert!(resp.threshold_exceeded);
        assert!((resp.entropy - 0.9).abs() < f32::EPSILON);
        assert!((resp.threshold - 0.7).abs() < f32::EPSILON);
        assert!(resp.message.contains("queued"));
        assert!(resp.message.contains("another workflow"));
        println!("[PASS] TriggerMentalCheckResponse::queued() works correctly");
    }

    #[test]
    fn test_trigger_mental_check_response_serialize() {
        let resp = TriggerMentalCheckResponse::initiated(
            0.8,
            0.7,
            TriggerReasonDto::HighEntropy,
            None, // No workflow_id - should be omitted
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"status\":\"initiated\""));
        assert!(json.contains("\"threshold_exceeded\":true"));
        // workflow_id should be absent when None due to skip_serializing_if
        assert!(!json.contains("workflow_id"));
        println!("[PASS] TriggerMentalCheckResponse serializes correctly with skip_serializing_if");
    }
}
