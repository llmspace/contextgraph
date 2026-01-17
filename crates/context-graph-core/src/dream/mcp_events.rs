//! MCP Event Integration for Dream Layer
//!
//! Defines MCP events for broadcasting dream state to the GWT workspace.
//! These events enable other subsystems to react to dream cycles.
//!
//! ## Event Categories
//!
//! 1. **Lifecycle Events**: DreamCycleStarted, DreamCycleCompleted
//! 2. **Phase Events**: NremPhaseCompleted, RemPhaseCompleted
//! 3. **Discovery Events**: BlindSpotDiscovered, ShortcutCreated
//! 4. **Resource Events**: GpuBudgetWarning, WakeTriggered

use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[cfg(test)]
use super::types::DreamPhase;
use super::types::ExtendedTriggerReason;
use super::WakeReason;

/// Base trait for dream events.
pub trait DreamEvent: Serialize + Clone + Send + Sync {
    /// Event type identifier for routing.
    fn event_type(&self) -> &'static str;

    /// Session ID for correlation.
    fn session_id(&self) -> Uuid;

    /// Timestamp in milliseconds since epoch.
    fn timestamp_ms(&self) -> u64;
}

/// Dream cycle started event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamCycleStarted {
    pub session_id: Uuid,
    pub trigger_reason: String,
    pub timestamp_ms: u64,
    pub expected_nrem_duration_ms: u64,
    pub expected_rem_duration_ms: u64,
}

impl DreamEvent for DreamCycleStarted {
    fn event_type(&self) -> &'static str {
        "dream_cycle_started"
    }
    fn session_id(&self) -> Uuid {
        self.session_id
    }
    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

impl DreamCycleStarted {
    pub fn new(trigger_reason: ExtendedTriggerReason) -> Self {
        Self {
            session_id: Uuid::new_v4(),
            trigger_reason: trigger_reason.to_string(),
            timestamp_ms: current_timestamp_ms(),
            expected_nrem_duration_ms: 180_000, // 3 min per Constitution
            expected_rem_duration_ms: 120_000,  // 2 min per Constitution
        }
    }

    pub fn with_session_id(session_id: Uuid, trigger_reason: ExtendedTriggerReason) -> Self {
        Self {
            session_id,
            trigger_reason: trigger_reason.to_string(),
            timestamp_ms: current_timestamp_ms(),
            expected_nrem_duration_ms: 180_000,
            expected_rem_duration_ms: 120_000,
        }
    }
}

/// NREM phase completed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NremPhaseCompleted {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub memories_replayed: usize,
    pub edges_strengthened: usize,
    pub edges_pruned: usize,
    pub duration_ms: u64,
    pub compression_ratio: f32,
}

impl DreamEvent for NremPhaseCompleted {
    fn event_type(&self) -> &'static str {
        "nrem_phase_completed"
    }
    fn session_id(&self) -> Uuid {
        self.session_id
    }
    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

impl NremPhaseCompleted {
    pub fn new(
        session_id: Uuid,
        memories_replayed: usize,
        edges_strengthened: usize,
        edges_pruned: usize,
        duration: Duration,
    ) -> Self {
        Self {
            session_id,
            timestamp_ms: current_timestamp_ms(),
            memories_replayed,
            edges_strengthened,
            edges_pruned,
            duration_ms: duration.as_millis() as u64,
            compression_ratio: if memories_replayed > 0 {
                edges_pruned as f32 / memories_replayed as f32
            } else {
                0.0
            },
        }
    }
}

/// REM phase completed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemPhaseCompleted {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub queries_generated: usize,
    pub blind_spots_found: usize,
    pub walk_distance: f32,
    pub duration_ms: u64,
    pub average_semantic_leap: f32,
}

impl DreamEvent for RemPhaseCompleted {
    fn event_type(&self) -> &'static str {
        "rem_phase_completed"
    }
    fn session_id(&self) -> Uuid {
        self.session_id
    }
    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

impl RemPhaseCompleted {
    pub fn new(
        session_id: Uuid,
        queries_generated: usize,
        blind_spots_found: usize,
        walk_distance: f32,
        duration: Duration,
        average_semantic_leap: f32,
    ) -> Self {
        Self {
            session_id,
            timestamp_ms: current_timestamp_ms(),
            queries_generated,
            blind_spots_found,
            walk_distance,
            duration_ms: duration.as_millis() as u64,
            average_semantic_leap,
        }
    }
}

/// Dream cycle completed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamCycleCompleted {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub completed: bool,
    pub wake_reason: String,
    pub shortcuts_created: usize,
    pub total_duration_ms: u64,
    pub wake_latency_ms: u64,
}

impl DreamEvent for DreamCycleCompleted {
    fn event_type(&self) -> &'static str {
        "dream_cycle_completed"
    }
    fn session_id(&self) -> Uuid {
        self.session_id
    }
    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

impl DreamCycleCompleted {
    pub fn new(
        session_id: Uuid,
        completed: bool,
        wake_reason: WakeReason,
        shortcuts_created: usize,
        total_duration: Duration,
        wake_latency: Duration,
    ) -> Self {
        Self {
            session_id,
            timestamp_ms: current_timestamp_ms(),
            completed,
            wake_reason: wake_reason.to_string(),
            shortcuts_created,
            total_duration_ms: total_duration.as_millis() as u64,
            wake_latency_ms: wake_latency.as_millis() as u64,
        }
    }
}

/// Blind spot discovered event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindSpotDiscovered {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub blind_spot_id: Uuid,
    pub poincare_position: Vec<f32>, // Variable length to avoid serde issues with [f32; 64]
    pub semantic_distance: f32,
    pub confidence: f32,
    pub discovery_step: usize,
}

impl DreamEvent for BlindSpotDiscovered {
    fn event_type(&self) -> &'static str {
        "blind_spot_discovered"
    }
    fn session_id(&self) -> Uuid {
        self.session_id
    }
    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

impl BlindSpotDiscovered {
    pub fn new(
        session_id: Uuid,
        poincare_position: Vec<f32>,
        semantic_distance: f32,
        confidence: f32,
        discovery_step: usize,
    ) -> Self {
        Self {
            session_id,
            timestamp_ms: current_timestamp_ms(),
            blind_spot_id: Uuid::new_v4(),
            poincare_position,
            semantic_distance,
            confidence,
            discovery_step,
        }
    }
}

/// Shortcut edge created event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortcutCreated {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub hop_count: usize,
    pub combined_weight: f32,
    pub traversal_count: usize,
}

impl DreamEvent for ShortcutCreated {
    fn event_type(&self) -> &'static str {
        "shortcut_created"
    }
    fn session_id(&self) -> Uuid {
        self.session_id
    }
    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

impl ShortcutCreated {
    pub fn new(
        session_id: Uuid,
        source_id: Uuid,
        target_id: Uuid,
        hop_count: usize,
        combined_weight: f32,
        traversal_count: usize,
    ) -> Self {
        Self {
            session_id,
            timestamp_ms: current_timestamp_ms(),
            source_id,
            target_id,
            hop_count,
            combined_weight,
            traversal_count,
        }
    }
}

/// GPU budget warning event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBudgetWarning {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub current_usage: f32,
    pub budget: f32,
    pub action_taken: String,
}

impl DreamEvent for GpuBudgetWarning {
    fn event_type(&self) -> &'static str {
        "gpu_budget_warning"
    }
    fn session_id(&self) -> Uuid {
        self.session_id
    }
    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

impl GpuBudgetWarning {
    pub fn new(session_id: Uuid, current_usage: f32, budget: f32, action_taken: &str) -> Self {
        Self {
            session_id,
            timestamp_ms: current_timestamp_ms(),
            current_usage,
            budget,
            action_taken: action_taken.to_string(),
        }
    }
}

/// Wake triggered event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeTriggered {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub reason: String,
    pub phase: String,
    pub latency_ms: u64,
}

impl DreamEvent for WakeTriggered {
    fn event_type(&self) -> &'static str {
        "wake_triggered"
    }
    fn session_id(&self) -> Uuid {
        self.session_id
    }
    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

impl WakeTriggered {
    pub fn new(session_id: Uuid, reason: WakeReason, phase: &str, latency: Duration) -> Self {
        Self {
            session_id,
            timestamp_ms: current_timestamp_ms(),
            reason: reason.to_string(),
            phase: phase.to_string(),
            latency_ms: latency.as_millis() as u64,
        }
    }
}

/// MCP event broadcaster interface.
///
/// Implementations should connect to actual MCP transport.
pub trait DreamEventBroadcaster: Send + Sync {
    /// Broadcast a dream event.
    fn broadcast<E: DreamEvent>(&self, event: &E) -> Result<(), BroadcastError>;

    /// Check if broadcaster is connected.
    fn is_connected(&self) -> bool;
}

/// Broadcast error type.
#[derive(Debug, thiserror::Error)]
pub enum BroadcastError {
    #[error("Not connected to MCP")]
    NotConnected,

    #[error("Serialization failed: {0}")]
    SerializationFailed(String),

    #[error("Transport error: {0}")]
    TransportError(String),
}

/// No-op broadcaster for testing.
#[derive(Debug, Default)]
pub struct NoOpBroadcaster;

impl DreamEventBroadcaster for NoOpBroadcaster {
    fn broadcast<E: DreamEvent>(&self, _event: &E) -> Result<(), BroadcastError> {
        Ok(())
    }

    fn is_connected(&self) -> bool {
        false
    }
}

/// Logging broadcaster for development.
#[derive(Debug, Default)]
pub struct LoggingBroadcaster;

impl DreamEventBroadcaster for LoggingBroadcaster {
    fn broadcast<E: DreamEvent>(&self, event: &E) -> Result<(), BroadcastError> {
        let json = serde_json::to_string(event)
            .map_err(|e| BroadcastError::SerializationFailed(e.to_string()))?;
        tracing::info!(target: "mcp_events", event_type = %event.event_type(), "{}", json);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        true
    }
}

/// Get current timestamp in milliseconds.
fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dream_cycle_started() {
        let event = DreamCycleStarted::new(ExtendedTriggerReason::HighEntropy);

        assert_eq!(event.event_type(), "dream_cycle_started");
        assert_eq!(event.trigger_reason, "high_entropy");
        assert!(event.timestamp_ms > 0);
        assert!(!event.session_id.is_nil());
    }

    #[test]
    fn test_dream_cycle_started_with_session_id() {
        let session_id = Uuid::new_v4();
        let event = DreamCycleStarted::with_session_id(
            session_id,
            ExtendedTriggerReason::Manual {
                phase: DreamPhase::FullCycle,
            },
        );

        assert_eq!(event.session_id, session_id);
        assert_eq!(event.trigger_reason, "manual");
        assert_eq!(event.expected_nrem_duration_ms, 180_000);
        assert_eq!(event.expected_rem_duration_ms, 120_000);
    }

    #[test]
    fn test_dream_cycle_completed() {
        let session_id = Uuid::new_v4();
        let event = DreamCycleCompleted::new(
            session_id,
            true,
            WakeReason::CycleComplete,
            5,
            Duration::from_secs(300),
            Duration::from_millis(50),
        );

        assert_eq!(event.event_type(), "dream_cycle_completed");
        assert!(event.completed);
        assert_eq!(event.shortcuts_created, 5);
        assert_eq!(event.total_duration_ms, 300_000);
        assert_eq!(event.wake_latency_ms, 50);
    }

    #[test]
    fn test_nrem_phase_completed() {
        let session_id = Uuid::new_v4();
        let event = NremPhaseCompleted::new(
            session_id,
            100, // memories replayed
            75,  // edges strengthened
            10,  // edges pruned
            Duration::from_secs(180),
        );

        assert_eq!(event.event_type(), "nrem_phase_completed");
        assert_eq!(event.memories_replayed, 100);
        assert_eq!(event.edges_strengthened, 75);
        assert_eq!(event.edges_pruned, 10);
        assert_eq!(event.duration_ms, 180_000);
        assert_eq!(event.compression_ratio, 0.1); // 10/100
    }

    #[test]
    fn test_nrem_phase_completed_zero_memories() {
        let session_id = Uuid::new_v4();
        let event = NremPhaseCompleted::new(
            session_id,
            0, // zero memories
            0,
            0,
            Duration::from_secs(180),
        );

        assert_eq!(
            event.compression_ratio, 0.0,
            "Division by zero should return 0.0"
        );
    }

    #[test]
    fn test_rem_phase_completed() {
        let session_id = Uuid::new_v4();
        let event = RemPhaseCompleted::new(
            session_id,
            50,  // queries generated
            3,   // blind spots found
            0.8, // walk distance
            Duration::from_secs(120),
            0.75, // average semantic leap
        );

        assert_eq!(event.event_type(), "rem_phase_completed");
        assert_eq!(event.queries_generated, 50);
        assert_eq!(event.blind_spots_found, 3);
        assert_eq!(event.duration_ms, 120_000);
        assert_eq!(event.average_semantic_leap, 0.75);
    }

    #[test]
    fn test_blind_spot_discovered() {
        let session_id = Uuid::new_v4();
        let position = vec![0.1, 0.2, 0.3];
        let event = BlindSpotDiscovered::new(session_id, position.clone(), 0.85, 0.9, 42);

        assert_eq!(event.event_type(), "blind_spot_discovered");
        assert_eq!(event.session_id, session_id);
        assert_eq!(event.poincare_position, position);
        assert_eq!(event.semantic_distance, 0.85);
        assert_eq!(event.confidence, 0.9);
        assert_eq!(event.discovery_step, 42);
        assert!(!event.blind_spot_id.is_nil());
    }

    #[test]
    fn test_shortcut_created() {
        let session_id = Uuid::new_v4();
        let source_id = Uuid::new_v4();
        let target_id = Uuid::new_v4();
        let event = ShortcutCreated::new(
            session_id, source_id, target_id, 4,    // hop count
            0.85, // combined weight
            7,    // traversal count
        );

        assert_eq!(event.event_type(), "shortcut_created");
        assert_eq!(event.source_id, source_id);
        assert_eq!(event.target_id, target_id);
        assert_eq!(event.hop_count, 4);
        assert_eq!(event.combined_weight, 0.85);
        assert_eq!(event.traversal_count, 7);
    }

    #[test]
    fn test_gpu_budget_warning() {
        let session_id = Uuid::new_v4();
        let event = GpuBudgetWarning::new(session_id, 0.35, 0.30, "Initiating wake sequence");

        assert_eq!(event.event_type(), "gpu_budget_warning");
        assert_eq!(event.current_usage, 0.35);
        assert_eq!(event.budget, 0.30);
        assert_eq!(event.action_taken, "Initiating wake sequence");
    }

    #[test]
    fn test_wake_triggered() {
        let session_id = Uuid::new_v4();
        let event = WakeTriggered::new(
            session_id,
            WakeReason::ExternalQuery,
            "nrem",
            Duration::from_millis(45),
        );

        assert_eq!(event.event_type(), "wake_triggered");
        assert_eq!(event.reason, "external_query");
        assert_eq!(event.phase, "nrem");
        assert_eq!(event.latency_ms, 45);
    }

    #[test]
    fn test_logging_broadcaster() {
        let broadcaster = LoggingBroadcaster;
        let event = DreamCycleStarted::new(ExtendedTriggerReason::Manual {
            phase: DreamPhase::FullCycle,
        });

        assert!(broadcaster.is_connected());
        broadcaster.broadcast(&event).unwrap();
    }

    #[test]
    fn test_noop_broadcaster() {
        let broadcaster = NoOpBroadcaster;
        let event = DreamCycleStarted::new(ExtendedTriggerReason::IdleTimeout);

        assert!(!broadcaster.is_connected());
        broadcaster.broadcast(&event).unwrap();
    }

    #[test]
    fn test_events_serialize_to_json() {
        let session_id = Uuid::new_v4();

        // Test DreamCycleStarted
        let event1 = DreamCycleStarted::new(ExtendedTriggerReason::Manual {
            phase: DreamPhase::FullCycle,
        });
        let json1 = serde_json::to_string(&event1);
        assert!(
            json1.is_ok(),
            "DreamCycleStarted serialization failed: {:?}",
            json1.err()
        );

        // Test DreamCycleCompleted
        let event2 = DreamCycleCompleted::new(
            session_id,
            true,
            WakeReason::CycleComplete,
            0,
            Duration::ZERO,
            Duration::ZERO,
        );
        let json2 = serde_json::to_string(&event2);
        assert!(
            json2.is_ok(),
            "DreamCycleCompleted serialization failed: {:?}",
            json2.err()
        );

        // Test NremPhaseCompleted
        let event3 = NremPhaseCompleted::new(session_id, 0, 0, 0, Duration::ZERO);
        let json3 = serde_json::to_string(&event3);
        assert!(
            json3.is_ok(),
            "NremPhaseCompleted serialization failed: {:?}",
            json3.err()
        );

        // Test RemPhaseCompleted
        let event4 = RemPhaseCompleted::new(session_id, 0, 0, 0.0, Duration::ZERO, 0.0);
        let json4 = serde_json::to_string(&event4);
        assert!(
            json4.is_ok(),
            "RemPhaseCompleted serialization failed: {:?}",
            json4.err()
        );

        // Test WakeTriggered
        let event5 = WakeTriggered::new(
            session_id,
            WakeReason::ExternalQuery,
            "test",
            Duration::from_millis(50),
        );
        let json5 = serde_json::to_string(&event5);
        assert!(
            json5.is_ok(),
            "WakeTriggered serialization failed: {:?}",
            json5.err()
        );

        // Test BlindSpotDiscovered
        let event6 = BlindSpotDiscovered::new(session_id, vec![0.1, 0.2], 0.8, 0.9, 1);
        let json6 = serde_json::to_string(&event6);
        assert!(
            json6.is_ok(),
            "BlindSpotDiscovered serialization failed: {:?}",
            json6.err()
        );

        // Test ShortcutCreated
        let event7 = ShortcutCreated::new(session_id, Uuid::new_v4(), Uuid::new_v4(), 3, 0.8, 5);
        let json7 = serde_json::to_string(&event7);
        assert!(
            json7.is_ok(),
            "ShortcutCreated serialization failed: {:?}",
            json7.err()
        );

        // Test GpuBudgetWarning
        let event8 = GpuBudgetWarning::new(session_id, 0.35, 0.30, "test");
        let json8 = serde_json::to_string(&event8);
        assert!(
            json8.is_ok(),
            "GpuBudgetWarning serialization failed: {:?}",
            json8.err()
        );
    }

    #[test]
    fn test_events_deserialize_from_json() {
        let session_id = Uuid::new_v4();

        // Serialize and deserialize DreamCycleStarted
        let event1 = DreamCycleStarted::new(ExtendedTriggerReason::HighEntropy);
        let json1 = serde_json::to_string(&event1).unwrap();
        let deserialized1: DreamCycleStarted = serde_json::from_str(&json1).unwrap();
        assert_eq!(deserialized1.session_id, event1.session_id);
        assert_eq!(deserialized1.trigger_reason, "high_entropy");

        // Serialize and deserialize WakeTriggered
        let event2 = WakeTriggered::new(
            session_id,
            WakeReason::ManualAbort,
            "rem",
            Duration::from_millis(75),
        );
        let json2 = serde_json::to_string(&event2).unwrap();
        let deserialized2: WakeTriggered = serde_json::from_str(&json2).unwrap();
        assert_eq!(deserialized2.reason, "manual_abort");
        assert_eq!(deserialized2.phase, "rem");
        assert_eq!(deserialized2.latency_ms, 75);
    }

    #[test]
    fn test_current_timestamp_ms_is_reasonable() {
        let ts = current_timestamp_ms();
        // Timestamp should be after 2020-01-01 (1577836800000 ms)
        assert!(ts > 1577836800000, "Timestamp seems too old: {}", ts);
        // Timestamp should be before 2100-01-01 (4102444800000 ms)
        assert!(
            ts < 4102444800000,
            "Timestamp seems too far in future: {}",
            ts
        );
    }
}
