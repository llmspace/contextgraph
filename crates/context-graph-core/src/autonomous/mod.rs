//! Autonomous North Star system
//!
//! This module provides types and services for autonomous operation of the
//! North Star alignment system including bootstrap, drift detection,
//! memory curation, and goal evolution.
//!
//! # Architecture
//!
//! The autonomous system is organized into two layers:
//!
//! ## Foundation Layer (Types)
//! - `bootstrap` - North Star goal initialization types
//! - `thresholds` - Adaptive threshold calibration types
//! - `drift` - Drift detection and monitoring types
//! - `curation` - Memory pruning and consolidation types
//! - `evolution` - Goal hierarchy evolution types
//! - `workflow` - Autonomous workflow orchestration types
//!
//! ## Logic Layer (Services)
//! - `services` - Active logic services implementing NORTH-008 to NORTH-020

pub mod bootstrap;
pub mod curation;
pub mod drift;
pub mod evolution;
pub mod thresholds;
pub mod workflow;
pub mod services;

pub use bootstrap::*;
pub use curation::*;
pub use drift::*;
pub use evolution::*;
pub use thresholds::*;
pub use workflow::*;

// Explicit service re-exports to avoid ambiguous glob re-exports
// (services module defines extended versions of some foundation types)
pub use services::{
    // NORTH-008: BootstrapService
    BootstrapService, BootstrapServiceConfig, GoalCandidate,
    // NORTH-009: ThresholdLearner
    ThresholdLearner, ThresholdLearnerConfig, ThompsonState, EmbedderLearningState, BayesianObservation,
    // NORTH-010: DriftDetector
    DriftDetector, DetectorDataPoint, DriftRecommendation, DetectorState,
    // NORTH-011: DriftCorrector
    DriftCorrector, DriftCorrectorConfig, CorrectionResult, CorrectionStrategy,
    // NORTH-012: PruningService
    PruningService, MemoryMetadata, ExtendedPruningConfig,
    // NORTH-013: ConsolidationService
    ConsolidationService, MemoryContent, MemoryPair, ServiceConsolidationCandidate,
    // NORTH-014: GapDetectionService
    GapDetectionService, GapType, GoalWithMetrics, GapDetectionConfig, GapReport,
    // NORTH-015: SubGoalDiscovery
    SubGoalDiscovery, MemoryCluster, DiscoveryConfig, DiscoveryResult,
    // NORTH-016: WeightAdjuster
    WeightAdjuster, WeightAdjusterConfig, AdjustmentReport,
    // NORTH-017: ObsolescenceDetector
    ObsolescenceDetector,
    // NORTH-018: DailyScheduler
    DailyScheduler, ScheduleResult, ScheduledTask, SchedulerCheckType,
    // NORTH-019: EventOptimizer
    EventOptimizer, EventOptimizerConfig, OptimizationTrigger, OptimizationAction,
    SystemMetrics, OptimizationEventRecord, OptimizationPlan,
    // NORTH-020: SelfHealingManager
    SelfHealingManager, SelfHealingConfig, HealingAction, HealingResult, HealthIssue, IssueSeverity,
};
