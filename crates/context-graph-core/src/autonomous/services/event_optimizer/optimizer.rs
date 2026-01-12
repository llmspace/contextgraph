//! Main EventOptimizer service implementation.

use chrono::Utc;
use std::collections::VecDeque;

use super::action::OptimizationAction;
use super::config::EventOptimizerConfig;
use super::metrics::SystemMetrics;
use super::plan::OptimizationPlan;
use super::record::OptimizationEventRecord;
use super::trigger::OptimizationTrigger;

/// Event-driven optimizer service
#[derive(Debug)]
pub struct EventOptimizer {
    config: EventOptimizerConfig,
    event_history: VecDeque<OptimizationEventRecord>,
    current_metrics: SystemMetrics,
    optimization_count: u64,
    total_improvement: f32,
}

impl Default for EventOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl EventOptimizer {
    /// Create a new EventOptimizer with default configuration
    pub fn new() -> Self {
        Self {
            config: EventOptimizerConfig::default(),
            event_history: VecDeque::with_capacity(1000),
            current_metrics: SystemMetrics::default(),
            optimization_count: 0,
            total_improvement: 0.0,
        }
    }

    /// Create an EventOptimizer with custom configuration
    pub fn with_config(config: EventOptimizerConfig) -> Self {
        let capacity = config.max_history_size;
        Self {
            config,
            event_history: VecDeque::with_capacity(capacity),
            current_metrics: SystemMetrics::default(),
            optimization_count: 0,
            total_improvement: 0.0,
        }
    }

    /// Handle an optimization trigger event and return a plan
    pub fn on_event(&mut self, trigger: OptimizationTrigger) -> OptimizationPlan {
        self.plan_optimization(&trigger)
    }

    /// Create an optimization plan for a given trigger
    pub fn plan_optimization(&self, trigger: &OptimizationTrigger) -> OptimizationPlan {
        let mut actions = OptimizationAction::recommended_for_trigger(trigger);
        self.prioritize_actions(&mut actions);
        OptimizationPlan::new(trigger.clone(), actions)
    }

    /// Execute an optimization plan and return the event record
    pub fn execute_plan(&mut self, plan: &OptimizationPlan) -> OptimizationEventRecord {
        let start_time = std::time::Instant::now();
        let metrics_before = self.current_metrics.clone();

        // Simulate execution of actions
        let (success, error) = self.execute_actions(&plan.actions);

        let duration_ms = start_time.elapsed().as_millis() as u64;
        let metrics_after = self.current_metrics.clone();

        let record = OptimizationEventRecord {
            event_type: plan.trigger.trigger_type_name().to_string(),
            timestamp: Utc::now(),
            metrics_before,
            metrics_after,
            success,
            actions_executed: plan.actions.clone(),
            duration_ms,
            error,
        };

        self.record_event(record.clone());
        record
    }

    /// Execute a list of optimization actions
    fn execute_actions(&mut self, actions: &[OptimizationAction]) -> (bool, Option<String>) {
        if actions.is_empty() {
            return (true, None);
        }

        for action in actions {
            // Apply simulated improvements based on action type
            match action {
                OptimizationAction::ReindexMemories => {
                    self.current_metrics.index_health =
                        (self.current_metrics.index_health + 0.1).min(1.0);
                    self.current_metrics.avg_latency_ms =
                        (self.current_metrics.avg_latency_ms * 0.9).max(1.0);
                }
                OptimizationAction::RecomputeAlignments => {
                    self.current_metrics.avg_alignment =
                        (self.current_metrics.avg_alignment + 0.05).min(1.0);
                }
                OptimizationAction::PruneStaleData => {
                    self.current_metrics.memory_usage_percent =
                        (self.current_metrics.memory_usage_percent - 5.0).max(0.0);
                }
                OptimizationAction::RebalanceWeights => {
                    self.current_metrics.avg_alignment =
                        (self.current_metrics.avg_alignment + 0.02).min(1.0);
                }
                OptimizationAction::CompactStorage => {
                    self.current_metrics.memory_usage_percent =
                        (self.current_metrics.memory_usage_percent - 3.0).max(0.0);
                }
            }
        }

        (true, None)
    }

    /// Estimate the impact of a set of actions
    pub fn estimate_impact(&self, actions: &[OptimizationAction]) -> f32 {
        if actions.is_empty() {
            return 0.0;
        }

        let total_benefit: f32 = actions.iter().map(|a| a.estimated_benefit()).sum();
        (total_benefit / actions.len() as f32).clamp(0.0, 1.0)
    }

    /// Prioritize actions based on current system state
    pub fn prioritize_actions(&self, actions: &mut [OptimizationAction]) {
        // Sort by benefit-to-cost ratio, descending
        actions.sort_by(|a, b| {
            let ratio_a = if a.estimated_cost() > 0.0 {
                a.estimated_benefit() / a.estimated_cost()
            } else {
                f32::MAX
            };
            let ratio_b = if b.estimated_cost() > 0.0 {
                b.estimated_benefit() / b.estimated_cost()
            } else {
                f32::MAX
            };
            ratio_b
                .partial_cmp(&ratio_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Additional prioritization based on current metrics
        let needs_memory_relief = self.current_metrics.memory_usage_percent > 80.0;
        let needs_index_repair = self.current_metrics.index_health < 0.7;

        if needs_memory_relief {
            // Move PruneStaleData and CompactStorage to the front
            actions.sort_by(|a, b| {
                let priority_a = matches!(
                    a,
                    OptimizationAction::PruneStaleData | OptimizationAction::CompactStorage
                );
                let priority_b = matches!(
                    b,
                    OptimizationAction::PruneStaleData | OptimizationAction::CompactStorage
                );
                priority_b.cmp(&priority_a)
            });
        } else if needs_index_repair {
            // Move ReindexMemories to the front
            actions.sort_by(|a, b| {
                let priority_a = matches!(a, OptimizationAction::ReindexMemories);
                let priority_b = matches!(b, OptimizationAction::ReindexMemories);
                priority_b.cmp(&priority_a)
            });
        }
    }

    /// Record an optimization event in history
    pub fn record_event(&mut self, event: OptimizationEventRecord) {
        if event.success {
            self.optimization_count += 1;
            self.total_improvement += event.improvement();
        }

        self.event_history.push_back(event);

        // Trim history if needed
        while self.event_history.len() > self.config.max_history_size {
            self.event_history.pop_front();
        }
    }

    /// Get the event history
    pub fn event_history(&self) -> &VecDeque<OptimizationEventRecord> {
        &self.event_history
    }

    /// Get the current metrics
    pub fn current_metrics(&self) -> &SystemMetrics {
        &self.current_metrics
    }

    /// Set the current metrics (for testing or external updates)
    pub fn set_current_metrics(&mut self, metrics: SystemMetrics) {
        self.current_metrics = metrics;
    }

    /// Get total optimization count
    pub fn optimization_count(&self) -> u64 {
        self.optimization_count
    }

    /// Get average improvement across all optimizations
    pub fn average_improvement(&self) -> f32 {
        if self.optimization_count > 0 {
            self.total_improvement / self.optimization_count as f32
        } else {
            0.0
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &EventOptimizerConfig {
        &self.config
    }
}
