//! Self-Healing Manager implementation (NORTH-020)
//!
//! This module contains the main `SelfHealingManager` struct that coordinates
//! health monitoring, issue diagnosis, and healing actions.

use chrono::{DateTime, Utc};
use std::collections::HashMap;

use super::types::{
    HealingAction, HealingResult, HealthIssue, IssueSeverity, SelfHealingConfig, SystemHealthState,
    SystemOperationalStatus,
};

/// Self-healing manager service
#[derive(Debug)]
pub struct SelfHealingManager {
    config: SelfHealingConfig,
    pub(crate) recovery_history: Vec<HealingResult>,
    pub(crate) last_healing_time: Option<DateTime<Utc>>,
    pub(crate) healing_attempts: HashMap<String, u32>,
}

impl Default for SelfHealingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfHealingManager {
    /// Create a new self-healing manager with default config
    pub fn new() -> Self {
        Self {
            config: SelfHealingConfig::default(),
            recovery_history: Vec::new(),
            last_healing_time: None,
            healing_attempts: HashMap::new(),
        }
    }

    /// Create a self-healing manager with custom config
    pub fn with_config(config: SelfHealingConfig) -> Self {
        Self {
            config,
            recovery_history: Vec::new(),
            last_healing_time: None,
            healing_attempts: HashMap::new(),
        }
    }

    /// Check the overall health of the system
    pub fn check_health(&self) -> SystemHealthState {
        // Build health state from component checks
        let mut health = SystemHealthState::healthy();

        // Add standard components with their health scores
        // In a real implementation, these would probe actual subsystems
        health.add_component("memory", 1.0);
        health.add_component("drift_detector", 1.0);
        health.add_component("threshold_learner", 1.0);
        health.add_component("pruning_service", 1.0);
        health.add_component("consolidation_service", 1.0);

        health.last_check = Utc::now();
        health
    }

    /// Diagnose issues from the current health state
    pub fn diagnose_issues(&self, health: &SystemHealthState) -> Vec<HealthIssue> {
        let mut issues = Vec::new();

        // Check overall health
        if health.overall_score < self.config.health_threshold {
            issues.push(HealthIssue::error(
                "system",
                format!(
                    "Overall health score {:.2} below threshold {:.2}",
                    health.overall_score, self.config.health_threshold
                ),
            ));
        }

        // Check each component
        for (component, score) in &health.component_scores {
            if *score < self.config.health_threshold {
                let severity = if *score < 0.3 {
                    IssueSeverity::Critical
                } else if *score < 0.5 {
                    IssueSeverity::Error
                } else {
                    IssueSeverity::Warning
                };

                issues.push(HealthIssue::new(
                    component.clone(),
                    severity,
                    format!("Component health {:.2} below threshold", score),
                ));
            }
        }

        // Include any pre-existing unresolved issues
        for issue in &health.issues {
            if !issue.resolved {
                issues.push(issue.clone());
            }
        }

        issues
    }

    /// Select the appropriate healing action for an issue
    pub fn select_healing_action(&self, issue: &HealthIssue) -> HealingAction {
        // Check if we've exceeded healing attempts for this component
        if let Some(attempts) = self.healing_attempts.get(&issue.component) {
            if *attempts >= self.config.max_healing_attempts {
                return HealingAction::Escalate {
                    reason: format!(
                        "Max healing attempts ({}) exceeded for {}",
                        self.config.max_healing_attempts, issue.component
                    ),
                };
            }
        }

        // Select action based on severity and component
        match issue.severity {
            IssueSeverity::Critical => {
                // Critical issues need restart
                HealingAction::RestartComponent {
                    name: issue.component.clone(),
                }
            }
            IssueSeverity::Error => {
                // Error issues try reset first
                HealingAction::ResetState {
                    component: issue.component.clone(),
                }
            }
            IssueSeverity::Warning => {
                // Warnings try cache clear
                HealingAction::ClearCache {
                    component: issue.component.clone(),
                }
            }
            IssueSeverity::Info => {
                // Info issues need no action
                HealingAction::NoAction
            }
        }
    }

    /// Apply a healing action and return the result
    pub fn apply_healing(&mut self, action: &HealingAction) -> HealingResult {
        let health_before = self.check_health().overall_score;

        // Check cooldown
        if let Some(last_time) = self.last_healing_time {
            let elapsed = Utc::now()
                .signed_duration_since(last_time)
                .num_seconds()
                .unsigned_abs();
            if elapsed < self.config.healing_cooldown_secs {
                return HealingResult::failure(
                    action.clone(),
                    health_before,
                    health_before,
                    format!(
                        "Healing cooldown active ({} seconds remaining)",
                        self.config.healing_cooldown_secs - elapsed
                    ),
                );
            }
        }

        // Track healing attempt
        let component_name = match action {
            HealingAction::RestartComponent { name } => Some(name.clone()),
            HealingAction::ClearCache { component } => Some(component.clone()),
            HealingAction::ResetState { component } => Some(component.clone()),
            _ => None,
        };

        if let Some(ref name) = component_name {
            *self.healing_attempts.entry(name.clone()).or_insert(0) += 1;
        }

        // Execute the healing action
        let success = match action {
            HealingAction::RestartComponent { .. } => {
                // Simulate component restart
                true
            }
            HealingAction::ClearCache { .. } => {
                // Simulate cache clear
                true
            }
            HealingAction::ResetState { .. } => {
                // Simulate state reset
                true
            }
            HealingAction::Escalate { .. } => {
                // Escalation is logged, not executed
                false
            }
            HealingAction::NoAction => {
                // No action means nothing to do
                true
            }
        };

        self.last_healing_time = Some(Utc::now());

        let health_after = if success {
            // Assume health improves after successful healing
            (health_before + 0.1).min(1.0)
        } else {
            health_before
        };

        if success {
            HealingResult::success(action.clone(), health_before, health_after)
        } else {
            HealingResult::failure(
                action.clone(),
                health_before,
                health_after,
                "Healing action could not be completed",
            )
        }
    }

    /// Check if the system is healthy based on the health state
    pub fn is_healthy(&self, health: &SystemHealthState) -> bool {
        health.overall_score >= self.config.health_threshold && !health.has_unresolved_issues()
    }

    /// Get the system status based on health state
    pub fn get_status(&self, health: &SystemHealthState) -> SystemOperationalStatus {
        if health.overall_score >= 0.9 && !health.has_unresolved_issues() {
            SystemOperationalStatus::Running
        } else if health.overall_score >= self.config.health_threshold {
            if health.has_unresolved_issues() {
                SystemOperationalStatus::Degraded
            } else {
                SystemOperationalStatus::Running
            }
        } else if health.overall_score >= 0.3 {
            SystemOperationalStatus::Recovering
        } else {
            SystemOperationalStatus::Failed
        }
    }

    /// Record a recovery result in history
    pub fn record_recovery(&mut self, issue: &HealthIssue) {
        let action = self.select_healing_action(issue);
        let result = self.apply_healing(&action);
        self.recovery_history.push(result);
    }

    /// Get the recovery history
    pub fn get_recovery_history(&self) -> Vec<HealingResult> {
        self.recovery_history.clone()
    }

    /// Clear healing attempts counter (for testing or reset)
    pub fn reset_healing_attempts(&mut self) {
        self.healing_attempts.clear();
    }

    /// Get current config
    pub fn config(&self) -> &SelfHealingConfig {
        &self.config
    }
}
