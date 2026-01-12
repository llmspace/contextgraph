//! Tests for EventOptimizer service.

use super::*;
use chrono::Utc;

#[test]
fn test_event_optimizer_new() {
    let optimizer = EventOptimizer::new();
    assert_eq!(optimizer.optimization_count(), 0);
    assert!(optimizer.event_history().is_empty());
    println!("[PASS] test_event_optimizer_new");
}

#[test]
fn test_event_optimizer_with_config() {
    let config = EventOptimizerConfig {
        max_history_size: 500,
        high_drift_threshold: 0.15,
        low_performance_threshold: 0.40,
        memory_pressure_threshold: 80.0,
        auto_optimize: false,
    };
    let optimizer = EventOptimizer::with_config(config.clone());
    assert_eq!(optimizer.config().max_history_size, 500);
    assert!(!optimizer.config().auto_optimize);
    println!("[PASS] test_event_optimizer_with_config");
}

#[test]
fn test_event_optimizer_on_event() {
    let mut optimizer = EventOptimizer::new();
    let trigger = OptimizationTrigger::HighDrift { severity: 0.15 };

    let plan = optimizer.on_event(trigger.clone());

    assert_eq!(plan.trigger, trigger);
    assert!(!plan.is_empty());
    println!("[PASS] test_event_optimizer_on_event");
}

#[test]
fn test_event_optimizer_plan_optimization() {
    let optimizer = EventOptimizer::new();
    let trigger = OptimizationTrigger::MemoryPressure {
        usage_percent: 90.0,
    };

    let plan = optimizer.plan_optimization(&trigger);

    assert!(plan.actions.contains(&OptimizationAction::PruneStaleData));
    assert!(plan.actions.contains(&OptimizationAction::CompactStorage));
    println!("[PASS] test_event_optimizer_plan_optimization");
}

#[test]
fn test_event_optimizer_execute_plan() {
    let mut optimizer = EventOptimizer::new();
    optimizer.set_current_metrics(SystemMetrics::new(0.70, 100.0, 80.0, 1000, 0.8));

    let trigger = OptimizationTrigger::HighDrift { severity: 0.15 };
    let plan = optimizer.plan_optimization(&trigger);
    let event = optimizer.execute_plan(&plan);

    assert!(event.success);
    assert_eq!(event.event_type, "high_drift");
    assert!(!event.actions_executed.is_empty());
    assert_eq!(optimizer.optimization_count(), 1);
    println!("[PASS] test_event_optimizer_execute_plan");
}

#[test]
fn test_event_optimizer_estimate_impact() {
    let optimizer = EventOptimizer::new();
    let actions = vec![
        OptimizationAction::RecomputeAlignments,
        OptimizationAction::RebalanceWeights,
    ];

    let impact = optimizer.estimate_impact(&actions);
    assert!(impact > 0.0);
    assert!(impact <= 1.0);
    println!("[PASS] test_event_optimizer_estimate_impact");
}

#[test]
fn test_event_optimizer_estimate_impact_empty() {
    let optimizer = EventOptimizer::new();
    let impact = optimizer.estimate_impact(&[]);
    assert!((impact - 0.0).abs() < f32::EPSILON);
    println!("[PASS] test_event_optimizer_estimate_impact_empty");
}

#[test]
fn test_event_optimizer_prioritize_actions() {
    let mut optimizer = EventOptimizer::new();
    optimizer.set_current_metrics(SystemMetrics::new(0.70, 100.0, 85.0, 1000, 0.8));

    let mut actions = vec![
        OptimizationAction::ReindexMemories,
        OptimizationAction::RecomputeAlignments,
        OptimizationAction::PruneStaleData,
        OptimizationAction::CompactStorage,
    ];

    optimizer.prioritize_actions(&mut actions);

    // With high memory usage, PruneStaleData and CompactStorage should be prioritized
    assert!(
        actions[0] == OptimizationAction::PruneStaleData
            || actions[0] == OptimizationAction::CompactStorage
    );
    println!("[PASS] test_event_optimizer_prioritize_actions");
}

#[test]
fn test_event_optimizer_prioritize_actions_index_repair() {
    let mut optimizer = EventOptimizer::new();
    optimizer.set_current_metrics(SystemMetrics::new(0.70, 100.0, 50.0, 1000, 0.5));

    let mut actions = vec![
        OptimizationAction::PruneStaleData,
        OptimizationAction::ReindexMemories,
        OptimizationAction::RebalanceWeights,
    ];

    optimizer.prioritize_actions(&mut actions);

    // With low index health, ReindexMemories should be prioritized
    assert_eq!(actions[0], OptimizationAction::ReindexMemories);
    println!("[PASS] test_event_optimizer_prioritize_actions_index_repair");
}

#[test]
fn test_event_optimizer_record_event() {
    let mut optimizer = EventOptimizer::new();

    let record = OptimizationEventRecord {
        event_type: "test".to_string(),
        timestamp: Utc::now(),
        metrics_before: SystemMetrics::default(),
        metrics_after: SystemMetrics::default(),
        success: true,
        actions_executed: vec![OptimizationAction::PruneStaleData],
        duration_ms: 100,
        error: None,
    };

    optimizer.record_event(record);

    assert_eq!(optimizer.event_history().len(), 1);
    assert_eq!(optimizer.optimization_count(), 1);
    println!("[PASS] test_event_optimizer_record_event");
}

#[test]
fn test_event_optimizer_record_event_failed() {
    let mut optimizer = EventOptimizer::new();

    let record = OptimizationEventRecord {
        event_type: "test".to_string(),
        timestamp: Utc::now(),
        metrics_before: SystemMetrics::default(),
        metrics_after: SystemMetrics::default(),
        success: false,
        actions_executed: vec![],
        duration_ms: 100,
        error: Some("Test error".to_string()),
    };

    optimizer.record_event(record);

    assert_eq!(optimizer.event_history().len(), 1);
    assert_eq!(optimizer.optimization_count(), 0); // Failed events don't count
    println!("[PASS] test_event_optimizer_record_event_failed");
}

#[test]
fn test_event_optimizer_history_trimming() {
    let config = EventOptimizerConfig {
        max_history_size: 5,
        ..Default::default()
    };
    let mut optimizer = EventOptimizer::with_config(config);

    for i in 0..10 {
        let record = OptimizationEventRecord {
            event_type: format!("test_{}", i),
            timestamp: Utc::now(),
            metrics_before: SystemMetrics::default(),
            metrics_after: SystemMetrics::default(),
            success: true,
            actions_executed: vec![],
            duration_ms: 100,
            error: None,
        };
        optimizer.record_event(record);
    }

    assert_eq!(optimizer.event_history().len(), 5);
    println!("[PASS] test_event_optimizer_history_trimming");
}

#[test]
fn test_event_optimizer_average_improvement() {
    let mut optimizer = EventOptimizer::new();

    // Record some events with improvements
    for _ in 0..5 {
        optimizer.set_current_metrics(SystemMetrics::new(0.70, 100.0, 80.0, 1000, 0.8));
        let trigger = OptimizationTrigger::HighDrift { severity: 0.12 };
        let plan = optimizer.plan_optimization(&trigger);
        optimizer.execute_plan(&plan);
    }

    assert!(optimizer.average_improvement() >= 0.0);
    println!("[PASS] test_event_optimizer_average_improvement");
}

#[test]
fn test_event_optimizer_average_improvement_no_events() {
    let optimizer = EventOptimizer::new();
    assert!((optimizer.average_improvement() - 0.0).abs() < f32::EPSILON);
    println!("[PASS] test_event_optimizer_average_improvement_no_events");
}

#[test]
fn test_event_optimizer_set_current_metrics() {
    let mut optimizer = EventOptimizer::new();
    let metrics = SystemMetrics::new(0.85, 25.0, 45.0, 5000, 0.95);

    optimizer.set_current_metrics(metrics.clone());

    assert!((optimizer.current_metrics().avg_alignment - 0.85).abs() < f32::EPSILON);
    assert!((optimizer.current_metrics().avg_latency_ms - 25.0).abs() < f32::EPSILON);
    println!("[PASS] test_event_optimizer_set_current_metrics");
}

// Integration tests
#[test]
fn test_full_optimization_cycle() {
    let mut optimizer = EventOptimizer::new();

    // Set initial metrics with some issues
    optimizer.set_current_metrics(SystemMetrics::new(0.65, 150.0, 85.0, 10000, 0.7));

    // Trigger high drift event
    let trigger = OptimizationTrigger::HighDrift { severity: 0.18 };
    let plan = optimizer.on_event(trigger);

    assert!(!plan.is_empty());
    assert!(plan.priority >= 8);

    // Execute the plan
    let event = optimizer.execute_plan(&plan);

    assert!(event.success);
    assert!(!event.actions_executed.is_empty());

    // Verify metrics improved
    assert!(optimizer.current_metrics().avg_alignment > 0.65);
    println!("[PASS] test_full_optimization_cycle");
}

#[test]
fn test_multiple_triggers() {
    let mut optimizer = EventOptimizer::new();
    optimizer.set_current_metrics(SystemMetrics::new(0.60, 200.0, 90.0, 15000, 0.6));

    // Execute multiple optimization cycles
    let triggers = [
        OptimizationTrigger::HighDrift { severity: 0.15 },
        OptimizationTrigger::MemoryPressure {
            usage_percent: 92.0,
        },
        OptimizationTrigger::LowPerformance {
            metric: "search_speed".to_string(),
            value: 0.35,
        },
    ];

    for trigger in triggers {
        let plan = optimizer.on_event(trigger);
        optimizer.execute_plan(&plan);
    }

    assert_eq!(optimizer.optimization_count(), 3);
    assert_eq!(optimizer.event_history().len(), 3);
    println!("[PASS] test_multiple_triggers");
}
