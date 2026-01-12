//! Performance Benchmark tests - CRITICAL <10ms requirement.

use std::time::Instant;

use crate::layers::learning::{LearningLayer, UtlWeightComputer, DEFAULT_LEARNING_RATE};
use crate::traits::NervousLayer;
use crate::types::LayerInput;

#[tokio::test]
async fn test_learning_layer_latency_benchmark() {
    let layer = LearningLayer::new();

    let iterations = 10_000;
    let mut total_us: u64 = 0;
    let mut max_us: u64 = 0;

    for i in 0..iterations {
        let mut input =
            LayerInput::new(format!("bench-{}", i), format!("Benchmark content {}", i));
        input.context.pulse.entropy = (i as f32 / iterations as f32).clamp(0.0, 1.0);
        input.context.pulse.coherence = 0.5 + (i as f32 / iterations as f32 * 0.5);

        let start = Instant::now();
        let _ = layer.process(input).await;
        let elapsed = start.elapsed().as_micros() as u64;

        total_us += elapsed;
        max_us = max_us.max(elapsed);
    }

    let avg_us = total_us / iterations as u64;

    println!("Learning Layer Benchmark Results:");
    println!("  Iterations: {}", iterations);
    println!("  Avg latency: {} us", avg_us);
    println!("  Max latency: {} us", max_us);
    println!("  Budget: 10000 us (10ms)");

    // Average should be well under budget
    assert!(
        avg_us < 10_000,
        "Average latency {} us exceeds 10ms budget",
        avg_us
    );

    // Even max should be under budget for a well-behaved layer
    assert!(
        max_us < 10_000,
        "Max latency {} us exceeds 10ms budget",
        max_us
    );

    println!("[VERIFIED] Average latency {} us < 10000 us budget", avg_us);
}

#[test]
fn test_utl_computation_benchmark() {
    let computer = UtlWeightComputer::new(DEFAULT_LEARNING_RATE);

    let iterations = 100_000;
    let start = Instant::now();

    for i in 0..iterations {
        let s = (i as f32 / iterations as f32).clamp(0.0, 1.0);
        let c = 1.0 - s;
        let _ = computer.compute_update(s, c);
    }

    let total_us = start.elapsed().as_micros();
    let avg_ns = (total_us * 1000) / iterations as u128;

    println!("UTL Computation Benchmark:");
    println!("  Iterations: {}", iterations);
    println!("  Total time: {} us", total_us);
    println!("  Avg per op: {} ns", avg_ns);

    // Each computation should be sub-microsecond
    assert!(avg_ns < 1000, "UTL computation too slow: {} ns", avg_ns);

    println!("[VERIFIED] UTL computation avg {} ns < 1000 ns", avg_ns);
}
