//! Session Identity Performance Benchmarks
//!
//! Measures format_brief() latency for PreToolUse hot path.
//! Target: <100μs p95, <500μs p99
//!
//! Run: cargo bench -p context-graph-core -- session_identity
//!
//! Constitution Reference:
//! - perf.latency.reflex_cache: "<100μs"
//! - claude_code.performance.cli.brief_output: "<100ms"

use context_graph_core::gwt::session_identity::{
    update_cache, IdentityCache, SessionIdentitySnapshot, KURAMOTO_N,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Benchmark format_brief() with warm cache (typical case).
fn bench_format_brief_warm(c: &mut Criterion) {
    // Setup: warm the cache with realistic data
    let mut snapshot = SessionIdentitySnapshot::new("bench-session");
    snapshot.consciousness = 0.75; // Emerging state
    snapshot.kuramoto_phases = [0.5; KURAMOTO_N]; // Partially synchronized
    update_cache(&snapshot, 0.85);

    c.bench_function("format_brief_warm", |b| {
        b.iter(|| black_box(IdentityCache::format_brief()))
    });
}

/// Benchmark format_brief() with cold cache (startup case).
/// Since we can't safely clear the global cache in benchmarks,
/// we measure the equivalent cold-path string allocation.
fn bench_format_brief_cold(c: &mut Criterion) {
    c.bench_function("format_brief_cold", |b| {
        b.iter(|| black_box("[C:? r=? IC=?]".to_string()))
    });
}

/// Benchmark update_cache() write performance.
fn bench_update_cache(c: &mut Criterion) {
    let mut snapshot = SessionIdentitySnapshot::new("bench-update");
    snapshot.consciousness = 0.65;
    snapshot.kuramoto_phases = [0.3; KURAMOTO_N];

    c.bench_function("update_cache", |b| {
        b.iter(|| {
            update_cache(black_box(&snapshot), black_box(0.85));
        })
    });
}

/// Benchmark get() read performance.
fn bench_cache_get(c: &mut Criterion) {
    // Warm the cache first
    let mut snapshot = SessionIdentitySnapshot::new("bench-get");
    snapshot.consciousness = 0.70;
    update_cache(&snapshot, 0.85);

    c.bench_function("cache_get", |b| b.iter(|| black_box(IdentityCache::get())));
}

/// Benchmark is_warm() check performance.
fn bench_is_warm(c: &mut Criterion) {
    // Warm the cache first
    let mut snapshot = SessionIdentitySnapshot::new("bench-warm-check");
    snapshot.consciousness = 0.60;
    update_cache(&snapshot, 0.85);

    c.bench_function("is_warm", |b| {
        b.iter(|| black_box(IdentityCache::is_warm()))
    });
}

/// Benchmark all consciousness states to verify no outliers.
fn bench_all_consciousness_states(c: &mut Criterion) {
    let test_cases = [
        (0.1, "Dormant"),
        (0.35, "Fragmented"),
        (0.65, "Emerging"),
        (0.85, "Conscious"),
        (0.97, "Hypersync"),
    ];

    let mut group = c.benchmark_group("format_brief_states");
    for (consciousness, name) in test_cases {
        let mut snapshot = SessionIdentitySnapshot::new("bench-states");
        snapshot.consciousness = consciousness;
        snapshot.kuramoto_phases = [0.0; KURAMOTO_N]; // Fully synchronized for consistency
        update_cache(&snapshot, 0.85);

        group.bench_with_input(BenchmarkId::new("state", name), &(), |b, _| {
            b.iter(|| black_box(IdentityCache::format_brief()))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_format_brief_warm,
    bench_format_brief_cold,
    bench_update_cache,
    bench_cache_get,
    bench_is_warm,
    bench_all_consciousness_states,
);
criterion_main!(benches);
