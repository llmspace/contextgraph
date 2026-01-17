//! MaxSim Late Interaction Benchmarks (TASK-STORAGE-P2-001)
//!
//! Performance validation for Stage 5 of the retrieval pipeline.
//!
//! # Targets
//!
//! - 50 candidates reranked in <15ms
//! - SIMD vs scalar: >4x speedup
//! - Token retrieval 50 IDs: <5ms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use context_graph_storage::teleological::search::{
    compute_maxsim_direct, cosine_similarity_128d, E12_TOKEN_DIM,
};

// ============================================================================
// TOKEN GENERATION
// ============================================================================

/// Generate a normalized random token of dimension E12_TOKEN_DIM (128).
fn generate_random_token(rng: &mut StdRng) -> Vec<f32> {
    let mut token: Vec<f32> = (0..E12_TOKEN_DIM).map(|_| rng.gen::<f32>() - 0.5).collect();

    // Normalize to unit length
    let norm: f32 = token.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut token {
            *x /= norm;
        }
    }

    token
}

/// Generate a document with n tokens.
fn generate_document(rng: &mut StdRng, n_tokens: usize) -> Vec<Vec<f32>> {
    (0..n_tokens).map(|_| generate_random_token(rng)).collect()
}

// ============================================================================
// BENCHMARKS
// ============================================================================

/// Benchmark cosine similarity for 128D vectors (SIMD target).
fn bench_cosine_128d(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a = generate_random_token(&mut rng);
    let b = generate_random_token(&mut rng);

    c.bench_function("cosine_similarity_128d", |bench| {
        bench.iter(|| cosine_similarity_128d(black_box(&a), black_box(&b)))
    });
}

/// Benchmark MaxSim scoring with varying candidate counts.
///
/// Target: 50 candidates in <15ms
fn bench_maxsim_scoring(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    // Typical query: 10 tokens
    let query = generate_document(&mut rng, 10);

    let mut group = c.benchmark_group("maxsim_scoring");

    // Benchmark different candidate counts
    for n_candidates in [10, 25, 50, 100] {
        // Generate candidate documents (15 tokens each, typical)
        let candidates: Vec<Vec<Vec<f32>>> = (0..n_candidates)
            .map(|_| generate_document(&mut rng, 15))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("candidates", n_candidates),
            &candidates,
            |bench, docs| {
                bench.iter(|| {
                    docs.iter()
                        .map(|doc| compute_maxsim_direct(black_box(&query), black_box(doc)))
                        .collect::<Vec<f32>>()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark single MaxSim computation with varying token counts.
fn bench_maxsim_token_scaling(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut group = c.benchmark_group("maxsim_token_scaling");

    for (n_query, n_doc) in [(5, 10), (10, 15), (15, 20), (20, 30)] {
        let query = generate_document(&mut rng, n_query);
        let doc = generate_document(&mut rng, n_doc);

        group.bench_with_input(
            BenchmarkId::new(format!("q{}_d{}", n_query, n_doc), n_query * n_doc),
            &(&query, &doc),
            |bench, (q, d)| bench.iter(|| compute_maxsim_direct(black_box(q), black_box(d))),
        );
    }

    group.finish();
}

/// Benchmark batch MaxSim with parallel execution.
fn bench_maxsim_batch(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut rng = StdRng::seed_from_u64(42);

    // Typical query: 10 tokens
    let query = generate_document(&mut rng, 10);

    // 50 candidates with 15 tokens each (typical Stage 5 input)
    let candidates: Vec<Vec<Vec<f32>>> = (0..50).map(|_| generate_document(&mut rng, 15)).collect();

    let mut group = c.benchmark_group("maxsim_batch");

    // Sequential baseline
    group.bench_function("sequential_50", |bench| {
        bench.iter(|| {
            candidates
                .iter()
                .map(|doc| compute_maxsim_direct(black_box(&query), black_box(doc)))
                .collect::<Vec<f32>>()
        })
    });

    // Parallel with rayon
    group.bench_function("parallel_50", |bench| {
        bench.iter(|| {
            candidates
                .par_iter()
                .map(|doc| compute_maxsim_direct(black_box(&query), black_box(doc)))
                .collect::<Vec<f32>>()
        })
    });

    group.finish();
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group!(
    benches,
    bench_cosine_128d,
    bench_maxsim_scoring,
    bench_maxsim_token_scaling,
    bench_maxsim_batch,
);
criterion_main!(benches);
