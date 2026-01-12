//! Real UTL processor implementing the constitution-specified formulas.
//!
//! Implements the Unified Theory of Learning per constitution.yaml:
//! - Canonical: L = f((ΔS × ΔC) · wₑ · cos φ)
//! - Multi-embedding: L_multi = sigmoid(2.0 · (Σᵢ τᵢλ_S·ΔSᵢ) · (Σⱼ τⱼλ_C·ΔCⱼ) · wₑ · cos φ)
//!
//! ΔS computed via KNN distance: ΔS = σ((d_k - μ)/σ_d)
//! ΔC computed via connectivity: ΔC = |{neighbors: sim(e, n) > θ_edge}| / max_edges

mod math;
mod processor;

#[cfg(test)]
mod tests;

// Re-export the main processor type
pub use processor::StubUtlProcessor;
