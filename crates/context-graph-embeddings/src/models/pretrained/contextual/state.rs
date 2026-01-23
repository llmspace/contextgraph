//! Model state for ContextualModel.
//!
//! Manages the loaded/unloaded state of the contextual embedding model.
//!
//! # E5-base-v2 Architecture
//!
//! E5-base-v2 uses prefix-based asymmetric encoding, so no projection
//! weights are needed. The model handles asymmetry through:
//! - "query: " prefix for intent embeddings
//! - "passage: " prefix for context embeddings

use tokenizers::Tokenizer;

use crate::gpu::BertWeights;

/// Internal state that varies based on whether the model is loaded.
#[allow(dead_code)]
pub enum ModelState {
    /// Unloaded - no weights in memory.
    Unloaded,

    /// Loaded with BERT-compatible weights and tokenizer.
    /// E5-base-v2 uses prefix-based asymmetry, no projection weights needed.
    Loaded {
        /// Model weights on GPU.
        weights: Box<BertWeights>,
        /// HuggingFace tokenizer for text encoding (boxed to reduce enum size).
        tokenizer: Box<Tokenizer>,
    },
}
