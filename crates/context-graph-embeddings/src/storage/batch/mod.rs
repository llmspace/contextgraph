//! Batch encoding for GDS-compatible embedding files.
//!
//! Provides efficient multi-embedding serialization with 4KB page alignment
//! for optimal GPU Direct Storage (GDS) performance.
//!
//! # File Formats
//!
//! - `.cgeb` - Data file containing page-aligned embeddings
//! - `.cgei` - Index file with offset table for O(1) seeking
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::storage::BatchBinaryEncoder;
//!
//! let mut encoder = BatchBinaryEncoder::with_capacity(1000);
//! for embedding in embeddings {
//!     encoder.push(&embedding)?;
//! }
//! encoder.write_gds_file(Path::new("embeddings"))?;
//! // Creates: embeddings.cgeb (data) + embeddings.cgei (index)
//! ```

mod encoder;
mod types;

#[cfg(test)]
mod tests;

// Re-export public API for backwards compatibility
pub use encoder::BatchBinaryEncoder;
pub use types::{EmbeddingIndexHeader, INDEX_MAGIC, INDEX_VERSION};
