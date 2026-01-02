//! GDS file reader for batch embeddings.
//!
//! Provides O(1) seeking to any embedding by index using the index file.
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::storage::GdsFile;
//!
//! let mut gds = GdsFile::open(Path::new("embeddings"))?;
//! println!("File contains {} embeddings", gds.len());
//!
//! // O(1) random access
//! let embedding = gds.read(42)?;
//! println!("Content hash: {:#x}", embedding.content_hash);
//! ```

mod error;
mod iter;
mod reader;

#[cfg(test)]
mod tests;

// Re-export public API for backwards compatibility
pub use error::GdsFileError;
pub use iter::GdsFileIter;
pub use reader::GdsFile;
