//! TeleologicalMemoryStore trait for 5-stage teleological retrieval.
//!
//! This module defines the core storage trait for the Context Graph system's
//! teleological memory architecture. It supports:
//! - CRUD operations for TeleologicalFingerprint
//! - Multi-space semantic search (13 embedding spaces)
//! - Purpose vector alignment search
//! - Sparse (SPLADE) search for Stage 1 recall
//! - Batch operations for efficiency
//! - Persistence and checkpointing
//!
//! # 5-Stage Retrieval Pipeline Support
//!
//! | Stage | Name | Method |
//! |-------|------|--------|
//! | 1 | Recall | `search_sparse()` - E13 SPLADE inverted index |
//! | 2 | Semantic | `search_semantic()` - E1 Matryoshka 128D ANN |
//! | 3 | Precision | `search_semantic()` - Full E1-E12 dense embeddings |
//! | 4 | Rerank | External - E12 ColBERT late interaction |
//! | 5 | Teleological | `search_purpose()` - 13D purpose vector alignment |
//!
//! # Design Philosophy
//!
//! - **NO BACKWARDS COMPATIBILITY**: Old MemoryStore trait deleted
//! - **FAIL FAST**: All errors return `CoreError` variants with context
//! - **UNIT TESTS**: Tests use stub `InMemoryTeleologicalStore` (not persistent RocksDB)
//! - **13 EMBEDDERS**: Full E1-E13 semantic fingerprint support
//!
//! # Module Structure
//!
//! - [`backend`]: Storage backend enum (`TeleologicalStorageBackend`)
//! - [`options`]: Search options (`TeleologicalSearchOptions`)
//! - [`result`]: Search result type (`TeleologicalSearchResult`)
//! - [`store`]: Core trait (`TeleologicalMemoryStore`)
//! - [`ext`]: Extension trait (`TeleologicalMemoryStoreExt`)

mod backend;
mod defaults;
mod ext;
mod options;
mod result;
mod store;

// Re-export all public types
pub use backend::TeleologicalStorageBackend;
pub use ext::TeleologicalMemoryStoreExt;
pub use options::TeleologicalSearchOptions;
pub use result::TeleologicalSearchResult;
pub use store::TeleologicalMemoryStore;
