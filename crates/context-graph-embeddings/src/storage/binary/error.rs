//! Error types for binary encoding/decoding.

use std::io;

use super::types::EMBEDDING_BINARY_VERSION;

/// Errors during encoding.
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("Buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall { needed: usize, available: usize },

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}

/// Errors during decoding.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("Invalid magic bytes: expected 'CGEB' (0x43474542)")]
    InvalidMagic,

    #[error("Unsupported format version: {0} (max supported: {EMBEDDING_BINARY_VERSION})")]
    UnsupportedVersion(u16),

    #[error("Buffer too short: need {needed} bytes, have {available}")]
    BufferTooShort { needed: usize, available: usize },

    #[error("Content hash mismatch: file may be corrupted")]
    HashMismatch,

    #[error("Alignment error: expected {expected}-byte alignment, got offset {actual}")]
    AlignmentError { expected: usize, actual: usize },

    #[error("Auxiliary data corrupted: {0}")]
    AuxDataCorrupted(String),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}
