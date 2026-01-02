//! GDS file error types.
//!
//! This module contains the error enum for GDS file operations.

use crate::storage::binary::DecodeError;
use std::io;

/// Errors for GDS file operations.
#[derive(Debug, thiserror::Error)]
pub enum GdsFileError {
    /// Invalid index file magic bytes.
    #[error("Invalid index file magic bytes: expected 'CGEI' (0x43474549)")]
    InvalidIndexMagic,

    /// Index out of bounds access.
    #[error("Index out of bounds: {index} >= {len}")]
    IndexOutOfBounds {
        /// The requested index.
        index: usize,
        /// The total length.
        len: usize,
    },

    /// Decode error when reading embedding.
    #[error("Decode error: {0}")]
    Decode(#[from] DecodeError),

    /// IO error during file operations.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}
