//! Iterator implementation for GDS file reader.
//!
//! Provides sequential iteration over embeddings in a GDS file.

use super::error::GdsFileError;
use super::reader::GdsFile;
use crate::types::FusedEmbedding;

/// Iterator over embeddings in a GDS file.
pub struct GdsFileIter<'a> {
    file: &'a mut GdsFile,
    current: usize,
}

impl<'a> GdsFileIter<'a> {
    /// Create a new iterator over the GDS file.
    pub(crate) fn new(file: &'a mut GdsFile) -> Self {
        Self { file, current: 0 }
    }
}

impl<'a> Iterator for GdsFileIter<'a> {
    type Item = Result<FusedEmbedding, GdsFileError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.file.len() {
            return None;
        }
        let result = self.file.read(self.current);
        self.current += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.file.len() - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for GdsFileIter<'a> {}
