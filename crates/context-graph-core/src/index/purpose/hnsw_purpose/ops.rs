//! PurposeIndexOps trait implementation for HnswPurposeIndex.

use uuid::Uuid;

use crate::index::config::PURPOSE_VECTOR_DIM;

use super::super::entry::PurposeIndexEntry;
use super::super::error::{PurposeIndexError, PurposeIndexResult};
use super::super::query::{PurposeQuery, PurposeQueryTarget, PurposeSearchResult};
use super::HnswPurposeIndex;
use super::PurposeIndexOps;

impl PurposeIndexOps for HnswPurposeIndex {
    fn insert(&mut self, entry: PurposeIndexEntry) -> PurposeIndexResult<()> {
        let memory_id = entry.memory_id;
        let alignments = entry.purpose_vector.alignments;

        // Validate dimension
        if alignments.len() != PURPOSE_VECTOR_DIM {
            return Err(PurposeIndexError::dimension_mismatch(
                PURPOSE_VECTOR_DIM,
                alignments.len(),
            ));
        }

        // Remove existing entry if present (update case)
        if self.metadata.contains_key(&memory_id) {
            // Get old metadata for secondary index cleanup
            if let Some(old_metadata) = self.metadata.get(&memory_id) {
                let old_metadata_clone = old_metadata.clone();
                self.remove_from_secondary_indexes(memory_id, &old_metadata_clone);
            }
            self.inner.remove(memory_id);
        }

        // Insert into HNSW
        self.inner.add(memory_id, &alignments)?;

        // Store metadata and vector
        self.metadata.insert(memory_id, entry.metadata.clone());
        self.vectors.insert(memory_id, entry.purpose_vector.clone());

        // Update secondary indexes
        self.update_secondary_indexes(memory_id, &entry.metadata);

        Ok(())
    }

    fn remove(&mut self, memory_id: Uuid) -> PurposeIndexResult<()> {
        // Fail-fast: check existence
        let metadata = self
            .metadata
            .get(&memory_id)
            .ok_or_else(|| PurposeIndexError::not_found(memory_id))?
            .clone();

        // Remove from HNSW
        self.inner.remove(memory_id);

        // Remove from primary storage
        self.metadata.remove(&memory_id);
        self.vectors.remove(&memory_id);

        // Remove from secondary indexes
        self.remove_from_secondary_indexes(memory_id, &metadata);

        Ok(())
    }

    fn search(&self, query: &PurposeQuery) -> PurposeIndexResult<Vec<PurposeSearchResult>> {
        // Validate query
        query.validate()?;

        // Handle different query targets
        match &query.target {
            PurposeQueryTarget::Vector(pv) => self.search_vector(pv, query),

            PurposeQueryTarget::Pattern {
                min_cluster_size,
                coherence_threshold,
            } => self.search_pattern(*min_cluster_size, *coherence_threshold, query),

            PurposeQueryTarget::FromMemory(memory_id) => {
                // Fail-fast: memory must exist
                let vector = self
                    .vectors
                    .get(memory_id)
                    .ok_or_else(|| PurposeIndexError::not_found(*memory_id))?
                    .clone();

                self.search_vector(&vector, query)
            }
        }
    }

    fn get(&self, memory_id: Uuid) -> PurposeIndexResult<PurposeIndexEntry> {
        let metadata = self
            .metadata
            .get(&memory_id)
            .ok_or_else(|| PurposeIndexError::not_found(memory_id))?
            .clone();

        let vector = self
            .vectors
            .get(&memory_id)
            .ok_or_else(|| PurposeIndexError::not_found(memory_id))?
            .clone();

        Ok(PurposeIndexEntry::new(memory_id, vector, metadata))
    }

    fn contains(&self, memory_id: Uuid) -> bool {
        self.metadata.contains_key(&memory_id)
    }

    fn len(&self) -> usize {
        self.metadata.len()
    }

    fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    fn clear(&mut self) {
        // Get all IDs first to avoid borrowing issues during removal
        let ids: Vec<Uuid> = self.metadata.keys().copied().collect();

        // Clear primary storage
        self.metadata.clear();
        self.vectors.clear();
        self.goal_index.clear();

        // Remove all entries from HNSW index
        for id in ids {
            self.inner.remove(id);
        }
    }
}
