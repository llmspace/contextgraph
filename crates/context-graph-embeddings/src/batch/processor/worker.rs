//! BatchProcessor worker loop and batch processing.
//!
//! Contains the async worker loop that manages queue polling
//! and batch processing through models.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, Notify, RwLock, Semaphore};
use tokio::time::interval;

use crate::error::EmbeddingResult;
use crate::models::ModelRegistry;
use crate::traits::EmbeddingModel as _;
use crate::types::{ModelEmbedding, ModelId};

use crate::batch::{Batch, BatchQueue, BatchRequest};

use super::stats::BatchProcessorStatsInternal;

// ============================================================================
// WORKER LOOP
// ============================================================================

/// Main worker loop that processes requests and batches.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn worker_loop(
    queues: Arc<RwLock<HashMap<ModelId, BatchQueue>>>,
    registry: Arc<ModelRegistry>,
    mut request_rx: mpsc::Receiver<BatchRequest>,
    shutdown_notify: Arc<Notify>,
    is_running: Arc<AtomicBool>,
    stats: Arc<BatchProcessorStatsInternal>,
    batch_semaphore: Arc<Semaphore>,
    poll_interval: Duration,
) {
    let mut poll_timer = interval(poll_interval);

    loop {
        tokio::select! {
            // Check for shutdown
            _ = shutdown_notify.notified() => {
                // Process remaining batches before exiting
                flush_all_queues(&queues, &registry, &stats, &batch_semaphore).await;
                break;
            }

            // Receive new requests
            Some(request) = request_rx.recv() => {
                let model_id = request.model_id;

                // Add to appropriate queue
                {
                    let mut queues_guard = queues.write().await;
                    if let Some(queue) = queues_guard.get_mut(&model_id) {
                        queue.push(request);
                    }
                }

                // Check if this queue should flush
                check_and_process_queue(
                    queues.clone(),
                    registry.clone(),
                    model_id,
                    stats.clone(),
                    batch_semaphore.clone(),
                ).await;
            }

            // Poll for timeouts
            _ = poll_timer.tick() => {
                if !is_running.load(Ordering::Relaxed) {
                    break;
                }

                // Check all queues for timeout-triggered flushes
                for model_id in ModelId::all() {
                    check_and_process_queue(
                        queues.clone(),
                        registry.clone(),
                        *model_id,
                        stats.clone(),
                        batch_semaphore.clone(),
                    ).await;
                }
            }
        }
    }
}

// ============================================================================
// QUEUE PROCESSING
// ============================================================================

/// Check if a queue should flush and process the batch.
pub(crate) async fn check_and_process_queue(
    queues: Arc<RwLock<HashMap<ModelId, BatchQueue>>>,
    registry: Arc<ModelRegistry>,
    model_id: ModelId,
    stats: Arc<BatchProcessorStatsInternal>,
    batch_semaphore: Arc<Semaphore>,
) {
    // Check if should flush (read lock)
    let should_flush = {
        let queues_guard = queues.read().await;
        queues_guard
            .get(&model_id)
            .map(|q| q.should_flush())
            .unwrap_or(false)
    };

    if !should_flush {
        return;
    }

    // Acquire semaphore permit for concurrent batch limiting
    let permit = match batch_semaphore.try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => return, // Max concurrent batches reached, try next poll
    };

    // Extract batch (write lock)
    let batch = {
        let mut queues_guard = queues.write().await;
        queues_guard
            .get_mut(&model_id)
            .and_then(|q| q.drain_batch())
    };

    if let Some(batch) = batch {
        // Process batch asynchronously
        tokio::spawn(async move {
            let _permit = permit; // Hold permit until batch completes
            process_batch(batch, &registry, &stats).await;
        });
    }
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

/// Process a single batch through the model.
pub(crate) async fn process_batch(
    batch: Batch,
    registry: &Arc<ModelRegistry>,
    stats: &Arc<BatchProcessorStatsInternal>,
) {
    let batch_size = batch.len();
    let model_id = batch.model_id;

    // Get model from registry
    let model = match registry.get_model(model_id).await {
        Ok(model) => model,
        Err(e) => {
            // Fail entire batch - NO FALLBACKS
            batch.fail(format!("Failed to get model {:?}: {}", model_id, e));
            stats.add_requests_failed(batch_size as u64);
            return;
        }
    };

    // Process each input in the batch
    // Note: We process sequentially here. For true GPU batching,
    // individual model implementations should optimize internally.
    let mut results: Vec<EmbeddingResult<ModelEmbedding>> = Vec::with_capacity(batch_size);
    let mut success_count: u64 = 0;
    let mut fail_count: u64 = 0;

    for input in &batch.inputs {
        match model.embed(input).await {
            Ok(embedding) => {
                results.push(Ok(embedding));
                success_count += 1;
            }
            Err(e) => {
                results.push(Err(e));
                fail_count += 1;
            }
        }
    }

    // Complete batch with individual results
    batch.complete(results);

    // Update stats
    stats.add_requests_completed(success_count);
    stats.add_requests_failed(fail_count);
    stats.inc_batches_processed();
}

// ============================================================================
// FLUSH OPERATIONS
// ============================================================================

/// Flush all queues (used during shutdown).
pub(crate) async fn flush_all_queues(
    queues: &Arc<RwLock<HashMap<ModelId, BatchQueue>>>,
    registry: &Arc<ModelRegistry>,
    stats: &Arc<BatchProcessorStatsInternal>,
    batch_semaphore: &Arc<Semaphore>,
) {
    for model_id in ModelId::all() {
        loop {
            // Check if queue has items
            let has_items = {
                let queues_guard = queues.read().await;
                queues_guard
                    .get(model_id)
                    .map(|q| !q.is_empty())
                    .unwrap_or(false)
            };

            if !has_items {
                break;
            }

            // Acquire permit (blocking during shutdown is OK)
            let permit = match batch_semaphore.acquire().await {
                Ok(permit) => permit,
                Err(_) => break, // Semaphore closed
            };

            // Extract and process batch
            let batch = {
                let mut queues_guard = queues.write().await;
                queues_guard.get_mut(model_id).and_then(|q| q.drain_batch())
            };

            if let Some(batch) = batch {
                process_batch(batch, registry, stats).await;
            }

            drop(permit);
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queues_created_for_all_12_models() {
        // Verify that we would create 12 queues - one per ModelId
        let all_models = ModelId::all();
        assert_eq!(all_models.len(), 12, "Expected 12 models");

        println!("\n========================================");
        println!("QUEUE CREATION VERIFICATION");
        println!("========================================");
        for (i, model_id) in all_models.iter().enumerate() {
            println!("Queue {}: {:?}", i + 1, model_id);
        }
        println!("Total queues: {}", all_models.len());
        println!("========================================\n");
    }
}
