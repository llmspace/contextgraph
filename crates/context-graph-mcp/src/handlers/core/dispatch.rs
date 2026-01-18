//! Request dispatch logic for MCP handlers.
//!
//! Contains the dispatch method for routing JSON-RPC requests to appropriate handlers.

use tracing::debug;

use crate::protocol::{error_codes, methods, JsonRpcRequest, JsonRpcResponse};

use super::handlers::Handlers;

impl Handlers {
    /// Dispatch a request to the appropriate handler.
    pub async fn dispatch(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Dispatching method: {}", request.method);

        match request.method.as_str() {
            // MCP lifecycle methods
            methods::INITIALIZE => self.handle_initialize(request.id).await,
            "notifications/initialized" => self.handle_initialized_notification(),
            methods::SHUTDOWN => self.handle_shutdown(request.id).await,

            // MCP tools protocol
            methods::TOOLS_LIST => self.handle_tools_list(request.id).await,
            methods::TOOLS_CALL => self.handle_tools_call(request.id, request.params).await,

            // Legacy direct methods (kept for backward compatibility)
            methods::MEMORY_STORE => self.handle_memory_store(request.id, request.params).await,
            methods::MEMORY_RETRIEVE => {
                self.handle_memory_retrieve(request.id, request.params)
                    .await
            }
            methods::MEMORY_SEARCH => self.handle_memory_search(request.id, request.params).await,
            methods::MEMORY_DELETE => self.handle_memory_delete(request.id, request.params).await,

            // Memory injection and comparison operations (TASK-INTEG-001)
            methods::MEMORY_INJECT => self.handle_memory_inject(request.id, request.params).await,
            methods::MEMORY_INJECT_BATCH => {
                self.handle_memory_inject_batch(request.id, request.params)
                    .await
            }
            methods::MEMORY_SEARCH_MULTI_PERSPECTIVE => {
                self.handle_memory_search_multi_perspective(request.id, request.params)
                    .await
            }
            methods::MEMORY_COMPARE => self.handle_memory_compare(request.id, request.params).await,
            methods::MEMORY_BATCH_COMPARE => {
                self.handle_memory_batch_compare(request.id, request.params)
                    .await
            }
            methods::MEMORY_SIMILARITY_MATRIX => {
                self.handle_memory_similarity_matrix(request.id, request.params)
                    .await
            }

            // Search operations (TASK-S002)
            methods::SEARCH_MULTI => self.handle_search_multi(request.id, request.params).await,
            methods::SEARCH_SINGLE_SPACE => {
                self.handle_search_single_space(request.id, request.params)
                    .await
            }
            methods::SEARCH_BY_PURPOSE => {
                self.handle_search_by_purpose(request.id, request.params)
                    .await
            }
            methods::SEARCH_WEIGHT_PROFILES => self.handle_get_weight_profiles(request.id).await,

            // Purpose/goal operations (TASK-S003)
            // NOTE: Legacy alignment methods removed per TASK-CORE-001 (ARCH-03)
            // Goals now emerge autonomously from topic clustering.
            methods::PURPOSE_QUERY => self.handle_purpose_query(request.id, request.params).await,
            methods::GOAL_HIERARCHY_QUERY => {
                self.handle_goal_hierarchy_query(request.id, request.params)
                    .await
            }
            methods::GOAL_ALIGNED_MEMORIES => {
                self.handle_goal_aligned_memories(request.id, request.params)
                    .await
            }
            methods::PURPOSE_DRIFT_CHECK => {
                self.handle_purpose_drift_check(request.id, request.params)
                    .await
            }

            methods::UTL_COMPUTE => self.handle_utl_compute(request.id, request.params).await,
            methods::UTL_METRICS => self.handle_utl_metrics(request.id, request.params).await,

            // Meta-UTL operations (TASK-S005)
            methods::META_UTL_LEARNING_TRAJECTORY => {
                self.handle_meta_utl_learning_trajectory(request.id, request.params)
                    .await
            }
            methods::META_UTL_HEALTH_METRICS => {
                self.handle_meta_utl_health_metrics(request.id, request.params)
                    .await
            }
            methods::META_UTL_PREDICT_STORAGE => {
                self.handle_meta_utl_predict_storage(request.id, request.params)
                    .await
            }
            methods::META_UTL_PREDICT_RETRIEVAL => {
                self.handle_meta_utl_predict_retrieval(request.id, request.params)
                    .await
            }
            methods::META_UTL_VALIDATE_PREDICTION => {
                self.handle_meta_utl_validate_prediction(request.id, request.params)
                    .await
            }
            methods::META_UTL_OPTIMIZED_WEIGHTS => {
                self.handle_meta_utl_optimized_weights(request.id, request.params)
                    .await
            }

            methods::SYSTEM_STATUS => self.handle_system_status(request.id).await,
            methods::SYSTEM_HEALTH => self.handle_system_health(request.id).await,
            _ => JsonRpcResponse::error(
                request.id,
                error_codes::METHOD_NOT_FOUND,
                format!("Method not found: {}", request.method),
            ),
        }
    }
}
