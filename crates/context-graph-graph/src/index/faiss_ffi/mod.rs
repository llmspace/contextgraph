//! FAISS FFI bindings - re-exported from context-graph-cuda.
//!
//! All FAISS FFI is now consolidated in `context-graph-cuda::ffi::faiss`.
//! This module re-exports for backwards compatibility with existing imports.
//!
//! # Constitution Reference
//!
//! - ARCH-06: CUDA FFI only in context-graph-cuda

pub use context_graph_cuda::ffi::faiss::{
    // Helpers
    check_faiss_result,
    // FFI functions
    faiss_IndexIVF_set_nprobe,
    faiss_Index_add_with_ids,
    faiss_Index_free,
    faiss_Index_is_trained,
    faiss_Index_ntotal,
    faiss_Index_search,
    faiss_Index_train,
    faiss_StandardGpuResources_free,
    faiss_StandardGpuResources_new,
    faiss_get_num_gpus,
    faiss_index_cpu_to_gpu,
    faiss_index_factory,
    faiss_read_index,
    faiss_write_index,
    gpu_available,
    // Types
    FaissGpuResourcesProvider,
    FaissIndex,
    FaissStandardGpuResources,
    GpuResources,
    MetricType,
    FAISS_OK,
};

#[cfg(feature = "faiss-gpu")]
pub use context_graph_cuda::ffi::faiss::gpu_count_direct;
