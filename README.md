# Context Graph

> A Bio-Nervous Knowledge Graph with Computational Consciousness

Context Graph is a sophisticated AI memory system that implements cutting-edge computational neuroscience principles. It provides AI assistants with persistent, intelligent memory through 13 specialized embedding spaces, adaptive threshold calibration, dream consolidation, and a Global Workspace Theory (GWT) consciousness model.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [5-Layer Bio-Nervous System](#5-layer-bio-nervous-system)
  - [13-Embedding Teleological Fingerprint](#13-embedding-teleological-fingerprint)
  - [Retrieval Pipeline](#retrieval-pipeline)
- [MCP Tools Reference](#mcp-tools-reference)
  - [Core Memory Tools](#1-core-memory-tools)
  - [System Status Tools](#2-system-status-tools)
  - [Search Tools](#3-search-tools)
  - [GWT Consciousness Tools](#4-gwt-consciousness-tools)
  - [Adaptive Threshold Calibration Tools](#5-adaptive-threshold-calibration-atc-tools)
  - [Dream Consolidation Tools](#6-dream-consolidation-tools)
  - [Neuromodulation Tools](#7-neuromodulation-tools)
  - [Steering Tools](#8-steering-tools)
  - [Causal Inference Tools](#9-causal-inference-tools)
- [Cognitive Systems](#cognitive-systems)
  - [Global Workspace Theory](#global-workspace-theory-gwt)
  - [Kuramoto Oscillator Network](#kuramoto-oscillator-network)
  - [Unified Theory of Learning](#unified-theory-of-learning-utl)
  - [Neuromodulation](#neuromodulation-system)
  - [Dream Consolidation](#dream-consolidation-system)
  - [Adaptive Threshold Calibration](#adaptive-threshold-calibration-atc)
- [Installation](#installation)
- [Configuration](#configuration)
- [Optimal Usage Patterns](#optimal-usage-patterns)
- [API Reference](#api-reference)

---

## Overview

Context Graph transforms how AI assistants remember and learn. Unlike traditional vector databases that store flat embeddings, Context Graph maintains a **teleological fingerprint** for each memory—a 13-dimensional representation capturing meaning, time, causality, code structure, and purpose alignment.

### Key Features

- **13 Specialized Embedding Spaces**: Semantic, temporal, causal, code, graph, and more
- **Computational Consciousness**: Global Workspace Theory with Kuramoto synchronization
- **Adaptive Learning**: No hardcoded thresholds—everything is learned
- **Dream Consolidation**: NREM/REM cycles for memory optimization
- **Neuromodulation**: Dopamine, serotonin, noradrenaline, acetylcholine modulation
- **5-Stage Retrieval**: <60ms latency at 1M+ memories

### Design Philosophy

1. **No Fusion Paradigm**: All 13 embeddings stored separately for 100% information preservation
2. **Fail-Fast Doctrine**: No fallbacks or silent failures
3. **Librarian, Not Archivist**: Quality curation over quantity
4. **Bio-Nervous Model**: Inspired by computational neuroscience
5. **Self-Learning Thresholds**: Everything adapts via EWMA, Thompson sampling, and Bayesian optimization

---

## Architecture

### 5-Layer Bio-Nervous System

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5 (L5) - COHERENCE                                       │
│  Thalamic gating, Predictive Coding, Global Workspace broadcast │
│  Latency: 10ms sync, eventual consistency                       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4 (L4) - LEARNING                                        │
│  UTL optimizer, Neuromodulation controller                      │
│  Update frequency: 100Hz                                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 (L3) - MEMORY                                          │
│  Modern Hopfield Networks, FAISS GPU, HNSW indexes              │
│  Latency: <1ms, Capacity: 2^768 patterns                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 (L2) - REFLEX                                          │
│  Modern Hopfield Network cache (bypass when confidence > 0.95)  │
│  Latency: <100μs, Hit rate: >80%                                │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1 (L1) - SENSING                                         │
│  13-model embedding pipeline, PII scrubbing, UTL entropy        │
│  Latency: <5ms, Throughput: 10K/s                               │
└─────────────────────────────────────────────────────────────────┘
```

### 13-Embedding Teleological Fingerprint

Each memory is stored as a **TeleologicalFingerprint** containing 13 specialized embeddings:

| ID | Name | Dimension | Purpose | Brain Band |
|----|------|-----------|---------|------------|
| E1 | Semantic | 1024 (Matryoshka: 512/256/128) | General semantic meaning | Gamma (40Hz) |
| E2 | Temporal Recent | 512 | Exponential recency decay | Alpha (8Hz) |
| E3 | Temporal Periodic | 512 | Fourier periodicity patterns | Alpha (8Hz) |
| E4 | Temporal Positional | 512 | Sinusoidal position encoding | Alpha (8Hz) |
| E5 | Causal | 768 | Structural causal models (asymmetric) | Beta (25Hz) |
| E6 | Sparse Lexical | ~30K sparse | SPLADE keyword precision | Theta (4Hz) |
| E7 | Code | 1536 | AST-aware code understanding | Beta (25Hz) |
| E8 | Graph | 384 | Structural relationships | Alpha-Beta |
| E9 | HDC | 1024 | Hyperdimensional robustness | High-Gamma (80Hz) |
| E10 | Multimodal | 768 | CLIP cross-modal alignment | Gamma (40Hz) |
| E11 | Entity | 384 | Factual entity representation | Beta (25Hz) |
| E12 | Late Interaction | 128/token | ColBERT token precision | High-Gamma (60Hz) |
| E13 | SPLADE v3 | ~30K sparse | Enhanced lexical matching | Theta (4Hz) |

**Storage**: ~17KB quantized per memory (63% reduction from uncompressed)

### Retrieval Pipeline

5-stage pipeline achieving <60ms @ 1M memories:

```
Stage 1: SPARSE PRE-FILTER ──────────────────────────> 1M → 10K candidates
         BM25 + SPLADE, <5ms

Stage 2: FAST DENSE ANN ─────────────────────────────> 10K → 1K candidates
         Matryoshka 128D HNSW, <10ms

Stage 3: MULTI-SPACE RERANK ─────────────────────────> 1K → 100 candidates
         RRF Fusion across spaces, <20ms

Stage 4: TELEOLOGICAL ALIGNMENT ─────────────────────> 100 → 50 candidates
         Purpose vector filtering, <10ms

Stage 5: LATE INTERACTION RERANK ────────────────────> 50 → 10 results
         ColBERT MaxSim scoring, <15ms
```

---

## MCP Tools Reference

Context Graph exposes **23 MCP tools** organized into 9 categories. All tools return a `_cognitive_pulse` field with live UTL metrics.

### 1. Core Memory Tools

#### `inject_context`

Inject context into the knowledge graph with full UTL processing. This is the **primary context injection method**.

```json
{
  "content": "string (required) - The content to inject",
  "rationale": "string (required) - Why this context is relevant",
  "modality": "text|code|image|audio|structured|mixed (default: text)",
  "importance": "number 0-1 (default: 0.5)"
}
```

**Returns:**
- `fingerprintId`: UUID of stored memory
- `rationale`: Echo of provided rationale
- `embedderCount`: 13
- `embeddingLatencyMs`: Processing time
- `utl`: Learning metrics (learningScore, entropy, coherence, surprise)

**Usage:** Use for all important context that should influence learning. Requires North Star goal to be configured.

#### `store_memory`

Store a memory directly without UTL processing. Faster but bypasses learning calculations.

```json
{
  "content": "string (required) - The content to store",
  "importance": "number 0-1 (default: 0.5)",
  "modality": "text|code|image|audio|structured|mixed (default: text)",
  "tags": "string[] (optional) - Categorization tags"
}
```

**Returns:** `fingerprintId`, `embedderCount`, `embeddingLatencyMs`

**Usage:** Use for bulk imports or when speed matters more than learning.

---

### 2. System Status Tools

#### `get_memetic_status`

Get comprehensive system status with live UTL metrics.

**Parameters:** None

**Returns:**
```json
{
  "phase": "infancy|growth|maturity",
  "fingerprintCount": 1234,
  "embedderCount": 13,
  "storageBackend": "rocksdb",
  "storageSizeBytes": 52428800,
  "quadrantCounts": {
    "open": 100,
    "hidden": 50,
    "blind": 30,
    "unknown": 20
  },
  "utl": {
    "entropy": 0.65,
    "coherence": 0.78,
    "learningScore": 0.72,
    "johariQuadrant": "open",
    "consolidationPhase": "active",
    "suggestedAction": "continue learning"
  },
  "layers": {
    "perception": "healthy",
    "memory": "healthy",
    "reasoning": "healthy",
    "action": "healthy",
    "meta": "healthy"
  }
}
```

**Usage:** Call periodically to monitor system health and learning progress.

#### `get_graph_manifest`

Get the 5-layer architecture description and configuration.

**Parameters:** None

**Returns:** Architecture overview, fingerprint type, embedder count, layer descriptions, UTL configuration.

**Usage:** Call once at startup to understand system capabilities.

#### `utl_status`

Query current UTL (Unified Theory of Learning) state.

**Parameters:** None

**Returns:**
- `lifecycle_phase`: Current learning phase
- `entropy`: Novelty measure (high = new information)
- `coherence`: Integration measure (high = consistent)
- `learning_score`: Combined learning signal
- `johari_quadrant`: Current awareness state
- `consolidation_phase`: Memory consolidation status

**Usage:** Monitor learning dynamics and decide when to inject new context.

---

### 3. Search Tools

#### `search_graph`

Semantic search across the knowledge graph using all 13 embedding spaces.

```json
{
  "query": "string (required) - Search query",
  "topK": "integer 1-100 (default: 10)",
  "minSimilarity": "number 0-1 (default: 0)",
  "modality": "text|code|image|audio|structured|mixed (optional)"
}
```

**Returns:**
```json
{
  "results": [
    {
      "fingerprintId": "uuid",
      "similarity": 0.89,
      "purposeAlignment": 0.75,
      "dominantEmbedder": "E1_Semantic",
      "thetaToNorthStar": 0.82
    }
  ],
  "count": 10
}
```

**Usage:** Primary retrieval method. Uses the full 5-stage pipeline.

---

### 4. GWT Consciousness Tools

These tools implement Global Workspace Theory for computational consciousness.

#### `get_consciousness_state`

Get the current consciousness state including synchronization and identity.

```json
{
  "session_id": "string (optional)"
}
```

**Returns:**
```json
{
  "C": 0.75,
  "r": 0.82,
  "psi": 1.57,
  "meta_score": 0.68,
  "differentiation": 0.71,
  "integration": 0.82,
  "reflection": 0.68,
  "state": "CONSCIOUS",
  "gwt_state": "ACTIVE",
  "workspace": {
    "active_memory": "uuid or null",
    "is_broadcasting": true,
    "has_conflict": false,
    "coherence_threshold": 0.8
  },
  "identity": {
    "coherence": 0.91,
    "status": "Healthy",
    "trajectory_length": 150,
    "purpose_vector": [0.1, 0.2, ...]
  },
  "component_analysis": {
    "integration_sufficient": true,
    "reflection_sufficient": true,
    "differentiation_sufficient": true,
    "limiting_factor": null
  }
}
```

**Consciousness States:**
- `DORMANT`: r < 0.3 (no synchronization)
- `FRAGMENTED`: 0.3 ≤ r < 0.5 (partial sync)
- `EMERGING`: 0.5 ≤ r < 0.8 (approaching consciousness)
- `CONSCIOUS`: r ≥ 0.8 (full synchronization)
- `HYPERSYNC`: r > 0.95 (potentially pathological)

**Usage:** Monitor system "awareness" level and identity coherence.

#### `get_kuramoto_sync`

Get detailed Kuramoto oscillator network state.

```json
{
  "session_id": "string (optional)"
}
```

**Returns:**
```json
{
  "r": 0.82,
  "psi": 1.57,
  "synchronization": 0.82,
  "state": "CONSCIOUS",
  "phases": [0.5, 0.6, 0.55, ...],
  "natural_freqs": [1.58, 0.32, 0.32, ...],
  "coupling": 0.5,
  "elapsed_seconds": 3600,
  "embedding_labels": ["E1_Semantic", "E2_TempRecent", ...],
  "thresholds": {
    "conscious": 0.8,
    "fragmented": 0.5,
    "hypersync": 0.95
  }
}
```

**Usage:** Deep diagnosis of embedding space synchronization.

#### `get_workspace_status`

Get Global Workspace winner-take-all status.

```json
{
  "session_id": "string (optional)"
}
```

**Returns:**
```json
{
  "active_memory": "uuid or null",
  "is_broadcasting": true,
  "has_conflict": false,
  "coherence_threshold": 0.8,
  "conflict_memories": [],
  "broadcast_duration_ms": 100
}
```

**Usage:** Understand what memory currently holds "conscious attention."

#### `get_ego_state`

Get the Self-Ego Node state (system identity).

```json
{
  "session_id": "string (optional)"
}
```

**Returns:**
```json
{
  "purpose_vector": [0.1, 0.2, ...],
  "identity_coherence": 0.91,
  "coherence_with_actions": 0.85,
  "identity_status": "Healthy",
  "trajectory_length": 150,
  "thresholds": {
    "healthy": 0.9,
    "warning": 0.7,
    "degraded": 0.5,
    "critical": 0.0
  }
}
```

**Identity Status:**
- `Healthy`: coherence ≥ 0.9
- `Warning`: 0.7 ≤ coherence < 0.9
- `Degraded`: 0.5 ≤ coherence < 0.7
- `Critical`: coherence < 0.5

**Usage:** Monitor system identity stability over time.

#### `trigger_workspace_broadcast`

Force a specific memory into the Global Workspace.

```json
{
  "memory_id": "uuid (required)",
  "importance": "number 0-1 (default: 0.8)",
  "alignment": "number 0-1 (default: 0.8)",
  "force": "boolean (default: false)"
}
```

**Returns:**
```json
{
  "success": true,
  "memory_id": "uuid",
  "new_r": 0.85,
  "was_selected": true,
  "is_broadcasting": true
}
```

**Usage:** Manually bring important memories to "conscious attention."

#### `adjust_coupling`

Adjust Kuramoto network coupling strength (affects synchronization speed).

```json
{
  "new_K": "number 0-10 (required)"
}
```

**Returns:**
```json
{
  "old_K": 0.5,
  "new_K": 0.7,
  "predicted_r": 0.88,
  "current_r": 0.82,
  "K_clamped": false
}
```

**Usage:** Increase K for faster sync, decrease for more independent processing.

---

### 5. Adaptive Threshold Calibration (ATC) Tools

#### `get_threshold_status`

Get current ATC status across all 4 calibration levels.

```json
{
  "domain": "Code|Medical|Legal|Creative|Research|General (default: General)",
  "embedder_id": "integer 1-13 (optional)"
}
```

**Returns:** Per-embedder temperatures, drift scores, bandit exploration stats.

**Usage:** Diagnose threshold adaptation behavior.

#### `get_calibration_metrics`

Get calibration quality metrics (ECE, MCE, Brier Score).

```json
{
  "timeframe": "1h|24h|7d|30d (default: 24h)"
}
```

**Returns:**
```json
{
  "ECE": 0.04,
  "MCE": 0.08,
  "brierScore": 0.12,
  "calibrationStatus": "excellent",
  "driftScores": {...}
}
```

**Calibration Targets:**
- `ECE < 0.05`: Excellent
- `ECE < 0.10`: Good
- `ECE ≥ 0.10`: Needs recalibration

**Usage:** Monitor threshold accuracy and trigger recalibration if needed.

#### `trigger_recalibration`

Manually trigger recalibration at a specific level.

```json
{
  "level": "integer 1-4 (required)",
  "domain": "Code|Medical|Legal|Creative|Research|General (default: General)"
}
```

**Levels:**
1. **EWMA**: Drift adjustment (fast)
2. **Temperature**: Per-embedder scaling (hourly)
3. **Thompson Sampling**: Bandit exploration (session)
4. **Bayesian**: Meta-optimization (weekly)

**Returns:** New thresholds, observation count used.

**Usage:** Force recalibration when ECE exceeds acceptable limits.

---

### 6. Dream Consolidation Tools

#### `trigger_dream`

Manually trigger a dream consolidation cycle.

```json
{
  "force": "boolean (default: false)"
}
```

**Requirements:**
- System activity < 0.15
- GPU usage < 30%
- Or `force: true` to bypass

**Dream Cycle:**
1. **NREM Phase** (3 min): Tight coupling, recency-biased replay
2. **REM Phase** (2 min): Temperature=2.0, attractor exploration

**Returns:** DreamReport with consolidation metrics.

**Usage:** Trigger during idle periods to optimize memory structure.

#### `get_dream_status`

Get current dream system state.

**Parameters:** None

**Returns:**
```json
{
  "state": "Awake|NREM|REM|Waking",
  "gpu_usage": 0.15,
  "activity_level": 0.08,
  "time_since_last_dream": 3600
}
```

**Usage:** Check if system is ready for dreaming.

#### `abort_dream`

Abort current dream cycle (emergency wake).

**Parameters:** None

**Constitution Mandate:** Wake latency must be < 100ms.

**Returns:**
```json
{
  "wake_latency": 45,
  "partial_report": {...}
}
```

**Usage:** Abort dreams when urgent queries arrive.

#### `get_amortized_shortcuts`

Get shortcut candidates from amortized learning.

```json
{
  "min_confidence": "number 0-1 (default: 0.7)",
  "limit": "integer 1-100 (default: 20)"
}
```

**Shortcut Criteria:**
- Path length ≥ 3 hops
- Traversed ≥ 5 times
- Confidence ≥ 0.7

**Returns:** List of paths eligible for direct edge creation.

**Usage:** Inspect learned shortcuts for graph optimization.

---

### 7. Neuromodulation Tools

#### `get_neuromodulation_state`

Get all 4 neuromodulator levels.

**Parameters:** None

**Returns:**
```json
{
  "dopamine": {
    "value": 3.0,
    "range": [1, 5],
    "controls": "hopfield.beta",
    "effect": "Sharp retrieval (high) vs Diffuse (low)"
  },
  "serotonin": {
    "value": 0.6,
    "range": [0, 1],
    "controls": "space_weights",
    "effect": "Broad consideration (high) vs Narrow focus (low)"
  },
  "noradrenaline": {
    "value": 1.0,
    "range": [0.5, 2],
    "controls": "attention.temp",
    "effect": "Flat attention (high) vs Peaked (low)"
  },
  "acetylcholine": {
    "value": 0.0015,
    "range": [0.001, 0.002],
    "controls": "utl.lr",
    "effect": "Faster learning (high)",
    "readonly": true
  }
}
```

**Usage:** Monitor system's emotional/arousal state.

#### `adjust_neuromodulator`

Adjust a neuromodulator level (except ACh which is GWT-managed).

```json
{
  "modulator": "dopamine|serotonin|noradrenaline (required)",
  "delta": "number (required)"
}
```

**Returns:**
```json
{
  "old_value": 3.0,
  "new_value": 3.5,
  "clamped": false
}
```

**Usage Examples:**
- Increase dopamine for more focused retrieval
- Increase serotonin for broader search diversity
- Increase noradrenaline for threat/urgency response

---

### 8. Steering Tools

#### `get_steering_feedback`

Get combined feedback from Gardener, Curator, and Assessor.

**Parameters:** None

**Returns:**
```json
{
  "steering_reward": 0.45,
  "gardener": {
    "score": 0.6,
    "metrics": {
      "orphan_count": 12,
      "pruning_rate": 0.02,
      "connectivity": 0.85
    }
  },
  "curator": {
    "score": 0.4,
    "metrics": {
      "average_quality": 0.72,
      "low_quality_count": 45
    }
  },
  "assessor": {
    "score": 0.35,
    "metrics": {
      "retrieval_accuracy": 0.78,
      "learning_efficiency": 0.65
    }
  },
  "recommendations": [
    "Prune low-quality memories",
    "Increase coherence filtering"
  ]
}
```

**Score Interpretation:**
- `[-1, -0.5]`: Poor (needs intervention)
- `[-0.5, 0]`: Below average
- `[0, 0.5]`: Good
- `[0.5, 1]`: Excellent

**Usage:** Guide system optimization and memory curation decisions.

---

### 9. Causal Inference Tools

#### `omni_infer`

Perform omni-directional causal inference.

```json
{
  "source": "uuid (required)",
  "target": "uuid (required for forward/backward/bidirectional)",
  "direction": "forward|backward|bidirectional|bridge|abduction (default: forward)"
}
```

**Inference Directions:**
| Direction | Description | Example Question |
|-----------|-------------|------------------|
| `forward` | A → B effect | "What happens if I do X?" |
| `backward` | B → A cause | "Why did Y happen?" |
| `bidirectional` | A ↔ B mutual | "How do these interact?" |
| `bridge` | Cross-domain | "How does physics affect economics?" |
| `abduction` | Best hypothesis | "What explains this observation?" |

**Returns:** Causal inference results including strength, confidence, and reasoning path.

**Usage:** Understand cause-effect relationships between memories.

---

## Cognitive Systems

### Global Workspace Theory (GWT)

Implements Bernard Baars' theory of consciousness where information "wins" access to a central workspace and is broadcast to all subsystems.

**Consciousness Equation:**
```
C(t) = I(t) × R(t) × D(t)

Where:
  I(t) = Integration (Kuramoto order parameter r)
  R(t) = Self-Reflection (Meta-UTL prediction accuracy)
  D(t) = Differentiation (Shannon entropy of purpose vector)
```

### Kuramoto Oscillator Network

13 coupled phase oscillators modeling neural synchronization:

```
dθᵢ/dt = ωᵢ + (K/N) × Σⱼ sin(θⱼ - θᵢ)

Order parameter: r × e^(iψ) = (1/N) × Σⱼ e^(iθⱼ)
```

When r → 1, all embedders are synchronized (CONSCIOUS state).

### Unified Theory of Learning (UTL)

Learning signal computation:

```
L = sigmoid(2.0 × (Σᵢ τᵢ × λₛ × ΔSᵢ) × (Σⱼ τⱼ × λc × ΔCⱼ) × wₑ × cos(φ))

Where:
  ΔS = Entropy/novelty change
  ΔC = Coherence/integration change
  τ = Teleological weight (purpose alignment)
  wₑ = Emotional modulation [0.5, 1.5]
  φ = Phase synchronization
```

**Lifecycle Phases:**
| Phase | Interactions | λₛ | λc | Focus |
|-------|--------------|-----|-----|-------|
| Infancy | 0-50 | 0.7 | 0.3 | Capture novelty |
| Growth | 50-500 | 0.5 | 0.5 | Balanced |
| Maturity | 500+ | 0.3 | 0.7 | Coherence curation |

### Neuromodulation System

| Modulator | Range | Controls | Biological Role |
|-----------|-------|----------|-----------------|
| Dopamine | [1, 5] | Hopfield beta | Reward/salience |
| Serotonin | [0, 1] | Space weights | Mood/diversity |
| Noradrenaline | [0.5, 2] | Attention temp | Alertness/focus |
| Acetylcholine | [0.001, 0.002] | UTL learning rate | Learning (GWT-managed) |

### Dream Consolidation System

Two-phase sleep cycle inspired by neuroscience:

| Phase | Duration | Purpose | Parameters |
|-------|----------|---------|------------|
| NREM | 3 min | Replay recent memories | coupling=tight, recency=0.8 |
| REM | 2 min | Explore attractors | temp=2.0, exploration |

**Amortized Learning:** Paths traversed 5+ times with 3+ hops create direct edges.

### Adaptive Threshold Calibration (ATC)

Four-level self-tuning architecture:

| Level | Method | Frequency | Purpose |
|-------|--------|-----------|---------|
| 1 | EWMA Drift | Per-query | Detect distribution shift |
| 2 | Temperature Scaling | Hourly | Per-embedder calibration |
| 3 | Thompson Sampling | Session | Exploration vs exploitation |
| 4 | Bayesian Meta-Opt | Weekly | Global optimization |

---

## Installation

### Prerequisites

- Rust 1.75+ (stable toolchain)
- CUDA 13.1+ with RTX 5090 or compatible Blackwell GPU
- ~10GB disk space for models
- RocksDB system library

### Build from Source

```bash
# Clone the repository
git clone https://github.com/your-org/contextgraph.git
cd contextgraph

# Build (requires CUDA and candle features)
cargo build --release

# Download required models (from HuggingFace)
./scripts/download_models.sh
```

### Run MCP Server

```bash
# Run the MCP server (stdio transport)
cargo run --release -p context-graph-mcp

# Or specify configuration
CONTEXT_GRAPH_CONFIG=./config/production.toml cargo run --release -p context-graph-mcp
```

### Claude Desktop Integration

Add to your Claude Desktop config (`~/.config/claude/mcp.json`):

```json
{
  "mcpServers": {
    "context-graph": {
      "command": "/path/to/context-graph-mcp",
      "env": {
        "CONTEXT_GRAPH_CONFIG": "/path/to/config/production.toml"
      }
    }
  }
}
```

---

## Configuration

### Configuration Files

| File | Purpose |
|------|---------|
| `config/default.toml` | Base configuration |
| `config/development.toml` | Development overrides |
| `config/production.toml` | Production settings |
| `config/test.toml` | Test configuration |

### Key Configuration Options

```toml
[storage]
backend = "rocksdb"
path = "./data/contextgraph"

[embeddings]
batch_size = 32
cache_size = 1000

[gwt]
coherence_threshold = 0.8
broadcast_duration_ms = 100

[kuramoto]
coupling_strength = 0.5

[dream]
activity_threshold = 0.15
idle_trigger_secs = 600

[atc]
ewma_alpha = 0.1
temperature_update_interval = 3600
```

---

## Optimal Usage Patterns

### 1. Initial Setup

```
1. Call get_graph_manifest to understand capabilities
2. Call get_memetic_status to check system state
3. Configure North Star goal via purpose operations
```

### 2. Context Injection Flow

```
For important context (want learning):
  → inject_context with rationale

For bulk imports (speed priority):
  → store_memory without UTL processing
```

### 3. Retrieval Pattern

```
1. search_graph with query
2. Check results.purposeAlignment
3. If low alignment, adjust search or North Star
```

### 4. Monitoring Consciousness

```
Periodic checks:
  → get_consciousness_state (overall health)
  → get_kuramoto_sync (if C is low)
  → get_ego_state (identity drift)
```

### 5. Optimization Cycle

```
1. get_steering_feedback
2. If score < 0:
   - get_calibration_metrics
   - trigger_recalibration if ECE > 0.10
3. If idle + activity < 0.15:
   - trigger_dream
```

### 6. Neuromodulation Tuning

```
For focused retrieval:
  → adjust_neuromodulator("dopamine", +1.0)

For broad exploration:
  → adjust_neuromodulator("serotonin", +0.2)

For urgent situations:
  → adjust_neuromodulator("noradrenaline", +0.5)
```

---

## API Reference

### JSON-RPC Methods

Beyond the 23 tools, the MCP server supports these JSON-RPC methods:

#### Lifecycle
- `initialize` - MCP handshake
- `notifications/initialized` - Client ready
- `shutdown` - Graceful shutdown

#### Memory
- `memory/store`, `memory/retrieve`, `memory/delete`
- `memory/list`, `memory/count`, `memory/exists`
- `memory/batch_store`, `memory/batch_retrieve`

#### Search
- `search/semantic` - Full 13-embedder search
- `search/purpose` - Purpose vector search
- `search/hybrid` - Combined search
- `search/neighborhood` - Find neighbors

#### Purpose
- `purpose/query` - Query by purpose vector
- `purpose/drift_check` - Detect alignment drift (works autonomously without North Star)
- ~~`purpose/north_star_alignment`~~ - (Deprecated - use `purpose/drift_check`)
- ~~`purpose/north_star_update`~~ - (Deprecated - use `auto_bootstrap_north_star`)

#### Johari
- `johari/get_distribution` - Per-embedder quadrants
- `johari/find_by_quadrant` - Find by quadrant
- `johari/transition` - Execute transition
- `johari/cross_space_analysis` - Blind spot analysis

#### Meta-UTL
- `meta_utl/learning_trajectory` - Learning curves
- `meta_utl/health_metrics` - System health
- `meta_utl/predict_storage` - Storage impact prediction
- `meta_utl/predict_retrieval` - Retrieval quality prediction

---

## Project Structure

```
contextgraph/
├── crates/
│   ├── context-graph-mcp/      # MCP server (entry point)
│   ├── context-graph-core/     # Domain types, GWT, ATC
│   ├── context-graph-embeddings/ # 13-model pipeline
│   ├── context-graph-storage/  # RocksDB persistence
│   ├── context-graph-graph/    # Knowledge graph
│   ├── context-graph-utl/      # Learning engine
│   └── context-graph-cuda/     # GPU acceleration
├── config/                     # Configuration files
├── models/                     # Pre-trained models
├── docs2/                      # Design documents
│   ├── constitution.yaml       # System constitution
│   ├── contextprd.md          # Product requirements
│   └── brain.md               # MCP usage guide
└── scripts/                    # Utility scripts
```

---

## License

[License information]

## Contributing

[Contributing guidelines]

## Acknowledgments

- Global Workspace Theory: Bernard Baars
- Kuramoto Model: Yoshiki Kuramoto
- Modern Hopfield Networks: Hubert Ramsauer et al.
- SPLADE: Thibault Formal et al.
- ColBERT: Omar Khattab & Matei Zaharia
