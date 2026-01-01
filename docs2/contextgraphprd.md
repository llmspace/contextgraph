# Context Graph - Product Requirements Document

**Version**: 2.0.0 | **Status**: Draft

---

## Executive Summary

Context Graph is a bio-nervous inspired knowledge system that provides AI agents with persistent, associative memory. It implements a 5-layer neural architecture using the Unified Theory of Learning (UTL) for optimal knowledge retrieval and consolidation, enabling agents to learn, remember, and reason across sessions.

---

## 1. Vision & Core Concepts

### 1.1 Problem Statement

AI agents suffer from:
- **No persistent memory** - Context lost between sessions
- **Poor retrieval** - Standard vector search misses semantic relationships
- **No learning loop** - No feedback mechanism for knowledge quality
- **Context bloat** - Agents don't know how to compress/distill retrieved information

### 1.2 Solution Overview

A brain-inspired memory system where:
- **Passive capture** stores conversation automatically via host hooks
- **Active curation** lets agents refine quality (merge, annotate, forget)
- **Dream consolidation** discovers novel connections during idle periods
- **UTL optimization** continuously improves retrieval quality

### 1.3 Unified Theory of Learning (UTL)

The core learning equation:
```
L = f((ΔS × ΔC) × wₑ × cos φ)

Where:
- ΔS: Entropy change (novelty/surprise) ∈ [0,1]
- ΔC: Coherence change (understanding) ∈ [0,1]
- wₑ: Emotional modulation weight ∈ [0.5, 1.5]
- φ: Phase synchronization angle ∈ [0, π]
```

**Johari Window Classification**:
| ΔS | ΔC | Quadrant | Meaning |
|----|----|----|---------|
| Low | High | Open | Known, confident |
| High | Low | Blind | Confused, uncertain |
| Low | Low | Hidden | Stale, drifting |
| High | High | Unknown | Novel, adapting |

---

## 2. System Architecture

### 2.1 5-Layer Bio-Nervous System

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: COHERENCE - Global consistency, Thalamic Gate    │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: LEARNING - UTL optimization loop                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: MEMORY - Modern Hopfield Networks                │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: REFLEX - Sub-millisecond cached responses        │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: SENSING - Multi-modal input processing           │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Function | Latency Target | Key Component |
|-------|----------|----------------|---------------|
| Sensing | Input normalization | <5ms | Embedding Pipeline |
| Reflex | Pattern-matched responses | <100μs | Hopfield Cache |
| Memory | Associative storage | <1ms | Modern Hopfield |
| Learning | Weight optimization | 10ms | UTL Optimizer |
| Coherence | Global consistency | 10ms | Thalamic Gate |

### 2.2 12-Model Embedding Architecture

| ID | Model | Dimension | Purpose |
|----|-------|-----------|---------|
| E1 | Semantic | 1024D | Dense meaning |
| E2-E4 | Temporal | 512D × 3 | Recency, periodicity, position |
| E5 | Causal | 768D | Cause-effect relationships |
| E6 | Sparse | ~30K | Top-K activation patterns |
| E7 | Code | 1536D | AST-aware programming |
| E8 | Graph/GNN | 1536D | Structural relationships |
| E9 | HDC | 10K-bit | Hyperdimensional computing |
| E10 | Multimodal | 1024D | Images, audio (CLIP/SigLIP) |
| E11 | Entity | 256D | Knowledge graph triples |
| E12 | Late-Interaction | 128D/token | ColBERT-style retrieval |

**Fusion**: FuseMoE (Mixture of Experts) with Laplace-smoothed top-k routing combines embeddings into unified 1536D representation.

### 2.3 Knowledge Graph Structure

```rust
KnowledgeNode {
    id: Uuid,
    content: String,
    embedding: Vector1536,
    importance: f32,              // [0, 1]
    johari_quadrant: Quadrant,
    utl_state: UTLState,
    agent_id: Option<String>,     // Multi-agent provenance
    priors_vibe_check: Vector128, // Memetic drift prevention
}

GraphEdge {
    source: Uuid,
    target: Uuid,
    edge_type: Semantic | Temporal | Causal | Hierarchical | Relational,
    weight: f32,
    neurotransmitter: Option<NeurotransmitterWeights>, // Domain modulation
    is_amortized_shortcut: bool,  // Dream-created shortcuts
}
```

---

## 3. Core Features

### 3.1 Context Injection (`inject_context`)

Primary retrieval tool with automatic distillation:

**Parameters**:
- `query`: Natural language query
- `max_tokens`: Token budget (default: 2048)
- `distillation_mode`: auto | raw | narrative | structured | code_focused
- `verbosity_level`: 0 (~100 tokens) | 1 (~200, default) | 2 (~800, full insights)
- `include_metadata`: Pre-emptive bundling (causal_links, entailment_cones, etc.)

**Response includes**:
- `context`: Distilled text
- `utl_metrics`: Current entropy/coherence
- `conflict_alert`: Semantic conflict detection
- `tool_gating_warning`: When entropy > 0.8 (suggests refining query first)

### 3.2 Cognitive Pulse (Meta-Cognitive Loop)

Every MCP response includes a mandatory Pulse header:
```
Pulse: { Entropy: 0.72, Coherence: 0.41, Suggested: "trigger_dream" }
```

| Entropy | Coherence | Suggested Action |
|---------|-----------|------------------|
| > 0.7 | > 0.5 | `epistemic_action` |
| > 0.7 | < 0.4 | `trigger_dream` |
| < 0.4 | > 0.7 | Continue working |
| < 0.4 | < 0.4 | `get_neighborhood` |

### 3.3 Dream Layer (Offline Consolidation)

Bio-inspired memory consolidation during idle periods:

**Phases**:
1. **NREM (3 min)**: Replay recent memories, Hebbian weight updates
2. **REM (2 min)**: Synthetic queries, discover blind spots, create edges
3. **Amortize (1 min)**: Create shortcut edges from multi-hop paths

**Triggers**: Activity < 15% for 10+ minutes
**Interrupt**: Instant wake on user query (<100ms)

### 3.4 Memory Curation Tools

| Tool | Purpose |
|------|---------|
| `merge_concepts` | Combine duplicate nodes (checks priors compatibility) |
| `annotate_node` | Add corrections, references, deprecation notes |
| `forget_concept` | Soft-delete (30-day recovery) or permanent removal |
| `boost_importance` | Mark content as critical |

**Priors Vibe Check**: Prevents memetic drift during merges. If nodes have incompatible priors (cosine sim < 0.7), creates Relational Edge instead: "In Python, X; but in Java, Y"

### 3.5 Neuromodulation (Global State Control)

Dynamic parameter adjustment based on UTL state:

| Modulator | Maps To | Effect |
|-----------|---------|--------|
| Dopamine | `hopfield.beta` | Retrieval sharpness |
| Serotonin | `fuse_moe.top_k` | Expert diversity |
| Noradrenaline | `attention.temp` | Attention distribution |
| Acetylcholine | `learning_rate` | Memory plasticity |

### 3.6 Active Inference (Epistemic Actions)

System-generated clarifying questions when coherence < 0.4:
```json
{
  "action_type": "ask_user",
  "question": "What causes X to lead to Y?",
  "expected_entropy_reduction": 0.35
}
```

---

## 4. MCP Interface

### 4.1 Protocol

```json
{
  "protocol": "JSON-RPC 2.0",
  "version": "2024-11-05",
  "transport": ["stdio", "sse"],
  "capabilities": { "tools": true, "resources": true, "prompts": true }
}
```

### 4.2 Core Tools

| Tool | Purpose |
|------|---------|
| `inject_context` | Retrieve & distill relevant context |
| `search_graph` | Vector similarity search with perspective filtering |
| `store_memory` | Store new knowledge (requires rationale) |
| `get_memetic_status` | Dashboard: entropy, coherence, curation tasks |
| `get_graph_manifest` | Meta-cognitive system prompt fragment |
| `trigger_dream` | Manual dream consolidation |
| `query_causal` | Causal relationship queries |
| `epistemic_action` | System-generated clarifying questions |

### 4.3 Curation Tools

| Tool | Purpose |
|------|---------|
| `merge_concepts` | Combine semantically identical nodes |
| `annotate_node` | Add marginalia (corrections, deprecation) |
| `forget_concept` | Soft-delete or permanent removal |
| `boost_importance` | Increase node importance |

### 4.4 Navigation Tools

| Tool | Purpose |
|------|---------|
| `get_neighborhood` | Browse local graph topology |
| `get_recent_context` | Temporal navigation |
| `find_causal_path` | Direct A→B path finding |
| `entailment_query` | Hyperbolic hierarchy queries |
| `generate_search_plan` | Query synthesis from goals |

### 4.5 Diagnostic Tools

| Tool | Purpose |
|------|---------|
| `homeostatic_status` | Graph health metrics |
| `check_adversarial` | Scan for attacks before storage |
| `get_system_logs` | See why nodes were pruned/quarantined |
| `critique_context` | Self-contradiction detection |
| `hydrate_citation` | Expand citation tags to raw content |
| `get_node_lineage` | Node evolution history |

### 4.6 Resources

| URI | Purpose |
|-----|---------|
| `context://{scope}` | Current context state |
| `graph://{node_id}` | Specific graph nodes |
| `utl://{session}/state` | UTL learning state |
| `utl://current_session/pulse` | Subscribable cognitive pulse |
| `admin://manifest` | Human-editable knowledge manifest |
| `visualize://{scope}/{topic}` | Graph visualization (Mermaid/D3) |

---

## 5. System Lifecycle

The system adapts thresholds based on graph maturity:

| Stage | Interactions | UTL Thresholds | Lambda Weights | Stance |
|-------|--------------|----------------|----------------|--------|
| Infancy | 0-50 | entropy: 0.9, coherence: 0.2 | λ_ΔS: 0.7, λ_ΔC: 0.3 | Capture-heavy |
| Growth | 50-500 | entropy: 0.7, coherence: 0.4 | λ_ΔS: 0.5, λ_ΔC: 0.5 | Balanced |
| Maturity | 500+ | entropy: 0.6, coherence: 0.5 | λ_ΔS: 0.3, λ_ΔC: 0.7 | Curation-heavy |

**Synthetic Data Seeding**: Bootstrap empty graphs with project documentation:
```bash
bin/reasoning seed --source ./README.md --target-nodes 200
```

---

## 6. Multi-Agent Safety

### 6.1 Perspective Filtering

Prevents cross-agent "memetic interference":
```json
{
  "perspective_lock": {
    "domain": "code",
    "exclude_agent_ids": ["creative-writer-agent"]
  }
}
```

### 6.2 Priors Vibe Check

128D vector encoding agent assumptions at storage time. During merge:
- If compatible (sim > 0.7): Normal merge
- If incompatible: Create Relational Edge instead

### 6.3 Scratchpad Isolation

Per-session, per-agent scratchpads with privacy controls:
- `private`: Only this agent
- `team`: Agents in same session
- `shared`: All agents with graph access

---

## 7. Security & Defense

### 7.1 Adversarial Detection

| Layer | Attack Type | Detection | Response |
|-------|-------------|-----------|----------|
| Input | Prompt injection | Regex + semantic | Block + log |
| Embedding | Space manipulation | Outlier detection | Quarantine |
| Graph | Circular logic | Cycle detection | Prune edges |
| Output | Data exfiltration | Content filter | Redact |

### 7.2 Homeostatic Plasticity

- Importance scaling prevents runaway excitation
- Semantic cancer detection (high importance + high neighbor entropy)
- Quarantine mechanism preserves nodes for review

### 7.3 PII Scrubbing

Layer 1 pre-processing with NER before embedding pipeline:
- API keys, passwords, SSN, credit cards
- Pattern matching (<1ms) + NER (<100ms)

---

## 8. Steering Subsystem (Dopamine Feedback)

The system provides reward signals to teach agents what to store:

```
Agent stores node → Steering assesses → Dopamine signal returned
      ↑                                              │
      └──────────── Agent adjusts behavior ──────────┘
```

**Lifecycle-Aware Rewards**:
- **Infancy**: +reward for high ΔS (novelty)
- **Growth**: Balanced ΔS/ΔC
- **Maturity**: +reward for high ΔC (coherence)

**Universal Penalties**:
- Near-duplicate storage: -0.4
- Low priors confidence: -0.3
- Missing rationale: -0.5

---

## 9. Human-in-the-Loop

### 9.1 Knowledge Manifest

Human-editable markdown at `~/.context-graph/manifest.md`:
```markdown
## Active Concepts
- UserAuthentication
- JWTTokenValidation

## Pending Actions
[MERGE: JWTTokenValidation, OAuth2Validation]

## Notes
[NOTE: RateLimiting] Deprecated in v2.0
```

### 9.2 Visualization

`visualize://topic/authentication` returns Mermaid/D3.js diagram for human review.

### 9.3 Undo Log

Every merge/forget generates `reversal_hash` for one-click rollback (30-day retention).

---

## 10. Performance Targets

### 10.1 Latency

| Operation | Target |
|-----------|--------|
| Context Injection P95 | <25ms |
| Context Injection P99 | <50ms |
| Hopfield Retrieval | <1ms |
| Vector Search (FAISS) | <2ms |
| Dream Wake | <100ms |

### 10.2 Quality

| Metric | Target |
|--------|--------|
| UTL Score Average | >0.6 |
| Reflex Cache Hit | >80% |
| Distillation Information Loss | <15% |
| Compression Ratio | >60% |

### 10.3 Scale

| Metric | Target |
|--------|--------|
| Memory Capacity | >10M nodes |
| Batch Throughput | >1000/sec |
| GPU Memory Usage | <24GB |

---

## 11. Hardware Requirements

### 11.1 Target Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5090 (Blackwell) |
| VRAM | 32GB GDDR7 |
| CUDA | 13.1 |
| Storage | NVMe Gen5 (GPU Direct Storage) |

### 11.2 CUDA Features

- **Green Contexts**: SM partitioning for concurrent kernels
- **FP8/FP4 Precision**: Tensor Core inference
- **CUDA Tile**: Memory-efficient attention
- **GDS**: Direct NVMe→GPU transfers

---

## 12. Implementation Roadmap

### Phase 0: Ghost System (2-4 weeks)
- Full MCP interface with mocked UTL
- SQLite storage + external embeddings
- Synthetic data seeding
- Tool gating enforcement

### Phase 1-4: Core Engine (16 weeks)
- MCP server + embedding pipeline
- Knowledge graph + FAISS GPU
- UTL integration + Bio-Nervous layers

### Phase 5-7: Advanced Features (9 weeks)
- Dream layer + Neuromodulation
- Immune system + Active inference
- CUDA optimization + GDS

### Phase 8-10: Production (11 weeks)
- MCP hardening + Testing
- Monitoring + Deployment

**Total**: ~49 weeks

---

## 13. Quality Gates

### 13.1 Coverage

| Metric | Threshold | Blocker |
|--------|-----------|---------|
| Unit Test Coverage | ≥90% | Yes |
| Integration Test Coverage | ≥80% | Yes |
| Clippy Warnings | 0 | Yes |
| Unsafe Blocks/Module | ≤5 | Yes |

### 13.2 Performance Gates

| Metric | Threshold |
|--------|-----------|
| P95 Latency | <25ms |
| P99 Latency | <50ms |
| Throughput | >1000/sec |
| GPU Memory | <24GB |
| Error Rate | <0.1% |

### 13.3 UTL Gates

| Metric | Threshold |
|--------|-----------|
| UTL Score Average | >0.6 |
| Coherence Recovery | <10s |
| Entropy Variance | <0.2 |

---

## 14. Agent's Quick Reference

### First Contact Protocol
1. Call `get_system_instructions` → Keep ~300 token fragment in context
2. Call `get_graph_manifest` → Understand 5-layer system
3. Call `get_memetic_status` → Check current state
4. Process any `curation_tasks`

### Core Behaviors

**When to Dream**:
- Entropy > 0.7 for 5+ minutes
- Working 30+ minutes without pause

**How to Curate**:
- ALWAYS check `curation_tasks` before merging
- ALWAYS include `rationale` when storing
- Use `summarize` strategy for important nodes

**When Confused**:
- Search empty → Adjust noradrenaline, search broader
- Search irrelevant → Call `reflect_on_memory`
- Search conflicting → Check `conflict_alert`, merge or ask user
- User asks "why don't you remember X?" → Check `get_system_logs`

---

## 15. References

### Research Papers
- **NeuroDream**: [SSRN 2025](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5377250)
- **Sleep Replay Consolidation**: [Nature Communications](https://www.nature.com/articles/s41467-022-34938-7)
- **Active Inference**: [MIT Press](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)
- **Predictive Coding**: [Nature Communications 2025](https://www.nature.com/articles/s41467-025-64234-z)
- **Neuromodulation**: [Trends in Neurosciences](https://www.cell.com/trends/neurosciences/abstract/S0166-2236(21)00256-3)
- **Homeostatic Plasticity**: [eLife 2025](https://elifesciences.org/articles/88376)
- **Hyperbolic Entailment Cones**: [ICML](http://proceedings.mlr.press/v80/ganea18a/ganea18a.pdf)
- **UniGuardian Defense**: [arXiv 2025](https://arxiv.org/abs/2502.13141)
- **OWASP LLM Top 10**: [OWASP GenAI](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
