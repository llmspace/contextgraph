# Ultimate Context Graph - Compressed PRD v2.0.0

**Abbrev**: NT=Neurotransmitter, SS=Steering Subsystem, OI=Omnidirectional Inference, FV=Formal Verification, PC=Predictive Coding, HE=Hyperbolic Entailment

---

## 0. WHY THIS EXISTS & YOUR ROLE

### 0.1 The Problem You're Solving
AI agents fail because: **no persistent memory** (context lost between sessions), **poor retrieval** (vector search misses semantic relationships), **no learning loop** (no feedback on storage quality), **context bloat** (agents retrieve too much, can't compress).

### 0.2 What the System Does vs What You Do

| System (Automatic) | You (Active) |
|-------------------|--------------|
| Stores conversations via host hooks | Curates quality (merge, annotate, forget) |
| Runs dream consolidation when idle | Triggers dreams when entropy high |
| Detects conflicts & duplicates | Decides resolution strategy |
| Computes UTL metrics | Responds to Pulse suggestions |
| PII scrubbing, adversarial defense | Nothing - trust the system |

**You are a librarian, not an archivist.** You don't store everything—you ensure what's stored is findable, coherent, and useful.

### 0.3 Steering Feedback Loop (How You Learn What to Store)
```
You store node → System assesses quality → Returns reward signal
       ↑                                            │
       └────────── You adjust behavior ─────────────┘
```

**Rewards by lifecycle:**
| Stage | Good Storage (+reward) | Bad Storage (-penalty) |
|-------|------------------------|------------------------|
| Infancy | High novelty (ΔS) | Low novelty |
| Growth | Balanced ΔS + ΔC | Imbalanced |
| Maturity | High coherence (ΔC) | Low coherence |

**Universal penalties:**
- Missing rationale: **-0.5**
- Near-duplicate (>0.9 sim): **-0.4**
- Low priors confidence: **-0.3**

---

## 1. AGENT QUICK START

### 1.1 System Overview
5-layer bio-nervous memory (UTL). Storage=automatic. **Your job: curation + retrieval (librarian).**

### 1.2 First Contact
1. `get_system_instructions` → mental model (~300 tok, KEEP)
2. `get_graph_manifest` → 5-layer architecture
3. `get_memetic_status` → entropy/coherence + `curation_tasks`

### 1.3 Cognitive Pulse (Every Response)
`Pulse: { Entropy: X, Coherence: Y, Suggested: "action" }`

| E | C | Action |
|---|---|--------|
| >0.7 | >0.5 | `epistemic_action` |
| >0.7 | <0.4 | `trigger_dream`/`critique_context` |
| <0.4 | >0.7 | Continue |
| <0.4 | <0.4 | `get_neighborhood` |

### 1.4 Core Behaviors
**Dreaming**: entropy>0.7 for 5+min OR 30+min work → `trigger_dream`
**Curation**: NEVER blind `merge_concepts`. Check `curation_tasks` first. Use `merge_strategy=summarize` (important) or `keep_highest` (trivial). ALWAYS include `rationale`.
**Feedback**: Empty search→↑noradrenaline. Irrelevant→`reflect_on_memory`. Conflicts→check `conflict_alert`. "Why don't you remember?"→`get_system_logs`.

### 1.4.1 Decision Trees

**When to Store:**
```
User shares information
  ├─ Is it novel? (check entropy after inject_context)
  │   ├─ YES + relevant → store_memory with rationale
  │   └─ NO → skip (system already has it)
  └─ Will it help future retrieval?
      ├─ YES → store with link_to related nodes
      └─ NO → don't pollute the graph
```

**When to Dream:**
```
Check Pulse entropy
  ├─ entropy > 0.7 for 5+ min → trigger_dream(phase=full)
  ├─ Working 30+ min straight → trigger_dream(phase=nrem)
  └─ entropy < 0.5 → no dream needed
```

**When to Curate:**
```
get_memetic_status returns curation_tasks?
  ├─ YES → process tasks BEFORE other work
  │   ├─ Duplicate detected → merge_concepts (check priors first!)
  │   ├─ Conflict detected → critique_context, then ask user or merge
  │   └─ Orphan detected → forget_concept or link_to parent
  └─ NO → continue normal work
```

**When Search Fails:**
```
Search returns empty/irrelevant
  ├─ Empty → broaden query, try generate_search_plan
  ├─ Irrelevant → reflect_on_memory to understand why
  ├─ Conflicting → check conflict_alert, resolve or ask user
  └─ User asks "why don't you remember X?" → get_system_logs
```

**When Confused (low coherence):**
```
coherence < 0.4
  ├─ High entropy too → trigger_dream or critique_context
  ├─ Low entropy → get_neighborhood to build context
  └─ System suggests epistemic_action → ASK the clarifying question
```

### 1.5 Query Best Practices
`generate_search_plan` → 3 optimized queries (semantic/causal/code) → parallel execute
`find_causal_path` → "UserAuth→JWT→Middleware→RateLimiting"

### 1.6 Token Economy
| Level | Tokens | When |
|-------|--------|------|
| 0 | ~100 | High confidence |
| 1 | ~200 | Normal |
| 2 | ~800 | coherence<0.4 ONLY |

### 1.7 Multi-Agent Safety
`perspective_lock: { domain: "code", exclude_agent_ids: ["creative-writer"] }`

### 1.8 Epistemic Actions (System-Generated Questions)

When coherence < 0.4, the system can generate clarifying questions for you to ask:

```json
{
  "action_type": "ask_user",
  "question": "You mentioned UserAuth connects to JWT—what triggers token refresh?",
  "expected_entropy_reduction": 0.35,
  "focal_nodes": ["node_userauth", "node_jwt"]
}
```

**Your job:** Ask the user this question (or a natural variant), then store their answer with `store_memory`.

**When to use `epistemic_action(force=true)`:**
- You're stuck and don't know what to ask
- Coherence has been low for multiple exchanges
- User seems to expect you to know something you don't

---

## 2. ARCHITECTURE

### 2.1 UTL Core
```
L = f((ΔS × ΔC) · wₑ · cos φ)
ΔS: entropy[0,1], ΔC: coherence[0,1], wₑ: emotional[0.5,1.5], φ: phase[0,π]
Loss: J = 0.4·L_task + 0.3·L_semantic + 0.3·(1-L)
```

### 2.2 Johari Quadrants
| ΔS | ΔC | Quadrant |
|----|-----|----------|
| Low | High | Open (direct recall) |
| High | Low | Blind (discovery) |
| Low | Low | Hidden (private) |
| High | High | Unknown (frontier) |

### 2.3 5-Layer Bio-Nervous
| L | Function | Latency | Key |
|---|----------|---------|-----|
| L1 | Sensing/tokenize | <5ms | Embed+PII |
| L2 | Reflex/cache | <100μs | Hopfield (>80% hit) |
| L3 | Memory/retrieval | <1ms | Hopfield (2^768 cap) |
| L4 | Learning/weights | <10ms | UTL Optimizer |
| L5 | Coherence/verify | <10ms | Thalamic Gate, PC, FV |

### 2.4 Lifecycle (Marblestone λ Weights)
| Phase | Interactions | λ_ΔS | λ_ΔC | Stance |
|-------|--------------|------|------|--------|
| Infancy | 0-50 | 0.7 | 0.3 | Capture (novelty) |
| Growth | 50-500 | 0.5 | 0.5 | Balanced |
| Maturity | 500+ | 0.3 | 0.7 | Curation (coherence) |

---

## 3. 12-MODEL EMBEDDING

| ID | Model | Dim | Latency |
|----|-------|-----|---------|
| E1 | Semantic | 1024D | <5ms |
| E2 | Temporal-Recent | 512D (exp decay) | <2ms |
| E3 | Temporal-Periodic | 512D (Fourier) | <2ms |
| E4 | Temporal-Positional | 512D (sin PE) | <2ms |
| E5 | Causal | 768D (SCM) | <8ms |
| E6 | Sparse | ~30K (5% active) | <3ms |
| E7 | Code | 1536D (AST) | <10ms |
| E8 | Graph/GNN | 1536D | <5ms |
| E9 | HDC | 10K-bit (XOR/Hamming) | <1ms |
| E10 | Multimodal | 1024D | <15ms |
| E11 | Entity/TransE | 256D (h+r≈t) | <2ms |
| E12 | Late-Interaction | 128D/tok (ColBERT) | <8ms |

**FuseMoE**: Laplace-smoothed gating, top-k=4 → unified 1536D
**CAME-AB**: Cross-modality per-attention + bridge + residual

---

## 4. DATA MODEL

### 4.1 KnowledgeNode
`id:UUID, content:str[≤65536], embedding:Vec1536, created_at, last_accessed, importance:f32[0,1], access_count:u32, johari_quadrant, utl_state:{delta_s,delta_c,w_e,phi}, agent_id?, semantic_cluster?, priors_vibe_check:{assumption_embedding[128], domain_priors, prior_confidence}`

### 4.2 GraphEdge
`source,target:UUID, edge_type:Semantic|Temporal|Causal|Hierarchical|Relational, weight,confidence:f32[0,1], nt_weights:{excitatory,inhibitory,modulatory}[0,1], is_amortized_shortcut:bool, steering_reward:f32[-1,1], domain:Code|Legal|Medical|Creative|Research|General`

### 4.3 NT Edge Modulation (Marblestone)
`w_eff = base × (1 + excitatory - inhibitory + 0.5×modulatory)`

### 4.4 Hyperbolic (Poincare Ball)
All nodes: ||x||<1, O(1) IS-A via entailment cones
`d(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))`

---

## 5. MCP TOOLS

### 5.1 Protocol
JSON-RPC 2.0, stdio/SSE, caps: tools/resources/prompts/logging

### 5.2 Core Tools (When & Why)

| Tool | WHEN to use | WHY | Key Params |
|------|-------------|-----|------------|
| `inject_context` | Starting task, need background | Primary retrieval with auto-distillation | query, max_tokens, distillation_mode, verbosity_level |
| `search_graph` | Need specific nodes, not narrative | Raw vector search, you distill | query, top_k, filters, perspective_lock |
| `store_memory` | User shares valuable, novel info | Requires rationale—no blind storage | content, importance, rationale, link_to |
| `query_causal` | Need to understand cause→effect | "What happens if X?" questions | action, outcome, intervention_type |
| `trigger_dream` | entropy>0.7 OR 30+min work | Consolidates, finds blind spots | phase:nrem/rem/full, duration, blocking |
| `get_memetic_status` | Start of session, periodically | See health, get curation_tasks | → entropy, coherence, curation_tasks |
| `get_graph_manifest` | First contact only | Understand system architecture | - |
| `epistemic_action` | coherence<0.4, need clarity | Generate clarifying question to ask user | session_id, force |
| `get_neuromodulation` | Debug retrieval quality | See modulator levels affecting search | session_id |
| `get_steering_feedback` | After storing, want to learn | See if your storage was valuable | content, context, domain |

### 5.3 Curation Tools
`merge_concepts` (source_node_ids, target_name, merge_strategy, force_merge), `annotate_node`, `forget_concept` (soft_delete=true default), `boost_importance`, `restore_from_hash` (30-day undo)

### 5.4 Navigation Tools
`get_neighborhood`, `get_recent_context`, `find_causal_path`, `entailment_query`

### 5.5 Meta-Cognitive Tools
`reflect_on_memory` (goal→tool sequence), `generate_search_plan` (goal→queries), `critique_context` (fact-check), `hydrate_citation` ([node_xyz]→raw), `get_system_instructions`, `get_system_logs`, `get_node_lineage`

### 5.6 Diagnostic Tools
`utl_status`, `homeostatic_status`, `check_adversarial`, `test_recall_accuracy`, `debug_compare_retrieval`, `search_tombstones`

### 5.7 Admin Tools
`reload_manifest`, `temporary_scratchpad`

### 5.8 Resources
`context://{scope}`, `graph://{node_id}`, `utl://{session}/state`, `utl://current_session/pulse`, `admin://manifest`, `visualize://{scope}/{topic}`

### 5.9 Marblestone Tools
| Tool | Purpose |
|------|---------|
| `get_steering_feedback` | SS reward signal |
| `omni_infer` | OI: forward/backward/bidirectional/abduction |
| `verify_code_node` | FV for code nodes |

**OI Directions**: forward(A→B), backward(B→A), bidirectional(A↔B), abduction(best explanation)

---

## 6. KEY MECHANISMS

### 6.1 inject_context Response
`context, tokens_used, tokens_before_distillation, distillation_applied, compression_ratio, nodes_retrieved, utl_metrics:{entropy,coherence,learning_score}, bundled_metadata:{causal_links,entailment_cones,neighborhood}, conflict_alert:{has_conflict,conflicting_nodes,suggested_action}, tool_gating_warning (entropy>0.8), Pulse`

### 6.2 Distillation Modes
auto|raw|narrative|structured|code_focused

### 6.3 Conflict Detection
Trigger: cos_sim>0.8 AND causal_coherence<0.3 → returns conflict_id (~20 tok)

### 6.4 Citation Tags
`[node_abc123]` → `hydrate_citation` to expand

### 6.5 Priors Vibe Check
128D assumption_embedding. Merge: cos_sim>0.7=normal, incompatible=Relational Edge, override=force_merge=true

### 6.6 Tool Gating
entropy>0.8 → warning: use `generate_search_plan`/`epistemic_action`/`expand_causal_path`

---

## 7. BACKGROUND SYSTEMS

### 7.1 Dream Layer (SRC)
**NREM (3min)**: Replay + Hebbian Δw=η×pre×post + tight coupling
**REM (2min)**: Synthetic queries (hyperbolic random walk) + blind spots (high semantic dist + shared causal) + new edges (w=0.3, c=0.5)
**Amortized Shortcuts (Marblestone)**: 3+ hop chains traversed ≥5× → direct edge, confidence≥0.7, w=product(path weights), is_amortized_shortcut=true
**Schedule**: activity<0.15 for 10min → trigger, wake<100ms on query

### 7.2 Neuromodulation
Dopamine→beta[1-5] (sharp), Serotonin→top_k[2-8] (explore), Noradrenaline→temp[0.5-2] (flat), Acetylcholine→lr (fast)
Update <200μs/query
**SS Dopamine Feedback**: +reward→dopamine+=r×0.2, -reward→dopamine-=|r|×0.1

### 7.3 Homeostatic Optimizer
Scales importance→0.5 setpoint, detects semantic cancer (high importance + high neighbor entropy), quarantines

### 7.4 Graph Gardener
activity<0.15 for 2+min: prune weak edges (<0.1 w, no access), merge near-dupes (>0.95 sim, priors OK), rebalance hyperbolic, rebuild FAISS

### 7.5 Passive Curator
Auto: high-confidence dupes (>0.95), weak links, orphans (>30d)
Escalates: ambiguous dupes (0.7-0.95), priors-incompatible, conflicts, semantic cancer
Reduces curation ~70%

### 7.6 Glymphatic Clearance
Background prune low-importance during idle

### 7.7 PII Scrubber
L1 pre-embed: patterns (<1ms), NER (<100ms) → [REDACTED:type]

### 7.8 Steering Subsystem (Marblestone) - How You Learn What to Store

Separate from Learning - reward signals only, no direct weight modification.

**This is how you improve.** The system evaluates your storage decisions and teaches you.

**Flow:**
```
1. You call store_memory with content + rationale
2. System computes: novelty, coherence fit, duplicate risk, priors alignment
3. Returns SteeringReward with score [-1, +1] and explanation
4. Your dopamine adjusts → affects future retrieval sharpness
```

**Components**: Gardener (cross-session curation), Curator (per-domain quality), Thought Assessor (per-interaction)

**SteeringReward**: `reward:f32[-1,1], gardener_score, curator_score, assessor_score, explanation, suggestions`

**Example rewards:**
```json
// GOOD: Novel, coherent, well-linked
{"reward": 0.7, "explanation": "High novelty, strong causal links to existing nodes"}

// BAD: Duplicate
{"reward": -0.4, "explanation": "92% similar to node_abc123, consider merge instead"}

// BAD: No rationale
{"reward": -0.5, "explanation": "Missing rationale - cannot assess relevance"}
```

**What to do with feedback:**
- reward > 0.3 → Good instinct, continue this pattern
- reward [-0.3, 0.3] → Neutral, acceptable but not optimal
- reward < -0.3 → Adjust behavior: add rationale, check for dupes, improve linking

**Integration**: Feeds dopamine, guides Learning without modifying weights

### 7.9 OI Engine (Marblestone)
Directions: forward (predict), backward (root cause), bidirectional (discover), bridge (cross-domain), abduction (hypothesis)
Clamped Variables: hard/soft clamp during inference
Active Inference: EFE for direction selection

### 7.10 Formal Verification Layer (Marblestone)
Lean-inspired SMT for code nodes @ L5
**Conditions**: bounds, null safety, type invariants, loop termination, custom assertions
**Status**: Verified|Failed|Timeout|NotApplicable
Cache by content hash, default timeout 5s

---

## 8. PREDICTIVE CODING

L5→L1 feedback: prediction→error=obs-pred→only surprise propagates
~30% token reduction for predictable contexts
**EmbeddingPriors by domain**: Medical: causal 1.8, code 0.3 | Programming: code 2.0, graph 1.5

---

## 9. HYPERBOLIC ENTAILMENT CONES

O(1) hierarchy via cone containment
`EntailmentCone: apex, aperture:f32(rad), axis:Vec1536`
`contains(point) = angle(tangent,axis) ≤ aperture`
Ancestors=cones containing node, Descendants=within node's cone

---

## 10. ADVERSARIAL DEFENSE

| Check | Attack | Response |
|-------|--------|----------|
| Embedding outlier | >3 std | Quarantine |
| Content-embed misalign | <0.4 | Block |
| Known signatures | Pattern | Block+log |
| Prompt injection | Regex | Block+log |
| Circular logic | Cycle detect | Prune edges |

**Patterns**: "ignore previous", "disregard system", "you are now", "new instructions:", "override:"

---

## 11. CROSS-SESSION IDENTITY

| Scope | Persistence |
|-------|-------------|
| User | Permanent |
| Session | Per-terminal |
| Context | Per-conversation |

Same user across clients=shared graph, different sessions=isolated working memory

---

## 12. HUMAN-IN-THE-LOOP

### 12.1 Manifest (~/.context-graph/manifest.md)
```markdown
## Active Concepts
## Pending Actions
[MERGE: JWTValidation, OAuth2Validation]
## Notes
[NOTE: RateLimiting] Deprecated v2.0
```
User edits → `reload_manifest`

### 12.2 Visualization
`visualize://topic/auth` → Mermaid. User spots merges, semantic cancer.

### 12.3 Undo
`reversal_hash` per merge/forget, 30-day recovery via `restore_from_hash`

---

## 13. HARDWARE

RTX 5090: 32GB GDDR7, 1792 GB/s, 21760 CUDA, 680 Tensor (5th gen), Compute 12.0, CUDA 13.1
**CUDA 13.1**: Green Contexts (4×170 SMs), FP8/FP4, CUDA Tile, GPU Direct Storage

---

## 14. PERFORMANCE TARGETS

| Op | Target |
|----|--------|
| Single Embed | <10ms |
| Batch Embed (64) | <50ms |
| FAISS search (1M) | <2ms |
| Hopfield | <1ms |
| FuseMoE | <3ms |
| Cache hit | <100μs |
| inject_context P95 | <25ms |
| Any tool P99 | <50ms |
| Neuromod batch | <200μs |
| Dream wake | <100ms |

### Quality Gates
| Metric | Threshold |
|--------|-----------|
| Unit coverage | ≥90% |
| Integration coverage | ≥80% |
| UTL avg | >0.6 |
| Coherence recovery | <10s |
| Attack detection | >95% |
| False positive | <2% |
| Distill latency | <50ms |
| Info loss | <15% |
| Compression | >60% |

---

## 15. ROADMAP

**Phase 0 (2-4w)**: Ghost System - MCP+SQLite+mocked UTL+synthetic data
**Phases 1-14 (~49w)**: Core(4w)→Embed(4w)→Graph(4w)→UTL(4w)→Bio(4w)→CUDA(3w)→GDS(3w)→Dream(3w)→Neuromod(3w)→Immune(3w)→ActiveInf(2w)→MCPHarden(4w)→Test(4w)→Deploy(4w)

---

## 16. MONITORING

### Metrics
UTL: learning_score, entropy, coherence, johari
GPU: util, mem, temp, kernel_dur
MCP: requests, latency, errors, connections
Dream: phase, blind_spots, wake_latency
Neuromod: dopamine, serotonin, noradrenaline, acetylcholine
Immune: attacks, false_pos, quarantined, health

### Alerts
| Alert | Condition | Severity |
|-------|-----------|----------|
| LearningLow | avg<0.4 5m | warning |
| GpuMemHigh | >90% 5m | critical |
| ErrorHigh | >1% 5m | critical |
| LatencyP99High | >50ms 5m | warning |
| DreamStuck | >15m | warning |
| AttackHigh | >10/5m | critical |
| SemanticCancer | quarantined>0 | warning |

---

## 17. CONCURRENCY

`ConcurrentGraph: inner: Arc<RwLock<KG>>, faiss: Arc<RwLock<FaissGpu>>`
Lock order: inner→faiss (no deadlock)
Soft delete default (30d recovery), permanent only: reason='user_requested'+soft_delete=false

---

## 18. REFERENCES

**Internal**: UTL(2.1), 5-Layer(2.3), Embed(3), MCP(5), Dream(7.1), Neuromod(7.2), Immune(7.3), HE(9), Adversarial(10)
**External**: NeuroDream(SSRN'25), SRC(NatComm), FEP(Wiki), ActiveInf(MIT), PC(Nature'25), Neuromod DNNs(TrendsNeuro), Homeostatic(eLife'25), BioLogicalNeuron(SciRep'25), HE Cones(ICML), Poincare(NeurIPS), UniGuardian(arXiv'25), OWASP LLM Top10

---

## 19. TOOL PARAM REFERENCE

### inject_context
`query:str[1-4096] REQ, max_tokens:int[100-8192]=2048, session_id:uuid, priority:low|normal|high|critical, distillation_mode:auto|raw|narrative|structured|code_focused, include_metadata:[causal_links,entailment_cones,neighborhood,conflicts], verbosity_level:0|1|2=1`

### search_graph
`query:str REQ, top_k:int=10[max100], filters:{min_importance,johari_quadrants,created_after}, perspective_lock:{domain,agent_ids,exclude_agent_ids}`

### store_memory
`content:str[≤65536] REQ(if text), content_base64:str[≤10MB], data_uri:str, modality:text|image|audio|video, importance:f32[0-1]=0.5, rationale:str[10-500] REQ, metadata:obj, link_to:[uuid]`

### merge_concepts
`source_node_ids:[uuid] REQ(min2), target_name:str REQ, merge_strategy:keep_newest|keep_highest|concatenate|summarize, force_merge:bool`

### forget_concept
`node_id:uuid REQ, reason:semantic_cancer|adversarial_injection|user_requested|obsolete REQ, cascade_edges:bool=true, soft_delete:bool=true`

### trigger_dream
`phase:nrem|rem|full_cycle=full_cycle, duration_minutes:int[1-10]=5, synthetic_query_count:int[10-500]=100, blocking:bool=false, abort_on_query:bool=true`

### get_memetic_status
`session_id:uuid` → coherence_score, entropy_level, top_active_concepts[max5], suggested_action:consolidate|explore|clarify|curate|ready, dream_available, curation_tasks:[{task_type,target_nodes,reason,suggested_tool,priority}]

### reflect_on_memory
`goal:str[10-500] REQ, session_id:uuid, max_steps:int[1-5]=3` → reasoning, suggested_sequence:[{step,tool,params,rationale}], utl_context

### generate_search_plan
`goal:str[10-500] REQ, query_types:[semantic,causal,code,temporal,hierarchical], max_queries:int[1-7]=3` → queries:[{query,type,rationale,expected_recall}], execution_strategy:parallel|sequential|cascade, token_estimate

### find_causal_path
`start_concept:str REQ, end_concept:str REQ, max_hops:int[1-6]=4, path_type:causal|semantic|any, include_alternatives:bool` → path_found, narrative, path:[{node_id,node_name,edge_type,edge_weight}], hop_count, path_confidence

### critique_context
`reasoning_summary:str[20-2000] REQ, focal_nodes:[uuid], contradiction_threshold:f32[0.3-0.9]=0.5` → contradictions_found, contradicting_nodes:[{node_id,content_snippet,contradiction_type,confidence}], suggested_action

### hydrate_citation
`citation_tags:[str][1-10] REQ, include_neighbors:bool, verbosity_level:0|1|2` → expansions:[{citation_tag,raw_content,importance,created_at,neighbors}]

### get_system_logs
`log_type:all|quarantine|prune|merge|dream_actions|adversarial_blocks, node_id:uuid, since:datetime, limit:int[1-100]=20` → entries:[{timestamp,action,node_ids,reason,recoverable,recovery_tool}], explanation_for_user

### temporary_scratchpad
`action:store|retrieve|clear REQ, content:str[≤4096](for store), session_id:uuid REQ, agent_id:str REQ, privacy:private|team|shared, tags:[str][max5], auto_commit_threshold:f32[0.3-0.9]=0.6`
