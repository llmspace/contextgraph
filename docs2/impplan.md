# Dynamic Multi-Purpose Vector Implementation Plan

## Executive Summary

This document synthesizes research from six parallel investigations to create a comprehensive implementation plan for transforming the contextgraph memory system from a **single North Star** architecture to a **dynamic multi-cluster purpose vector** system that:

1. Works perfectly from 0 memories (graceful degradation)
2. Eliminates bias from historical data dominating future results
3. Discovers emergent topics via hierarchical clustering (3+ close fingerprints = topic)
4. Uses frequency/recency-based importance (not explicit goals)
5. Operates fully autonomously without user-set goals

---

## Part 1: Architecture Overview

### Current State (Problems)

```
┌─────────────────────────────────────────────────────────────────┐
│ CURRENT: Single North Star Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│ • K-means discovers ONE goal from stored fingerprints           │
│ • Requires minimum 3 memories to bootstrap (FAILS at 0-2)       │
│ • Early memories bias the North Star permanently                │
│ • Stage 4 retrieval FAILS without purpose vector                │
│ • No mechanism to track multiple interests                      │
└─────────────────────────────────────────────────────────────────┘
```

### Target State (Solution)

```
┌─────────────────────────────────────────────────────────────────┐
│ TARGET: Dynamic Multi-Cluster Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│ • HDBSCAN + BIRCH hybrid discovers N emergent topic clusters    │
│ • Works from 0 memories with 6-tier progressive activation      │
│ • Thompson Sampling + IPS prevents historical bias              │
│ • EWMA + BM25 saturation for fair importance scoring            │
│ • Multi-interest portfolio with exploration-exploitation        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Progressive Feature Activation (Zero-Memory Operation)

### 6-Tier Maturity Model

The system MUST work at every memory count. Features activate progressively:

| Tier | Memory Count | Features Available | Confidence Level |
|------|--------------|-------------------|------------------|
| **0** | 0 | Storage, Status, Basic Retrieval | None |
| **1** | 1-2 | Exact Retrieval, Pairwise Similarity | None |
| **2** | 3-9 | Basic Clustering, Similarity Search | Very Low |
| **3** | 10-29 | Multiple Clusters, Anomaly Detection | Low |
| **4** | 30-99 | Reliable Statistics, Drift Detection | Moderate |
| **5** | 100-499 | Sub-clustering, Trend Analysis | High |
| **6** | 500+ | Full Personalization, Predictive Features | Very High |

### Tier 0: Zero Memories (MUST Work)

```rust
pub fn get_retrieval_purpose(&self) -> Option<DynamicPurposeVector> {
    match self.memory_count {
        0 => None,  // Skip Stage 4 alignment entirely
        1..=2 => None,  // Semantic search only
        3..=9 => Some(self.provisional_purpose_vector()),
        _ => Some(self.dynamic_purpose_vector()),
    }
}

pub fn stage_alignment_filter(&self, candidates: Vec<Candidate>) -> Vec<Candidate> {
    match self.get_retrieval_purpose() {
        None => candidates,  // Pass through unchanged (graceful degradation)
        Some(pv) => self.apply_multi_goal_filter(candidates, pv),
    }
}
```

### Default Values at Each Tier

| Metric | Tier 0-2 Default | Rationale |
|--------|-----------------|-----------|
| Alignment Score | 0.5 (neutral) | Uninformative prior |
| Confidence | 0.0 | No data to be confident about |
| Cluster Assignment | -1 (unclustered) | No clusters exist yet |
| Purpose Vector | [0.5; 13] uniform | Maximum entropy |
| IC (Identity Continuity) | 1.0 | First state = healthy by definition |

---

## Part 3: Hierarchical Clustering for Topic Discovery

### Algorithm Selection: HDBSCAN + BIRCH Hybrid

**Why this combination:**

| Algorithm | Role | Strength |
|-----------|------|----------|
| **HDBSCAN** | Batch clustering | Auto-determines cluster count, handles variable density |
| **BIRCH** | Online insertion | O(1) incremental updates via CF-trees |
| **Growing Neural Gas** | Topology learning | Continuous learning, no stopping condition |

### Cluster Formation Rules

```
RULE: Cluster Formation Criteria

1. min_cluster_size = 3  (≥3 fingerprints required)
2. max_cluster_radius = adaptive_threshold (cosine distance)
3. silhouette_score > 0.3 (quality validation)

is_valid_cluster(points):
    IF len(points) < 3: RETURN false
    centroid = mean(points)
    max_distance = max(cosine_distance(p, centroid) for p in points)
    silhouette = compute_silhouette(points)
    RETURN max_distance <= threshold AND silhouette > 0.3
```

### Mathematical Formulas

**Cosine Distance (primary metric for 13D purpose vectors):**
```
cosine_distance(A, B) = 1 - (A · B) / (||A|| × ||B||)
```

**Silhouette Score (cluster quality):**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

where:
  a(i) = average distance to points in same cluster (cohesion)
  b(i) = average distance to nearest other cluster (separation)

Interpretation:
  s > 0.5 = good clustering
  s ≈ 0 = on boundary
  s < 0 = likely misclassified
```

**Ward's Criterion (merge decision):**
```
merge_cost(C1, C2) = (n1 × n2) / (n1 + n2) × ||c1 - c2||²
```

**Incremental Centroid Update (Welford's algorithm):**
```
new_mean = old_mean + (new_point - old_mean) / new_count
```

### Implementation: IncrementalPurposeCluster

```rust
pub struct PurposeClusterManager {
    clusters: HashMap<ClusterId, TopicCluster>,
    unclustered_queue: VecDeque<MemoryId>,

    // Configuration
    min_cluster_size: usize,        // 3
    merge_threshold: f32,           // 0.85 cosine similarity
    split_threshold: f32,           // 0.5 internal silhouette
    rebalance_interval: Duration,   // 1 hour
}

pub struct TopicCluster {
    centroid: [f32; 13],
    members: Vec<MemoryId>,
    cf_entry: ClusteringFeature,  // BIRCH-style (N, LS, SS)
    importance: f32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    access_count: u64,
}

impl PurposeClusterManager {
    pub fn add_memory(&mut self, memory: &Memory) -> Result<()> {
        let pv = memory.purpose_vector();
        let nearest = self.find_nearest_clusters(pv, k=3);

        if nearest[0].similarity > self.merge_threshold {
            // Add to existing cluster
            self.update_cluster_incremental(nearest[0].id, memory)?;
        } else {
            // Queue for potential new cluster
            self.unclustered_queue.push_back(memory.id);
            self.attempt_new_cluster_formation()?;
        }
        Ok(())
    }

    fn attempt_new_cluster_formation(&mut self) -> Result<()> {
        for seed in self.unclustered_queue.iter() {
            let neighbors = self.find_neighbors(seed, radius=1.0 - self.merge_threshold);
            if neighbors.len() >= self.min_cluster_size {
                let new_cluster = TopicCluster::from_members(neighbors)?;
                if self.validate_cluster(&new_cluster) {
                    self.clusters.insert(new_cluster.id, new_cluster);
                    self.remove_from_queue(&neighbors);
                }
            }
        }
        Ok(())
    }
}
```

---

## Part 4: Bias Mitigation Strategies

### Problem: Rich-Get-Richer Effect

> "The exposure mechanism determines user behaviors, which are circled back as training data. Such feedback loops not only create biases but also intensify them over time."

### Solution 1: Thompson Sampling for Exploration-Exploitation

**Mathematical Framework:**

For each topic cluster `i`, maintain a Beta distribution:
```
Prior: Beta(α_i, β_i)

Update Rules:
  - Cluster retrieved and deemed relevant: α_i ← α_i + 1
  - Cluster retrieved and deemed irrelevant: β_i ← β_i + 1

Selection Process:
  1. Sample θ_i ~ Beta(α_i, β_i) for each cluster
  2. Select cluster with highest sampled value
  3. This naturally explores uncertain clusters while exploiting known good ones
```

**Key Benefit:** New clusters start with uniform priors `Beta(1, 1)`, giving them fair exploration opportunity.

```rust
pub struct ThompsonSampler {
    beta_params: HashMap<ClusterId, (f32, f32)>,  // (alpha, beta)
}

impl ThompsonSampler {
    pub fn sample(&self, cluster_id: ClusterId) -> f32 {
        let (alpha, beta) = self.beta_params.get(&cluster_id)
            .unwrap_or(&(1.0, 1.0));  // Uniform prior for new clusters

        // Sample from Beta distribution
        rand_distr::Beta::new(*alpha, *beta).unwrap().sample(&mut rand::thread_rng())
    }

    pub fn update(&mut self, cluster_id: ClusterId, was_relevant: bool) {
        let (alpha, beta) = self.beta_params.entry(cluster_id)
            .or_insert((1.0, 1.0));

        if was_relevant {
            *alpha += 1.0;
        } else {
            *beta += 1.0;
        }
    }
}
```

### Solution 2: Inverse Propensity Scoring (IPS)

**Formula:**
```
L_IPS = Σ (r_i / p_i) × loss(f(x_i), y_i)

where:
  r_i = relevance signal (1 if accessed, 0 otherwise)
  p_i = propensity (probability the item was shown/considered)
```

**For Memory Systems:**
- Track how often each cluster could have been retrieved (denominator)
- Weight learning signals inversely to exposure frequency
- New/rare clusters get higher weights to compensate for fewer opportunities

```rust
pub struct PropensityEstimator {
    exposure_counts: HashMap<ClusterId, u64>,
    total_queries: u64,
}

impl PropensityEstimator {
    pub fn get_propensity(&self, cluster_id: ClusterId) -> f32 {
        let exposures = self.exposure_counts.get(&cluster_id).unwrap_or(&0);
        (*exposures as f32 / self.total_queries as f32).max(0.1)  // Floor at 0.1
    }

    pub fn compute_ips_weight(&self, cluster_id: ClusterId) -> f32 {
        1.0 / self.get_propensity(cluster_id)
    }
}
```

### Solution 3: Maximal Marginal Relevance (MMR) for Diversity

**Formula:**
```
MMR = λ × Sim(d_i, Q) - (1-λ) × max[Sim(d_i, d_j)] for d_j ∈ S

where:
  Sim(d_i, Q) = relevance of memory to query
  Sim(d_i, d_j) = similarity to already-selected memories
  λ = diversity parameter (0.7 recommended)
  S = set of already-selected memories
```

**Implementation:**
```rust
pub fn mmr_selection(
    candidates: &[MemoryCandidate],
    k: usize,
    lambda: f32,
) -> Vec<MemoryCandidate> {
    let mut selected = Vec::with_capacity(k);
    let mut remaining: Vec<_> = candidates.to_vec();

    while selected.len() < k && !remaining.is_empty() {
        let best_idx = remaining.iter().enumerate()
            .map(|(idx, candidate)| {
                let relevance = candidate.relevance_score;
                let max_sim_to_selected = selected.iter()
                    .map(|s| cosine_similarity(&candidate.embedding, &s.embedding))
                    .fold(0.0_f32, f32::max);

                let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim_to_selected;
                (idx, mmr_score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        selected.push(remaining.remove(best_idx));
    }

    selected
}
```

### Solution 4: Elastic Weight Consolidation (EWC)

**Formula:**
```
L_total = L_new + (λ/2) × Σ F_i × (θ_i - θ*_i)²

where:
  L_new = loss on new task
  F_i = Fisher information for parameter i (importance weight)
  θ*_i = optimal parameter value from previous task
  λ = regularization strength
```

**Purpose:** Allows learning new patterns without catastrophic forgetting of important old ones.

---

## Part 5: Importance Scoring (Frequency + Recency)

### Core Formula

```
Importance(topic) = Frequency_Score × Recency_Weight × Confidence_Adjustment

where:
  Frequency_Score = log(1 + count) / log(1 + max_count)  // Log normalized
  Recency_Weight = e^(-λ × t)  // Exponential decay
  Confidence_Adjustment = Wilson_Score if count < 10 else 1.0
```

### Component 1: EWMA (Exponentially Weighted Moving Average)

**Formula:**
```
EWMA_t = α × current_value + (1 - α) × EWMA_{t-1}

where: α = 2 / (N + 1)
```

**Half-life parameterization:**
```
λ = ln(2) / half_life

Recommended half-lives:
  - Fast-changing topics: 7 days (λ = 0.099)
  - Moderate drift: 30 days (λ = 0.023)
  - Stable preferences: 150 days (λ = 0.0046)
```

### Component 2: BM25 Saturation (Prevents Frequency Domination)

**Formula:**
```
tf_saturated = (frequency × (k1 + 1)) / (frequency + k1)

where: k1 = 1.2 (standard)
```

**Effect:** A topic with 100 interactions doesn't get 100× the score of one with 1 interaction.

### Component 3: Wilson Score (Cold Start Adjustment)

**Formula:**
```
Lower_Bound = (p̂ + z²/2n - z√(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)

where:
  p̂ = observed positive rate
  n = number of observations
  z = 1.96 for 95% confidence
```

**Purpose:** New topics with few interactions get lower confidence bounds, preventing random spikes from dominating.

### Component 4: Burst Damping

**Formula:**
```
Is_Burst = current_frequency > μ + k×σ

Damped_Frequency = min(raw_frequency, μ + burst_cap × σ)

where:
  μ = historical mean frequency
  σ = historical standard deviation
  k = 2-3 (sensitivity)
  burst_cap = 1.5 (maximum burst multiplier)
```

### Complete Implementation

```rust
pub fn compute_topic_importance(
    interaction_count: u64,
    last_interaction_time: DateTime<Utc>,
    total_interactions_all_topics: u64,
    global_mean_count: f32,
    global_std_count: f32,
    half_life_days: f32,
) -> f32 {
    let now = Utc::now();
    let days_since_last = (now - last_interaction_time).num_seconds() as f32 / 86400.0;

    // 1. RECENCY: Exponential decay
    let lambda_decay = (2.0_f32).ln() / half_life_days;
    let recency_weight = (-lambda_decay * days_since_last).exp();

    // 2. FREQUENCY: Log-normalized with BM25 saturation
    let k1 = 1.2;
    let log_freq = (1.0 + interaction_count as f32).ln();
    let saturated_freq = (log_freq * (k1 + 1.0)) / (log_freq + k1);

    // 3. BURST DAMPING: Cap extreme bursts
    let burst_threshold = global_mean_count + 2.0 * global_std_count;
    let damped_count = (interaction_count as f32).min(burst_threshold * 1.5);

    // 4. FREQUENCY SCORE: Normalized
    let frequency_score = (1.0 + damped_count).ln() / (1.0 + burst_threshold * 1.5).ln();

    // 5. COLD START: Wilson confidence adjustment
    let confidence_penalty = if interaction_count < 10 {
        interaction_count as f32 / 10.0
    } else {
        1.0
    };

    // 6. FINAL SCORE
    frequency_score * recency_weight * confidence_penalty
}
```

### Recommended Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Half-life | 30-60 days | Balances recency with historical value |
| BM25 k1 | 1.2 | Standard saturation factor |
| Prior weight (Bayesian) | 5-10 | Moderate smoothing for new topics |
| Burst threshold | μ + 2σ | Standard anomaly detection |
| Wilson confidence | 95% (z=1.96) | Reasonable uncertainty bound |
| Min interactions for full weight | 10 | Cold start threshold |

---

## Part 6: Emergent Goal Discovery (Without Presets)

### Multi-Interest Portfolio Model

Instead of a single North Star, maintain a **portfolio of interests**:

```rust
pub struct MultiInterestPortfolio {
    interests: HashMap<ClusterId, InterestProfile>,
    max_interests: usize,  // 20
    exploration_rate: f32,  // 0.15
}

pub struct InterestProfile {
    id: ClusterId,
    importance: f32,
    confidence: f32,
    attention_share: f32,
    last_update: DateTime<Utc>,
    interaction_count: u64,
    success_rate: f32,

    // Thompson Sampling parameters
    alpha: f32,  // successes + 1
    beta: f32,   // failures + 1

    // Lifecycle
    lifecycle_phase: LifecyclePhase,  // Emerging, Stable, Declining
}

pub enum LifecyclePhase {
    Emerging,   // New, high exploration weight
    Stable,     // Established, balanced
    Declining,  // Waning interest, check if still relevant
}
```

### Attention Allocation Algorithm

```rust
impl MultiInterestPortfolio {
    pub fn allocate_attention(&mut self) -> HashMap<ClusterId, f32> {
        // Reserve exploration budget
        let exploration_budget = self.exploration_rate;  // 15%
        let exploitation_budget = 1.0 - exploration_budget;

        // Exploitation: proportional to importance × confidence
        let exploitation_weights: HashMap<_, _> = self.interests.iter()
            .map(|(id, p)| (*id, p.importance * p.confidence))
            .collect();
        let total_weight: f32 = exploitation_weights.values().sum();

        // Exploration: Thompson Sampling
        let exploration_samples: HashMap<_, _> = self.interests.iter()
            .map(|(id, p)| (*id, self.thompson_sample(p)))
            .collect();
        let total_samples: f32 = exploration_samples.values().sum();

        // Combine allocations
        let mut allocations = HashMap::new();
        for id in self.interests.keys() {
            let exploit_share = (exploitation_weights[id] / total_weight) * exploitation_budget;
            let explore_share = (exploration_samples[id] / total_samples) * exploration_budget;

            let allocation = (exploit_share + explore_share)
                .max(0.02)   // Minimum 2% attention
                .min(0.40);  // Maximum 40% attention

            allocations.insert(*id, allocation);
        }

        // Normalize to sum to 1.0
        let total: f32 = allocations.values().sum();
        for v in allocations.values_mut() {
            *v /= total;
        }

        allocations
    }
}
```

### Habit vs. True Interest Detection

**Key Discriminators:**

| Signal | Indicates Habit | Indicates True Interest |
|--------|----------------|------------------------|
| Timing | Regular schedule | Irregular, opportunistic |
| Initiation | Triggered by context | Self-initiated |
| Friction tolerance | Abandons if hard | Persists through difficulty |
| Completion | Often incomplete | High completion rate |
| Post-engagement | Immediate exit | Follow-up actions |

```rust
pub fn compute_desire_score(
    intent_signals: f32,
    engagement_quality: f32,
    habit_score: f32,
) -> f32 {
    // High habit + low intent = ROUTINE (user does, may not want)
    // High intent + high engagement = TRUE_INTEREST
    // High frequency + low completion = ASPIRATIONAL

    (intent_signals * engagement_quality) / habit_score.max(0.1)
}
```

---

## Part 7: Identity Continuity for Multiple Goals

### Current (Single IC)

```rust
pub struct IdentityContinuity {
    identity_coherence: f32,  // Single IC value
    status: IdentityStatus,
}
```

### Target (Multi-Goal IC)

```rust
pub struct MultiGoalIdentityContinuity {
    goal_ics: HashMap<ClusterId, f32>,
    aggregate_ic: f32,
    dominant_goal_id: ClusterId,
    last_switch: DateTime<Utc>,
    switch_count: u32,  // Switches in 24h
}

impl MultiGoalIdentityContinuity {
    pub fn update(
        &mut self,
        purpose_vectors: &HashMap<ClusterId, PurposeVector>,
        previous_vectors: &HashMap<ClusterId, PurposeVector>,
        kuramoto_r: f32,
        importance_weights: &HashMap<ClusterId, f32>,
    ) {
        // Compute IC per goal
        for (goal_id, pv) in purpose_vectors {
            if let Some(prev_pv) = previous_vectors.get(goal_id) {
                let cosine = pv.similarity(prev_pv);
                let ic = (cosine * kuramoto_r).clamp(0.0, 1.0);
                self.goal_ics.insert(*goal_id, ic);
            }
        }

        // Compute weighted aggregate
        let mut aggregate = 0.0;
        let mut total_weight = 0.0;
        for (goal_id, ic) in &self.goal_ics {
            let weight = importance_weights.get(goal_id).unwrap_or(&0.1);
            aggregate += ic * weight;
            total_weight += weight;
        }
        self.aggregate_ic = (aggregate / total_weight).clamp(0.0, 1.0);

        // Update dominant goal
        let new_dominant = self.goal_ics.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, _)| *id);

        if let Some(new_id) = new_dominant {
            if new_id != self.dominant_goal_id {
                self.last_switch = Utc::now();
                self.switch_count += 1;
            }
            self.dominant_goal_id = new_id;
        }
    }

    pub fn is_in_crisis(&self) -> bool {
        self.aggregate_ic < 0.7
    }

    pub fn should_trigger_dream(&self) -> bool {
        self.aggregate_ic < 0.5
    }
}
```

---

## Part 8: Retrieval Pipeline Changes

### Current Stage 4 (Single Purpose Vector)

```rust
// CURRENT: Fails without North Star
let purpose_vector = self.config.purpose_vector
    .ok_or(PipelineError::MissingPurposeVector)?;
```

### Target Stage 4 (Multi-Goal Cascade)

```rust
pub fn stage_alignment_filter_multi(
    &self,
    candidates: Vec<PipelineCandidate>,
    config: &StageConfig,
) -> Result<StageResult, PipelineError> {
    // Graceful degradation: If no clusters exist, pass through
    let dynamic_pv = match self.config.dynamic_purpose_vector() {
        Some(pv) => pv,
        None => return Ok(StageResult::passthrough(candidates)),
    };

    // Try each goal in importance order
    for (idx, (pv, weight)) in dynamic_pv.iter_weighted().enumerate() {
        let filtered: Vec<_> = candidates.iter()
            .filter(|c| self.compute_alignment(c, pv) >= config.min_score_threshold)
            .cloned()
            .collect();

        // If we have enough results, or this is the last goal, return
        if filtered.len() >= config.min_candidates || idx == dynamic_pv.len() - 1 {
            return Ok(StageResult {
                candidates: filtered,
                stage: PipelineStage::AlignmentFilter,
                latency: Instant::now().elapsed(),
            });
        }
    }

    // Fallback: return top results from Stage 3 without filtering
    Ok(StageResult::passthrough(candidates.into_iter().take(config.k).collect()))
}
```

---

## Part 9: Constitution Modifications Required

### ARCH-03 (Modified)

**Current:**
```yaml
ARCH-03: "Autonomous operation - goals emerge from data, no manual set_north_star()"
```

**Modified:**
```yaml
ARCH-03: |
  Autonomous operation:
  - Multiple goals emerge via hierarchical clustering
  - No manual goal setting (set_north_star() remains forbidden)
  - Topic importance derived from frequency + recency
  - System operates from 0 memories with graceful degradation
```

### New Architecture Rules

```yaml
ARCH-10: |
  Progressive feature activation:
  - Tier 0 (0 memories): Storage, status, basic retrieval
  - Tier 2 (3+ memories): Basic clustering enabled
  - Tier 4 (30+ memories): Statistical reliability
  - Tier 6 (500+ memories): Full personalization

ARCH-11: |
  Bias mitigation requirements:
  - Thompson Sampling for exploration-exploitation (15% exploration)
  - IPS weighting for learning signals
  - MMR diversity in retrieval (λ=0.7)
  - EWC for continual learning

ARCH-12: |
  Multi-cluster purpose vectors:
  - min_cluster_size = 3 fingerprints
  - HDBSCAN for batch, BIRCH for online
  - Silhouette score > 0.3 for cluster validity
  - Cascade filtering across importance-ranked goals
```

---

## Part 10: Implementation Roadmap

### Phase 1: Multi-Goal Storage (Week 1)

**Objective:** Enable storing multiple discovered goals while maintaining backward compatibility.

**Tasks:**
1. Create `DynamicPurposeVector` struct with weighted goals
2. Modify `GoalHierarchy` to support multi-root forest
3. Update `auto_bootstrap_north_star` to keep all discovered clusters
4. Add importance metadata per goal

**Files to Modify:**
- `crates/context-graph-core/src/types/fingerprint/purpose.rs`
- `crates/context-graph-mcp/src/handlers/autonomous/bootstrap.rs`
- `crates/context-graph-core/src/autonomous/goal_hierarchy.rs`

### Phase 2: Importance Weighting (Week 1-2)

**Objective:** Implement frequency + recency based importance scoring.

**Tasks:**
1. Implement EWMA tracker per cluster
2. Add BM25 saturation for frequency
3. Implement Wilson Score for cold start
4. Add burst detection and damping

**Files to Create:**
- `crates/context-graph-core/src/importance/mod.rs`
- `crates/context-graph-core/src/importance/ewma.rs`
- `crates/context-graph-core/src/importance/scoring.rs`

### Phase 3: Bias Mitigation (Week 2)

**Objective:** Implement Thompson Sampling, IPS, and MMR.

**Tasks:**
1. Create `ThompsonSampler` struct
2. Implement `PropensityEstimator`
3. Add MMR diversity selection
4. Integrate into retrieval pipeline

**Files to Create:**
- `crates/context-graph-core/src/bias/thompson.rs`
- `crates/context-graph-core/src/bias/ips.rs`
- `crates/context-graph-core/src/bias/mmr.rs`

### Phase 4: Progressive Activation (Week 2-3)

**Objective:** Make system work from 0 memories.

**Tasks:**
1. Implement 6-tier maturity model
2. Add graceful degradation to Stage 4
3. Create default/neutral values for Tier 0-2
4. Add capability status reporting

**Files to Modify:**
- `crates/context-graph-storage/src/teleological/search/pipeline/stages.rs`
- `crates/context-graph-mcp/src/handlers/autonomous/bootstrap.rs`

### Phase 5: Multi-Goal IC Tracking (Week 3)

**Objective:** Track identity continuity across multiple goals.

**Tasks:**
1. Create `MultiGoalIdentityContinuity` struct
2. Implement weighted aggregate IC
3. Update dream trigger to use aggregate IC
4. Add goal switching detection

**Files to Modify:**
- `crates/context-graph-core/src/gwt/ego_node/identity_continuity.rs`
- `crates/context-graph-core/src/gwt/consciousness.rs`

### Phase 6: Hierarchical Clustering (Week 3-4)

**Objective:** Implement HDBSCAN + BIRCH hybrid clustering.

**Tasks:**
1. Integrate HDBSCAN for batch clustering
2. Implement BIRCH CF-trees for online updates
3. Add silhouette score validation
4. Implement cluster lifecycle management

**Files to Create:**
- `crates/context-graph-core/src/clustering/hdbscan.rs`
- `crates/context-graph-core/src/clustering/birch.rs`
- `crates/context-graph-core/src/clustering/manager.rs`

---

## Part 11: Validation Checklist

### Before Deployment

- [ ] Zero-memory system starts successfully
- [ ] Stage 4 passes through candidates when no clusters exist
- [ ] Multi-goal discovery stores all valid clusters (not just max)
- [ ] Importance weighting computes correctly (EWMA + BM25 + Wilson)
- [ ] Thompson Sampling provides 15% exploration budget
- [ ] MMR ensures diversity in results (λ=0.7)
- [ ] DynamicPurposeVector aggregation produces valid [0,1] vectors
- [ ] Multi-goal IC computation works with weighted aggregate
- [ ] Dream consolidation triggers on aggregate IC < 0.5
- [ ] Cascade filtering handles empty results gracefully
- [ ] Goal rebalancing triggers at correct intervals (hourly)
- [ ] Backward compatibility: existing queries still work

### Test Coverage Required

| Test Category | Scenarios |
|---------------|-----------|
| Bootstrap | 0, 1, 3, 5, 10, 30, 100, 500 memories |
| Retrieval | Per-goal, cascade, empty-result, diverse |
| IC Tracking | Single goal vs multi-goal equivalence |
| Rebalancing | Top goal changes, weights update |
| Bias | New topic gets fair exposure |
| Regression | All existing tests pass |

---

## Part 12: Summary

### What Changes

| Component | Current | Target |
|-----------|---------|--------|
| Goal Discovery | Single North Star via K-means | Multi-cluster via HDBSCAN+BIRCH |
| Purpose Vector | Single 13D vector | Dynamic portfolio of weighted vectors |
| Importance | Confidence-based (discovery) | Frequency + Recency (EWMA, BM25) |
| Zero-Memory | FAILS with error | Graceful degradation, pass-through |
| Bias Mitigation | None | Thompson Sampling, IPS, MMR, EWC |
| IC Tracking | Single scalar | Multi-goal weighted aggregate |
| Stage 4 Filter | Hard-coded single PV | Cascade filter across importance-ranked goals |

### What Stays the Same

| Component | Status |
|-----------|--------|
| 13-embedder TeleologicalFingerprint | 100% preserved |
| 5-stage retrieval pipeline structure | 80% preserved (Stage 4 modified) |
| GWT consciousness equation | 100% preserved (D factor from aggregate PV) |
| Kuramoto oscillators | 100% preserved |
| RocksDB storage | 100% preserved (metadata expanded) |
| Dream consolidation triggers | 95% preserved (uses aggregate IC) |

### Key Metrics

| Metric | Target |
|--------|--------|
| Bootstrap success at 0 memories | 100% |
| New topic fair exposure rate | 15% (exploration budget) |
| Cluster formation threshold | 3 fingerprints |
| IC crisis threshold | aggregate < 0.5 |
| Feature activation tiers | 6 levels |
| Importance half-life | 30-60 days (configurable) |

---

## References

### Hierarchical Clustering
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)
- [BIRCH - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)
- [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

### Cold Start / Graceful Degradation
- [Cold Start in Recommender Systems](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems))
- [NCHS Reliability of Estimates](https://wwwn.cdc.gov/nchs/nhanes/tutorials/reliabilityofestimates.aspx)

### Bias Mitigation
- [Thompson Sampling Tutorial - Stanford](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
- [Unbiased Learning to Rank - SIGIR](https://dl.acm.org/doi/10.1145/3209978.3209986)
- [Elastic Weight Consolidation - PNAS](https://www.pnas.org/doi/10.1073/pnas.1611835114)

### Importance Scoring
- [EWMA - Corporate Finance Institute](https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/exponentially-weighted-moving-average-ewma/)
- [BM25 Retrieval](https://www.emergentmind.com/topics/bm25-retrieval)
- [Wilson Score - Evan Miller](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html)

### Emergent Goal Discovery
- [Multi-armed Bandit - Wikipedia](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [Multiple Goal Management - Springer](https://link.springer.com/article/10.1007/s10588-023-09377-4)
