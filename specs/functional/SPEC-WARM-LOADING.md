# Warm Model Loading System - Functional Specification

```xml
<functional_spec id="SPEC-WARM-LOADING" version="1.0">
<metadata>
  <title>Warm Model Loading System - VRAM-Resident Embedding Models</title>
  <module>Module 3 Extension</module>
  <status>draft</status>
  <owner>Lead Architect / CUDA Engineer</owner>
  <created>2026-01-03</created>
  <last_updated>2026-01-03</last_updated>
  <related_specs>
    <spec_ref>SPEC-EMBED (Module 3) - 12-Model Embedding Pipeline</spec_ref>
    <spec_ref>SPEC-CUDA (Module 7) - CUDA Optimization</spec_ref>
  </related_specs>
  <dependencies>
    <dependency>Module 2 (Core Infrastructure) - Storage layer, configuration</dependency>
    <dependency>Module 7 (CUDA Optimization) - GPU memory management</dependency>
  </dependencies>
  <critical_constraints>
    <constraint>NO FALLBACKS - System fails fast with clear errors</constraint>
    <constraint>NO MOCKS - All tests use real GPU, real models, real data</constraint>
    <constraint>NO BACKWARDS COMPATIBILITY - Clean implementation only</constraint>
  </critical_constraints>
</metadata>

<overview>
The Warm Model Loading System ensures all 12 embedding models plus the FuseMoE fusion layer are pre-loaded into GPU VRAM at system startup and remain resident throughout operation. This eliminates cold-start latency for embedding requests and provides predictable, consistent performance.

Key Objectives:
1. **Startup Loading**: All 12 models loaded into VRAM before system accepts requests
2. **Persistent Residency**: Models never unloaded during normal operation
3. **Health Verification**: API to verify all models are warm and operational
4. **Hard Failure**: System refuses to start if any model fails to load
5. **VRAM Budget**: Strict enforcement of memory limits with 8GB headroom

Target Hardware: RTX 5090 32GB VRAM
- Available VRAM: ~24GB (8GB reserved for headroom and working memory)
- Total model memory: ~5.5GB FP32 or ~2.8GB FP16/FP8

Model Inventory:
| Model | Size (FP32) | Size (FP16) | Dimension |
|-------|-------------|-------------|-----------|
| E1 Semantic | 1.4GB | 700MB | 1024D |
| E2 Temporal-Recent | 15MB | 7.5MB | 512D |
| E3 Temporal-Periodic | 15MB | 7.5MB | 512D |
| E4 Temporal-Positional | 15MB | 7.5MB | 512D |
| E5 Causal | 650MB | 325MB | 768D |
| E6 Sparse | 550MB | 275MB | ~30K |
| E7 Code | 550MB | 275MB | 1536D |
| E8 Graph/GNN | 120MB | 60MB | 1536D |
| E9 HDC | 60MB | 30MB | 10K-bit |
| E10 Multimodal | 1.6GB | 800MB | 1024D |
| E11 Entity/TransE | 120MB | 60MB | 256D |
| E12 Late-Interaction | 450MB | 225MB | 128D/tok |
| FuseMoE Layer | 1.7GB | 850MB | 4096D->1536D |
| **Total** | **~7.2GB** | **~3.6GB** | - |

This specification defines the WHAT and WHY of warm loading. The Technical Specification (TECH-WARM-LOADING) will define the HOW.
</overview>

<!-- ============================================================================ -->
<!-- USER STORIES -->
<!-- ============================================================================ -->

<user_stories>

<story id="US-WARM-01" priority="must-have">
  <narrative>
    As a system operator
    I want all embedding models loaded into VRAM at startup
    So that the first embedding request has the same latency as subsequent requests
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-WARM-01-01">
      <given>The system is starting up</given>
      <when>Initialization completes successfully</when>
      <then>All 12 embedding models are resident in VRAM</then>
    </criterion>
    <criterion id="AC-WARM-01-02">
      <given>All models are loaded</given>
      <when>The first embedding request arrives</when>
      <then>Latency is within 5% of steady-state latency (no cold-start penalty)</then>
    </criterion>
    <criterion id="AC-WARM-01-03">
      <given>The system is starting up</given>
      <when>FuseMoE fusion layer loading completes</when>
      <then>FuseMoE weights and gating network are resident in VRAM</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-WARM-02" priority="must-have">
  <narrative>
    As a system operator
    I want models to remain in VRAM during operation
    So that embedding performance is consistent and predictable
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-WARM-02-01">
      <given>Models are loaded and system is operational</given>
      <when>The system operates for 24+ hours under load</when>
      <then>No model is ever unloaded or swapped out of VRAM</then>
    </criterion>
    <criterion id="AC-WARM-02-02">
      <given>Models are loaded</given>
      <when>Memory pressure increases from other operations</when>
      <then>Embedding models maintain VRAM residency (protected allocation)</then>
    </criterion>
    <criterion id="AC-WARM-02-03">
      <given>Models are loaded</given>
      <when>VRAM usage is queried</when>
      <then>Model memory allocations are marked as non-evictable</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-WARM-03" priority="must-have">
  <narrative>
    As a system operator
    I want a health check API to verify model warmth
    So that I can confirm all models are ready before routing traffic
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-WARM-03-01">
      <given>Health check endpoint is called</given>
      <when>All models are warm and operational</when>
      <then>Response includes per-model status showing "warm" state</then>
    </criterion>
    <criterion id="AC-WARM-03-02">
      <given>Health check endpoint is called</given>
      <when>Any model is not warm</when>
      <then>Response indicates which specific model(s) are not warm</then>
    </criterion>
    <criterion id="AC-WARM-03-03">
      <given>Health check is requested</given>
      <when>Verifying model readiness</when>
      <then>Each model's VRAM address and allocation size is reported</then>
    </criterion>
    <criterion id="AC-WARM-03-04">
      <given>Health check is requested</given>
      <when>Verifying inference readiness</when>
      <then>A test inference is run on each model to confirm operational status</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-WARM-04" priority="must-have">
  <narrative>
    As a system operator
    I want the system to fail at startup if any model cannot be loaded
    So that I am immediately aware of configuration or hardware problems
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-WARM-04-01">
      <given>System is starting up</given>
      <when>Any model file is missing or corrupted</when>
      <then>System exits with error code and specific error message identifying the failed model</then>
    </criterion>
    <criterion id="AC-WARM-04-02">
      <given>System is starting up</given>
      <when>VRAM is insufficient for all models</when>
      <then>System exits with error code showing required vs available VRAM</then>
    </criterion>
    <criterion id="AC-WARM-04-03">
      <given>System is starting up</given>
      <when>CUDA initialization fails</when>
      <then>System exits with error code and full GPU state dump</then>
    </criterion>
    <criterion id="AC-WARM-04-04">
      <given>System is starting up</given>
      <when>Model validation fails (wrong dimensions, corrupted weights)</when>
      <then>System exits with validation error details</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-WARM-05" priority="must-have">
  <narrative>
    As a system operator
    I want VRAM budget enforcement with hard limits
    So that the system operates within hardware constraints
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-WARM-05-01">
      <given>System configuration specifies VRAM budget</given>
      <when>Models are loaded</when>
      <then>Total model VRAM usage does not exceed configured budget (default: 24GB)</then>
    </criterion>
    <criterion id="AC-WARM-05-02">
      <given>VRAM budget is configured</given>
      <when>Model loading would exceed budget</when>
      <then>System fails with clear error showing budget vs required</then>
    </criterion>
    <criterion id="AC-WARM-05-03">
      <given>Models are loaded within budget</given>
      <when>Querying VRAM usage</when>
      <then>Actual usage matches predicted usage within 5%</then>
    </criterion>
    <criterion id="AC-WARM-05-04">
      <given>Headroom is configured (default: 8GB)</given>
      <when>Working memory is needed for inference</when>
      <then>Headroom is always available for temporary allocations</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-WARM-06" priority="must-have">
  <narrative>
    As a developer
    I want comprehensive error logging with GPU state
    So that I can diagnose any loading or operational failures
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-WARM-06-01">
      <given>Any GPU error occurs</given>
      <when>Error is logged</when>
      <then>Log includes: CUDA error code, GPU memory state, model loading progress, driver version</then>
    </criterion>
    <criterion id="AC-WARM-06-02">
      <given>Model loading fails</given>
      <when>Error is reported</when>
      <then>Error includes: model ID, file path, expected vs actual file size, load progress percentage</then>
    </criterion>
    <criterion id="AC-WARM-06-03">
      <given>VRAM allocation fails</given>
      <when>Error is reported</when>
      <then>Error includes: requested size, available VRAM, fragmentation state, allocation history</then>
    </criterion>
    <criterion id="AC-WARM-06-04">
      <given>Any failure occurs during startup</given>
      <when>System exits</when>
      <then>Complete diagnostic dump is written to configured log path</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-WARM-07" priority="must-have">
  <narrative>
    As a developer
    I want real integration tests verifying VRAM residency
    So that I can prove the warm loading system works correctly
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-WARM-07-01">
      <given>Integration tests are run</given>
      <when>Testing VRAM residency</when>
      <then>Tests use REAL GPU with actual model files (no mocks)</then>
    </criterion>
    <criterion id="AC-WARM-07-02">
      <given>Residency test is run</given>
      <when>Verifying model location</when>
      <then>Test queries nvidia-smi or CUDA API to confirm VRAM allocation</then>
    </criterion>
    <criterion id="AC-WARM-07-03">
      <given>Inference test is run</given>
      <when>Comparing cold vs warm latency</when>
      <then>Warm inference latency is within 5% of subsequent inferences</then>
    </criterion>
    <criterion id="AC-WARM-07-04">
      <given>Stress test is run</given>
      <when>Running 10,000 consecutive inferences</when>
      <then>No model is ever observed outside VRAM</then>
    </criterion>
  </acceptance_criteria>
</story>

</user_stories>

<!-- ============================================================================ -->
<!-- FUNCTIONAL REQUIREMENTS -->
<!-- ============================================================================ -->

<requirements>

<!-- ==================== Startup Loading Requirements ==================== -->

<requirement id="REQ-WARM-001" story_ref="US-WARM-01" priority="must">
  <description>The system SHALL load all 12 embedding models into VRAM during system initialization, before accepting any embedding requests.</description>
  <rationale>Cold-start latency is unacceptable for production use. All models must be ready before the system is operational.</rationale>
  <acceptance_criteria>
    <criterion>All 12 models loaded: E1-E12 as defined in SPEC-EMBED</criterion>
    <criterion>Loading occurs during init phase, blocking startup completion</criterion>
    <criterion>No embedding API responds until all models are warm</criterion>
    <criterion>Loading order: largest models first to fail fast on VRAM issues</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-002" story_ref="US-WARM-01" priority="must">
  <description>The system SHALL load the FuseMoE fusion layer weights and gating network into VRAM during system initialization.</description>
  <rationale>FuseMoE is required for embedding fusion and must also be warm.</rationale>
  <acceptance_criteria>
    <criterion>FuseMoE expert weights loaded (~1.7GB FP32)</criterion>
    <criterion>Gating network weights loaded</criterion>
    <criterion>Output projection layer loaded (4096D->1536D)</criterion>
    <criterion>CAME-AB cross-attention weights loaded</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-003" story_ref="US-WARM-01" priority="must">
  <description>The first embedding request after startup SHALL have latency within 5% of steady-state latency.</description>
  <rationale>Zero cold-start penalty proves warm loading is effective.</rationale>
  <acceptance_criteria>
    <criterion>First request latency measured against mean of requests 100-200</criterion>
    <criterion>Difference must be less than 5% of steady-state mean</criterion>
    <criterion>Measured on real GPU with real model files</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Persistent Residency Requirements ==================== -->

<requirement id="REQ-WARM-004" story_ref="US-WARM-02" priority="must">
  <description>The system SHALL maintain all embedding models in VRAM continuously during operation, never unloading them.</description>
  <rationale>Unpredictable latency from model swapping is unacceptable.</rationale>
  <acceptance_criteria>
    <criterion>Model memory allocations are non-evictable (cudaMallocManaged with preferred location GPU or cudaMalloc)</criterion>
    <criterion>No automatic memory management can evict model weights</criterion>
    <criterion>Models remain resident for entire process lifetime</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-005" story_ref="US-WARM-02" priority="must">
  <description>The system SHALL protect model VRAM allocations from eviction under memory pressure.</description>
  <rationale>Other GPU operations should not cause model eviction.</rationale>
  <acceptance_criteria>
    <criterion>Model allocations use cudaMalloc (device-only) not UVM</criterion>
    <criterion>Working memory pool is separate from model allocations</criterion>
    <criterion>If working memory exhausted, fail operation rather than evict models</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Health Check Requirements ==================== -->

<requirement id="REQ-WARM-006" story_ref="US-WARM-03" priority="must">
  <description>The system SHALL provide a health check API endpoint that returns the warm status of all embedding models.</description>
  <rationale>Operators need to verify system readiness before routing traffic.</rationale>
  <acceptance_criteria>
    <criterion>Endpoint: GET /health/models or MCP tool health.check_models</criterion>
    <criterion>Response includes status for each of 12 models plus FuseMoE</criterion>
    <criterion>Status includes: model_id, is_warm, vram_address, allocation_size_bytes</criterion>
    <criterion>Response latency less than 10ms (no heavy computation)</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-007" story_ref="US-WARM-03" priority="must">
  <description>The health check SHALL perform a test inference on each model to verify operational readiness.</description>
  <rationale>VRAM presence alone does not prove the model can run inference.</rationale>
  <acceptance_criteria>
    <criterion>Test inference uses small fixed input</criterion>
    <criterion>Output is validated against known expected output</criterion>
    <criterion>Any inference failure marks model as not-ready</criterion>
    <criterion>Full health check with inference tests completes in less than 100ms</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Startup Failure Requirements ==================== -->

<requirement id="REQ-WARM-008" story_ref="US-WARM-04" priority="must">
  <description>The system SHALL exit with a non-zero exit code and specific error message if any model fails to load at startup.</description>
  <rationale>Partial loading is not acceptable - all or nothing.</rationale>
  <acceptance_criteria>
    <criterion>Exit code: ERR-WARM-MODEL-LOAD (specific non-zero value)</criterion>
    <criterion>Error message includes: model_id that failed, reason, file path</criterion>
    <criterion>No partial operation - if one model fails, none are used</criterion>
    <criterion>Error written to stderr and log file</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-009" story_ref="US-WARM-04" priority="must">
  <description>The system SHALL exit with specific error if VRAM is insufficient for all models.</description>
  <rationale>Attempting to run with insufficient VRAM would cause failures during operation.</rationale>
  <acceptance_criteria>
    <criterion>Exit code: ERR-WARM-VRAM-INSUFFICIENT</criterion>
    <criterion>Error includes: required_vram, available_vram, model_breakdown</criterion>
    <criterion>Checked before any model loading begins (pre-flight check)</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-010" story_ref="US-WARM-04" priority="must">
  <description>The system SHALL exit with specific error if CUDA initialization fails.</description>
  <rationale>Without CUDA, no GPU operations are possible.</rationale>
  <acceptance_criteria>
    <criterion>Exit code: ERR-WARM-CUDA-INIT</criterion>
    <criterion>Error includes: CUDA error code, driver version, GPU name</criterion>
    <criterion>Full GPU state dump in error output</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-011" story_ref="US-WARM-04" priority="must">
  <description>The system SHALL validate model dimensions and perform integrity checks after loading.</description>
  <rationale>Corrupted or wrong models would produce garbage output.</rationale>
  <acceptance_criteria>
    <criterion>Each model's output dimension verified against expected</criterion>
    <criterion>Weight matrix checksums verified if available</criterion>
    <criterion>Test inference output validated against known reference</criterion>
    <criterion>Exit code ERR-WARM-MODEL-VALIDATION on failure</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== VRAM Budget Requirements ==================== -->

<requirement id="REQ-WARM-012" story_ref="US-WARM-05" priority="must">
  <description>The system SHALL enforce a configurable VRAM budget with default of 24GB for models.</description>
  <rationale>Must leave headroom for working memory on 32GB GPU.</rationale>
  <acceptance_criteria>
    <criterion>Configuration: vram_budget_bytes (default: 24GB)</criterion>
    <criterion>Configuration: vram_headroom_bytes (default: 8GB)</criterion>
    <criterion>Pre-flight check: sum(model_sizes) less than vram_budget</criterion>
    <criterion>System refuses to start if budget exceeded</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-013" story_ref="US-WARM-05" priority="must">
  <description>The system SHALL report predicted VRAM usage before loading and verify actual usage after loading.</description>
  <rationale>Operators need visibility into VRAM consumption.</rationale>
  <acceptance_criteria>
    <criterion>Pre-load: log predicted VRAM for each model</criterion>
    <criterion>Post-load: query actual VRAM usage via CUDA API</criterion>
    <criterion>Alert if actual exceeds predicted by more than 5%</criterion>
    <criterion>Metrics exposed: vram.predicted, vram.actual, vram.available</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Error Logging Requirements ==================== -->

<requirement id="REQ-WARM-014" story_ref="US-WARM-06" priority="must">
  <description>The system SHALL log comprehensive GPU state on any error related to warm loading.</description>
  <rationale>Diagnosis requires full context.</rationale>
  <acceptance_criteria>
    <criterion>CUDA error code and description</criterion>
    <criterion>GPU memory: total, used, free, fragmentation</criterion>
    <criterion>Driver version, runtime version, GPU name, compute capability</criterion>
    <criterion>Model loading progress at time of failure</criterion>
    <criterion>Recent allocation history (last 100 allocations)</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-015" story_ref="US-WARM-06" priority="must">
  <description>The system SHALL produce a complete diagnostic dump on startup failure.</description>
  <rationale>Post-mortem analysis requires complete information.</rationale>
  <acceptance_criteria>
    <criterion>Dump written to configurable path (default: /var/log/context-graph/startup-failure-{timestamp}.dump)</criterion>
    <criterion>Includes: full configuration, GPU state, model loading log, error details</criterion>
    <criterion>Machine-readable format (JSON) for automated analysis</criterion>
    <criterion>Human-readable summary at top of dump</criterion>
  </acceptance_criteria>
</requirement>

<!-- ==================== Integration Test Requirements ==================== -->

<requirement id="REQ-WARM-016" story_ref="US-WARM-07" priority="must">
  <description>The system SHALL include integration tests that verify VRAM residency using real GPU and real model files.</description>
  <rationale>Only real tests prove the system works.</rationale>
  <acceptance_criteria>
    <criterion>Tests require actual CUDA-capable GPU</criterion>
    <criterion>Tests use actual model weight files</criterion>
    <criterion>No mocks, stubs, or simulated VRAM</criterion>
    <criterion>Tests query CUDA API to verify device memory allocation</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-017" story_ref="US-WARM-07" priority="must">
  <description>Integration tests SHALL verify zero cold-start penalty by comparing first inference latency to steady-state.</description>
  <rationale>Core requirement validation.</rationale>
  <acceptance_criteria>
    <criterion>Measure latency of inference #1 (after startup)</criterion>
    <criterion>Measure mean latency of inferences #100-200</criterion>
    <criterion>Assert: inference_1_latency less than 1.05 * mean_steady_state</criterion>
    <criterion>Test runs on real GPU with real models</criterion>
  </acceptance_criteria>
</requirement>

<requirement id="REQ-WARM-018" story_ref="US-WARM-07" priority="must">
  <description>Integration tests SHALL verify models remain resident under sustained load.</description>
  <rationale>Residency must be maintained not just at startup but continuously.</rationale>
  <acceptance_criteria>
    <criterion>Run 10,000 consecutive embedding requests</criterion>
    <criterion>Query VRAM state before and after each batch of 1000</criterion>
    <criterion>Assert: model VRAM addresses unchanged throughout test</criterion>
    <criterion>Assert: no latency spikes indicating swap-out/swap-in</criterion>
  </acceptance_criteria>
</requirement>

</requirements>

<!-- ============================================================================ -->
<!-- EDGE CASES -->
<!-- ============================================================================ -->

<edge_cases>

<edge_case id="EC-WARM-001" req_ref="REQ-WARM-001">
  <scenario>Model file is missing from configured path</scenario>
  <expected_behavior>
    Startup fails immediately with ERR-WARM-MODEL-MISSING.
    Error message includes: model_id, expected_path, suggestion to check configuration.
    NO fallback to stub embeddings.
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-002" req_ref="REQ-WARM-001">
  <scenario>Model file exists but is corrupted (invalid weights)</scenario>
  <expected_behavior>
    Loading proceeds, but validation fails.
    Startup fails with ERR-WARM-MODEL-VALIDATION.
    Error includes: model_id, validation failure details (e.g., NaN weights, wrong shape).
    NO fallback to previous version.
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-003" req_ref="REQ-WARM-009">
  <scenario>GPU has exactly enough VRAM for models but no headroom</scenario>
  <expected_behavior>
    Startup fails with ERR-WARM-VRAM-HEADROOM-INSUFFICIENT.
    Error shows: models_require=X, available=Y, headroom_required=8GB, deficit=Z.
    NO attempt to load with reduced headroom.
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-004" req_ref="REQ-WARM-010">
  <scenario>CUDA driver is too old for required compute capability</scenario>
  <expected_behavior>
    Startup fails with ERR-WARM-CUDA-CAPABILITY.
    Error shows: required_capability=12.0, available_capability=X.Y.
    Suggestion to update drivers.
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-005" req_ref="REQ-WARM-004">
  <scenario>Another process starts using GPU VRAM heavily after our startup</scenario>
  <expected_behavior>
    Our model allocations are protected (cudaMalloc, not UVM).
    Other process may fail or slow down, but our models stay resident.
    If headroom is consumed, our working memory allocations fail gracefully.
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-006" req_ref="REQ-WARM-006">
  <scenario>Health check called during model loading (startup)</scenario>
  <expected_behavior>
    Health check returns "loading" status for each model.
    Response includes: loading_progress percentage, models_loaded count, estimated_time_remaining.
    Clear indication that system is not ready.
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-007" req_ref="REQ-WARM-007">
  <scenario>Model is in VRAM but test inference fails</scenario>
  <expected_behavior>
    Health check marks model as "warm_but_failing".
    Error details included in health response.
    System may attempt automatic recovery (reload model once).
    If reload fails, system enters degraded state (or fails if critical).
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-008" req_ref="REQ-WARM-001">
  <scenario>Multiple GPUs present, models should load on specific GPU</scenario>
  <expected_behavior>
    Configuration specifies target GPU device_id.
    All models load on specified GPU.
    Other GPUs are ignored (no automatic distribution).
    Error if specified GPU is unavailable.
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-009" req_ref="REQ-WARM-011">
  <scenario>Model file has valid format but wrong model architecture</scenario>
  <expected_behavior>
    Validation catches dimension mismatch.
    Startup fails with ERR-WARM-MODEL-DIMENSION-MISMATCH.
    Error shows: expected_dim=X, actual_dim=Y, model_id.
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-010" req_ref="REQ-WARM-014">
  <scenario>CUDA error occurs with unknown error code</scenario>
  <expected_behavior>
    Log includes raw error code and any available description.
    Error handling does not crash even for unknown codes.
    Diagnostic dump still produced with all available information.
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-011" req_ref="REQ-WARM-015">
  <scenario>Diagnostic dump path is not writable</scenario>
  <expected_behavior>
    Attempt write to fallback path (/tmp/context-graph-dump-{timestamp}.json).
    If fallback fails, write to stderr.
    Error about unwritable dump path included in output.
  </expected_behavior>
</edge_case>

<edge_case id="EC-WARM-012" req_ref="REQ-WARM-005">
  <scenario>GPU driver crash and recovery during operation</scenario>
  <expected_behavior>
    System detects lost CUDA context.
    Immediate shutdown with ERR-WARM-CUDA-CONTEXT-LOST.
    No attempt to recover (requires full restart).
    Diagnostic dump produced if possible.
  </expected_behavior>
</edge_case>

</edge_cases>

<!-- ============================================================================ -->
<!-- ERROR STATES -->
<!-- ============================================================================ -->

<error_states>

<error id="ERR-WARM-001" code="1001" category="startup">
  <name>ERR-WARM-MODEL-MISSING</name>
  <condition>Model file not found at configured path</condition>
  <message>Warm loading failed: Model {model_id} not found at {path}</message>
  <recovery>Verify model file exists at configured path. Check configuration file.</recovery>
  <logging>model_id, expected_path, config_source, directory_listing</logging>
  <exit_code>101</exit_code>
</error>

<error id="ERR-WARM-002" code="1002" category="startup">
  <name>ERR-WARM-MODEL-LOAD</name>
  <condition>Model file exists but cannot be loaded (read error, parse error)</condition>
  <message>Warm loading failed: Cannot load model {model_id} from {path}: {reason}</message>
  <recovery>Check file permissions, file integrity. Re-download model if corrupted.</recovery>
  <logging>model_id, path, file_size, bytes_read, parse_error, cuda_error</logging>
  <exit_code>102</exit_code>
</error>

<error id="ERR-WARM-003" code="1003" category="startup">
  <name>ERR-WARM-MODEL-VALIDATION</name>
  <condition>Model loaded but validation failed (wrong dimensions, NaN weights, test inference failed)</condition>
  <message>Warm loading failed: Model {model_id} validation failed: {validation_error}</message>
  <recovery>Model file may be corrupted or incompatible version. Obtain correct model file.</recovery>
  <logging>model_id, expected_dims, actual_dims, test_output, expected_output</logging>
  <exit_code>103</exit_code>
</error>

<error id="ERR-WARM-004" code="1004" category="startup">
  <name>ERR-WARM-VRAM-INSUFFICIENT</name>
  <condition>Total VRAM required exceeds available</condition>
  <message>Warm loading failed: Insufficient VRAM. Required: {required_gb}GB, Available: {available_gb}GB</message>
  <recovery>Use smaller models, enable quantization, or upgrade GPU.</recovery>
  <logging>required_bytes, available_bytes, per_model_breakdown, gpu_info</logging>
  <exit_code>104</exit_code>
</error>

<error id="ERR-WARM-005" code="1005" category="startup">
  <name>ERR-WARM-VRAM-HEADROOM</name>
  <condition>Models fit but insufficient headroom for working memory</condition>
  <message>Warm loading failed: Insufficient headroom. Models: {model_gb}GB, Available: {available_gb}GB, Headroom required: {headroom_gb}GB</message>
  <recovery>Reduce model count, enable quantization, or upgrade GPU.</recovery>
  <logging>model_total, available, headroom_required, headroom_available</logging>
  <exit_code>105</exit_code>
</error>

<error id="ERR-WARM-006" code="1006" category="startup">
  <name>ERR-WARM-CUDA-INIT</name>
  <condition>CUDA initialization failed</condition>
  <message>Warm loading failed: CUDA initialization error {cuda_code}: {cuda_message}</message>
  <recovery>Check GPU driver installation. Verify CUDA compatibility. Check nvidia-smi output.</recovery>
  <logging>cuda_error_code, cuda_error_string, driver_version, runtime_version, gpu_list</logging>
  <exit_code>106</exit_code>
</error>

<error id="ERR-WARM-007" code="1007" category="startup">
  <name>ERR-WARM-CUDA-CAPABILITY</name>
  <condition>GPU compute capability insufficient</condition>
  <message>Warm loading failed: GPU compute capability {actual_cc} insufficient. Required: {required_cc}</message>
  <recovery>Upgrade GPU to one with compute capability 12.0 or higher.</recovery>
  <logging>gpu_name, actual_capability, required_capability, gpu_id</logging>
  <exit_code>107</exit_code>
</error>

<error id="ERR-WARM-008" code="1008" category="runtime">
  <name>ERR-WARM-CUDA-ALLOC</name>
  <condition>CUDA memory allocation failed during model loading</condition>
  <message>Warm loading failed: CUDA allocation failed for model {model_id}. Requested: {size_mb}MB</message>
  <recovery>Check VRAM usage with nvidia-smi. Kill other GPU processes. Restart system if fragmented.</recovery>
  <logging>model_id, requested_bytes, vram_free, vram_total, allocation_history</logging>
  <exit_code>108</exit_code>
</error>

<error id="ERR-WARM-009" code="1009" category="runtime">
  <name>ERR-WARM-CUDA-CONTEXT-LOST</name>
  <condition>CUDA context became invalid during operation (driver crash)</condition>
  <message>Fatal: CUDA context lost. GPU driver may have crashed. System requires restart.</message>
  <recovery>Restart the application. Check GPU driver stability. Monitor GPU temperature.</recovery>
  <logging>last_successful_operation, cuda_context_state, gpu_temperature, driver_logs</logging>
  <exit_code>109</exit_code>
</error>

<error id="ERR-WARM-010" code="1010" category="runtime">
  <name>ERR-WARM-MODEL-DIMENSION-MISMATCH</name>
  <condition>Model output dimension does not match expected</condition>
  <message>Warm loading failed: Model {model_id} output dimension {actual_dim} does not match expected {expected_dim}</message>
  <recovery>Verify correct model version is installed. Check model configuration.</recovery>
  <logging>model_id, expected_dim, actual_dim, model_file_hash</logging>
  <exit_code>110</exit_code>
</error>

</error_states>

<!-- ============================================================================ -->
<!-- TEST PLAN -->
<!-- ============================================================================ -->

<test_plan>

<!-- ==================== Unit Tests ==================== -->

<test_case id="TC-WARM-001" type="unit" req_ref="REQ-WARM-012">
  <description>VRAM budget calculation is accurate</description>
  <preconditions>Model size configuration available</preconditions>
  <inputs>List of model specifications with sizes</inputs>
  <expected>Calculated total matches expected (~5.5GB FP32 or ~2.8GB FP16)</expected>
  <data_requirements>REAL model size specifications, NO hardcoded values</data_requirements>
</test_case>

<test_case id="TC-WARM-002" type="unit" req_ref="REQ-WARM-009">
  <description>Pre-flight VRAM check correctly detects insufficient memory</description>
  <preconditions>CUDA context available</preconditions>
  <inputs>Required VRAM exceeds available (simulated low-memory GPU)</inputs>
  <expected>Returns ERR-WARM-VRAM-INSUFFICIENT with accurate numbers</expected>
  <data_requirements>REAL CUDA API calls</data_requirements>
</test_case>

<test_case id="TC-WARM-003" type="unit" req_ref="REQ-WARM-011">
  <description>Model dimension validation catches mismatched models</description>
  <preconditions>Model loaded into memory</preconditions>
  <inputs>Model with wrong output dimension</inputs>
  <expected>Validation fails with specific dimension mismatch error</expected>
  <data_requirements>REAL model file with intentionally wrong dimensions</data_requirements>
</test_case>

<!-- ==================== Integration Tests ==================== -->

<test_case id="TC-WARM-004" type="integration" req_ref="REQ-WARM-001">
  <description>All 12 models load successfully into VRAM</description>
  <preconditions>RTX 5090 or equivalent GPU available, all model files present</preconditions>
  <inputs>Full model set</inputs>
  <expected>
    All 12 models loaded, VRAM allocations verified via CUDA API,
    health check returns all models as "warm"
  </expected>
  <data_requirements>REAL GPU, REAL model files, NO mocks</data_requirements>
</test_case>

<test_case id="TC-WARM-005" type="integration" req_ref="REQ-WARM-002">
  <description>FuseMoE layer loads successfully into VRAM</description>
  <preconditions>GPU available, FuseMoE weight files present</preconditions>
  <inputs>FuseMoE configuration and weights</inputs>
  <expected>
    FuseMoE weights resident in VRAM,
    gating network weights resident,
    test fusion produces correct output
  </expected>
  <data_requirements>REAL GPU, REAL FuseMoE weights</data_requirements>
</test_case>

<test_case id="TC-WARM-006" type="integration" req_ref="REQ-WARM-003">
  <description>First inference has no cold-start penalty</description>
  <preconditions>System fully started, all models warm</preconditions>
  <inputs>Standard embedding request</inputs>
  <expected>
    first_latency < 1.05 * mean_latency(requests 100-200)
  </expected>
  <data_requirements>REAL GPU, REAL models, REAL timing measurements</data_requirements>
</test_case>

<test_case id="TC-WARM-007" type="integration" req_ref="REQ-WARM-004">
  <description>Models remain resident under sustained load</description>
  <preconditions>System fully started</preconditions>
  <inputs>10,000 consecutive embedding requests</inputs>
  <expected>
    Model VRAM addresses unchanged throughout,
    no latency spikes > 2x mean,
    CUDA memory queries show consistent allocation
  </expected>
  <data_requirements>REAL GPU, REAL sustained load</data_requirements>
</test_case>

<test_case id="TC-WARM-008" type="integration" req_ref="REQ-WARM-006, REQ-WARM-007">
  <description>Health check returns accurate warm status</description>
  <preconditions>System fully started</preconditions>
  <inputs>Health check API call</inputs>
  <expected>
    All 13 components (12 models + FuseMoE) show "warm",
    each has valid VRAM address,
    each has allocation size matching expected,
    test inference passes for each
  </expected>
  <data_requirements>REAL GPU, REAL health check execution</data_requirements>
</test_case>

<test_case id="TC-WARM-009" type="integration" req_ref="REQ-WARM-008">
  <description>Missing model file causes startup failure</description>
  <preconditions>One model file intentionally removed</preconditions>
  <inputs>Startup attempt with missing model</inputs>
  <expected>
    Exit code 101 (ERR-WARM-MODEL-MISSING),
    stderr contains model_id and expected path,
    no partial loading
  </expected>
  <data_requirements>REAL startup process with REAL error handling</data_requirements>
</test_case>

<test_case id="TC-WARM-010" type="integration" req_ref="REQ-WARM-009">
  <description>Insufficient VRAM causes pre-flight failure</description>
  <preconditions>Simulated low-memory constraint (or smaller GPU)</preconditions>
  <inputs>Startup attempt with insufficient VRAM configured</inputs>
  <expected>
    Exit code 104 (ERR-WARM-VRAM-INSUFFICIENT),
    error shows required vs available,
    no model loading attempted
  </expected>
  <data_requirements>REAL CUDA, REAL pre-flight check</data_requirements>
</test_case>

<test_case id="TC-WARM-011" type="integration" req_ref="REQ-WARM-014, REQ-WARM-015">
  <description>Startup failure produces complete diagnostic dump</description>
  <preconditions>Intentional startup failure condition</preconditions>
  <inputs>Trigger any startup failure</inputs>
  <expected>
    Diagnostic dump written to configured path,
    dump is valid JSON,
    dump contains: error details, GPU state, config, timestamps
  </expected>
  <data_requirements>REAL failure, REAL diagnostic generation</data_requirements>
</test_case>

<!-- ==================== Stress Tests ==================== -->

<test_case id="TC-WARM-012" type="stress" req_ref="REQ-WARM-004, REQ-WARM-005">
  <description>24-hour sustained operation maintains VRAM residency</description>
  <preconditions>System fully started on production-class GPU</preconditions>
  <inputs>Continuous embedding requests for 24 hours at 100 req/sec</inputs>
  <expected>
    No model evictions detected,
    latency remains stable (no spikes from reload),
    VRAM allocation addresses constant throughout
  </expected>
  <data_requirements>REAL GPU, REAL 24-hour test</data_requirements>
</test_case>

<test_case id="TC-WARM-013" type="stress" req_ref="REQ-WARM-005">
  <description>Memory pressure from other processes does not evict models</description>
  <preconditions>System fully started</preconditions>
  <inputs>
    Concurrent process allocating VRAM up to headroom limit,
    embedding requests during pressure
  </inputs>
  <expected>
    Models remain resident,
    embedding requests succeed (may use limited working memory),
    other process may fail/slow but our models protected
  </expected>
  <data_requirements>REAL GPU, REAL concurrent memory pressure</data_requirements>
</test_case>

<!-- ==================== Benchmark Tests ==================== -->

<test_case id="TC-WARM-014" type="benchmark" req_ref="REQ-WARM-001, REQ-WARM-002">
  <description>Startup time is acceptable</description>
  <preconditions>Cold start (no prior VRAM allocation)</preconditions>
  <inputs>Full startup with all 13 components</inputs>
  <expected>
    Total startup time < 30 seconds on NVMe storage,
    Total startup time < 60 seconds on SATA SSD,
    Time breakdown logged per model
  </expected>
  <data_requirements>REAL hardware, REAL model files</data_requirements>
</test_case>

<test_case id="TC-WARM-015" type="benchmark" req_ref="REQ-WARM-013">
  <description>Actual VRAM usage matches prediction</description>
  <preconditions>System fully started</preconditions>
  <inputs>Query predicted and actual VRAM</inputs>
  <expected>
    abs(actual - predicted) / predicted < 0.05 (5% tolerance)
  </expected>
  <data_requirements>REAL CUDA memory queries</data_requirements>
</test_case>

</test_plan>

<!-- ============================================================================ -->
<!-- CONSTRAINTS -->
<!-- ============================================================================ -->

<constraints>
  <constraint id="CON-WARM-001">NO fallback to CPU embeddings - GPU is required</constraint>
  <constraint id="CON-WARM-002">NO mock embeddings in any test - all tests use REAL GPU and REAL models</constraint>
  <constraint id="CON-WARM-003">NO automatic model eviction - allocations must be protected</constraint>
  <constraint id="CON-WARM-004">NO partial loading - all models or none</constraint>
  <constraint id="CON-WARM-005">Target hardware: RTX 5090 32GB or equivalent with 24GB usable</constraint>
  <constraint id="CON-WARM-006">Minimum compute capability: 12.0</constraint>
  <constraint id="CON-WARM-007">Maximum model load time: 30 seconds per model (NVMe storage)</constraint>
  <constraint id="CON-WARM-008">Health check latency: < 100ms with test inference</constraint>
  <constraint id="CON-WARM-009">Cold-start penalty: < 5% of steady-state latency</constraint>
  <constraint id="CON-WARM-010">All code must be Rust 2021 edition with stable toolchain</constraint>
  <constraint id="CON-WARM-011">Maximum 500 lines per source file (excluding tests)</constraint>
  <constraint id="CON-WARM-012">All public APIs must have rustdoc comments</constraint>
</constraints>

<!-- ============================================================================ -->
<!-- DEPENDENCIES -->
<!-- ============================================================================ -->

<dependencies>
  <dependency type="module" name="Module 3 (Embedding Pipeline)" version="1.0">
    <provides>
      <item>12 embedding model specifications (E1-E12)</item>
      <item>FuseMoE fusion layer specification</item>
      <item>Model file format and loading interface</item>
    </provides>
  </dependency>
  <dependency type="module" name="Module 7 (CUDA Optimization)" version="1.0">
    <provides>
      <item>CudaContext initialization</item>
      <item>Memory pool management</item>
      <item>Device buffer allocation (non-evictable)</item>
      <item>GPU state querying</item>
    </provides>
  </dependency>
  <dependency type="rust_crate" name="cudarc" version="0.10+" purpose="CUDA device management"/>
  <dependency type="rust_crate" name="safetensors" version="0.4+" purpose="Model weight loading"/>
  <dependency type="rust_crate" name="tracing" version="0.1+" purpose="Logging and diagnostics"/>
  <dependency type="external" name="CUDA 13.1" purpose="GPU compute"/>
  <dependency type="hardware" name="RTX 5090 or equivalent" purpose="32GB VRAM target"/>
</dependencies>

</functional_spec>
```

---

## Appendix A: Model Size Reference

| Model ID | Model Name | FP32 Size | FP16 Size | FP8 Size | Output Dim |
|----------|------------|-----------|-----------|----------|------------|
| E1 | Semantic | 1.4 GB | 700 MB | 350 MB | 1024D |
| E2 | Temporal-Recent | 15 MB | 7.5 MB | 3.75 MB | 512D |
| E3 | Temporal-Periodic | 15 MB | 7.5 MB | 3.75 MB | 512D |
| E4 | Temporal-Positional | 15 MB | 7.5 MB | 3.75 MB | 512D |
| E5 | Causal | 650 MB | 325 MB | 162 MB | 768D |
| E6 | Sparse | 550 MB | 275 MB | 137 MB | ~30K |
| E7 | Code | 550 MB | 275 MB | 137 MB | 1536D |
| E8 | Graph/GNN | 120 MB | 60 MB | 30 MB | 1536D |
| E9 | HDC | 60 MB | 30 MB | 15 MB | 10K-bit |
| E10 | Multimodal | 1.6 GB | 800 MB | 400 MB | 1024D |
| E11 | Entity/TransE | 120 MB | 60 MB | 30 MB | 256D |
| E12 | Late-Interaction | 450 MB | 225 MB | 112 MB | 128D/tok |
| FuseMoE | Fusion Layer | 1.7 GB | 850 MB | 425 MB | 4096->1536 |
| **Total** | | **~7.2 GB** | **~3.6 GB** | **~1.8 GB** | - |

---

## Appendix B: VRAM Budget Calculation

```
Target GPU: RTX 5090 (32 GB VRAM)

Reserved:
  - System/Driver overhead: ~500 MB
  - Working memory headroom: 8 GB
  - Available for models: ~23.5 GB

Model memory (FP16 recommended):
  - 12 embedding models: ~2.8 GB
  - FuseMoE layer: ~850 MB
  - Total model memory: ~3.65 GB

Working memory per batch:
  - Embedding input buffers: ~100 MB
  - Intermediate activations: ~500 MB
  - Output buffers: ~100 MB
  - Scratch space: ~300 MB
  - Total per batch: ~1 GB

Concurrent operations:
  - 4 concurrent batches: ~4 GB working memory
  - Peak working memory: ~4 GB

Total peak usage:
  - Models: 3.65 GB
  - Working: 4 GB
  - Total: ~7.65 GB

Margin of safety:
  - Available: 23.5 GB
  - Peak usage: 7.65 GB
  - Remaining: ~16 GB (comfortable margin)

Recommendation: Use FP16 precision for excellent balance of
performance, memory efficiency, and quality retention.
```

---

## Appendix C: Error Code Reference

| Exit Code | Error Name | Category | Recovery |
|-----------|------------|----------|----------|
| 101 | ERR-WARM-MODEL-MISSING | Startup | Check file paths |
| 102 | ERR-WARM-MODEL-LOAD | Startup | Check file integrity |
| 103 | ERR-WARM-MODEL-VALIDATION | Startup | Get correct model |
| 104 | ERR-WARM-VRAM-INSUFFICIENT | Startup | Smaller models or bigger GPU |
| 105 | ERR-WARM-VRAM-HEADROOM | Startup | Reduce model count |
| 106 | ERR-WARM-CUDA-INIT | Startup | Check driver install |
| 107 | ERR-WARM-CUDA-CAPABILITY | Startup | Upgrade GPU |
| 108 | ERR-WARM-CUDA-ALLOC | Runtime | Check VRAM usage |
| 109 | ERR-WARM-CUDA-CONTEXT-LOST | Runtime | Restart application |
| 110 | ERR-WARM-MODEL-DIMENSION-MISMATCH | Startup | Get correct model version |

---

## Appendix D: Health Check Response Schema

```json
{
  "status": "healthy" | "degraded" | "unhealthy" | "loading",
  "timestamp": "2026-01-03T12:00:00Z",
  "gpu": {
    "name": "NVIDIA GeForce RTX 5090",
    "vram_total_bytes": 34359738368,
    "vram_used_bytes": 7654321000,
    "vram_free_bytes": 26705417368,
    "compute_capability": "12.0",
    "driver_version": "565.57",
    "cuda_version": "13.1"
  },
  "models": [
    {
      "model_id": "E1_Semantic",
      "status": "warm",
      "vram_address": "0x7f8b00000000",
      "allocation_bytes": 734003200,
      "last_inference_latency_us": 4823,
      "test_inference_passed": true
    },
    // ... remaining 11 models ...
  ],
  "fusemoe": {
    "status": "warm",
    "vram_address": "0x7f8b30000000",
    "allocation_bytes": 891289600,
    "gating_network_loaded": true,
    "output_projection_loaded": true,
    "test_fusion_passed": true
  },
  "metrics": {
    "total_model_vram_bytes": 3823000000,
    "startup_duration_ms": 12543,
    "first_inference_latency_ms": 8.2,
    "steady_state_latency_mean_ms": 7.9
  }
}
```

---

## Appendix E: Requirement Traceability Matrix

| Requirement ID | User Story | Test Cases | Priority | Status |
|---------------|------------|------------|----------|--------|
| REQ-WARM-001 | US-WARM-01 | TC-WARM-004 | must | pending |
| REQ-WARM-002 | US-WARM-01 | TC-WARM-005 | must | pending |
| REQ-WARM-003 | US-WARM-01 | TC-WARM-006 | must | pending |
| REQ-WARM-004 | US-WARM-02 | TC-WARM-007, TC-WARM-012 | must | pending |
| REQ-WARM-005 | US-WARM-02 | TC-WARM-013 | must | pending |
| REQ-WARM-006 | US-WARM-03 | TC-WARM-008 | must | pending |
| REQ-WARM-007 | US-WARM-03 | TC-WARM-008 | must | pending |
| REQ-WARM-008 | US-WARM-04 | TC-WARM-009 | must | pending |
| REQ-WARM-009 | US-WARM-04 | TC-WARM-010 | must | pending |
| REQ-WARM-010 | US-WARM-04 | TC-WARM-011 | must | pending |
| REQ-WARM-011 | US-WARM-04 | TC-WARM-003 | must | pending |
| REQ-WARM-012 | US-WARM-05 | TC-WARM-001, TC-WARM-002 | must | pending |
| REQ-WARM-013 | US-WARM-05 | TC-WARM-015 | must | pending |
| REQ-WARM-014 | US-WARM-06 | TC-WARM-011 | must | pending |
| REQ-WARM-015 | US-WARM-06 | TC-WARM-011 | must | pending |
| REQ-WARM-016 | US-WARM-07 | All integration tests | must | pending |
| REQ-WARM-017 | US-WARM-07 | TC-WARM-006 | must | pending |
| REQ-WARM-018 | US-WARM-07 | TC-WARM-007, TC-WARM-012 | must | pending |

---

*Document Version: 1.0*
*Generated: 2026-01-03*
*Specification Agent: Functional Spec Agent #1/6*
