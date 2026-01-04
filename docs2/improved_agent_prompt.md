# AI Agent Task Specification: Claude Flow Orchestration System

## METADATA
- Task ID: CFLOW-001
- Project: steveprog/docs3
- Priority: CRITICAL
- Risk Level: HIGH
- Execution Model: Synchronous Subagent Orchestration

---

## 1. INTENT & PURPOSE

### High-Level Goal
Execute a complex multi-agent workflow using Claude Flow to coordinate multiple subagents that share memory, work synchronously, and collectively accomplish the specified objective as a unified system.

### Business Context
This task requires orchestrating multiple AI subagents through the Claude Flow system to complete work that exceeds the capacity of a single agent. Agents must share state through the memory system, execute in correct dependency order, and produce verifiable outputs with physical evidence of completion.

### Success Definition
When this task is complete:
1. All subagents have executed in correct dependency order
2. All memory handoffs between agents are documented and verified
3. All outputs exist in their expected locations (database, files, tables, etc.)
4. The sherlock-holmes verification agent confirms 100% task completion
5. Physical evidence of all claimed work exists and has been manually inspected

---

## 2. CONTEXT & ENVIRONMENT

### Technical Stack
- Configuration: `.env` file contains all required credentials
- Database Guide: `/home/cabdru/steveprog/docs3/supabaseguide.md`
- Constitution: `/home/cabdru/steveprog/docs3/constitution.md`
- Orchestration Guide: `/home/cabdru/steveprog/docs3/claudeflow.md`

### Required Pre-Reading
Before any execution, the orchestrating agent MUST read these documents in order:
1. `/home/cabdru/steveprog/docs3/constitution.md` - Governance rules
2. `/home/cabdru/steveprog/docs3/claudeflow.md` - Orchestration patterns
3. `/home/cabdru/steveprog/docs3/supabaseguide.md` - Database configuration

### Environment Constraints
- All database credentials are in `.env` - DO NOT hardcode credentials
- Use real Supabase instance - NO mock databases
- Network calls must succeed or fail with clear errors

---

## 3. DETAILED REQUIREMENTS

### Functional Requirements
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | Read and parse all listed documentation files | MUST | Agent confirms understanding of each document before proceeding |
| FR-2 | Execute subagents synchronously (not in parallel) | MUST | Each subagent completes before next starts |
| FR-3 | Maintain correct dependency order | MUST | No agent launches until all its dependencies have completed |
| FR-4 | Share state via memory system | MUST | Each subagent documents which memories it created and where |
| FR-5 | Verify subagent outputs personally | MUST | Orchestrator validates subagent claims against actual system state |
| FR-6 | Use real data in all tests | MUST | Zero mock data, zero stubs, zero fakes |
| FR-7 | Fail fast on errors | MUST | No silent failures, no workarounds, no fallbacks |
| FR-8 | Execute sherlock-holmes verification | MUST | Final verification pass identifies any incomplete items |

### Non-Functional Requirements
| ID | Category | Requirement | Measurement |
|----|----------|-------------|-------------|
| NFR-1 | Reliability | Every error must be logged with full context | Stack trace, input state, expected vs actual |
| NFR-2 | Traceability | Every subagent reports memory locations used/created | Memory key names documented in output |
| NFR-3 | Verifiability | All outputs must be physically verifiable | Database queries, file existence checks, API responses |
| NFR-4 | Integrity | Tests must fail when system is broken | No passing tests on broken states |

---

## 4. SCOPE BOUNDARIES

### Explicitly In Scope
- [x] Reading and applying guidance from constitution.md
- [x] Reading and applying guidance from claudeflow.md
- [x] Reading and applying guidance from supabaseguide.md
- [x] Spawning and coordinating subagents via Claude Flow
- [x] Managing memory handoffs between agents
- [x] Verifying physical existence of all outputs
- [x] Running sherlock-holmes agent for final verification
- [x] Fixing any issues sherlock-holmes identifies

### Explicitly Out of Scope - DO NOT MODIFY
- [ ] Creating workarounds when things fail
- [ ] Using mock data or stubs in any form
- [ ] Creating fallback behavior that masks failures
- [ ] Accepting subagent claims without verification
- [ ] Launching agents out of dependency order
- [ ] Running agents in parallel/asynchronously
- [ ] Maintaining backwards compatibility with broken states

### Boundary Conditions
- IF a subagent fails → STOP, log complete error context, DO NOT proceed
- IF a dependency is missing → STOP, report which dependency is needed, DO NOT create a workaround
- IF verification finds missing output → STOP, report discrepancy, DO NOT claim success
- IF database operation fails → STOP, log connection details (not credentials), DO NOT use mock data

---

## 5. TECHNICAL CONSTRAINTS

### Must Use
- Real Supabase database via credentials in `.env`
- Claude Flow orchestration system as documented
- Memory system for inter-agent communication
- Synchronous execution order
- sherlock-holmes subagent for final verification

### Must Not Use
- Mock data of any kind
- Stub implementations
- Fake responses
- Parallel/async subagent execution
- Hardcoded credentials
- Fallback behavior that masks failures
- Tests that pass when functionality is broken

### Must Preserve
- Error propagation (errors bubble up, not swallowed)
- Execution order integrity (dependencies before dependents)
- Data integrity (real data only)
- Verification chain (every claim verified against physical evidence)

---

## 6. EDGE CASES & ERROR HANDLING

### Edge Cases
| Scenario | Expected Behavior |
|----------|-------------------|
| Subagent returns success but output doesn't exist | FAIL - Report "Subagent [X] claimed completion but [output Y] not found at [location Z]" |
| Database connection fails | FAIL - Log full connection error, connection string (masked), timestamp |
| Memory key not found | FAIL - Report "Expected memory key [X] from agent [Y] not found" |
| Document file not found | FAIL - Report exact path attempted, list available files in directory |
| Subagent reports wrong memory location | FAIL - Report expected vs actual memory locations |
| Test passes but feature is broken | FAIL - This indicates test is invalid, report test methodology flaw |

### Error Handling Requirements
| Error Type | Detection | Response | Recovery |
|------------|-----------|----------|----------|
| Configuration Error | .env missing or malformed | Log exact missing/invalid keys, STOP | None - fix config manually |
| Database Error | Connection/query failure | Log full error with query (not credentials), STOP | None - fix database issue |
| Subagent Failure | Non-zero exit or exception | Log subagent ID, error message, full state, STOP | None - debug subagent |
| Verification Failure | Output not found where claimed | Log expected location, actual state, who claimed it, STOP | None - investigate discrepancy |
| Memory System Error | Read/write failure | Log operation attempted, key, payload size, STOP | None - fix memory system |

### Error Logging Format
Every error MUST include:
```
[ERROR] [TIMESTAMP] [AGENT_ID]
- Operation: [what was being attempted]
- Expected: [what should have happened]
- Actual: [what actually happened]
- Context: [relevant state at time of failure]
- Stack Trace: [full trace if available]
- Recovery: NONE - Fix the root cause
```

---

## 7. QUALITY REQUIREMENTS

### Code Quality
- [ ] All error paths produce actionable log messages
- [ ] No silent exception swallowing
- [ ] No generic catch-all error handlers that hide specifics
- [ ] Every external call has explicit error handling
- [ ] No "TODO" or "FIXME" comments in shipped code

### Testing Requirements
- [ ] Tests use real data from real database
- [ ] Tests MUST fail when functionality is broken
- [ ] Tests verify actual system state, not return values alone
- [ ] No mocking, stubbing, or faking
- [ ] Each test documents what physical evidence it verifies

### Verification Requirements
- [ ] Every claimed output is verified to physically exist
- [ ] Database writes verified by subsequent read
- [ ] File creation verified by file system check
- [ ] API responses verified by inspecting actual response content
- [ ] Memory writes verified by subsequent memory read

---

## 8. FORBIDDEN ACTIONS

The following actions are explicitly PROHIBITED:

1. **DO NOT** create workarounds when something fails
2. **DO NOT** use mock data, stubs, or fakes under any circumstance
3. **DO NOT** create fallback behavior that masks failures
4. **DO NOT** accept subagent claims without personal verification
5. **DO NOT** launch agents out of dependency order
6. **DO NOT** run agents in parallel or asynchronously
7. **DO NOT** write tests that pass when functionality is broken
8. **DO NOT** swallow exceptions or errors silently
9. **DO NOT** proceed when a dependency is missing or failed
10. **DO NOT** skip reading the required documentation files
11. **DO NOT** trust return values alone without checking source of truth
12. **DO NOT** claim success without physical evidence
13. **DO NOT** hardcode any credentials
14. **DO NOT** maintain backwards compatibility with broken states

---

## 9. SUBAGENT ORCHESTRATION PROTOCOL

### Launch Sequence Rules
1. Build dependency graph from task requirements
2. Identify agents with no unmet dependencies (ready to launch)
3. Launch ONE agent at a time (synchronous execution)
4. Wait for agent completion
5. Verify agent's claimed outputs actually exist
6. Record memory locations agent created
7. Mark agent as complete
8. Repeat from step 2 until all agents complete or failure occurs

### Memory Handoff Protocol
Each subagent MUST report:
```
AGENT_ID: [identifier]
MEMORIES_READ: [list of memory keys consumed]
MEMORIES_WRITTEN: [list of memory keys created]
MEMORY_LOCATIONS: [exact paths/keys where next agent should look]
OUTPUT_LOCATIONS: [physical locations of outputs created]
```

The orchestrator MUST:
1. Record all memory locations from each agent
2. Pass relevant memory locations to dependent agents
3. Verify memories exist before launching dependent agents
4. Include memory location information in agent launch instructions

### Subagent Trust Protocol
```
RULE: NEVER trust subagent claims without verification

VERIFICATION STEPS:
1. Receive subagent completion report
2. Parse claimed outputs and locations
3. Independently verify each claim:
   - Database records: Execute query, confirm rows exist
   - Files: Check filesystem, confirm file exists with expected content
   - Memory: Read memory key, confirm data exists
   - API state: Call API, confirm expected state
4. IF any claim fails verification → Report discrepancy, STOP
5. IF all claims verified → Proceed to next agent
```

---

## 10. FULL STATE VERIFICATION PROTOCOL

### Source of Truth Identification
Before executing any logic:
1. Identify where the final result will be stored
2. Document the "source of truth" location explicitly
3. Plan the verification query/check to confirm success

### Execute & Inspect Pattern
```
1. Execute the operation
2. DO NOT trust the return value alone
3. Perform independent READ operation on source of truth
4. Compare expected state vs actual state
5. Log both states for audit trail
6. IF mismatch → FAIL with detailed discrepancy report
```

### Boundary & Edge Case Audit
For each major operation, manually simulate 3 edge cases:
```
FOR EACH edge_case IN [empty_input, maximum_limit, invalid_format]:
    1. Log: "TESTING EDGE CASE: [edge_case]"
    2. Log: "STATE BEFORE: [capture system state]"
    3. Execute operation with edge case input
    4. Log: "STATE AFTER: [capture system state]"
    5. Log: "EXPECTED: [what should have happened]"
    6. Log: "ACTUAL: [what did happen]"
    7. IF expected != actual → FAIL with detailed report
```

### Evidence of Success Requirement
Every successful operation MUST produce:
```
SUCCESS EVIDENCE LOG:
- Operation: [what was done]
- Source of Truth: [where result is stored]
- Verification Query: [how we checked]
- Expected Result: [what we expected to find]
- Actual Result: [what we actually found]
- Physical Proof: [screenshot/query result/file contents]
- Timestamp: [when verified]
- Verified By: [agent ID]
```

---

## 11. SHERLOCK-HOLMES FINAL VERIFICATION

### Invocation Requirement
The sherlock-holmes subagent MUST be invoked as the final step after all other agents complete.

### Sherlock-Holmes Mandate
```
AGENT: sherlock-holmes
MISSION: Comprehensive verification that entire task is complete

INSTRUCTIONS:
1. Review all claimed completions from all agents
2. Independently verify EVERY claim against physical evidence
3. Check database for expected records
4. Check filesystem for expected files
5. Check memory system for expected state
6. Check API endpoints for expected responses
7. Identify ANY discrepancy or missing item
8. Report findings with evidence

OUTPUT FORMAT:
{
  "verification_status": "PASS" | "FAIL",
  "items_verified": [list of verified items with evidence],
  "discrepancies_found": [list of issues with details],
  "missing_items": [list of expected but not found],
  "recommendations": [how to fix any issues]
}
```

### Post-Sherlock Protocol
```
IF sherlock-holmes.verification_status == "FAIL":
    1. Parse discrepancies and missing items
    2. Determine root cause of each issue
    3. Fix each identified issue
    4. Re-run sherlock-holmes verification
    5. Repeat until PASS or determine issue is unfixable
    6. IF unfixable → Report complete failure with all details

IF sherlock-holmes.verification_status == "PASS":
    1. Document final success state
    2. Log all evidence collected
    3. Mark task as complete
```

---

## 12. ACCEPTANCE CRITERIA CHECKLIST

Before marking task complete, verify ALL of the following:

### Documentation Phase
- [ ] constitution.md has been read and understood
- [ ] claudeflow.md has been read and understood  
- [ ] supabaseguide.md has been read and understood
- [ ] Orchestration strategy has been planned based on documentation

### Execution Phase
- [ ] All subagents executed synchronously (one at a time)
- [ ] Dependency order was respected (no agent launched before its dependencies)
- [ ] Each subagent reported its memory locations
- [ ] Each subagent's outputs were personally verified
- [ ] No mock data was used anywhere
- [ ] No workarounds were created for failures

### Verification Phase
- [ ] Source of truth identified for each operation
- [ ] Independent read operations confirmed all writes
- [ ] Edge cases tested with before/after state logging
- [ ] Physical evidence collected for all claimed outputs
- [ ] sherlock-holmes verification completed
- [ ] All sherlock-holmes findings addressed

### Quality Phase
- [ ] All errors produced actionable log messages
- [ ] No silent failures occurred
- [ ] Tests fail when functionality is broken (verified)
- [ ] No fallback behavior masked any issues

---

## 13. EMERGENCY CLAUSES

### The Fail-Fast Clause
```
IF any operation fails for any reason:
    1. DO NOT create a workaround
    2. DO NOT use mock data as fallback
    3. DO NOT proceed to next step
    4. DO log complete error context
    5. DO report exactly what failed and why
    6. DO stop execution immediately
    7. DO provide actionable information for debugging
```

### The No-Cover-Up Clause
```
Tests and verification exist to EXPOSE problems, not hide them.
IF a test passes but functionality is broken:
    - The test is invalid
    - Report the test methodology flaw
    - DO NOT claim success based on passing invalid test
```

### The Verification-Required Clause
```
No claim of success is valid without physical evidence.
"It returned success" is NOT sufficient verification.
"I confirmed the record exists in the database" IS sufficient verification.
"I confirmed the file exists at path X with content Y" IS sufficient verification.
```

### The Intent Override Clause
```
The intent of this specification is:
- To produce verifiable, working functionality
- Using real systems and real data
- With complete transparency about what succeeded and what failed
- And physical evidence of all claimed outputs

Any interpretation that would hide failures, use fake data, or claim 
success without evidence violates this intent and is PROHIBITED.
```

### The Clarification Clause
```
IF uncertain about any requirement:
    1. DO NOT assume
    2. DO NOT create a workaround
    3. STOP and request clarification
    4. Document what is unclear and what options exist
```

---

## 14. FINAL VERIFICATION

This task shall be completed in whatever manner an experienced senior engineer who:
- Deeply understands distributed systems and multi-agent coordination
- Prioritizes correctness over convenience
- Values transparency over false confidence  
- Insists on physical evidence of success
- Would never ship a feature they haven't personally verified

Would choose to execute it.

Any shortcut that would result in:
- Unverified claims of success
- Mock data hiding real failures
- Tests passing on broken functionality
- Workarounds masking root causes
- Subagent claims accepted without verification

Is PROHIBITED and violates the intent of this specification.

---

## APPENDIX: Quick Reference

### When Something Fails
```
1. STOP immediately
2. DO NOT create workaround
3. Log: What failed, Why it failed, Full context, Stack trace
4. Report: Clear description of failure
5. Wait: For root cause to be addressed
```

### When Subagent Completes
```
1. Receive completion report
2. Parse claimed outputs
3. Verify EACH claim independently
4. Check physical evidence exists
5. IF all verified → Record and proceed
6. IF any unverified → STOP and report
```

### When Task Completes
```
1. Run sherlock-holmes verification
2. Address any findings
3. Collect all evidence
4. Confirm ALL acceptance criteria met
5. Document final state with proof
```
