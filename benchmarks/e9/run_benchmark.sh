#!/bin/bash
# E9 HDC Benchmark Runner
#
# This script runs the E9 blind-spot detection benchmarks
# and produces a report comparing E1-only vs E1+E9 search quality.
#
# Prerequisites:
# - MCP server must be running
# - Test memories must be seeded (run seed_memories.sh first)
#
# Usage:
#   ./run_benchmark.sh [--seed] [--verbose]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS_FILE="$SCRIPT_DIR/test_corpus.json"
SEED_FILE="$SCRIPT_DIR/seed_memories.json"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark_$TIMESTAMP.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  E9 HDC Benchmark Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse arguments
SEED=false
VERBOSE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if [ ! -f "$CORPUS_FILE" ]; then
    echo -e "${RED}ERROR: Test corpus not found at $CORPUS_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Test corpus found${NC}"

if [ ! -f "$SEED_FILE" ]; then
    echo -e "${RED}ERROR: Seed memories not found at $SEED_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Seed memories found${NC}"

# Seed memories if requested
if [ "$SEED" = true ]; then
    echo ""
    echo -e "${YELLOW}Seeding test memories...${NC}"

    # Count memories
    MEMORY_COUNT=$(jq '.memories | length' "$SEED_FILE")
    echo "  Found $MEMORY_COUNT memories to seed"

    # This would call the MCP store_memory tool for each memory
    # For now, print instructions
    echo -e "${YELLOW}  NOTE: Manual seeding required via MCP tools${NC}"
    echo "  Use inject_context or store_memory MCP tool to add memories from:"
    echo "    $SEED_FILE"
fi

echo ""
echo -e "${YELLOW}Running benchmarks...${NC}"
echo ""

# Initialize counters
TYPO_TOTAL=0
TYPO_SUCCESS=0
CODE_TOTAL=0
CODE_SUCCESS=0
BASELINE_TOTAL=0
BASELINE_FALSE_POSITIVE=0

# Process each query category
echo -e "${BLUE}=== TYPO CORPUS ===${NC}"
TYPO_QUERIES=$(jq -r '.queries[] | select(.category == "typo") | .query' "$CORPUS_FILE")
for query in $TYPO_QUERIES; do
    ((TYPO_TOTAL++))
    echo -e "  Query: ${YELLOW}$query${NC}"

    if [ "$VERBOSE" = true ]; then
        echo "    Expected: E9 should find correct spelling"
    fi

    # In a full implementation, this would:
    # 1. Call search_robust MCP tool
    # 2. Check if blind_spot_discoveries is non-empty
    # 3. Verify the discovery matches ground truth

    echo "    [Would call search_robust MCP tool]"
done

echo ""
echo -e "${BLUE}=== CODE IDENTIFIER CORPUS ===${NC}"
CODE_QUERIES=$(jq -r '.queries[] | select(.category == "code_id") | .query' "$CORPUS_FILE")
for query in $CODE_QUERIES; do
    ((CODE_TOTAL++))
    echo -e "  Query: ${YELLOW}$query${NC}"

    if [ "$VERBOSE" = true ]; then
        echo "    Expected: E9 should find alternate naming convention"
    fi

    echo "    [Would call search_robust MCP tool]"
done

echo ""
echo -e "${BLUE}=== BASELINE CORPUS ===${NC}"
BASELINE_QUERIES=$(jq -r '.queries[] | select(.category == "baseline") | .query' "$CORPUS_FILE")
for query in $BASELINE_QUERIES; do
    ((BASELINE_TOTAL++))
    echo -e "  Query: ${YELLOW}$query${NC}"

    if [ "$VERBOSE" = true ]; then
        echo "    Expected: E9 should NOT produce false discoveries"
    fi

    echo "    [Would call search_robust MCP tool]"
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Benchmark Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Queries processed:"
echo "  Typo corpus:     $TYPO_TOTAL queries"
echo "  Code ID corpus:  $CODE_TOTAL queries"
echo "  Baseline corpus: $BASELINE_TOTAL queries"
echo ""
echo -e "${YELLOW}NOTE: Full benchmark requires MCP tool integration${NC}"
echo ""
echo "To run actual benchmarks:"
echo "  1. Start the context-graph MCP server"
echo "  2. Seed memories using store_memory/inject_context tools"
echo "  3. Use Claude Code with search_robust tool to run queries"
echo "  4. Compare results against ground truth in test_corpus.json"
echo ""
echo "Results would be saved to: $RESULTS_FILE"
