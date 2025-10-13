# Welfare Diplomacy C Port - Implementation Plan

## Overview

Porting welfare-diplomacy from Python to C for PufferLib, achieving 10-100x speedup while maintaining exact functional equivalence.

**Current**: 80/160 DATC tests passing (50.0%)
**Goal**: 100 tests → 150+ tests → 160/160 (100%)

## Architecture

### Core Components
- **C Engine** (~2,500 lines): Game state, order validation, adjudication
- **Python Wrapper** (~200 lines): PufferEnv interface, minimal overhead
- **Adapter Layer** (~650 lines): Compatibility bridge for original tests
- **Original Python Tests** (~6,500 lines): Gold standard for correctness

### Key Simplifications
- No networking, DAIDE protocol, web interface
- No messaging system (can add later for LLM agents)
- Standard map only (hardcoded)
- Fixed 7 players (AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY)
- Local simulation only

## Implementation Phases

### Phase 1: Project Structure ✅ COMPLETE
**Deliverables:**
- Directory structure: `pufferlib/ocean/diplomacy/`
- Files: `binding.c`, `diplomacy.h`, `diplomacy.c`, `diplomacy.py`
- Build system integrated into `setup.py`
- C extension imports successfully

### Phase 2: Core Data Structures ✅ COMPLETE
**Deliverables:**
- Map structure (76 locations, adjacency cache)
- Power structure (units, centers, orders, welfare tracking)
- Game state structure (phase progression, combat resolution)
- PufferLib environment structure

### Phase 3: Core Game Logic ✅ MOSTLY COMPLETE

#### 3.1 Movement Adjudication ✅ COMPLETE
- Order parsing, validation, basic movement resolution
- Support mechanics (strength calculation, dislodgement, support cutting)
- Result tracking (OK, VOID, BOUNCE, CUT, DISLODGED)

#### 3.2 Convoy Mechanics ✅ MOSTLY WORKING
- Multi-fleet convoy pathfinding ✅
- Convoy disruption detection ✅
- Basic paradox resolution (Szykman rule) ✅
- Complex paradoxes and multi-route scenarios ⏳

**Current**: 14/24 convoy tests passing (58%)
**Remaining**: ~10 tests

#### 3.3 Split Coasts ✅ COMPLETE
- STP, BUL, SPA coast variants implemented ✅
- Coast-specific adjacencies ✅
- Order parsing with coast specifications ✅
- Some tests blocked by retreat/build phases

**Current**: 3/14 split coast tests passing (21%)
**Blocked**: ~11 tests need retreat/build phases

#### 3.4 Circular Movement ✅ MOSTLY WORKING
- 2-way and 3-way cycles resolved ✅
- Disrupted circular movements handled ✅
- Support orders in circular movements ✅

**Current**: 6/7 circular movement tests passing (86%)
**Remaining**: 1 convoy+circular edge case

### Phase 4: Retreat Phase ✅ MOSTLY COMPLETE
**Deliverables:**
- ✅ Calculate valid retreat destinations
- ✅ Exclude attacker's origin and contested locations
- ✅ Handle multiple retreats to same location (all disband)
- ✅ Update phase progression for retreat phases
- ⏳ Convoy-based retreat rules (3 tests)
- ⏳ Support validation in retreats (1 test)

**Status**: 12/16 retreat tests passing (75%)
**Remaining**: Convoy retreat rules, support validation

### Phase 5: Adjustment Phase 🎯 CRITICAL PRIORITY
**Deliverables:**
- Build validation (home center, owned, empty)
- Disband mechanics and civil disorder rules
- Welfare-specific rules (voluntary disbands)
- Welfare point calculation: `max(0, num_centers - num_units)`

**Impact**: +18 direct tests + unblocks remaining split coast tests = +25 tests
**Target**: ~99 → ~124 tests (77%)

### Phase 6: RL Interface
**Deliverables:**
- Observation space (board, units, phase, year, centers, welfare, valid actions)
- Action space (discrete, ~1000 orders)
- Action masking for invalid orders
- Reward calculation (welfare deltas)
- Multi-agent support (7 powers)

### Phase 7: Testing & Polish
**Deliverables:**
- 100% DATC pass rate (all 159 tests)
- All original Python tests passing
- Edge case handling
- Memory leak checks
- Performance benchmarks

### Phase 8: Optimization
**Deliverables:**
- Memory layout optimization
- Lookup table pre-computation
- SIMD vectorization (if beneficial)
- Multi-threading for vectorized envs

## Current Status & Roadmap

### Current: 68/160 Tests Passing (42.5%) ✅
- Basic movement, support mechanics, convoy basics working
- Split coasts infrastructure complete
- Circular movement mostly working

### Path to 150+ Tests

#### Milestone 1: Retreat Phase 🎯 HIGHEST PRIORITY
**Target**: 68 → 99 tests
- Implement retreat destination calculation
- Handle multiple retreats to same location
- **Impact**: +16 direct tests + unblocks ~15 tests

#### Milestone 2: Adjustment Phase 🎯 HIGHEST PRIORITY
**Target**: 99 → 124 tests
- Implement build/disband mechanics
- Civil disorder rules
- Welfare calculations
- **Impact**: +18 direct tests + unblocks ~7 tests

#### Milestone 3: Convoy Edge Cases 🎯 MEDIUM PRIORITY
**Target**: 124 → 144 tests
- Complex paradox resolution
- Multi-route convoys
- Adjacent convoy edge cases
- **Impact**: +10 convoy tests + +10 adjacent convoy tests

#### Milestone 4: Polish & Edge Cases 🎯 LOW PRIORITY
**Target**: 144 → 160 tests
- Fix remaining split coast tests
- Circular movement + convoy edge case
- Custom tests
- **Impact**: +16 remaining tests

## Testing Strategy

### Test Hierarchy

#### Level 1: C-Specific Tests
**Location**: `tests/diplomacy/c_tests/`
- `test_c_binding.py` - C extension loading, memory management
- `test_c_map.py` - Map data structures
- `test_c_game_state.py` - Game state management
- `test_c_orders.py` - Order parsing and validation
- `test_c_welfare.py` - Welfare calculations
- `test_simple_adjudication.py` - Simple movement scenarios

**Purpose**: Test C implementation in isolation

#### Level 2: Original Python Tests (Gold Standard)
**Location**: `tests/diplomacy/original/`
- `test_map.py` - Map functionality (238 lines)
- `test_game.py` - Game logic (669 lines)
- `test_datc.py` - DATC compliance (5478 lines, ~159 tests)

**Purpose**: Prove exact functional equivalence with Python

#### Level 3: Adapter Layer
**Location**: `tests/diplomacy/adapters.py`
- `GameAdapter` - Bridges C implementation to Python Game API
- `MapAdapter` - Bridges C map to Python Map API

**Purpose**: Make C implementation compatible with original tests

### DATC Test Coverage by Section (Current: 68/160)

| Section | Description | Passing | Priority |
|---------|-------------|---------|----------|
| 6.A | Basic Validity | 8/12 (67%) | ✅ Done |
| 6.B | Coastal Issues | 3/14 (21%) | ⏳ Blocked by retreat/build |
| 6.C | Circular Movement | 6/7 (86%) | ⏳ 1 edge case |
| 6.D | Supports & Dislodges | 14/34 (41%) | ✅ Core done |
| 6.E | Head-to-Head | 6/15 (40%) | ✅ Core done |
| 6.F | Convoys | 14/24 (58%) | ⏳ Edge cases remain |
| 6.G | Adjacent Convoys | 3/18 (17%) | 🎯 Priority 3 |
| 6.H | Retreats | 0/16 (0%) | 🎯 Priority 1 |
| 6.I | Building | 3/7 (43%) | 🎯 Priority 2 |
| 6.J | Civil Disorder | 1/11 (9%) | 🎯 Priority 2 |
| 6.K | Custom | 1/2 (50%) | ⏳ Edge cases |
| **Total** | | **68/160 (42.5%)** | **Target: 150+** |

### Test Execution Strategy

**Fast Feedback Loop**:
```bash
# C-specific tests only (fast)
pytest tests/diplomacy/c_tests/ -v

# Specific DATC section
pytest tests/diplomacy/original/test_datc.py -k "test_6_d" -v

# All passing tests
pytest tests/diplomacy/original/test_datc.py -v
```

**Full Validation**:
```bash
# All tests (slow)
pytest tests/diplomacy/ -v

# DATC suite only
pytest tests/diplomacy/original/test_datc.py -v
```

## Technical Decisions

### Language Choice
- **C** (not C++) for simplicity and PufferLib consistency
- C99 standard features
- Clean, readable code prioritized over performance (initially)

### Map Data
- Compile-time data structures (no runtime parsing)
- Standard Diplomacy map hardcoded as static arrays
- 76 locations, 34 supply centers, 7 powers

### Order Representation
- String format for human readability during development
- Integer encoding deferred until optimization phase

### State Hashing
- Skip Zobrist hashing initially
- Add later if needed for transposition tables

### Vectorization
- Follow PufferLib `vec_init`/`vec_step` pattern
- Support 100+ parallel games in single process

## Priorities for 68 → 150 Tests

### Priority 1: Retreat Phase (2-3 days) 🎯
- **Impact**: +31 tests → 99 total (62%)
- Calculate valid retreat destinations
- Handle multiple retreats to same location
- Unblocks many split coast and other tests

### Priority 2: Adjustment Phase (3-4 days) 🎯
- **Impact**: +25 tests → 124 total (77%)
- Build/disband validation and execution
- Civil disorder rules
- Welfare calculations
- Unblocks remaining blocked tests

### Priority 3: Convoy Edge Cases (2-3 days)
- **Impact**: +20 tests → 144 total (90%)
- Complex paradoxes
- Multi-route convoys
- Adjacent convoy scenarios

### Priority 4: Polish (1-2 days)
- **Impact**: +16 tests → 160 total (100%)
- Remaining edge cases
- Final test fixes

## Success Criteria

1. **Correctness**: 100% DATC test pass rate (159/159)
2. **Equivalence**: All original Python tests pass
   - test_map.py ✓
   - test_game.py ✓
   - test_datc.py ✓
3. **Performance**: 10-100x faster than Python
4. **Stability**: No memory leaks, no crashes
5. **Usability**: Clean Python API, easy PufferLib integration

## Quick Action Plan

**Fastest path to 150+ tests:**

1. **Implement Retreat Phase** → 99 tests (3 days)
   - Valid retreat destinations
   - Multiple retreats to same location

2. **Implement Adjustment Phase** → 124 tests (4 days)
   - Build/disband mechanics
   - Civil disorder
   - Welfare calculations

3. **Fix Convoy Edge Cases** → 144 tests (3 days)
   - Complex paradoxes
   - Multi-route scenarios

4. **Polish Remaining** → 160 tests (2 days)

**Estimated time to 150 tests**: 7-8 days
**Estimated time to 160 tests**: 12-14 days

## Notes

- This plan evolves as implementation progresses
- Correctness comes before performance
- Testing is not optional - exact equivalence required
- All deviations from Python behavior must be documented
- Original Python tests at `../welfare-diplomacy` serve as gold standard
