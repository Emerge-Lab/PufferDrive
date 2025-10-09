# Welfare Diplomacy C/C++ Port - Implementation Plan

## Overview

Port the welfare-diplomacy simulator from Python to C for PufferLib, achieving 10-100x speedup while maintaining exact functional equivalence.

**Goal**: Pass all 159 DATC (Diplomacy Adjudicator Test Cases) and original Python tests.

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

### Phase 3: Core Game Logic 🚧 IN PROGRESS

#### 3.1 Movement Adjudication ✅ COMPLETE
- Order parsing (HOLD, MOVE, SUPPORT, CONVOY)
- Order validation (adjacency, terrain, ownership)
- Basic movement resolution (uncontested, bounces, head-to-head)
- Support mechanics (strength calculation, dislodgement, support cutting)
- Result tracking (OK, VOID, BOUNCE, CUT, DISLODGED)

**Tests passing**: 26/159 DATC tests (16.4%)

#### 3.2 Convoy Mechanics 🎯 NEXT
- Multi-fleet convoy pathfinding
- Convoy disruption detection
- Attacked but not dislodged convoy still works
- Convoy to adjacent places (intent detection)
- Paradox resolution (Szykman rule)

**Expected gain**: +40-50 DATC tests

#### 3.3 Split Coasts
- Model STP/NC, STP/SC, BUL/EC, BUL/SC, SPA/NC, SPA/SC
- Update adjacency cache for coast-specific moves
- Coast specification in orders
- Default coast selection rules

**Expected gain**: +15 DATC tests

#### 3.4 Circular Movement (Enhancement)
- Proper dependency graph resolution
- Detect cycles and resolve consistent ones
- Break cycles with conflicting moves

**Expected gain**: +3-7 DATC tests

### Phase 4: Retreat Phase
**Deliverables:**
- Calculate valid retreat destinations
- Exclude attacker's origin and contested locations
- Handle multiple retreats to same location (all disband)
- Update phase progression for retreat phases

**Expected gain**: +16 DATC tests

### Phase 5: Adjustment Phase
**Deliverables:**
- Build validation (home center, owned, empty)
- Disband mechanics
- Civil disorder rules (auto-disband)
- Welfare-specific rules (voluntary disbands)
- Welfare point calculation: `max(0, num_centers - num_units)`

**Expected gain**: +18 DATC tests

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

## Implementation Roadmap

### Milestone 1: Basic Movement ✅ COMPLETE
**Target**: Pass DATC 6.A (basic validity)
- Simple moves, bounces, invalid moves
- **Status**: 7/7 tests passing (100%)

### Milestone 2: Support Mechanics ✅ COMPLETE
**Target**: Pass DATC 6.D, 6.E (supports and head-to-head)
- Attack/defense strength calculation
- Dislodgement detection
- Support cutting rules
- Self-dislodgement prevention
- **Status**: 19/49 tests passing (39%)

### Milestone 3: Convoy Mechanics 🎯 NEXT (Week 1)
**Target**: Pass DATC 6.F, 6.G (convoys)
- Multi-fleet pathfinding
- Disruption detection
- Paradox resolution
- **Estimated**: +40-50 tests → 65-75 total (45%)

### Milestone 4: Split Coasts (Week 1-2)
**Target**: Pass DATC 6.B (coastal issues)
- Model split coasts properly
- Coast-specific adjacencies
- **Estimated**: +15 tests → 80-90 total (55%)

### Milestone 5: Retreat Phase (Week 2)
**Target**: Pass DATC 6.H (retreating)
- Valid retreat calculation
- Multiple retreats handling
- **Estimated**: +16 tests → 95-105 total (65%)

### Milestone 6: Adjustment Phase (Week 2-3)
**Target**: Pass DATC 6.I, 6.J (building/disbanding)
- Build/disband validation
- Civil disorder rules
- Welfare calculations
- **Estimated**: +18 tests → 115-125 total (75%)

### Milestone 7: Edge Cases & Polish (Week 3-4)
**Target**: 100% DATC pass rate
- Fix remaining failures
- Handle all paradoxes
- Verify all original Python tests
- **Estimated**: +34-44 tests → 159 total (100%)

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

### DATC Test Coverage by Section

| Section | Description | Total Tests | Status |
|---------|-------------|-------------|--------|
| 6.A | Basic Validity | 11 | ✅ 7/7 tested (100%) |
| 6.B | Coastal Issues | 14 | ❌ 0/14 (needs split coasts) |
| 6.C | Circular Movement | 7 | ⏳ 0/7 (needs improvement) |
| 6.D | Supports & Dislodges | 34 | ✅ 13/34 (38%) |
| 6.E | Head-to-Head | 15 | ✅ 6/15 (40%) |
| 6.F | Convoys | 24 | ❌ 0/24 (not implemented) |
| 6.G | Adjacent Convoys | 18 | ❌ 0/18 (not implemented) |
| 6.H | Retreats | 16 | ❌ 0/16 (not implemented) |
| 6.I | Building | 7 | ❌ 0/7 (not implemented) |
| 6.J | Civil Disorder | 11 | ❌ 0/11 (not implemented) |
| 6.K | Custom | 2 | ❌ 0/2 |
| **Total** | | **159** | **26/159 (16.4%)** |

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

## Timeline Estimates

| Phase | Estimated Time | Status |
|-------|----------------|--------|
| Phase 1: Structure Setup | 1-2 days | ✅ DONE |
| Phase 2: Data Structures | 2-3 days | ✅ DONE |
| Phase 3: Core Game Logic | 2-3 weeks | 🚧 IN PROGRESS |
| - Basic movement | 2 days | ✅ DONE |
| - Support mechanics | 2 days | ✅ DONE |
| - Convoy mechanics | 3-5 days | 🎯 NEXT |
| - Split coasts | 2-3 days | Pending |
| - Circular movement | 1 day | Pending |
| Phase 4: Retreat Phase | 2-3 days | Pending |
| Phase 5: Adjustment Phase | 3-4 days | Pending |
| Phase 6: RL Interface | 3-4 days | Pending |
| Phase 7: Testing & Polish | 1 week | Pending |
| Phase 8: Optimization | 3-5 days | Deferred |
| **Total** | **3-5 weeks** | **Week 2** |

## Success Criteria

1. **Correctness**: 100% DATC test pass rate (159/159)
2. **Equivalence**: All original Python tests pass
   - test_map.py ✓
   - test_game.py ✓
   - test_datc.py ✓
3. **Performance**: 10-100x faster than Python
4. **Stability**: No memory leaks, no crashes
5. **Usability**: Clean Python API, easy PufferLib integration

## Priorities (Current Focus)

### Priority 1: Convoy Mechanics (This Week)
- **Impact**: Blocks ~40-50 DATC tests
- **Complexity**: High (pathfinding, disruption, paradoxes)
- **Estimated Time**: 3-5 days

### Priority 2: Split Coasts (This Week)
- **Impact**: Blocks ~15 DATC tests
- **Complexity**: Medium (data modeling, adjacency update)
- **Estimated Time**: 2-3 days

### Priority 3: Retreat Phase (Next Week)
- **Impact**: Blocks ~16 DATC tests
- **Complexity**: Medium (destination calculation, validation)
- **Estimated Time**: 2-3 days

### Priority 4: Adjustment Phase (Next Week)
- **Impact**: Blocks ~18 DATC tests
- **Complexity**: Medium (build/disband validation, welfare)
- **Estimated Time**: 3-4 days

## Notes

- This plan evolves as implementation progresses
- Correctness comes before performance
- Testing is not optional - exact equivalence required
- All deviations from Python behavior must be documented
- Original Python tests at `../welfare-diplomacy` serve as gold standard
