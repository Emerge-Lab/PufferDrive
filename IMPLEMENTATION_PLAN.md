# Welfare Diplomacy C/C++ Port for PufferLib - Implementation Plan

## Overview
Port the welfare-diplomacy simulator to C/C++ following PufferLib's ocean environment pattern for high-performance RL training.

**Goal**: Achieve 10-100x speedup over Python implementation while maintaining exact functional equivalence.

## Architecture Summary
- **Core Engine**: C (~6,000 lines from ~8,400 Python lines)
- **Python Wrapper**: Minimal Python interface using env_binding.h pattern
- **Omit**: Networking, DAIDE protocol, web interface, messaging (keep in Python if needed later)
- **Focus**: Barebones game state + order validation + order adjudication + game progression

## Key Simplifications
- **No networking**: Local simulation only
- **No rendering**: Optional Python-side rendering only (if needed)
- **No DAIDE protocol**: Direct order API
- **No messaging system**: Can add later in Python if needed for LLM agents
- **Standard map only**: Hardcode standard Diplomacy map data
- **Fixed 7 players**: Always 7 powers (AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY)

---

## Phase 1: Project Structure Setup

### Deliverables
1. **Directory**: `pufferlib/ocean/diplomacy/`
2. **Files**:
   - `binding.c` - Python C API interface (follows env_binding.h pattern)
   - `diplomacy.h` - Main header with structs and function declarations
   - `diplomacy.c` - Core game engine implementation
   - `diplomacy.py` - Python PufferEnv wrapper class
   - `test_diplomacy.py` - Comprehensive test suite
   - `test_datc_c.py` - DATC (Diplomacy Adjudicator Test Cases) tests
3. **Build System**: Update `setup.py` to compile C extension

### Success Criteria
- All files created and compile successfully
- Can import the module in Python
- Basic directory structure matches PufferLib conventions

---

## Phase 2: Core Data Structures (C)

### 2.1 Map Data Structures (`diplomacy.h`)

```c
// Location types
typedef enum {
    LOC_LAND,
    LOC_COAST,
    LOC_WATER,
    LOC_PORT
} LocationType;

// Unit types
typedef enum {
    UNIT_ARMY,
    UNIT_FLEET
} UnitType;

// Location structure
typedef struct {
    char name[4];           // 3-letter code + null terminator
    LocationType type;
    int adjacencies[20];    // Indices of adjacent locations
    int num_adjacent;
    int has_supply_center;
    int owner_power;        // -1 if neutral, 0-6 for power index
} Location;

// Map structure (Standard Diplomacy map)
typedef struct {
    Location locations[75];  // Standard map has 75 locations
    int num_locations;

    // Power data
    char power_names[7][10]; // AUSTRIA, ENGLAND, etc.
    char power_abbrev[7][2]; // A, E, F, G, I, R, T

    // Home centers for each power
    int home_centers[7][4];  // Max 4 home centers per power
    int num_homes[7];

    // Adjacency cache for fast lookups
    // [unit_type][from_loc][to_loc] -> can_move
    int adjacency_cache[2][75][75];
} Map;
```

### 2.2 Power Structure

```c
typedef struct {
    int power_id;           // 0-6
    char name[10];
    char abbrev[2];

    // Supply centers controlled (location indices)
    int centers[34];        // Max 34 supply centers on map
    int num_centers;

    // Units (army/fleet + location)
    struct {
        UnitType type;
        int location;
    } units[34];            // Max units = max centers
    int num_units;

    // Welfare Diplomacy specific
    int welfare_points;

    // Orders for current phase
    char orders[34][64];    // String representation of orders
    int num_orders;

    // Retreating units
    struct {
        UnitType type;
        int from_location;
        int possible_retreats[10];
        int num_possible;
    } retreats[34];
    int num_retreats;

    // Adjustment count (positive = builds, negative = disbands)
    int adjustment;
} Power;
```

### 2.3 Game State Structure

```c
typedef enum {
    PHASE_SPRING_MOVEMENT,
    PHASE_SPRING_RETREAT,
    PHASE_FALL_MOVEMENT,
    PHASE_FALL_RETREAT,
    PHASE_WINTER_ADJUSTMENT,
    PHASE_COMPLETED
} PhaseType;

typedef struct {
    Map* map;
    Power powers[7];

    // Current phase
    int year;
    PhaseType phase;

    // Combat resolution data
    struct {
        int location;
        int attack_strength;
        int attacker_power;
        int defender_power;
    } combats[75];
    int num_combats;

    // Dislodged units
    struct {
        UnitType type;
        int power_id;
        int from_location;
        int dislodged_by_location;
    } dislodged[34];
    int num_dislodged;

    // Game settings (Welfare Diplomacy)
    int max_years;          // Game ends after N years
    int welfare_mode;       // 1 for Welfare variant, 0 for standard
} GameState;
```

### 2.4 PufferLib Environment Structure

```c
typedef struct {
    // Game state
    GameState* game;

    // PufferLib standard fields
    void* observations;
    void* actions;
    float* rewards;
    unsigned char* terminals;

    // Multi-agent support (7 powers)
    int current_power_idx;

    // Random seed
    int seed;

    // Logging
    Log log;
} Env;
```

### Success Criteria for Phase 2
- All structs defined and documented
- Compiles without errors
- Memory layout is efficient (consider padding)
- Can allocate and free structures

---

## Phase 3: Core Game Logic (C)

### 3.1 Initialization
- `void init_map(Map* map)` - Load standard map data
- `void init_game(GameState* game, Map* map, int welfare_mode)` - Initialize game
- `void c_init(Env* env, PyObject* args, PyObject* kwargs)` - PufferLib init

### 3.2 Reset
- `void reset_game(GameState* game)` - Reset to Spring 1901
- `void c_reset(Env* env)` - PufferLib reset

### 3.3 Order Processing
- `int validate_order(GameState* game, int power_id, const char* order)` - Check validity
- `void parse_order(const char* order_str, Order* order)` - Parse string to struct
- `void get_possible_orders(GameState* game, int power_id, char** orders)` - List valid orders

### 3.4 Order Adjudication (Core Logic)
- `void resolve_orders(GameState* game)` - Main adjudication engine
- `void resolve_movement(GameState* game)` - Movement phase resolution
- `void resolve_retreats(GameState* game)` - Retreat phase resolution
- `void resolve_adjustments(GameState* game)` - Adjustment phase resolution
- `void calculate_welfare_points(GameState* game)` - Welfare variant scoring

### 3.5 Phase Progression
- `void advance_phase(GameState* game)` - Move to next phase
- `void c_step(Env* env)` - PufferLib step function

---

## Phase 4: Welfare Diplomacy Specifics

### Key Rules to Implement
1. **Voluntary Disbanding**: In adjustment phase, powers can disband units even with surplus centers
2. **Welfare Points**: After each adjustment phase:
   ```
   welfare_points += (num_centers - num_units)
   ```
   Only positive values count (can't go negative)
3. **No Victory Condition**: Game runs to fixed year limit (e.g., 10 years = 1910)
4. **Final Scoring**: Total utility = cumulative welfare_points

### Implementation
- Add `welfare_mode` flag to GameState
- Modify adjustment phase to allow voluntary disbands
- Calculate and accumulate welfare_points after adjustments
- Disable victory checking in welfare mode
- Add welfare_points to observations

---

## Phase 5: RL Interface (C + Python)

### 5.1 Observation Space
```python
observation_space = Dict({
    'board': Box(0, 7, shape=(75,), dtype=np.int8),      # Power controlling each location
    'units': Box(0, 2, shape=(75,), dtype=np.int8),      # Unit type at location (0=none)
    'phase': Box(0, 5, shape=(1,), dtype=np.int8),       # Current phase
    'year': Box(1901, 2000, shape=(1,), dtype=np.int16), # Current year
    'centers': Box(0, 34, shape=(7,), dtype=np.int8),    # Centers per power
    'units_count': Box(0, 34, shape=(7,), dtype=np.int8), # Units per power
    'welfare': Box(0, 500, shape=(7,), dtype=np.int16),  # Welfare points
    'valid_actions': MultiBinary(1000),                   # Valid action mask
})
```

### 5.2 Action Space
- Discrete action space mapping integers to order strings
- Build lookup table of all possible orders (~1000 orders)
- Action masking for invalid orders

### 5.3 Reward Calculation
```python
# Welfare Diplomacy: Reward = delta in welfare points
reward = new_welfare_points - old_welfare_points

# Can add reward shaping:
# - Small reward for gaining supply centers
# - Small penalty for losing units
```

### 5.4 Multi-Agent Support
- Support all 7 powers as separate agents
- Option for self-play or fixed opponents
- Vectorization across multiple games

---

## Phase 6: Python Wrapper (diplomacy.py)

### Structure
```python
class Diplomacy(pufferlib.PufferEnv):
    def __init__(self, num_players=7, max_years=10, welfare_mode=True, ...):
        # Initialize observation/action spaces
        # Call C binding to create environment

    def reset(self, seed=None):
        # Call c_reset
        # Return observations

    def step(self, actions):
        # Call c_step
        # Return obs, rewards, terminals, truncations, infos
```

---

## Phase 7: Testing & Validation (CRITICAL!)

### 7.1 Test Directory Structure

```
tests/diplomacy/
├── __init__.py
├── conftest.py                      # Pytest fixtures
│
# === C-SPECIFIC TESTS (New) ===
├── c_tests/
│   ├── __init__.py
│   ├── test_c_binding.py           # C extension loading, memory
│   ├── test_c_map.py               # Map data structures in C
│   ├── test_c_game_state.py        # C game state management
│   └── test_c_welfare.py           # C welfare calculations
│
# === ORIGINAL PYTHON TESTS (Copied from welfare-diplomacy) ===
├── original/
│   ├── __init__.py                 # Copied
│   ├── test_map.py                 # Copied (12K lines)
│   ├── test_game.py                # Copied (27K lines)
│   ├── test_datc.py                # Copied (296K lines, ~200 test cases)
│   ├── test_datc_no_check.py       # Copied (4K lines)
│   ├── test_datc_no_expand.py      # Copied (5K lines)
│   └── test_pytest.py              # Copied (1.1K lines)
│
# === ADAPTER LAYER ===
└── adapters.py                      # Run Python tests against C
```

### 7.2 C-Specific Tests (~550 lines total)

#### test_c_binding.py
- C extension loading and imports
- Environment creation/destruction
- Memory management (no leaks)
- Multiple environment instances
- C function exposure verification

#### test_c_map.py
- Map structure allocation
- 76 locations loaded correctly
- Location names, types (LAND/WATER/COAST)
- 34 supply centers identified
- Home centers for all 7 powers
- Adjacency lists loaded
- find_location_by_name() works
- can_move() adjacency queries
- Adjacency cache built correctly

#### test_c_game_state.py
- Game initialization
- Game reset functionality
- Initial phase (Spring 1901 Movement)
- 7 powers initialized correctly
- Phase progression logic
- Year increment after winter
- Game-over condition at max_years

#### test_c_welfare.py
- Welfare mode flag
- Welfare calculation: max(0, centers - units)
- Welfare point accumulation
- No 18-center victory in welfare mode

### 7.3 Original Python Tests (~6,614 lines copied)

**Direct copies from `/scratch/mmk9418/projects/welfare-diplomacy/diplomacy/tests/`**:
- `test_map.py` - Map functionality tests
- `test_game.py` - Game logic tests
- `test_datc.py` - Full DATC compliance suite (~200 test cases)
- `test_datc_no_check.py` - DATC without validation
- `test_datc_no_expand.py` - DATC without expansion
- `test_pytest.py` - Pytest integration tests

These define the **gold standard** - our C implementation must pass ALL of these.

### 7.4 Adapter Layer (`adapters.py` ~300 lines)

**Purpose**: Make C implementation compatible with Python test interface

```python
class CGameAdapter:
    """Adapter: C implementation → Python Game API"""
    def __init__(self):
        from pufferlib.ocean.diplomacy import Diplomacy
        self.env = Diplomacy()

    def clear_units(self): ...
    def set_units(self, power, units): ...
    def set_orders(self, power, orders): ...
    def process(self): ...
    # ... all methods Python tests expect

class CMapAdapter:
    """Adapter: C map → Python Map API"""
    @property
    def locs(self): ...
    @property
    def scs(self): ...
    # ... all properties Python tests expect
```

### 7.5 Test Execution Strategy

**Phase 1 (Current)**: C-Specific Tests Only
```bash
pytest tests/diplomacy/c_tests/ -v
```
- Verify C basics work
- Test as we build features

**Phase 2 (Map Complete)**: Map Tests
```bash
pytest tests/diplomacy/c_tests/test_c_map.py -v
pytest tests/diplomacy/original/test_map.py -v  # via adapter
```
- Validate map data matches Python

**Phase 3 (Game Logic Complete)**: Game Tests
```bash
pytest tests/diplomacy/original/test_game.py -v  # via adapter
```
- Validate game logic matches Python

**Phase 4 (Adjudication Complete)**: DATC Suite
```bash
pytest tests/diplomacy/original/test_datc.py -v  # via adapter
```
- **Ultimate validation**: All ~200 DATC test cases pass
- Proves 100% equivalence with Python

### 7.6 Pytest Markers

```python
# In conftest.py
pytest.mark.c_only       # C-specific tests
pytest.mark.python_gold  # Original Python tests (gold standard)
pytest.mark.slow         # Slow tests (DATC suite)
pytest.mark.map          # Map tests
pytest.mark.game         # Game logic tests
pytest.mark.welfare      # Welfare-specific tests
```

Run subsets:
```bash
pytest -m c_only           # C tests only
pytest -m python_gold      # Gold standard tests only
pytest -m "not slow"       # Skip DATC (fast feedback)
pytest -m map              # Map tests only
```

### 7.7 Success Criteria

**Immediate**: C-specific tests pass
**Short-term**: Original test_map.py passes via adapter
**Medium-term**: Original test_game.py passes via adapter
**Long-term**: All DATC tests pass via adapter = **100% equivalence**

---

## Phase 8: Optimization (After Correctness)

### 8.1 Memory Layout
- Struct packing to minimize cache misses
- Array-of-structs vs struct-of-arrays analysis

### 8.2 Lookup Tables
- Pre-compute adjacency matrices
- Cache valid order lists per location
- Hash table for order parsing

### 8.3 SIMD (if beneficial)
- Vectorize combat resolution across locations
- Batch order validation

### 8.4 Multi-threading
- Thread pool for vectorized environments
- Lock-free game state copies

---
## Technical Decisions

### Language
- **C** (not C++) for simplicity and PufferLib consistency
- Use C99 standard features

### Map Data
- **Compile-time data structures**: Avoid runtime parsing overhead
- Standard Diplomacy map hardcoded as static arrays

### Order Representation
- String format for human readability during development
- Consider integer encoding for performance later

### State Hashing
- Skip Zobrist hashing initially (used in Python for caching)
- Add later if needed for transposition tables

### Vectorization
- Follow PufferLib `vec_init` / `vec_step` pattern
- Support 100+ parallel games in single process

---


## Timeline Estimate

| Phase | Estimated Time |
|-------|----------------|
| Phase 1: Structure Setup | ✅ 1-2 days (DONE) |
| Phase 2: Data Structures | ✅ 2-3 days (DONE) |
| **Testing Setup** | **1 day** |
| - Copy original tests | 1 hour |
| - Create C-specific tests | 3 hours |
| - Build adapter layer | 4 hours |
| Phase 3: Core Game Logic | 1-2 weeks |
| Phase 4: Welfare Specifics | 2-3 days |
| Phase 5: RL Interface | 3-4 days |
| Phase 6: Python Wrapper | 2-3 days |
| Phase 7: Test Validation | 1 week |
| - Fix DATC failures | 3-5 days |
| - Welfare test validation | 1-2 days |
| Phase 8: Optimization | 3-5 days |
| **Total** | **3-4 weeks** |

---

## Success Metrics

1. **Correctness**: 100% DATC test pass rate (~200 test cases)
2. **Equivalence**: All original Python tests pass against C implementation
   - test_map.py ✓
   - test_game.py ✓
   - test_datc.py ✓ (all variants)
3. **Performance**: 10-100x faster than Python
4. **Stability**: No memory leaks, no crashes
5. **Usability**: Clean Python API, easy to integrate with PufferLib training
6. **Coverage**: ~7,464 lines of test code (550 C-specific + 6,614 copied + 300 adapter)

---

## Current Status

**Phase**: 2 (Core Data Structures) - In Progress

**Completed**:
- ✅ Phase 1: Project structure, compilation, basic imports
- ✅ Created map data generation script (generates C code from standard.map)
- ✅ Updated structures for 76 locations
- ✅ Basic test framework created

**Next Steps**:
1. Insert generated map data into init_standard_map()
2. Build adjacency cache
3. Create test directory structure (tests/diplomacy/)
4. Copy original Python tests as gold standard
5. Create C-specific tests for incremental validation
6. Build adapter layer to run Python tests against C
7. Implement core game logic (Phase 3)

---
## Current Status

**Phase**: 1 (Project Structure Setup)

**Next Steps**:
1. Create directory structure
2. Create initial header files
3. Set up compilation
4. Write basic initialization code

---

## Notes

- This plan should be updated as implementation progresses
- Any deviations from Python behavior must be documented and justified
- Testing is not optional - exact functional equivalence is required
- Performance comes AFTER correctness
