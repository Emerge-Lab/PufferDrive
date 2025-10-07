#ifndef DIPLOMACY_H
#define DIPLOMACY_H

#include <stdint.h>

// Constants
#define MAX_LOCATIONS 76
#define MAX_POWERS 7
#define MAX_UNITS 34
#define MAX_ADJACENCIES 20
#define MAX_HOME_CENTERS 4
#define MAX_ORDER_LENGTH 64
#define MAX_POWER_NAME_LENGTH 10
#define MAX_LOCATION_NAME_LENGTH 4

// Location types
typedef enum {
    LOC_LAND = 0,
    LOC_COAST = 1,
    LOC_WATER = 2,
    LOC_PORT = 3
} LocationType;

// Unit types
typedef enum {
    UNIT_NONE = 0,
    UNIT_ARMY = 1,
    UNIT_FLEET = 2
} UnitType;

// Phase types
typedef enum {
    PHASE_SPRING_MOVEMENT = 0,
    PHASE_SPRING_RETREAT = 1,
    PHASE_FALL_MOVEMENT = 2,
    PHASE_FALL_RETREAT = 3,
    PHASE_WINTER_ADJUSTMENT = 4,
    PHASE_COMPLETED = 5
} PhaseType;

// Order types
typedef enum {
    ORDER_NONE = 0,
    ORDER_HOLD = 1,        // - (hold)
    ORDER_MOVE = 2,        // - (move)
    ORDER_SUPPORT_HOLD = 3,// S (support hold)
    ORDER_SUPPORT_MOVE = 4,// S (support move)
    ORDER_CONVOY = 5,      // C (convoy)
    ORDER_BUILD = 6,       // B (build)
    ORDER_DISBAND = 7,     // D (disband)
    ORDER_RETREAT = 8      // R (retreat)
} OrderType;

// Location structure
typedef struct {
    char name[MAX_LOCATION_NAME_LENGTH];  // 3-letter code + null terminator
    LocationType type;
    int adjacencies[MAX_ADJACENCIES];      // Indices of adjacent locations (-1 terminated)
    int num_adjacent;
    int has_supply_center;                 // 1 if supply center, 0 otherwise
    int owner_power;                       // -1 if neutral, 0-6 for power index
    int is_home_center;                    // Which power's home center (0-6), or -1
} Location;

// Map structure (Standard Diplomacy map)
typedef struct {
    Location locations[MAX_LOCATIONS];
    int num_locations;

    // Power data
    char power_names[MAX_POWERS][MAX_POWER_NAME_LENGTH];
    char power_abbrev[MAX_POWERS][2];

    // Home centers for each power (location indices)
    int home_centers[MAX_POWERS][MAX_HOME_CENTERS];
    int num_homes[MAX_POWERS];

    // Adjacency cache for fast lookups
    // [unit_type][from_loc][to_loc] -> can_move (0 or 1)
    uint8_t adjacency_cache[3][MAX_LOCATIONS][MAX_LOCATIONS];  // indexed by UnitType
} Map;

// Unit structure
typedef struct {
    UnitType type;
    int location;          // Location index
    int power_id;          // Owner power (0-6)
    int can_retreat;       // Flag for dislodged units
} Unit;

// Order structure (parsed order)
typedef struct {
    OrderType type;
    int unit_location;     // Location of unit giving order
    UnitType unit_type;
    int target_location;   // For move/support/convoy
    int target_unit_location; // For support move (where supported unit is)
    int dest_location;     // For support move (where supported unit is going)
    int power_id;          // Power giving order
} Order;

// Retreat option
typedef struct {
    int location;          // Where unit can retreat to
} RetreatOption;

// Dislodged unit
typedef struct {
    UnitType type;
    int power_id;
    int from_location;
    int dislodged_by_location;
    RetreatOption possible_retreats[MAX_ADJACENCIES];
    int num_possible_retreats;
} DislodgedUnit;

// Power structure
typedef struct {
    int power_id;                          // 0-6
    char name[MAX_POWER_NAME_LENGTH];
    char abbrev[2];

    // Supply centers controlled (location indices, -1 terminated)
    int centers[MAX_UNITS];
    int num_centers;

    // Units
    Unit units[MAX_UNITS];
    int num_units;

    // Welfare Diplomacy specific
    int welfare_points;

    // Orders for current phase
    Order orders[MAX_UNITS];
    int num_orders;

    // Retreating units
    DislodgedUnit retreats[MAX_UNITS];
    int num_retreats;

    // Adjustment count (positive = can build, negative = must disband)
    int adjustment;
} Power;

// Combat resolution data
typedef struct {
    int location;
    int attack_strength;
    int attacker_location;  // Where attack is coming from
    int attacker_power;
    int defender_power;
    int successful;         // 1 if attack succeeded, 0 if bounced
} Combat;

// Game state structure
typedef struct {
    Map* map;
    Power powers[MAX_POWERS];

    // Current phase
    int year;
    PhaseType phase;

    // Combat resolution data
    Combat combats[MAX_LOCATIONS];
    int num_combats;

    // Dislodged units (accumulated during resolution)
    DislodgedUnit dislodged[MAX_UNITS];
    int num_dislodged;

    // Game settings
    int max_years;          // Game ends after N years
    int welfare_mode;       // 1 for Welfare variant, 0 for standard
    int is_game_over;       // 1 if game completed
} GameState;

// Logging structure (for PufferLib)
typedef struct {
    float n;                // Number of episodes
    float avg_welfare;      // Average welfare points across powers
    float avg_centers;      // Average centers per power
    float avg_units;        // Average units per power
} Log;

// Environment structure (PufferLib integration)
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

// Function declarations

// Map initialization
void init_standard_map(Map* map);
void free_map(Map* map);

// Game initialization and lifecycle
void init_game(GameState* game, Map* map, int welfare_mode, int max_years);
void reset_game(GameState* game);
void free_game(GameState* game);

// Order handling
int parse_order(const char* order_str, Order* order, GameState* game);
int validate_order(GameState* game, int power_id, const Order* order);
void get_possible_orders(GameState* game, int power_id, int location, char orders[][MAX_ORDER_LENGTH], int* num_orders);

// Phase processing
void process_orders(GameState* game);
void resolve_movement_phase(GameState* game);
void resolve_retreat_phase(GameState* game);
void resolve_adjustment_phase(GameState* game);
void advance_phase(GameState* game);

// Welfare Diplomacy specific
void calculate_welfare_points(GameState* game);

// Utility functions
const char* phase_to_string(PhaseType phase);
const char* unit_type_to_string(UnitType type);
int find_location_by_name(Map* map, const char* name);
int get_unit_at_location(GameState* game, int location);
int can_move(Map* map, UnitType unit_type, int from_loc, int to_loc);

// PufferLib integration functions
void c_init(Env* env);
void c_reset(Env* env);
void c_step(Env* env);
void c_render(Env* env);
void c_close(Env* env);

#endif // DIPLOMACY_H
