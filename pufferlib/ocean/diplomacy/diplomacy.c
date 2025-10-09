#include "diplomacy.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

// ============================================================================
// Utility Functions
// ============================================================================

const char* phase_to_string(PhaseType phase) {
    switch (phase) {
        case PHASE_SPRING_MOVEMENT: return "SPRING 1901 MOVEMENT";
        case PHASE_SPRING_RETREAT: return "SPRING 1901 RETREAT";
        case PHASE_FALL_MOVEMENT: return "FALL 1901 MOVEMENT";
        case PHASE_FALL_RETREAT: return "FALL 1901 RETREAT";
        case PHASE_WINTER_ADJUSTMENT: return "WINTER 1901 ADJUSTMENT";
        case PHASE_COMPLETED: return "COMPLETED";
        default: return "UNKNOWN";
    }
}

const char* unit_type_to_string(UnitType type) {
    switch (type) {
        case UNIT_ARMY: return "A";
        case UNIT_FLEET: return "F";
        case UNIT_NONE: return " ";
        default: return "?";
    }
}

int find_location_by_name(Map* map, const char* name) {
    for (int i = 0; i < map->num_locations; i++) {
        if (strcmp(map->locations[i].name, name) == 0) {
            return i;
        }
    }
    return -1;  // Not found
}

int get_unit_at_location(GameState* game, int location) {
    for (int p = 0; p < MAX_POWERS; p++) {
        for (int u = 0; u < game->powers[p].num_units; u++) {
            if (game->powers[p].units[u].location == location) {
                return p;  // Return power ID
            }
        }
    }
    return -1;  // No unit at location
}

int can_move(Map* map, UnitType unit_type, int from_loc, int to_loc) {
    if (from_loc < 0 || from_loc >= map->num_locations ||
        to_loc < 0 || to_loc >= map->num_locations) {
        return 0;
    }
    return map->adjacency_cache[unit_type][from_loc][to_loc];
}

// ============================================================================
// Game State Query Functions (for testing)
// ============================================================================

int get_current_year(GameState* game) {
    return game->year;
}

PhaseType get_current_phase(GameState* game) {
    return game->phase;
}

int get_num_units(GameState* game, int power_id) {
    if (power_id < 0 || power_id >= MAX_POWERS) {
        return 0;
    }
    return game->powers[power_id].num_units;
}

void get_unit_info(GameState* game, int power_id, int unit_idx, UnitType* type, int* location) {
    if (power_id < 0 || power_id >= MAX_POWERS) {
        return;
    }
    if (unit_idx < 0 || unit_idx >= game->powers[power_id].num_units) {
        return;
    }
    *type = game->powers[power_id].units[unit_idx].type;
    *location = game->powers[power_id].units[unit_idx].location;
}

int get_num_centers(GameState* game, int power_id) {
    if (power_id < 0 || power_id >= MAX_POWERS) {
        return 0;
    }
    return game->powers[power_id].num_centers;
}

void get_center_locations(GameState* game, int power_id, int* centers, int* num_centers) {
    if (power_id < 0 || power_id >= MAX_POWERS) {
        *num_centers = 0;
        return;
    }
    *num_centers = game->powers[power_id].num_centers;
    for (int i = 0; i < *num_centers; i++) {
        centers[i] = game->powers[power_id].centers[i];
    }
}

int get_welfare_points(GameState* game, int power_id) {
    if (power_id < 0 || power_id >= MAX_POWERS) {
        return 0;
    }
    return game->powers[power_id].welfare_points;
}

const char* get_location_name(Map* map, int location_idx) {
    if (location_idx < 0 || location_idx >= map->num_locations) {
        return "";
    }
    return map->locations[location_idx].name;
}

int get_num_locations(Map* map) {
    return map->num_locations;
}

LocationType get_location_type(Map* map, int location_idx) {
    if (location_idx < 0 || location_idx >= map->num_locations) {
        return LOC_LAND;  // Default
    }
    return map->locations[location_idx].type;
}

int is_supply_center(Map* map, int location_idx) {
    if (location_idx < 0 || location_idx >= map->num_locations) {
        return 0;
    }
    return map->locations[location_idx].has_supply_center;
}

// ============================================================================
// Map Initialization - Generated from standard.map
// ============================================================================

void init_standard_map(Map* map) {
    memset(map, 0, sizeof(Map));

    // Power names
    strcpy(map->power_names[0], "AUSTRIA");
    strcpy(map->power_names[1], "ENGLAND");
    strcpy(map->power_names[2], "FRANCE");
    strcpy(map->power_names[3], "GERMANY");
    strcpy(map->power_names[4], "ITALY");
    strcpy(map->power_names[5], "RUSSIA");
    strcpy(map->power_names[6], "TURKEY");

    // Power abbreviations
    strcpy(map->power_abbrev[0], "A");
    strcpy(map->power_abbrev[1], "E");
    strcpy(map->power_abbrev[2], "F");
    strcpy(map->power_abbrev[3], "G");
    strcpy(map->power_abbrev[4], "I");
    strcpy(map->power_abbrev[5], "R");
    strcpy(map->power_abbrev[6], "T");

    // Initialize all locations (generated from standard.map)
    map->num_locations = 76;

// 0: ADR
strcpy(map->locations[0].name, "ADR");
map->locations[0].type = LOC_WATER;
map->locations[0].has_supply_center = 0;
map->locations[0].owner_power = -1;  // Neutral initially
map->locations[0].num_adjacent = 5;
map->locations[0].adjacencies[0] = 2;
map->locations[0].adjacencies[1] = 4;
map->locations[0].adjacencies[2] = 29;
map->locations[0].adjacencies[3] = 64;
map->locations[0].adjacencies[4] = 70;

// 1: AEG
strcpy(map->locations[1].name, "AEG");
map->locations[1].type = LOC_WATER;
map->locations[1].has_supply_center = 0;
map->locations[1].owner_power = -1;  // Neutral initially
map->locations[1].num_adjacent = 6;
map->locations[1].adjacencies[0] = 15;
map->locations[1].adjacencies[1] = 18;
map->locations[1].adjacencies[2] = 20;
map->locations[1].adjacencies[3] = 26;
map->locations[1].adjacencies[4] = 29;
map->locations[1].adjacencies[5] = 58;

// 2: ALB
strcpy(map->locations[2].name, "ALB");
map->locations[2].type = LOC_COAST;
map->locations[2].has_supply_center = 0;
map->locations[2].owner_power = -1;  // Neutral initially
map->locations[2].num_adjacent = 5;
map->locations[2].adjacencies[0] = 0;
map->locations[2].adjacencies[1] = 26;
map->locations[2].adjacencies[2] = 29;
map->locations[2].adjacencies[3] = 54;
map->locations[2].adjacencies[4] = 64;

// 3: ANK
strcpy(map->locations[3].name, "ANK");
map->locations[3].type = LOC_COAST;
map->locations[3].has_supply_center = 1;
map->locations[3].owner_power = -1;  // Neutral initially
map->locations[3].num_adjacent = 4;
map->locations[3].adjacencies[0] = 5;
map->locations[3].adjacencies[1] = 10;
map->locations[3].adjacencies[2] = 18;
map->locations[3].adjacencies[3] = 58;

// 4: APU
strcpy(map->locations[4].name, "APU");
map->locations[4].type = LOC_COAST;
map->locations[4].has_supply_center = 0;
map->locations[4].owner_power = -1;  // Neutral initially
map->locations[4].num_adjacent = 5;
map->locations[4].adjacencies[0] = 0;
map->locations[4].adjacencies[1] = 29;
map->locations[4].adjacencies[2] = 42;
map->locations[4].adjacencies[3] = 51;
map->locations[4].adjacencies[4] = 70;

// 5: ARM
strcpy(map->locations[5].name, "ARM");
map->locations[5].type = LOC_COAST;
map->locations[5].has_supply_center = 0;
map->locations[5].owner_power = -1;  // Neutral initially
map->locations[5].num_adjacent = 5;
map->locations[5].adjacencies[0] = 3;
map->locations[5].adjacencies[1] = 10;
map->locations[5].adjacencies[2] = 55;
map->locations[5].adjacencies[3] = 58;
map->locations[5].adjacencies[4] = 63;

// 6: BAL
strcpy(map->locations[6].name, "BAL");
map->locations[6].type = LOC_WATER;
map->locations[6].has_supply_center = 0;
map->locations[6].owner_power = -1;  // Neutral initially
map->locations[6].num_adjacent = 7;
map->locations[6].adjacencies[0] = 9;
map->locations[6].adjacencies[1] = 12;
map->locations[6].adjacencies[2] = 19;
map->locations[6].adjacencies[3] = 33;
map->locations[6].adjacencies[4] = 31;
map->locations[6].adjacencies[5] = 50;
map->locations[6].adjacencies[6] = 61;

// 7: BAR
strcpy(map->locations[7].name, "BAR");
map->locations[7].type = LOC_WATER;
map->locations[7].has_supply_center = 0;
map->locations[7].owner_power = -1;  // Neutral initially
map->locations[7].num_adjacent = 3;
map->locations[7].adjacencies[0] = 45;
map->locations[7].adjacencies[1] = 44;
map->locations[7].adjacencies[2] = 60;

// 8: BEL
strcpy(map->locations[8].name, "BEL");
map->locations[8].type = LOC_COAST;
map->locations[8].has_supply_center = 1;
map->locations[8].owner_power = -1;  // Neutral initially
map->locations[8].num_adjacent = 6;
map->locations[8].adjacencies[0] = 16;
map->locations[8].adjacencies[1] = 22;
map->locations[8].adjacencies[2] = 28;
map->locations[8].adjacencies[3] = 43;
map->locations[8].adjacencies[4] = 47;
map->locations[8].adjacencies[5] = 52;

// 9: BER
strcpy(map->locations[9].name, "BER");
map->locations[9].type = LOC_COAST;
map->locations[9].has_supply_center = 0;
map->locations[9].owner_power = -1;  // Neutral initially
map->locations[9].num_adjacent = 5;
map->locations[9].adjacencies[0] = 6;
map->locations[9].adjacencies[1] = 31;
map->locations[9].adjacencies[2] = 39;
map->locations[9].adjacencies[3] = 50;
map->locations[9].adjacencies[4] = 56;

// 10: BLA
strcpy(map->locations[10].name, "BLA");
map->locations[10].type = LOC_WATER;
map->locations[10].has_supply_center = 0;
map->locations[10].owner_power = -1;  // Neutral initially
map->locations[10].num_adjacent = 6;
map->locations[10].adjacencies[0] = 3;
map->locations[10].adjacencies[1] = 5;
map->locations[10].adjacencies[2] = 15;
map->locations[10].adjacencies[3] = 18;
map->locations[10].adjacencies[4] = 53;
map->locations[10].adjacencies[5] = 55;

// 11: BOH
strcpy(map->locations[11].name, "BOH");
map->locations[11].type = LOC_LAND;
map->locations[11].has_supply_center = 0;
map->locations[11].owner_power = -1;  // Neutral initially
map->locations[11].num_adjacent = 5;
map->locations[11].adjacencies[0] = 24;
map->locations[11].adjacencies[1] = 39;
map->locations[11].adjacencies[2] = 56;
map->locations[11].adjacencies[3] = 67;
map->locations[11].adjacencies[4] = 71;

// 12: BOT
strcpy(map->locations[12].name, "BOT");
map->locations[12].type = LOC_WATER;
map->locations[12].has_supply_center = 0;
map->locations[12].owner_power = -1;  // Neutral initially
map->locations[12].num_adjacent = 5;
map->locations[12].adjacencies[0] = 6;
map->locations[12].adjacencies[1] = 23;
map->locations[12].adjacencies[2] = 33;
map->locations[12].adjacencies[3] = 60;
map->locations[12].adjacencies[4] = 61;

// 13: BRE
strcpy(map->locations[13].name, "BRE");
map->locations[13].type = LOC_COAST;
map->locations[13].has_supply_center = 1;
map->locations[13].owner_power = -1;  // Neutral initially
map->locations[13].num_adjacent = 5;
map->locations[13].adjacencies[0] = 22;
map->locations[13].adjacencies[1] = 25;
map->locations[13].adjacencies[2] = 36;
map->locations[13].adjacencies[3] = 46;
map->locations[13].adjacencies[4] = 47;

// 14: BUD
strcpy(map->locations[14].name, "BUD");
map->locations[14].type = LOC_LAND;
map->locations[14].has_supply_center = 1;
map->locations[14].owner_power = -1;  // Neutral initially
map->locations[14].num_adjacent = 5;
map->locations[14].adjacencies[0] = 24;
map->locations[14].adjacencies[1] = 53;
map->locations[14].adjacencies[2] = 54;
map->locations[14].adjacencies[3] = 64;
map->locations[14].adjacencies[4] = 71;

// 15: BUL
strcpy(map->locations[15].name, "BUL");
map->locations[15].type = LOC_COAST;
map->locations[15].has_supply_center = 1;
map->locations[15].owner_power = -1;  // Neutral initially
map->locations[15].num_adjacent = 6;
map->locations[15].adjacencies[0] = 1;
map->locations[15].adjacencies[1] = 10;
map->locations[15].adjacencies[2] = 18;
map->locations[15].adjacencies[3] = 26;
map->locations[15].adjacencies[4] = 53;
map->locations[15].adjacencies[5] = 54;

// 16: BUR
strcpy(map->locations[16].name, "BUR");
map->locations[16].type = LOC_LAND;
map->locations[16].has_supply_center = 0;
map->locations[16].owner_power = -1;  // Neutral initially
map->locations[16].num_adjacent = 8;
map->locations[16].adjacencies[0] = 8;
map->locations[16].adjacencies[1] = 25;
map->locations[16].adjacencies[2] = 52;
map->locations[16].adjacencies[3] = 37;
map->locations[16].adjacencies[4] = 39;
map->locations[16].adjacencies[5] = 46;
map->locations[16].adjacencies[6] = 47;
map->locations[16].adjacencies[7] = 62;

// 17: CLY
strcpy(map->locations[17].name, "CLY");
map->locations[17].type = LOC_COAST;
map->locations[17].has_supply_center = 0;
map->locations[17].owner_power = -1;  // Neutral initially
map->locations[17].num_adjacent = 4;
map->locations[17].adjacencies[0] = 21;
map->locations[17].adjacencies[1] = 34;
map->locations[17].adjacencies[2] = 41;
map->locations[17].adjacencies[3] = 44;

// 18: CON
strcpy(map->locations[18].name, "CON");
map->locations[18].type = LOC_COAST;
map->locations[18].has_supply_center = 1;
map->locations[18].owner_power = -1;  // Neutral initially
map->locations[18].num_adjacent = 6;
map->locations[18].adjacencies[0] = 1;
map->locations[18].adjacencies[1] = 15;
map->locations[18].adjacencies[2] = 15;
map->locations[18].adjacencies[3] = 10;
map->locations[18].adjacencies[4] = 3;
map->locations[18].adjacencies[5] = 58;

// 19: DEN
strcpy(map->locations[19].name, "DEN");
map->locations[19].type = LOC_COAST;
map->locations[19].has_supply_center = 1;
map->locations[19].owner_power = -1;  // Neutral initially
map->locations[19].num_adjacent = 6;
map->locations[19].adjacencies[0] = 6;
map->locations[19].adjacencies[1] = 27;
map->locations[19].adjacencies[2] = 31;
map->locations[19].adjacencies[3] = 43;
map->locations[19].adjacencies[4] = 57;
map->locations[19].adjacencies[5] = 61;

// 20: EAS
strcpy(map->locations[20].name, "EAS");
map->locations[20].type = LOC_WATER;
map->locations[20].has_supply_center = 0;
map->locations[20].owner_power = -1;  // Neutral initially
map->locations[20].num_adjacent = 4;
map->locations[20].adjacencies[0] = 1;
map->locations[20].adjacencies[1] = 29;
map->locations[20].adjacencies[2] = 58;
map->locations[20].adjacencies[3] = 63;

// 21: EDI
strcpy(map->locations[21].name, "EDI");
map->locations[21].type = LOC_COAST;
map->locations[21].has_supply_center = 0;
map->locations[21].owner_power = -1;  // Neutral initially
map->locations[21].num_adjacent = 5;
map->locations[21].adjacencies[0] = 17;
map->locations[21].adjacencies[1] = 34;
map->locations[21].adjacencies[2] = 43;
map->locations[21].adjacencies[3] = 44;
map->locations[21].adjacencies[4] = 75;

// 22: ENG
strcpy(map->locations[22].name, "ENG");
map->locations[22].type = LOC_WATER;
map->locations[22].has_supply_center = 0;
map->locations[22].owner_power = -1;  // Neutral initially
map->locations[22].num_adjacent = 8;
map->locations[22].adjacencies[0] = 8;
map->locations[22].adjacencies[1] = 13;
map->locations[22].adjacencies[2] = 30;
map->locations[22].adjacencies[3] = 32;
map->locations[22].adjacencies[4] = 36;
map->locations[22].adjacencies[5] = 43;
map->locations[22].adjacencies[6] = 47;
map->locations[22].adjacencies[7] = 72;

// 23: FIN
strcpy(map->locations[23].name, "FIN");
map->locations[23].type = LOC_COAST;
map->locations[23].has_supply_center = 0;
map->locations[23].owner_power = -1;  // Neutral initially
map->locations[23].num_adjacent = 4;
map->locations[23].adjacencies[0] = 12;
map->locations[23].adjacencies[1] = 45;
map->locations[23].adjacencies[2] = 60;
map->locations[23].adjacencies[3] = 61;

// 24: GAL
strcpy(map->locations[24].name, "GAL");
map->locations[24].type = LOC_LAND;
map->locations[24].has_supply_center = 0;
map->locations[24].owner_power = -1;  // Neutral initially
map->locations[24].num_adjacent = 7;
map->locations[24].adjacencies[0] = 11;
map->locations[24].adjacencies[1] = 14;
map->locations[24].adjacencies[2] = 53;
map->locations[24].adjacencies[3] = 56;
map->locations[24].adjacencies[4] = 69;
map->locations[24].adjacencies[5] = 71;
map->locations[24].adjacencies[6] = 73;

// 25: GAS
strcpy(map->locations[25].name, "GAS");
map->locations[25].type = LOC_COAST;
map->locations[25].has_supply_center = 0;
map->locations[25].owner_power = -1;  // Neutral initially
map->locations[25].num_adjacent = 6;
map->locations[25].adjacencies[0] = 16;
map->locations[25].adjacencies[1] = 13;
map->locations[25].adjacencies[2] = 36;
map->locations[25].adjacencies[3] = 37;
map->locations[25].adjacencies[4] = 46;
map->locations[25].adjacencies[5] = 59;

// 26: GRE
strcpy(map->locations[26].name, "GRE");
map->locations[26].type = LOC_COAST;
map->locations[26].has_supply_center = 1;
map->locations[26].owner_power = -1;  // Neutral initially
map->locations[26].num_adjacent = 5;
map->locations[26].adjacencies[0] = 1;
map->locations[26].adjacencies[1] = 2;
map->locations[26].adjacencies[2] = 15;
map->locations[26].adjacencies[3] = 29;
map->locations[26].adjacencies[4] = 54;

// 27: HEL
strcpy(map->locations[27].name, "HEL");
map->locations[27].type = LOC_WATER;
map->locations[27].has_supply_center = 0;
map->locations[27].owner_power = -1;  // Neutral initially
map->locations[27].num_adjacent = 4;
map->locations[27].adjacencies[0] = 19;
map->locations[27].adjacencies[1] = 28;
map->locations[27].adjacencies[2] = 31;
map->locations[27].adjacencies[3] = 43;

// 28: HOL
strcpy(map->locations[28].name, "HOL");
map->locations[28].type = LOC_COAST;
map->locations[28].has_supply_center = 1;
map->locations[28].owner_power = -1;  // Neutral initially
map->locations[28].num_adjacent = 5;
map->locations[28].adjacencies[0] = 8;
map->locations[28].adjacencies[1] = 27;
map->locations[28].adjacencies[2] = 31;
map->locations[28].adjacencies[3] = 43;
map->locations[28].adjacencies[4] = 52;

// 29: ION
strcpy(map->locations[29].name, "ION");
map->locations[29].type = LOC_WATER;
map->locations[29].has_supply_center = 0;
map->locations[29].owner_power = -1;  // Neutral initially
map->locations[29].num_adjacent = 9;
map->locations[29].adjacencies[0] = 0;
map->locations[29].adjacencies[1] = 1;
map->locations[29].adjacencies[2] = 2;
map->locations[29].adjacencies[3] = 4;
map->locations[29].adjacencies[4] = 20;
map->locations[29].adjacencies[5] = 26;
map->locations[29].adjacencies[6] = 42;
map->locations[29].adjacencies[7] = 65;
map->locations[29].adjacencies[8] = 68;

// 30: IRI
strcpy(map->locations[30].name, "IRI");
map->locations[30].type = LOC_WATER;
map->locations[30].has_supply_center = 0;
map->locations[30].owner_power = -1;  // Neutral initially
map->locations[30].num_adjacent = 5;
map->locations[30].adjacencies[0] = 22;
map->locations[30].adjacencies[1] = 34;
map->locations[30].adjacencies[2] = 36;
map->locations[30].adjacencies[3] = 41;
map->locations[30].adjacencies[4] = 72;

// 31: KIE
strcpy(map->locations[31].name, "KIE");
map->locations[31].type = LOC_COAST;
map->locations[31].has_supply_center = 0;
map->locations[31].owner_power = -1;  // Neutral initially
map->locations[31].num_adjacent = 7;
map->locations[31].adjacencies[0] = 6;
map->locations[31].adjacencies[1] = 9;
map->locations[31].adjacencies[2] = 19;
map->locations[31].adjacencies[3] = 27;
map->locations[31].adjacencies[4] = 28;
map->locations[31].adjacencies[5] = 39;
map->locations[31].adjacencies[6] = 52;

// 32: LON
strcpy(map->locations[32].name, "LON");
map->locations[32].type = LOC_COAST;
map->locations[32].has_supply_center = 0;
map->locations[32].owner_power = -1;  // Neutral initially
map->locations[32].num_adjacent = 4;
map->locations[32].adjacencies[0] = 22;
map->locations[32].adjacencies[1] = 43;
map->locations[32].adjacencies[2] = 75;
map->locations[32].adjacencies[3] = 72;

// 33: LVN
strcpy(map->locations[33].name, "LVN");
map->locations[33].type = LOC_COAST;
map->locations[33].has_supply_center = 0;
map->locations[33].owner_power = -1;  // Neutral initially
map->locations[33].num_adjacent = 6;
map->locations[33].adjacencies[0] = 6;
map->locations[33].adjacencies[1] = 12;
map->locations[33].adjacencies[2] = 38;
map->locations[33].adjacencies[3] = 50;
map->locations[33].adjacencies[4] = 60;
map->locations[33].adjacencies[5] = 73;

// 34: LVP
strcpy(map->locations[34].name, "LVP");
map->locations[34].type = LOC_COAST;
map->locations[34].has_supply_center = 0;
map->locations[34].owner_power = -1;  // Neutral initially
map->locations[34].num_adjacent = 6;
map->locations[34].adjacencies[0] = 17;
map->locations[34].adjacencies[1] = 21;
map->locations[34].adjacencies[2] = 30;
map->locations[34].adjacencies[3] = 41;
map->locations[34].adjacencies[4] = 72;
map->locations[34].adjacencies[5] = 75;

// 35: LYO
strcpy(map->locations[35].name, "LYO");
map->locations[35].type = LOC_WATER;
map->locations[35].has_supply_center = 0;
map->locations[35].owner_power = -1;  // Neutral initially
map->locations[35].num_adjacent = 6;
map->locations[35].adjacencies[0] = 37;
map->locations[35].adjacencies[1] = 48;
map->locations[35].adjacencies[2] = 59;
map->locations[35].adjacencies[3] = 66;
map->locations[35].adjacencies[4] = 68;
map->locations[35].adjacencies[5] = 74;

// 36: MAO
strcpy(map->locations[36].name, "MAO");
map->locations[36].type = LOC_WATER;
map->locations[36].has_supply_center = 0;
map->locations[36].owner_power = -1;  // Neutral initially
map->locations[36].num_adjacent = 10;
map->locations[36].adjacencies[0] = 13;
map->locations[36].adjacencies[1] = 22;
map->locations[36].adjacencies[2] = 25;
map->locations[36].adjacencies[3] = 30;
map->locations[36].adjacencies[4] = 40;
map->locations[36].adjacencies[5] = 41;
map->locations[36].adjacencies[6] = 49;
map->locations[36].adjacencies[7] = 59;
map->locations[36].adjacencies[8] = 59;
map->locations[36].adjacencies[9] = 74;

// 37: MAR
strcpy(map->locations[37].name, "MAR");
map->locations[37].type = LOC_COAST;
map->locations[37].has_supply_center = 1;
map->locations[37].owner_power = -1;  // Neutral initially
map->locations[37].num_adjacent = 6;
map->locations[37].adjacencies[0] = 16;
map->locations[37].adjacencies[1] = 25;
map->locations[37].adjacencies[2] = 35;
map->locations[37].adjacencies[3] = 48;
map->locations[37].adjacencies[4] = 59;
map->locations[37].adjacencies[5] = 62;

// 38: MOS
strcpy(map->locations[38].name, "MOS");
map->locations[38].type = LOC_LAND;
map->locations[38].has_supply_center = 0;
map->locations[38].owner_power = -1;  // Neutral initially
map->locations[38].num_adjacent = 5;
map->locations[38].adjacencies[0] = 33;
map->locations[38].adjacencies[1] = 55;
map->locations[38].adjacencies[2] = 60;
map->locations[38].adjacencies[3] = 69;
map->locations[38].adjacencies[4] = 73;

// 39: MUN
strcpy(map->locations[39].name, "MUN");
map->locations[39].type = LOC_LAND;
map->locations[39].has_supply_center = 0;
map->locations[39].owner_power = -1;  // Neutral initially
map->locations[39].num_adjacent = 8;
map->locations[39].adjacencies[0] = 9;
map->locations[39].adjacencies[1] = 11;
map->locations[39].adjacencies[2] = 16;
map->locations[39].adjacencies[3] = 31;
map->locations[39].adjacencies[4] = 52;
map->locations[39].adjacencies[5] = 56;
map->locations[39].adjacencies[6] = 67;
map->locations[39].adjacencies[7] = 62;

// 40: NAF
strcpy(map->locations[40].name, "NAF");
map->locations[40].type = LOC_COAST;
map->locations[40].has_supply_center = 0;
map->locations[40].owner_power = -1;  // Neutral initially
map->locations[40].num_adjacent = 3;
map->locations[40].adjacencies[0] = 36;
map->locations[40].adjacencies[1] = 65;
map->locations[40].adjacencies[2] = 74;

// 41: NAO
strcpy(map->locations[41].name, "NAO");
map->locations[41].type = LOC_WATER;
map->locations[41].has_supply_center = 0;
map->locations[41].owner_power = -1;  // Neutral initially
map->locations[41].num_adjacent = 5;
map->locations[41].adjacencies[0] = 17;
map->locations[41].adjacencies[1] = 30;
map->locations[41].adjacencies[2] = 34;
map->locations[41].adjacencies[3] = 36;
map->locations[41].adjacencies[4] = 44;

// 42: NAP
strcpy(map->locations[42].name, "NAP");
map->locations[42].type = LOC_COAST;
map->locations[42].has_supply_center = 1;
map->locations[42].owner_power = -1;  // Neutral initially
map->locations[42].num_adjacent = 4;
map->locations[42].adjacencies[0] = 4;
map->locations[42].adjacencies[1] = 29;
map->locations[42].adjacencies[2] = 51;
map->locations[42].adjacencies[3] = 68;

// 43: NTH
strcpy(map->locations[43].name, "NTH");
map->locations[43].type = LOC_WATER;
map->locations[43].has_supply_center = 0;
map->locations[43].owner_power = -1;  // Neutral initially
map->locations[43].num_adjacent = 11;
map->locations[43].adjacencies[0] = 8;
map->locations[43].adjacencies[1] = 19;
map->locations[43].adjacencies[2] = 21;
map->locations[43].adjacencies[3] = 22;
map->locations[43].adjacencies[4] = 32;
map->locations[43].adjacencies[5] = 27;
map->locations[43].adjacencies[6] = 28;
map->locations[43].adjacencies[7] = 45;
map->locations[43].adjacencies[8] = 44;
map->locations[43].adjacencies[9] = 57;
map->locations[43].adjacencies[10] = 75;

// 44: NWG
strcpy(map->locations[44].name, "NWG");
map->locations[44].type = LOC_WATER;
map->locations[44].has_supply_center = 0;
map->locations[44].owner_power = -1;  // Neutral initially
map->locations[44].num_adjacent = 6;
map->locations[44].adjacencies[0] = 7;
map->locations[44].adjacencies[1] = 17;
map->locations[44].adjacencies[2] = 21;
map->locations[44].adjacencies[3] = 41;
map->locations[44].adjacencies[4] = 45;
map->locations[44].adjacencies[5] = 43;

// 45: NWY
strcpy(map->locations[45].name, "NWY");
map->locations[45].type = LOC_COAST;
map->locations[45].has_supply_center = 1;
map->locations[45].owner_power = -1;  // Neutral initially
map->locations[45].num_adjacent = 7;
map->locations[45].adjacencies[0] = 7;
map->locations[45].adjacencies[1] = 23;
map->locations[45].adjacencies[2] = 43;
map->locations[45].adjacencies[3] = 44;
map->locations[45].adjacencies[4] = 57;
map->locations[45].adjacencies[5] = 60;
map->locations[45].adjacencies[6] = 61;

// 46: PAR
strcpy(map->locations[46].name, "PAR");
map->locations[46].type = LOC_LAND;
map->locations[46].has_supply_center = 1;
map->locations[46].owner_power = -1;  // Neutral initially
map->locations[46].num_adjacent = 4;
map->locations[46].adjacencies[0] = 16;
map->locations[46].adjacencies[1] = 13;
map->locations[46].adjacencies[2] = 25;
map->locations[46].adjacencies[3] = 47;

// 47: PIC
strcpy(map->locations[47].name, "PIC");
map->locations[47].type = LOC_COAST;
map->locations[47].has_supply_center = 0;
map->locations[47].owner_power = -1;  // Neutral initially
map->locations[47].num_adjacent = 5;
map->locations[47].adjacencies[0] = 8;
map->locations[47].adjacencies[1] = 13;
map->locations[47].adjacencies[2] = 16;
map->locations[47].adjacencies[3] = 22;
map->locations[47].adjacencies[4] = 46;

// 48: PIE
strcpy(map->locations[48].name, "PIE");
map->locations[48].type = LOC_COAST;
map->locations[48].has_supply_center = 0;
map->locations[48].owner_power = -1;  // Neutral initially
map->locations[48].num_adjacent = 6;
map->locations[48].adjacencies[0] = 35;
map->locations[48].adjacencies[1] = 37;
map->locations[48].adjacencies[2] = 66;
map->locations[48].adjacencies[3] = 67;
map->locations[48].adjacencies[4] = 70;
map->locations[48].adjacencies[5] = 62;

// 49: POR
strcpy(map->locations[49].name, "POR");
map->locations[49].type = LOC_COAST;
map->locations[49].has_supply_center = 1;
map->locations[49].owner_power = -1;  // Neutral initially
map->locations[49].num_adjacent = 3;
map->locations[49].adjacencies[0] = 36;
map->locations[49].adjacencies[1] = 59;
map->locations[49].adjacencies[2] = 59;

// 50: PRU
strcpy(map->locations[50].name, "PRU");
map->locations[50].type = LOC_COAST;
map->locations[50].has_supply_center = 0;
map->locations[50].owner_power = -1;  // Neutral initially
map->locations[50].num_adjacent = 5;
map->locations[50].adjacencies[0] = 6;
map->locations[50].adjacencies[1] = 9;
map->locations[50].adjacencies[2] = 33;
map->locations[50].adjacencies[3] = 56;
map->locations[50].adjacencies[4] = 73;

// 51: ROM
strcpy(map->locations[51].name, "ROM");
map->locations[51].type = LOC_COAST;
map->locations[51].has_supply_center = 1;
map->locations[51].owner_power = -1;  // Neutral initially
map->locations[51].num_adjacent = 5;
map->locations[51].adjacencies[0] = 4;
map->locations[51].adjacencies[1] = 42;
map->locations[51].adjacencies[2] = 66;
map->locations[51].adjacencies[3] = 68;
map->locations[51].adjacencies[4] = 70;

// 52: RUH
strcpy(map->locations[52].name, "RUH");
map->locations[52].type = LOC_LAND;
map->locations[52].has_supply_center = 0;
map->locations[52].owner_power = -1;  // Neutral initially
map->locations[52].num_adjacent = 5;
map->locations[52].adjacencies[0] = 8;
map->locations[52].adjacencies[1] = 16;
map->locations[52].adjacencies[2] = 28;
map->locations[52].adjacencies[3] = 31;
map->locations[52].adjacencies[4] = 39;

// 53: RUM
strcpy(map->locations[53].name, "RUM");
map->locations[53].type = LOC_COAST;
map->locations[53].has_supply_center = 1;
map->locations[53].owner_power = -1;  // Neutral initially
map->locations[53].num_adjacent = 7;
map->locations[53].adjacencies[0] = 10;
map->locations[53].adjacencies[1] = 14;
map->locations[53].adjacencies[2] = 15;
map->locations[53].adjacencies[3] = 24;
map->locations[53].adjacencies[4] = 54;
map->locations[53].adjacencies[5] = 55;
map->locations[53].adjacencies[6] = 69;

// 54: SER
strcpy(map->locations[54].name, "SER");
map->locations[54].type = LOC_LAND;
map->locations[54].has_supply_center = 1;
map->locations[54].owner_power = -1;  // Neutral initially
map->locations[54].num_adjacent = 6;
map->locations[54].adjacencies[0] = 2;
map->locations[54].adjacencies[1] = 14;
map->locations[54].adjacencies[2] = 15;
map->locations[54].adjacencies[3] = 26;
map->locations[54].adjacencies[4] = 53;
map->locations[54].adjacencies[5] = 64;

// 55: SEV
strcpy(map->locations[55].name, "SEV");
map->locations[55].type = LOC_COAST;
map->locations[55].has_supply_center = 0;
map->locations[55].owner_power = -1;  // Neutral initially
map->locations[55].num_adjacent = 5;
map->locations[55].adjacencies[0] = 5;
map->locations[55].adjacencies[1] = 10;
map->locations[55].adjacencies[2] = 38;
map->locations[55].adjacencies[3] = 53;
map->locations[55].adjacencies[4] = 69;

// 56: SIL
strcpy(map->locations[56].name, "SIL");
map->locations[56].type = LOC_LAND;
map->locations[56].has_supply_center = 0;
map->locations[56].owner_power = -1;  // Neutral initially
map->locations[56].num_adjacent = 6;
map->locations[56].adjacencies[0] = 9;
map->locations[56].adjacencies[1] = 11;
map->locations[56].adjacencies[2] = 24;
map->locations[56].adjacencies[3] = 39;
map->locations[56].adjacencies[4] = 50;
map->locations[56].adjacencies[5] = 73;

// 57: SKA
strcpy(map->locations[57].name, "SKA");
map->locations[57].type = LOC_WATER;
map->locations[57].has_supply_center = 0;
map->locations[57].owner_power = -1;  // Neutral initially
map->locations[57].num_adjacent = 4;
map->locations[57].adjacencies[0] = 19;
map->locations[57].adjacencies[1] = 45;
map->locations[57].adjacencies[2] = 43;
map->locations[57].adjacencies[3] = 61;

// 58: SMY
strcpy(map->locations[58].name, "SMY");
map->locations[58].type = LOC_COAST;
map->locations[58].has_supply_center = 1;
map->locations[58].owner_power = -1;  // Neutral initially
map->locations[58].num_adjacent = 6;
map->locations[58].adjacencies[0] = 1;
map->locations[58].adjacencies[1] = 3;
map->locations[58].adjacencies[2] = 5;
map->locations[58].adjacencies[3] = 18;
map->locations[58].adjacencies[4] = 20;
map->locations[58].adjacencies[5] = 63;

// 59: SPA
strcpy(map->locations[59].name, "SPA");
map->locations[59].type = LOC_COAST;
map->locations[59].has_supply_center = 1;
map->locations[59].owner_power = -1;  // Neutral initially
map->locations[59].num_adjacent = 6;
map->locations[59].adjacencies[0] = 25;
map->locations[59].adjacencies[1] = 35;
map->locations[59].adjacencies[2] = 36;
map->locations[59].adjacencies[3] = 37;
map->locations[59].adjacencies[4] = 49;
map->locations[59].adjacencies[5] = 74;

// 60: STP
strcpy(map->locations[60].name, "STP");
map->locations[60].type = LOC_COAST;
map->locations[60].has_supply_center = 0;
map->locations[60].owner_power = -1;  // Neutral initially
map->locations[60].num_adjacent = 6;
map->locations[60].adjacencies[0] = 7;
map->locations[60].adjacencies[1] = 12;
map->locations[60].adjacencies[2] = 23;
map->locations[60].adjacencies[3] = 33;
map->locations[60].adjacencies[4] = 38;
map->locations[60].adjacencies[5] = 45;

// 61: SWE
strcpy(map->locations[61].name, "SWE");
map->locations[61].type = LOC_COAST;
map->locations[61].has_supply_center = 1;
map->locations[61].owner_power = -1;  // Neutral initially
map->locations[61].num_adjacent = 6;
map->locations[61].adjacencies[0] = 6;
map->locations[61].adjacencies[1] = 12;
map->locations[61].adjacencies[2] = 19;
map->locations[61].adjacencies[3] = 23;
map->locations[61].adjacencies[4] = 45;
map->locations[61].adjacencies[5] = 57;

// 62: SWI
strcpy(map->locations[62].name, "SWI");
map->locations[62].type = LOC_COAST;
map->locations[62].has_supply_center = 0;
map->locations[62].owner_power = -1;  // Neutral initially
map->locations[62].num_adjacent = 5;
map->locations[62].adjacencies[0] = 37;
map->locations[62].adjacencies[1] = 16;
map->locations[62].adjacencies[2] = 39;
map->locations[62].adjacencies[3] = 67;
map->locations[62].adjacencies[4] = 48;

// 63: SYR
strcpy(map->locations[63].name, "SYR");
map->locations[63].type = LOC_COAST;
map->locations[63].has_supply_center = 0;
map->locations[63].owner_power = -1;  // Neutral initially
map->locations[63].num_adjacent = 3;
map->locations[63].adjacencies[0] = 5;
map->locations[63].adjacencies[1] = 20;
map->locations[63].adjacencies[2] = 58;

// 64: TRI
strcpy(map->locations[64].name, "TRI");
map->locations[64].type = LOC_COAST;
map->locations[64].has_supply_center = 1;
map->locations[64].owner_power = -1;  // Neutral initially
map->locations[64].num_adjacent = 7;
map->locations[64].adjacencies[0] = 0;
map->locations[64].adjacencies[1] = 2;
map->locations[64].adjacencies[2] = 14;
map->locations[64].adjacencies[3] = 54;
map->locations[64].adjacencies[4] = 67;
map->locations[64].adjacencies[5] = 70;
map->locations[64].adjacencies[6] = 71;

// 65: TUN
strcpy(map->locations[65].name, "TUN");
map->locations[65].type = LOC_COAST;
map->locations[65].has_supply_center = 1;
map->locations[65].owner_power = -1;  // Neutral initially
map->locations[65].num_adjacent = 4;
map->locations[65].adjacencies[0] = 29;
map->locations[65].adjacencies[1] = 40;
map->locations[65].adjacencies[2] = 68;
map->locations[65].adjacencies[3] = 74;

// 66: TUS
strcpy(map->locations[66].name, "TUS");
map->locations[66].type = LOC_COAST;
map->locations[66].has_supply_center = 0;
map->locations[66].owner_power = -1;  // Neutral initially
map->locations[66].num_adjacent = 5;
map->locations[66].adjacencies[0] = 35;
map->locations[66].adjacencies[1] = 48;
map->locations[66].adjacencies[2] = 51;
map->locations[66].adjacencies[3] = 68;
map->locations[66].adjacencies[4] = 70;

// 67: TYR
strcpy(map->locations[67].name, "TYR");
map->locations[67].type = LOC_LAND;
map->locations[67].has_supply_center = 0;
map->locations[67].owner_power = -1;  // Neutral initially
map->locations[67].num_adjacent = 7;
map->locations[67].adjacencies[0] = 11;
map->locations[67].adjacencies[1] = 39;
map->locations[67].adjacencies[2] = 48;
map->locations[67].adjacencies[3] = 64;
map->locations[67].adjacencies[4] = 70;
map->locations[67].adjacencies[5] = 71;
map->locations[67].adjacencies[6] = 62;

// 68: TYS
strcpy(map->locations[68].name, "TYS");
map->locations[68].type = LOC_WATER;
map->locations[68].has_supply_center = 0;
map->locations[68].owner_power = -1;  // Neutral initially
map->locations[68].num_adjacent = 7;
map->locations[68].adjacencies[0] = 29;
map->locations[68].adjacencies[1] = 35;
map->locations[68].adjacencies[2] = 51;
map->locations[68].adjacencies[3] = 42;
map->locations[68].adjacencies[4] = 65;
map->locations[68].adjacencies[5] = 66;
map->locations[68].adjacencies[6] = 74;

// 69: UKR
strcpy(map->locations[69].name, "UKR");
map->locations[69].type = LOC_LAND;
map->locations[69].has_supply_center = 0;
map->locations[69].owner_power = -1;  // Neutral initially
map->locations[69].num_adjacent = 5;
map->locations[69].adjacencies[0] = 24;
map->locations[69].adjacencies[1] = 38;
map->locations[69].adjacencies[2] = 53;
map->locations[69].adjacencies[3] = 55;
map->locations[69].adjacencies[4] = 73;

// 70: VEN
strcpy(map->locations[70].name, "VEN");
map->locations[70].type = LOC_COAST;
map->locations[70].has_supply_center = 1;
map->locations[70].owner_power = -1;  // Neutral initially
map->locations[70].num_adjacent = 7;
map->locations[70].adjacencies[0] = 0;
map->locations[70].adjacencies[1] = 4;
map->locations[70].adjacencies[2] = 48;
map->locations[70].adjacencies[3] = 51;
map->locations[70].adjacencies[4] = 64;
map->locations[70].adjacencies[5] = 66;
map->locations[70].adjacencies[6] = 67;

// 71: VIE
strcpy(map->locations[71].name, "VIE");
map->locations[71].type = LOC_LAND;
map->locations[71].has_supply_center = 1;
map->locations[71].owner_power = -1;  // Neutral initially
map->locations[71].num_adjacent = 5;
map->locations[71].adjacencies[0] = 11;
map->locations[71].adjacencies[1] = 14;
map->locations[71].adjacencies[2] = 24;
map->locations[71].adjacencies[3] = 64;
map->locations[71].adjacencies[4] = 67;

// 72: WAL
strcpy(map->locations[72].name, "WAL");
map->locations[72].type = LOC_COAST;
map->locations[72].has_supply_center = 0;
map->locations[72].owner_power = -1;  // Neutral initially
map->locations[72].num_adjacent = 5;
map->locations[72].adjacencies[0] = 22;
map->locations[72].adjacencies[1] = 30;
map->locations[72].adjacencies[2] = 32;
map->locations[72].adjacencies[3] = 34;
map->locations[72].adjacencies[4] = 75;

// 73: WAR
strcpy(map->locations[73].name, "WAR");
map->locations[73].type = LOC_LAND;
map->locations[73].has_supply_center = 0;
map->locations[73].owner_power = -1;  // Neutral initially
map->locations[73].num_adjacent = 6;
map->locations[73].adjacencies[0] = 24;
map->locations[73].adjacencies[1] = 33;
map->locations[73].adjacencies[2] = 38;
map->locations[73].adjacencies[3] = 50;
map->locations[73].adjacencies[4] = 56;
map->locations[73].adjacencies[5] = 69;

// 74: WES
strcpy(map->locations[74].name, "WES");
map->locations[74].type = LOC_WATER;
map->locations[74].has_supply_center = 0;
map->locations[74].owner_power = -1;  // Neutral initially
map->locations[74].num_adjacent = 6;
map->locations[74].adjacencies[0] = 36;
map->locations[74].adjacencies[1] = 35;
map->locations[74].adjacencies[2] = 40;
map->locations[74].adjacencies[3] = 59;
map->locations[74].adjacencies[4] = 65;
map->locations[74].adjacencies[5] = 68;

// 75: YOR
strcpy(map->locations[75].name, "YOR");
map->locations[75].type = LOC_COAST;
map->locations[75].has_supply_center = 0;
map->locations[75].owner_power = -1;  // Neutral initially
map->locations[75].num_adjacent = 5;
map->locations[75].adjacencies[0] = 21;
map->locations[75].adjacencies[1] = 32;
map->locations[75].adjacencies[2] = 34;
map->locations[75].adjacencies[3] = 43;
map->locations[75].adjacencies[4] = 72;


// Home center assignments
// AUSTRIA
map->home_centers[0][0] = 14;  // BUD
map->locations[14].is_home_center = 0;
map->home_centers[0][1] = 64;  // TRI
map->locations[64].is_home_center = 0;
map->home_centers[0][2] = 71;  // VIE
map->locations[71].is_home_center = 0;
map->home_centers[0][4] = 14;  // BUD
map->locations[14].is_home_center = 0;
map->home_centers[0][6] = 71;  // VIE
map->locations[71].is_home_center = 0;
map->home_centers[0][8] = 64;  // TRI
map->locations[64].is_home_center = 0;
map->num_homes[0] = 10;

// FRANCE
map->home_centers[2][0] = 13;  // BRE
map->locations[13].is_home_center = 2;
map->home_centers[2][1] = 37;  // MAR
map->locations[37].is_home_center = 2;
map->home_centers[2][2] = 46;  // PAR
map->locations[46].is_home_center = 2;
map->home_centers[2][4] = 13;  // BRE
map->locations[13].is_home_center = 2;
map->home_centers[2][6] = 37;  // MAR
map->locations[37].is_home_center = 2;
map->home_centers[2][8] = 46;  // PAR
map->locations[46].is_home_center = 2;
map->num_homes[2] = 10;

// ITALY
map->home_centers[4][0] = 42;  // NAP
map->locations[42].is_home_center = 4;
map->home_centers[4][1] = 51;  // ROM
map->locations[51].is_home_center = 4;
map->home_centers[4][2] = 70;  // VEN
map->locations[70].is_home_center = 4;
map->home_centers[4][4] = 42;  // NAP
map->locations[42].is_home_center = 4;
map->home_centers[4][6] = 51;  // ROM
map->locations[51].is_home_center = 4;
map->home_centers[4][8] = 70;  // VEN
map->locations[70].is_home_center = 4;
map->num_homes[4] = 10;

// TURKEY
map->home_centers[6][0] = 3;  // ANK
map->locations[3].is_home_center = 6;
map->home_centers[6][1] = 18;  // CON
map->locations[18].is_home_center = 6;
map->home_centers[6][2] = 58;  // SMY
map->locations[58].is_home_center = 6;
map->home_centers[6][4] = 3;  // ANK
map->locations[3].is_home_center = 6;
map->home_centers[6][6] = 18;  // CON
map->locations[18].is_home_center = 6;
map->home_centers[6][8] = 58;  // SMY
map->locations[58].is_home_center = 6;
map->home_centers[6][10] = 8;  // BEL
map->locations[8].is_home_center = 6;
map->home_centers[6][11] = 15;  // BUL
map->locations[15].is_home_center = 6;
map->home_centers[6][12] = 19;  // DEN
map->locations[19].is_home_center = 6;
map->home_centers[6][13] = 26;  // GRE
map->locations[26].is_home_center = 6;
map->home_centers[6][14] = 28;  // HOL
map->locations[28].is_home_center = 6;
map->home_centers[6][15] = 45;  // NWY
map->locations[45].is_home_center = 6;
map->num_homes[6] = 17;

    // Override and correct home center assignments and supply centers
    // Reset all is_home_center to -1
    for (int i = 0; i < map->num_locations; i++) {
        map->locations[i].is_home_center = -1;
    }

    // Helper macro to set home centers
    #define SET_HOME(power_idx, name) \
        do { \
            int idx = find_location_by_name(map, name); \
            if (idx >= 0) { \
                int n = map->num_homes[power_idx]; \
                if (n < MAX_HOME_CENTERS) { \
                    map->home_centers[power_idx][n] = idx; \
                } \
                map->num_homes[power_idx] = n + 1; \
                map->locations[idx].is_home_center = power_idx; \
                map->locations[idx].has_supply_center = 1; \
            } \
        } while(0)

    // Clear counts
    for (int p = 0; p < MAX_POWERS; p++) map->num_homes[p] = 0;

    // Austria
    SET_HOME(0, "BUD"); SET_HOME(0, "TRI"); SET_HOME(0, "VIE");
    // England
    SET_HOME(1, "EDI"); SET_HOME(1, "LON"); SET_HOME(1, "LVP");
    // France
    SET_HOME(2, "BRE"); SET_HOME(2, "MAR"); SET_HOME(2, "PAR");
    // Germany
    SET_HOME(3, "BER"); SET_HOME(3, "KIE"); SET_HOME(3, "MUN");
    // Italy
    SET_HOME(4, "NAP"); SET_HOME(4, "ROM"); SET_HOME(4, "VEN");
    // Russia (4 homes)
    SET_HOME(5, "MOS"); SET_HOME(5, "SEV"); SET_HOME(5, "STP"); SET_HOME(5, "WAR");
    // Turkey
    SET_HOME(6, "ANK"); SET_HOME(6, "CON"); SET_HOME(6, "SMY");

    // Ensure neutral supply centers are marked
    const char* neutrals[] = {"BEL","BUL","DEN","GRE","HOL","NWY","POR","RUM","SER","SPA","SWE","TUN"};
    for (int i = 0; i < 12; i++) {
        int idx = find_location_by_name(map, neutrals[i]);
        if (idx >= 0) map->locations[idx].has_supply_center = 1;
    }

    #undef SET_HOME

    // Build adjacency cache for fast movement lookups
    // Initialize entire cache to 0
    memset(map->adjacency_cache, 0, sizeof(map->adjacency_cache));

    // For each location, mark valid adjacencies based on unit type
    for (int from = 0; from < map->num_locations; from++) {
        Location* from_loc = &map->locations[from];

        for (int adj_idx = 0; adj_idx < from_loc->num_adjacent; adj_idx++) {
            int to = from_loc->adjacencies[adj_idx];
            Location* to_loc = &map->locations[to];

            // ARMY can move: LAND<->LAND, LAND<->COAST, COAST<->COAST
            // (basically, if neither is pure water)
            if (from_loc->type != LOC_WATER && to_loc->type != LOC_WATER) {
                map->adjacency_cache[UNIT_ARMY][from][to] = 1;
            }

            // FLEET can move: WATER<->WATER, WATER<->COAST, COAST<->COAST
            // (basically, if neither is pure land)
            if (from_loc->type != LOC_LAND && to_loc->type != LOC_LAND) {
                map->adjacency_cache[UNIT_FLEET][from][to] = 1;
            }
        }
    }
}

void free_map(Map* map) {
    // No dynamic allocation in Map currently, but keep for future
    (void)map;  // Suppress unused parameter warning
}

// ============================================================================
// Game Initialization and Lifecycle
// ============================================================================

void init_game(GameState* game, Map* map, int welfare_mode, int max_years) {
    memset(game, 0, sizeof(GameState));

    game->map = map;
    game->year = 1901;
    game->phase = PHASE_SPRING_MOVEMENT;
    game->welfare_mode = welfare_mode;
    game->max_years = max_years;
    game->is_game_over = 0;

    // Initialize powers
    for (int p = 0; p < MAX_POWERS; p++) {
        game->powers[p].power_id = p;
        strcpy(game->powers[p].name, map->power_names[p]);
        strcpy(game->powers[p].abbrev, map->power_abbrev[p]);
        game->powers[p].welfare_points = 0;
        game->powers[p].num_centers = 0;
        game->powers[p].num_units = 0;
        game->powers[p].num_orders = 0;
        game->powers[p].num_retreats = 0;
        game->powers[p].adjustment = 0;
    }

    // Set up initial unit positions and supply centers
    // Power IDs: 0=AUSTRIA, 1=ENGLAND, 2=FRANCE, 3=GERMANY, 4=ITALY, 5=RUSSIA, 6=TURKEY

    // AUSTRIA (0): F TRI, A VIE, A BUD
    game->powers[0].units[0] = (Unit){UNIT_FLEET, 64, 0, 0}; // F TRI
    game->powers[0].units[1] = (Unit){UNIT_ARMY, 71, 0, 0};  // A VIE
    game->powers[0].units[2] = (Unit){UNIT_ARMY, 14, 0, 0};  // A BUD
    game->powers[0].num_units = 3;
    game->powers[0].centers[0] = 64; // TRI
    game->powers[0].centers[1] = 71; // VIE
    game->powers[0].centers[2] = 14; // BUD
    game->powers[0].num_centers = 3;

    // ENGLAND (1): F LON, F EDI, A LVP
    game->powers[1].units[0] = (Unit){UNIT_FLEET, 32, 1, 0}; // F LON
    game->powers[1].units[1] = (Unit){UNIT_FLEET, 21, 1, 0}; // F EDI
    game->powers[1].units[2] = (Unit){UNIT_ARMY, 34, 1, 0};  // A LVP
    game->powers[1].num_units = 3;
    game->powers[1].centers[0] = 32; // LON
    game->powers[1].centers[1] = 21; // EDI
    game->powers[1].centers[2] = 34; // LVP
    game->powers[1].num_centers = 3;

    // FRANCE (2): F BRE, A PAR, A MAR
    game->powers[2].units[0] = (Unit){UNIT_FLEET, 13, 2, 0}; // F BRE
    game->powers[2].units[1] = (Unit){UNIT_ARMY, 46, 2, 0};  // A PAR
    game->powers[2].units[2] = (Unit){UNIT_ARMY, 37, 2, 0};  // A MAR
    game->powers[2].num_units = 3;
    game->powers[2].centers[0] = 13; // BRE
    game->powers[2].centers[1] = 46; // PAR
    game->powers[2].centers[2] = 37; // MAR
    game->powers[2].num_centers = 3;

    // GERMANY (3): F KIE, A BER, A MUN
    game->powers[3].units[0] = (Unit){UNIT_FLEET, 31, 3, 0}; // F KIE
    game->powers[3].units[1] = (Unit){UNIT_ARMY, 9, 3, 0};   // A BER
    game->powers[3].units[2] = (Unit){UNIT_ARMY, 39, 3, 0};  // A MUN
    game->powers[3].num_units = 3;
    game->powers[3].centers[0] = 31; // KIE
    game->powers[3].centers[1] = 9;  // BER
    game->powers[3].centers[2] = 39; // MUN
    game->powers[3].num_centers = 3;

    // ITALY (4): F NAP, A ROM, A VEN
    game->powers[4].units[0] = (Unit){UNIT_FLEET, 42, 4, 0}; // F NAP
    game->powers[4].units[1] = (Unit){UNIT_ARMY, 51, 4, 0};  // A ROM
    game->powers[4].units[2] = (Unit){UNIT_ARMY, 70, 4, 0};  // A VEN
    game->powers[4].num_units = 3;
    game->powers[4].centers[0] = 42; // NAP
    game->powers[4].centers[1] = 51; // ROM
    game->powers[4].centers[2] = 70; // VEN
    game->powers[4].num_centers = 3;

    // RUSSIA (5): F SEV, F STP(sc), A MOS, A WAR
    game->powers[5].units[0] = (Unit){UNIT_FLEET, 55, 5, 0}; // F SEV
    game->powers[5].units[1] = (Unit){UNIT_FLEET, 60, 5, 0}; // F STP
    game->powers[5].units[2] = (Unit){UNIT_ARMY, 38, 5, 0};  // A MOS
    game->powers[5].units[3] = (Unit){UNIT_ARMY, 73, 5, 0};  // A WAR
    game->powers[5].num_units = 4;
    game->powers[5].centers[0] = 55; // SEV
    game->powers[5].centers[1] = 60; // STP
    game->powers[5].centers[2] = 38; // MOS
    game->powers[5].centers[3] = 73; // WAR
    game->powers[5].num_centers = 4;

    // TURKEY (6): F ANK, A CON, A SMY
    game->powers[6].units[0] = (Unit){UNIT_FLEET, 3, 6, 0};  // F ANK
    game->powers[6].units[1] = (Unit){UNIT_ARMY, 18, 6, 0};  // A CON
    game->powers[6].units[2] = (Unit){UNIT_ARMY, 58, 6, 0};  // A SMY
    game->powers[6].num_units = 3;
    game->powers[6].centers[0] = 3;  // ANK
    game->powers[6].centers[1] = 18; // CON
    game->powers[6].centers[2] = 58; // SMY
    game->powers[6].num_centers = 3;
}

void reset_game(GameState* game) {
    int welfare_mode = game->welfare_mode;
    int max_years = game->max_years;
    Map* map = game->map;

    init_game(game, map, welfare_mode, max_years);
}

void free_game(GameState* game) {
    // No dynamic allocation in GameState currently, but keep for future
    (void)game;
}

// ============================================================================
// Order Handling
// ============================================================================

int parse_order(const char* order_str, Order* order, GameState* game) {
    // Parse standard Diplomacy order notation
    // Format examples:
    // "A PAR - MAR" (move)
    // "A PAR H" or "A PAR HOLDS" (hold)
    // "A PAR S A MAR - BUR" (support move)
    // "A PAR S A MAR" (support hold)
    // "F ENG C A WAL - BRE" (convoy)

    if (!order_str || !order || !game) {
        return -1;
    }

    memset(order, 0, sizeof(Order));

    char buffer[256];
    strncpy(buffer, order_str, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';

    // Convert to uppercase for easier parsing
    for (char* p = buffer; *p; p++) {
        *p = toupper(*p);
    }

    // Parse unit type (A or F)
    char* token = strtok(buffer, " ");
    if (!token) return -1;

    if (token[0] == 'A') {
        order->unit_type = UNIT_ARMY;
    } else if (token[0] == 'F') {
        order->unit_type = UNIT_FLEET;
    } else {
        return -1;  // Invalid unit type
    }

    // Parse unit location
    token = strtok(NULL, " ");
    if (!token) return -1;

    int unit_loc = find_location_by_name(game->map, token);
    if (unit_loc < 0) return -1;
    order->unit_location = unit_loc;

    // Parse order type
    token = strtok(NULL, " ");
    if (!token) return -1;

    if (strcmp(token, "H") == 0 || strcmp(token, "HOLDS") == 0) {
        // HOLD order
        order->type = ORDER_HOLD;
        return 0;

    } else if (strcmp(token, "-") == 0 || strcmp(token, "->") == 0) {
        // MOVE order
        order->type = ORDER_MOVE;

        // Parse destination
        token = strtok(NULL, " ");
        if (!token) return -1;

        int dest_loc = find_location_by_name(game->map, token);
        if (dest_loc < 0) return -1;
        order->target_location = dest_loc;
        return 0;

    } else if (strcmp(token, "S") == 0 || strcmp(token, "SUPPORT") == 0 ||
               strcmp(token, "SUPPORTS") == 0) {
        // SUPPORT order

        // Parse supported unit type
        token = strtok(NULL, " ");
        if (!token) return -1;

        // Skip unit type (already know from next token)
        // Parse supported unit location
        token = strtok(NULL, " ");
        if (!token) return -1;

        int supported_loc = find_location_by_name(game->map, token);
        if (supported_loc < 0) return -1;
        order->target_unit_location = supported_loc;

        // Check if it's support to hold or support to move
        token = strtok(NULL, " ");
        if (token && (strcmp(token, "-") == 0 || strcmp(token, "->") == 0)) {
            // Support to move
            order->type = ORDER_SUPPORT_MOVE;
            token = strtok(NULL, " ");
            if (!token) return -1;

            int dest_loc = find_location_by_name(game->map, token);
            if (dest_loc < 0) return -1;
            order->dest_location = dest_loc;
        } else {
            // Support to hold (no destination)
            order->type = ORDER_SUPPORT_HOLD;
            order->dest_location = supported_loc;
        }
        return 0;

    } else if (strcmp(token, "C") == 0 || strcmp(token, "CONVOY") == 0 ||
               strcmp(token, "CONVOYS") == 0) {
        // CONVOY order
        order->type = ORDER_CONVOY;

        // Parse convoyed unit type (should be A)
        token = strtok(NULL, " ");
        if (!token) return -1;

        // Parse convoyed unit location
        token = strtok(NULL, " ");
        if (!token) return -1;

        int convoyed_loc = find_location_by_name(game->map, token);
        if (convoyed_loc < 0) return -1;
        order->target_unit_location = convoyed_loc;

        // Parse "-"
        token = strtok(NULL, " ");
        if (!token || (strcmp(token, "-") != 0 && strcmp(token, "->") != 0)) {
            return -1;
        }

        // Parse destination
        token = strtok(NULL, " ");
        if (!token) return -1;

        int dest_loc = find_location_by_name(game->map, token);
        if (dest_loc < 0) return -1;
        order->dest_location = dest_loc;
        return 0;

    } else if (strcmp(token, "B") == 0 || strcmp(token, "BUILD") == 0 ||
               strcmp(token, "BUILDS") == 0) {
        // BUILD order
        order->type = ORDER_BUILD;
        return 0;

    } else if (strcmp(token, "D") == 0 || strcmp(token, "DISBAND") == 0 ||
               strcmp(token, "DISBANDS") == 0 || strcmp(token, "REMOVE") == 0) {
        // DISBAND order
        order->type = ORDER_DISBAND;
        return 0;

    } else {
        return -1;  // Unknown order type
    }
}

int validate_order(GameState* game, int power_id, const Order* order) {
    // Validate that an order is legal for the current game state
    // Returns 0 if valid, -1 if invalid

    if (!game || !order || power_id < 0 || power_id >= MAX_POWERS) {
        return -1;
    }

    Power* power = &game->powers[power_id];
    Map* map = game->map;

    // Find unit at the specified location
    int unit_idx = -1;
    for (int i = 0; i < power->num_units; i++) {
        if (power->units[i].location == order->unit_location) {
            unit_idx = i;
            break;
        }
    }

    // Check unit exists and belongs to this power
    if (unit_idx < 0) {
        return -1;  // No unit at this location
    }

    Unit* unit = &power->units[unit_idx];

    // Check unit type matches
    if (unit->type != order->unit_type) {
        return -1;  // Unit type mismatch
    }

    // Validate based on order type
    switch (order->type) {
        case ORDER_HOLD:
            // HOLD is always valid
            return 0;

        case ORDER_MOVE:
            // Check destination is adjacent and reachable by unit type
            if (order->target_location < 0 || order->target_location >= map->num_locations) {
                return -1;  // Invalid destination
            }
            // Check not moving to same location
            if (order->target_location == unit->location) {
                return -1;  // Cannot move to own sector
            }
            // Check if move is adjacent (normal move)
            if (can_move(map, unit->type, unit->location, order->target_location)) {
                return 0;  // Valid adjacent move
            }
            // If not adjacent and unit is army, check for convoy path
            if (unit->type == UNIT_ARMY && is_convoyed_move(game, unit->location, order->target_location)) {
                return 0;  // Valid convoy move
            }
            return -1;  // Not adjacent and no convoy path

        case ORDER_SUPPORT_HOLD:
        case ORDER_SUPPORT_MOVE:
            // Check target unit location exists
            if (order->target_unit_location < 0 || order->target_unit_location >= map->num_locations) {
                return -1;
            }
            // Check supporting unit can reach the target location or destination
            // (supporting unit must be adjacent to either the supported unit or the destination)
            int can_support = 0;
            if (can_move(map, unit->type, unit->location, order->target_unit_location)) {
                can_support = 1;  // Adjacent to supported unit
            }
            if (order->dest_location >= 0 && order->dest_location < map->num_locations) {
                if (can_move(map, unit->type, unit->location, order->dest_location)) {
                    can_support = 1;  // Adjacent to destination
                }
            }
            if (!can_support) {
                return -1;  // Can't support this move
            }
            return 0;

        case ORDER_CONVOY:
            // Convoy must be given by a fleet
            if (unit->type != UNIT_FLEET) {
                return -1;  // Only fleets can convoy
            }
            // Check fleet is in WATER (not COAST) - can't convoy from coastal areas
            if (!can_fleet_convoy(map, unit->location)) {
                return -1;  // Fleet must be in water to convoy
            }
            // Check convoyed unit location exists
            if (order->target_unit_location < 0 || order->target_unit_location >= map->num_locations) {
                return -1;
            }
            // Check destination exists
            if (order->dest_location < 0 || order->dest_location >= map->num_locations) {
                return -1;
            }
            // Convoy order is valid if fleet can convoy
            return 0;

        case ORDER_BUILD:
            // Can only build in home centers during adjustment phase
            if (game->phase != PHASE_WINTER_ADJUSTMENT) {
                return -1;  // Wrong phase
            }
            // TODO: Check location is home center and unoccupied
            return 0;

        case ORDER_DISBAND:
            // Can always disband own units (in welfare diplomacy)
            // In standard diplomacy, only during adjustment phase if over limit
            if (!game->welfare_mode && game->phase != PHASE_WINTER_ADJUSTMENT) {
                return -1;  // Wrong phase for standard diplomacy
            }
            return 0;

        default:
            return -1;  // Unknown order type
    }
}

void get_possible_orders(GameState* game, int power_id, int location,
                         char orders[][MAX_ORDER_LENGTH], int* num_orders) {
    *num_orders = 0;
    if (!game || power_id < 0 || power_id >= MAX_POWERS) return;
    Power* power = &game->powers[power_id];
    int unit_idx = -1;
    for (int i = 0; i < power->num_units; i++) {
        if (power->units[i].location == location) {
            unit_idx = i;
            break;
        }
    }
    if (unit_idx < 0) return; // no unit
    Unit* unit = &power->units[unit_idx];

    // HOLD always available
    snprintf(orders[*num_orders], MAX_ORDER_LENGTH, "%s %s H",
             unit->type == UNIT_ARMY ? "A" : "F",
             game->map->locations[unit->location].name);
    (*num_orders)++;

    // MOVE to any adjacent and reachable location
    for (int a = 0; a < game->map->locations[location].num_adjacent; a++) {
        int to = game->map->locations[location].adjacencies[a];
        if (can_move(game->map, unit->type, location, to)) {
            snprintf(orders[*num_orders], MAX_ORDER_LENGTH, "%s %s - %s",
                     unit->type == UNIT_ARMY ? "A" : "F",
                     game->map->locations[unit->location].name,
                     game->map->locations[to].name);
            (*num_orders)++;
            if (*num_orders >= MAX_UNITS) break; // cap defensively
        }
    }
}

// ============================================================================
// Convoy Pathfinding
// ============================================================================

// Check if a fleet at given location can convoy (must be in water, not coast)
int can_fleet_convoy(Map* map, int location) {
    if (location < 0 || location >= map->num_locations) {
        return 0;
    }

    Location* loc = &map->locations[location];
    // Fleet can convoy only if in WATER (not COAST, not LAND)
    return (loc->type == LOC_WATER);
}

// BFS to find if there's a convoy path from start to end through convoying fleets
// Returns 1 if path exists, 0 otherwise
// convoying_fleets: array of locations where fleets are offering to convoy
// num_convoying_fleets: number of such fleets
int find_convoy_path(Map* map, int start, int end,
                     int* convoying_fleets, int num_convoying_fleets) {
    if (start == end) return 0;  // Can't convoy to same location

    // Check if start and end are coastal (armies can only convoy from/to coast)
    Location* start_loc = &map->locations[start];
    Location* end_loc = &map->locations[end];

    if (start_loc->type != LOC_COAST && start_loc->type != LOC_PORT) return 0;
    if (end_loc->type != LOC_COAST && end_loc->type != LOC_PORT) return 0;

    // BFS to find path through convoying fleets
    // Queue for BFS
    int queue[MAX_LOCATIONS];
    int visited[MAX_LOCATIONS] = {0};
    int queue_start = 0, queue_end = 0;

    // Start with convoying fleets adjacent to start location
    for (int i = 0; i < num_convoying_fleets; i++) {
        int fleet_loc = convoying_fleets[i];

        // Check if this fleet is adjacent to start
        int adjacent = 0;
        for (int j = 0; j < start_loc->num_adjacent; j++) {
            if (start_loc->adjacencies[j] == fleet_loc) {
                adjacent = 1;
                break;
            }
        }

        if (adjacent && can_fleet_convoy(map, fleet_loc)) {
            queue[queue_end++] = fleet_loc;
            visited[fleet_loc] = 1;
        }
    }

    // BFS through convoying fleets
    while (queue_start < queue_end) {
        int current = queue[queue_start++];
        Location* current_loc = &map->locations[current];

        // Check if this fleet is adjacent to destination
        for (int i = 0; i < current_loc->num_adjacent; i++) {
            if (current_loc->adjacencies[i] == end) {
                return 1;  // Found a path!
            }
        }

        // Expand to adjacent convoying fleets
        for (int i = 0; i < current_loc->num_adjacent; i++) {
            int adj = current_loc->adjacencies[i];

            if (visited[adj]) continue;

            // Check if there's a convoying fleet at this location
            for (int j = 0; j < num_convoying_fleets; j++) {
                if (convoying_fleets[j] == adj && can_fleet_convoy(map, adj)) {
                    queue[queue_end++] = adj;
                    visited[adj] = 1;
                    break;
                }
            }
        }
    }

    return 0;  // No path found
}

// Get list of fleets offering to convoy an army from start to end
// Returns number of convoying fleets found
int get_convoying_fleets(GameState* game, int start, int end, int* convoying_fleets) {
    int count = 0;

    // Look through all powers' orders
    for (int p = 0; p < MAX_POWERS; p++) {
        Power* power = &game->powers[p];

        for (int o = 0; o < power->num_orders; o++) {
            Order* order = &power->orders[o];

            // Check if this is a convoy order for the right army move
            if (order->type == ORDER_CONVOY &&
                order->target_unit_location == start &&
                order->dest_location == end) {

                // Find the fleet's location (order->unit_location is where the fleet is)
                int fleet_loc = order->unit_location;

                // Only add if fleet is in water and can actually convoy
                if (can_fleet_convoy(game->map, fleet_loc)) {
                    convoying_fleets[count++] = fleet_loc;
                    if (count >= MAX_LOCATIONS) break;
                }
            }
        }
    }

    return count;
}

// Check if an army move should be via convoy (has valid convoy path)
int is_convoyed_move(GameState* game, int from, int to) {
    // Get all fleets offering to convoy this move
    int convoying_fleets[MAX_LOCATIONS];
    int num_fleets = get_convoying_fleets(game, from, to, convoying_fleets);

    if (num_fleets == 0) {
        return 0;  // No fleets offering to convoy
    }

    // Check if there's a valid path through these fleets
    return find_convoy_path(game->map, from, to, convoying_fleets, num_fleets);
}

int detect_paradox(GameState* game, int starting_location, int convoying_fleet_location) {
    // Paradox detection: Follow chain from starting_location to see if we eventually
    // find a support order supporting the convoying fleet
    // Returns 1 if paradox detected, 0 otherwise

    int visited_locations[MAX_LOCATIONS];
    int num_visited = 0;
    int current_location = starting_location;

    // Follow the chain for up to MAX_LOCATIONS steps
    while (num_visited < MAX_LOCATIONS) {
        // Check if we've visited this location before (cycle detection)
        for (int i = 0; i < num_visited; i++) {
            if (visited_locations[i] == current_location) {
                return 0;  // Cycle detected, no paradox
            }
        }
        visited_locations[num_visited++] = current_location;

        // Find unit at current location
        int unit_power = -1;
        Order* current_order = NULL;

        for (int p = 0; p < MAX_POWERS; p++) {
            Power* power = &game->powers[p];
            for (int u = 0; u < power->num_units; u++) {
                if (power->units[u].location == current_location) {
                    // Found unit, now find its order
                    for (int o = 0; o < power->num_orders; o++) {
                        if (power->orders[o].unit_location == current_location) {
                            current_order = &power->orders[o];
                            unit_power = p;
                            break;
                        }
                    }
                    break;
                }
            }
            if (current_order) break;
        }

        if (!current_order) {
            return 0;  // No unit or no order at this location
        }

        // Check if this is a support order for the convoying fleet
        if (current_order->type == ORDER_SUPPORT_HOLD || current_order->type == ORDER_SUPPORT_MOVE) {
            // Check if this support is for the fleet at convoying_fleet_location
            if (current_order->target_unit_location == convoying_fleet_location) {
                return 1;  // Paradox detected!
            }
        }

        // Only continue chain if order is Support or Convoy
        if (current_order->type != ORDER_SUPPORT_HOLD &&
            current_order->type != ORDER_SUPPORT_MOVE &&
            current_order->type != ORDER_CONVOY) {
            return 0;  // Chain ends
        }

        // Move to the next location in the chain
        if (current_order->type == ORDER_SUPPORT_MOVE || current_order->type == ORDER_CONVOY) {
            current_location = current_order->dest_location;
        } else {
            return 0;  // Hold support doesn't continue the chain
        }
    }

    return 0;  // No paradox found
}

// ============================================================================
// Phase Processing
// ============================================================================

void process_orders(GameState* game) {
    // Dispatch to appropriate phase handler
    switch (game->phase) {
        case PHASE_SPRING_MOVEMENT:
        case PHASE_FALL_MOVEMENT:
            resolve_movement_phase(game);
            break;
        case PHASE_SPRING_RETREAT:
        case PHASE_FALL_RETREAT:
            resolve_retreat_phase(game);
            break;
        case PHASE_WINTER_ADJUSTMENT:
            resolve_adjustment_phase(game);
            break;
        case PHASE_COMPLETED:
            // No processing needed
            break;
    }
}

void resolve_movement_phase(GameState* game) {
    // Implementation of Diplomacy movement adjudication
    // Algorithm:
    // 1. Validate all orders
    // 2. Calculate destination conflicts
    // 3. Calculate attack/defense strengths
    // 4. Determine successful moves and dislodgements
    // 5. Apply moves and record dislodged units

    // Clear any previous dislodgements
    game->num_dislodged = 0;
    game->num_combats = 0;

    // Structure to track move attempts
    typedef struct {
        int unit_power;          // Which power owns the unit
        int unit_idx;            // Index in power's unit array
        int from_location;       // Where unit is
        int to_location;         // Where it's trying to go (-1 for hold)
        UnitType unit_type;
        int is_valid;            // Is this a valid order?
        int attack_strength;     // Strength moving into to_location
        int defend_strength;     // Strength defending from_location
        int can_move;            // Final determination: can this unit move?
        int support_cut;         // Has this unit's support been cut?
        int is_convoyed;         // Is this a convoyed move?
        int convoy_disrupted;    // Has the convoy been disrupted?
    } MoveAttempt;
    
    // Structure to track support orders
    typedef struct {
        int supporter_power;     // Power giving support
        int supporter_location;  // Where supporter is
        int supported_location;  // Where supported unit is (or -1 for hold)
        int destination;         // Where supported unit is going (or where it is for hold support)
        int is_valid;            // Is this support valid?
        int is_cut;              // Has support been cut by attack?
    } SupportOrder;

    MoveAttempt attempts[MAX_POWERS * MAX_UNITS];
    int num_attempts = 0;
    
    SupportOrder supports[MAX_POWERS * MAX_UNITS];
    int num_supports = 0;

    // Step 1: Collect all move attempts from orders
    for (int p = 0; p < MAX_POWERS; p++) {
        Power* power = &game->powers[p];
        
        // Create default HOLD orders for units without explicit orders
        for (int u = 0; u < power->num_units; u++) {
            Unit* unit = &power->units[u];
            int has_order = 0;
            
            // Check if unit has an explicit order
            for (int o = 0; o < power->num_orders; o++) {
                if (power->orders[o].unit_location == unit->location) {
                    has_order = 1;
                    break;
                }
            }
            
            // If no order, create implicit HOLD
            if (!has_order && power->num_orders < MAX_UNITS) {
                Order hold_order;
                memset(&hold_order, 0, sizeof(Order));
                hold_order.type = ORDER_HOLD;
                hold_order.unit_location = unit->location;
                hold_order.unit_type = unit->type;
                hold_order.power_id = p;
                power->orders[power->num_orders++] = hold_order;
            }
        }
        
        // Process all orders for this power
        for (int o = 0; o < power->num_orders; o++) {
            Order* order = &power->orders[o];
            
            // Find the unit for this order
            int unit_idx = -1;
            for (int u = 0; u < power->num_units; u++) {
                if (power->units[u].location == order->unit_location) {
                    unit_idx = u;
                    break;
                }
            }
            
            if (unit_idx < 0) {
                continue;  // Order for non-existent unit
            }
            
            // Handle different order types
            if (order->type == ORDER_HOLD || order->type == ORDER_MOVE) {
                MoveAttempt* attempt = &attempts[num_attempts++];
                attempt->unit_power = p;
                attempt->unit_idx = unit_idx;
                attempt->from_location = order->unit_location;
                attempt->unit_type = order->unit_type;
                attempt->attack_strength = 1;  // Base strength (supports added later)
                attempt->defend_strength = 1;  // Base strength (supports added later)
                attempt->can_move = 0;         // Determined later
                attempt->support_cut = 0;      // Not supporting, so can't be cut
                attempt->is_convoyed = 0;      // Determined below for armies
                attempt->convoy_disrupted = 0; // Checked later

                if (order->type == ORDER_HOLD) {
                    attempt->to_location = -1;  // Holding
                    attempt->is_valid = 1;
                } else {  // ORDER_MOVE
                    attempt->to_location = order->target_location;
                    // Validate move
                    int valid = validate_order(game, p, order);
                    attempt->is_valid = (valid == 0);

                    // Check if this is a convoyed move (army moving to non-adjacent location)
                    if (attempt->is_valid && order->unit_type == UNIT_ARMY) {
                        int is_adjacent = can_move(game->map, order->unit_type,
                                                   order->unit_location, order->target_location);
                        if (!is_adjacent) {
                            // Army move to non-adjacent location requires convoy
                            attempt->is_convoyed = 1;
                        }
                    }
                }
            } 
            else if (order->type == ORDER_SUPPORT_HOLD || order->type == ORDER_SUPPORT_MOVE) {
                // Collect support orders
                SupportOrder* support = &supports[num_supports++];
                support->supporter_power = p;
                support->supporter_location = order->unit_location;
                support->supported_location = order->target_unit_location;
                support->destination = (order->type == ORDER_SUPPORT_MOVE) ?
                                      order->dest_location : order->target_unit_location;
                support->is_valid = (validate_order(game, p, order) == 0);
                support->is_cut = 0;  // Determined later

                // Also add this unit to move attempts as HOLD (supporting units hold their position)
                MoveAttempt* attempt = &attempts[num_attempts++];
                attempt->unit_power = p;
                attempt->unit_idx = unit_idx;
                attempt->from_location = order->unit_location;
                attempt->to_location = -1;  // Supporting units hold
                attempt->unit_type = order->unit_type;
                attempt->attack_strength = 1;
                attempt->defend_strength = 1;
                attempt->can_move = 0;
                attempt->is_valid = 1;
                attempt->support_cut = 0;  // Will be set if attacked
                attempt->is_convoyed = 0;  // Not a move
                attempt->convoy_disrupted = 0;
            }
            else if (order->type == ORDER_CONVOY) {
                // Convoying units also hold their position and can defend
                MoveAttempt* attempt = &attempts[num_attempts++];
                attempt->unit_power = p;
                attempt->unit_idx = unit_idx;
                attempt->from_location = order->unit_location;
                attempt->to_location = -1;  // Convoying units hold
                attempt->unit_type = order->unit_type;
                attempt->attack_strength = 1;
                attempt->defend_strength = 1;
                attempt->can_move = 0;
                attempt->is_valid = (validate_order(game, p, order) == 0);
                attempt->support_cut = 0;  // Not used for convoy, but initialize
                attempt->is_convoyed = 0;  // Not a move
                attempt->convoy_disrupted = 0;
            }
        }
    }

    // Step 2: Determine which supports are cut by attacks
    // A support is cut if the supporter is attacked by any unit (except the supported unit)
    // NOTE: Convoyed moves are checked later after convoy disruption (Step 6b)
    for (int i = 0; i < num_attempts; i++) {
        MoveAttempt* attacker = &attempts[i];

        if (!attacker->is_valid || attacker->to_location < 0) {
            continue;  // Not attacking
        }

        // Skip convoyed moves - they only cut support if convoy succeeds
        // (checked later after convoy disruption)
        if (attacker->is_convoyed) {
            continue;
        }

        // Check if this attack cuts any supports
        for (int s = 0; s < num_supports; s++) {
            SupportOrder* support = &supports[s];

            if (!support->is_valid || support->is_cut) {
                continue;  // Already invalid or cut
            }

            // Support is cut if attacker is moving to supporter's location
            if (attacker->to_location == support->supporter_location) {
                // Exception: attack from the unit being supported doesn't cut support
                if (attacker->from_location != support->supported_location) {
                    // Exception: own units don't cut support
                    if (attacker->unit_power != support->supporter_power) {
                        support->is_cut = 1;
                    }
                }
            }
        }
    }
    
    // Step 3: Calculate attack and defense strengths with valid supports
    for (int i = 0; i < num_attempts; i++) {
        MoveAttempt* attempt = &attempts[i];
        
        if (!attempt->is_valid) {
            continue;
        }
        
        // Calculate attack strength (if moving)
        if (attempt->to_location >= 0) {
            // Count valid supports for this move
            for (int s = 0; s < num_supports; s++) {
                SupportOrder* support = &supports[s];
                
                if (!support->is_valid || support->is_cut) {
                    continue;
                }
                
                // Check if this support applies to this move
                if (support->supported_location == attempt->from_location &&
                    support->destination == attempt->to_location) {
                    
                    // Additional check: supporter's power can't support dislodging own unit
                    // Find if there's own unit at destination
                    int own_unit_at_dest = 0;
                    for (int j = 0; j < num_attempts; j++) {
                        if (attempts[j].from_location == attempt->to_location &&
                            attempts[j].unit_power == support->supporter_power) {
                            own_unit_at_dest = 1;
                            break;
                        }
                    }
                    
                    if (!own_unit_at_dest) {
                        attempt->attack_strength++;
                    } else {
                        // Can't support dislodging own unit - mark support as invalid
                        support->is_valid = 0;
                    }
                }
            }
        }
        
        // Calculate defense strength (if holding or destination of attack)
        // Units get hold support
        for (int s = 0; s < num_supports; s++) {
            SupportOrder* support = &supports[s];
            
            if (!support->is_valid || support->is_cut) {
                continue;
            }
            
            // Hold support: supported unit is at its location
            if (support->supported_location == attempt->from_location &&
                support->destination == attempt->from_location) {
                attempt->defend_strength++;
            }
        }
    }
    
    // Step 4: Detect conflicts and determine outcomes using strength calculations
    for (int i = 0; i < num_attempts; i++) {
        MoveAttempt* attacker = &attempts[i];

        if (!attacker->is_valid || attacker->to_location < 0) {
            continue;  // Invalid order or holding
        }

        int destination = attacker->to_location;
        int source = attacker->from_location;
        
        // Find all competing moves to this destination
        int max_attack_strength = 0;
        int num_with_max_strength = 0;
        int head_to_head_opponent_idx = -1;
        
        for (int j = 0; j < num_attempts; j++) {
            MoveAttempt* other = &attempts[j];

            if (!other->is_valid || other->to_location != destination) {
                continue;  // Not attacking this destination
            }

            // Track max attack strength
            if (other->attack_strength > max_attack_strength) {
                max_attack_strength = other->attack_strength;
                num_with_max_strength = 1;
            } else if (other->attack_strength == max_attack_strength) {
                num_with_max_strength++;
            }
            
            // Check for head-to-head battle
            if (other->from_location == destination && other->to_location == source) {
                head_to_head_opponent_idx = j;
            }
        }
        
        // Find defender at destination (if any)
        int defender_idx = -1;
        int defender_strength = 0;
        for (int j = 0; j < num_attempts; j++) {
            if (attempts[j].from_location == destination) {
                defender_idx = j;
                defender_strength = attempts[j].defend_strength;
                break;
            }
        }
        
        // Determine outcome based on strengths
        if (num_with_max_strength > 1) {
            // Multiple attackers with equal max strength → all bounce
            continue;  // can_move stays 0
        }
        
        // Single strongest attacker
        if (attacker->attack_strength == max_attack_strength) {
            if (defender_idx >= 0) {
                // There's a defender
                if (head_to_head_opponent_idx >= 0) {
                    // Head-to-head battle
                    MoveAttempt* opponent = &attempts[head_to_head_opponent_idx];
                    if (attacker->attack_strength > opponent->attack_strength) {
                        // We win - can move, they're dislodged
                        attacker->can_move = 1;
                        // Mark for dislodgement (tracked later)
                    } else {
                        // Equal or weaker - both bounce
                        continue;
                    }
                } else {
                    // Not head-to-head, defender holding or moving elsewhere
                    if (attacker->attack_strength > defender_strength) {
                        // Successful attack - defender dislodged
                        attacker->can_move = 1;
                        // Mark defender for dislodgement
                    } else {
                        // Attack bounced
                        continue;
                    }
                }
            } else {
                // No defender - move succeeds
                attacker->can_move = 1;
            }
        }
    }
    
    // Step 5: Handle circular movements and chains
    // Detect and resolve cycles like A→B, B→C, C→A
    for (int i = 0; i < num_attempts; i++) {
        MoveAttempt* attempt = &attempts[i];

        if (attempt->can_move || !attempt->is_valid || attempt->to_location < 0) {
            continue;  // Already resolved or not moving
        }

        // Try to find a cycle starting from this unit
        int visited[MAX_UNITS] = {0};
        int cycle_indices[MAX_UNITS];
        int cycle_len = 0;

        int current_idx = i;
        int found_cycle = 0;

        // Follow the chain of moves
        while (current_idx >= 0 && cycle_len < MAX_UNITS) {
            if (visited[current_idx]) {
                // Found a cycle - check if it starts at our original unit
                for (int c = 0; c < cycle_len; c++) {
                    if (cycle_indices[c] == current_idx) {
                        found_cycle = 1;
                        // Trim cycle to only include units in the actual cycle
                        for (int k = c; k < cycle_len; k++) {
                            cycle_indices[k - c] = cycle_indices[k];
                        }
                        cycle_len -= c;
                        break;
                    }
                }
                break;
            }

            visited[current_idx] = 1;
            cycle_indices[cycle_len++] = current_idx;

            // Find unit at destination
            int next_idx = -1;
            int dest = attempts[current_idx].to_location;
            for (int j = 0; j < num_attempts; j++) {
                if (attempts[j].from_location == dest && attempts[j].to_location >= 0) {
                    next_idx = j;
                    break;
                }
            }

            current_idx = next_idx;
        }

        // If we found a cycle, check if all units in cycle can move
        if (found_cycle && cycle_len >= 2) {
            int can_resolve_cycle = 1;

            // Check if any external unit is attacking into the cycle
            for (int c = 0; c < cycle_len; c++) {
                int cycle_dest = attempts[cycle_indices[c]].to_location;
                int cycle_strength = attempts[cycle_indices[c]].attack_strength;

                // Check for external attackers
                for (int j = 0; j < num_attempts; j++) {
                    if (attempts[j].to_location == cycle_dest && attempts[j].is_valid) {
                        // Is this attacker part of the cycle?
                        int in_cycle = 0;
                        for (int k = 0; k < cycle_len; k++) {
                            if (j == cycle_indices[k]) {
                                in_cycle = 1;
                                break;
                            }
                        }

                        if (!in_cycle && attempts[j].attack_strength >= cycle_strength) {
                            // External unit blocks the cycle
                            can_resolve_cycle = 0;
                            break;
                        }
                    }
                }

                if (!can_resolve_cycle) break;
            }

            // If cycle can resolve, mark all units in cycle as can_move
            if (can_resolve_cycle) {
                for (int c = 0; c < cycle_len; c++) {
                    attempts[cycle_indices[c]].can_move = 1;
                }
            }
        }
    }

    // Step 5b: Iteratively resolve remaining moves where destination is being vacated
    int changed = 1;
    int iterations = 0;
    while (changed && iterations < 20) {
        changed = 0;
        iterations++;

        for (int i = 0; i < num_attempts; i++) {
            MoveAttempt* attempt = &attempts[i];

            if (attempt->can_move || !attempt->is_valid || attempt->to_location < 0) {
                continue;  // Already resolved or not moving
            }

            int destination = attempt->to_location;

            // Check if destination unit is moving away successfully
            int dest_unit_moving = 0;
            for (int j = 0; j < num_attempts; j++) {
                if (attempts[j].from_location == destination &&
                    attempts[j].to_location >= 0 &&
                    attempts[j].can_move) {
                    dest_unit_moving = 1;
                    break;
                }
            }

            if (!dest_unit_moving) {
                continue;  // Destination not being vacated
            }

            // Check if we're the only/strongest attacker to this destination
            int is_strongest = 1;
            int max_str = attempt->attack_strength;
            int num_at_max = 1;

            for (int j = 0; j < num_attempts; j++) {
                if (i == j) continue;
                if (!attempts[j].is_valid || attempts[j].to_location != destination) {
                    continue;
                }

                if (attempts[j].attack_strength > max_str) {
                    is_strongest = 0;
                    break;
                } else if (attempts[j].attack_strength == max_str) {
                    num_at_max++;
                }
            }

            // Can move if strongest attacker
            if (is_strongest && num_at_max == 1) {
                attempt->can_move = 1;
                changed = 1;
            }
        }
    }
    
    // Step 6: Determine and record dislodgements
    for (int i = 0; i < num_attempts; i++) {
        MoveAttempt* attacker = &attempts[i];
        
        if (!attacker->can_move || attacker->to_location < 0) {
            continue;  // Not moving
        }
        
        // Check if we're dislodging a unit
        for (int j = 0; j < num_attempts; j++) {
            MoveAttempt* defender = &attempts[j];
            
            if (defender->from_location == attacker->to_location) {
                // There's a unit at our destination
                // It's dislodged if we have strength to dislodge it
                if (attacker->attack_strength > defender->defend_strength) {
                    // Record dislodgement
                    DislodgedUnit* dislodged = &game->dislodged[game->num_dislodged++];
                    dislodged->type = defender->unit_type;
                    dislodged->power_id = defender->unit_power;
                    dislodged->from_location = defender->from_location;
                    dislodged->dislodged_by_location = attacker->from_location;
                    dislodged->num_possible_retreats = 0;  // Calculate later in retreat phase
                    
                    // Mark defender as not able to stay
                    // We'll remove the unit when applying moves
                }
                break;
            }
        }
    }
    
    // Step 6b: Check convoy disruption and handle support cutting for convoyed moves
    // For convoyed moves:
    // - If convoy is disrupted (fleet dislodged), move fails and support is NOT cut
    // - If convoy is valid, check if move succeeds and cut support if it attacks
    for (int i = 0; i < num_attempts; i++) {
        MoveAttempt* attempt = &attempts[i];

        if (!attempt->is_convoyed || attempt->to_location < 0) {
            continue;  // Not a convoyed move
        }

        // This is a convoyed move - check if convoy is still valid after dislodgements
        // Build list of non-dislodged convoying fleets
        int convoying_fleets[MAX_LOCATIONS];
        int num_fleets = 0;
        int dislodged_fleet_locs[MAX_LOCATIONS];
        int num_dislodged_fleets = 0;

        // Get all fleets offering to convoy this move
        for (int p = 0; p < MAX_POWERS; p++) {
            Power* power = &game->powers[p];
            for (int o = 0; o < power->num_orders; o++) {
                Order* order = &power->orders[o];

                if (order->type == ORDER_CONVOY &&
                    order->target_unit_location == attempt->from_location &&
                    order->dest_location == attempt->to_location) {

                    int fleet_loc = order->unit_location;

                    // Check if this fleet is dislodged
                    int is_dislodged = 0;
                    for (int d = 0; d < game->num_dislodged; d++) {
                        if (game->dislodged[d].from_location == fleet_loc) {
                            is_dislodged = 1;
                            dislodged_fleet_locs[num_dislodged_fleets++] = fleet_loc;
                            break;
                        }
                    }

                    // Only include non-dislodged fleets in water
                    if (!is_dislodged && can_fleet_convoy(game->map, fleet_loc)) {
                        convoying_fleets[num_fleets++] = fleet_loc;
                    }
                }
            }
        }

        // Check if convoy path still exists with remaining fleets
        int convoy_valid = (num_fleets > 0 &&
                           find_convoy_path(game->map, attempt->from_location,
                                          attempt->to_location,
                                          convoying_fleets, num_fleets));

        if (!convoy_valid && num_dislodged_fleets > 0) {
            // Convoy appears disrupted - check for paradox
            // Paradox: destination unit supports a dislodged convoying fleet
            int is_paradox = 0;
            for (int df = 0; df < num_dislodged_fleets; df++) {
                if (detect_paradox(game, attempt->to_location, dislodged_fleet_locs[df])) {
                    is_paradox = 1;
                    break;
                }
            }

            if (!is_paradox) {
                // Not a paradox - convoy is disrupted, move fails and does NOT cut support
                attempt->convoy_disrupted = 1;
                attempt->can_move = 0;
            }
            // If paradox, don't mark as disrupted - convoy succeeds despite dislodged fleet
        } else if (!convoy_valid) {
            // No valid path and no dislodged fleets (invalid from start)
            attempt->convoy_disrupted = 1;
            attempt->can_move = 0;
        }

        // If convoy is valid (or paradox), check if it cuts any supports
        if (!attempt->convoy_disrupted) {
            // Convoy is valid - check if it cuts any supports
            // (convoyed moves cut support just like normal moves, if convoy succeeds)
            for (int s = 0; s < num_supports; s++) {
                SupportOrder* support = &supports[s];

                if (!support->is_valid || support->is_cut) {
                    continue;  // Already invalid or cut
                }

                // Support is cut if attacker is moving to supporter's location
                if (attempt->to_location == support->supporter_location) {
                    // Exception: attack from the unit being supported doesn't cut support
                    if (attempt->from_location != support->supported_location) {
                        // Exception: own units don't cut support
                        if (attempt->unit_power != support->supporter_power) {
                            support->is_cut = 1;
                        }
                    }
                }
            }
        }
    }

    // Step 6c: Resolve valid convoyed moves
    // Now that we know which convoys are disrupted, resolve the valid ones
    for (int i = 0; i < num_attempts; i++) {
        MoveAttempt* attempt = &attempts[i];

        if (!attempt->is_convoyed || attempt->to_location < 0) {
            continue;  // Not a convoyed move
        }

        if (attempt->convoy_disrupted) {
            continue;  // Convoy failed, already marked
        }

        // Convoy is valid - check if move can succeed
        int destination = attempt->to_location;

        // Check if destination is vacant or being vacated
        int dest_vacant = 1;
        int dest_unit_moving = 0;

        for (int j = 0; j < num_attempts; j++) {
            if (attempts[j].from_location == destination) {
                dest_vacant = 0;
                if (attempts[j].to_location >= 0 && attempts[j].can_move) {
                    dest_unit_moving = 1;
                }
                break;
            }
        }

        if (!dest_vacant && !dest_unit_moving) {
            continue;  // Destination occupied and not moving - convoy bounces
        }

        // Check if we're the strongest attacker (among other convoyed moves and already-resolved moves)
        int is_strongest = 1;
        int max_str = attempt->attack_strength;

        for (int j = 0; j < num_attempts; j++) {
            if (i == j) continue;
            if (!attempts[j].is_valid || attempts[j].to_location != destination) {
                continue;
            }

            // Check other convoyed moves
            if (attempts[j].is_convoyed && !attempts[j].convoy_disrupted) {
                if (attempts[j].attack_strength >= max_str) {
                    is_strongest = 0;
                    break;
                }
            }
            // Check already-resolved non-convoyed moves
            else if (!attempts[j].is_convoyed && attempts[j].can_move) {
                // There's already a stronger move succeeding
                is_strongest = 0;
                break;
            }
        }

        if (is_strongest) {
            attempt->can_move = 1;
        }
    }

    // Step 7: Apply successful moves and remove dislodged units
    for (int i = 0; i < num_attempts; i++) {
        MoveAttempt* attempt = &attempts[i];

        if (!attempt->can_move || attempt->to_location < 0) {
            continue;  // Stay in place
        }
        
        // Check if this unit was dislodged
        int was_dislodged = 0;
        for (int d = 0; d < game->num_dislodged; d++) {
            if (game->dislodged[d].power_id == attempt->unit_power &&
                game->dislodged[d].from_location == attempt->from_location) {
                was_dislodged = 1;
                break;
            }
        }
        
        if (was_dislodged) {
            // Don't apply move for dislodged unit - it will be removed/retreated
            continue;
        }
        
        // Move the unit
        Power* power = &game->powers[attempt->unit_power];
        Unit* unit = &power->units[attempt->unit_idx];
        unit->location = attempt->to_location;
    }
    
    // Step 8: Remove dislodged units from their powers
    for (int d = 0; d < game->num_dislodged; d++) {
        DislodgedUnit* dislodged = &game->dislodged[d];
        Power* power = &game->powers[dislodged->power_id];
        
        // Remove unit from power's unit list
        for (int u = 0; u < power->num_units; u++) {
            if (power->units[u].location == dislodged->from_location &&
                power->units[u].type == dislodged->type) {
                // Remove by shifting remaining units
                for (int k = u + 1; k < power->num_units; k++) {
                    power->units[k - 1] = power->units[k];
                }
                power->num_units--;
                break;
            }
        }
    }
}

void resolve_retreat_phase(GameState* game) {
    // Retreat phase implementation
    // For now, all dislodged units are disbanded (no retreat orders processed yet)
    
    // TODO: Implement full retreat resolution:
    // 1. Calculate valid retreat destinations for each dislodged unit
    // 2. Process retreat orders
    // 3. Check for conflicts (two units retreating to same space → both disband)
    // 4. Move retreating units or disband if no valid retreat/order
    
    // For now: just clear dislodged units (they're already removed from power.units)
    game->num_dislodged = 0;
}

void resolve_adjustment_phase(GameState* game) {
    // Simple implementation for Iteration 3: No builds/disbands yet
    // Just calculate adjustment counts and welfare points

    // Calculate adjustment count for each power
    for (int p = 0; p < MAX_POWERS; p++) {
        Power* power = &game->powers[p];
        power->adjustment = power->num_centers - power->num_units;
    }

    // Welfare Diplomacy: Calculate welfare points after adjustments
    if (game->welfare_mode) {
        calculate_welfare_points(game);
    }

    // TODO: In future iterations, implement full adjustment resolution:
    // 1. Process build orders
    // 2. Process disband orders
    // 3. In welfare mode: allow voluntary disbands
}

void advance_phase(GameState* game) {
    // Advance to next phase
    switch (game->phase) {
        case PHASE_SPRING_MOVEMENT:
            // Check if any units were dislodged
            if (game->num_dislodged > 0) {
                game->phase = PHASE_SPRING_RETREAT;
            } else {
                game->phase = PHASE_FALL_MOVEMENT;
            }
            break;
        case PHASE_SPRING_RETREAT:
            game->phase = PHASE_FALL_MOVEMENT;
            break;
        case PHASE_FALL_MOVEMENT:
            // Check if any units were dislodged
            if (game->num_dislodged > 0) {
                game->phase = PHASE_FALL_RETREAT;
            } else {
                game->phase = PHASE_WINTER_ADJUSTMENT;
            }
            break;
        case PHASE_FALL_RETREAT:
            game->phase = PHASE_WINTER_ADJUSTMENT;
            break;
        case PHASE_WINTER_ADJUSTMENT:
            // Move to next year
            game->year++;
            game->phase = PHASE_SPRING_MOVEMENT;

            // Check if game should end
            // max_years means "play for N years starting from 1901"
            // e.g., max_years=10 means play years 1901-1910, end when year becomes 1911
            if (game->year > 1900 + game->max_years) {
                game->phase = PHASE_COMPLETED;
                game->is_game_over = 1;
            }
            break;
        case PHASE_COMPLETED:
            // Game is over, no more phases
            break;
    }

    // Clear orders for new phase
    for (int p = 0; p < MAX_POWERS; p++) {
        game->powers[p].num_orders = 0;
    }
}

// ============================================================================
// Welfare Diplomacy Specific
// ============================================================================

void calculate_welfare_points(GameState* game) {
    for (int p = 0; p < MAX_POWERS; p++) {
        Power* power = &game->powers[p];

        // Welfare points = centers - units (only if positive)
        int welfare_gain = power->num_centers - power->num_units;
        if (welfare_gain > 0) {
            power->welfare_points += welfare_gain;
        }
    }
}

// ============================================================================
// PufferLib Integration Functions
// ============================================================================

void c_init(Env* env) {
    // Allocate and initialize game state
    env->game = (GameState*)calloc(1, sizeof(GameState));
    if (!env->game) {
        fprintf(stderr, "Failed to allocate GameState\n");
        return;
    }

    // Allocate and initialize map
    Map* map = (Map*)calloc(1, sizeof(Map));
    if (!map) {
        fprintf(stderr, "Failed to allocate Map\n");
        free(env->game);
        return;
    }

    init_standard_map(map);
    init_game(env->game, map, 1, 10);  // Default: Welfare mode, 10 year limit

    // Initialize log
    memset(&env->log, 0, sizeof(Log));

    // Initialize reward bookkeeping
    for (int i = 0; i < MAX_POWERS; i++) {
        env->last_welfare[i] = 0;
    }
}

void c_reset(Env* env) {
    if (env->game) {
        reset_game(env->game);
    }
    // Reset rewards and terminals
    if (env->rewards) {
        for (int i = 0; i < MAX_POWERS; i++) env->rewards[i] = 0.0f;
    }
    if (env->terminals) {
        for (int i = 0; i < MAX_POWERS; i++) env->terminals[i] = 0;
    }
    // Initialize last welfare snapshot
    for (int i = 0; i < MAX_POWERS; i++) env->last_welfare[i] = env->game->powers[i].welfare_points;

    // Encode observations once on reset
    if (env->observations) {
        float* obs = (float*)env->observations;
        int stride = 175;
        for (int agent = 0; agent < MAX_POWERS; agent++) {
            float* base = obs + agent * stride;
            // Board ownership
            for (int i = 0; i < env->game->map->num_locations; i++) {
                base[i] = (float)env->game->map->locations[i].owner_power;
            }
            // Unit type at each location (0 none, 1 army, 2 fleet)
            int offset = 75;
            for (int i = 0; i < env->game->map->num_locations; i++) {
                int owner = get_unit_at_location(env->game, i);
                if (owner >= 0) {
                    UnitType t = UNIT_NONE;
                    for (int u = 0; u < env->game->powers[owner].num_units; u++) {
                        if (env->game->powers[owner].units[u].location == i) {
                            t = env->game->powers[owner].units[u].type;
                            break;
                        }
                    }
                    base[offset + i] = (float)t;
                } else {
                    base[offset + i] = 0.0f;
                }
            }
            // Centers per power
            offset += 75;
            for (int p = 0; p < MAX_POWERS; p++) {
                base[offset + p] = (float)env->game->powers[p].num_centers;
            }
            // Units per power
            offset += 7;
            for (int p = 0; p < MAX_POWERS; p++) {
                base[offset + p] = (float)env->game->powers[p].num_units;
            }
            // Welfare per power
            offset += 7;
            for (int p = 0; p < MAX_POWERS; p++) {
                base[offset + p] = (float)env->game->powers[p].welfare_points;
            }
            // Phase and year
            offset += 7;
            base[offset + 0] = (float)env->game->phase;
            base[offset + 1] = (float)env->game->year;
        }
    }
}

void c_step(Env* env) {
    if (!env->game || env->game->is_game_over) {
        return;
    }

    GameState* game = env->game;

    // For Iteration 3: Simple implementation with HOLD orders only
    // Process current phase (all units hold)
    process_orders(game);

    // Advance to next phase
    advance_phase(game);

    // TODO: Process actions from env->actions (parse orders from action space)
    // Update observations (encode without resetting)
    if (env->observations) {
        float* obs = (float*)env->observations;
        int stride = 175;
        for (int agent = 0; agent < MAX_POWERS; agent++) {
            float* base = obs + agent * stride;
            for (int i = 0; i < env->game->map->num_locations; i++) {
                base[i] = (float)env->game->map->locations[i].owner_power;
            }
            int offset = 75;
            for (int i = 0; i < env->game->map->num_locations; i++) {
                int owner = get_unit_at_location(env->game, i);
                if (owner >= 0) {
                    UnitType t = UNIT_NONE;
                    for (int u = 0; u < env->game->powers[owner].num_units; u++) {
                        if (env->game->powers[owner].units[u].location == i) {
                            t = env->game->powers[owner].units[u].type;
                            break;
                        }
                    }
                    base[offset + i] = (float)t;
                } else {
                    base[offset + i] = 0.0f;
                }
            }
            offset += 75;
            for (int p = 0; p < MAX_POWERS; p++) {
                base[offset + p] = (float)env->game->powers[p].num_centers;
            }
            offset += 7;
            for (int p = 0; p < MAX_POWERS; p++) {
                base[offset + p] = (float)env->game->powers[p].num_units;
            }
            offset += 7;
            for (int p = 0; p < MAX_POWERS; p++) {
                base[offset + p] = (float)env->game->powers[p].welfare_points;
            }
            offset += 7;
            base[offset + 0] = (float)env->game->phase;
            base[offset + 1] = (float)env->game->year;
        }
    }

    // Calculate rewards at end of adjustment phase based on welfare deltas
    if (game->phase == PHASE_WINTER_ADJUSTMENT) {
        for (int i = 0; i < MAX_POWERS; i++) {
            int current = game->powers[i].welfare_points;
            int delta = current - env->last_welfare[i];
            if (env->rewards) env->rewards[i] = (float)delta;
            env->last_welfare[i] = current;
        }
    } else {
        if (env->rewards) {
            for (int i = 0; i < MAX_POWERS; i++) env->rewards[i] = 0.0f;
        }
    }

    // Set terminals if game over
    if (game->is_game_over) {
        for (int i = 0; i < MAX_POWERS; i++) {
            env->terminals[i] = 1;
        }
    }
}

void c_render(Env* env) {
    // TODO: Implement basic text rendering of game state
    if (!env->game) {
        return;
    }

    GameState* game = env->game;
    printf("Year: %d, Phase: %s\n", game->year, phase_to_string(game->phase));

    for (int p = 0; p < MAX_POWERS; p++) {
        Power* power = &game->powers[p];
        printf("%s: Centers=%d, Units=%d, Welfare=%d\n",
               power->name, power->num_centers, power->num_units, power->welfare_points);
    }
}

void c_close(Env* env) {
    if (env->game) {
        if (env->game->map) {
            free_map(env->game->map);
            free(env->game->map);
        }
        free_game(env->game);
        free(env->game);
        env->game = NULL;
    }
}

void c_configure(Env* env, int welfare_mode, int max_years) {
    if (!env || !env->game) {
        return;
    }
    // Re-init game in-place preserving map pointer
    Map* map = env->game->map;
    init_game(env->game, map, welfare_mode, max_years);
}
