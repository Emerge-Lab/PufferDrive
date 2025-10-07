"""
Script to parse the standard.map file and generate C code for init_standard_map()
"""

import re
from collections import defaultdict

# Read the standard.map file
map_file = "/scratch/mmk9418/projects/welfare-diplomacy/diplomacy/maps/standard.map"

locations = []
loc_types = {}
adjacencies = {}
aliases = {}
home_centers = {}
starting_units = {}

with open(map_file, 'r') as f:
    content = f.read()

# Parse initial power setup
power_pattern = r'(\w+)\s+\((\w+)\)\s+([\w\s]+)'
for match in re.finditer(power_pattern, content[:500]):
    power = match.group(1)
    centers = match.group(3).split()
    home_centers[power] = centers

# Parse location aliases (to get canonical 3-letter codes)
alias_section = re.findall(r'^(.+?)\s+=\s+(.+)$', content, re.MULTILINE)
canonical_names = {}
for full_name, alias_list in alias_section[:82]:  # Only location aliases
    aliases_list = alias_list.strip().split()
    if aliases_list:
        canonical = aliases_list[0].upper().split('/')[0].split('(')[0][:3]
        canonical_names[full_name.strip()] = canonical

# Parse location types and adjacencies
adjacency_pattern = r'(\w+)\s+(\w+(?:/\w+)?)\s+ABUTS\s+(.+)'
for match in re.finditer(adjacency_pattern, content):
    loc_type = match.group(1)
    location = match.group(2).upper()
    abuts_str = match.group(3)

    # Get canonical 3-letter code
    if '/' in location or '(' in location:
        loc = location.split('/')[0].split('(')[0]
    else:
        loc = location

    if loc not in loc_types:
        loc_types[loc] = loc_type
        locations.append(loc)

    # Parse adjacencies
    adj_list = []
    for adj in abuts_str.split():
        adj_clean = adj.upper().split('/')[0].split('(')[0]
        adj_list.append(adj_clean)

    adjacencies[loc] = adj_list

# Sort locations alphabetically for consistency
locations.sort()

# Map location types to enum
type_map = {
    'LAND': 'LOC_LAND',
    'COAST': 'LOC_COAST',
    'WATER': 'LOC_WATER',
    'PORT': 'LOC_PORT'
}

# Generate C code
print(f"// Total locations: {len(locations)}")
print()
print("// Location initialization")
print(f"map->num_locations = {len(locations)};")
print()

for i, loc in enumerate(locations):
    loc_type_enum = type_map.get(loc_types.get(loc, 'COAST'), 'LOC_COAST')
    print(f"// {i}: {loc}")
    print(f'strcpy(map->locations[{i}].name, "{loc}");')
    print(f'map->locations[{i}].type = {loc_type_enum};')

    # Check if it's a supply center
    is_sc = any(loc in centers for centers in home_centers.values())
    if loc == 'BEL' or loc == 'BUL' or loc == 'DEN' or loc == 'GRE' or loc == 'HOL' or \
       loc == 'NWY' or loc == 'POR' or loc == 'RUM' or loc == 'SER' or loc == 'SPA' or \
       loc == 'SWE' or loc == 'TUN':
        is_sc = True

    print(f'map->locations[{i}].has_supply_center = {1 if is_sc else 0};')
    print(f'map->locations[{i}].owner_power = -1;  // Neutral initially')

    # Adjacencies
    if loc in adjacencies:
        adj_indices = []
        for adj in adjacencies[loc]:
            if adj in locations:
                adj_idx = locations.index(adj)
                adj_indices.append(adj_idx)

        print(f'map->locations[{i}].num_adjacent = {len(adj_indices)};')
        for j, adj_idx in enumerate(adj_indices):
            print(f'map->locations[{i}].adjacencies[{j}] = {adj_idx};')
    else:
        print(f'map->locations[{i}].num_adjacent = 0;')

    print()

# Print home center assignments
print("\n// Home center assignments")
power_list = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
for p_idx, power in enumerate(power_list):
    if power in home_centers:
        print(f"// {power}")
        centers = home_centers[power]
        for c_idx, center in enumerate(centers):
            if center in locations:
                loc_idx = locations.index(center)
                print(f'map->home_centers[{p_idx}][{c_idx}] = {loc_idx};  // {center}')
                print(f'map->locations[{loc_idx}].is_home_center = {p_idx};')
        print(f'map->num_homes[{p_idx}] = {len(centers)};')
        print()

print("\n// Location name to index mapping (for reference):")
for i, loc in enumerate(locations):
    print(f"// {i:2d}: {loc}")
