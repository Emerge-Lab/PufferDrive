#!/bin/bash
# Render driving behaviour evaluation videos from a weights .bin file.
#
# Usage: bash scripts/render_behaviours.sh <weights.bin> [output_dir]
#
# Requires: visualize binary built locally (bash scripts/build_ocean.sh visualize fast)

set -e

BIN_PATH="$1"
OUTPUT_DIR="${2:-/tmp/behaviour_videos}"
BASE_INI="pufferlib/config/ocean/drive.ini"

if [ -z "$BIN_PATH" ]; then
    echo "Usage: bash scripts/render_behaviours.sh <weights.bin> [output_dir]"
    exit 1
fi

if [ ! -f "$BIN_PATH" ]; then
    echo "Weights file not found: $BIN_PATH"
    exit 1
fi

if [ ! -f "./visualize" ]; then
    echo "Building visualize binary..."
    bash scripts/build_ocean.sh visualize fast
fi

mkdir -p "$OUTPUT_DIR"

# Generate a behavioural eval ini: control_sdc_only, create_all_valid, goal_respawn
BEHAV_INI=$(mktemp /tmp/behav_render_XXXXXX.ini)
python3 -c "
import configparser
c = configparser.ConfigParser()
c.read('$BASE_INI')
c.set('env', 'init_mode', 'create_all_valid')
c.set('env', 'control_mode', 'control_sdc_only')
c.set('env', 'goal_behavior', '0')  # GOAL_RESPAWN
c.set('env', 'episode_length', '1000')
with open('$BEHAV_INI', 'w') as f:
    c.write(f)
print('Generated behavioural eval ini: $BEHAV_INI')
"

SCENARIOS=(
    "lead_vehicle_interaction"
    "lane_change"
    "dense_traffic"
    "obstacles"
    "vru_interaction"
)

total=0
for scenario in "${SCENARIOS[@]}"; do
    MAP_DIR="pufferlib/resources/drive/binaries/$scenario"
    if [ ! -d "$MAP_DIR" ]; then
        MAP_DIR="resources/drive/binaries/$scenario"
    fi
    if [ ! -d "$MAP_DIR" ]; then
        echo "SKIP $scenario: map dir not found"
        continue
    fi

    echo "=== Rendering $scenario ==="
    for map_file in "$MAP_DIR"/map_*.bin; do
        map_name=$(basename "$map_file" .bin)
        topdown="$OUTPUT_DIR/${scenario}_${map_name}_topdown.mp4"
        agent="$OUTPUT_DIR/${scenario}_${map_name}_agent.mp4"

        echo "  $map_name..."
        ./visualize \
            --config "$BEHAV_INI" \
            --map-name "$map_file" \
            --policy-name "$BIN_PATH" \
            --output-topdown "$topdown" \
            --output-agent "$agent" \
            --view both \
            2>/dev/null && total=$((total + 2)) || echo "  FAILED: $map_name"
    done
done

rm -f "$BEHAV_INI"
echo ""
echo "Videos saved to $OUTPUT_DIR"
echo "Total: $total videos"
