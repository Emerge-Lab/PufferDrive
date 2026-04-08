#!/bin/bash
# Render behavior eval videos locally using the visualize binary.
#
# Usage:
#   bash scripts/render_behavior_evals.sh <policy.bin> [output_dir] [view_mode]
#
# Example:
#   bash scripts/render_behavior_evals.sh resources/drive/seed99_003815.bin behavior_eval_videos both

set -euo pipefail

POLICY_NAME="${1:?Usage: $0 <policy.bin> [output_dir] [view_mode]}"
OUTPUT_DIR="${2:-behavior_eval_videos}"
VIEW_MODE="${3:-both}"

BEHAVIOR_INI="pufferlib/config/ocean/driving_behaviours_eval.ini"
BASE_INI="pufferlib/config/ocean/drive.ini"

if [ ! -f "$POLICY_NAME" ]; then
    echo "Error: policy file not found: $POLICY_NAME"
    exit 1
fi
if [ ! -f ./visualize ]; then
    echo "Error: visualize binary not found. Run: bash scripts/build_ocean.sh visualize fast"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Write reward bound sed commands to a file (avoids shell word-splitting issues)
REWARD_SED_FILE="/tmp/behavior_eval_rewards_$$.sed"
.venv/bin/python3 -c "
import configparser, sys
cp = configparser.ConfigParser(comment_prefixes=(';','#'), inline_comment_prefixes=(';','#'))
cp.read('$BEHAVIOR_INI')
if not cp.has_section('eval_driving_rewards'):
    sys.exit(0)
for key, val in cp.items('eval_driving_rewards'):
    print(f's/reward_bound_{key}_min = .*/reward_bound_{key}_min = {val}/')
    print(f's/reward_bound_{key}_max = .*/reward_bound_{key}_max = {val}/')
" > "$REWARD_SED_FILE"

# Build a temp INI from drive.ini with control_sdc_only + create_all_valid + pinned rewards
make_render_ini() {
    local map_dir="$1"
    local tmp_ini="/tmp/behavior_eval_render_$$.ini"

    sed -e 's/control_mode = .*/control_mode = "control_sdc_only"/' \
        -e 's/init_mode = .*/init_mode = "create_all_valid"/' \
        -e "s|map_dir = .*|map_dir = \"${map_dir}\"|" \
        "$BASE_INI" | sed -f "$REWARD_SED_FILE" > "$tmp_ini"

    echo "$tmp_ini"
}

# Extract class names (skip eval_driving_rewards)
CLASSES=$(grep -E '^\[eval_' "$BEHAVIOR_INI" | grep -v 'eval_driving_rewards' | sed 's/\[eval_//;s/\]//')

for CLASS in $CLASSES; do
    MAP_DIR=$(grep -A5 "^\[eval_${CLASS}\]" "$BEHAVIOR_INI" | grep 'map_dir' | sed 's/.*= *//;s/"//g' | tr -d ' ')

    if [ ! -d "$MAP_DIR" ]; then
        echo "Skipping $CLASS: $MAP_DIR not found"
        continue
    fi

    MAPS=$(find "$MAP_DIR" -name '*.bin' | sort)
    NUM_MAPS=$(echo "$MAPS" | wc -l | tr -d ' ')

    echo ""
    echo "=== $CLASS: $NUM_MAPS maps in $MAP_DIR ==="

    RENDER_INI=$(make_render_ini "$MAP_DIR")

    i=0
    for MAP in $MAPS; do
        i=$((i + 1))
        MAP_BASE=$(basename "$MAP" .bin)

        OUT_TD="$OUTPUT_DIR/${CLASS}_${MAP_BASE}_topdown.mp4"
        OUT_AG="$OUTPUT_DIR/${CLASS}_${MAP_BASE}_agent.mp4"

        printf "  [%d/%d] %s... " "$i" "$NUM_MAPS" "$(basename "$MAP")"

        if ./visualize \
            --config "$RENDER_INI" \
            --map-name "$MAP" \
            --policy-name "$POLICY_NAME" \
            --view "$VIEW_MODE" \
            --output-topdown "$OUT_TD" \
            --output-agent "$OUT_AG" \
            > /dev/null 2>&1; then
            echo "OK"
        else
            echo "FAILED"
        fi
    done

    rm -f "$RENDER_INI"
done

rm -f "$REWARD_SED_FILE"

echo ""
echo "Done. Videos in $OUTPUT_DIR/"
