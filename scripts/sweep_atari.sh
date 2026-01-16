#!/bin/bash

environments=(
    "pong"
    "breakout"
    "beam_rider"
    "enduro"
    "qbert"
    "space_invaders"
    "seaquest"
)

for env in "${environments[@]}"; do
    echo "Training: $env"
    python demo.py --mode sweep-carbs --vec multiprocessing --env "$env"
done
