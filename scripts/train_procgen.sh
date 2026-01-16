#!/bin/bash

environments=(
    "bigfish"
    "bossfight"
    "caveflyer"
    "chaser"
    "climber"
    "coinrun"
    "dodgeball"
    "fruitbot"
    "heist"
    "jumper"
    "leaper"
    "maze"
    "miner"
    "ninja"
    "plunder"
    "starpilot"
)

for env in "${environments[@]}"; do
    echo "Training on environment: $env"
    python demo.py --mode train --vec multiprocessing --track --env "$env"
done
