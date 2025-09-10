#!/bin/bash

# Prepare Jsons
python /puffertank/pufferlib/pufferlib/ocean/drive/drive.py

# Run training
python -m pufferlib.pufferl train puffer_drive
