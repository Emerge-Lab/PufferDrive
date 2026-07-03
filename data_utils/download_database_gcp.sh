#!/bin/bash

# This script downloads the first 1000 map files from the WOMD dataset in parallel.

# Create the destination directory if it doesn't exist
mkdir -p ressources/drive/binaries/training

# Use a wildcard to specify the first 1000 map files (000-999)
GS_URI_PATTERN="gs://valeo-cp2386-datasets/pufferdrive/v1.0/womd/training/map_[0-9][0-9][0-9].bin"

# Use gsutil with the -m option to download files in parallel
# The -m option performs a multi-threaded/multi-processing copy.
echo "Downloading map files from gs://valeo-cp2386-datasets/pufferdrive/v1.0/womd/training/ ..."
gsutil -m cp $GS_URI_PATTERN resources/drive/binaries/training/
echo "Download complete."
