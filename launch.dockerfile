ARG BUILD_MODE="fast"

# Copier uniquement les fichiers nécessaires pour l'installation de pufferdrive
COPY . .

RUN chmod +x /pufferdrive/automation/run_training.sh

RUN python setup.py build_ext --inplace --force

RUN /bin/bash /pufferdrive/scripts/build_ocean.sh drive ${BUILD_MODE}

# Set the entrypoint for the training job.
ENTRYPOINT ["/pufferdrive/automation/run_training.sh"]
