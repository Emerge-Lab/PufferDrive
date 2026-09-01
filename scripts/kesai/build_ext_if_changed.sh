#!/bin/bash
# Rebuild env before training if the source has changed.
# Minimize rebuilds as they crash running jobs that are using the env.
REPO_DIR=$1
# .so files are tagged per python version; track builds per interpreter
PYTHON_TAG=$(python -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
BUILD_HASH_FILE=${REPO_DIR}/experiments/logs/build_source_hash_${PYTHON_TAG}

SOURCE_HASH=$({ find ${REPO_DIR}/pufferlib \( -name '*.c' -o -name '*.h' \) -type f; echo ${REPO_DIR}/setup.py; } | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)

if [ -f ${BUILD_HASH_FILE} ] && [ "$(cat ${BUILD_HASH_FILE})" = "${SOURCE_HASH}" ]; then
    echo "C sources unchanged (${SOURCE_HASH:0:12}); skipping rebuild."
    exit 0
fi

echo "C sources changed; rebuilding extension."
cd ${REPO_DIR} || exit 1
python setup.py build_ext --inplace --force
BUILD_STATUS=$?
if [ ${BUILD_STATUS} -eq 0 ]; then
    echo ${SOURCE_HASH} > ${BUILD_HASH_FILE}
fi
exit ${BUILD_STATUS}
