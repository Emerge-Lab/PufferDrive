#!/bin/bash
# Rebuild the C extension when its sources changed, only then: a rebuild crashes the running jobs that use it.
REPO_DIR=$1
# .so files are tagged per python version; track builds per interpreter
PYTHON_TAG=$(python -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
BUILD_HASH_FILE=${REPO_DIR}/experiments/logs/build_source_hash_${PYTHON_TAG}
# jobs of one interpreter that start together (e.g. both nuPlan evals) would otherwise build into the same files at once
mkdir -p "$(dirname "${BUILD_HASH_FILE}")"
exec 9> "${BUILD_HASH_FILE}.lock"
flock 9

SOURCE_NAME_PATTERNS=(-name '*.c' -o -name '*.h')
# NO_TRAIN builds skip the torch extension, so only full builds track its C++/CUDA sources
if [ "${NO_TRAIN:-0}" != "1" ]; then
    SOURCE_NAME_PATTERNS+=(-o -name '*.cpp' -o -name '*.cu')
fi
SOURCE_HASH=$({ find ${REPO_DIR}/opengigaflow \( "${SOURCE_NAME_PATTERNS[@]}" \) -type f; echo ${REPO_DIR}/setup.py; } | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)

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
