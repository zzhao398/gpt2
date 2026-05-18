#!/bin/bash
# Usage: ./test_req.sh <req_num>
# Example: ./test_req.sh 0

REQ=$1
KERNELS_DIR="kernels"
REQ_DIR="kernels_req_${REQ}"

if [ ! -d "$REQ_DIR" ]; then
    echo "Directory $REQ_DIR not found"
    exit 1
fi

echo "=== Testing req_${REQ} ==="

# backup and replace kernels
for f in ${REQ_DIR}/*.cuh; do
    fname=$(basename $f)
    if [ -f "${KERNELS_DIR}/${fname}" ]; then
        cp ${KERNELS_DIR}/${fname} ${KERNELS_DIR}/${fname}.bak
    fi
    cp $f ${KERNELS_DIR}/${fname}
    echo "Replaced kernels/${fname}"
done

# build and test
make test_gpt2_kernels 2>&1
echo "---"
srun ./test_gpt2_kernels

# restore originals
for f in ${REQ_DIR}/*.cuh; do
    fname=$(basename $f)
    if [ -f "${KERNELS_DIR}/${fname}.bak" ]; then
        mv ${KERNELS_DIR}/${fname}.bak ${KERNELS_DIR}/${fname}
        echo "Restored kernels/${fname}"
    fi
done
