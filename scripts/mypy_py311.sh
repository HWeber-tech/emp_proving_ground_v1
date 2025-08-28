#!/usr/bin/env bash
set -euo pipefail

# Timestamp for artifacts (UTC, ISO-like, filesystem-safe)
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
ROOT="$(pwd)"
SNAP_DIR="${ROOT}/mypy_snapshots"
mkdir -p "${SNAP_DIR}"

IMG_TAG="mypy-py311:latest"
DOCKERFILE="docker/mypy311/Dockerfile"

SNAPSHOT_FILE="${SNAP_DIR}/mypy_snapshot_py311_${TS}.txt"
SUMMARY_FILE="${SNAP_DIR}/mypy_summary_py311_${TS}.txt"
RANKED_FILE="${SNAP_DIR}/mypy_ranked_offenders_py311_${TS}.csv"
ENV_FILE="${SNAP_DIR}/env_py311_${TS}.txt"

echo "[mypy-py311] TS=${TS}" | tee "${SNAPSHOT_FILE}"
echo "[mypy-py311] Building image ${IMG_TAG} from ${DOCKERFILE} ..." | tee -a "${SNAPSHOT_FILE}"

# Build image, capture failures without aborting the script
set +e
BUILD_OUT="$(docker build -f "${DOCKERFILE}" -t "${IMG_TAG}" . 2>&1)"
BUILD_RC=$?
set -e

echo "${BUILD_OUT}" | tee -a "${SNAPSHOT_FILE}"
if [ ${BUILD_RC} -ne 0 ]; then
  echo "[mypy-py311] Docker build failed with exit code ${BUILD_RC}" | tee -a "${SNAPSHOT_FILE}"
  echo "mypy (py311) could not run: docker build failed (exit ${BUILD_RC}); see snapshot ${SNAPSHOT_FILE}" > "${SUMMARY_FILE}"
  # Still attempt to capture environment best-effort (will likely fail)
  echo "[mypy-py311] Skipping environment capture due to build failure" | tee -a "${SNAPSHOT_FILE}"
  exit 2
fi

echo "[mypy-py311] Running mypy in container ..." | tee -a "${SNAPSHOT_FILE}"
# Run docker, capture stdout/stderr and exit code without aborting
set +e
RUN_OUT="$(docker run --rm -t \
  -v "${ROOT}:/workspace" \
  -w /workspace \
  -e MYPY_CACHE_DIR=/tmp/mypy_cache \
  "${IMG_TAG}" \
  sh -lc 'set -eu; mkdir -p "${MYPY_CACHE_DIR}"; mypy --no-incremental --show-error-codes --no-color-output 2>&1')"
RUN_RC=$?
set -e

echo "${RUN_OUT}" | tee -a "${SNAPSHOT_FILE}"

# Try to extract totals line from RUN_OUT
TOTALS_LINE="$(printf "%s\n" "${RUN_OUT}" | grep -E 'Found [0-9]+ errors? in [0-9]+ files? \(checked [0-9]+ (source )?files?\)' | tail -n 1 || true)"

# Always attempt env capture (after image built)
echo "[mypy-py311] Capturing environment ..." | tee -a "${SNAPSHOT_FILE}"
set +e
ENV_OUT="$(docker run --rm \
  -v "${ROOT}:/workspace" \
  -w /workspace \
  "${IMG_TAG}" \
  sh -lc 'python --version; mypy --version; echo "--- pip freeze ---"; pip freeze' 2>&1)"
ENV_RC=$?
set -e
printf "%s\n" "${ENV_OUT}" > "${ENV_FILE}"

# Write summary based on availability of totals or failure modes
if [ -n "${TOTALS_LINE}" ]; then
  printf "%s\n" "${TOTALS_LINE}" > "${SUMMARY_FILE}"
else
  if [ ${RUN_RC} -ne 0 ]; then
    echo "mypy (py311) could not run: docker run failed (exit ${RUN_RC}); see snapshot ${SNAPSHOT_FILE}" > "${SUMMARY_FILE}"
  else
    echo "mypy (py311) completed; no summary totals detected — check snapshot ${SNAPSHOT_FILE}" > "${SUMMARY_FILE}"
  fi
fi

# Ranked offenders CSV (only meaningful if we have RUN_OUT)
echo "[mypy-py311] Generating ranked offenders CSV ..." | tee -a "${SNAPSHOT_FILE}"
{
  echo "file,error_count"
  printf "%s\n" "${RUN_OUT}" \
    | awk -F: '/: error: / {file=$1; count[file]++} END {for (f in count) printf("%s,%d\n", f, count[f])}' \
    | sort -t, -k2,2nr
} > "${RANKED_FILE}"

echo "[mypy-py311] Artifacts:" | tee -a "${SNAPSHOT_FILE}"
echo "  Summary:  ${SUMMARY_FILE}" | tee -a "${SNAPSHOT_FILE}"
echo "  Snapshot: ${SNAPSHOT_FILE}" | tee -a "${SNAPSHOT_FILE}"
echo "  Ranked:   ${RANKED_FILE}" | tee -a "${SNAPSHOT_FILE}"
echo "  Env:      ${ENV_FILE}" | tee -a "${SNAPSHOT_FILE}"

# Exit code logic:
# 0 => proper summary with 0 errors
# 1 => proper summary with non-zero errors
# 2 => no totals (runner issue) or docker failure
if [ -s "${SUMMARY_FILE}" ] && grep -qE '^Found 0 errors? in [0-9]+ files? \(checked [0-9]+ (source )?files?\)$' "${SUMMARY_FILE}"; then
  exit 0
fi
if [ -s "${SUMMARY_FILE}" ] && grep -qE '^Found [1-9][0-9]* errors? in [0-9]+ files? \(checked [0-9]+ (source )?files?\)$' "${SUMMARY_FILE}"; then
  exit 1
fi
exit 2
