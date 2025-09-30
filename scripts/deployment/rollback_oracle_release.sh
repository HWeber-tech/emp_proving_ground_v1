#!/usr/bin/env bash
set -euo pipefail

# Placeholder rollback script referenced by the smoke-test plan.
# Update the namespace and deployment name if they differ in your cluster.

NAMESPACE="emp-system"
DEPLOYMENT="emp-app"
PREVIOUS_REVISION="$(kubectl rollout history deployment/${DEPLOYMENT} -n ${NAMESPACE} | awk '/\(current\)/{print prev} {prev=$1}')"

if [[ -z "${PREVIOUS_REVISION}" ]]; then
  echo "Unable to determine previous revision for ${DEPLOYMENT}" >&2
  exit 1
fi

echo "Rolling back ${DEPLOYMENT} in ${NAMESPACE} to revision ${PREVIOUS_REVISION}"
kubectl rollout undo deployment/${DEPLOYMENT} -n ${NAMESPACE} --to-revision="${PREVIOUS_REVISION}"
