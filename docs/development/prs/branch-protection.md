---
# Branch Protection: Main

This repository uses a single CI workflow named "CI" with job checks:
- CI / policy
- CI / lint
- CI / types
- CI / tests

Backtest is optional and not required for merging.

## One-time setup (repo admin)

1) Create a GitHub secret:
   - Name: REPO_ADMIN_TOKEN
   - Value: A Personal Access Token with permissions to administer this repository.
     - Classic PAT: scope "repo".
     - Fine-grained PAT: Repository permissions → Administration: Read and write (and access to this repo).
   - The token owner must have admin permissions on this repository.

2) Apply branch protection
   - Trigger workflow: Actions → "Enforce Branch Protection" → Run workflow.
   - Input "branch": main (default).
   - The workflow configures:
     - Require status checks: "CI / policy", "CI / lint", "CI / types", "CI / tests"
     - Strict status checks (must be up-to-date)
     - Dismiss stale approvals on new commits
     - Require at least 1 approving review
     - Disallow force pushes and deletions
     - Require linear history
     - Require conversation resolution

3) Verification
   - Settings → Branches → Branch protection rules → main
   - Confirm the required checks and restrictions match the above.

## Notes
- If job names change, update contexts in [.github/workflows/enforce-branch-protection.yml](.github/workflows/enforce-branch-protection.yml:1).
- Keep workflow name "CI" to preserve compatibility with the Kilocode bridge in [.github/workflows/kilocode-ci-bridge.yml](.github/workflows/kilocode-ci-bridge.yml:1).
---