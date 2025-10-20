# Kubernetes RBAC Guide for EMP

This guide explains how role-based access control (RBAC) is wired into the EMP deployment and how to extend it safely. The canonical manifests live in `config/security/security-policy.yaml` and are applied alongside the other cluster security primitives (network policies, pod security context, Vault integration).

## Current RBAC Layout

- **Service Account: `emp-service-account` (namespace `emp-system`)** – Pods for the core EMP application run under this identity. It is referenced by deployments and grants the pod a non-default identity.
- **ClusterRole: `emp-cluster-role`** – Grants read-only access (`get`, `list`, `watch`) to pods, services, endpoints, nodes, ConfigMaps, Secrets, and namespaces so the app can observe cluster state without mutating it.
- **ClusterRoleBinding: `emp-cluster-role-binding`** – Binds the above role to the EMP service account. Because it is a cluster role binding, the permissions apply across all namespaces.
- **Vault RBAC (service account `vault-service-account`)** – Vault pods receive a separate cluster role (`vault-cluster-role`) that enables `*` verbs on Secrets and ConfigMaps. This is intentionally broader and isolated from the EMP role binding so Vault can manage secrets without granting the EMP pods mutation rights.

All of the objects sit in the same manifest so applying it keeps the security model converged.

## Common Extension Patterns

1. **Grant additional read-only resources to EMP**
   - Edit the rules block under `emp-cluster-role` in `config/security/security-policy.yaml`.
   - Add the resource (or API group) with the verbs you need, keeping read-only verbs whenever possible.
   - Apply the change: `kubectl apply -f config/security/security-policy.yaml`.
   - Validate with `kubectl auth can-i list <resource> --as system:serviceaccount:emp-system:emp-service-account`.

2. **Give EMP write access to a resource**
   - Add a new rule to `emp-cluster-role` scoped to the resource and namespace in question.
   - Restrict verbs to only what is necessary (`create`, `patch`, `update`, `delete`). Consider splitting into a dedicated role if the permissions are broad.
   - Re-apply the manifest and confirm using `kubectl auth can-i`.

3. **Onboard a new EMP subsystem**
   - Create a new service account block in the manifest for the subsystem namespace.
   - Define a namespaced `Role` or a `ClusterRole` if it needs cluster-wide scope.
   - Bind with `RoleBinding`/`ClusterRoleBinding` targeting the new service account.
   - Reuse existing rule snippets as templates and keep namespaces explicit to avoid accidental cluster-wide access.

4. **Add permissions for Vault or other infrastructure controllers**
   - Vault already has a dedicated cluster role; extend it by appending rules under `vault-cluster-role`.
   - For brand‑new controllers, mirror the structure used for EMP: service account → role → binding, keeping them in the same manifest to simplify reconciliation.

## Good Practices When Extending

- **Principle of least privilege:** default to read-only access, and isolate write access to dedicated roles when possible.
- **Repeatable deliveries:** treat the manifest as the single source of truth and avoid `kubectl edit`; instead, change the file and re-apply it.
- **Drift checks:** after applying, run `kubectl auth can-i` checks for critical verbs/resources and `kubectl describe` on the binding to ensure the subjects are correct.
- **Review impact:** when adding `*` verbs or cluster-scoped rules, review with security/operations because these changes affect all namespaces.

## Testing and Verification Workflow

1. Update `config/security/security-policy.yaml` with the new rule(s).
2. Apply to the cluster: `kubectl apply -f config/security/security-policy.yaml`.
3. Confirm the permissions:
   ```bash
   kubectl auth can-i get secrets --as system:serviceaccount:emp-system:emp-service-account
   kubectl auth can-i update secrets --as system:serviceaccount:emp-system:emp-service-account
   ```
4. Observe audit trails: Kubernetes audit logging (configured later in the same manifest) captures authentication and RBAC denials. Monitor the audit sink or logging backend to verify no unexpected denials occur.

Keeping the documentation and manifest synchronized ensures operators understand the current permission set and how to evolve it responsibly.
