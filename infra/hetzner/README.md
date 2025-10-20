# Hetzner single-box provisioning

This Terraform module provisions the Phase J single-box host in Hetzner Cloud
with the hardware requirements captured in the roadmap (8+ vCPU, 32-64 GB RAM,
NVMe storage, Ubuntu 22.04). The configuration creates a server, registers an
SSH key, applies an optional SSH-only firewall, and bootstraps a sudo-capable
operator via cloud-init.

## Prerequisites

- Terraform >= 1.5.0
- Hetzner Cloud API token with write scope (`HCLOUD_TOKEN`)
- SSH public key that should be authorized on the host

## Usage

1. Copy the example variable file and populate required values:

   ```bash
   cd infra/hetzner
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Edit `terraform.tfvars` and set:

   - `hcloud_token` (or export `HCLOUD_TOKEN`)
   - `server_name`
   - `ssh_public_key`
   - Optional: `trusted_admin_cidrs`, `labels`, `server_type`

3. Initialise and review the plan:

   ```bash
   terraform init
   terraform plan
   ```

4. Provision the server:

   ```bash
   terraform apply
   ```

   The outputs include the assigned IPv4/IPv6 addresses and a ready-to-copy SSH
   command using the configured admin user.

To destroy the server when no longer needed:

```bash
terraform destroy
```

## Defaults and safeguards

- `server_type` defaults to `cx52` (8 vCPU / 32 GB / NVMe). Validation restricts
  overrides to instance classes that satisfy the hardware envelope.
- `trusted_admin_cidrs` controls the optional SSH-only firewall. Provide at
  least one CIDR (e.g. a VPN range) to enable it; leave empty to skip firewall
  attachment.
- Cloud-init disables password authentication and root SSH logins, installs the
  Hetzner guest agent, and creates the `empops` user with passwordless sudo.

## Example `terraform.tfvars`

```hcl
hcloud_token        = "${HCLOUD_TOKEN}"
server_name         = "emp-hetzner-single-box"
ssh_public_key      = "ssh-ed25519 AAAA... user@example"
trusted_admin_cidrs = ["203.0.113.0/24"]
labels = {
  environment = "production"
  role        = "emp-single-box"
  owner       = "emp"
}
```

The resulting host satisfies deliverable **J.1** and can immediately run the
bootstrap script in `scripts/deployment/bootstrap_hetzner_stack.sh` to complete
Phase J.2.
