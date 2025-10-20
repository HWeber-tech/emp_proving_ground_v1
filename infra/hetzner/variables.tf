variable "hcloud_token" {
  description = "Hetzner Cloud API token with write scope."
  type        = string
  sensitive   = true
}

variable "server_name" {
  description = "Name for the Hetzner server resource."
  type        = string

  validation {
    condition     = length(trimspace(var.server_name)) > 0
    error_message = "server_name cannot be blank."
  }
}

variable "server_type" {
  description = "Hetzner server type (must satisfy 8+ vCPU and 32-64 GB RAM requirement)."
  type        = string
  default     = "cx52"

  validation {
    condition     = contains(["cx51", "cx52", "ccx33", "ccx43", "ax41", "ax51", "ax61"], var.server_type)
    error_message = "server_type must map to an 8+ core, 32-64 GB NVMe instance class."
  }
}

variable "location" {
  description = "Hetzner datacenter location slug (e.g. hel1, fsn1, nbg1)."
  type        = string
  default     = "hel1"
}

variable "ssh_key_name" {
  description = "Name to register the SSH public key under in Hetzner."
  type        = string
  default     = "emp-hetzner-admin"
}

variable "ssh_public_key" {
  description = "SSH public key material (single line) to authorize on the server."
  type        = string

  validation {
    condition     = can(regex("^(ssh|ecdsa)-", trimspace(var.ssh_public_key)))
    error_message = "ssh_public_key must be an OpenSSH-formatted public key."
  }
}

variable "admin_username" {
  description = "Primary sudo-capable user provisioned via cloud-init."
  type        = string
  default     = "empops"
}

variable "trusted_admin_cidrs" {
  description = "CIDR blocks allowed to access SSH on the firewall. Leave empty to disable firewall creation."
  type        = list(string)
  default     = []
}

variable "enable_backups" {
  description = "Toggle Hetzner backups for the server (adds monthly cost)."
  type        = bool
  default     = true
}

variable "user_data_override" {
  description = "Optional cloud-init user_data to override the default bootstrap."
  type        = string
  default     = null
  nullable    = true
}

variable "labels" {
  description = "Labels applied to the server for inventory grouping."
  type        = map(string)
  default = {
    environment = "production"
    role        = "emp-single-box"
    owner       = "emp"
  }
}
