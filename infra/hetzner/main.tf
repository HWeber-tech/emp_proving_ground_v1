locals {
  rendered_user_data = coalesce(
    var.user_data_override,
    templatefile("${path.module}/templates/cloud_init.yaml.tftpl", {
      hostname       = var.server_name
      admin_username = var.admin_username
      ssh_public_key = trimspace(var.ssh_public_key)
    })
  )
  firewall_ids = [for fw in hcloud_firewall.admin : fw.id]
}

resource "hcloud_ssh_key" "admin" {
  name       = var.ssh_key_name
  public_key = trimspace(var.ssh_public_key)
}

resource "hcloud_firewall" "admin" {
  count = length(var.trusted_admin_cidrs) > 0 ? 1 : 0

  name = "${var.server_name}-ssh"

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = var.trusted_admin_cidrs
  }

  rule {
    direction        = "out"
    protocol         = "tcp"
    port             = "1-65535"
    destination_ips  = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction        = "out"
    protocol         = "udp"
    port             = "1-65535"
    destination_ips  = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction       = "out"
    protocol        = "icmp"
    destination_ips = ["0.0.0.0/0", "::/0"]
  }
}

resource "hcloud_server" "single_box" {
  name        = var.server_name
  server_type = var.server_type
  image       = "ubuntu-22.04"
  location    = var.location
  backups     = var.enable_backups

  ssh_keys = [
    hcloud_ssh_key.admin.id,
  ]

  firewalls = local.firewall_ids

  labels = var.labels

  user_data = local.rendered_user_data

  public_net {
    ipv4_enabled = true
    ipv6_enabled = true
  }
}
