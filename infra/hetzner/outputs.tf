output "server_id" {
  description = "Hetzner server identifier."
  value       = hcloud_server.single_box.id
}

output "server_ipv4" {
  description = "Public IPv4 address assigned to the server."
  value       = hcloud_server.single_box.ipv4_address
}

output "server_ipv6" {
  description = "Public IPv6 address assigned to the server."
  value       = hcloud_server.single_box.ipv6_address
}

output "ssh_command" {
  description = "Convenience SSH command for the provisioned host."
  value       = "ssh ${var.admin_username}@${hcloud_server.single_box.ipv4_address}"
}
