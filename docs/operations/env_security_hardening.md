# Secrets and Network Hardening Runbook

Lock down the environment files that hold data service and broker credentials,
enforce a host firewall, and optionally expose the runtime through a WireGuard
tunnel. These steps assume Ubuntu 22.04 on bare metal or a cloud instance such
as the Hetzner profile defined in the deployment roadmap.

## 1. Lock down the dotenv files

1. Create a dedicated secrets directory owned by the service account that runs
   the EMP processes (replace `emp` with the actual user):

   ```bash
   sudo install -d -m 750 -o emp -g emp /etc/emp/secrets
   ```

2. Copy the required template (`env_templates/*.env`, `config/test_fix.env`,
   etc.), fill in production credentials, and write it to the secrets
   directory with restrictive permissions:

   ```bash
   sudo install -m 640 -o emp -g emp \ 
     /tmp/institutional.env /etc/emp/secrets/institutional.env
   ```

   `640` keeps the file readable by the service group but blocks other users.
   Use `600` when the runtime user does not need to share credentials with any
   helpers.

3. Verify the permissions before wiring services:

   ```bash
   sudo ls -al /etc/emp/secrets
   # -rw-r----- 1 emp emp 1.8K Apr  6  institutional.env
   ```

4. Reference the locked-down file instead of copying secrets into the repo:

   - Python tooling: `python -m tools.operations.managed_ingest_connectors \
     --env-file /etc/emp/secrets/institutional.env`
   - Systemd unit: add `EnvironmentFile=/etc/emp/secrets/runtime.env`
   - Docker Compose override: `env_file: /etc/emp/secrets/runtime.env`

   The helper scripts now look for `EMP_SECRETS_ENV_FILE`; export it once per
   shell (for example, `export EMP_SECRETS_ENV_FILE=/etc/emp/secrets/runtime.env`)
   so ad-hoc diagnostics pick up the locked-down dotenv automatically.

5. Rotate credentials by replacing the file in place and reloading the
   relevant services. Keep an encrypted offline backup (for example, using
   `age` or `sops`) instead of storing plaintext copies in version control.

## 2. Enforce a host firewall

Restrict inbound traffic to the SSH bastion and the handful of services the EMP
stack exposes internally. The commands below assume `ufw` (Uncomplicated
Firewall); adapt to `nftables` or `firewalld` if preferred.

```bash
sudo apt-get update && sudo apt-get install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH administration
sudo ufw allow 22/tcp

# Internal data plane (only from the trusted subnet)
sudo ufw allow from 10.20.0.0/16 to any port 5432 proto tcp  # TimescaleDB
sudo ufw allow from 10.20.0.0/16 to any port 6379 proto tcp  # Redis
sudo ufw allow from 10.20.0.0/16 to any port 9094 proto tcp  # Kafka

# Runtime API (paper/live health)
sudo ufw allow from 10.20.0.0/16 to any port 8000 proto tcp

# Optional WireGuard tunnel (section 3)
sudo ufw allow 51820/udp

sudo ufw enable
sudo ufw status verbose
```

Adjust the trusted subnet (`10.20.0.0/16` in the example) to match the actual
VPN or VPC range. When exposing public health checks use specific source CIDRs
or add reverse proxy authentication; never leave Redis, Timescale, or Kafka
open to the internet.

## 3. Optional WireGuard access

Use WireGuard when operators need a private, encrypted tunnel into the data
plane instead of punching permanent holes through the firewall.

1. Install WireGuard and enable IP forwarding:

   ```bash
   sudo apt-get install -y wireguard
   echo 'net.ipv4.ip_forward=1' | sudo tee /etc/sysctl.d/99-emp.conf
   echo 'net.ipv6.conf.all.forwarding=1' | sudo tee -a /etc/sysctl.d/99-emp.conf
   sudo sysctl --system
   ```

2. Generate keys and create `/etc/wireguard/wg0.conf`:

   ```bash
   umask 077
   wg genkey | tee server.key | wg pubkey > server.pub
   ```

   ```ini
   [Interface]
   Address = 10.88.0.1/24
   ListenPort = 51820
   PrivateKey = <contents of server.key>
   PostUp   = ufw route allow in on wg0 out on eth0
   PostUp   = iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
   PostDown = ufw route delete allow in on wg0 out on eth0
   PostDown = iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

   [Peer]
   PublicKey = <peer-public-key>
   AllowedIPs = 10.88.0.10/32
   ```

3. Bring up the tunnel:

   ```bash
   sudo systemctl enable --now wg-quick@wg0
   sudo wg show
   ```

4. On each peer, configure a complementary interface that points `AllowedIPs`
   to the runtime subnet (`10.20.0.0/16` in the firewall example). Only export
   the minimal routes required for operations.

Keep the WireGuard private keys outside of configuration management, rotate
them on operator offboarding, and remove peers with `sudo wg set wg0 peer
<key> remove` when access is no longer required.

## 4. Operational verification

- `sudo systemctl show <unit> --property=EnvironmentFile` confirms services are
  loading the locked-down dotenv files.
- `sudo ufw status verbose` (or the equivalent `nft list ruleset`) should list
  only the expected ports and subnets.
- `sudo wg show` must report the handshake timestamps for active operators; the
  command returns `latest handshake: (none)` when peers are offline.
- Incorporate these checks into the promotion and incident runbooks so
  credential leakage and inadvertent port exposure surface immediately.
