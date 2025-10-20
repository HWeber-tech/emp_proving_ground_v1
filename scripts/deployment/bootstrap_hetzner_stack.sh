#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[hetzner-bootstrap] %s\n' "$*"
}

ensure_command() {
  command -v "$1" >/dev/null 2>&1
}

main() {
  local sudo_cmd=""
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    sudo_cmd="sudo"
  fi

  if ! ensure_command docker; then
    log 'Installing Docker Engine and compose plugin'
    ${sudo_cmd} apt-get update
    ${sudo_cmd} apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release
    ${sudo_cmd} install -m 0755 -d /etc/apt/keyrings
    if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | ${sudo_cmd} gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    fi
    local codename
    codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
    local arch
    arch=$(dpkg --print-architecture)
    if [[ ! -f /etc/apt/sources.list.d/docker.list ]]; then
      echo "deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${codename} stable" | ${sudo_cmd} tee /etc/apt/sources.list.d/docker.list >/dev/null
    fi
    ${sudo_cmd} apt-get update
    ${sudo_cmd} apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ${sudo_cmd} systemctl enable --now docker
  else
    log 'Docker already present; skipping installation step'
  fi

  if [[ "${EUID:-$(id -u)}" -ne 0 ]] && ! id -nG "${USER}" | grep -qw docker; then
    log "Adding ${USER} to docker group"
    ${sudo_cmd} usermod -aG docker "${USER}"
    log 'Log out/in to refresh the docker group membership'
  fi

  local root_dir
  root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  local compose_file
  compose_file="${root_dir}/docker/hetzner/docker-compose.yml"

  if [[ ! -f "${compose_file}" ]]; then
    log "Missing compose file at ${compose_file}"
    exit 1
  fi

  local docker_cmd="docker"
  if [[ -n "${sudo_cmd}" ]]; then
    docker_cmd="${sudo_cmd} docker"
  fi

  log 'Building engine image'
  ${docker_cmd} compose -f "${compose_file}" build engine
  log 'Starting TimescaleDB, Redis, Kafka, and engine'
  ${docker_cmd} compose -f "${compose_file}" up -d timescale redis kafka engine
  log 'Stack status'
  ${docker_cmd} compose -f "${compose_file}" ps
}

main "$@"
