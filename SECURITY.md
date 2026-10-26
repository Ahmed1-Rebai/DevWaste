# Security Policy

## Supported versions

DevWaste AI is an MVP. Only the latest commit on `main` receives fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/Ahmed1-Rebai/DevWaste/security/advisories/new)
for this repository. Include the affected command, the version or commit,
and steps to reproduce.

## Security model

- DevWaste only reads from the Docker API (list, inspect, stats). It never
  starts, stops, or removes containers, images, networks, or volumes.
- `cleanup` is dry-run only; `--no-dry-run` is refused.
- It does not read environment variables, logs, or container filesystems,
  and it makes no network calls besides the local Docker daemon.
- Access to the Docker socket is equivalent to root on the host. The
  development `docker-compose.yml` mounts it; remove that mount if you only
  use synthetic data.
- Model artifacts are loaded with `torch.load`. Only load model directories
  that you trained yourself or trust.
