# Security Policy

## Supported versions

DevWaste AI is pre-release (`0.1.0`, unreleased). Only the latest commit on
`main` receives fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
**Security → Report a vulnerability** on
[github.com/Ahmed1-Rebai/DevWaste](https://github.com/Ahmed1-Rebai/DevWaste/security/advisories/new).
Do not open a public issue for a security problem.

Include the affected command or module, steps to reproduce, and the impact you
expect. You should get a first reply within 7 days.

## Scope notes

- DevWaste talks to the Docker daemon through the Docker socket. Access to that
  socket is equivalent to root on the host; the provided `docker-compose.yml`
  mounts it only so that `scan`, `collect` and `cleanup` work.
- The tool is designed to be read-only: it lists and inspects Docker objects and
  never starts, stops or removes anything. Any code path that mutates Docker
  state is a security bug.
- It collects resource metrics, names, IDs and state only. Collection of
  environment variables, logs, payloads or filesystem contents is a bug.
