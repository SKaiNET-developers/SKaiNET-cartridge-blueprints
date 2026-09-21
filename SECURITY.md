# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/SKaiNET-developers/SKaiNET-cartridge-blueprints/security/advisories/new)
(Security tab → *Report a vulnerability*). Do not open a public issue for security reports.

You will receive an acknowledgement within a few days. Please give us reasonable time to assess and fix the
issue before disclosing it publicly.

## Scope

In scope:

- The materializer plugin (`blueprint-gradle-plugin/`): source fetching and digest verification, credential
  handling, path handling when caching and packing, manifest construction and signing.
- The blueprints in this repository: pinned source URIs, revisions and digests.

Out of scope:

- Vulnerabilities in model weights or in the publishers' distribution channels.
- Cartridges materialized by third parties.
- Third-party dependency CVEs that are not reachable from this repository's code.
