# Contributing

Thanks for helping. This repository holds cartridge **blueprints** and the reference **materializer**; the
specification they implement lives in [SKaiNET-cartridge](https://github.com/SKaiNET-developers/SKaiNET-cartridge)
— changes to terms, schemas or rules go there first (as an ADR, or a SKEEP for anything durable).

## Ground rules for a blueprint

These are the spec's MUST NOTs, restated because they are the easiest to break by accident:

- **No weights, no compiled models, no built runtimes** — nothing of manifest role `weights`, `model` or a built
  `runtime`. `.gitignore` blocks the usual extensions; a tiny synthetic test fixture is the only exception.
- **No measured numbers as descriptor content.** `reference_measurements` is the place for what you saw.
- **No license acceptance.** A blueprint says acceptance is required; only a profile records that someone accepted.
- **No device, fleet, registry, customer or product names.** Targets are a hardware class and an ABI. Product
  content (a tool catalog, a vocabulary) is a declared *input*, never a sample in the tree.
- **Licenses are verified at the publisher** for the exact checkpoint and revision you pin — the model card, its
  README, the publisher's repository — and recorded with `license_url`. Do not infer them from the model family.

## Workflow

1. Open an issue describing the change.
2. Branch from `main`, keep commits focused, run `./gradlew build` before pushing.
3. Open a pull request; CI must be green.

New code needs tests. For the materializer that usually means a case in `MaterializerTest` and, when Gradle wiring
is involved, in `MaterializeFunctionalTest`.

## Licensing of contributions

Contributions are accepted under the repository's MIT license. The project follows [REUSE](https://reuse.software/);
`REUSE.toml` covers files without their own header.
