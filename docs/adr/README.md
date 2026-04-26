# Architecture Decision Records

Every non-obvious decision that shapes zeroshot-detect's design lands here as an ADR.

## Index

- [0001 — Scope and model choice](0001-scope-and-model-choice.md)
- [0002 — Detection data contract](0002-detection-data-contract.md)
- [0003 — Rendering strategy](0003-rendering-strategy.md)
- [0004 — UI and deployment](0004-ui-and-deployment.md)

## How to add one

1. Copy `TEMPLATE.md` to `NNNN-short-title.md` (next integer, zero-padded to 4).
2. Fill in Context, Decision, Alternatives, Consequences.
3. Add a one-line entry to the index above in the same commit.
4. Reference the ADR from the commit body and from code comments where the decision is load-bearing.
