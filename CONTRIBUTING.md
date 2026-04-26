# Contributing to zeroshot-detect

Same hard rules as the rest of the portfolio — written down here so they're not implicit.

---

## 1. DRY — Don't Repeat Yourself

- **Second occurrence extracts.** First time inline; second time refactor.
- **No copy-paste.** Lift into a function, class, or constant instead.
- **One source of truth per fact.** Model names, thresholds, prompts, paths declared once and imported.

Exceptions need a one-line comment explaining why duplication is load-bearing.

---

## 2. Code comments — WHAT + WHY, never HOW

The code is the HOW. Comments add what the code can't say.

- **WHAT** — a one-line summary of intent when the function name alone isn't enough.
- **WHY** — the non-obvious reason behind a choice, constraint, or workaround.

Good:
```python
# WHAT: clamp confidence threshold to >= 0.05.
# WHY: OWLv2 emits ~hundreds of low-score boxes that flood the UI; below
# 0.05 the noise dominates and the visualisation becomes unreadable.
```

Bad:
```python
for box in boxes:   # iterate over the boxes
    ...
```

---

## 3. Commit messages — WHAT + WHY + WHERE

Subject line ≤72 chars, imperative. Body answers WHAT changed, WHY it changed, WHERE it landed. One logical change per commit. ADRs land in the same commit as the code they describe.

Template at `.gitmessage`. Configure with `git config commit.template .gitmessage`.

---

## 4. Docs updated continuously

- A change that alters public behaviour updates the affected doc in the same commit.
- Architectural decisions live in `docs/adr/NNNN-short-title.md`. Numbers monotonic; never reused.
- `README.md` reflects current state, not aspiration.

---

## 5. Testing

- New logic ships with tests. Bug fixes ship with a regression test that fails before the fix.
- `pytest` must pass before commit. CI enforces.
- Tests live in `tests/` mirroring `zsdetect/`.
- Tests run **offline** — the OWLv2 model is mocked. Real inference is gated behind the `integration` mark and skipped by default.

---

## 6. Tooling

- `ruff check . && ruff format .` before commit.
- `mypy zsdetect/ app.py` must pass (strict).
- Python ≥ 3.11.
