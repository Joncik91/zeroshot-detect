# ADR-0004: UI and deployment

- **Status:** accepted
- **Date:** 2026-04-26
- **Deciders:** Jounes

## Context

Once the detector + renderer + data contract are settled (ADRs 0001-0003), the remaining choices are:

- What UI framework hosts the demo?
- What does the layout actually look like?
- How does the demo get from a `git push` to a public URL?
- How is the "I clicked Detect, is anything happening?" UX handled given 5-15 s CPU inference?

Forces:
- Already shipped paperQA on Gradio + HF Spaces. Reusing what worked beats inventing.
- Inference is slow (~5-15 s on CPU, ~30-60 s for first-call cold start). UI must signal progress or visitors will think the page is broken.
- The deploy procedure must survive future re-runs without surprises.

## Decision

1. **UI framework: Gradio Blocks** (sdk version 6.13.0, matching what HF auto-selects). Same choice as paperQA. Cheapest path from a Python function to a hosted UI; the `<gradio-app>` web component lets the demo also embed in the portfolio site at https://joncik91.github.io.

2. **Layout: two-column, image-led.**
   - **Left column:** image upload (`gr.Image`, `type="pil"`, sources = upload + clipboard), labels textbox (comma-separated), confidence-threshold slider (0.05-0.50, default 0.10), "Detect" button, status markdown line.
   - **Right column:** annotated image output, detections dataframe (`label`, `score`, `box`).
   - Same scale ratio as paperQA (1:2) so the user's eye lands on the output, not the controls.

3. **Loading-state pattern: chained click handlers.** Same `submit.click(...).then(_detect, ...).then(...)` pattern as paperQA's app. Pass 1 immediately disables the button and shows `⏳ Running OWLv2…` in the status line. Pass 2 runs the actual detection. Pass 3 re-enables the button.

4. **Deployment target:** Hugging Face Space, **CPU basic** (free tier), Gradio SDK, blank template, no persistent storage. Same procedure as paperQA's `docs/deploying.md` (orphan-worktree push to the Space's `main`).

5. **Same deploy gotchas documented up-front:**
   - HF rejects binary fixtures from regular git → exclude `tests/` from the deploy tree.
   - `short_description` must be ≤ 60 characters → README frontmatter checked before push.
   - `requirements.txt` must list every runtime dep explicitly (HF Spaces ignores `pyproject.toml` extras).

## Alternatives considered

- **Streamlit instead of Gradio.** Has a comparable HF Spaces story but no `<gradio-app>`-equivalent web component for embedding into the portfolio site. Plus, paperQA's Gradio code is reusable here.

- **FastAPI + a separate HTML page.** More flexible but requires its own deploy story and front-end discipline. Premature for v1; reach for it when a Gradio limitation actually bites.

- **Server-Sent Events / streaming progress** instead of the chained-click pattern. More elegant but Gradio's component model already gives us the chained version for free. SSE is a v2 polish item if cold starts get longer.

- **GPU Space** instead of CPU basic. Would cut inference to ~1 s, but introduces a billing line that doesn't fit "free portfolio demo." Documented as an upgrade path in the README; not the v1 default.

## Consequences

- **Positive:**
  - Familiar tooling — second-time-through on Gradio + Spaces means fewer unknowns.
  - Loading-state pattern makes the slow CPU inference feel acknowledged rather than broken.
  - Embeddable in the portfolio site via `<gradio-app>`.
  - Same deploy runbook as paperQA — a single mental model covers both projects.

- **Negative:**
  - First request after a cold start pays a ~30-60 s OWLv2 download. Documented in the status line and the README.
  - 5-15 s/inference on CPU caps the perceived snappiness regardless of UI polish. Mitigated, not eliminated, by the loading state.
  - Blank-template Space means we own the entire build artifact (requirements.txt, app.py, frontmatter). That's the right trade for transparency but means tiny mistakes (gradio version skew, missing dep) surface as build errors not silent fixes.

- **Follow-ups:**
  - If the demo gets enough traffic to make GPU billing worth it, document the GPU-Space deploy variant alongside the CPU one.
  - If a v2 image-guided detection mode is added (per ADR-0001's follow-up list), it gets a second tab in the same Blocks app.
