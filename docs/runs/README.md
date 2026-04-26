# Test runs

Snapshots of real demo runs against non-trivial inputs, kept here as
informal evidence the live Space works on cases the unit tests don't
cover. **Not** the README's headline screenshot — those should show
crisp boxes on natural photos. These are the messier edge cases.

## Index

- [2026-04-26 — Coolblue homepage screenshot (UI edge case)](2026-04-26-coolblue-ui-edgecase.png)

  Webpage screenshot input — far outside OWLv2's natural-photo training
  distribution. With threshold 0.10 and labels
  `apple icon, smartphone, red circle`, the demo correctly finds the
  Apple logo (0.475), three smartphones in the right-column ads
  (0.501-0.546), and three small red notification badges next to news
  timestamps (0.211-0.265). First label attempt used the Dutch typo
  "cirkle" which CLIP couldn't match — corrected to "red circle"
  surfaced them. Confirms the threshold slider + multi-label parsing
  do what they should under unfamiliar input.
