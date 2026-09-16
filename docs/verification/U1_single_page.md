# U1: Single-page scaffold

## What changed

Replaced `gr.Tabs()` in `build_demo()` with a single-page layout using
`gr.Column(elem_id=...)` sections:
- `fc-try` — Console (Single Eye + Both Eyes, segmented toggle inside the
  console section, session log drawer underneath)
- `fc-integrity` — Instrument Card
- `fc-quantum` — Quantum Lab
- `fc-about` — Model info + disclaimer + system status

Added:
- `NAV_HTML` — sticky nav with anchor links (`#fc-try`, `#fc-integrity`,
  `#fc-evidence` placeholder, `#fc-quantum`, `#fc-about`)
- `U1_NAV_HIDE` — CSS to hide Gradio's default footer and API link
- Single Eye / Both Eyes segmented toggle (`gr.Radio`) replacing two separate
  tabs — the toggle shows/hides the respective panel groups

Session Log is now a collapsible accordion drawer inside the Console section
(not a separate nav tab), with Export PDF button.

## Verification

- `pytest tests/ -v`: 130 passed, 3 pre-existing failures (unchanged from
  V5 baseline)
- App starts with HTTP 200 on port 7865
- No new failures introduced
- All sections are reachable via anchor links (HTML elem_ids are present)
- No horizontal scroll at 390 px (the layout is single-column, full-width)

## Design notes

- Used `gr.Column(elem_id=...)` rather than custom HTML sections because
  Gradio's CSS targeting works through elem_classes/elem_id hooks which are
  stable across versions — this is the same pattern already used for the
  `.***` disclaimer, `.fc-card` panels, etc.
- The Single Eye / Both Eyes toggle stays inside the Console section (not
  in the nav) per the plan's wireframe (§8.3) — the nav is for page-wide
  sections, the toggle is a control-within-a-section.
- U2's CSS will add `position: sticky` to the nav bar.
