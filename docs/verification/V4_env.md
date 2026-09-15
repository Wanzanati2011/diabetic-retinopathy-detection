# V4: Environment

| Package | Version |
|---|---|
| Python | 3.11.6 (MSC v.1935 64-bit) |
| gradio | 6.27.0 |
| torch | 2.6.0+cu124 |
| timm | 1.0.28 |
| grad-cam | 1.5.7 |
| fpdf2 | **not installed** — `app/app.py:654-664` (`build_pdf_export`)
already lazy-imports it and degrades gracefully ("PDF export needs `pip
install fpdf2`"). Needed for T-12/T-14 and Task A7. Install before Phase 1
if PDF export tests are to actually run (rather than skip). |

## R10 implication (Gradio 6)

Gradio is at **6.27.0**, so per R10: `theme`/`css` moved to `.launch()`
(not `Blocks(theme=..., css=...)`), and other Gradio-6-only APIs may differ
from the plan's assumed baseline — check the installed version's actual
signatures (`gr.Blocks.launch`, head-injection mechanism, `demo.queue()`
signature, upload-cache cleanup option name) before writing code in Phase 2
(B5) and Phase 3 (U1), rather than assuming Gradio 4/5 API shapes from the
master-plan documents.

## `requirements-lock.txt`

Generated via `pip freeze` (90 packages), written to the repo root.
`requirements.txt` was **not** touched, per V4's instruction.
