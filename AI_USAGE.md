# AI Usage Log — Group 99 - PTQ

This file logs all generative-AI tool use across Assignments 1–3, per the
course AI-use policy.

## Log

| # | Tool (name/version) | Used by | Stage/date | Purpose | Section(s) affected | Prompt summary / log link | AI contribution | Verification Sources | Student verification | Responsible member |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | OpenAI ChatGPT 5.6 Luna | Nguyen Thanh The | Repository initialization, 8–9 Sep 2026 | Review repository structure and handbook compliance | README.md, docs/*.md, AI_USAGE.md | Asked whether the course-project skeleton satisfied the handbook | Suggested documentation, Pages, disclosure, and reproducibility improvements; no autonomous edits | Course handbook; GitHub Pages documentation | Suggestions manually reviewed; Pages URL and links tested | Nguyen Thanh The |
| 2 | OpenAI Codex GPT-5 | Nguyen Minh Quang | Assignment 1 planning, 9 Sep 2026 | Create the Assignment 1 configuration structure | Assignment 1 configs and docs | Separate shared rules from model-specific architecture and training settings | Drafted YAML configs and protocol documentation; no model code or results | Course handbook; local repository structure | Config structure reviewed locally; protocol remains draft | Nguyen Minh Quang |
| 3 | OpenAI Codex GPT-5 | Nguyen Minh Quang | Assignment 1 implementation, 19–20 Sep 2026 | Design and verify the Fashion-MNIST experiment workflow | `assignment1/src/`, `assignment1/configs/`, `assignment1/tests/` | Requested support for a readable terminal workflow for configuring training/evaluation, saving checkpoints and metrics, and organizing plots and run metadata | Supported CLI design, module responsibilities, configuration flow, run-directory layout, checkpoint/result naming, reproducibility safeguards, and test cases | PyTorch and torchvision runtime behaviour; repository protocol | The implementation was reviewed locally; thirteen unit tests passed; smoke runs and the resulting outputs were checked carefully | Nguyen Minh Quang |
| 4 | OpenAI Codex GPT-5 | Nguyen Minh Quang | Assignment 1 milestone experiments and report, 20 Sep 2026 | Check experiment logs, curate artifacts, aggregate three-seed results, and draft milestone documentation | `assignment1/results/`, `assignment1/README.md`, `docs/assignment1*.md`, `environment/README.md` | Requested support for checking the 50,000/10,000 split, organizing the experiment outputs, and preparing the milestone material | Calculated aggregate summaries from saved outputs, checked seed/checkpoint mappings, produced error-analysis statistics, organized artifacts, and supported documentation planning | Saved histories, test-result JSON files, confusion matrices, split manifest, environment queries | All six run/seed mappings were checked; metrics were regenerated from local artifacts; no result was invented | Nguyen Minh Quang |
| 5 | OpenAI Codex | Nguyen Minh Quang | Assignment 1 result analysis, October 2026 | Summarize and analyze the five-model comparison and ablation results | Assignment 1 report results tables, training-curve summary, ablation discussion, and saved benchmark artifacts | Requested support for organizing, checking, and comparing the saved test results | Summarized and analyzed the saved metrics, histories, and figures to support the result analysis; no experimental result was created by the tool | Saved training summaries, histories, test-result JSON files, confusion matrices, and aggregate CSV files | Reported values and interpretations were checked carefully against the saved artifacts; final responsibility remains with the student | Nguyen Minh Quang |

### Field guide

- **Tool and model:** exact tool name and model/version if known.
- **Used by:** member name.
- **Stage/date:** development stage or timestamp.
- **Purpose:** e.g. literature search, concept explanation, architecture/experiment
  design, coding assistance, debugging, test generation, report writing,
  grammar check, figure/slide generation, result analysis.
- **Section(s) affected:** e.g. `src/train.py`, report §3.2.
- **Prompt summary:** short paraphrase or link to full prompt log.
- **AI contribution:** what the AI actually produced/suggested.
- **Student verification:** how it was checked (docs, tests, papers, manual run).
- **Responsible member:** who is accountable for the final, verified content.

## Reminders

- Do not submit unchecked AI-generated code or text.
- Do not use AI to fabricate data, results, citations, or references.
- Do not claim experiments that were not actually run.
- Do not paste private/restricted/credential-bearing data into AI tools.
- Missing or false disclosure may be treated as an academic integrity violation.
