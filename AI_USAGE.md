# AI Usage Log — Group 99 - PTQ

This file logs all generative-AI tool use across Assignments 1–3, per the
course AI-use policy.

## Log

| # | Tool (name/version) | Used by | Stage/date | Purpose | Section(s) affected | Prompt summary / log link | AI contribution | Verification Sources | Student verification | Responsible member |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | OpenAI ChatGPT 5.6 Luna | Nguyen Thanh The | Repository initialization, 8–9 Sep 2026 | Review repository structure and handbook compliance | README.md, docs/*.md, AI_USAGE.md | Asked whether the course-project skeleton satisfied the handbook | Suggested documentation, Pages, disclosure, and reproducibility improvements; no autonomous edits | Course handbook; GitHub Pages documentation | Suggestions manually reviewed; Pages URL and links tested | Nguyen Thanh The |
| 2 | OpenAI Codex GPT-5 | Nguyen Minh Quang | Assignment 1 planning, 9 Sep 2026 | Create the Assignment 1 configuration structure | Assignment 1 configs and docs | Separate shared rules from model-specific architecture and training settings | Drafted YAML configs and protocol documentation; no model code or results | Course handbook; local repository structure | Config structure reviewed locally; protocol remains draft | Nguyen Minh Quang |
| 3 | OpenAI Codex GPT-5 | Nguyen Minh Quang | Assignment 1 implementation, 19–20 Sep 2026 | Design and verify the Fashion-MNIST experiment workflow | `assignment1/src/`, `assignment1/configs/`, `assignment1/tests/` | Requested a readable terminal workflow for configuring and running training/evaluation, plus a consistent way to save checkpoints, metrics, plots, and run metadata | Assisted with the CLI design, module responsibilities, configuration flow, run-directory layout, checkpoint/result naming, reproducibility safeguards, and test cases | PyTorch and torchvision runtime behaviour; repository protocol | The implementation was reviewed locally; thirteen unit tests passed; smoke runs were checked; final Linear/MLP runs were executed manually | Nguyen Minh Quang |
| 4 | OpenAI Codex GPT-5 | Nguyen Minh Quang | Assignment 1 milestone experiments and report, 20 Sep 2026 | Check experiment logs, curate artifacts, aggregate three-seed results, and draft milestone documentation | `assignment1/results/`, `assignment1/README.md`, `docs/assignment1*.md`, `environment/README.md` | Requested verification of the 50,000/10,000 split and six Linear/MLP training/evaluation runs | Calculated aggregate summaries from saved outputs, checked seed/checkpoint mappings, produced error-analysis statistics, organized artifacts, and drafted documentation | Saved histories, test-result JSON files, confusion matrices, split manifest, environment queries | All six run/seed mappings were checked; metrics were regenerated from local artifacts; no result was invented | Nguyen Minh Quang |
| 5 | Google Antigravity | Nguyen Thanh The | Assignment 2 implementation, 20 Oct 2026 | Implement baseline CNN, pretrained ResNet-50, training & evaluation pipelines, configs, and unit tests | `assignment2/src/`, `assignment2/configs/`, `assignment2/tests/`, `assignment2/README.md` | Requested complete implementation of Assignment 2 pipeline for Food-101 based on dataset proposal | Assisted with model architectures (SimpleCNN, ResNet50Classifier), training/evaluation scripts, data loader with stratified split, metrics calculation, and test suite | PyTorch / torchvision documentation, repository protocol | The implementation was verified locally; all six unit tests passed; end-to-end training and evaluation runs were tested on a dummy dataset | Nguyen Thanh The |
| 6 | Google Antigravity | Nguyen Thanh The | Assignment 2 M2 Draft milestone, 21 Oct 2026 | Curate baseline and frozen ResNet-50 experimental results, generate plots, and format error analysis for M2 Draft | `assignment2/results/`, `docs/assignment2.md`, `docs/assets/a2/`, `assignment2/README.md` | Requested verification of experimental outputs from Kaggle/Colab runs, generation of training curves and confusion matrix artifacts, and update of M2 draft documentation | Parsed training logs, generated quantitative comparison tables, formatted top confused pairs for error analysis, and linked visual artifacts | Training history JSON/CSV files, test metric outputs | Metrics verified directly against test set evaluation logs; numbers and plots match actual training runs | Nguyen Thanh The |
| 7 | Google Antigravity | Nguyen Thanh The | Assignment 2 final experiments, 26 Oct 2026 | Record E3 (ResNet-50 full fine-tuning) results, update comparative tables, error analysis, and finalize M3 report draft | `assignment2/results/resnet50_finetune/`, `docs/assignment2.md`, `docs/assets/a2/`, `assignment2/README.md` | Requested processing of Kaggle training log output for E3 fine-tuning, generation of training curve plots, and comprehensive report update including all three experiments | Generated history JSON/CSV, summary and test metrics files, training curves PNG, updated quantitative comparison table, cost analysis, and limitations section | Kaggle training/evaluation console output logs | All numbers match console output verbatim; training curves generated from raw epoch logs; no result was invented | Nguyen Thanh The |
| 8 | OpenAI Codex | Nguyen Minh Quang | Assignment 1 result analysis, October 2026 | Summarize and analyze the five-model comparison and ablation results | Assignment 1 report results tables, training-curve summary, ablation discussion, and saved benchmark artifacts | Requested support for organizing, checking, and comparing the saved test results | Summarized and analyzed the saved metrics, histories, and figures to support the result analysis; no experimental result was created by the tool | Saved training summaries, histories, test-result JSON files, confusion matrices, and aggregate CSV files | Reported values and interpretations were checked carefully against the saved artifacts; final responsibility remains with the student | Nguyen Minh Quang |

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
