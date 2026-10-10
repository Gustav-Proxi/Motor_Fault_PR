# Repository Guidelines

## Project Structure & Module Organization

This project studies six-class induction-motor fault recognition from smartphone vibration recordings. Read `CSE455_Proposal_VaishakGKumar.pdf` (submitted proposal) and `PROJECT_IDEA.md` (expanded plan) before implementing experiments. The base pipeline lives in `src/motorpr/`; see `README.md` for commands.

Use `src/` for reusable Python modules, `tests/` for tests, `notebooks/` for exploration, and `results/` for generated figures and metrics. Store recordings in `data/raw/` and features in `data/processed/`. Exclude datasets, virtual environments, and reproducible outputs from version control.

## Build, Test, and Development Commands

Dependencies are listed in `pyproject.toml` (NumPy, SciPy, pandas, pyarrow, scikit-learn, Matplotlib; pytest for tests).

- `python scripts/download_data.py` — fetch the 36 recordings into `data/raw/`.
- `python scripts/run_baseline.py` — clean front-end × classifier grid into `results/baseline/`.
- `python scripts/run_corruption.py --corrupt both|test` — corruption sweep.
- `python -m pytest` — run tests.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for modules, functions, and variables, and `PascalCase` for classes. Add public-function type hints and docstrings explaining units, array shapes, and feature definitions. Keep numerical logic in reusable modules. No formatter or linter is configured.

## Testing Guidelines

Name test files `tests/test_<module>.py` and functions `test_<behavior>`. Prioritize synthetic signals with known statistics or frequencies to validate handcrafted features, window boundaries, and corruption routines. Test that recording IDs never overlap between training and evaluation partitions. No coverage threshold is established.

## Experiment & Data Requirements

The proposal compares PCA/LDA and includes Dynamic Time Warping; `PROJECT_IDEA.md` adds Kernel PCA, Subspace Alignment, and explicit OOD splits. Document which scope each experiment implements.

Use handcrafted features and classical methods only; no pretrained models or deep learning. Preserve recording IDs, class labels, speed, and load metadata. Use recording-grouped cross-validation, fitting learned preprocessing and projections only on training data. Report naive random splits separately as a leakage comparison. Apply corruption to raw signals before feature extraction. Record seeds, parameters, dependency versions, corruption levels, and OOD splits. Retain dataset attribution and license information.

## Commit & Pull Request Guidelines

This directory has no Git history, so no existing commit convention can be inferred. Use short, imperative messages such as `Add Welch band-power features`. PRs should describe the change, validation performed, and effects on experiment results. Link related issues when available and include plots for changes to reported findings.
