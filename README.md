Domain: Healthcare

# PharmEasy Healthcare Decision Intelligence

This repository is a reproducible, offline Python project scaffold for the Healthcare Decision Intelligence assignment.

## Status and scope

The supplied message contains submission guidelines and the required repository layout, but not the Part 1–4 task statements or the exact provided `generate_data.py`. Those materials are necessary to implement and verify the required calculations.

**This is not a completed graded submission.** Placeholder files are clearly marked; no reference values have been invented or hard-coded.

## Requirements

- Python 3.10 or newer
- NumPy 2.2.6
- No API keys, paid services, or dataset downloads

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Intended run order

After restoring the official generator and implementing each task from the full brief, run from the repository root:

```bash
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

Each script must generate its results under `outputs/`.

## Layout

```text
.
├── README.md
├── requirements.txt
├── generate_data.py
├── data/records.csv
├── part1.py
├── part2.py
├── part3.py
├── part4.py
└── outputs/
    ├── part1_results.txt
    ├── part2_results.txt
    ├── part3_results.txt
    ├── part4_results.txt
    ├── part2_thresholds.csv
    ├── part2_knn_cv.csv
    ├── part3_segments.csv
    ├── part3_hierarchical.csv
    ├── part4_splits.csv
    ├── part4_voting.csv
    ├── part4_adaboost.csv
    └── part4_architecture.csv
```

## Reproducibility rules

- Graded quantities must be calculated by project code.
- Use only Python's standard library and NumPy for graded calculations.
- Follow all split, tie-break, initialization, standard-deviation, threshold, and rounding conventions in the complete assignment.
- Keep all work on `main`.
- Commit only text, source code, data, and CSV files.

## Missing source material

To turn this scaffold into the actual assignment submission, add:

1. Complete task statements for Parts 1–4, including required result keys, CSV columns, and every pinned convention.
2. The exact official `generate_data.py` source.
3. Any additional source data or constants referenced by those tasks.

The current repository establishes the requested structure without pretending that missing computations or reference outputs have been completed.
