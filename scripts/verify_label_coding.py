"""Verify the coding of `target` in data/heart.csv against the raw UCI Cleveland file.

Finding (see README): in this Kaggle copy, target == 1 means NO heart disease (UCI num == 0)
and target == 0 means disease present (UCI num > 0).

Usage
-----
1. Download the UCI "Heart Disease" archive (CC BY 4.0, doi:10.24432/C52P4X):
   https://archive.ics.uci.edu/dataset/45/heart+disease  and unzip `processed.cleveland.data`.
2. Run (needs only numpy):

    python scripts/verify_label_coding.py path/to/processed.cleveland.data

Expected output: 302 of 302 de-duplicated rows matched, with target == 1 agreeing with
"no disease" in all of them and with "disease" in none.
"""
import sys
from pathlib import Path

import numpy as np

HEART_CSV = Path(__file__).resolve().parent.parent / "data" / "heart.csv"


def load_uci(path):
    """Read processed.cleveland.data -> list of (age, sex, trestbps, chol, num)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) == 14:
            rows.append((int(float(parts[0])), int(float(parts[1])),
                         int(float(parts[3])), int(float(parts[4])), int(float(parts[13]))))
    return rows


def main(uci_path):
    uci = load_uci(uci_path)
    print(f"UCI Cleveland rows: {len(uci)} | no disease (num == 0): "
          f"{sum(r[4] == 0 for r in uci)} | disease (num > 0): {sum(r[4] > 0 for r in uci)}")

    raw = np.genfromtxt(HEART_CSV, delimiter=",", names=True)
    data = np.array([list(r) for r in raw], dtype=float)
    _, first = np.unique(data, axis=0, return_index=True)
    data = data[np.sort(first)]  # drop the duplicate rows, as the notebook does
    cols = list(raw.dtype.names)
    i_age, i_sex, i_bp, i_chol, i_target = (cols.index(c) for c in ("age", "sex", "trestbps", "chol", "target"))

    by_key = {(r[0], r[1], r[2], r[3]): r[4] for r in uci}
    as_no_disease = as_disease = unmatched = 0
    for row in data:
        key = (int(row[i_age]), int(row[i_sex]), int(row[i_bp]), int(row[i_chol]))
        if key not in by_key:
            unmatched += 1
            continue
        uci_disease = by_key[key] > 0
        if (row[i_target] == 1) == uci_disease:
            as_disease += 1
        else:
            as_no_disease += 1

    print(f"de-duplicated rows in heart.csv: {len(data)} | unmatched: {unmatched}")
    print(f"target == 1 agrees with UCI 'no disease': {as_no_disease}")
    print(f"target == 1 agrees with UCI 'disease'   : {as_disease}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
