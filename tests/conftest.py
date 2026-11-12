from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hsboa.data import CLASSES, FEATURE_COLUMNS, FILES, LABEL_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_TRAIN = PROJECT_ROOT / FILES["train"]

# Class sizes mimic UNSW-NB15 imbalance (Worms rarest); all multiples of 5 so a
# 20% stratified split is exact.
_SYNTH_SIZES = dict(zip(CLASSES, (300, 200, 150, 100, 60, 50, 20, 20, 15, 10)))


def _synthetic_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = sum(_SYNTH_SIZES.values())
    df = pd.DataFrame({"id": np.arange(1, n + 1)})
    for c in FEATURE_COLUMNS:
        df[c] = rng.random(n)
    df["proto"] = rng.choice(["tcp", "udp", "arp"], n)
    df["service"] = rng.choice(["-", "http", "dns", "ftp"], n)
    df["state"] = rng.choice(["FIN", "INT", "CON"], n)
    labels = np.repeat(list(_SYNTH_SIZES), list(_SYNTH_SIZES.values()))
    df[LABEL_COLUMN] = labels
    df["label"] = (labels != "Normal").astype(int)
    # Make two features informative so selection and ordering are testable.
    codes = pd.Series(labels).map({k: i for i, k in enumerate(_SYNTH_SIZES)}).to_numpy()
    df["sbytes"] = codes + 0.05 * rng.random(n)
    df["ct_srv_src"] = codes * 0.5 + 0.3 * rng.random(n)
    return df


@pytest.fixture
def synthetic_file(tmp_path) -> Path:
    path = tmp_path / "synthetic.csv"
    _synthetic_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def real_train() -> Path:
    if not REAL_TRAIN.exists():
        pytest.skip(f"{REAL_TRAIN} not present")
    return REAL_TRAIN
