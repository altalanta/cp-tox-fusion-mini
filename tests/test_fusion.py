import numpy as np
import pandas as pd

from src.fuse_modalities import viability_labels, group_split


def test_viability_labels_marks_low_counts():
    df = pd.DataFrame(
        {
            "plate": ["Week1", "Week1"],
            "compound_id": ["000001", "000002"],
            "well": ["A01", "A02"],
            "cell_count": [50, 200],
        }
    )
    labelled = viability_labels(df, control_ids=["000002"])
    assert labelled.loc[labelled["compound_id"] == "000001", "viability_label"].iloc[0] == 1
    assert labelled.loc[labelled["compound_id"] == "000002", "viability_label"].iloc[0] == 0


def test_group_split_uses_all_groups():
    df = pd.DataFrame({"compound_id": ["c1", "c1", "c2", "c3"], "value": np.arange(4)})
    splits = group_split(df, "compound_id", random_state=0)
    all_ids = set().union(*splits.values())
    assert set(df["compound_id"].unique()) == all_ids
    assert len(set(splits["train"]).intersection(splits["val"])) == 0
