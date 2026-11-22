from scripts.run_repeat_purchase_ablation import ANCHOR, feature_groups
from src.features.churn_features import FEATURE_COLS


def test_ablation_groups_use_only_known_features():
    groups = feature_groups()
    assert len(groups) >= 5
    for columns in groups.values():
        assert columns
        assert set(columns).issubset(FEATURE_COLS)
        assert ANCHOR in columns


def test_ablation_has_frequency_only_control_and_full_set():
    groups = feature_groups()
    assert groups["lifetime_frequency_only"] == [ANCHOR]
    assert set(groups["full_feature_set"]) == set(FEATURE_COLS)
    assert len(groups["full_feature_set"]) > len(groups["lifetime_frequency_only"])
