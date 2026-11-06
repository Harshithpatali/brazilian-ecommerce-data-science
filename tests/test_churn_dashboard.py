import ast
from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1] / "app" / "pages" / "churn.py"


def test_dashboard_render_binds_test_metrics_before_using_them():
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    render = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render"
    )
    assigned_names = set()
    loaded_names = set()
    for node in ast.walk(render):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                assigned_names.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loaded_names.add(node.id)

    assert "test_metrics" in assigned_names
    assert "test_metrics" in loaded_names


def test_dashboard_uses_shared_metadata_contract_not_training_module():
    source = DASHBOARD.read_text(encoding="utf-8")
    assert "validate_model_metadata" in source
    assert "from src.models.train_churn import" not in source



def test_dashboard_uses_explicit_repeat_purchase_curve_threshold():
    source = DASHBOARD.read_text(encoding="utf-8")
    assert "repeat_purchase_probability_threshold" in source
    assert "cv_repeat_purchase_pr_auc" in source
    assert '"class_1_precision"' in source


def test_churn_probability_uses_class_label_not_probability_column_position():
    import numpy as np
    import pandas as pd

    from app.pages.churn import _predict_churn_probability

    class ReversedClassEstimator:
        classes_ = np.array([1, 0])

        def predict_proba(self, features):
            return np.array([[0.80, 0.20], [0.25, 0.75]])

    result = _predict_churn_probability(
        ReversedClassEstimator(), pd.DataFrame({"feature": [1, 2]})
    )
    np.testing.assert_allclose(result, [0.80, 0.25])


def test_churn_probability_rejects_unverifiable_class_order():
    import pandas as pd
    import pytest

    from app.pages.churn import _predict_churn_probability

    class InvalidEstimator:
        classes_ = ["repeat", "unknown"]

        def predict_proba(self, features):
            return [[0.5, 0.5]]

    with pytest.raises(ValueError, match="Expected binary churn classes"):
        _predict_churn_probability(
            InvalidEstimator(), pd.DataFrame({"feature": [1]})
        )
