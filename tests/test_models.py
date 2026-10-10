import numpy as np
import pandas as pd
import pytest

from src.models import FEATURE_SETS, FORBIDDEN_FEATURES, feature_columns, make_model


def _toy_frame():
    cols = ["txId", "time_step", "local_1", "local_2", "agg_1", "label"]
    return pd.DataFrame([[1, 1, 0.1, 0.2, 0.3, 0]], columns=cols)


def test_local_and_all_feature_sets():
    df = _toy_frame()
    assert feature_columns(df, "local") == ["local_1", "local_2"]
    assert feature_columns(df, "all") == ["local_1", "local_2", "agg_1"]


def test_forbidden_columns_are_never_features():
    df = _toy_frame()
    for fs in FEATURE_SETS:
        assert not set(feature_columns(df, fs)) & set(FORBIDDEN_FEATURES)


def test_unknown_feature_set_and_family_raise():
    with pytest.raises(ValueError):
        feature_columns(_toy_frame(), "graph")
    with pytest.raises(ValueError):
        make_model("svm", 0)


def test_missing_feature_columns_raise():
    with pytest.raises(ValueError):
        feature_columns(pd.DataFrame({"txId": [1], "time_step": [1], "label": [0]}), "local")


@pytest.mark.parametrize("family", ["lr", "rf"])
def test_models_fit_and_predict_probabilities(family):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + 0.2 * rng.normal(size=200) > 0.8).astype(int)
    model = make_model(family, 0).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (200,) and ((p >= 0) & (p <= 1)).all()