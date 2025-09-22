
# starter_hull_model.py
# Baseline implementation for Hull Tactical Market Prediction
# - Stateful inference with Kaggle evaluation API
# - Defensive feature/target inference
# - Lightweight ridge-like regression (no sklearn)
# - Volatility targeting using lagged_forward_returns if available
#
# Place this file in your Kaggle Notebook and run.
# It will create /kaggle/working/submission.parquet when using the local gateway.

import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import polars as pl

import sys
from pathlib import Path

try:  # default_inference_server is only available in the Kaggle environment
    from default_inference_server import DefaultInferenceServer  # type: ignore
except ModuleNotFoundError:
    DefaultInferenceServer = None  # type: ignore[assignment]
    server = None
else:  # pragma: no cover - requires Kaggle runtime
    server = DefaultInferenceServer()
    server.run_for_test(("/kaggle/input/hull-tactical-market-prediction",))

EVAL_ROOT = Path('/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation')
sys.path.append(str(EVAL_ROOT))

WORK = Path('/kaggle/working')
sys.path.append(str(WORK))

os.environ['HULL_DATA_DIR'] = '/kaggle/input/hull-tactical-market-prediction'
os.environ['HULL_MODEL_PATH'] = '/kaggle/working/model_baseline.pkl'

try:
    import kaggle_evaluation.default_inference_server
except Exception:  # pragma: no cover
    kaggle_evaluation = None

KAGGLE_INPUT_DIR = Path("/kaggle/input/hull-tactical-market-prediction/")
KAGGLE_WORKING_DIR = Path("/kaggle/working/")
SUBMISSION_PATH = KAGGLE_WORKING_DIR / "submission.parquet"


def _to_numpy(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 1), dtype=np.float32)
    return df.select(cols).to_numpy().astype(np.float32, copy=False)


def _infer_cols(df: pl.DataFrame) -> Tuple[List[str], Optional[str], Optional[str]]:
    exclude = {"date_id", "is_scored", "row_id", "time_id", "timestamp", "allocation"}
    possible_target_names = ["forward_returns", "target"]
    all_cols = df.columns

    # FIX: check numeric types using is_numeric_dtype
    numeric_cols = [
        c for c, dt in zip(all_cols, df.dtypes)
        if pl.datatypes.is_numeric_dtype(dt)
    ]

    feat_cols = [c for c in numeric_cols if c not in exclude and c not in possible_target_names]
    target_col = None
    for name in possible_target_names:
        if name in all_cols:
            target_col = name
            break
    row_id_col = "row_id" if "row_id" in all_cols else None
    return feat_cols, target_col, row_id_col


class OnlineStandardizer:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.eps = 1e-6
    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + self.eps
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            return X
        return (X - self.mean_) / self.std_
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class RidgeLikeRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape
        X_aug = np.hstack([X, np.ones((n, 1), dtype=X.dtype)])
        reg = self.alpha * np.eye(d + 1, dtype=X.dtype)
        reg[-1, -1] = 0.0
        XtX = X_aug.T @ X_aug + reg
        Xty = X_aug.T @ y.reshape(-1, 1)
        w = np.linalg.solve(XtX, Xty).ravel()
        self.coef_ = w[:d]
        self.intercept_ = float(w[-1])
        return self
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.coef_ is None:
            return np.zeros(X.shape[0], dtype=X.dtype)
        return X @ self.coef_.astype(X.dtype, copy=False) + self.intercept_


class VolTarget:
    def __init__(self, span: int = 60, min_vol: float = 1e-4, max_leverage: float = 2.0):
        self.span = span
        self.min_vol = float(min_vol)
        self.max_leverage = float(max_leverage)
        self.ewm_var: Optional[float] = None
        self.alpha = 2.0 / (span + 1.0)
        self.target_vol = 0.01  # ~1% daily target
    def update_and_scale(self, base_signal: np.ndarray, proxy_ret: Optional[np.ndarray]) -> np.ndarray:
        if proxy_ret is not None:
            for r in proxy_ret:
                r2 = float(r) ** 2
                if self.ewm_var is None:
                    self.ewm_var = r2
                else:
                    self.ewm_var = (1 - self.alpha) * self.ewm_var + self.alpha * r2
        pred_vol = np.sqrt(self.ewm_var) if self.ewm_var is not None else self.target_vol
        scale = self.target_vol / max(pred_vol, self.min_vol)
        alloc = base_signal * scale
        return np.clip(alloc, 0.0, self.max_leverage)


class StrategyModel:
    def __init__(self, alpha: float = 10.0):
        self.std = OnlineStandardizer()
        self.model = RidgeLikeRegressor(alpha=alpha)
        self.feat_cols: List[str] = []
        self.row_id_col: Optional[str] = None
        self.proxy_col: Optional[str] = None
        self.voltgt = VolTarget(span=60, max_leverage=2.0)
        self.fitted = False
    def fit_from_train(self, train_df: pl.DataFrame):
        feat_cols, target_col, row_id_col = _infer_cols(train_df)
        self.feat_cols = feat_cols
        self.row_id_col = row_id_col
        if "lagged_forward_returns" in train_df.columns:
            self.proxy_col = "lagged_forward_returns"
        else:
            for c in train_df.columns:
                if "lag" in c and "ret" in c:
                    self.proxy_col = c
                    break
        if target_col is not None and target_col in train_df.columns:
            X = _to_numpy(train_df, feat_cols)
            Xs = self.std.fit_transform(X)
            y = train_df[target_col].to_numpy().astype(np.float32, copy=False)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            self.model.fit(Xs, y)
            self.fitted = True
        else:
            self.std.fit(np.zeros((1, 1), dtype=np.float32))
            self.model.coef_ = np.zeros((len(feat_cols) if feat_cols else 1,), dtype=np.float32)
            self.model.intercept_ = 0.0
            self.fitted = True
    def _base_signal(self, X_batch: np.ndarray) -> np.ndarray:
        raw = self.model.predict(self.std.transform(X_batch))
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        base_long = (np.tanh(raw) + 1.0)  # in [0, 2]
        return base_long
    def predict_allocations(self, batch: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        Xb = _to_numpy(batch, self.feat_cols)
        base_signal = self._base_signal(Xb)
        proxy_arr = None
        if self.proxy_col and self.proxy_col in batch.columns:
            proxy_arr = batch[self.proxy_col].to_numpy().astype(np.float32, copy=False)
        alloc = self.voltgt.update_and_scale(base_signal.astype(np.float32, copy=False), proxy_arr)
        if "row_id" in batch.columns:
            out = pl.DataFrame({"row_id": batch["row_id"], "allocation": alloc})
        else:
            out = pl.DataFrame({"allocation": alloc})
        return out


_MODEL: Optional[StrategyModel] = None

def _load_train_dataframe() -> Optional[pl.DataFrame]:
    try:
        path_csv = KAGGLE_INPUT_DIR / "train.csv"
        if path_csv.exists():
            return pl.read_csv(path_csv, low_memory=True)
    except Exception:
        pass
    return None

def _ensure_model_initialized(test_batch: pl.DataFrame):
    global _MODEL
    if _MODEL is not None:
        return
    _MODEL = StrategyModel(alpha=10.0)
    train_df = _load_train_dataframe()
    if train_df is not None:
        _MODEL.fit_from_train(train_df)
    else:
        fake = pl.DataFrame({c: np.zeros(1) for c in test_batch.columns if pl.datatypes.is_numeric(test_batch.schema[c])})
        _MODEL.fit_from_train(fake)

def predict(test: pl.DataFrame):
    _ensure_model_initialized(test)
    out = _MODEL.predict_allocations(test)
    try:
        if len(out) > 0:
            os.makedirs(KAGGLE_WORKING_DIR, exist_ok=True)
            out.write_parquet(SUBMISSION_PATH)
    except Exception:
        pass
    return out

def _serve_or_run_local():
    try:
        inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)  # type: ignore[attr-defined]
    except Exception:
        fake = pl.DataFrame({
            "row_id": [0, 1, 2, 3],
            "lagged_forward_returns": [0.001, -0.0005, 0.002, -0.0015],
            "feature_a": [1.0, 0.5, -0.1, 0.0],
            "feature_b": [10.0, 11.0, 9.5, 10.2],
        })
        out = predict(fake)
        print(out)
        return
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway((str(KAGGLE_INPUT_DIR),))

if __name__ == "__main__":
    _serve_or_run_local()
