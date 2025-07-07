import numpy as np
import pickle
import os
import pandas as pd

def moneyline_to_prob(ml: float) -> float:
    """Convert moneyline into implied probability."""
    if ml < 0:
        return -ml / (-ml + 100)
    else:
        return 100 / (ml + 100)

def normalize_book_odds(p_book_raw: float) -> float:
    """Remove vig for a binary market (home vs away)."""
    p_raw   = moneyline_to_prob(p_book_raw)
    p_away  = 1 - p_raw
    total   = p_raw + p_away
    return p_raw / total

class BMAWeights:
    def __init__(self, num_models: int, alpha: float = 0.98,
                 prior: np.ndarray = np.array([1/3, 1/3, 1/3]), eps: float = 1e-3):
        self.alpha = alpha
        self.eps   = eps
        if prior is not None:
            assert len(prior) == num_models
            self.logw = np.log(prior)
        else:
            self.logw = np.log(np.ones(num_models) / num_models)
        self._renormalize()

    def _renormalize(self):
        self.logw -= np.max(self.logw)
        w = np.exp(self.logw)
        w = np.clip(w / w.sum(), self.eps, 1 - self.eps)
        self.w = w / w.sum()

    def update(self, y_true: int, p_preds: np.ndarray) -> np.ndarray:
        lik = np.where(y_true==1, p_preds, 1 - p_preds)
        self.logw = self.alpha * self.logw + np.log(lik + 1e-15)
        self._renormalize()
        return self.w

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'logw': self.logw,
                'alpha': self.alpha,
                'eps': self.eps
            }, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls(
            num_models=len(state['logw']),
            alpha=state['alpha'],
            prior=np.array([1/3, 1/3, 1/3]),
            eps=state['eps']
        )
        obj.logw = state['logw']
        obj._renormalize()
        return obj

class FairValueEngine:
    def __init__(self,
                 num_models: int = 3,
                 alpha: float = 0.98,
                 state_path: str = "bma_state.pkl"):
        self.state_path = state_path
        if os.path.exists(state_path):
            self.bma = BMAWeights.load(state_path)
        else:
            self.bma = BMAWeights(num_models=num_models, alpha=alpha)

    def update_from_history(self, history: list[tuple[int, list[float]]]):
        for y_true, p_list in history:
            preds = np.array(p_list)
            lik = preds if y_true==1 else (1 - preds)
            self.bma.update(y_true, lik)
        self.bma.save(self.state_path)

    def compute(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        df = signals_df.copy()
        df['p_book'] = df['p_book_raw'].apply(normalize_book_odds)
        def _fv(row):
            ps = np.array([row['p_xgb'],row['p_book'],row['p_polymarket']])
            return float(np.dot(self.bma.w, ps))
        df['fair_p'] = df.apply(_fv, axis=1)
        df['fair_odds'] = 1 / df['fair_p']
        return df 