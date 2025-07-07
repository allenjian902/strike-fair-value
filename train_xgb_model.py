import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle

def train_and_save_xgb_model(
    n_samples=500,
    model_path='xgb_pseudo_model.pkl',
    calibration_curve_path='calibration_curve.png',
    random_seed=42
):
    os.environ.pop('MPLBACKEND', None)
    np.random.seed(random_seed)
    df = pd.DataFrame({
        'team_stat_diff': np.random.normal(0, 1, n_samples),
        'rest_day_diff': np.random.randint(-1, 3, n_samples),
        'home_field_advantage': np.random.binomial(1, 0.6, n_samples),
    })
    logit = 0.5 * df['team_stat_diff'] + 0.3 * df['rest_day_diff'] + 0.8 * df['home_field_advantage']
    prob = 1 / (1 + np.exp(-logit))
    df['home_win'] = np.random.binomial(1, prob)

    TARGET = 'home_win'
    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=random_seed
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1,    
        'seed': random_seed
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    y_pred_val = model.predict(dval)
    print("Validation Log-loss:", log_loss(y_val, y_pred_val))
    print("ROC AUC:", roc_auc_score(y_val, y_pred_val))
    print("Brier Score:", brier_score_loss(y_val, y_pred_val))

    prob_true, prob_pred = calibration_curve(y_val, y_pred_val, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve on Pseudo Data')
    plt.savefig(calibration_curve_path)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Pseudo-model saved to {model_path}")

payload = {
    "start_date": "2025-06-03",
    "end_date": "2025-06-26",
    "interval": 10,
    "max_time_diff": 86400,
    "kalshi_tickers": ["KXUFCFIGHT-25JUN28PRISMI-SMI"],
    "polymarket_slugs": ["ufc-317-price-vs-smith-883"]
} 