import requests
import pandas as pd
import numpy as np
import pickle
from fairvalue_engine import FairValueEngine

DATA_PIPELINE_URL = "http://data-pipeline-9722-465efba1-egsa9f4a.onporter.run/api/v1/historical"
KALSHI_MARKET_URL = "https://trading-api.kalshi.com/trade-api/v2/markets/{}"  #(ticker)
POLYMARKET_MARKET_URL = "https://api.polymarket.com/v4/markets/{}"  #(slug)
XGB_MODEL_PATH = "xgb_pseudo_model.pkl"
BMA_STATE_PATH = "bma_state.pkl"

payload = {
    "start_date": "2025-06-03",
    "end_date": "2025-06-26",
    "interval": 10,
    "max_time_diff": 86400,
    "kalshi_tickers": ["KXUFCFIGHT-25JUN28PRISMI-SMI"],
    "polymarket_slugs": ["ufc-317-price-vs-smith-883"]
}
resp = requests.post(DATA_PIPELINE_URL, json=payload)
print("Status code:", resp.status_code)
print("Response text:", resp.text)
data = resp.json()

kalshi_tickers = data.get("kalshi_tickers", [])
polymarket_slugs = data.get("polymarket_slugs", [])

def get_kalshi_prob(ticker):
    url = KALSHI_MARKET_URL.format(ticker)
    resp = requests.get(url)
    d = resp.json()
    # Try to extract the 'yes' price. Adjust the key as needed based on actual API response.
    # Common fields: d['market']['yes_price'] or d['market']['last_trade_price']
    market = d.get('market', {})
    # Use 'last_trade_price' as a fallback if 'yes_price' is not available
    price = market.get('yes_price')
    if price is None:
        price = market.get('last_trade_price')
    if price is None:
        raise ValueError(f"Could not find a valid price for Kalshi ticker {ticker}")
    return float(price)

def get_polymarket_prob(slug):
    url = POLYMARKET_MARKET_URL.format(slug)
    resp = requests.get(url)
    d = resp.json()
    # Try to extract the 'yes' probability. Adjust the key as needed based on actual API response.
    # Example: d['market']['outcomes'][0]['probability'] or d['market']['yesPrice']
    market = d.get('market', {})
    # Try common keys
    prob = market.get('yesPrice')
    if prob is None:
        # Try outcomes[0]['probability']
        outcomes = market.get('outcomes', [])
        if outcomes and 'probability' in outcomes[0]:
            prob = outcomes[0]['probability']
    if prob is None:
        raise ValueError(f"Could not find a valid probability for Polymarket slug {slug}")
    return float(prob)

# --- 3. Load XGBoost model (if available) ---
try:
    with open(XGB_MODEL_PATH, 'rb') as f:
        xgb_model = pickle.load(f)
    def get_p_xgb():
        # Placeholder: no features, so just return 0.5
        # If you have features, use xgb_model.predict(DMatrix(features))
        return 0.5
except Exception:
    xgb_model = None
    def get_p_xgb():
        return 0.5

# --- 4. Assemble DataFrame ---
data_rows = []
for ticker, slug in zip(kalshi_tickers, polymarket_slugs):
    try:
        p_book_raw = get_kalshi_prob(ticker)
    except Exception as e:
        print(f"Error fetching Kalshi price for {ticker}: {e}")
        p_book_raw = np.nan
    try:
        p_polymarket = get_polymarket_prob(slug)
    except Exception as e:
        print(f"Error fetching Polymarket price for {slug}: {e}")
        p_polymarket = np.nan
    p_xgb = get_p_xgb()
    data_rows.append({
        'game_id': f'{ticker}|{slug}',
        'p_xgb': p_xgb,
        'p_book_raw': p_book_raw,
        'p_polymarket': p_polymarket
    })

signals_df = pd.DataFrame(data_rows)

# --- 5. Run FairValueEngine ---
engine = FairValueEngine(num_models=3, alpha=0.98, state_path=BMA_STATE_PATH)
result_df = engine.compute(signals_df)
print(result_df) 