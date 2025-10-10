import pandas as pd
import numpy as np
from io import StringIO
import random
from pypfopt import risk_models

# --- CACHED AI PREDICTION DATA ---
# This simulates the final output of your trained LSTM/XGBoost models 
# for a set of high-potential assets.
AI_PREDICTIONS_CSV_CONTENT = """
Ticker,Predicted_Return,Volatility,AI_Score
MSFT,0.032,0.012,2.67
VRTX,0.025,0.010,2.50
AAPL,0.035,0.015,2.33
GOOG,0.028,0.014,2.0
AMZN,0.040,0.020,2.0
NVDA,0.050,0.025,2.0
COST,0.015,0.008,1.88
WMT,0.012,0.007,1.71
TSLA,0.060,0.040,1.5
JPM,0.020,0.013,1.54
"""

# --- CACHING AND FILTERING FUNCTION ---
def generate_ai_inputs(num_selected: int = 4):
    """
    Simulates the AI Model's job by loading cached predictions,
    performing the final pre-selection, and generating data structures 
    needed by the Classical and Quantum optimizers.
    """
    
    # Load the predictions from the mock CSV content
    ai_output_df = pd.read_csv(StringIO(AI_PREDICTIONS_CSV_CONTENT))

    # 1. Pre-selection: Sort by the AI score and select the top K assets
    ai_output_df = ai_output_df.sort_values(
        by='AI_Score', ascending=False
    ).head(num_selected).reset_index(drop=True)
    
    # Convert to the list-of-dicts format expected by the quantum optimizer
    ai_output = ai_output_df.rename(columns={'Predicted_Return': 'predicted_return', 
                                             'Volatility': 'volatility', 
                                             'AI_Score': 'ai_score'}).to_dict('records')
    tickers = ai_output_df['Ticker'].tolist()

    # 2. Generate Mock Historical Data for Covariance (50 days of data)
    # This is a synthetic dataset based on the AI's volatility and return estimates.
    days = 50
    returns_matrix = np.zeros((days, num_selected))
    
    for i, data in enumerate(ai_output):
        # Center random noise around the predicted return, scaled by volatility
        vol = data['volatility']
        returns_matrix[:, i] = np.random.normal(0, vol, days)
        returns_matrix[:, i] += data['predicted_return'] / days 

    mock_historical_data = pd.DataFrame(returns_matrix, columns=tickers)
    
    # Create the predicted mu series (Expected Return Vector)
    mu_series = pd.Series(ai_output_df['Predicted_Return'].values, index=tickers)

    return ai_output, mock_historical_data, mu_series
