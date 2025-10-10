import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier
from pypfopt import plotting

def classical_portfolio_allocation(data_df, mu, budget=2000):
    """
    data_df: DataFrame of Historical Prices (needed for proper covariance).
    mu: pd.Series of predicted returns from AI model.
    budget: total investment amount.
    """
    
    # 1. Calculate REAL Covariance Matrix using Historical Data
    # Use a robust method like Ledoit-Wolf shrinkage for better stability
    S = risk_models.CovarianceShrinkage(data_df).ledoit_wolf()
    
    # Ensure mu and S are aligned
    tickers = mu.index.tolist()
    S_filtered = S.loc[tickers, tickers]

    # 2. Optimize Max Sharpe Ratio 
    ef = EfficientFrontier(mu, S_filtered, verbose=False)
    
    # Calculate Max Sharpe Ratio weights
    weights = ef.max_sharpe() 
    cleaned_weights = ef.clean_weights()

    # 3. Calculate metrics for the output log
    performance = ef.portfolio_performance(verbose=False)
    total_return = performance[0]
    annual_volatility = performance[1]
    sharpe_ratio = performance[2]

    # 4. Calculate allocation in $ based on budget
    allocation = {
        ticker: round(weight * budget, 2) 
        for ticker, weight in cleaned_weights.items() 
        if weight > 1e-4 # Only include significant allocations
    }
    
    return {
        "allocation": allocation,
        "performance": {
            "Sharpe Ratio": round(sharpe_ratio, 3),
            "Predicted Return": f"{round(total_return * 100, 2)}%",
            "Predicted Volatility": f"{round(annual_volatility * 100, 2)}%"
        }
    }

# This file now only contains the function, not the __main__ block
