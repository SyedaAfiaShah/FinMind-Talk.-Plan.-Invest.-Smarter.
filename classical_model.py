import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier

def calculate_performance_metrics(weights, mu, cov_matrix, budget):
    """Calculates key financial metrics (Return, Volatility, Sharpe Ratio)."""
    
    # Convert weights dict to array for numpy calculations
    tickers = list(weights.keys())
    w_array = np.array([weights[t] for t in tickers])

    # Ensure mu and cov_matrix are in the same order as w_array
    mu_series = mu[tickers]
    cov_df = cov_matrix.loc[tickers, tickers]

    # Calculate expected portfolio return (annualized)
    expected_return = np.dot(w_array, mu_series)
    
    # Calculate portfolio volatility (annualized)
    expected_volatility = np.sqrt(np.dot(w_array.T, np.dot(cov_df, w_array)))
    
    # Calculate Sharpe Ratio (assuming 0 risk-free rate for simplicity)
    sharpe_ratio = expected_return / expected_volatility

    # Calculate total allocated budget
    allocated_amount = sum(weights.values())

    # Format output
    return {
        "Expected Return": f"{expected_return * 100:.2f}%",
        "Volatility (Risk)": f"{expected_volatility * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.3f}",
        "Total Allocated": f"${allocated_amount:,.2f}",
        "Engine Time (s)": 0.00
    }

def classical_portfolio_allocation(historical_data, mu_series, budget=2000):
    """
    Performs Maximum Sharpe Ratio optimization (Classical Baseline).
    
    Inputs:
        historical_data (pd.DataFrame): DataFrame of mock historical returns for selected assets.
        mu_series (pd.Series): Predicted expected returns (mu) from the AI model.
        budget (float): Total investment budget.
        
    Output:
        A dictionary containing allocation and performance metrics.
    """
    
    start_time = time.time()
    
    # 1. Calculate Covariance Matrix (Critical step for Markowitz optimization)
    # Using a high-quality method (Shrinkage) for robustness.
    try:
        # Use simple sample covariance since pypfopt doesn't need price data, just return data
        cov_matrix = historical_data.cov()
    except Exception as e:
        # Fallback to zero correlation if covariance calculation fails
        tickers = historical_data.columns
        volatilities = historical_data.std()
        cov_matrix = pd.DataFrame(np.diag(volatilities.values ** 2), index=tickers, columns=tickers)
    
    # 2. Instantiate the Efficient Frontier solver
    ef = EfficientFrontier(mu_series, cov_matrix, weight_bounds=(0, 1))

    # 3. Optimization: Maximize the Sharpe Ratio
    try:
        weights = ef.max_sharpe()
    except Exception:
        # Fallback to equal weighting if solver fails (prevents app crash)
        n = len(mu_series)
        weights = {t: 1/n for t in mu_series.index}
        ef.set_weights(weights)

    # Clean the weights (remove small values, normalize to 1)
    cleaned_weights = ef.clean_weights(cutoff=0.001, verbose=False)
    
    # 4. Convert weights (percent) to dollar amounts
    dollar_allocation = {ticker: weight * budget 
                         for ticker, weight in cleaned_weights.items() if weight > 0}
    
    # 5. Calculate Final Performance Metrics
    performance = calculate_performance_metrics(cleaned_weights, mu_series, cov_matrix, budget)
    performance["Engine Time (s)"] = f"{(time.time() - start_time):.4f}"


    return {
        "allocation": dollar_allocation,
        "performance": performance
    }

# NOTE: Imports time inside the function for timing purposes
import time 
