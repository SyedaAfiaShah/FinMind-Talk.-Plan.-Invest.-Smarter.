import time
import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier


def calculate_performance_metrics(weights, mu, cov_matrix, budget):
    """Calculates key financial metrics (Return, Volatility, Sharpe Ratio)."""

    # Convert weights dict to array for numpy calculations
    tickers = list(weights.keys())
    w_vals = [weights[t] for t in tickers]
    w_array = np.array(w_vals)

    # Ensure mu and cov_matrix are in the same order as w_array
    mu_series = mu[tickers]
    cov_df = cov_matrix.loc[tickers, tickers]

    # Calculate expected portfolio return (annualized)
    expected_return = np.dot(w_array, mu_series)

    # Calculate portfolio volatility (annualized)
    expected_volatility = np.sqrt(np.dot(w_array.T, np.dot(cov_df, w_array)))

    # Calculate Sharpe Ratio (assuming 0 risk-free rate for simplicity)
    sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0.0

    # Determine total allocated amount. If weights are fractional (sum approx 1),
    # report the provided budget as total allocated. If weights are dollar amounts,
    # sum them directly.
    total_weights = sum(w_vals)
    if abs(total_weights - 1.0) < 1e-6:
        allocated_amount = budget
    else:
        # weights appear to be dollar amounts already
        allocated_amount = total_weights

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
    """

    start_time = time.time()

    # If budget is very small for demo, use an internal effective budget to keep numerical stability
    # but always return allocations scaled to the user's actual budget.
    effective_budget = budget
    DEMO_INTERNAL_MIN = 1000.0
    use_internal_scaling = False
    if budget < DEMO_INTERNAL_MIN:
        effective_budget = DEMO_INTERNAL_MIN
        use_internal_scaling = True

    # 1. Calculate Covariance Matrix
    try:
        cov_matrix = historical_data.cov()
    except Exception as e:
        tickers = historical_data.columns
        volatilities = historical_data.std()
        cov_matrix = pd.DataFrame(np.diag(volatilities.values ** 2), index=tickers, columns=tickers)

    # 2. Instantiate the Efficient Frontier solver
    ef = EfficientFrontier(mu_series, cov_matrix, weight_bounds=(0, 1))

    # 3. Optimization: Maximize the Sharpe Ratio
    try:
        weights = ef.max_sharpe()
    except Exception:
        # Fallback to equal weighting if solver fails
        n = len(mu_series)
        weights = {t: 1/n for t in mu_series.index}
        ef.set_weights(weights)

    # Clean the weights (remove small values, normalize to 1)
    cleaned_weights = ef.clean_weights(cutoff=0.001, verbose=False)

    # 4. Convert weights (fraction) to dollar amounts using effective_budget
    dollar_allocation_internal = {ticker: weight * effective_budget
                                  for ticker, weight in cleaned_weights.items() if weight > 0}

    # If we used internal scaling, scale dollar allocations back to the real user budget
    if use_internal_scaling and effective_budget != 0:
        scale_factor = budget / effective_budget
        dollar_allocation = {t: round(a * scale_factor, 2) for t, a in dollar_allocation_internal.items()}
    else:
        dollar_allocation = {t: round(a, 2) for t, a in dollar_allocation_internal.items()}

    # 5. Calculate Final Performance Metrics (pass fractional weights so expected return is correct)
    performance = calculate_performance_metrics(cleaned_weights, mu_series, cov_matrix, budget)
    performance["Engine Time (s)"] = f"{(time.time() - start_time):.4f}"

    return {
        "allocation": dollar_allocation,
        "performance": performance
    }

