import pandas as pd
import numpy as np
import time
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

def calculate_quantum_metrics(weights, mu, ai_output, budget, total_return):
    """Calculates financial metrics for the portfolio selected by the quantum model."""

    # Detect tickers in ai_output entries: items may have 'ticker' or 'Ticker'
    selected_tickers = list(weights.keys())

    selected_volatilities = []
    for tick in selected_tickers:
        vol = None
        for item in ai_output:
            # robust matching for 'ticker' or 'Ticker'
            if ('ticker' in item and item['ticker'] == tick) or ('Ticker' in item and item['Ticker'] == tick):
                vol_key = 'volatility' if 'volatility' in item else 'Volatility' if 'Volatility' in item else None
                if vol_key:
                    vol = item[vol_key]
                break
        # if not found, fallback to a small volatility to avoid divide by zero
        if vol is None:
            vol = 0.01
        selected_volatilities.append(vol)

    # Simple portfolio volatility proxy: weighted average volatility
    weights_frac = [weights[t] / budget for t in selected_tickers] if budget > 0 else [1/len(selected_tickers)]*len(selected_tickers)
    expected_volatility = np.average(selected_volatilities, weights=weights_frac)

    # Calculate Sharpe Ratio
    sharpe_ratio = total_return / expected_volatility if expected_volatility > 0 else 0

    return {
        "Expected Return": f"{total_return * 100:.2f}%",
        "Volatility (Risk)": f"{expected_volatility * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.3f}",
        "Total Allocated": f"${budget:,.2f}",
        "Engine Time (s)": 0.00
    }


def quantum_portfolio_allocation_local(ai_output: list, budget: float, max_assets: int):
    """
    Formulates the portfolio selection problem as a QUBO and solves it using
    a Simulated Annealer.
    """

    start_time = time.time()

    # Internal scaling to avoid instability for tiny demo budgets
    effective_budget = budget
    DEMO_INTERNAL_MIN = 1000.0
    use_internal_scaling = False
    if budget < DEMO_INTERNAL_MIN:
        effective_budget = DEMO_INTERNAL_MIN
        use_internal_scaling = True

    df = pd.DataFrame(ai_output)
    # allow either 'ticker' or 'Ticker' column
    if 'ticker' not in df.columns and 'Ticker' in df.columns:
        df = df.rename(columns={'Ticker': 'ticker'})
    tickers = df['ticker'].tolist()
    num_stocks = len(tickers)

    mu = df['predicted_return'].values
    sigma = df['volatility'].values

    A = 5.0
    P = 10.0

    bqm = BinaryQuadraticModel('BINARY')

    # Risk quadratic terms
    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):
            bqm.add_interaction(i, j, A * sigma[i] * sigma[j])

    # Return linear terms
    for i in range(num_stocks):
        bqm.add_variable(i, -mu[i])

    # Cardinality constraint linear correction
    for i in range(num_stocks):
        # add -2*P*K times the existing linear term (which is -mu[i])
        linear_coeff = bqm.linear.get(i, 0.0)
        bqm.add_variable(i, -2 * P * max_assets * linear_coeff)

    # Cardinality quadratic terms
    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):
            bqm.add_interaction(i, j, P)

    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100)

    best_solution = sampleset.first.sample

    selected_stocks = [tickers[i] for i, v in best_solution.items() if v == 1]

    if not selected_stocks:
        selected_stocks = [tickers[np.argmax(mu)]]

    selected_df = df[df['ticker'].isin(selected_stocks)].copy()

    total_selected_return = selected_df['predicted_return'].sum()
    if total_selected_return > 0:
        selected_df['weight'] = selected_df['predicted_return'] / total_selected_return
    else:
        selected_df['weight'] = 1 / len(selected_stocks)

    # Use effective_budget for stable internal amounts, then scale back if needed
    selected_df['amount_internal'] = selected_df['weight'] * effective_budget

    if use_internal_scaling and effective_budget != 0:
        scale_factor = budget / effective_budget
        selected_df['amount'] = selected_df['amount_internal'] * scale_factor
    else:
        selected_df['amount'] = selected_df['amount_internal']

    dollar_allocation = {row['ticker']: round(row['amount'], 2) for index, row in selected_df.iterrows()}

    total_portfolio_return = np.dot(selected_df['weight'].values, selected_df['predicted_return'].values)

    performance = calculate_quantum_metrics(
        dollar_allocation, mu, ai_output, budget, total_portfolio_return
    )
    performance["Engine Time (s)"] = f"{(time.time() - start_time):.4f}"

    return {
        "allocation": dollar_allocation,
        "performance": performance
    }
