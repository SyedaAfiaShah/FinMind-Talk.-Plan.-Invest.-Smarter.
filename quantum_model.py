import pandas as pd
import numpy as np
import time
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

def calculate_quantum_metrics(weights, mu, ai_output, budget, total_return):
    """Calculates financial metrics for the portfolio selected by the quantum model."""
    
    # Calculate portfolio volatility (We use the simple mean volatility of selected assets 
    # since the QUBO formulation simplifies covariance for tractability)
    selected_tickers = list(weights.keys())
    selected_volatilities = [
        item['volatility'] for item in ai_output if item['ticker'] in selected_tickers
    ]
    # Simple portfolio volatility proxy: weighted average volatility
    expected_volatility = np.average(selected_volatilities, weights=[weights[t]/budget for t in selected_tickers])
    
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
    D-Wave's local Simulated Annealer (proxy for Quantum Annealer).

    Inputs:
        ai_output (list): List of dicts with ticker, predicted_return, volatility, ai_score.
        budget (float): Total investment budget.
        max_assets (int): K, the hard constraint on the number of assets to select (Cardinality).
        
    Output:
        A dictionary containing allocation and performance metrics.
    """
    
    start_time = time.time()
    
    df = pd.DataFrame(ai_output)
    tickers = df['ticker'].tolist()
    num_stocks = len(tickers)
    
    # --- QUBO PARAMETERS ---
    # Convert returns to a series and create a simple volatility matrix for coupling terms
    mu = df['predicted_return'].values
    sigma = df['volatility'].values
    
    # A (Risk-Aversion Parameter): Controls the balance between risk and return.
    # Higher A means more risk aversion. We set it high to favor low risk.
    A = 5.0 
    
    # P (Penalty Parameter): Controls how strictly the constraint is enforced.
    # Must be larger than A * max_return * max_volatility^2 to dominate
    P = 10.0 
    
    # Initialize Binary Quadratic Model (BQM)
    bqm = BinaryQuadraticModel('BINARY')

    # --- 1. Objective Function (Minimize Risk - Maximize Return) ---
    # We want to MINIMIZE H = A * Risk - Return
    
    # Risk (Quadratic Term): Simple interaction based on volatilities (Proxy for covariance)
    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):
            # Q[i, j] = A * sigma_i * sigma_j
            bqm.add_interaction(i, j, A * sigma[i] * sigma[j])
            
    # Return (Linear Term): We negate returns because QUBO solves minimization problems
    for i in range(num_stocks):
        # Q[i, i] = -mu_i
        bqm.add_variable(i, -mu[i])
        
    # --- 2. Cardinality Constraint (Select Exactly K Assets) ---
    # H_constraint = P * (Sum(x_i) - K)^2
    # Expanding this results in linear and quadratic terms added to the BQM.
    
    # The linear term correction: -2*P*K*x_i
    for i in range(num_stocks):
        bqm.add_variable(i, -2 * P * max_assets * bqm.linear[i])

    # The quadratic term: P * x_i * x_j (for all i != j)
    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):
            bqm.add_interaction(i, j, P)

    # --- 3. Solve QUBO via Simulated Annealing (Hackathon Proxy) ---
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100) # Run 100 times for best result
    
    best_solution = sampleset.first.sample
    
    # --- 4. Post-Processing: Map Binary Result to Dollar Allocation ---
    selected_stocks = [tickers[i] for i, v in best_solution.items() if v == 1]
    
    # Fallback to equal weighting if constraint failure leads to zero selection
    if not selected_stocks:
        selected_stocks = [tickers[np.argmax(mu)]] # Select best predicted stock as last resort

    # Final Allocation Logic: Proportional weighting based on Predicted Return (AI Signal)
    # This ensures the quantum selection uses the AI's risk/return profile for weighting.
    selected_df = df[df['ticker'].isin(selected_stocks)].copy()
    
    # Calculate weights proportional to predicted return
    total_selected_return = selected_df['predicted_return'].sum()
    if total_selected_return > 0:
        selected_df['weight'] = selected_df['predicted_return'] / total_selected_return
    else:
        # If returns are all negative/zero, fall back to equal weighting among selected assets
        selected_df['weight'] = 1 / len(selected_stocks)
        
    # Convert weights to dollar amounts
    selected_df['amount'] = selected_df['weight'] * budget
    
    dollar_allocation = {row['ticker']: round(row['amount'], 2) for index, row in selected_df.iterrows()}
    
    # Calculate Total Portfolio Return (for metrics display)
    total_portfolio_return = np.dot(selected_df['weight'].values, selected_df['predicted_return'].values)

    # 5. Calculate Final Performance Metrics
    performance = calculate_quantum_metrics(
        dollar_allocation, mu, ai_output, budget, total_portfolio_return
    )
    performance["Engine Time (s)"] = f"{(time.time() - start_time):.4f}"

    return {
        "allocation": dollar_allocation,
        "performance": performance
    }
