import gradio as gr
import pandas as pd
import numpy as np
import time
import re
from math import sqrt
from pypfopt import EfficientFrontier
from dimod import SimulatedAnnealingSampler, BinaryQuadraticModel

# -------------------------
# Backend: mocked AI + solvers
# -------------------------
def generate_ai_inputs(num_selected=4):
    MOCK_FULL_AI_OUTPUT = [
        {"ticker": "NVDA", "predicted_return": 0.045, "volatility": 0.035, "ai_score": 1.28},
        {"ticker": "MSFT", "predicted_return": 0.038, "volatility": 0.016, "ai_score": 2.37},
        {"ticker": "AAPL", "predicted_return": 0.032, "volatility": 0.018, "ai_score": 1.78},
        {"ticker": "GOOGL", "predicted_return": 0.035, "volatility": 0.019, "ai_score": 1.84},
        {"ticker": "COST", "predicted_return": 0.027, "volatility": 0.011, "ai_score": 2.45},
        {"ticker": "PEP", "predicted_return": 0.022, "volatility": 0.009, "ai_score": 2.44},
        {"ticker": "AMD", "predicted_return": 0.051, "volatility": 0.042, "ai_score": 1.21},
        {"ticker": "JPM", "predicted_return": 0.015, "volatility": 0.010, "ai_score": 1.50},
        {"ticker": "V", "predicted_return": 0.018, "volatility": 0.008, "ai_score": 2.25},
        {"ticker": "META", "predicted_return": 0.040, "volatility": 0.030, "ai_score": 1.33},
    ]
    df_full = pd.DataFrame(MOCK_FULL_AI_OUTPUT).sort_values(by="ai_score", ascending=False)
    df_selected = df_full.head(num_selected)
    ai_output_list = df_selected.to_dict("records")

    tickers = df_selected["ticker"].tolist()
    np.random.seed(42)
    mock_daily_returns = np.random.normal(
        loc=df_selected["predicted_return"].values / 252,
        scale=df_selected["volatility"].values / sqrt(252),
        size=(252, len(tickers)),
    )
    mock_daily_returns_df = pd.DataFrame(mock_daily_returns, columns=tickers)
    mu_series = pd.Series(df_selected["predicted_return"].values, index=tickers)
    return ai_output_list, mock_daily_returns_df, mu_series

def classical_portfolio_allocation(historical_data, mu_series, budget=2000):
    try:
        S = historical_data.cov() * 252
        ef = EfficientFrontier(mu_series, S)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        allocation = {t: round(w * budget, 2) for t, w in cleaned_weights.items() if w > 1e-4}
        perf = ef.portfolio_performance(verbose=False)
        metrics = {
            "Expected Annual Return": f"{perf[0]*100:.2f}%",
            "Annual Volatility": f"{perf[1]*100:.2f}%",
            "Sharpe Ratio": f"{perf[2]:.3f}",
            "Optimization Type": "Convex (Max Sharpe)",
        }
        return {"allocation": allocation, "performance": metrics}
    except Exception as e:
        return {"allocation": {}, "performance": {"Status": f"Classical solver error: {e}"}}

def quantum_portfolio_allocation_local(ai_output, budget=2000, max_assets=4):
    df = pd.DataFrame(ai_output)
    tickers = df["ticker"].tolist()
    mu = df["predicted_return"].values
    sigma = df["volatility"].values
    n = len(tickers)

    lambda_B = 100
    lambda_A = 1
    bqm = BinaryQuadraticModel("BINARY")

    # linear reward/penalty
    for i in range(n):
        linear_coeff = -mu[i] + lambda_A * (sigma[i] ** 2)
        bqm.add_variable(i, linear_coeff)

    # approximate cardinality penalty
    for i in range(n):
        bqm.add_linear(i, -2 * lambda_B * max_assets)
        for j in range(i + 1, n):
            bqm.add_quadratic(i, j, 2 * lambda_B)

    bqm.offset = lambda_B * max_assets ** 2

    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100)
    best_solution = sampleset.first.sample
    selected_indices = [i for i, v in best_solution.items() if v == 1]
    selected_tickers = [tickers[i] for i in selected_indices]

    if not selected_tickers:
        return {"allocation": {}, "performance": {"Status": "Solver Failed"}}

    sel = df[df["ticker"].isin(selected_tickers)]
    total_pred = sel["predicted_return"].sum()
    if total_pred == 0:
        return {"allocation": {}, "performance": {"Status": "Zero predicted return sum"}}

    weights = sel["predicted_return"] / total_pred
    allocation = {t: round(w * budget, 2) for t, w in zip(selected_tickers, weights)}

    # mock metrics
    try:
        mock_data = pd.DataFrame(np.random.rand(10, len(selected_tickers)), columns=selected_tickers)
        S = mock_data.cov() * 252
        mu_mock = pd.Series(sel["predicted_return"].values, index=selected_tickers)
        ef = EfficientFrontier(mu_mock, S)
        ef.max_sharpe()
        perf = ef.portfolio_performance(verbose=False)
        metrics = {
            "Expected Annual Return": f"{perf[0]*100:.2f}%",
            "Annual Volatility": f"{perf[1]*100:.2f}%",
            "Sharpe Ratio": f"{perf[2]:.3f}",
            "Optimization Type": "QUBO (Simulated Annealing)",
            "Assets Selected (K)": len(selected_tickers),
        }
    except Exception:
        metrics = {
            "Expected Annual Return": "N/A",
            "Annual Volatility": "N/A",
            "Sharpe Ratio": "N/A",
            "Optimization Type": "QUBO (Simulated Annealing)",
            "Assets Selected (K)": len(selected_tickers),
        }

    return {"allocation": allocation, "performance": metrics}

# -------------------------
# Helpers
# -------------------------
def simple_financial_health(income, expenses, savings):
    try:
        income = float(income)
        expenses = float(expenses)
        savings = float(savings)
        if income <= 0:
            return 20, 0.0
        emergency_ratio = savings / (expenses * 3 + 1e-9)
        score = min(100, max(10, int(50 * emergency_ratio + (income - expenses) / income * 50)))
        disposable = max(0.0, income - expenses)
        invest_pct = 0.05 + (min(score, 100) / 100) * 0.25
        recommended = round(disposable * invest_pct, 2)
        return score, recommended
    except Exception:
        return 30, 0.0

def robust_parse_budget(budget_input):
    try:
        if isinstance(budget_input, (int, float)):
            return float(budget_input)
        s = str(budget_input)
        s = s.replace("$", "").replace(",", "").strip()
        s = re.sub(r"[^\d.]", "", s)
        return float(s) if s else 0.0
    except:
        return 0.0

# -------------------------
# Main advisor logic
# -------------------------
def finmind_advisor(budget_input, mode, k, profile, income, expenses, savings, show_explain):
    start = time.time()

    total_budget = robust_parse_budget(budget_input)
    fin_score, rec_month = simple_financial_health(income or 0, expenses or 0, savings or 0)
    if total_budget <= 0 and rec_month > 0:
        total_budget = rec_month * 12

    scale = {"Conservative": 0.9, "Balanced": 1.0, "Aggressive": 1.1}.get(profile, 1.0)

    ai_out, hist, mu = generate_ai_inputs(num_selected=k)
    mu = mu * scale
    for item in ai_out:
        item["predicted_return"] = item.get("predicted_return", 0) * scale

    if mode == "Classical (Max Sharpe)":
        result = classical_portfolio_allocation(hist, mu, total_budget)
        engine = "Classical (Max Sharpe)"
    else:
        result = quantum_portfolio_allocation_local(ai_out, total_budget, k)
        engine = "Quantum (QUBO / SA)"

    if not result.get("allocation"):
        runtime = round(time.time() - start, 2)
        tech = f"Engine: {engine}. Filtered k={k}. Runtime {runtime}s. Status: no allocation"
        return f"⚠️ Optimization failed ({engine})", pd.DataFrame(), pd.DataFrame(), "Explainability unavailable", tech

    # Build allocations dataframe
    rows = []
    total_alloc = 0.0
    for t, amt in result["allocation"].items():
        ai_item = next((x for x in ai_out if x["ticker"] == t), {})
        pred = ai_item.get("predicted_return", 0)
        aisc = ai_item.get("ai_score", 0)
        rows.append({"Ticker": t, "Weight(%)": (amt / total_budget) * 100 if total_budget else 0, "Amount($)": amt, "Predicted Return": pred, "AI Score": aisc})
        total_alloc += amt

    df = pd.DataFrame(rows)
    df_display = df.copy()
    if not df_display.empty:
        df_display["Weight(%)"] = df_display["Weight(%)"].round(2).astype(str) + "%"
        df_display["Predicted Return"] = (df_display["Predicted Return"] * 100).round(2).astype(str) + "%"
        df_display["Amount($)"] = df_display["Amount($)"].apply(lambda x: f"${x:,.2f}")

    metrics_df = pd.DataFrame(result["performance"].items(), columns=["Metric", "Value"])
    runtime = round(time.time() - start, 2)

    summary = (
        f"### ✅ FinMind Analysis Complete\n"
        f"Engine: {engine}\n"
        f"Budget: ${total_budget:,.2f} (Allocated: ${total_alloc:,.2f})\n"
        f"Financial Health Score: {fin_score}/100  •  Recommended Monthly: ${rec_month:.2f}\n"
        f"Profile: {profile}\nRuntime: {runtime}s"
    )

    if show_explain:
        explain_lines = ["#### 🧠 Why these stocks were chosen"]
        for _, r in df.iterrows():
            explain_lines.append(f"• **{r['Ticker']}** chosen: AI Score {r['AI Score']}, Pred. Return {(r['Predicted Return']*100):.2f}%")
        explain_text = "\n".join(explain_lines)
    else:
        explain_text = "Explainability is hidden. Toggle to show the AI reasoning."

    tech_text = f"Filtered to top {k} by AI score. Engine: {engine}. Runtime: {runtime}s."

    return summary, df_display, metrics_df, explain_text, tech_text

# -------------------------
# CSS (polish)
# -------------------------
css = """
/* simple modern fintech look */
body { background: #071019; color: #e6eef1; }
.gradio-container { padding: 18px; }
.card { background: #0f1720; border-radius: 10px; padding: 12px; }
button { background: linear-gradient(135deg,#00c9a7 0%,#92fe9d 100%) !important; color: #06201f !important; font-weight:700 !important; border-radius:10px !important; }
button:hover { transform: scale(1.02); }
"""

# -------------------------
# Gradio UI (complete)
# -------------------------
with gr.Blocks(css=css) as app:
    gr.Markdown("<h2 style='color: #ffd166'>🚀 FinMind: Talk. Plan. Invest. Smarter</h2>")
    gr.Markdown("AI-driven, hybrid classical+quantum portfolio optimization prototype")

    with gr.Row():
        with gr.Column(scale=1):
            # Inputs
            budget = gr.Number(label="💰 Investment ($)", value=2000)
            with gr.Accordion("Financial Health Quick Check (optional)", open=False):
                income = gr.Number(label="Monthly Income ($)", value=5000)
                expenses = gr.Number(label="Monthly Expenses ($)", value=3000)
                savings = gr.Number(label="Liquid Savings ($)", value=5000)

            profile = gr.Radio(["Conservative", "Balanced", "Aggressive"], label="Investor Profile", value="Balanced")
            mode = gr.Radio(["Classical (Max Sharpe)", "Quantum (QUBO / SA)"], label="Select Optimization Engine", value="Classical (Max Sharpe)")
            k = gr.Slider(minimum=3, maximum=8, value=4, step=1, label="Max Assets to Select (K)")
            show_explain_toggle = gr.Checkbox(label="Show AI Explanation", value=True)

            with gr.Row():
                run_btn = gr.Button("🚀 Run FinMind Advisor", variant="primary")
                clear_btn = gr.Button("Clear inputs")

            # Quick action example buttons (autofill)
            gr.Markdown("#### Quick Examples")
            with gr.Row():
                ex1 = gr.Button("Try $2000 (Classical)")
                ex2 = gr.Button("Try $5000 (Quantum)")
                ex3 = gr.Button("Try $10000 (Balanced)")

        with gr.Column(scale=1):
            # Outputs (declared before use)
            summary_output = gr.Markdown()
            with gr.Accordion("AI Insights: Why these stocks were chosen", open=False):
                explain_output = gr.Markdown()
            with gr.Accordion("Allocation Table", open=True):
                allocation_output = gr.Dataframe(headers=["Ticker", "Weight(%)", "Amount($)", "Predicted Return", "AI Score"])
            with gr.Accordion("Performance & Risk Metrics", open=False):
                metrics_output = gr.Dataframe(headers=["Metric", "Value"])
            with gr.Accordion("Technical Details", open=False):
                technical_output = gr.Markdown()

    # Quick examples wiring (note values match component types)
    ex1.click(fn=lambda: (2000,  "Classical (Max Sharpe)", 3, "Balanced", 5000, 3000, 5000, True),
              inputs=None,
              outputs=[budget, mode, k, profile, income, expenses, savings, show_explain_toggle])

    ex2.click(fn=lambda: (5000, "Quantum (QUBO / SA)", 4, "Aggressive", 7000, 3000, 10000, True),
              inputs=None,
              outputs=[budget, mode, k, profile, income, expenses, savings, show_explain_toggle])

    ex3.click(fn=lambda: (10000, "Classical (Max Sharpe)", 5, "Balanced", 9000, 4000, 15000, True),
              inputs=None,
              outputs=[budget, mode, k, profile, income, expenses, savings, show_explain_toggle])

    # Clear button behavior
    def clear_inputs():
        return 2000, "Classical (Max Sharpe)", 4, "Balanced", 5000, 3000, 5000, True

    clear_btn.click(fn=clear_inputs, inputs=None, outputs=[budget, mode, k, profile, income, expenses, savings, show_explain_toggle])

    # Run button action
    run_btn.click(
        fn=finmind_advisor,
        inputs=[budget, mode, k, profile, income, expenses, savings, show_explain_toggle],
        outputs=[summary_output, allocation_output, metrics_output, explain_output, technical_output]
    )

if __name__ == "__main__":
    app.launch()
