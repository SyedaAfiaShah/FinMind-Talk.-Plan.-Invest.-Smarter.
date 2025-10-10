import streamlit as st
import pandas as pd
import numpy as np
import time
import re  # Added for regex

# Import the core logic functions
from classical_optimization import classical_portfolio_allocation
from quantum_optimization import quantum_portfolio_allocation_local
from mock_ai import generate_mock_ai_output  # <- fixed circular import

# --- Configuration ---
st.set_page_config(
    page_title="QuantumFin Advisor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.set_page_config(
    page_title="FinMind: Talk. Plan. Invest. Smarter.",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "👋 Welcome to **FinMind**! I am your AI Investment Advisor. Let's talk, plan, and invest smarter. To get started, tell me your total budget (e.g., **$2000**)."}
    ]
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Custom Functions for Streamlit UI ---
def display_chat():
    """Displays the conversational history."""
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]

        if role == "system":
            st.markdown(f'<div style="background-color: #E8F8F5; padding: 10px; border-radius: 8px 8px 8px 2px; '
                        f'margin-bottom: 10px; border-left: 5px solid #2ECC71; color: #1C2833; max-width: 90%; font-size: 14px;">'
                        f'{content}</div>', unsafe_allow_html=True)
        else:  # user
            st.markdown(f'<div style="background-color: #3498DB; padding: 10px; border-radius: 8px 8px 2px 8px; '
                        f'margin-bottom: 10px; color: white; margin-left: 10%; text-align: right; font-size: 14px;">'
                        f'{content}</div>', unsafe_allow_html=True)

def display_results(result_data):
    """Displays the final allocation table and technical log."""
    st.subheader(f"💰 Final Recommendation: Allocation Details")
    st.markdown(f"**Engine Used:** {result_data['engine']}")
    st.markdown("---")

    # Display Allocation Table
    allocations_df = pd.DataFrame(result_data['allocations'])
    allocations_df['Amount ($)'] = allocations_df['amount'].apply(lambda x: f"${x:,.2f}")
    allocations_df['Weight (%)'] = (allocations_df['weight'] * 100).round(2).astype(str) + '%'
    allocations_df['Predicted Return'] = (allocations_df['predicted_return'] * 100).round(2).astype(str) + '%'

    # Select columns to display
    display_cols = ['ticker', 'Weight (%)', 'Amount ($)', 'Predicted Return', 'ai_score']
    allocations_df = allocations_df[display_cols].rename(columns={'ticker': 'Ticker', 'ai_score': 'AI Score'})

    st.dataframe(
        allocations_df,
        use_container_width=True,
        hide_index=True
    )

    # Display Performance Metrics
    st.subheader("📊 Performance & Risk Metrics")
    performance_df = pd.DataFrame(result_data['performance'].items(), columns=['Metric', 'Value'])
    st.table(performance_df)

    st.subheader("⚙️ Backend Technical Log (Narrative)")
    log = f"""
    1. **Data Engineer:** Fetched historical data and calculated rolling features.
    2. **AI Model:** LSTM/XGBoost ran, filtered universe down to **{len(result_data['allocations'])}** assets, 
       and outputted $\\mu$ (Predicted Returns) and $\\Sigma$ (Covariance).
    3. **Optimization:** **{result_data['engine']}** ran its core algorithm.
       - **Classical Mode:** Used a convex solver (e.g., CVXPY) to find the globally optimal **Max Sharpe** portfolio.
       - **Quantum Mode:** Formulated the problem as a **QUBO** matrix to enforce the **Cardinality Constraint** (max assets). Solved via Simulated Annealing.
    4. **Cloud Deployment:** Results wrapped and served in **{round(time.time() - st.session_state.start_time, 2)} seconds**.
    """
    st.markdown(log)
    st.caption("Disclaimer: This is a prototype simulation for a university project. Do not use for real financial decisions.")

def process_user_input(user_input, mode, max_assets):
    """Handles conversational logic and triggers the Python pipeline."""
    st.session_state.start_time = time.time()
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # --- Extract budget using regex ---
    budget_match = re.search(r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", user_input)
    total_budget = 0
    if budget_match:
        total_budget = float(budget_match.group(1).replace(',', ''))

    if total_budget < 1000:
        st.session_state.chat_history.append({"role": "system",
                                              "content": "For optimal diversification and accurate modeling, "
                                                         "please provide a budget of at least **$1,000**."})
        st.session_state.results = None
        st.rerun()

    # --- Processing Message ---
    processing_msg = f"*{mode.capitalize()} engine is running... (AI filtering {max_assets} stocks).* " \
                     f"Please wait (simulated latency added for realism)."
    st.session_state.chat_history.append({"role": "system", "content": processing_msg})


    try:
        # Generate mock AI output
        ai_output, mock_historical_data, mu_series = generate_mock_ai_output(num_selected=max_assets)

        if mode == 'classical':
            result = classical_portfolio_allocation(mock_historical_data, mu_series, total_budget)
            engine_name = "Classical (Max Sharpe / CVaR Proxy)"
        else:
            result = quantum_portfolio_allocation_local(ai_output, total_budget, max_assets)
            engine_name = "Quantum (QUBO / Simulated Annealing)"

       
        final_allocations_list = []
        if result.get("allocation"):  # <-- ensure allocation exists
            for ticker, amount in result["allocation"].items():
                ai_data = next((item for item in ai_output if item["ticker"] == ticker), {})
                final_allocations_list.append({
            "ticker": ticker,
            "amount": amount,
            "weight": amount / total_budget,
            "predicted_return": ai_data.get("predicted_return", 0),
            "volatility": ai_data.get("volatility", 0),
            "ai_score": ai_data.get("score", 0),
        })
        else:
            st.session_state.chat_history.append({
        "role": "system",
        "content": "⚠️ No allocation could be computed. Please try adjusting your budget or asset selection."
    })

        st.session_state.results = {
            "engine": engine_name,
            "allocations": final_allocations_list,
            "performance": result["performance"]
        }

        summary_message = f"**✅ FinMind Analysis Complete!** The **{engine_name}** model optimized your **${total_budget:,.2f}** investment. Scroll right for detailed allocation and risk metrics."
        st.session_state.chat_history.append({"role": "system", "content": summary_message})

    except Exception as e:
        st.session_state.chat_history.append({"role": "system",
                                              "content": f"❌ An error occurred during computation: {e}. Please try again."})
        st.session_state.results = None

# --- Streamlit UI Layout ---
st.title("FinMind 🚀")
st.markdown("**Talk. Plan. Invest. Smarter.** — Your AI-powered hybrid investment advisor.")


# Engine Selection
with st.expander("⚙️ Optimization Engine Settings"):
    engine_mode = st.radio(
        "Select Optimization Engine:",
        ('classical', 'quantum'),
        format_func=lambda x: "Classical (Max Sharpe / CVaR Proxy)" if x == 'classical' else "Quantum (QUBO / Simulated Annealing)",
        help="Classical is the robust baseline. Quantum demonstrates QUBO constraint handling."
    )
    max_assets = st.slider(
        "Max Assets to Select (K):", 3, 8, 4,
        help="Simulates the AI's pre-selection / Quantum constraint (K)."
    )
    st.markdown("💡 **Tip:** FinMind recommends starting with 3-5 assets for optimal diversification in your first run.")

# Display Chat and Results
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Chat Interface")
    display_chat()
with col2:
    st.subheader("Investment Results")
    if st.session_state.results:
        display_results(st.session_state.results)
    else:
        st.info("Results will appear here after analysis.")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask about your budget:", 
        key='user_input_key', 
        placeholder="I have $5000 to invest."
    )
    submitted = st.form_submit_button("Analyze & Optimize")
    
    if submitted and user_input:
        process_user_input(user_input, engine_mode, max_assets)
        
        # --- Optional: Explicitly clear user_input after submission ---
        st.session_state.user_input_key = ""

# Hackathon Note
st.markdown("---")
st.markdown("### Hackathon Note")
st.markdown("To run this on Streamlit Cloud, you need a `requirements.txt` listing all dependencies "
            "(`streamlit`, `pandas`, `numpy`, `pypfopt`, `dwave-ocean-sdk`). All project files (`.py`) "
            "must be in the same folder (`deployment/`).")
