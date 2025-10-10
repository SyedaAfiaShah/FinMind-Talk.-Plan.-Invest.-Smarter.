import streamlit as st
import pandas as pd
import numpy as np
import time
# Import the core logic functions
from classical_optimization import classical_portfolio_allocation
from quantum_optimization import quantum_portfolio_allocation_local
from app import generate_mock_ai_output # Import the mock data generator

# --- Configuration ---
st.set_page_config(
    page_title="QuantumFin Advisor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "👋 Welcome! I am the **QuantumFin Investment Advisor**. To get started, tell me your total budget for investment (e.g., **$2000**)."}
    ]
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Custom Functions for Streamlit UI ---

def display_chat():
    """Displays the conversational history."""
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        # Use Markdown to style the chat bubbles simply
        if role == "system":
            st.markdown(f'<div style="background-color: #E8F8F5; padding: 10px; border-radius: 8px 8px 8px 2px; margin-bottom: 10px; border-left: 5px solid #2ECC71; color: #1C2833; max-width: 90%; font-size: 14px;">{content}</div>', unsafe_allow_html=True)
        else: # user
            st.markdown(f'<div style="background-color: #3498DB; padding: 10px; border-radius: 8px 8px 2px 8px; margin-bottom: 10px; color: white; margin-left: 10%; text-align: right; font-size: 14px;">{content}</div>', unsafe_allow_html=True)

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
    # This is where we show the steps taken by the different team members
    log = f"""
    1. **Data Engineer:** Fetched historical data and calculated rolling features.
    2. **AI Model:** LSTM/XGBoost ran, filtered universe down to **{len(result_data['allocations'])}** assets, and outputted $\\mu$ (Predicted Returns) and $\\Sigma$ (Covariance).
    3. **Optimization:** **{result_data['engine']}** ran its core algorithm.
       - **Classical Mode:** Used a convex solver (e.g., CVXPY) to find the globally optimal **Max Sharpe** portfolio.
       - **Quantum Mode:** Formulated the problem as a **QUBO** matrix to enforce the **Cardinality Constraint** (max assets). Solved via Simulated Annealing.
    4. **Cloud Deployment:** Results wrapped and served in **{round(time.time() - st.session_state.start_time, 2)} seconds**.
    """
    st.markdown(log)
    st.caption("Disclaimer: This is a prototype simulation for a university project. Do not use for real financial decisions.")

def process_user_input(user_input, mode, max_assets):
    """Handles conversational logic and triggers the Python pipeline."""
    
    # 1. Start timer and append user message
    st.session_state.start_time = time.time()
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 2. Extract budget
    budget_match = st.session_state.user_input.match(r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
    total_budget = 0
    if budget_match and budget_match.group(1):
        total_budget = float(budget_match.group(1).replace(',', ''))

    if total_budget < 1000:
        st.session_state.chat_history.append({"role": "system", "content": "For optimal diversification and accurate modeling, please provide a budget of at least **$1,000**."})
        st.session_state.results = None
        st.rerun()

    # 3. Processing Message
    processing_msg = f"*{mode.capitalize()} engine is running... (AI filtering {max_assets} stocks).* Please wait (simulated latency added for realism)."
    st.session_state.chat_history.append({"role": "system", "content": processing_msg})

    # Add a short delay to simulate network latency and computation
    time.sleep(2) 
    
    # 4. Run the Core Python Pipeline
    try:
        # We need the mu_series for the classical model, but the quantum model needs the ai_output dict.
        # Since the functions expect the raw data frame for real covariance, we slightly modify the flow here
        # to ensure the mock historical data is generated.
        ai_output, mock_historical_data, mu_series = generate_mock_ai_output(num_selected=max_assets)

        if mode == 'classical':
            result = classical_portfolio_allocation(mock_historical_data, mu_series, total_budget)
            engine_name = "Classical (Max Sharpe / CVaR Proxy)"
        else: # quantum
            result = quantum_portfolio_allocation_local(ai_output, total_budget, max_assets)
            engine_name = "Quantum (QUBO / Simulated Annealing)"
            
        # 5. Format and store final results
        st.session_state.results = {
            "engine": engine_name,
            "allocations": result["allocation"], # This needs to be a list of dicts for easy display
            "performance": result["performance"]
        }
        
        # We need to manually convert the allocation output from $ to the full dict structure
        # to simplify the Streamlit display function
        final_allocations_list = []
        for ticker, amount in result["allocation"].items():
            # Find the corresponding original AI data to get return/volatility/score
            ai_data = next((item for item in ai_output if item["ticker"] == ticker), {})
            
            final_allocations_list.append({
                "ticker": ticker,
                "amount": amount,
                "weight": amount / total_budget,
                "predicted_return": ai_data.get("predicted_return", 0),
                "volatility": ai_data.get("volatility", 0),
                "ai_score": ai_data.get("score", 0),
            })

        st.session_state.results['allocations'] = final_allocations_list

        # 6. Final conversational response
        summary_message = f"**✅ Analysis Complete!** The **{engine_name}** model found the optimal allocation for your **${total_budget:,.2f}** budget. See the details below."
        st.session_state.chat_history.append({"role": "system", "content": summary_message})

    except Exception as e:
        st.session_state.chat_history.append({"role": "system", "content": f"❌ An error occurred during computation: {e}. Please try again."})
        st.session_state.results = None

# --- Streamlit UI Layout ---

st.title("QuantumFin Advisor 🚀")
st.markdown("A **Hybrid AI-Quantum-Cloud** Framework Prototype.")

# Engine Selection and Parameters in Sidebar/Expander
with st.expander("⚙️ Optimization Engine Settings"):
    engine_mode = st.radio(
        "Select Optimization Engine:",
        ('classical', 'quantum'),
        format_func=lambda x: "Classical (Max Sharpe / CVaR Proxy)" if x == 'classical' else "Quantum (QUBO / Simulated Annealing)",
        help="Classical is the robust baseline. Quantum demonstrates QUBO constraint handling."
    )
    max_assets = st.slider("Max Assets to Select (K):", 3, 8, 4, help="Simulates the AI's pre-selection/Quantum constraint (K).")
    
# Display Chat and Results side-by-side (using columns)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Chat Interface")
    # Chat display
    display_chat()
    
with col2:
    st.subheader("Investment Results")
    if st.session_state.results:
        display_results(st.session_state.results)
    else:
        st.info("Results will appear here after analysis.")


# Input Form at the bottom
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask about your budget:", key='user_input_key', placeholder="I have $5000 to invest.")
    submitted = st.form_submit_button("Analyze & Optimize")

    if submitted and user_input:
        process_user_input(st.session_state.user_input_key, engine_mode, max_assets)

# --- Important Final Hackathon Note ---
st.markdown("---")
st.markdown("### Hackathon Note")
st.markdown("To run this on Streamlit Cloud, you need a `requirements.txt` listing all dependencies (`streamlit`, `pandas`, `numpy`, `pypfopt`, `dwave-ocean-sdk`). All project files (`.py`) must be in the same folder (`deployment/`).")
