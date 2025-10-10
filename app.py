import streamlit as st
import pandas as pd
import numpy as np
import time
import re

# --- CRITICAL IMPORTS: Synchronized with filenames in deployment/ ---
from classical_optimization import classical_portfolio_allocation
from quantum_optimization import quantum_portfolio_allocation_local
from pipeline_wrapper import generate_ai_inputs 

# --- Configuration ---
st.set_page_config(
    page_title="FinMind: Talk. Plan. Invest. Smarter.",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom Styling for Chat Bubbles (using st.markdown and HTML/CSS) ---
CHAT_STYLES = """
<style>
/* Streamlit main content padding adjustment */
.main .block-container {
    padding-top: 2rem;
    padding-right: 2rem;
    padding-left: 2rem;
    padding-bottom: 2rem;
}
/* Custom Chat Bubble Styles */
.chat-system {
    background-color: #E8F8F5; 
    padding: 10px; 
    border-radius: 8px 8px 8px 2px;
    margin-bottom: 10px; 
    border-left: 5px solid #2ECC71; 
    color: #1C2833; 
    max-width: 95%; 
    font-size: 14px;
}
.chat-user {
    background-color: #3498DB; 
    padding: 10px; 
    border-radius: 8px 8px 2px 8px;
    margin-bottom: 10px; 
    color: white; 
    margin-left: 10%; 
    text-align: right; 
    font-size: 14px;
}
</style>
"""
st.markdown(CHAT_STYLES, unsafe_allow_html=True)


# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "👋 Welcome to **FinMind**! I am your AI Investment Advisor. To get started, tell me your total budget (e.g., **$5000**)."}
    ]
if 'results' not in st.session_state:
    st.session_state.results = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()


# --- Core UI Functions ---
def display_chat():
    """Displays the conversational history using custom styled markdown."""
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]

        if role == "system":
            st.markdown(f'<div class="chat-system">{content}</div>', unsafe_allow_html=True)
        else:  # user
            st.markdown(f'<div class="chat-user">{content}</div>', unsafe_allow_html=True)

def display_results(result_data):
    """Displays the final allocation table and technical log."""
    
    st.markdown(f"### 💰 Final Recommendation")
    st.markdown(f"**Engine Used:** **`{result_data['engine']}`**", unsafe_allow_html=True)
    st.markdown("---")

    # Display Allocation Table
    allocations_df = pd.DataFrame(result_data['allocations'])
    # Format columns for display
    allocations_df['Amount ($)'] = allocations_df['amount'].apply(lambda x: f"${x:,.2f}")
    allocations_df['Weight (%)'] = (allocations_df['weight'] * 100).round(2).astype(str) + '%'
    
    # Check for keys in allocation list items before accessing
    predicted_return_col = allocations_df['predicted_return'].apply(lambda x: f"{x * 100:.2f}%") if 'predicted_return' in allocations_df.columns else "N/A"
    ai_score_col = allocations_df['ai_score'] if 'ai_score' in allocations_df.columns else "N/A"
    
    allocations_df['Predicted Return'] = predicted_return_col
    allocations_df['AI Score'] = ai_score_col


    display_cols = ['ticker', 'Weight (%)', 'Amount ($)', 'Predicted Return', 'AI Score']
    # Select columns while handling potential missing columns gracefully
    display_df = allocations_df[[col for col in display_cols if col in allocations_df.columns]]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Display Performance Metrics
    st.markdown("### 📊 Performance & Risk Metrics")
    performance_df = pd.DataFrame(result_data['performance'].items(), columns=['Metric', 'Value'])
    st.table(performance_df)

    st.markdown("### ⚙️ Backend Technical Log (Narrative)")
    # This narrative is crucial for the pitch, detailing the three-engine system
    log = f"""
    1. **Data Engineer:** Data acquired and rolling features calculated.
    2. **AI Predictive Engine:** Cached LSTM/XGBoost output filtered the universe to the **Top {len(result_data['allocations'])}** assets based on Predicted Sharpe.
    3. **Optimization Engine:** The **{result_data['engine']}** ran its core optimization algorithm ({'Max Sharpe' if 'Classical' in result_data['engine'] else 'QUBO'}).
    4. **Performance:** Full analysis served in **{round(time.time() - st.session_state.start_time, 2)} seconds**, proving the efficiency of the **Cloud-Native Framework**.
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
                                              "content": "For optimal diversification, please provide a budget of at least **$1,000**."})
        st.session_state.results = None
        st.rerun()

    # --- Processing Message & Simulated Latency ---
    processing_msg = f"*{mode.capitalize()} engine running... (AI filtering to {max_assets} stocks).* " \
                     f"Please wait while the cloud performs optimization."
    st.session_state.chat_history.append({"role": "system", "content": processing_msg})

    # Add simulated latency to mimic real-world processing (keep it short!)
    time.sleep(1)

    try:
        # 1. AI Layer: Get cached AI predictions and mock historical data
        ai_output, mock_historical_data, mu_series = generate_ai_inputs(num_selected=max_assets)

        # 2. Optimization Layer: Select Engine
        if mode == 'classical':
            result = classical_portfolio_allocation(mock_historical_data, mu_series, total_budget)
            engine_name = "Classical (Max Sharpe / CVaR Proxy)"
        else:
            # NOTE: quantum_optimization_local takes the raw ai_output list
            result = quantum_portfolio_allocation_local(ai_output, total_budget, max_assets)
            engine_name = "Quantum (QUBO / Simulated Annealing)"
        
        # 3. Process and format results
        final_allocations_list = []
        if result and result.get("allocation") and total_budget > 0:
            for ticker, amount in result["allocation"].items():
                # Find the full AI data for the selected ticker
                ai_data = next((item for item in ai_output if item["ticker"] == ticker), {})

                final_allocations_list.append({
                    "ticker": ticker,
                    "amount": amount,
                    "weight": amount / total_budget,
                    "predicted_return": ai_data.get("predicted_return", 0),
                    "ai_score": ai_data.get("ai_score", 0),
                })
        
        if not final_allocations_list:
            st.session_state.chat_history.append({
                "role": "system",
                "content": "⚠️ Optimization engine returned no valid allocation. Try adjusting your constraints or budget."
            })
            st.session_state.results = None
            st.rerun()

        st.session_state.results = {
            "engine": engine_name,
            "allocations": final_allocations_list,
            "performance": result["performance"]
        }

        summary_message = f"**✅ Analysis Complete!** The **{engine_name}** model optimized your **${total_budget:,.2f}** investment. Scroll right for detailed allocation and risk metrics."
        st.session_state.chat_history.append({"role": "system", "content": summary_message})

    except Exception as e:
        st.session_state.chat_history.append({"role": "system",
                                              "content": f"❌ An unexpected runtime error occurred: {e}. Check the logs."})
        st.session_state.results = None
        
    st.rerun()
        

# --- Streamlit UI Layout ---
st.title("FinMind 🚀")
st.markdown("**Talk. Plan. Invest. Smarter.** — Your AI-powered hybrid investment advisor.")


# Engine Selection (Expander is used to save screen space)
with st.expander("⚙️ Optimization Engine Settings"):
    engine_mode = st.radio(
        "Select Optimization Engine:",
        ('classical', 'quantum'),
        format_func=lambda x: "Classical (Max Sharpe / CVaR Proxy)" if x == 'classical' else "Quantum (QUBO / Simulated Annealing)",
        help="Classical is the robust baseline. Quantum demonstrates QUBO constraint handling."
    )
    max_assets = st.slider(
        "Max Assets to Select (K):", 3, 8, 4,
        help="Simulates the AI's pre-selection / Quantum constraint (K). Set K=4 for the first run."
    )
    st.markdown("💡 **Tip:** The quantum engine solves the complex problem of selecting *exactly* K assets.")

# Display Chat and Results side-by-side
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Chat Interface")
    # Use a fixed-height container for chat history scrolling
    with st.container(height=300): 
        display_chat()
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask about your budget:",
            key='user_input_key',
            placeholder="I have $5000 to invest."
        )
        submitted = st.form_submit_button("Analyze & Optimize")

        if submitted and user_input:
            # Call the processing function and trigger a full app rerun
            process_user_input(user_input, engine_mode, max_assets)
            # Clearing the input key forces the text box to reset visually
            st.session_state.user_input_key = "" 

with col2:
    st.subheader("Investment Results")
    if st.session_state.results:
        display_results(st.session_state.results)
    else:
        st.info("Results will appear here after analysis. Try entering a budget like '$2000' and clicking 'Analyze & Optimize'.")

st.markdown("---")
st.markdown("### 🚀 Final Deployment Checklist")
st.markdown("Remember to push all required files to your GitHub `deployment/` folder: `requirements.txt`, `pipeline_wrapper.py`, `classical_optimization.py`, and `quantum_optimization.py`.")
```eof
