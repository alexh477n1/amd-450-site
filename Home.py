import streamlit as st
import utils
import warmup
import comparison

## PAGE CONFIG
st.set_page_config(layout="wide", page_title="AMD Benchmark Analysis Tool")

st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        background-color: #e0f7fa; 
        border-radius: 5px; 
        color: #333;
    }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b; color: white; }
    .stDataFrame { margin-bottom: 2rem; }
    div[data-testid="stDownloadButton"] { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)


## MAIN APP STRUCTURE

def main():
    st.sidebar.title("AMD Benchmark Data Analysis")
    st.sidebar.markdown("---")

    # Shared config
    st.sidebar.header("Global Visualization (Tab 1)")
    show_iqr = st.sidebar.checkbox("Show Tight IQR Bands", value=False)
    show_gmm = st.sidebar.checkbox("Show Dist. Plots", value=True)

    # Note: We are updating the CONFIG (but only for chunk size and user defined tolerance level in utils based on sidebar interactions if necessary.
    # Currently the config is static in utils, but if you wanted to make config dynamic,
    # you would update utils.CONFIG here.

    st.sidebar.markdown("---")
    st.sidebar.header("Comparison Settings (Tab 2)")
    chunk_size = st.sidebar.slider("Chunk Size", 50, 2000, 250, step=50,
                                   help="Size of data segments to analyze for local stability.")
    perf_tolerance = st.sidebar.number_input("Perf. Tolerance (%)", 0.0, 5.0, 1.0, step=0.1,
                                             help="Min % diff to declare a winner.")

    # Tabs
    tab1, tab2 = st.tabs(["Detailed Warmup Analysis", "A/B Benchmark Comparison"])

    # Tab 1: Detailed Warmup Analysis
    with tab1:
        warmup.render_warmup_tab(show_iqr, show_gmm)

    # Tab 2: A/B Benchmark Comparison
    with tab2:
        comparison.render_comparison_tab(chunk_size, perf_tolerance)


if __name__ == "__main__":
    main()