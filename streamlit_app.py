import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import io

# =====================================
# Page Config & Global Styling
# =====================================
st.set_page_config(page_title="RMB Capital Optimization Tool", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #EDF2FB;
        color: #333333;
    }
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
    }
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3, h4 {
        font-family: Arial, sans-serif;
    }
    .stMetric {
        background-color: #E8EBF0;
        padding: 0.4rem;
        border-radius: 0.25rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================
# Helper Functions
# =====================================
def calculate_capital(rwa, cap_ratio):
    """
    Capital = RWA * (cap_ratio / 100).
    Example: If RWA=1,000,000 and cap_ratio=12,
    capital usage = 1,000,000 * 0.12 = 120,000
    """
    return rwa * (cap_ratio / 100.0)


def scale_rwa_with_lgd(rwa_old, lgd_old, lgd_new):
    """
    Simplified formula to scale RWA if LGD changes:
    RWA_new = RWA_old * (LGD_new / LGD_old).

    Real Basel or internal models often use more complex
    equations involving PD, EAD, maturity, correlation, etc.
    """
    if lgd_old == 0:
        return 0
    scale_factor = lgd_new / lgd_old
    return rwa_old * scale_factor


def create_fictional_clients():
    """
    Returns a DataFrame of 5 fictional clients with:
      - Client Name
      - Loan Amount
      - Term (years)
      - PD (%)
      - LGD (%)
      - Current RWA
    """
    data = {
        "Client": [
            "Atlantis Manufacturing",
            "Zulu Shipping",
            "Shades Retail",
            "Green Earth Energy",
            "Solace Holdings"
        ],
        "Loan Amount (ZAR)": [500_000_000, 750_000_000, 1_000_000_000, 1_250_000_000, 1_800_000_000],
        "Term (Years)": [3, 4, 2, 7, 5],
        "PD (%)": [1.2, 2.0, 3.5, 1.0, 2.5],
        "LGD (%)": [30.0, 45.0, 50.0, 20.0, 40.0],
        "RWA (ZAR)": [300_000_000, 400_000_000, 650_000_000, 200_000_000, 800_000_000]
    }
    return pd.DataFrame(data)


def hide_dataframe_index(df_style):
    """Helper to hide the Styler index in st.dataframe()."""
    return df_style.hide(axis="index")


# =====================================
# Main Page: Title & Intro
# =====================================
st.title("RMB Capital Optimization Tool")

st.markdown(
    """
    This tool demonstrates how **credit protection** (e.g., via CLNs) 
    might reduce **Loss Given Default (LGD)**, thus lowering 
    **Risk-Weighted Assets (RWA)** and freeing up **capital**.
    """
)

st.info(
    """
    **How It Works**  
    - We have five fictional clients, each with different loan amounts, PD, LGD, and RWA.  
    - By reducing LGD, we recalculate RWA and see how much capital is freed.  
    - We also provide a simple "optimization" approach to allocate LGD reduction 
      among the clients who yield the biggest capital relief per LGD point.  
    - Finally, we discuss additional factors for **issuing CLNs**.
    """
)

# =====================================
# Sidebar: Inputs
# =====================================
st.sidebar.header("Global Parameters")

cap_ratio = st.sidebar.number_input(
    "Capital Ratio Requirement (%)",
    min_value=0.0,
    value=12.0,
    step=0.5,
    help="The bank's regulatory or internal capital ratio requirement."
)

protection_budget = st.sidebar.number_input(
    "Total LGD Reduction Budget (bps of LGD)",
    min_value=0.0,
    value=40.0,
    step=5.0,
    help=(
        "A simplified 'budget' for how many percentage points of LGD reduction you "
        "can allocate across all clients. E.g., if budget=40, you could allocate "
        "20 to one client, 20 to another, etc."
    )
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Helper Notes")
st.sidebar.info(
    """
    Each client starts with a 'base LGD.'  
    You can distribute the 'LGD reduction budget' across multiple clients.  
    This is purely illustrative—real-world deals are more complex.
    """
)

# =====================================
# Step 1: Five Fictional Clients (Baseline)
# =====================================
df_clients = create_fictional_clients()

# Calculate their current capital usage (no protection)
df_clients["Current Capital (ZAR)"] = [
    calculate_capital(rwa, cap_ratio) for rwa in df_clients["RWA (ZAR)"]
]

st.subheader("1. Baseline: Five Fictional Clients")
st.write("Below are five sample clients with pre-credit-protection data:")

# We'll style the baseline DataFrame with commas
format_dict_baseline = {
    "Loan Amount (ZAR)": "{:,.0f}",
    "Term (Years)": "{:,.0f}",
    "PD (%)": "{:,.2f}",
    "LGD (%)": "{:,.2f}",
    "RWA (ZAR)": "{:,.0f}",
    "Current Capital (ZAR)": "{:,.2f}",
}
styled_baseline = df_clients.style.format(format_dict_baseline)
styled_baseline = hide_dataframe_index(styled_baseline)
st.dataframe(styled_baseline, use_container_width=True)

# =====================================
# Step 2: Allocate LGD Reductions
# =====================================
st.subheader("2. Allocate LGD Reduction")

st.write(
    """
    We'll distribute the **LGD reduction budget** (in % points) across the 5 clients.
    Enter how many LGD points to allocate to each client. 
    """
)

client_lgd_allocations = []
col_list = st.columns(5)
for i, client_name in enumerate(df_clients["Client"]):
    with col_list[i]:
        base_lgd = df_clients.loc[i, "LGD (%)"]
        max_alloc = min(protection_budget, base_lgd)
        alloc_val = st.number_input(
            f"{client_name} LGD↓",
            min_value=0.0,
            max_value=float(max_alloc),
            value=0.0,
            step=5.0
        )
        client_lgd_allocations.append(alloc_val)

allocated_sum = sum(client_lgd_allocations)
if allocated_sum > protection_budget:
    st.warning(
        f"You've allocated a total of {allocated_sum:.2f} LGD points, "
        f"but your budget is only {protection_budget:.2f}. Please adjust."
    )

# =====================================
# Step 3: Recalculate RWA & Freed Capital
# =====================================
st.subheader("3. Recalculation & Results")

df_opt = df_clients.copy()

new_lgd_vals = []
new_rwa_vals = []
new_cap_vals = []
cap_freed_vals = []

for i, row in df_opt.iterrows():
    base_lgd = row["LGD (%)"]
    base_rwa = row["RWA (ZAR)"]
    old_cap = row["Current Capital (ZAR)"]
    allocated = client_lgd_allocations[i]

    if allocated_sum <= protection_budget:
        new_lgd = max(base_lgd - allocated, 0)
    else:
        # If user overallocated, revert to baseline
        new_lgd = base_lgd

    new_rwa = scale_rwa_with_lgd(base_rwa, base_lgd, new_lgd)
    new_cap = calculate_capital(new_rwa, cap_ratio)

    new_lgd_vals.append(new_lgd)
    new_rwa_vals.append(new_rwa)
    new_cap_vals.append(new_cap)
    cap_freed_vals.append(old_cap - new_cap)

df_opt["New LGD (%)"] = new_lgd_vals
df_opt["New RWA (ZAR)"] = new_rwa_vals
df_opt["New Capital (ZAR)"] = new_cap_vals
df_opt["Capital Freed (ZAR)"] = cap_freed_vals

# Portfolio-level summary
total_current_cap = df_opt["Current Capital (ZAR)"].sum()
total_new_cap = df_opt["New Capital (ZAR)"].sum()
portfolio_cap_freed = total_current_cap - total_new_cap

# Create a table of results
df_display = df_opt[
    [
        "Client",
        "Loan Amount (ZAR)",
        "Term (Years)",
        "PD (%)",
        "LGD (%)",
        "New LGD (%)",
        "RWA (ZAR)",
        "New RWA (ZAR)",
        "Current Capital (ZAR)",
        "New Capital (ZAR)",
        "Capital Freed (ZAR)"
    ]
].copy()

# Round numeric columns as needed
format_dict_final = {
    "Loan Amount (ZAR)": "{:,.0f}",
    "Term (Years)": "{:,.0f}",
    "PD (%)": "{:,.2f}",
    "LGD (%)": "{:,.2f}",
    "New LGD (%)": "{:,.2f}",
    "RWA (ZAR)": "{:,.0f}",
    "New RWA (ZAR)": "{:,.0f}",
    "Current Capital (ZAR)": "{:,.2f}",
    "New Capital (ZAR)": "{:,.2f}",
    "Capital Freed (ZAR)": "{:,.2f}"
}

styled_results = df_display.style.format(format_dict_final)
styled_results = hide_dataframe_index(styled_results)
st.dataframe(styled_results, use_container_width=True)

st.info(
    f"""
    **Portfolio Summary**  
    - Total Current Capital Usage: ZAR {total_current_cap:,.2f}  
    - Total New Capital Usage: ZAR {total_new_cap:,.2f}  
    - **Total Capital Freed**: ZAR {portfolio_cap_freed:,.2f}
    """
)

# =====================================
# Step 4: Simple "Optimization" Helper
# =====================================
st.subheader("4. Simple Optimization Suggestion")
st.write(
    """
    If you'd like the system to **auto-suggest** how to allocate 
    your LGD reduction budget, we'll do a basic ranking by 
    \"Capital Freed per 1 LGD point.\"
    """
)


def capital_freed_per_point(row):
    if row["LGD (%)"] == 0:
        return 0.0
    old_cap = row["Current Capital (ZAR)"]
    # Hypothetically reduce the LGD by 1 point
    new_rwa_test = scale_rwa_with_lgd(row["RWA (ZAR)"], row["LGD (%)"], row["LGD (%)"] - 1)
    new_cap_test = calculate_capital(new_rwa_test, cap_ratio)
    return old_cap - new_cap_test


df_opt["Freed per LGDpt (ZAR)"] = df_opt.apply(capital_freed_per_point, axis=1)
df_opt_sorted = df_opt.sort_values(by="Freed per LGDpt (ZAR)", ascending=False)

col_autoalloc, col_desc = st.columns([1, 2])
with col_autoalloc:
    st.write("**Allocation Suggestion**")
    remaining_budget = protection_budget
    suggested_allocs = []
    for _, row in df_opt_sorted.iterrows():
        base_lgd = row["LGD (%)"]
        if remaining_budget <= 0 or base_lgd <= 0:
            suggested_allocs.append(0.0)
            continue
        allocation = min(remaining_budget, base_lgd)
        suggested_allocs.append(allocation)
        remaining_budget -= allocation

    df_suggest = df_opt_sorted[["Client", "LGD (%)", "Freed per LGDpt (ZAR)"]].copy()
    df_suggest["Suggested LGD Reduction"] = suggested_allocs
    df_suggest["Final LGD (%)"] = df_suggest["LGD (%)"] - df_suggest["Suggested LGD Reduction"]

    format_dict_suggest = {
        "LGD (%)": "{:,.2f}",
        "Freed per LGDpt (ZAR)": "{:,.2f}",
        "Suggested LGD Reduction": "{:,.2f}",
        "Final LGD (%)": "{:,.2f}"
    }
    styled_suggest = df_suggest.style.format(format_dict_suggest).hide(axis="index")
    st.dataframe(styled_suggest, use_container_width=True)

with col_desc:
    st.markdown(
        """
        1. Start with the client offering the highest capital relief per 1 LGD point.  
        2. Allocate as many LGD points as possible until you run out of budget or 
           the client's LGD hits zero.  
        3. Move on to the next.  

        **Note**: Real optimization might require advanced solvers 
        (linear or mixed-integer programming) and factoring in 
        the *cost* of protection.
        """
    )

# =====================================
# Step 5: Visualization - Capital Freed
# =====================================
st.subheader("5. Visualization: Capital Freed by Client")

fig = px.bar(
    df_display,
    x="Client",
    y="Capital Freed (ZAR)",
    title="Capital Freed After LGD Reductions",
    color="Client",
    labels={
        "Capital Freed (ZAR)": "Capital Freed (ZAR)",
        "Client": "Client"
    }
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# =====================================
# Step 6: Download Section - Excel Export
# =====================================
st.subheader("6. Download Your Results")
st.write("Download the final table (no index) to Excel for further analysis.")

output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    df_display.to_excel(writer, index=False, sheet_name="Results")
    worksheet = writer.sheets["Results"]
    for idx, col in enumerate(df_display.columns):
        worksheet.set_column(idx, idx, 20)
excel_data = output.getvalue()

st.download_button(
    label="Download Excel",
    data=excel_data,
    file_name="Capital_Optimization_Results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")

# =====================================
# 7. CLN Issuance: Additional Considerations
# =====================================
st.subheader("7. CLN Issuance: Additional Considerations")
st.write(
    """
    Once you've determined where LGD reduction is most beneficial, you may decide 
    to **issue CLNs** (Credit-Linked Notes) for certain loans or clients. 
    Select the client(s) you actually want to issue CLNs for:
    """
)

all_clients = df_display["Client"].tolist()
selected_clients_for_clns = st.multiselect(
    "Choose which clients to cover with CLNs:",
    options=all_clients,
    default=[]
)

# For a real model, you'd incorporate the cost/spread of issuing CLNs and the coverage ratio.
if selected_clients_for_clns:
    st.markdown("**You have selected:**")
    chosen_df = df_display[df_display["Client"].isin(selected_clients_for_clns)].copy()

    st.markdown(
        """
        Below is an area to *optionally* specify a 'cost of protection' if you'd like 
        to approximate the net benefit.
        """
    )
    # Let the user add a cost of protection for each selected client
    cost_of_protection = {}
    for c in selected_clients_for_clns:
        # Could be bps of the loan, or a nominal cost, etc.
        val = st.number_input(
            f"Cost of protection for {c} (ZAR):",
            min_value=0.0,
            value=0.0,
            step=50000.0
        )
        cost_of_protection[c] = val

    # Quick calculation: net capital saving in currency = Freed capital minus cost
    chosen_df["Cost of Protection (ZAR)"] = chosen_df["Client"].apply(lambda x: cost_of_protection[x])
    chosen_df["Net Benefit (ZAR)"] = chosen_df["Capital Freed (ZAR)"] - chosen_df["Cost of Protection (ZAR)"]

    format_dict_cln = {
        "Loan Amount (ZAR)": "{:,.0f}",
        "Capital Freed (ZAR)": "{:,.2f}",
        "Cost of Protection (ZAR)": "{:,.2f}",
        "Net Benefit (ZAR)": "{:,.2f}"
    }
    st.dataframe(
        hide_dataframe_index(
            chosen_df[["Client", "Loan Amount (ZAR)", "Capital Freed (ZAR)",
                       "Cost of Protection (ZAR)", "Net Benefit (ZAR)"]]
            .style.format(format_dict_cln)
        ),
        use_container_width=True
    )

    st.info(
        """
        **Note**: In real-world deals, you'd factor in many details:
        - **Coverage Ratio**: Are we covering 100% of the loan or just a portion?
        - **Tenor Alignment**: The CLN maturity vs. the loan's maturity.
        - **Regulatory Recognition**: For partial or full capital relief, 
          does the CLN structure meet Basel/Regulator standards?
        - **Market Appetite & Pricing**: Are investors willing to buy the CLN at a cost that justifies it?
        - **Documentation & Legal Costs**: Setting up the CLN program and ensuring compliance.
        """
    )
else:
    st.markdown("_No clients selected for CLN coverage yet._")

st.markdown("---")

# =====================================
# Footer / Final Notes
# =====================================
st.write(
    """
    ### Final Thoughts
    1. **Simplified Model**: Real-world capital models can be more intricate.  
    2. **Cost vs. Benefit**: Always compare the cost of issuing credit protection to the 
       capital freed.  
    3. **Fictional Data**: All figures here are illustrative examples.  

    *For advanced multi-constraint optimization or different coverage scenarios, 
    consider integrating solvers like PuLP or OR-Tools.*
    """
)
