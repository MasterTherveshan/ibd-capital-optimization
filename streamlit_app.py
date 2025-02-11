import streamlit as st
import pandas as pd
import numpy as np
import random

# ---------------------------------------
# PAGE CONFIG & TITLE
# ---------------------------------------
st.set_page_config(page_title="RMB Capital Optimization Tool", layout="wide")
st.title("RMB Capital Optimization Tool")

st.write(
    """
    **Overview**:
    1. **Section 1**: Editable Portfolio (Drawn%, Undrawn%, PD).  
    2. **Section 2**: LGD Reversion Table (how much of each loan to sell 
       to restore Old LGD).  
    3. **Section 3**: Available Capital & Shortfall (defaults to match portfolio usage).  
    4. **Section 4**: New RCF creation.  
    5. **Section 5**: Industry Filter & Sell-off Options if a shortfall remains, 
       showing only clients that can individually solve the shortfall with \(\le 100\%\) partial coverage.
    """
)

st.markdown("---")

# ---------------------------------------
# GLOBALS / HELPER FUNCTIONS
# ---------------------------------------
SA_COMPANIES = [
    "Sasol", "Shoprite", "MTN", "Vodacom", "Standard Bank", "Absa",
    "Nedbank", "FirstRand", "Discovery", "Sanlam", "Old Mutual", "Anglo American",
    "BHP Billiton", "Exxaro", "Mondi", "Tiger Brands", "Aspen Pharmacare",
    "Naspers", "Woolworths", "Pick n Pay", "Mediclinic", "Massmart", "Netcare",
    "Impala Platinum", "Kumba Iron Ore", "Harmony Gold", "Sibanye-Stillwater",
    "African Rainbow Minerals", "Famous Brands", "Gold Fields"
]

SECTOR_MAP = {
    "Sasol": "Oil & Gas",
    "Shoprite": "Retail",
    "MTN": "Telecom",
    "Vodacom": "Telecom",
    "Standard Bank": "Financials",
    "Absa": "Financials",
    "Nedbank": "Financials",
    "FirstRand": "Financials",
    "Discovery": "Insurance",
    "Sanlam": "Insurance",
    "Old Mutual": "Insurance",
    "Anglo American": "Mining",
    "BHP Billiton": "Mining",
    "Exxaro": "Mining",
    "Mondi": "Manufacturing",
    "Tiger Brands": "Food & Beverage",
    "Aspen Pharmacare": "Healthcare",
    "Naspers": "Technology",
    "Woolworths": "Retail",
    "Pick n Pay": "Retail",
    "Mediclinic": "Healthcare",
    "Massmart": "Retail",
    "Netcare": "Healthcare",
    "Impala Platinum": "Mining",
    "Kumba Iron Ore": "Mining",
    "Harmony Gold": "Mining",
    "Sibanye-Stillwater": "Mining",
    "African Rainbow Minerals": "Mining",
    "Famous Brands": "Food & Beverage",
    "Gold Fields": "Mining"
}

CAP_RATIO = 0.12  # 12% capital ratio
random.seed(42)


def create_fictional_clients(num=30):
    """Generate 30 random clients with Old vs. New LGDs, PD, etc."""
    data_list = []
    for i in range(num):
        cname = SA_COMPANIES[i]
        sec = SECTOR_MAP[cname]

        loan_amt = random.randint(100_000_000, 2_000_000_000)
        old_lgd = random.randint(20, 50)
        new_lgd = old_lgd + random.randint(0, 10)  # "uplift"
        drawn_pct = random.randint(40, 70)
        undrawn_pct = 100 - drawn_pct
        pd_val = round(random.uniform(1.0, 5.0), 2)

        data_list.append({
            "Client": cname,
            "Sector": sec,
            "Loan Amount": loan_amt,
            "Old LGD (%)": old_lgd,
            "New LGD (%)": new_lgd,
            "Drawn (%)": drawn_pct,
            "Undrawn (%)": undrawn_pct,
            "PD (%)": pd_val
        })
    return pd.DataFrame(data_list)


def calc_portfolio_cap_usage(row):
    """
    RWA = Loan * (Drawn%/100) * (PD%/100) * (NewLGD%/100) * 12.5
    capital usage = RWA * CAP_RATIO
    """
    amt = row["Loan Amount"]
    drawn = row["Drawn (%)"]
    pd_ = row["PD (%)"]
    new_lgd = row["New LGD (%)"]
    rwa = amt * (drawn / 100) * (pd_ / 100) * (new_lgd / 100) * 12.5
    return rwa * CAP_RATIO


def calc_loan_to_sell_for_lgd_reversion(row):
    """
    If NewLGD > OldLGD:
      portion = (NewLGD - OldLGD)/NewLGD
      LoanToSell = portion * LoanAmt * (Drawn%/100)
    else 0
    """
    old_lgd = row["Old LGD (%)"]
    new_lgd = row["New LGD (%)"]
    if new_lgd <= old_lgd:
        return 0.0
    portion = (new_lgd - old_lgd) / new_lgd
    amt = row["Loan Amount"]
    drawn_fraction = row["Drawn (%)"] / 100
    return portion * amt * drawn_fraction


def fraction_of_drawn(row):
    """
    fraction_sold = (LoanToSell / (loan_amt * drawn%/100)) * 100
    """
    drawn_amt = row["Loan Amount"] * (row["Drawn (%)"] / 100)
    if drawn_amt == 0:
        return 0.0
    return (row["Loan to Sell (ZAR)"] / drawn_amt) * 100


def new_rcf_usage(amt, dr, pd_, lgd):
    """
    RWA = amt * (dr/100) * (pd_/100) * (lgd/100) * 12.5
    capital usage = RWA * CAP_RATIO
    """
    rwa = amt * (dr / 100) * (pd_ / 100) * (lgd / 100) * 12.5
    return rwa * CAP_RATIO


def format_num(x, decimals=2):
    return f"{x:,.{decimals}f}"


# ================================================
# SECTION 1: EDITABLE PORTFOLIO
# ================================================
st.subheader("Section 1: Editable Portfolio")

df_initial = create_fictional_clients()
df_edited = st.data_editor(
    df_initial,
    column_config={
        "Client": st.column_config.TextColumn(disabled=True),
        "Sector": st.column_config.TextColumn(disabled=True),
        "Loan Amount": st.column_config.NumberColumn(disabled=True),
        "Old LGD (%)": st.column_config.NumberColumn(disabled=True),
        "New LGD (%)": st.column_config.NumberColumn(disabled=True),
        "Drawn (%)": st.column_config.NumberColumn(),
        "Undrawn (%)": st.column_config.NumberColumn(),
        "PD (%)": st.column_config.NumberColumn(),
    },
    use_container_width=True
)
st.info(
    "You can modify Drawn%, Undrawn%, and PD% for each client. "
    "Old & New LGDs are read-only for demonstration."
)

df_portfolio = df_edited.copy()
df_portfolio["Capital Usage"] = df_portfolio.apply(calc_portfolio_cap_usage, axis=1)
total_port_usage = df_portfolio["Capital Usage"].sum()
st.write(f"**Current Portfolio Capital Usage**: {format_num(total_port_usage)}")

# ================================================
# SECTION 2: LGD REVERSION TABLE
# ================================================
st.markdown("---")
st.subheader("Section 2: LGD Reversion Table")

st.write(
    """
    For each client, how much of the loan (ZAR and %) must be sold 
    to revert from **New LGD** to **Old LGD**, ignoring any shortfall logic.
    """
)

df_lgd = df_portfolio.copy()
df_lgd["Loan to Sell (ZAR)"] = df_lgd.apply(calc_loan_to_sell_for_lgd_reversion, axis=1)
df_lgd["% of RCF to Sell"] = df_lgd.apply(fraction_of_drawn, axis=1)

cols_display = [
    "Client", "Sector", "Loan Amount", "Old LGD (%)", "New LGD (%)",
    "Drawn (%)", "PD (%)", "Loan to Sell (ZAR)", "% of RCF to Sell"
]
df_lgd_disp = df_lgd[cols_display].copy()

for col in cols_display[2:]:
    df_lgd_disp[col] = df_lgd_disp[col].apply(lambda x: format_num(x))

st.dataframe(df_lgd_disp, use_container_width=True)

# ================================================
# SECTION 3: AVAILABLE CAPITAL & SHORTFALL
# ================================================
st.markdown("---")
st.subheader("Section 3: Available Capital & Shortfall")

st.write(
    """
    We default your capital to match the current portfolio usage, 
    so you're at 0 shortfall to start. Adjust if desired.
    """
)
available_cap = st.number_input(
    "Your total capital (ZAR)?",
    min_value=0.0,
    value=float(total_port_usage),
    step=50_000_000.0
)

diff = available_cap - total_port_usage
if diff < 0:
    st.warning(
        f"Existing portfolio usage exceeds your capital by {format_num(abs(diff))}."
    )
else:
    st.info(
        f"You have {format_num(diff)} leftover after the existing portfolio."
    )

# ================================================
# SECTION 4: NEW RCF
# ================================================
st.markdown("---")
st.subheader("Section 4: Adding a New RCF")

new_amt = st.number_input(
    "New RCF Loan Amount",
    min_value=0.0,
    value=500_000_000.0,
    step=50_000_000.0
)
new_drawn = st.slider("Drawn (%) for New RCF", min_value=0, max_value=100, value=60)
new_pd = st.slider("PD (%) for New RCF", min_value=0.0, max_value=10.0, value=3.0)
new_lgd = st.slider("LGD (%) for New RCF", min_value=0, max_value=60, value=40)

rcf_use = new_rcf_usage(new_amt, new_drawn, new_pd, new_lgd)
combined_usage = total_port_usage + rcf_use
shortfall = combined_usage - available_cap

st.write(f"**New RCF Capital Usage**: {format_num(rcf_use)}")
if shortfall > 0:
    st.error(
        f"**Shortfall**: {format_num(shortfall)} above your available capital."
    )
else:
    st.success(
        f"No shortfall. Combined usage = {format_num(combined_usage)} ≤ {format_num(available_cap)}."
    )

# ================================================
# SECTION 5: SELL-OFF SUGGESTIONS (≤100%)
# ================================================
st.markdown("---")
st.subheader("Section 5: Industry Filter & Sell-off Options <100%")

st.write(
    """
    If there's a shortfall, pick which industries you'd consider selling from. 
    We'll show only the clients that could individually fix the shortfall 
    with ≤ 100% partial coverage.
    """
)

if shortfall <= 0:
    st.info("No shortfall → no sell-off needed.")
else:
    # We have a shortfall
    st.write(f"Your shortfall = **{format_num(shortfall)}**. ")

    all_industries = sorted(df_portfolio["Sector"].unique())
    # Pre-select all by default
    selected_inds = st.multiselect(
        "Which industries to consider selling from?",
        options=all_industries,
        default=all_industries
    )

    # We'll see for each client in those industries:
    # fraction_needed = shortfall / that_client_cap_usage * 100
    # If fraction_needed <= 100 => that single client alone can solve shortfall with partial coverage
    df_sell = df_portfolio[df_portfolio["Sector"].isin(selected_inds)].copy()

    if df_sell.empty:
        st.warning("No clients in those industries. Can't sell anything.")
    else:
        # Sort descending by capital usage
        df_sell.sort_values(by="Capital Usage", ascending=False, inplace=True)


        def fraction_needed(row):
            cap_use = row["Capital Usage"]
            if cap_use <= 0:
                return 99999.0
            return (shortfall / cap_use) * 100


        df_sell["% Sell-Off to Cover Shortfall"] = df_sell.apply(fraction_needed, axis=1)
        df_sell = df_sell[df_sell["% Sell-Off to Cover Shortfall"] <= 100]

        if df_sell.empty:
            st.warning("No single client can individually fix the shortfall with ≤100% coverage.")
        else:
            st.write(
                """
                Below are the clients that can fix the **entire** shortfall alone, 
                with ≤100% partial coverage. 
                """
            )

            show_cols = [
                "Client", "Sector", "Loan Amount", "Drawn (%)", "PD (%)",
                "New LGD (%)", "Capital Usage", "% Sell-Off to Cover Shortfall"
            ]
            df_disp = df_sell[show_cols].copy()

            for col in ["Loan Amount", "Drawn (%)", "PD (%)",
                        "New LGD (%)", "Capital Usage", "% Sell-Off to Cover Shortfall"]:
                df_disp[col] = df_disp[col].apply(lambda x: format_num(x))

            st.dataframe(df_disp, use_container_width=True)

            st.write(
                """
                **Interpretation**: If "% Sell-Off to Cover Shortfall" is, say, 40%, 
                it means selling 40% of this client's **drawn** loan portion 
                would fix the shortfall alone.
                """
            )

st.markdown("---")
st.write(
    """
    **Done**. You've seen:
    1. Editable portfolio & capital usage.
    2. LGD reversion amounts.
    3. Capital & shortfall logic.
    4. A new RCF that might cause a shortfall.
    5. Single-client partial coverage in selected industries for ≤100% sell-off to fix shortfall.
    """
)
