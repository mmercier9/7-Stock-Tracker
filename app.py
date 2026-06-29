from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh


# ============================================================
# DEFAULT PORTFOLIO SETTINGS
# ============================================================

DEFAULT_PORTFOLIO = [
    ("MDA.TO", 25.0),
    ("RKLB", 20.0),
    ("LUNR", 7.5),
    ("NOC", 12.5),
    ("IRDM", 15.0),
    ("ASTS", 7.5),
    ("LHX", 12.5),
]

EASTERN_TZ = ZoneInfo("America/New_York")


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Weighted Stock Portfolio Tracker",
    page_icon="📈",
    layout="wide",
)

st.title("Weighted Stock Portfolio Tracker")
st.caption(
    "Intraday tracker using individual stock percent changes and a weighted portfolio fund line."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def build_portfolio_from_inputs(tickers, weights):
    portfolio = {}

    for ticker, weight in zip(tickers, weights):
        ticker = normalize_ticker(ticker)

        if ticker:
            portfolio[ticker] = float(weight)

    return portfolio


def validate_portfolio(portfolio):
    if not portfolio:
        return False, "Please enter at least one ticker."

    total_weight = sum(portfolio.values())

    if total_weight <= 0:
        return False, "Total portfolio weight must be greater than zero."

    for ticker, weight in portfolio.items():
        if weight < 0:
            return False, f"{ticker} has a negative weight. Please use zero or a positive number."

    return True, ""


@st.cache_data(ttl=15, show_spinner=False)
def download_intraday_data(symbols_tuple):
    """
    Downloads 1-minute intraday close prices from yfinance.

    The cache TTL is intentionally short so auto-refresh can request
    updated data without repeatedly hammering the data source.
    """

    symbols = list(symbols_tuple)
    closes = pd.DataFrame()
    messages = []

    for symbol in symbols:
        try:
            data = yf.download(
                tickers=symbol,
                period="1d",
                interval="1m",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if data.empty:
                messages.append(f"{symbol}: no intraday data available.")
                continue

            # Convert timestamps to US Eastern Time.
            if data.index.tz is None:
                data.index = data.index.tz_localize("UTC").tz_convert(EASTERN_TZ)
            else:
                data.index = data.index.tz_convert(EASTERN_TZ)

            # Handle either normal or MultiIndex columns.
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    close_series = data["Close"][symbol]
                except Exception:
                    messages.append(f"{symbol}: could not read Close data.")
                    continue
            else:
                close_series = data["Close"]

            close_series = close_series.dropna()

            if close_series.empty:
                messages.append(f"{symbol}: no close data available.")
                continue

            closes[symbol] = close_series

        except Exception as e:
            messages.append(f"{symbol}: error downloading data: {e}")

    return closes, messages


def calculate_percent_change(closes):
    """
    Calculates percent change from first valid intraday price.
    """

    pct_change = pd.DataFrame(index=closes.index)

    for symbol in closes.columns:
        valid_prices = closes[symbol].dropna()

        if valid_prices.empty:
            continue

        start_price = valid_prices.iloc[0]

        if start_price == 0:
            continue

        pct_change[symbol] = (closes[symbol] - start_price) / start_price * 100

    return pct_change


def calculate_weighted_portfolio_change(pct_change, portfolio_weights):
    """
    Calculates weighted portfolio/fund percent change.

    Weights are normalized automatically, so the program still works
    if the entered weights do not add exactly to 100%.
    """

    if pct_change.empty:
        return pd.Series(dtype=float)

    total_weight = sum(portfolio_weights.values())

    if total_weight <= 0:
        return pd.Series(dtype=float)

    weighted_change = pd.Series(0.0, index=pct_change.index)

    for symbol in pct_change.columns:
        if symbol not in portfolio_weights:
            continue

        weight_decimal = portfolio_weights[symbol] / total_weight

        weighted_change = weighted_change.add(
            pct_change[symbol] * weight_decimal,
            fill_value=0,
        )

    return weighted_change


def make_chart(pct_change, weighted_portfolio_change, latest_portfolio_change):
    fig = go.Figure()

    # Individual stock lines
    for symbol in pct_change.columns:
        fig.add_trace(
            go.Scatter(
                x=pct_change.index,
                y=pct_change[symbol],
                mode="lines",
                name=symbol,
                line=dict(width=1.3),
                opacity=0.65,
            )
        )

    # Weighted portfolio line
    if not weighted_portfolio_change.empty:
        fig.add_trace(
            go.Scatter(
                x=weighted_portfolio_change.index,
                y=weighted_portfolio_change,
                mode="lines",
                name="Weighted Portfolio Fund",
                line=dict(width=4, color="black"),
            )
        )

    fig.add_hline(y=0, line_width=1)

    eastern_now = datetime.now(EASTERN_TZ)

    fig.update_layout(
        title=(
            "Intraday Performance — Individual Stocks + Weighted Portfolio Fund"
            f"<br><sup>Weighted Portfolio Change: {latest_portfolio_change:+.2f}% | "
            f"Updated: {eastern_now.strftime('%Y-%m-%d %I:%M:%S %p ET')}</sup>"
        ),
        xaxis_title="Time — Eastern",
        yaxis_title="% Change Since First Intraday Price",
        hovermode="x unified",
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=40, r=40, t=100, b=40),
    )

    fig.update_xaxes(tickformat="%I:%M %p")

    return fig


def make_summary_table(portfolio, pct_change):
    rows = []
    total_weight = sum(portfolio.values())

    for symbol, weight in portfolio.items():
        latest_change = None

        if symbol in pct_change.columns and not pct_change[symbol].dropna().empty:
            latest_change = pct_change[symbol].dropna().iloc[-1]

        normalized_weight = weight / total_weight * 100 if total_weight > 0 else 0

        contribution = (
            latest_change * normalized_weight / 100
            if latest_change is not None
            else None
        )

        rows.append(
            {
                "Ticker": symbol,
                "Entered Weight %": weight,
                "Normalized Weight %": normalized_weight,
                "Latest % Change": latest_change,
                "Weighted Contribution": contribution,
            }
        )

    summary_df = pd.DataFrame(rows)

    return summary_df


def format_summary_for_display(summary_df):
    display_df = summary_df.copy()

    numeric_columns = [
        "Entered Weight %",
        "Normalized Weight %",
        "Latest % Change",
        "Weighted Contribution",
    ]

    for column in numeric_columns:
        display_df[column] = display_df[column].map(
            lambda x: "" if pd.isna(x) else f"{x:.2f}"
        )

    return display_df


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Portfolio Inputs")

ticker_inputs = []
weight_inputs = []

for i, (default_ticker, default_weight) in enumerate(DEFAULT_PORTFOLIO, start=1):
    col1, col2 = st.sidebar.columns([1.2, 1])

    with col1:
        ticker_value = st.text_input(
            f"Ticker {i}",
            value=default_ticker,
            key=f"ticker_{i}",
        )

    with col2:
        weight_value = st.number_input(
            f"Weight {i} %",
            value=float(default_weight),
            min_value=0.0,
            step=0.5,
            key=f"weight_{i}",
        )

    ticker_inputs.append(ticker_value)
    weight_inputs.append(weight_value)

portfolio_weights = build_portfolio_from_inputs(ticker_inputs, weight_inputs)
is_valid, validation_message = validate_portfolio(portfolio_weights)

total_entered_weight = sum(portfolio_weights.values())

st.sidebar.divider()
st.sidebar.metric("Total entered weight", f"{total_entered_weight:.2f}%")

if abs(total_entered_weight - 100.0) > 0.01:
    st.sidebar.warning(
        "Weights do not add to 100%. The app will normalize them automatically."
    )
else:
    st.sidebar.success("Weights add to 100%.")

refresh_seconds = st.sidebar.number_input(
    "Auto-refresh seconds",
    min_value=15,
    max_value=600,
    value=60,
    step=15,
)

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

manual_refresh = st.sidebar.button("Refresh now")

st.sidebar.caption(
    "Note: MDA may need to be entered as MDA.TO for Yahoo Finance."
)

# ============================================================
# AUTO-REFRESH CONTROL
# ============================================================

if auto_refresh:
    refresh_count = st_autorefresh(
        interval=int(refresh_seconds * 1000),
        key="portfolio_auto_refresh",
    )
else:
    refresh_count = 0

if manual_refresh:
    st.cache_data.clear()


# ============================================================
# MAIN APP
# ============================================================

if not is_valid:
    st.error(validation_message)
    st.stop()

symbols = tuple(portfolio_weights.keys())

with st.spinner("Downloading intraday market data..."):
    closes, messages = download_intraday_data(symbols)

if messages:
    with st.expander("Data messages", expanded=False):
        for message in messages:
            st.write(message)

if closes.empty:
    st.warning("No intraday data available for the selected tickers.")
    st.stop()

pct_change = calculate_percent_change(closes)

if pct_change.empty:
    st.warning("Could not calculate percent changes from the downloaded data.")
    st.stop()

weighted_portfolio_change = calculate_weighted_portfolio_change(
    pct_change=pct_change,
    portfolio_weights=portfolio_weights,
)

if weighted_portfolio_change.empty:
    latest_portfolio_change = 0.0
else:
    latest_portfolio_change = weighted_portfolio_change.dropna().iloc[-1]

# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Weighted portfolio change",
        f"{latest_portfolio_change:+.2f}%",
    )

with col2:
    st.metric(
        "Active tickers",
        len(pct_change.columns),
    )

with col3:
    st.metric(
        "Last update",
        datetime.now(EASTERN_TZ).strftime("%I:%M:%S %p ET"),
    )

with col4:
    if auto_refresh:
        st.metric(
            "Refresh count",
            refresh_count,
        )
    else:
        st.metric(
            "Auto-refresh",
            "Off",
        )

# ============================================================
# CHART
# ============================================================

fig = make_chart(
    pct_change=pct_change,
    weighted_portfolio_change=weighted_portfolio_change,
    latest_portfolio_change=latest_portfolio_change,
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SUMMARY TABLE
# ============================================================

summary_df = make_summary_table(
    portfolio=portfolio_weights,
    pct_change=pct_change,
)

st.subheader("Portfolio Summary")

display_df = format_summary_for_display(summary_df)

st.dataframe(display_df, use_container_width=True, hide_index=True)

csv_data = summary_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download portfolio summary CSV",
    data=csv_data,
    file_name="weighted_portfolio_summary.csv",
    mime="text/csv",
)

# ============================================================
# STATUS MESSAGE
# ============================================================

if auto_refresh:
    st.caption(f"Auto-refresh is on. Refreshing every {refresh_seconds} seconds.")
else:
    st.caption("Auto-refresh is off. Use Refresh now to update the chart.")
