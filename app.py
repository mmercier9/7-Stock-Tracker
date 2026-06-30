from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh


# ============================================================
# DEFAULT PORTFOLIO SETTINGS
# ============================================================

DEFAULT_PORTFOLIO = [
    {"Stock Symbol": "MDA.TO", "Initial Weighting %": 25.0, "Initial Shares": 10.0},
    {"Stock Symbol": "RKLB", "Initial Weighting %": 20.0, "Initial Shares": 20.0},
    {"Stock Symbol": "LUNR", "Initial Weighting %": 7.5, "Initial Shares": 25.0},
    {"Stock Symbol": "NOC", "Initial Weighting %": 12.5, "Initial Shares": 2.0},
    {"Stock Symbol": "IRDM", "Initial Weighting %": 15.0, "Initial Shares": 15.0},
    {"Stock Symbol": "ASTS", "Initial Weighting %": 7.5, "Initial Shares": 15.0},
    {"Stock Symbol": "LHX", "Initial Weighting %": 12.5, "Initial Shares": 3.0},
]

EASTERN_TZ = ZoneInfo("America/New_York")

MARKET_OPEN_TIME = dt_time(9, 30)
MARKET_CLOSE_TIME = dt_time(16, 0)

COLOR_SEQUENCE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


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
    "Tracks actual intraday portfolio value using number of shares, current prices, percent change, and approximate signed volume."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_ticker(ticker: str) -> str:
    if pd.isna(ticker):
        return ""
    return str(ticker).strip().upper()


def create_default_input_df():
    df = pd.DataFrame(DEFAULT_PORTFOLIO)

    df["Current Price"] = None
    df["Current Value"] = None
    df["Current % Weighting"] = None
    df["Current % Change"] = None
    df["Dollar Gain/Loss Since Open"] = None

    return df


def clean_input_df(input_df):
    df = input_df.copy()

    df["Stock Symbol"] = df["Stock Symbol"].apply(normalize_ticker)

    df["Initial Weighting %"] = pd.to_numeric(
        df["Initial Weighting %"],
        errors="coerce"
    ).fillna(0.0)

    df["Initial Shares"] = pd.to_numeric(
        df["Initial Shares"],
        errors="coerce"
    ).fillna(0.0)

    df = df[df["Stock Symbol"] != ""].copy()
    df = df[df["Initial Shares"] >= 0].copy()
    df = df[df["Initial Weighting %"] >= 0].copy()

    df.reset_index(drop=True, inplace=True)

    return df


def validate_portfolio_input(df):
    if df.empty:
        return False, "Please enter at least one valid stock symbol."

    total_shares = df["Initial Shares"].sum()

    if total_shares <= 0:
        return False, "Total Initial Shares must be greater than zero."

    duplicate_symbols = df[df["Stock Symbol"].duplicated()]["Stock Symbol"].tolist()

    if duplicate_symbols:
        return False, f"Duplicate ticker symbols found: {duplicate_symbols}"

    return True, ""


def filter_regular_market_hours(data):
    if data.empty:
        return data

    data = data.sort_index()

    return data.between_time(
        MARKET_OPEN_TIME,
        MARKET_CLOSE_TIME,
        inclusive="both"
    )


@st.cache_data(ttl=15, show_spinner=False)
def download_intraday_data(symbols_tuple):
    """
    Downloads 1-minute intraday close prices and volume from yfinance.

    Timestamps are converted to Eastern Time and filtered to regular
    market hours only.
    """

    symbols = list(symbols_tuple)
    closes = pd.DataFrame()
    volumes = pd.DataFrame()
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
                prepost=False,
            )

            if data.empty:
                messages.append(f"{symbol}: no intraday data available.")
                continue

            if data.index.tz is None:
                data.index = data.index.tz_localize("UTC").tz_convert(EASTERN_TZ)
            else:
                data.index = data.index.tz_convert(EASTERN_TZ)

            data = filter_regular_market_hours(data)

            if data.empty:
                messages.append(f"{symbol}: no regular-hours intraday data available yet.")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                try:
                    close_series = data["Close"][symbol]
                    volume_series = data["Volume"][symbol]
                except Exception:
                    messages.append(f"{symbol}: could not read Close/Volume data.")
                    continue
            else:
                close_series = data["Close"]
                volume_series = data["Volume"] if "Volume" in data.columns else pd.Series(0, index=data.index)

            combined = pd.DataFrame(
                {
                    "Close": close_series,
                    "Volume": volume_series,
                }
            ).dropna(subset=["Close"])

            if combined.empty:
                messages.append(f"{symbol}: no usable close data available.")
                continue

            closes[symbol] = combined["Close"]
            volumes[symbol] = combined["Volume"].fillna(0)

        except Exception as e:
            messages.append(f"{symbol}: error downloading data: {e}")

    return closes, volumes, messages


def forward_fill_intraday_prices(closes):
    """
    Forward-fills missing 1-minute quote gaps.

    This prevents the portfolio total from dropping sharply when one
    ticker temporarily has no quote at a given minute.
    """

    if closes.empty:
        return closes

    filled_closes = closes.copy().sort_index()
    filled_closes = filled_closes.ffill()

    return filled_closes


def calculate_percent_change(closes):
    """
    Calculates percent change from the first valid regular-market intraday price.

    Missing quote gaps are forward-filled to prevent artificial chart dropouts.
    """

    filled_closes = forward_fill_intraday_prices(closes)

    pct_change = pd.DataFrame(index=filled_closes.index)

    for symbol in filled_closes.columns:
        valid_prices = filled_closes[symbol].dropna()

        if valid_prices.empty:
            continue

        open_price = valid_prices.iloc[0]

        if open_price == 0:
            continue

        pct_change[symbol] = (filled_closes[symbol] - open_price) / open_price * 100

    return pct_change


def calculate_dollar_values(closes, input_df):
    """
    Calculates actual dollar value over time:

        Stock Value = Number of Shares × Current Stock Price

    Missing quote gaps are forward-filled to prevent artificial portfolio dropouts.
    """

    filled_closes = forward_fill_intraday_prices(closes)

    dollar_values = pd.DataFrame(index=filled_closes.index)

    shares_by_symbol = dict(
        zip(
            input_df["Stock Symbol"],
            input_df["Initial Shares"]
        )
    )

    for symbol in filled_closes.columns:
        if symbol not in shares_by_symbol:
            continue

        shares = shares_by_symbol[symbol]

        dollar_values[symbol] = filled_closes[symbol] * shares

    return dollar_values


def calculate_opening_values(closes, input_df):
    """
    Calculates opening value at the first regular-market price:

        Opening Value = Number of Shares × First Regular-Market Price
    """

    filled_closes = forward_fill_intraday_prices(closes)

    opening_values = {}

    shares_by_symbol = dict(
        zip(
            input_df["Stock Symbol"],
            input_df["Initial Shares"]
        )
    )

    for symbol in input_df["Stock Symbol"]:
        if symbol in filled_closes.columns and not filled_closes[symbol].dropna().empty:
            open_price = filled_closes[symbol].dropna().iloc[0]
            shares = shares_by_symbol.get(symbol, 0.0)
            opening_values[symbol] = open_price * shares
        else:
            opening_values[symbol] = None

    return opening_values


def calculate_portfolio_total_value(dollar_values):
    """
    Calculates total portfolio value.

    Forward-fills component values so a missing 1-minute quote does not
    temporarily remove a stock from the portfolio total.
    """

    if dollar_values.empty:
        return pd.Series(dtype=float)

    filled_values = dollar_values.copy().sort_index()
    filled_values = filled_values.ffill()

    return filled_values.sum(axis=1, skipna=False)


def calculate_portfolio_percent_change(portfolio_total_value, initial_total_value):
    if portfolio_total_value.empty or initial_total_value <= 0:
        return pd.Series(dtype=float)

    return (portfolio_total_value - initial_total_value) / initial_total_value * 100


def calculate_signed_volume(closes, volumes):
    """
    Approximates net purchase/sale volume using price direction:

    - If current minute close > prior minute close: positive volume / purchase pressure
    - If current minute close < prior minute close: negative volume / sale pressure
    - If unchanged: zero

    This is not true bid/ask trade classification. It is a practical indicator
    using available yfinance 1-minute OHLCV data.
    """

    if closes.empty or volumes.empty:
        return pd.DataFrame()

    filled_closes = forward_fill_intraday_prices(closes)

    aligned_volumes = volumes.copy().sort_index()
    aligned_volumes = aligned_volumes.reindex(filled_closes.index).fillna(0)

    signed_volume = pd.DataFrame(index=filled_closes.index)

    for symbol in filled_closes.columns:
        if symbol not in aligned_volumes.columns:
            continue

        price_change = filled_closes[symbol].diff()

        direction = price_change.apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )

        signed_volume[symbol] = aligned_volumes[symbol] * direction

    return signed_volume


def calculate_signed_dollar_volume(closes, signed_volume):
    """
    Converts signed share volume to signed dollar volume:

        Signed Dollar Volume = Signed Share Volume × Current Price
    """

    if closes.empty or signed_volume.empty:
        return pd.DataFrame()

    filled_closes = forward_fill_intraday_prices(closes)

    signed_dollar_volume = pd.DataFrame(index=signed_volume.index)

    for symbol in signed_volume.columns:
        if symbol not in filled_closes.columns:
            continue

        price_series = filled_closes[symbol].reindex(signed_volume.index).ffill()

        signed_dollar_volume[symbol] = signed_volume[symbol] * price_series

    return signed_dollar_volume


def calculate_portfolio_signed_dollar_volume(signed_dollar_volume):
    """
    Sums signed dollar volume across all tickers to estimate overall
    portfolio-level buy/sell pressure.
    """

    if signed_dollar_volume.empty:
        return pd.Series(dtype=float)

    return signed_dollar_volume.sum(axis=1, skipna=True)


def update_tracking_table(input_df, closes, dollar_values, pct_change, opening_values):
    """
    Adds current price, current value, current weighting, percent change,
    and intraday gain/loss to the user input table.
    """

    filled_closes = forward_fill_intraday_prices(closes)

    output_df = input_df.copy()

    current_prices = {}
    current_values = {}
    current_pct_changes = {}

    for symbol in output_df["Stock Symbol"]:
        if symbol in filled_closes.columns and not filled_closes[symbol].dropna().empty:
            current_prices[symbol] = filled_closes[symbol].dropna().iloc[-1]
        else:
            current_prices[symbol] = None

        if symbol in dollar_values.columns and not dollar_values[symbol].dropna().empty:
            current_values[symbol] = dollar_values[symbol].dropna().iloc[-1]
        else:
            current_values[symbol] = None

        if symbol in pct_change.columns and not pct_change[symbol].dropna().empty:
            current_pct_changes[symbol] = pct_change[symbol].dropna().iloc[-1]
        else:
            current_pct_changes[symbol] = None

    current_total_value = sum(
        value for value in current_values.values()
        if value is not None and not pd.isna(value)
    )

    current_price_list = []
    current_value_list = []
    current_weighting_list = []
    current_pct_change_list = []
    gain_loss_list = []

    for _, row in output_df.iterrows():
        symbol = row["Stock Symbol"]

        current_price = current_prices.get(symbol)
        current_value = current_values.get(symbol)
        current_pct_change = current_pct_changes.get(symbol)
        opening_value = opening_values.get(symbol)

        if current_value is not None and current_total_value > 0:
            current_weighting = current_value / current_total_value * 100
        else:
            current_weighting = None

        if current_value is not None and opening_value is not None:
            gain_loss = current_value - opening_value
        else:
            gain_loss = None

        current_price_list.append(current_price)
        current_value_list.append(current_value)
        current_weighting_list.append(current_weighting)
        current_pct_change_list.append(current_pct_change)
        gain_loss_list.append(gain_loss)

    output_df["Current Price"] = current_price_list
    output_df["Current Value"] = current_value_list
    output_df["Current % Weighting"] = current_weighting_list
    output_df["Current % Change"] = current_pct_change_list
    output_df["Dollar Gain/Loss Since Open"] = gain_loss_list

    return output_df


def get_market_open_close_for_chart(index):
    if index.empty:
        chart_day = datetime.now(EASTERN_TZ).date()
    else:
        chart_day = index.max().date()

    market_open = datetime.combine(
        chart_day,
        MARKET_OPEN_TIME,
        tzinfo=EASTERN_TZ
    )

    market_close = datetime.combine(
        chart_day,
        MARKET_CLOSE_TIME,
        tzinfo=EASTERN_TZ
    )

    return market_open, market_close


def make_dual_axis_chart(dollar_values, pct_change, portfolio_total_value, portfolio_pct_change):
    """
    Creates one chart:
    - Left y-axis: total portfolio dollar value only
    - Right y-axis: percent changes for each stock and the total portfolio
    - Individual stock percent lines are dashed and colored
    - Portfolio total value is solid black
    - Portfolio percent change is dashed black
    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    color_by_symbol = {}

    for idx, symbol in enumerate(pct_change.columns):
        color_by_symbol[symbol] = COLOR_SEQUENCE[idx % len(COLOR_SEQUENCE)]

    if not portfolio_total_value.empty:
        fig.add_trace(
            go.Scatter(
                x=portfolio_total_value.index,
                y=portfolio_total_value,
                mode="lines",
                name="Portfolio $",
                line=dict(
                    color="black",
                    width=4.0,
                    dash="solid",
                ),
                opacity=1.0,
            ),
            secondary_y=False,
        )

    for symbol in pct_change.columns:
        color = color_by_symbol[symbol]

        fig.add_trace(
            go.Scatter(
                x=pct_change.index,
                y=pct_change[symbol],
                mode="lines",
                name=f"{symbol} %",
                line=dict(
                    color=color,
                    width=1.6,
                    dash="dash",
                ),
                opacity=0.65,
            ),
            secondary_y=True,
        )

    if not portfolio_pct_change.empty:
        fig.add_trace(
            go.Scatter(
                x=portfolio_pct_change.index,
                y=portfolio_pct_change,
                mode="lines",
                name="Portfolio %",
                line=dict(
                    color="black",
                    width=2.8,
                    dash="dash",
                ),
                opacity=0.75,
            ),
            secondary_y=True,
        )

    fig.add_hline(y=0, line_width=1, secondary_y=True)

    eastern_now = datetime.now(EASTERN_TZ)
    market_open, market_close = get_market_open_close_for_chart(portfolio_total_value.index)

    latest_portfolio_value = None
    latest_portfolio_pct = None

    if not portfolio_total_value.empty:
        latest_portfolio_value = portfolio_total_value.dropna().iloc[-1]

    if not portfolio_pct_change.empty:
        latest_portfolio_pct = portfolio_pct_change.dropna().iloc[-1]

    title_value = (
        f"${latest_portfolio_value:,.2f}"
        if latest_portfolio_value is not None
        else "Unavailable"
    )

    title_pct = (
        f"{latest_portfolio_pct:+.2f}%"
        if latest_portfolio_pct is not None
        else "Unavailable"
    )

    fig.update_layout(
        title=dict(
            text=(
                "Intraday Portfolio Value and Stock Percent Change"
                f"<br><sup>Portfolio Value: {title_value} | "
                f"Portfolio Change Since Open: {title_pct} | "
                f"Baseline resets at 9:30 AM ET | "
                f"Updated: {eastern_now.strftime('%Y-%m-%d %I:%M:%S %p ET')}</sup>"
            ),
            x=0.01,
            xanchor="left",
        ),
        hovermode="x unified",
        height=820,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            font=dict(size=11),
            traceorder="normal",
        ),
        margin=dict(
            l=70,
            r=80,
            t=105,
            b=150,
        ),
    )

    fig.update_xaxes(
        title_text="Time — Eastern",
        tickformat="%I:%M %p",
        range=[market_open, market_close],
    )

    fig.update_yaxes(
        title_text="Portfolio Dollar Value",
        tickprefix="$",
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="Percent Change Since Open",
        ticksuffix="%",
        secondary_y=True,
    )

    return fig


def make_volume_indicator_chart(signed_volume, portfolio_signed_dollar_volume):
    """
    Creates a row-based volume pressure chart:
    - One row for each stock symbol
    - One final row for overall portfolio buy/sell pressure
    - Green bars = estimated purchase pressure
    - Red bars = estimated sale pressure
    """

    if signed_volume.empty:
        return go.Figure()

    symbols = list(signed_volume.columns)
    row_titles = symbols + ["Portfolio"]

    fig = make_subplots(
        rows=len(row_titles),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_titles=row_titles,
    )

    # Individual symbol signed volume rows
    for row_index, symbol in enumerate(symbols, start=1):
        series = signed_volume[symbol].fillna(0)

        colors = [
            "green" if value > 0 else ("red" if value < 0 else "gray")
            for value in series
        ]

        fig.add_trace(
            go.Bar(
                x=series.index,
                y=series,
                marker_color=colors,
                name=f"{symbol} signed volume",
                showlegend=False,
            ),
            row=row_index,
            col=1,
        )

        fig.add_hline(
            y=0,
            line_width=1,
            line_color="gray",
            row=row_index,
            col=1,
        )

    # Portfolio signed dollar volume row
    portfolio_row = len(row_titles)

    if not portfolio_signed_dollar_volume.empty:
        portfolio_series = portfolio_signed_dollar_volume.fillna(0)

        portfolio_colors = [
            "green" if value > 0 else ("red" if value < 0 else "gray")
            for value in portfolio_series
        ]

        fig.add_trace(
            go.Bar(
                x=portfolio_series.index,
                y=portfolio_series,
                marker_color=portfolio_colors,
                name="Portfolio signed dollar volume",
                showlegend=False,
            ),
            row=portfolio_row,
            col=1,
        )

        fig.add_hline(
            y=0,
            line_width=1,
            line_color="gray",
            row=portfolio_row,
            col=1,
        )

    eastern_now = datetime.now(EASTERN_TZ)
    market_open, market_close = get_market_open_close_for_chart(signed_volume.index)

    fig.update_layout(
        title=dict(
            text=(
                "Estimated Purchase / Sale Volume Pressure"
                f"<br><sup>Green = estimated purchase pressure | "
                f"Red = estimated sale pressure | "
                f"Updated: {eastern_now.strftime('%Y-%m-%d %I:%M:%S %p ET')}</sup>"
            ),
            x=0.01,
            xanchor="left",
        ),
        height=max(520, 115 * len(row_titles)),
        margin=dict(
            l=90,
            r=50,
            t=95,
            b=60,
        ),
        bargap=0.05,
    )

    fig.update_xaxes(
        tickformat="%I:%M %p",
        range=[market_open, market_close],
    )

    fig.update_yaxes(
        title_text="Signed volume",
        showticklabels=True,
    )

    # Label the portfolio row differently because it is signed dollar volume
    fig.update_yaxes(
        title_text="Signed $ volume",
        row=portfolio_row,
        col=1,
    )

    return fig


def format_tracking_table_for_display(df):
    display_df = df.copy()

    money_columns = [
        "Current Price",
        "Current Value",
        "Dollar Gain/Loss Since Open",
    ]

    percent_columns = [
        "Initial Weighting %",
        "Current % Weighting",
        "Current % Change",
    ]

    share_columns = [
        "Initial Shares",
    ]

    for column in money_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda x: "" if pd.isna(x) else f"${x:,.2f}"
            )

    for column in percent_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda x: "" if pd.isna(x) else f"{x:.2f}%"
            )

    for column in share_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda x: "" if pd.isna(x) else f"{x:,.4f}"
            )

    return display_df


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Controls")

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
    "The chart uses regular market hours only: 9:30 AM to 4:00 PM ET."
)

st.sidebar.caption(
    "MDA may need to be entered as MDA.TO for Yahoo Finance."
)

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
# DATA ENTRY PANEL
# ============================================================

st.subheader("Portfolio Data Entry and Current Tracking")

# Use v5 session key so Streamlit does not reuse old table format from prior versions.
if "portfolio_input_df_v5" not in st.session_state:
    st.session_state["portfolio_input_df_v5"] = create_default_input_df()

entry_df = st.data_editor(
    st.session_state["portfolio_input_df_v5"],
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Stock Symbol": st.column_config.TextColumn(
            "Stock Symbol",
            help="Ticker symbol used by Yahoo Finance, for example RKLB, LHX, or MDA.TO.",
            required=True,
        ),
        "Initial Weighting %": st.column_config.NumberColumn(
            "Initial Weighting %",
            min_value=0.0,
            step=0.5,
            format="%.2f",
        ),
        "Initial Shares": st.column_config.NumberColumn(
            "Initial Shares",
            min_value=0.0,
            step=1.0,
            format="%.4f",
        ),
        "Current Price": st.column_config.NumberColumn(
            "Current Price",
            format="$%.2f",
            disabled=True,
        ),
        "Current Value": st.column_config.NumberColumn(
            "Current Value",
            format="$%.2f",
            disabled=True,
        ),
        "Current % Weighting": st.column_config.NumberColumn(
            "Current % Weighting",
            format="%.2f%%",
            disabled=True,
        ),
        "Current % Change": st.column_config.NumberColumn(
            "Current % Change",
            format="%.2f%%",
            disabled=True,
        ),
        "Dollar Gain/Loss Since Open": st.column_config.NumberColumn(
            "Dollar Gain/Loss Since Open",
            format="$%.2f",
            disabled=True,
        ),
    },
    disabled=[
        "Current Price",
        "Current Value",
        "Current % Weighting",
        "Current % Change",
        "Dollar Gain/Loss Since Open",
    ],
    key="portfolio_editor_v5",
)

input_df = clean_input_df(entry_df)
is_valid, validation_message = validate_portfolio_input(input_df)

if not is_valid:
    st.error(validation_message)
    st.stop()

st.session_state["portfolio_input_df_v5"] = entry_df.copy()

total_initial_weight = input_df["Initial Weighting %"].sum()

if abs(total_initial_weight - 100.0) > 0.01:
    st.warning(
        f"Initial Weighting totals {total_initial_weight:.2f}%, not 100%. "
        "This does not stop the tracker, because actual portfolio value is based on Initial Shares."
    )
else:
    st.success("Initial Weighting totals 100%.")

symbols = tuple(input_df["Stock Symbol"].tolist())


# ============================================================
# MARKET DATA
# ============================================================

with st.spinner("Downloading intraday market data..."):
    closes, volumes, messages = download_intraday_data(symbols)

if messages:
    with st.expander("Data messages", expanded=False):
        for message in messages:
            st.write(message)

if closes.empty:
    st.warning(
        "No regular-hours intraday data available for the selected tickers. "
        "This can happen before 9:30 AM ET, after market close, on weekends, "
        "or if Yahoo Finance has not published data yet."
    )
    st.stop()

pct_change = calculate_percent_change(closes)

dollar_values = calculate_dollar_values(
    closes=closes,
    input_df=input_df,
)

if dollar_values.empty:
    st.warning("Could not calculate dollar values from the downloaded data.")
    st.stop()

opening_values = calculate_opening_values(
    closes=closes,
    input_df=input_df,
)

initial_total_value = sum(
    value for value in opening_values.values()
    if value is not None and not pd.isna(value)
)

portfolio_total_value = calculate_portfolio_total_value(dollar_values)

portfolio_pct_change = calculate_portfolio_percent_change(
    portfolio_total_value=portfolio_total_value,
    initial_total_value=initial_total_value,
)

tracking_df = update_tracking_table(
    input_df=input_df,
    closes=closes,
    dollar_values=dollar_values,
    pct_change=pct_change,
    opening_values=opening_values,
)

signed_volume = calculate_signed_volume(
    closes=closes,
    volumes=volumes,
)

signed_dollar_volume = calculate_signed_dollar_volume(
    closes=closes,
    signed_volume=signed_volume,
)

portfolio_signed_dollar_volume = calculate_portfolio_signed_dollar_volume(
    signed_dollar_volume=signed_dollar_volume,
)


# ============================================================
# TOP METRICS
# ============================================================

latest_portfolio_value = portfolio_total_value.dropna().iloc[-1]

if not portfolio_pct_change.empty:
    latest_portfolio_pct = portfolio_pct_change.dropna().iloc[-1]
else:
    latest_portfolio_pct = 0.0

latest_gain_loss = latest_portfolio_value - initial_total_value

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Opening portfolio value",
        f"${initial_total_value:,.2f}",
    )

with col2:
    st.metric(
        "Current portfolio value",
        f"${latest_portfolio_value:,.2f}",
        delta=f"${latest_gain_loss:,.2f}",
    )

with col3:
    st.metric(
        "Portfolio % change since open",
        f"{latest_portfolio_pct:+.2f}%",
    )

with col4:
    st.metric(
        "Active tickers",
        len(dollar_values.columns),
    )

with col5:
    if auto_refresh:
        st.metric("Refresh count", refresh_count)
    else:
        st.metric("Auto-refresh", "Off")


# ============================================================
# UPDATED TRACKING TABLE
# ============================================================

st.subheader("Current Portfolio Status")

display_tracking_df = format_tracking_table_for_display(tracking_df)

st.dataframe(
    display_tracking_df,
    use_container_width=True,
    hide_index=True,
)

csv_data = tracking_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download current portfolio status CSV",
    data=csv_data,
    file_name="current_portfolio_status.csv",
    mime="text/csv",
)


# ============================================================
# TREND CHART
# ============================================================

fig = make_dual_axis_chart(
    dollar_values=dollar_values,
    pct_change=pct_change,
    portfolio_total_value=portfolio_total_value,
    portfolio_pct_change=portfolio_pct_change,
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# VOLUME INDICATOR CHART
# ============================================================

st.subheader("Estimated Purchase / Sale Volume Pressure")

st.caption(
    "This indicator approximates buy/sell pressure using minute-to-minute price direction. "
    "Green bars mean price rose during that minute, red bars mean price fell during that minute. "
    "This is not true bid/ask order-flow data."
)

volume_fig = make_volume_indicator_chart(
    signed_volume=signed_volume,
    portfolio_signed_dollar_volume=portfolio_signed_dollar_volume,
)

st.plotly_chart(volume_fig, use_container_width=True)


# ============================================================
# STATUS MESSAGE
# ============================================================

st.caption(
    "Dollar values are calculated as Initial Shares × Current Price. "
    "Missing 1-minute quote gaps are forward-filled to avoid artificial portfolio dropouts. "
    "Percent change is measured from the first regular-market price at or after 9:30 AM ET."
)

if auto_refresh:
    st.caption(
        f"Auto-refresh is on. Refreshing every {refresh_seconds} seconds."
    )
else:
    st.caption(
        "Auto-refresh is off. Use Refresh now to update the chart."
    )
