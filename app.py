import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import streamlit.components.v1 as components

# -------- PAGE CONFIG --------
st.set_page_config(page_title="PrediStock", layout="wide")

# Hide Streamlit menu & footer
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("📈 Stock Forecast System")

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("<h1 style='margin-bottom:5px;'>PrediStock</h1>", unsafe_allow_html=True)
st.sidebar.title("Settings")

n_years = st.sidebar.slider("Years of prediction:", 1, 4)
period = n_years * 365

chart_type = st.sidebar.selectbox(
    "Select Chart Type",
    ("Line Chart", "Candlestick")
)

show_raw = st.sidebar.checkbox("Show Raw Data", value=False)
show_confidence = st.sidebar.checkbox("Show Forecast Confidence", value=False)

# HOME BUTTON
st.sidebar.markdown("""
<a href="https://predistocks01.netlify.app/" target="_blank">
<button style="width:100%; padding:10px; background-color:#4CAF50; color:white; border:none; border-radius:5px;">
Home
</button>
</a>
""", unsafe_allow_html=True)

# -------- STOCK SEARCH FUNCTION --------
def search_stock(query):
    try:
        search = yf.Search(query, max_results=8)
        results = search.quotes
        suggestions = []

        for r in results:
            name = r.get("shortname", "")
            symbol = r.get("symbol", "")
            exchange = r.get("exchange", "")

            if exchange in ["NSI", "NSE", "BSE"]:
                suggestions.append(f"{name} ({symbol})")

        return suggestions

    except:
        return []

# -------- SEARCH INPUT --------
query = st.text_input("🔎 Search Company Name (Example: Zomato, Reliance, Tata)")

suggestions = []

if query:
    suggestions = search_stock(query)

selected_stock = None

if suggestions:
    selected_stock = st.selectbox("Suggestions", suggestions)

stock_symbol = None

if selected_stock:
    stock_symbol = selected_stock.split("(")[-1].replace(")", "")

# -------- STOCK DATA --------
if stock_symbol:

    try:
        stock = yf.Ticker(stock_symbol)
        info = stock.info

        st.subheader(info.get("shortName", stock_symbol))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Sector:", info.get("sector", "N/A"))

        with col2:
            market_cap = info.get("marketCap")
            if market_cap:
                st.write(f"Market Cap: ₹ {market_cap:,.0f}")

        with col3:
            current_price_data = stock.history(period="1d")
            if not current_price_data.empty:
                current_price = current_price_data.iloc[-1]["Close"]
                st.write(f"Current Price: ₹ {current_price:,.2f}")

        st.write(info.get("longBusinessSummary", "No description available.")[:400] + "...")

        # -------- LOAD DATA --------
        @st.cache_data
        def load_data(ticker):
            data = yf.download(ticker, period="max")

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data.reset_index(inplace=True)
            data = data.dropna()

            return data

        data = load_data(stock_symbol)

        if data.empty:
            st.error("No stock data found.")
            st.stop()

        # -------- RAW DATA --------
        if show_raw:
            st.subheader("Raw Data")
            st.write(data.tail())

        # -------- CHART --------
        st.subheader("📊 Stock Price Chart")

        fig = go.Figure()

        if chart_type == "Line Chart":

            fig.add_trace(
                go.Scatter(x=data["Date"], y=data["Open"], name="Open")
            )

            fig.add_trace(
                go.Scatter(x=data["Date"], y=data["Close"], name="Close")
            )

        else:

            fig.add_trace(
                go.Candlestick(
                    x=data["Date"],
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"]
                )
            )

        fig.update_layout(
            title="Stock Price",
            xaxis_rangeslider_visible=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------- FORECAST --------
        st.subheader("🔮 Stock Forecast")

        df_train = data[["Date", "Close"]]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        model = Prophet()
        model.fit(df_train)

        future = model.make_future_dataframe(periods=period)

        forecast = model.predict(future)

        st.write(forecast.tail())

        # Forecast plot
        if show_confidence:
            fig1 = plot_plotly(model, forecast)
        else:
            forecast["yhat_upper"] = forecast["yhat"]
            forecast["yhat_lower"] = forecast["yhat"]
            fig1 = plot_plotly(model, forecast)

        st.plotly_chart(fig1, use_container_width=True)

        # Forecast components
        st.subheader("Forecast Components")

        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # -------- INVESTMENT INSIGHT --------
        st.subheader("💡 Investment Insight")

        current_price = data.iloc[-1]["Close"]
        initial_price = data.iloc[0]["Close"]

        diff = current_price - initial_price
        percent_change = (diff / initial_price) * 100

        if diff > 0:

            st.success(
                f"Stock grew ₹ {diff:,.2f} ({percent_change:.2f}%). "
                f"Possible investment opportunity."
            )

            if st.button(f"Buy {stock_symbol}"):

                zerodha_symbol = stock_symbol.replace(".NS", "")

                zerodha_url = f"https://kite.zerodha.com/?symbol=NSE:{zerodha_symbol}"

                components.html(
                    f'<script>window.open("{zerodha_url}", "_blank");</script>',
                    height=0
                )

        elif diff < 0:

            st.error(
                f"Stock declined ₹ {abs(diff):,.2f} ({abs(percent_change):.2f}%). "
                "Better wait for recovery."
            )

        else:

            st.warning("Minimal price movement detected.")

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Type a company name to search stocks.")
