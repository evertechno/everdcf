import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
from fpdf import FPDF
import yfinance as yf
from sklearn.linear_model import LinearRegression

# Caching functions to improve performance
@st.cache_data
def calculate_fcf(data):
    """Calculate Free Cash Flow (FCF) based on the income statement and cash flow statement"""
    required_columns = ['Operating Income', 'Taxes', 'Depreciation', 'Capital Expenditures', 'Changes in Working Capital']
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")
    fcf = data['Operating Income'] - data['Taxes'] + data['Depreciation'] - data['Capital Expenditures'] - data['Changes in Working Capital']
    return fcf

@st.cache_data
def calculate_wacc(cost_of_equity, cost_of_debt, equity_value, debt_value, tax_rate):
    """Calculate Weighted Average Cost of Capital (WACC)"""
    total_value = equity_value + debt_value
    wacc = (equity_value / total_value) * cost_of_equity + (debt_value / total_value) * cost_of_debt * (1 - tax_rate)
    return wacc

@st.cache_data
def calculate_terminal_value(fcf, growth_rate, discount_rate, method='perpetuity'):
    """Calculate Terminal Value using different methods"""
    if method == 'perpetuity':
        terminal_value = fcf * (1 + growth_rate) / (discount_rate - growth_rate)
    elif method == 'exit_multiple':
        exit_multiple = 10  # Example multiple; can be adjusted based on user input
        terminal_value = fcf * exit_multiple
    return terminal_value

@st.cache_data
def dcf_valuation(fcf, wacc, terminal_value, years):
    """Calculate DCF valuation"""
    if years > len(fcf):
        raise ValueError("Forecast years exceed the available data length.")
    discounted_fcf = sum([fcf[i] / (1 + wacc)**(i+1) for i in range(years)])
    discounted_terminal_value = terminal_value / (1 + wacc)**years
    dcf_value = discounted_fcf + discounted_terminal_value
    return dcf_value

@st.cache_data
def sensitivity_analysis(fcf, wacc_range, growth_range, years):
    """Perform sensitivity analysis for DCF valuation"""
    sensitivity_results = []
    for wacc_value in wacc_range:
        for growth_value in growth_range:
            terminal_value = calculate_terminal_value(fcf.iloc[-1], growth_value, wacc_value)
            dcf_value = dcf_valuation(fcf, wacc_value, terminal_value, years)
            sensitivity_results.append((wacc_value, growth_value, dcf_value))
    sensitivity_df = pd.DataFrame(sensitivity_results, columns=['WACC', 'Growth Rate', 'DCF Value'])
    return sensitivity_df

@st.cache_data
def plot_sensitivity_analysis(sensitivity_df):
    """Plot sensitivity analysis results"""
    fig = px.scatter_3d(sensitivity_df, x='WACC', y='Growth Rate', z='DCF Value',
                        title='Sensitivity Analysis',
                        labels={'WACC': 'WACC (%)', 'Growth Rate': 'Growth Rate (%)', 'DCF Value': 'DCF Value ($)'})
    return fig

@st.cache_data
def generate_pdf_report(dcf_value, terminal_value, wacc, fcf_forecast, market_cap, net_income, pe_ratio, debt_equity_ratio, assumptions):
    """Generate a PDF report of the valuation results"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="DCF Valuation Report", ln=True, align='C')

    pdf.cell(200, 10, txt=f"DCF Value: ${dcf_value:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Terminal Value: ${terminal_value:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"WACC: {wacc * 100:.2f}%", ln=True)
    pdf.cell(200, 10, txt="Free Cash Flow Forecast:", ln=True)
    for year, fcf in enumerate(fcf_forecast, start=1):
        pdf.cell(200, 10, txt=f"Year {year}: ${fcf:,.2f}", ln=True)

    pdf.cell(200, 10, txt=f"Market Capitalization: ${market_cap:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Net Income: ${net_income:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"P/E Ratio: {pe_ratio:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Debt/Equity Ratio: {debt_equity_ratio:.2f}", ln=True)

    pdf.cell(200, 10, txt="Assumptions:", ln=True)
    for key, value in assumptions.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    return pdf.output(dest="S").encode("latin1")

@st.cache_data
def fetch_historical_data(ticker):
    """Fetch historical financial data for the given ticker"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")

    # Calculate yearly averages
    hist['Year'] = hist.index.year
    yearly_data = hist.groupby('Year').mean()
    return yearly_data

@st.cache_data
def industry_comparison(ticker, peer_tickers):
    """Compare the company's financial metrics with its peers"""
    peers = [yf.Ticker(t) for t in peer_tickers]
    peer_data = {peer.ticker: peer.history(period="5y").groupby('Year').mean() for peer in peers}

    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    hist['Year'] = hist.index.year
    company_data = hist.groupby('Year').mean()

    return company_data, peer_data

@st.cache_data
def monte_carlo_simulation(initial_value, num_simulations, num_days, vol):
    """Perform Monte Carlo simulations for risk analysis"""
    simulations = []
    for _ in range(num_simulations):
        prices = [initial_value]
        for _ in range(num_days):
            shock = np.random.normal(loc=0, scale=vol)
            prices.append(prices[-1] * (1 + shock))
        simulations.append(prices)
    return np.array(simulations)

@st.cache_data
def calculate_financial_ratios(data):
    """Calculate key financial ratios"""
    ratios = {
        'ROE': data['Net Income'] / data['Total Equity'],
        'ROA': data['Net Income'] / data['Total Assets'],
        'EBITDA Margin': data['EBITDA'] / data['Revenue'],
        'Net Profit Margin': data['Net Income'] / data['Revenue']
    }
    return ratios

def download_template():
    """Generate and provide a download link for the template"""
    template_data = {
        'Year': ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'],
        'Revenue': [None, None, None, None, None],
        'Operating Income': [None, None, None, None, None],
        'Taxes': [None, None, None, None, None],
        'Depreciation': [None, None, None, None, None],
        'Capital Expenditures': [None, None, None, None, None],
        'Changes in Working Capital': [None, None, None, None, None],
    }

    df_template = pd.DataFrame(template_data)

    # Convert the DataFrame to an Excel file and save it to a buffer
    from io import BytesIO
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_template.to_excel(writer, index=False, sheet_name='Financial Data')

    excel_buffer.seek(0)

    # Return as downloadable file
    return excel_buffer

# Streamlit interface
st.set_page_config(page_title="Automated DCF Valuation", layout="wide")
st.title('Automated DCF Valuation')

st.markdown("""
This application performs an **automated DCF valuation** of a company based on uploaded financial statements. You can also perform **sensitivity analysis**, adjust forecast scenarios, and analyze **industry benchmarks**.
""")

# User authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and password == "password":
            st.session_state['authenticated'] = True
        else:
            st.error("Invalid credentials")

if st.session_state['authenticated']:
    # Add custom CSS to hide the header and the top-right buttons
    hide_streamlit_style = """
        <style>
            .css-1r6p8d1 {display: none;} /* Hides the Streamlit logo in the top left */
            .css-1v3t3fg {display: none;} /* Hides the star button */
            .css-1r6p8d1 .st-ae {display: none;} /* Hides the Streamlit logo */
            header {visibility: hidden;} /* Hides the header */
            .css-1tqja98 {visibility: hidden;} /* Hides the header bar */
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Allow user to download the template
    st.download_button(
        label="Download Financial Data Template",
        data=download_template(),
        file_name="financial_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # File uploader for financial statement (CSV or Excel)
    uploaded_file = st.file_uploader("Upload your filled financial statement (CSV/Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        # Reading the uploaded data into a DataFrame
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file, sheet_name="Financial Data")

        st.write("Uploaded Data Preview:", data.head())

        try:
            # Automatically generate FCF and display financial summary
            fcf = calculate_fcf(data)
            st.write(f"Calculated Free Cash Flow: {fcf}")

            # Allow user to input other assumptions for the valuation (Cost of Equity, Cost of Debt, etc.)
            # Ensuring matching numeric types to fix the "StreamlitMixedNumericTypesError"

            # Set types consistently: make sure to use floats when needed
            cost_of_equity = st.number_input('Enter Cost of Equity (%)', min_value=0.0, max_value=100.0, value=8.0, step=0.1, format="%.2f")
            cost_of_debt = st.number_input('Enter Cost of Debt (%)', min_value=0.0, max_value=100.0, value=4.0, step=0.1, format="%.2f")
            equity_value = st.number_input('Enter Total Equity Value ($)', min_value=1.0, value=500000000.0, format="%.2f")
            debt_value = st.number_input('Enter Total Debt Value ($)', min_value=1.0, value=200000000.0, format="%.2f")
            tax_rate = st.number_input('Enter Tax Rate (%)', min_value=0.0, max_value=100.0, value=25.0, step=0.1, format="%.2f") / 100
            growth_rate = st.number_input('Enter Perpetuity Growth Rate (%)', min_value=0.0, max_value=100.0, value=2.5, step=0.1, format="%.2f") / 100
            forecast_years = int(st.number_input('Enter Number of Forecast Years', min_value=1, max_value=10, value=5, format="%.0f"))
            terminal_value_method = st.selectbox('Select Terminal Value Method', ['perpetuity', 'exit_multiple'])

            # Validate forecast years
            if forecast_years > len(fcf):
                st.error("Error: Forecast years exceed the available data length. Please enter a valid number of forecast years.")
            else:
                # Calculate WACC
                wacc = calculate_wacc(cost_of_equity / 100, cost_of_debt / 100, equity_value, debt_value, tax_rate)

                # Calculate Terminal Value
                terminal_value = calculate_terminal_value(fcf.iloc[-1], growth_rate, wacc, method=terminal_value_method)

                # DCF Valuation
                dcf_value = dcf_valuation(fcf, wacc, terminal_value, forecast_years)

                st.write(f"Calculated WACC: {wacc * 100:.2f}%")
                st.write(f"Calculated Terminal Value: ${terminal_value:,.2f}")
                st.write(f"DCF Valuation: ${dcf_value:,.2f}")

                # Sensitivity Analysis
                wacc_range = np.linspace(0.05, 0.15, 11)
                growth_range = np.linspace(0.01, 0.10, 10)
                sensitivity_df = sensitivity_analysis(fcf, wacc_range, growth_range, forecast_years)

                # Plot Sensitivity Analysis
                sensitivity_fig = plot_sensitivity_analysis(sensitivity_df)
                st.plotly_chart(sensitivity_fig)

                # Enhanced Financial Projections Visualization
                st.subheader("Financial Projections & Valuation")

                # Plot Free Cash Flow over the forecast years with bar chart and line chart
                years = np.arange(1, forecast_years + 1)
                fcf_forecast = fcf.head(forecast_years).values  # Use first n years' FCF
                fig_fcf = go.Figure()
                fig_fcf.add_trace(go.Bar(x=years, y=fcf_forecast, name="Free Cash Flow", marker_color='skyblue'))
                fig_fcf.add_trace(go.Scatter(x=years, y=fcf_forecast, mode='lines+markers', name="Free Cash Flow (Line)", marker=dict(color='red')))
                fig_fcf.update_layout(title='Free Cash Flow Projections', xaxis_title='Year', yaxis_title='FCF ($)', template='plotly_white')
                st.plotly_chart(fig_fcf)

                # Plot DCF Valuation
                fig_dcf = px.bar(x=[f'DCF Value'], y=[dcf_value], title='DCF Valuation', labels={'x': 'Valuation Type', 'y': 'Value ($)'})
                st.plotly_chart(fig_dcf)

                # Financial Metrics (Bar chart comparison of key financial metrics)
                st.subheader("Key Financial Metrics")

                # Check if 'Market Capitalization' exists or use manual input
                if 'Market Capitalization' in data.columns:
                    market_cap = data['Market Capitalization'].sum()
                else:
                    market_cap = st.number_input('Enter Market Capitalization ($)', min_value=1.0, value=1000000000.0, format="%.2f")

                # Check if 'Net Income' exists in the uploaded data, and if not, prompt for manual input
                if 'Net Income' in data.columns:
                    net_income = data['Net Income'].sum()  # Sum the Net Income for simplicity
                else:
                    net_income = st.number_input('Enter Net Income ($)', min_value=1.0, value=100000000.0, format="%.2f")

                # Calculate P/E ratio
                pe_ratio = market_cap / net_income  # Price/Earnings ratio
                debt_equity_ratio = data['Total Debt'].sum() / data['Total Equity'].sum() if 'Total Debt' in data.columns and 'Total Equity' in data.columns else 0

                st.write(f"P/E Ratio: {pe_ratio:.2f}")
                st.write(f"Debt/Equity Ratio: {debt_equity_ratio:.2f}")

                # Display other financial metrics
                st.subheader("DCF Assumptions")
                assumptions = {
                    "Cost of Equity": f"{cost_of_equity}%",
                    "Cost of Debt": f"{cost_of_debt}%",
                    "Tax Rate": f"{tax_rate * 100}%",
                    "Perpetuity Growth Rate": f"{growth_rate * 100}%",
                    "Terminal Value Method": terminal_value_method
                }
                for key, value in assumptions.items():
                    st.write(f"{key}: {value}")

                # Historical Data Analysis
                ticker = st.text_input("Enter Stock Ticker for Historical Data Analysis")
                if ticker:
                    historical_data = fetch_historical_data(ticker)
                    st.write("Historical Data:", historical_data)

                    # Linear Regression for Trend Analysis
                    x = np.array(historical_data.index).reshape(-1, 1)
                    y = historical_data['Close'].values
                    model = LinearRegression().fit(x, y)
                    trend = model.predict(x)

                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Historical Close'))
                    fig_trend.add_trace(go.Scatter(x=historical_data.index, y=trend, mode='lines', name='Trend', line=dict(dash='dash')))
                    fig_trend.update_layout(title='Historical Data and Trend Analysis', xaxis_title='Year', yaxis_title='Close Price ($)', template='plotly_white')
                    st.plotly_chart(fig_trend)

                # Industry Comparison
                peer_tickers = st.text_input("Enter Peer Tickers for Industry Comparison (comma separated)").split(',')
                if ticker and peer_tickers:
                    company_data, peer_data = industry_comparison(ticker, peer_tickers)
                    st.write("Company Data:", company_data)
                    for peer_ticker, data in peer_data.items():
                        st.write(f"Peer Data ({peer_ticker}):", data)

                # Monte Carlo Simulation
                initial_value = st.number_input("Enter Initial Value for Monte Carlo Simulation", min_value=1.0, value=100.0)
                num_simulations = st.number_input("Enter Number of Simulations", min_value=1, value=100)
                num_days = st.number_input("Enter Number of Days", min_value=1, value=252)
               
