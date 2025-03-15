import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import requests
import time

# Helper function to fetch data from Alpha Vantage
def fetch_alpha_vantage_data(ticker, api_key, data_type='TIME_SERIES_DAILY', interval='1d', outputsize='compact'):
    """Fetch financial data from Alpha Vantage API."""
    base_url = f'https://www.alphavantage.co/query?function={data_type}&symbol={ticker}&interval={interval}&apikey={api_key}'
    if data_type == 'TIME_SERIES_DAILY':
        url = f'{base_url}&outputsize={outputsize}'
    else:
        url = base_url
    
    try:
        response = requests.get(url)
        data = response.json()
        if 'Time Series (Daily)' in data:
            return data['Time Series (Daily)']
        elif 'Time Series (1min)' in data:
            return data['Time Series (1min)']
        elif 'Monthly Adjusted Time Series' in data:
            return data['Monthly Adjusted Time Series']
        elif 'quarterlyReports' in data:
            return data['quarterlyReports']
        else:
            st.error(f"Error fetching data: {data.get('Note', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None

# Helper functions for DCF Calculation
def calculate_fcf(data):
    """Calculate Free Cash Flow (FCF) based on the income statement and cash flow statement"""
    fcf = data['Operating Income'] - data['Taxes'] + data['Depreciation'] - data['Capital Expenditures'] - data['Changes in Working Capital']
    return fcf

def calculate_wacc(cost_of_equity, cost_of_debt, equity_value, debt_value, tax_rate):
    """Calculate Weighted Average Cost of Capital (WACC)"""
    total_value = equity_value + debt_value
    wacc = (equity_value / total_value) * cost_of_equity + (debt_value / total_value) * cost_of_debt * (1 - tax_rate)
    return wacc

def calculate_terminal_value(fcf, growth_rate, discount_rate, method='perpetuity'):
    """Calculate Terminal Value using different methods"""
    if method == 'perpetuity':
        terminal_value = fcf * (1 + growth_rate) / (discount_rate - growth_rate)
    elif method == 'exit_multiple':
        exit_multiple = 10  # Example multiple; can be adjusted based on user input
        terminal_value = fcf * exit_multiple
    return terminal_value

def dcf_valuation(fcf, wacc, terminal_value, years):
    """Calculate DCF valuation"""
    discounted_fcf = sum([fcf[i] / (1 + wacc)**(i+1) for i in range(years)])
    discounted_terminal_value = terminal_value / (1 + wacc)**years
    dcf_value = discounted_fcf + discounted_terminal_value
    return dcf_value

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

def plot_sensitivity_analysis(sensitivity_df):
    """Plot sensitivity analysis results"""
    fig = px.scatter_3d(sensitivity_df, x='WACC', y='Growth Rate', z='DCF Value', title='Sensitivity Analysis')
    return fig

# Streamlit interface
st.title('Automated DCF Valuation with Alpha Vantage Integration')

st.markdown("""
This application performs an **automated DCF valuation** of a company based on uploaded financial statements or external data from **Alpha Vantage**. You can also perform **sensitivity analysis**, adjust forecast scenarios, and analyze **industry benchmarks**.
""")

# Allow user to download the template
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

st.download_button(
    label="Download Financial Data Template",
    data=download_template(),
    file_name="financial_data_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# User inputs for Alpha Vantage API Key and Stock Ticker
api_key = st.text_input('Enter your Alpha Vantage API Key:')
ticker_input = st.text_input('Enter Stock Ticker (e.g., AAPL)', value="AAPL")

# Fetch data from Alpha Vantage if ticker is provided
if api_key and ticker_input:
    data = fetch_alpha_vantage_data(ticker_input, api_key, data_type='TIME_SERIES_DAILY', interval='1d')
    
    if data:
        # Convert the time series data into a DataFrame
        df = pd.DataFrame(data).T
        df['date'] = pd.to_datetime(df.index)
        df.set_index('date', inplace=True)
        df = df[['4. close']].rename(columns={'4. close': 'Close Price'})
        st.write(f"Fetched data for {ticker_input}:")
        st.write(df)

# File uploader for financial statement (CSV or Excel)
uploaded_file = st.file_uploader("Upload your filled financial statement (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Reading the uploaded data into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file, sheet_name="Financial Data")
    
    st.write("Uploaded Data Preview:", data.head())

    # Automatically generate FCF and display financial summary
    fcf = calculate_fcf(data)
    st.write(f"Calculated Free Cash Flow: {fcf}")

    # Allow user to input other assumptions for the valuation (Cost of Equity, Cost of Debt, etc.)
    cost_of_equity = st.number_input('Enter Cost of Equity (%)', min_value=0.0, max_value=100.0, value=8.0)
    cost_of_debt = st.number_input('Enter Cost of Debt (%)', min_value=0.0, max_value=100.0, value=4.0)
    equity_value = st.number_input('Enter Total Equity Value ($)', min_value=1.0, value=500000000)
    debt_value = st.number_input('Enter Total Debt Value ($)', min_value=1.0, value=200000000)
    tax_rate = st.number_input('Enter Tax Rate (%)', min_value=0.0, max_value=100.0, value=25.0) / 100
    growth_rate = st.number_input('Enter Perpetuity Growth Rate (%)', min_value=0.0, max_value=100.0, value=2.5) / 100
    forecast_years = st.number_input('Enter Number of Forecast Years', min_value=1, max_value=10, value=5)

    # Calculate WACC
    wacc = calculate_wacc(cost_of_equity / 100, cost_of_debt / 100, equity_value, debt_value, tax_rate)

    # Calculate Terminal Value
    terminal_value = calculate_terminal_value(fcf.iloc[-1], growth_rate, wacc)

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

    # Create Dashboard Visualizations
    st.subheader("Financial Projections & Valuation")

    # Plot Free Cash Flow over the forecast years
    years = np.arange(1, forecast_years + 1)
    fcf_forecast = fcf.head(forecast_years).values  # Use first n years' FCF
    fig_fcf = plt.figure(figsize=(10, 6))
    plt.plot(years, fcf_forecast, marker='o', label="Free Cash Flow")
    plt.title('Free Cash Flow Projections')
    plt.xlabel('Year')
    plt.ylabel('FCF ($)')
    plt.grid(True)
    plt.legend()
    st.pyplot(fig_fcf)

    # Plot DCF Valuation
    fig_dcf = px.bar(x=[f'DCF Value'], y=[dcf_value], title='DCF Valuation', labels={'x': 'Valuation Type', 'y': 'Value ($)'})
    st.plotly_chart(fig_dcf)

    # Display other financial metrics
    st.subheader("DCF Assumptions")
    st.write(f"Cost of Equity: {cost_of_equity}%")
    st.write(f"Cost of Debt: {cost_of_debt}%")
    st.write(f"Tax Rate: {tax_rate * 100}%")
    st.write(f"Perpetuity Growth Rate: {growth_rate * 100}%")

    # Downloadable Results (optional)
    df_results = pd.DataFrame({
        "Metric": ["DCF Value", "Terminal Value", "WACC", "FCF (Year 1)", "FCF (Year 2)", "FCF (Year 3)", "FCF (Year 4)", "FCF (Year 5)"],
        "Value": [dcf_value, terminal_value, wacc * 100, *fcf_forecast]
    })
    
    st.download_button(
        label="Download Valuation Results",
        data=df_results.to_csv(index=False),
        file_name="dcf_valuation_results.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a financial statement CSV or Excel file to proceed.")
