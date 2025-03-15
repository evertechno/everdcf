import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
from fpdf import FPDF
from sklearn.linear_model import LinearRegression

@st.cache_data
def calculate_fcf(data):
    required_columns = ['Operating Income', 'Taxes', 'Depreciation', 'Capital Expenditures', 'Changes in Working Capital']
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")
    fcf = data['Operating Income'] - data['Taxes'] + data['Depreciation'] - data['Capital Expenditures'] - data['Changes in Working Capital']
    return fcf

@st.cache_data
def calculate_wacc(cost_of_equity, cost_of_debt, equity_value, debt_value, tax_rate):
    total_value = equity_value + debt_value
    wacc = (equity_value / total_value) * cost_of_equity + (debt_value / total_value) * cost_of_debt * (1 - tax_rate)
    return wacc

@st.cache_data
def calculate_terminal_value(fcf, growth_rate, discount_rate, method='perpetuity'):
    if method == 'perpetuity':
        terminal_value = fcf * (1 + growth_rate) / (discount_rate - growth_rate)
    elif method == 'exit_multiple':
        exit_multiple = 10
        terminal_value = fcf * exit_multiple
    return terminal_value

@st.cache_data
def dcf_valuation(fcf, wacc, terminal_value, years):
    if years > len(fcf):
        raise ValueError("Forecast years exceed the available data length.")
    discounted_fcf = sum([fcf[i] / (1 + wacc)**(i+1) for i in range(years)])
    discounted_terminal_value = terminal_value / (1 + wacc)**years
    dcf_value = discounted_fcf + discounted_terminal_value
    return dcf_value

@st.cache_data
def sensitivity_analysis(fcf, wacc_range, growth_range, years):
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
    fig = px.scatter_3d(sensitivity_df, x='WACC', y='Growth Rate', z='DCF Value',
                        title='Sensitivity Analysis',
                        labels={'WACC': 'WACC (%)', 'Growth Rate': 'Growth Rate (%)', 'DCF Value': 'DCF Value ($)'})
    return fig

@st.cache_data
def generate_pdf_report(dcf_value, terminal_value, wacc, fcf_forecast, market_cap, net_income, pe_ratio, debt_equity_ratio, assumptions):
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
def fetch_historical_data():
    return pd.DataFrame()

@st.cache_data
def monte_carlo_simulation(initial_value, num_simulations, num_days, vol):
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
    ratios = {
        'ROE': data['Net Income'] / data['Total Equity'],
        'ROA': data['Net Income'] / data['Total Assets'],
        'EBITDA Margin': data['EBITDA'] / data['Revenue'],
        'Net Profit Margin': data['Net Income'] / data['Revenue']
    }
    return ratios

@st.cache_data
def calculate_profit_margin(data):
    return data['Net Income'] / data['Revenue']

@st.cache_data
def calculate_ebitda(data):
    return data['Operating Income'] + data['Depreciation']

@st.cache_data
def calculate_revenue_growth(data):
    return data['Revenue'].pct_change()

@st.cache_data
def calculate_debt_to_equity_ratio(data):
    return data['Total Debt'] / data['Total Equity']

@st.cache_data
def calculate_pe_ratio(market_cap, net_income):
    return market_cap / net_income

def download_template():
    template_data = {
        'Year': ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'],
        'Revenue': [None, None, None, None, None],
        'Operating Income': [None, None, None, None, None],
        'Taxes': [None, None, None, None, None],
        'Depreciation': [None, None, None, None, None],
        'Capital Expenditures': [None, None, None, None, None],
        'Changes in Working Capital': [None, None, None, None, None],
        'Net Income': [None, None, None, None, None],
        'Total Debt': [None, None, None, None, None],
        'Total Equity': [None, None, None, None, None]
    }

    df_template = pd.DataFrame(template_data)
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_template.to_excel(writer, index=False, sheet_name='Financial Data')
    excel_buffer.seek(0)
    return excel_buffer

st.set_page_config(page_title="Automated DCF Valuation", layout="wide")
st.title('Automated DCF Valuation')

st.markdown("""
This application performs an **automated DCF valuation** of a company based on uploaded financial statements. You can also perform **sensitivity analysis**, adjust forecast scenarios, and analyze **industry benchmarks**.
""")

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

st.download_button(
    label="Download Financial Data Template",
    data=download_template(),
    file_name="financial_data_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

uploaded_file = st.file_uploader("Upload your filled financial statement (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file, sheet_name="Financial Data")

    st.write("Uploaded Data Preview:", data.head())

    required_columns = ['Revenue', 'Operating Income', 'Taxes', 'Depreciation', 'Capital Expenditures', 'Changes in Working Capital', 'Net Income', 'Total Debt', 'Total Equity']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Error: Missing required columns: {', '.join(missing_columns)}")
    else:
        try:
            fcf = calculate_fcf(data)
            st.write(f"Calculated Free Cash Flow: {fcf}")

            cost_of_equity = st.number_input('Enter Cost of Equity (%)', min_value=0.0, max_value=100.0, value=8.0, step=0.1, format="%.2f")
            cost_of_debt = st.number_input('Enter Cost of Debt (%)', min_value=0.0, max_value=100.0, value=4.0, step=0.1, format="%.2f")
            equity_value = st.number_input('Enter Total Equity Value ($)', min_value=1.0, value=500000000.0, format="%.2f")
            debt_value = st.number_input('Enter Total Debt Value ($)', min_value=1.0, value=200000000.0, format="%.2f")
            tax_rate = st.number_input('Enter Tax Rate (%)', min_value=0.0, max_value=100.0, value=25.0, step=0.1, format="%.2f") / 100
            growth_rate = st.number_input('Enter Perpetuity Growth Rate (%)', min_value=0.0, max_value=100.0, value=2.5, step=0.1, format="%.2f") / 100
            forecast_years = int(st.number_input('Enter Number of Forecast Years', min_value=1, max_value=10, value=5, format="%.0f"))
            terminal_value_method = st.selectbox('Select Terminal Value Method', ['perpetuity', 'exit_multiple'])

            if forecast_years > len(fcf):
                st.error("Error: Forecast years exceed the available data length. Please enter a valid number of forecast years.")
            else:
                wacc = calculate_wacc(cost_of_equity / 100, cost_of_debt / 100, equity_value, debt_value, tax_rate)
                terminal_value = calculate_terminal_value(fcf.iloc[-1], growth_rate, wacc, method=terminal_value_method)
                dcf_value = dcf_valuation(fcf, wacc, terminal_value, forecast_years)

                st.write(f"Calculated WACC: {wacc * 100:.2f}%")
                st.write(f"Calculated Terminal Value: ${terminal_value:,.2f}")
                st.write(f"DCF Valuation: ${dcf_value:,.2f}")

                wacc_range = np.linspace(0.05, 0.15, 11)
                growth_range = np.linspace(0.01, 0.10, 10)
                sensitivity_df = sensitivity_analysis(fcf, wacc_range, growth_range, forecast_years)
                sensitivity_fig = plot_sensitivity_analysis(sensitivity_df)
                st.plotly_chart(sensitivity_fig)

                st.subheader("Financial Projections & Valuation")
                years = np.arange(1, forecast_years + 1)
                fcf_forecast = fcf.head(forecast_years).values
                fig_fcf = go.Figure()
                fig_fcf.add_trace(go.Bar(x=years, y=fcf_forecast, name="Free Cash Flow", marker_color='skyblue'))
                fig_fcf.add_trace(go.Scatter(x=years, y=fcf_forecast, mode='lines+markers', name="Free Cash Flow (Line)", marker=dict(color='red')))
                fig_fcf.update_layout(title='Free Cash Flow Projections', xaxis_title='Year', yaxis_title='FCF ($)', template='plotly_white')
                st.plotly_chart(fig_fcf)

                fig_dcf = px.bar(x=[f'DCF Value'], y=[dcf_value], title='DCF Valuation', labels={'x': 'Valuation Type', 'y': 'Value ($)'})
                st.plotly_chart(fig_dcf)

                st.subheader("Key Financial Metrics")
                if 'Market Capitalization' in data.columns:
                    market_cap = data['Market Capitalization'].sum()
                else:
                    market_cap = st.number_input('Enter Market Capitalization ($)', min_value=1.0, value=1000000000.0, format="%.2f")

                if 'Net Income' in data.columns:
                    net_income = data['Net Income'].sum()
                else:
                    net_income = st.number_input('Enter Net Income ($)', min_value=1.0, value=100000000.0, format="%.2f")

                pe_ratio = market_cap / net_income
                debt_equity_ratio = data['Total Debt'].sum() / data['Total Equity'].sum() if 'Total Debt' in data.columns and 'Total Equity' in data.columns else 0

                st.write(f"P/E Ratio: {pe_ratio:.2f}")
                st.write(f"Debt/Equity Ratio: {debt_equity_ratio:.2f}")

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

                profit_margin = calculate_profit_margin(data).mean()
                ebitda = calculate_ebitda(data)
                revenue_growth = calculate_revenue_growth(data).mean()
                debt_to_equity_ratio = calculate_debt_to_equity_ratio(data).mean()

                st.subheader("Additional Financial Metrics")
                st.write(f"Average Profit Margin: {profit_margin:.2f}")
                st.write(f"Total EBITDA: {ebitda.sum():,.2f}")
                st.write(f"Average Revenue Growth Rate: {revenue_growth:.2f}")
                st.write(f"Average Debt to Equity Ratio: {debt_to_equity_ratio:.2f}")

                fig_revenue = px.line(data, x=data.index, y='Revenue', title='Historical Revenue')
                st.plotly_chart(fig_revenue)

                fig_ebitda = px.line(data, x=data.index, y=ebitda, title='Historical EBITDA')
                st.plotly_chart(fig_ebitda)

                fig_profit_margin = px.line(data, x=data.index, y=calculate_profit_margin(data), title='Historical Profit Margin')
                st.plotly_chart(fig_profit_margin)

                fig_debt_equity = px.line(data, x=data.index, y=calculate_debt_to_equity_ratio(data), title='Debt to Equity Ratio')
                st.plotly_chart(fig_debt_equity)

                num_simulations = st.number_input("Enter Number of Simulations", min_value=1, value=100)
                num_days = st.number_input("Enter Number of Days", min_value=1, value=252)
                vol = st.number_input("Enter Volatility (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1) / 100

                simulations_revenue = monte_carlo_simulation(data['Revenue'].iloc[-1], num_simulations, num_days, vol)
                fig_mc_revenue = go.Figure()
                for simulation in simulations_revenue:
                    fig_mc_revenue.add_trace(go.Scatter(x=list(range(num_days + 1)), y=simulation, mode='lines', line=dict(width=1), opacity=0.5))
                fig_mc_revenue.update_layout(title='Monte Carlo Simulation - Revenue', xaxis_title='Day', yaxis_title='Revenue ($)', template='plotly_white')
                st.plotly_chart(fig_mc_revenue)

                simulations_ebitda = monte_carlo_simulation(ebitda.iloc[-1], num_simulations, num_days, vol)
                fig_mc_ebitda = go.Figure()
                for simulation in simulations_ebitda:
                    fig_mc_ebitda.add_trace(go.Scatter(x=list(range(num_days + 1)), y=simulation, mode='lines', line=dict(width=1), opacity=0.5))
                fig_mc_ebitda.update_layout(title='Monte Carlo Simulation - EBITDA', xaxis_title='Day', yaxis_title='EBITDA ($)', template='plotly_white')
                st.plotly_chart(fig_mc_ebitda)

                simulations_profit_margin = monte_carlo_simulation(profit_margin, num_simulations, num_days, vol)
                fig_mc_profit_margin = go.Figure()
                for simulation in simulations_profit_margin:
                    fig_mc_profit_margin.add_trace(go.Scatter(x=list(range(num_days + 1)), y=simulation, mode='lines', line=dict(width=1), opacity=0.5))
                fig_mc_profit_margin.update_layout(title='Monte Carlo Simulation - Profit Margin', xaxis_title='Day', yaxis_title='Profit Margin (%)', template='plotly_white')
                st.plotly_chart(fig_mc_profit_margin)

                df_results = pd.DataFrame({
                    "Metric": ["DCF Value", "Terminal Value", "WACC", "FCF (Year 1)", "FCF (Year 2)", "FCF (Year 3)", "FCF (Year 4)", "FCF (Year 5)"],
                    "Value": [dcf_value, terminal_value, wacc * 100, *fcf_forecast]
                })

                st.download_button(
                    label="Download Valuation Results (CSV)",
                    data=df_results.to_csv(index=False),
                    file_name="dcf_valuation_results.csv",
                    mime="text/csv"
                )

                pdf_report = generate_pdf_report(dcf_value, terminal_value, wacc, fcf_forecast, market_cap, net_income, pe_ratio, debt_equity_ratio, assumptions)
                st.download_button(
                    label="Download Valuation Report (PDF)",
                    data=pdf_report,
                    file_name="dcf_valuation_report.pdf",
                    mime="application/pdf"
                )

        except KeyError as e:
            st.error(f"Error: {e}")
        except ValueError as e:
            st.error(f"Error: {e}")
