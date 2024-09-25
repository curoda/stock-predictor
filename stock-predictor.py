import streamlit as st
import openai
import pandas as pd
import yfinance as yf

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Define functions

def get_stock_data(stock_symbol):
    """Fetches stock financial data and company information."""
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period='5y')
    stock_info = stock.info
    return data, stock_info

def generate_factors(stock_info):
    """Generates key stock factors dynamically using GPT-4 based on stock's sector and industry."""
    messages = [
        {"role": "system", "content": "You are a financial analyst."},
        {"role": "user", "content": f"Given the sector {stock_info['sector']} and industry {stock_info['industry']}, what are the key factors affecting a company's stock performance over the next 5 years?"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    factors = response['choices'][0]['message']['content'].strip().split('\n')
    return factors

def analyze_factor(factor, stock_info):
    """Uses GPT-4 to analyze each factor and assigns rating, confidence, and importance."""
    messages = [
        {"role": "system", "content": "You are a financial analyst."},
        {"role": "user", "content": f"Analyze the factor '{factor}' for stock performance in the {stock_info['sector']} sector. What is the rating, confidence, and importance for predicting the stock’s 5-year performance?"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )
    analysis = response['choices'][0]['message']['content'].strip()
    return analysis

def run_analysis(stock_symbol, time_horizon):
    """Main analysis workflow to generate factors, analyze them, and display results."""
    stock_data, stock_info = get_stock_data(stock_symbol)
    st.write(f"## {stock_info['longName']} ({stock_symbol.upper()}) - {stock_info['sector']} Sector")
    factors = generate_factors(stock_info)

    # Analyze each factor
    factor_results = []
    for factor in factors:
        result = analyze_factor(factor, stock_info)
        factor_results.append({"Factor": factor, "Analysis": result})

    # Display Factor Table
    st.write("### Key Factor Analysis Table")
    df = pd.DataFrame(factor_results)
    st.table(df)

def market_comparison(stock_symbol, time_horizon):
    """Compares stock's performance to S&P 500 and Nasdaq."""
    # Predicted growth rates for the stock
    stock_growth = predict_stock_growth(stock_symbol)

    # S&P 500 and Nasdaq historical growth rates
    sp500_growth = 0.07  # Example: 7% annually
    nasdaq_growth = 0.10  # Example: 10% annually

    # Calculate outperformance
    stock_vs_sp500 = (stock_growth - sp500_growth) * time_horizon
    stock_vs_nasdaq = (stock_growth - nasdaq_growth) * time_horizon

    # Display Comparison
    st.write(f"{stock_symbol} is expected to outperform the S&P 500 by {stock_vs_sp500:.2%} and the Nasdaq by {stock_vs_nasdaq:.2%} over {time_horizon} years.")

def display_comparison_table(stock_vs_sp500, stock_vs_nasdaq, stock_growth):
    """Displays performance comparison between stock, S&P 500, and Nasdaq."""
    comparison_data = {
        "Metric": ["Annual Growth Rate", "5-Year Outperformance"],
        "Stock": [f"{stock_growth:.2%}", f"+{stock_vs_sp500:.2%} vs S&P 500, +{stock_vs_nasdaq:.2%} vs Nasdaq"],
        "S&P 500": [f"{0.07:.2%}", "N/A"],
        "Nasdaq": [f"{0.10:.2%}", "N/A"]
    }
    df_comparison = pd.DataFrame(comparison_data)
    st.write("### Stock Performance vs Market Indices")
    st.table(df_comparison)

# Streamlit UI

st.title('Stock Performance Prediction')

# Input fields
stock_symbol = st.text_input('Enter Stock Symbol (e.g., NVDA):', 'NVDA')
time_horizon = st.slider('Select Time Horizon (years):', min_value=1, max_value=10, value=5)

# Analyze button
if st.button('Analyze'):
    run_analysis(stock_symbol, time_horizon)
    market_comparison(stock_symbol, time_horizon)
