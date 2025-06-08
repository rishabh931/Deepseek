# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Configure page
st.set_page_config(page_title="Stock P&L Analyzer", layout="wide")
st.title("📈 Indian Stock P&L Statement Analysis")

# Indian stock symbols mapping
INDIAN_STOCKS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS"
}

# Function to download financial data
def get_financials(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    
    # Get financial statements
    income_stmt = stock.financials
    balance_sheet = stock.balance_sheet
    
    # Extract relevant data
    financial_data = {}
    
    try:
        # Revenue
        financial_data['Revenue'] = income_stmt.loc['Total Revenue'].head(4).values[::-1]
        
        # Operating Profit
        financial_data['Operating Profit'] = income_stmt.loc['Operating Income'].head(4).values[::-1]
        
        # PBT
        financial_data['PBT'] = income_stmt.loc['Pretax Income'].head(4).values[::-1]
        
        # PAT
        financial_data['PAT'] = income_stmt.loc['Net Income'].head(4).values[::-1]
        
        # Shares Outstanding
        shares_outstanding = balance_sheet.loc['Ordinary Shares Number'].head(4).values[::-1]
        
        # Calculate EPS
        financial_data['EPS'] = financial_data['PAT'] / (shares_outstanding / 1e6)  # Convert to millions
        
        # Calculate metrics
        financial_data['OPM %'] = (financial_data['Operating Profit'] / financial_data['Revenue']) * 100
        financial_data['EPS Growth %'] = [0] + [((financial_data['EPS'][i] - financial_data['EPS'][i-1]) / 
                                               financial_data['EPS'][i-1] * 100) 
                                              for i in range(1, len(financial_data['EPS']))]
        
        # Get years
        years = income_stmt.columns[:4].strftime('%Y').values[::-1]
        
        return pd.DataFrame(financial_data, index=years), years
        
    except KeyError as e:
        st.error(f"Missing financial data field: {str(e)}")
        return None, None

# Function to generate analysis insights
def generate_analysis(df, years):
    insights = []
    
    # Revenue analysis
    rev_growth = df['Revenue'].pct_change().dropna() * 100
    insights.append("**Revenue Analysis:**")
    insights.append(f"- Latest revenue: ₹{df['Revenue'].iloc[-1]/1000:,.1f} Cr")
    insights.append(f"- 4-year CAGR: {((df['Revenue'].iloc[-1]/df['Revenue'].iloc[0])**(1/3)-1)*100:.1f}%")
    insights.append(f"- Trend: {'Growth' if rev_growth.mean() > 0 else 'Decline'} averaging {rev_growth.mean():.1f}% YoY")
    
    # OPM analysis
    opm_trend = "improving" if df['OPM %'].iloc[-1] > df['OPM %'].iloc[0] else "declining"
    insights.append("\n**Operating Profit Analysis:**")
    insights.append(f"- Current OPM: {df['OPM %'].iloc[-1]:.1f}%")
    insights.append(f"- Trend: {opm_trend} ({df['OPM %'].iloc[0]:.1f}% → {df['OPM %'].iloc[-1]:.1f}%)")
    insights.append(f"- Operating leverage: {((df['Operating Profit'].pct_change().mean() - df['Revenue'].pct_change().mean()))*100:.1f}%")
    
    # PAT analysis
    pat_growth = df['PAT'].pct_change().dropna() * 100
    insights.append("\n**Profit After Tax Analysis:**")
    insights.append(f"- Latest PAT: ₹{df['PAT'].iloc[-1]/1000:,.1f} Cr")
    insights.append(f"- 4-year CAGR: {((df['PAT'].iloc[-1]/df['PAT'].iloc[0])**(1/3)-1)*100:.1f}%")
    insights.append(f"- Margin trend: {'Expanding' if (df['PAT']/df['Revenue']).iloc[-1] > (df['PAT']/df['Revenue']).iloc[0] else 'Contracting'}")
    
    # EPS analysis
    eps_growth = df['EPS Growth %'].dropna()
    insights.append("\n**EPS Analysis:**")
    insights.append(f"- Latest EPS: ₹{df['EPS'].iloc[-1]:.1f}")
    insights.append(f"- 3-year average growth: {eps_growth.mean():.1f}%")
    insights.append(f"- Growth consistency: {'Stable' if eps_growth.std() < 15 else 'Volatile'}")
    
    return "\n".join(insights)

# UI Components
selected_stock = st.selectbox(
    "Select Indian Stock:",
    list(INDIAN_STOCKS.keys()),
    index=0
)

ticker_symbol = INDIAN_STOCKS[selected_stock]

if st.button("Analyze P&L"):
    with st.spinner(f"Fetching {selected_stock} financial data..."):
        financial_df, years = get_financials(ticker_symbol)
        
    if financial_df is not None:
        st.success(f"Data retrieved for {selected_stock} ({ticker_symbol})")
        
        # Section 1: Revenue, Operating Profit, OPM%
        st.header("1. Revenue, Operating Profit & OPM% Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Financial Metrics")
            st.dataframe(financial_df[['Revenue', 'Operating Profit', 'OPM %']].style
                         .format({
                             'Revenue': '{:,.0f}',
                             'Operating Profit': '{:,.0f}',
                             'OPM %': '{:.1f}%'
                         }))
            
            # Download button
            csv = financial_df[['Revenue', 'Operating Profit', 'OPM %']].to_csv()
            st.download_button(
                label="Download Revenue Data",
                data=csv,
                file_name=f"{selected_stock}_revenue_data.csv",
                mime="text/csv"
            )
        
        with col2:
            st.subheader("Performance Trend")
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Bar plot for Revenue and Operating Profit
            width = 0.35
            x = np.arange(len(years))
            ax1.bar(x - width/2, financial_df['Revenue']/1e7, width, label='Revenue (₹ Cr)', color='skyblue')
            ax1.bar(x + width/2, financial_df['Operating Profit']/1e7, width, label='Op Profit (₹ Cr)', color='lightgreen')
            ax1.set_ylabel('Amount (₹ Crores)')
            
            # Line plot for OPM%
            ax2 = ax1.twinx()
            ax2.plot(x, financial_df['OPM %'], 'r-o', linewidth=2, label='OPM %')
            ax2.set_ylabel('OPM %', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, max(financial_df['OPM %']) * 1.5)
            
            plt.title(f'{selected_stock} Revenue & Profitability')
            plt.xticks(x, years)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            st.pyplot(fig)
        
        # Analysis insights
        st.subheader("Key Observations")
        st.markdown(generate_analysis(financial_df, years))
        
        # Section 2: PBT, PAT, EPS
        st.header("2. PBT, PAT & EPS Analysis")
        
        col3, col4 = st.columns([1, 2])
        
        with col3:
            st.subheader("Profitability Metrics")
            st.dataframe(financial_df[['PBT', 'PAT', 'EPS', 'EPS Growth %']].style
                         .format({
                             'PBT': '{:,.0f}',
                             'PAT': '{:,.0f}',
                             'EPS': '{:.1f}',
                             'EPS Growth %': '{:.1f}%'
                         }))
            
            # Download button
            csv2 = financial_df[['PBT', 'PAT', 'EPS', 'EPS Growth %']].to_csv()
            st.download_button(
                label="Download Profitability Data",
                data=csv2,
                file_name=f"{selected_stock}_profitability_data.csv",
                mime="text/csv"
            )
        
        with col4:
            st.subheader("Profitability Trend")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # PAT and PBT trend
            ax1.bar(years, financial_df['PAT']/1e7, color='lightblue', label='PAT (₹ Cr)')
            ax1.bar(years, (financial_df['PBT'] - financial_df['PAT'])/1e7, 
                    bottom=financial_df['PAT']/1e7, color='lightcoral', label='Tax (₹ Cr)')
            ax1.set_ylabel('Amount (₹ Crores)')
            ax1.set_title('Profit Before Tax Composition')
            ax1.legend()
            
            # EPS Growth
            ax2.plot(years, financial_df['EPS'], 'g-o', label='EPS (₹)')
            ax2.set_ylabel('EPS', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            ax3 = ax2.twinx()
            ax3.bar(years, financial_df['EPS Growth %'], alpha=0.3, color='purple', label='EPS Growth %')
            ax3.set_ylabel('Growth %', color='purple')
            ax3.tick_params(axis='y', labelcolor='purple')
            ax3.axhline(0, color='grey', linestyle='--')
            ax3.set_title('EPS Performance')
            ax3.legend(loc='lower right')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # GitHub resources
        st.header("📦 GitHub Resources")
        st.info("All code and data files are available in the GitHub repository:")
        st.markdown("[![GitHub](https://img.shields.io/badge/Repository-100000?logo=github)](https://github.com/yourusername/stock-pl-analysis)")
        
        resources = {
            "app.py": "Main Streamlit application code",
            "requirements.txt": "Python dependencies",
            "stock_data.csv": "Sample dataset",
            "analysis_template.ipynb": "Jupyter Notebook for analysis"
        }
        
        for file, description in resources.items():
            with st.expander(f"Download {file}"):
                st.write(description)
                # Create dummy files for download
                content = f"This is a sample {file} file for {selected_stock} analysis"
                st.download_button(
                    label=f"Download {file}",
                    data=content,
                    file_name=file,
                    mime="text/plain"
                )
    else:
        st.error("Could not retrieve financial data. Please try another stock.")

# Add footer
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
- **Data Source**: Yahoo Finance
- **Financial Metrics**: Revenue, Operating Profit, OPM%, PBT, PAT, EPS
- **Analysis**: 4-year trend with visualizations and key insights
- **Updates**: Daily market data
""")
