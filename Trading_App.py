import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Trading App",
    page_icon="📈",
    layout="wide"
)

# Title and Header
st.title("Trading App 📈")
st.header("Empowering Your Investment Journey")

# Introduction with quotes
st.subheader("“An investment in knowledge pays the best interest.” – Benjamin Franklin")
st.write("Welcome to the **Trading App**, your trusted companion in navigating the financial markets. Whether you're a seasoned investor or just starting, our platform offers all the tools and insights you need to make informed decisions.")

# Stock Image
st.image('stock_image.jpg',use_container_width=True)

# Overview of App Features
st.markdown("## What We Offer")
st.write("""
Our app is designed to cater to every step of your investment process:
- **Stock Information Page**: Learn about key financial metrics, historical data, and current stock trends.
- **Stock Analysis Page**: Dive deep into stock fundamentals and explore technical analysis tools.
- **Stock Prediction Page**: Use advanced machine learning models to predict future stock performance.
""")

# Add a motivational quote
st.markdown("### 🌟 Quote of the Day")
st.info("“The stock market is filled with individuals who know the price of everything but the value of nothing.” – Philip Fisher")


st.markdown("### 📊 Did You Know?")
st.write("""
The New York Stock Exchange (NYSE) is the largest stock exchange in the world by market capitalization, 
with a total value exceeding **$25 trillion**! Understanding the markets begins with knowing the incredible scale of global finance.
""")

# App Navigation Details
st.markdown("### How to Get Started")
st.write("""
1. Navigate to the **Stock Information** page to explore comprehensive stock details.
2. Move to the **Stock Analysis** page for in-depth analysis of your favorite stocks.
3. Visit the **Stock Prediction** page to forecast stock prices using cutting-edge technology.
""")

# Footer
st.markdown("---")
st.markdown("### Ready to take charge of your financial future? 🚀")
st.markdown("“The best time to invest was yesterday. The second-best time is today!”")
