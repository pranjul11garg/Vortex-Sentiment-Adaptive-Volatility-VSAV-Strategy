#!/usr/bin/env python
# coding: utf-8

# ---
# title: "Vortex–Sentiment Adaptive Volatility (VSAV) Strategy"
# author:
#   - name: Group Data Oracles
#     affiliations:
#       - name: Boston University
#         city: Boston
#         state: MA
# format:
#   html:
#     toc: true
#     css: styles.css
#     html-math-method: katex
#     embed-resources: false
#     code-fold: true
# jupyter: python3
# execute:
#   eval: true
# ---
# 

# ## Importing Necessary Libraries for Analysis

# In[176]:


import yfinance as yf  # For downloading financial data
import numpy as np      # For numerical operations
import pandas as pd     # For data manipulation
import requests # For downloading the API data
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px # Import the Plotly Express module for interactive visualization
import json
import vectorbt as vbt
from plotly.subplots import make_subplots
import streamlit as st


# ## Data Collection

# ### Fetch daily OHLCV data 

# In[177]:


# Data for the TSLA, XLY, and SPY tickers is retrieved from the Yahoo Finance library, covering the period from January 1, 2019, 
# to March 5, 2025.
tsla = yf.download('TSLA', start='2019-01-01', end='2025-03-05') 
xly = yf.download('XLY', start='2019-01-01', end='2025-03-05')
spy = yf.download('SPY', start='2019-01-01', end='2025-03-05')


# In[178]:


# Displays a summary of the TSLA DataFrame, including column names, data types, non-null counts, and memory usage.
tsla.info()


# In[179]:


# Displays a summary of the XLY DataFrame, including column names, data types, non-null counts, and memory usage.
xly.info()


# In[180]:


# Displays a summary of the SPY DataFrame, including column names, data types, non-null counts, and memory usage.
spy.info()


# ### Fetch sentiment scores from the API

# In[181]:


# Defines the API endpoint URL for retrieving news sentiment data related to Tesla (TSLA) 
# from the Alpha Vantage service. The query specifies the function type, date range, result limit, 
# targeted ticker symbol, and a valid API key.
###url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from=20250101T0130&time_to=20250301T0130&limit=1000&tickers=TSLA&apikey=PNM5EHRALIOT1CKJ'

# Sends a GET request to the specified URL to initiate the API call.
###response = requests.get(url)

# Evaluates whether the API call was successful based on the HTTP response status code.
###if response.status_code == 200:
    # Parses the JSON response and extracts the 'feed' section containing sentiment data.
    ###sentiment_data = response.json()
    
    # Converts the extracted sentiment feed into a DataFrame for further analysis or visualization.
   ### sentiment_df = pd.DataFrame(sentiment_data['feed']) 
    
    # Displays the first five rows of the sentiment DataFrame to provide an overview of the retrieved content.
    ###print(sentiment_df.head())
###else:
    # Prints an error message if the API request was unsuccessful.
    ###print("API call failed:", response.status_code)

# Independently parses the full JSON response and prints its contents for inspection or debugging purposes.
###sentiment_json = response.json()
###print(sentiment_json)


# ## Indicator Calculation

# ### Compute VI+ and VI-

# In[182]:


# Defines a function to calculate the Vortex Indicator (VI) for a given DataFrame and ticker symbol.
# The calculation uses a default lookback period of 14 days unless specified otherwise.
def calculate_vortex(df, value, n=14):
    # Extracts the high, low, and close price series for the specified ticker.
    high = df[("High", value)]
    low = df[("Low", value)]
    close = df[("Close", value)]

    # Calculates the Vortex Movement values:
    # VM+ = absolute difference between today's high and yesterday's low
    # VM− = absolute difference between today's low and yesterday's high
    vm_plus = abs(high - low.shift(1))     # |Today's High – Yesterday’s Low|
    vm_minus = abs(low - high.shift(1))    # |Today's Low – Yesterday’s High|

    # Computes the True Range (TR) as the maximum of:
    # - High - Low
    # - Absolute difference between High and Previous Close
    # - Absolute difference between Low and Previous Close
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    # Applies a rolling window to compute the n-period sum of VM+ and VM− values
    # and the corresponding True Range values.
    sum_vm_plus = vm_plus.rolling(window=n).sum()
    sum_vm_minus = vm_minus.rolling(window=n).sum()
    sum_tr = tr.rolling(window=n).sum()

    # Calculates the Vortex Indicator components:
    # VI+ = sum of VM+ over n periods divided by sum of TR over n periods
    # VI− = sum of VM− over n periods divided by sum of TR over n periods
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr

    # Returns the VI+ and VI− series as output.
    return vi_plus, vi_minus


# In[183]:


# Calculates the Vortex Indicator values for TSLA and stores the results as new columns in the DataFrame.
tsla['VI+'], tsla['VI-'] = calculate_vortex(tsla, 'TSLA')

# Calculates the Vortex Indicator values for XLY and stores the results as new columns in the DataFrame.
xly['VI+'], xly['VI-'] = calculate_vortex(xly, 'XLY')

# Calculates the Vortex Indicator values for SPY and stores the results as new columns in the DataFrame.
spy['VI+'], spy['VI-'] = calculate_vortex(spy, 'SPY')


# In[184]:


# Displays the first 20 rows of the TSLA DataFrame to provide an initial overview of its structure and content with the new function applied.
tsla.head(20)


# ### Calculate Volume-Weighted Sentiment 

# In[185]:


# Load the sentiment JSON file from local storage
with open("TSLA_sentiment.json", "r") as f:
    sentiment_json = json.load(f)

# Extract the "feed" list from the top-level JSON dictionary.
# This section contains the array of sentiment articles or entries.
sentiment_feed = sentiment_json.get("feed", [])

# Initialize an empty list to hold cleaned and structured sentiment data
sentiment_data = []

# Iterate through each item in the sentiment feed to extract relevant fields
for item in sentiment_feed:
    try:
        sentiment_data.append({
            # Convert the timestamp to pandas datetime for proper indexing
            "time_published": pd.to_datetime(item["time_published"]),
            # Convert the sentiment score string to float
            "sentiment_score": float(item["overall_sentiment_score"]),
            # Store the sentiment label (e.g., Positive, Neutral, Negative)
            "sentiment_label": item["overall_sentiment_label"],
        })
    except (KeyError, ValueError, TypeError):
        # Skip malformed or incomplete entries that raise an error
        continue

# Convert the structured list of dictionaries into a pandas DataFrame
sentiment_df = pd.DataFrame(sentiment_data)

# Set the 'time_published' column as the DataFrame index to enable time-series operations
sentiment_df.set_index("time_published", inplace=True)

# Display the first few rows of the DataFrame to verify content and structure
print(sentiment_df.head())

# Output a summary of the DataFrame structure, including column types and memory usage
print(sentiment_df.info())


# In[186]:


# Initialize an empty list to store processed sentiment records
sentiment_data = []

# Iterate through each news item in the 'feed' section of the JSON object
for news_item in sentiment_json.get("feed", []):
    # Append a dictionary with selected and transformed fields to the sentiment list
    sentiment_data.append({
        # Convert the time of publication to datetime format
        "time_published": pd.to_datetime(news_item["time_published"]),
        # Extract the sentiment score (as-is; conversion to float may be handled separately if needed)
        "sentiment_score": news_item["overall_sentiment_score"],
        # Extract the sentiment label (e.g., Positive, Neutral, Negative)
        "sentiment_label": news_item["overall_sentiment_label"],
    })

# Convert the list of dictionaries into a pandas DataFrame
sentiment_data = pd.DataFrame(sentiment_data)


# In[187]:


# Sort the DataFrame by publication time in ascending order for chronological analysis
sentiment_data['time_published'].sort_values(ascending=True)


# In[188]:


# Convert the 'time_published' column to only retain the date portion (drop time-of-day)
sentiment_data['time_published'] = sentiment_data['time_published'].dt.date


# In[189]:


# Filter sentiment data to retain only those records that match dates present in the TSLA index
sentiment_scores_filtered = sentiment_data[
    pd.to_datetime(sentiment_data['time_published']).isin(tsla.index)
]

# Group the filtered data by publication date and calculate the average sentiment score per day
sentiment_scores_filtered = sentiment_scores_filtered.groupby('time_published')['sentiment_score'].mean().reset_index()


# In[190]:


# Fix the multi-level column issue by selecting the 'Volume' column and resetting its name
tsla_volume = tsla[('Volume', 'TSLA')].rename('Volume')

# Ensure the index of tsla_volume is a column and convert it to match the type of time_published
tsla_volume = tsla_volume.reset_index()
tsla_volume['Date'] = pd.to_datetime(tsla_volume['Date'])


# In[191]:


# Convert 'time_published' in the sentiment data to datetime to match volume data type
sentiment_scores_filtered['time_published'] = pd.to_datetime(sentiment_scores_filtered['time_published'])

# Perform an inner merge between sentiment scores and volume data based on matching dates
merged_data = pd.merge(
    tsla_volume,
    sentiment_scores_filtered,
    left_on='Date',
    right_on='time_published',
    how='inner'
)


# In[192]:


# Compute the weighted sentiment by multiplying raw sentiment by trading volume
merged_data['Weighted_Sentiment'] = merged_data['Volume'] * merged_data['sentiment_score']

# Calculate a 5-day rolling average of the weighted sentiment to smooth short-term noise
merged_data['5_day_avg_sentiment'] = merged_data['Weighted_Sentiment'].rolling(window=5).mean()

# Define a binary condition for when the average sentiment is positive
merged_data['Buy_Condition'] = merged_data['5_day_avg_sentiment'] > 0

# Normalize the rolling sentiment score by average volume to allow comparability across scales
merged_data['5_day_avg_sentiment_norm'] = (
    merged_data['5_day_avg_sentiment'] / merged_data['Volume'].mean()
)


# In[193]:


merged_data


# ### Derive ATR (10) for Volatility Adjustments

# In[194]:


# Flatten MultiIndex columns if present to simplify DataFrame operations
tsla.columns = [
    '_'.join(col).strip() if isinstance(col, tuple) else col
    for col in tsla.columns
]

# Calculate the previous closing price to support True Range computation
tsla["prev_close"] = tsla["Close_TSLA"].shift(1)

# Compute three True Range variations used in ATR calculation
tsla["tr1"] = tsla["High_TSLA"] - tsla["Low_TSLA"]
tsla["tr2"] = abs(tsla["High_TSLA"] - tsla["prev_close"])
tsla["tr3"] = abs(tsla["Low_TSLA"] - tsla["prev_close"])

# Derive the True Range (TR) as the maximum of the three variants
tsla["true_range"] = tsla[["tr1", "tr2", "tr3"]].max(axis=1)

# Compute the 10-day Average True Range (ATR) to measure market volatility
tsla["ATR_10"] = tsla["true_range"].rolling(window=10).mean()

# Calculate ATR as a percentage of the current closing price to normalize volatility
tsla["atr_pct"] = tsla["ATR_10"] / tsla["Close_TSLA"]

# Define a function to assign position size based on volatility levels
def position_size(row):
    if row["atr_pct"] < 0.03:
        return 0.01  # Allocate 1% of capital for low-volatility conditions
    else:
        return 0.005  # Allocate 0.5% of capital for high-volatility conditions

# Apply the position size function across all rows
tsla["position_size"] = tsla.apply(position_size, axis=1)

# Display the latest 10 rows with selected indicators for inspection
print(tsla[["Close_TSLA", "ATR_10", "atr_pct", "position_size"]].tail(10))


# In[195]:


# Create a line chart to visualize the ATR% (Average True Range as a percentage of price) over time
fig = px.line(
    tsla, 
    x=tsla.index, 
    y="atr_pct", 
    title="ATR% Over Time"  # Title of the chart
)

# Add a horizontal reference line at 3% to represent the low-volatility cutoff threshold
fig.add_hline(
    y=0.03, 
    line_dash="dot", 
    line_color="green", 
    annotation_text="Low Volatility Cutoff"
)

# Display the chart
fig.show()


# In[196]:


from IPython.display import IFrame

IFrame(src='figures/atr%_5y.html', width='100%', height='600px')


# The chart illustrates the historical volatility of TSLA, measured by the Average True Range (ATR) as a percentage of the closing price. Periods where the ATR% falls below the dotted green line at 3% indicate low volatility, which is typically associated with more stable market conditions. In contrast, noticeable spikes—such as those seen in 2020 and 2021—reflect periods of heightened volatility. More recently, ATR% values appear to remain closer to or slightly above the low-volatility threshold, suggesting relatively calmer market behavior compared to earlier years.

# In[197]:


# Filter the TSLA DataFrame to include only records from the year 2025
tsla_2025 = tsla[tsla.index.year == 2025]

# Create a line chart to visualize ATR% for TSLA during 2025
fig = px.line(
    tsla_2025,
    x=tsla_2025.index,
    y="atr_pct",
    title="ATR% Over Time (2025 Only)"
)

# Add a horizontal line at the 3% threshold to denote the low-volatility cutoff
fig.add_hline(
    y=0.03,
    line_dash="dot",
    line_color="green",
    annotation_text="Low Volatility Cutoff"
)

# Display the chart
fig.show()


# The chart displays ATR% for TSLA during 2025, reflecting how the stock's volatility has evolved since the start of the year. While ATR% began above the 7% mark in early January, it gradually declined and remained mostly between 4% and 6% throughout February. Although volatility did not breach the low-volatility threshold of 3%, the dip toward that level suggests a period of relative calm. Toward early March, ATR% showed a clear upward trend, indicating a potential resurgence in market volatility.

# In[198]:


# Create Buy Signal
tsla['Buy_Signal'] = tsla['VI+_'] > tsla['VI-_']  # Vortex crossover

# Create Sell Signal (basic)
tsla['Sell_Signal'] = tsla['VI-_'] > tsla['VI+_']

# Initialize the position tracking column with 0 (no active position)
tsla['Position'] = 0

# Initialize a variable to store the peak price during a position for trailing stop logic
peak_price = 0

# Iterate through the dataset starting from index 1 to access previous values
for i in range(1, len(tsla)):

    # Entry condition: enter a position if a buy signal is present
    if tsla['Buy_Signal'].iloc[i]:
        tsla.at[tsla.index[i], 'Position'] = 1  # Mark entry into a position
        peak_price = tsla['Close_TSLA'].iloc[i]  # Record the entry price as initial peak

    # If already in position, check for exit condition using trailing stop
    elif tsla['Position'].iloc[i - 1] == 1:
        current_price = tsla['Close_TSLA'].iloc[i]  # Current closing price
        peak_price = max(peak_price, current_price)  # Update peak price if current exceeds previous
        drawdown = (peak_price - current_price) / peak_price  # Compute drawdown from peak

        # Exit condition: drawdown exceeds 3%
        if drawdown >= 0.03:
            tsla.at[tsla.index[i], 'Sell_Signal'] = True  # Trigger a sell signal
            tsla.at[tsla.index[i], 'Position'] = 0        # Exit position
        else:
            tsla.at[tsla.index[i], 'Position'] = 1        # Maintain position

# Display the total number of buy and sell signals generated across the dataset
print("Buy signals:", tsla['Buy_Signal'].sum())
print("Sell signals:", tsla['Sell_Signal'].sum())


# In[199]:


# Create an empty figure object
fig = go.Figure()

# Plot the TSLA closing price as a continuous line
fig.add_trace(go.Scatter(
    x=tsla.index,
    y=tsla['Close_TSLA'],
    mode='lines',
    name='TSLA Price'
))

# Add markers to indicate Buy Signals using upward-pointing green triangles
fig.add_trace(go.Scatter(
    x=tsla[tsla['Buy_Signal']].index,
    y=tsla[tsla['Buy_Signal']]['Close_TSLA'],
    mode='markers',
    marker=dict(symbol='triangle-up', size=10, color='green'),
    name='Buy Signal'
))

# Add markers to indicate Sell Signals using downward-pointing red triangles
fig.add_trace(go.Scatter(
    x=tsla[tsla['Sell_Signal']].index,
    y=tsla[tsla['Sell_Signal']]['Close_TSLA'],
    mode='markers',
    marker=dict(symbol='triangle-down', size=10, color='red'),
    name='Sell Signal'
))

# Update layout settings including title and visual style
fig.update_layout(
    title='TSLA Buy & Sell Signals',
    template='plotly_white'
)

# Render the interactive plot
fig.show()


# The chart illustrates the closing price of Tesla stock over time, with overlaid trading signals generated by the strategy. Green upward triangles represent buy signals, while red downward triangles mark sell signals. These signals are distributed throughout periods of both rising and falling prices, reflecting how the algorithm dynamically enters and exits positions based on market conditions. Clusters of signals during high-volatility periods—such as 2020, 2021, and early 2025—indicate frequent entries and exits, whereas more stable phases show fewer trades.

# In[200]:


# Calculate ATR as a percentage of the closing price to normalize volatility
tsla['atr_pct'] = tsla['ATR_10'] / tsla['Close_TSLA']

# Define Vortex Indicator crossover signals:
# - VI_Cross_Up: Identifies when VI+ crosses above VI− (potential bullish signal)
# - VI_Cross_Down: Identifies when VI− crosses above VI+ (potential bearish signal)
tsla['VI_Cross_Up'] = (tsla['VI+_'] > tsla['VI-_']) & (tsla['VI+_'].shift(1) <= tsla['VI-_'].shift(1))
tsla['VI_Cross_Down'] = (tsla['VI-_'] > tsla['VI+_']) & (tsla['VI-_'].shift(1) <= tsla['VI+_'].shift(1))

# Initialize signal and state columns
tsla['Buy_Signal'] = False          # Flag for buy signal
tsla['Sell_Signal'] = False         # Flag for sell signal
tsla['Position'] = 0                # Position state: 1 = in position, 0 = no position
tsla['Entry_Type'] = None           # Strategy classification: 'aggressive' or 'conservative'

# Initialize control variables for trailing stop and price tracking
in_position = False                 # Boolean flag for current position state
peak_price = 0                      # Highest price observed during an open position

# Iterate through the DataFrame to simulate trading logic based on Vortex signals and volatility
for i in range(1, len(tsla)):
    row = tsla.iloc[i]
    idx = tsla.index[i]

    # Buy condition: Enter a new position if VI_Cross_Up occurs and no current position is held
    if not in_position and row['VI_Cross_Up']:
        tsla.at[idx, 'Buy_Signal'] = True
        tsla.at[idx, 'Position'] = 1
        in_position = True
        peak_price = row['Close_TSLA']

        # Classify entry type based on volatility threshold
        if row['atr_pct'] < 0.03:
            tsla.at[idx, 'Entry_Type'] = 'aggressive'
        else:
            tsla.at[idx, 'Entry_Type'] = 'conservative'

    # While in position, evaluate for trailing stop or VI_Cross_Down exit condition
    elif in_position:
        current_price = row['Close_TSLA']
        peak_price = max(peak_price, current_price)
        drawdown = (peak_price - current_price) / peak_price

        # Sell condition: Exit if drawdown exceeds 3% or VI_Cross_Down occurs
        if drawdown >= 0.03 or row['VI_Cross_Down']:
            tsla.at[idx, 'Sell_Signal'] = True
            tsla.at[idx, 'Position'] = 0
            in_position = False
        else:
            tsla.at[idx, 'Position'] = 1  # Maintain position

# Output the total count of each type of signal and entry classification
print("Buy signals:", tsla['Buy_Signal'].sum())
print("Sell signals:", tsla['Sell_Signal'].sum())
print("Aggressive entries:", (tsla['Entry_Type'] == 'aggressive').sum())
print("Conservative entries:", (tsla['Entry_Type'] == 'conservative').sum())


# In[201]:


# Create an empty figure to hold all plot layers
fig = go.Figure()

# Plot the TSLA closing price as a continuous blue line
fig.add_trace(go.Scatter(
    x=tsla.index,
    y=tsla['Close_TSLA'],
    mode='lines',
    name='TSLA Price',
    line=dict(color='blue')
))

# Add markers for aggressive buy signals (Entry_Type = 'aggressive')
fig.add_trace(go.Scatter(
    x=tsla[(tsla['Buy_Signal']) & (tsla['Entry_Type'] == 'aggressive')].index,
    y=tsla[(tsla['Buy_Signal']) & (tsla['Entry_Type'] == 'aggressive')]['Close_TSLA'],
    mode='markers',
    name='Buy (Aggressive)',
    marker=dict(symbol='triangle-up', color='limegreen', size=10)
))

# Add markers for conservative buy signals (Entry_Type = 'conservative')
fig.add_trace(go.Scatter(
    x=tsla[(tsla['Buy_Signal']) & (tsla['Entry_Type'] == 'conservative')].index,
    y=tsla[(tsla['Buy_Signal']) & (tsla['Entry_Type'] == 'conservative')]['Close_TSLA'],
    mode='markers',
    name='Buy (Conservative)',
    marker=dict(symbol='triangle-up', color='green', size=10)
))

# Add markers for sell signals using red downward-pointing triangles
fig.add_trace(go.Scatter(
    x=tsla[tsla['Sell_Signal']].index,
    y=tsla[tsla['Sell_Signal']]['Close_TSLA'],
    mode='markers',
    name='Sell Signal',
    marker=dict(symbol='triangle-down', color='red', size=10)
))

# Configure chart layout with appropriate title, axis labels, and style
fig.update_layout(
    title='TSLA Buy/Sell Signals Over Time',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    template='plotly_white',
    height=600
)

# Render the figure
fig.show()


# The chart displays the historical closing price of Tesla (TSLA) stock alongside algorithmically generated buy and sell signals. The blue line represents TSLA's closing price, while the green upward-pointing triangles indicate buy entries—distinguished by lime green for aggressive entries (lower volatility) and dark green for conservative entries (higher volatility). Red downward-pointing triangles represent sell signals.
# 
# The buy signals are generally aligned with upward momentum, and sell signals frequently follow periods of short-term retracement or heightened volatility. The system shows particularly dense activity around highly volatile phases, such as mid-2020 to early 2022, capturing many entries and exits. In contrast, during more stable periods, the signals are more spaced out. Overall, the plot provides a clear visual assessment of how the strategy adapts dynamically to changing market conditions by modulating its entries based on volatility and exiting with protective trailing logic.

# ## Tesla Analysis Results

# In[202]:


tsla_signals = tsla.reset_index()[['Date', 'VI_Cross_Up', 'VI_Cross_Down', 'atr_pct', 'Close_TSLA']]


# In[203]:


merged_data = pd.merge(merged_data, tsla, on='Date', how='left')


# In[204]:


# Calculate ATR percentage
merged_data['atr_pct'] = merged_data['ATR_10'] / merged_data['Close_TSLA']

# Vortex crossover logic
merged_data['VI_Cross_Up'] = (merged_data['VI+_'] > merged_data['VI-_']) & (merged_data['VI+_'].shift(1) <= merged_data['VI-_'].shift(1))
merged_data['VI_Cross_Down'] = (merged_data['VI-_'] > merged_data['VI+_']) & (merged_data['VI-_'].shift(1) <= merged_data['VI+_'].shift(1))

# Initialize signal & state columns
merged_data['Buy_Signal'] = False
merged_data['Sell_Signal'] = False
merged_data['Position'] = 0
merged_data['Entry_Type'] = None  # aggressive/conservative

# Trailing stop logic variables
in_position = False
peak_price = 0

for i in range(1, len(merged_data)):
    row = merged_data.iloc[i]
    idx = merged_data.index[i]
    # Buy condition
    if not in_position or row['VI_Cross_Up'] or row['5_day_avg_sentiment_norm']>0:
        merged_data.at[idx, 'Buy_Signal'] = True
        merged_data.at[idx, 'Position'] = 1
        in_position = True
        peak_price = row['Close_TSLA']

        # Entry Type: aggressive if ATR < 3%, else conservative
        if row['atr_pct'] < 0.03:
            merged_data.at[idx, 'Entry_Type'] = 'aggressive'
        else:
            merged_data.at[idx, 'Entry_Type'] = 'conservative'

    # While in position, check for trailing stop or VI cross down
    elif in_position:
        current_price = row['Close_TSLA']
        peak_price = max(peak_price, current_price)
        drawdown = (peak_price - current_price) / peak_price

        if drawdown >= 0.03 or row['VI_Cross_Down']:
            merged_data.at[idx, 'Sell_Signal'] = True
            merged_data.at[idx, 'Position'] = 0
            in_position = False
        else:
            merged_data.at[idx, 'Position'] = 1

# Show result counts
print("Buy signals:", merged_data['Buy_Signal'].sum())
print("Sell signals:", merged_data['Sell_Signal'].sum())
print("Aggressive entries:", (merged_data['Entry_Type'] == 'aggressive').sum())
print("Conservative entries:", (merged_data['Entry_Type'] == 'conservative').sum())


# In[205]:


# Ensure 'Date' is datetime and set as index if needed
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

fig = go.Figure()

# Plot 5-day Avg Sentiment
fig.add_trace(go.Scatter(
    x=merged_data['Date'],
    y=merged_data['5_day_avg_sentiment_norm'],
    mode='lines+markers',
    name='5-Day Avg Sentiment',
    line=dict(color='blue')
))

# Plot ATR %
fig.add_trace(go.Scatter(
    x=merged_data['Date'],
    y=merged_data['atr_pct'],
    mode='lines+markers',
    name='ATR %',
    yaxis='y2',
    line=dict(color='orange')
))

# Optional: Highlight Buy Signal Dates (even though there are none now)
fig.add_trace(go.Scatter(
    x=merged_data.loc[merged_data['Buy_Signal'], 'Date'],
    y=merged_data.loc[merged_data['Buy_Signal'], '5_day_avg_sentiment_norm'],
    mode='markers',
    marker=dict(color='green', size=10, symbol='star'),
    name='Buy Signal'
))

# Add dual axis layout
fig.update_layout(
    title="5-Day Sentiment vs ATR % (with Buy Signals)",
    xaxis_title='Date',
    yaxis=dict(title='5-Day Avg Sentiment'),
    yaxis2=dict(title='ATR %', overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99),
    height=500
)

fig.show()


# In[206]:


# Initialize portfolio variables
capital = 100000                   # Starting capital for the simulation
in_position = False               # Flag indicating whether a position is currently held
entry_price = 0                   # Entry price of the current position
position_value = 0                # Dollar value allocated to the position
cash = capital                    # Available cash (initially equal to capital)
returns = []                      # List to store profit/loss for each trade

# Iterate over the dataset to simulate trading
for i in range(len(merged_data)):
    row = merged_data.iloc[i]

    # ==== Buy Logic ====
    if row['Buy_Signal'] and not in_position:
        position_size = row['position_size']             # Fraction of capital to allocate
        position_value = cash * position_size            # Calculate how much capital to invest
        entry_price = row['Close_TSLA']                  # Record entry price
        shares_bought = position_value / entry_price     # Calculate number of shares to buy
        cash -= position_value                           # Deduct invested capital from cash
        in_position = True                               # Update position flag

    # ==== Sell Logic ====
    elif row['Sell_Signal'] and in_position:
        exit_price = row['Close_TSLA']                   # Get the exit price
        proceeds = shares_bought * exit_price            # Calculate proceeds from sale
        profit = proceeds - position_value               # Profit = proceeds - initial investment
        cash += proceeds                                 # Add proceeds back to cash
        returns.append(profit)                           # Record trade return
        in_position = False                              # Reset position state
        position_value = 0                               # Clear position value
        entry_price = 0                                  # Reset entry price

# ==== Final Capital Calculation ====
# If still holding a position, add unrealized value to cash
final_value = cash + (shares_bought * row['Close_TSLA'] if in_position else 0)
total_return = final_value - capital                    # Net profit/loss from strategy

# ==== Print Performance Metrics ====
print(f"Final Capital: ${final_value:,.2f}")
print(f"Total Return: ${total_return:.2f}")
print(f"Total Trades: {len(returns)}")
print(f"Average Profit per Trade: ${np.mean(returns):.2f}")


# In[207]:


# Make sure index is datetime and 'Close_TSLA' exists
price = tsla['Close_TSLA']

# Generate entries and exits from your signals
entries = tsla['Buy_Signal']
exits = tsla['Sell_Signal']

# Create portfolio
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    size=np.nan,  # Let it auto-calculate position size if fixed capital
    init_cash=100_000,
    fees=0.001,  # 0.1% per trade
    slippage=0.0005  # Optional
)


# In[208]:


# Summary stats
print(portfolio.stats())

# Equity curve
portfolio.plot().show()


# In[209]:


print(tsla['Buy_Signal'].sum())  # Should be > 0
print(tsla['Sell_Signal'].sum())  # Should also be > 0


# In[210]:


tsla = tsla.dropna(subset=['Close_TSLA'])
entries = tsla['Buy_Signal'].astype(bool)
exits = tsla['Sell_Signal'].astype(bool)


# In[211]:


price = tsla['Close_TSLA']
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=100_000,
    fees=0.001
)

print(portfolio.stats())
portfolio.plot().show()


# The backtest results show that while the strategy achieved a total return of approximately 62.76%, it significantly underperformed compared to a simple buy-and-hold strategy on TSLA, which yielded a 1215.81% return. The strategy executed 80 trades with a low win rate of 32.5%, indicating that most trades were unprofitable. Although it had a few strong winners, the average profit per trade was marginal, with a profit factor of 1.19. Additionally, the portfolio experienced a substantial maximum drawdown of 55.35% and a prolonged recovery period lasting two years, signaling high risk. Visuals further confirm that many trades resulted in small losses or gains, with only a few notable profitable exits. Overall, while the strategy demonstrates some profitability, its risk-return profile is weak and may require optimization in entry/exit logic, volatility filtering, or sentiment integration to compete with the benchmark performance.

# ## XLY Analysis Results

# In[212]:


#url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from=20250101T0130&time_to=20250301T0130&limit=1000&tickers=XLY&apikey=PNM5EHRALIOT1CKJ'

#response = requests.get(url)

#if response.status_code == 200:
 #   sentiment_data = response.json()
  #  sentiment_df = pd.DataFrame(sentiment_data['feed']) 
   # print(sentiment_df.head())
#else:
 #   print("API call failed:", response.status_code)

#sentiment_json = response.json()
#print(sentiment_json)


# In[213]:


sentiment_data = []
for news_item in sentiment_json.get("feed", []):
    sentiment_data.append({
            "time_published": pd.to_datetime(news_item["time_published"]),
            "sentiment_score": news_item["overall_sentiment_score"],
            "sentiment_label": news_item["overall_sentiment_label"],
    })
sentiment_data = pd.DataFrame(sentiment_data)


# In[214]:


sentiment_data['time_published'] = sentiment_data['time_published'].dt.date
sentiment_scores_filtered = sentiment_data[pd.to_datetime(sentiment_data['time_published']).isin(tsla.index)]
sentiment_scores_filtered = sentiment_scores_filtered.groupby('time_published')['sentiment_score'].mean().reset_index()


# In[215]:


# Fix the multi-level column issue by selecting the 'Volume' column and resetting its name
xly_volume = xly[('Volume', 'XLY')].rename('Volume')

# Ensure the index of tsla_volume is a column and convert it to match the type of time_published
xly_volume = xly_volume.reset_index()
xly_volume['Date'] = pd.to_datetime(xly_volume['Date'])

# Convert time_published to datetime
sentiment_scores_filtered['time_published'] = pd.to_datetime(sentiment_scores_filtered['time_published'])
# Merge the dataframes
merged_data = pd.merge(xly_volume, sentiment_scores_filtered, left_on='Date', right_on='time_published', how='inner')
merged_data['Weighted_Sentiment'] = merged_data['Volume'] * merged_data['sentiment_score']
merged_data['5_day_avg_sentiment'] = merged_data['Weighted_Sentiment'].rolling(window=5).mean()
merged_data['Buy_Condition'] = merged_data['5_day_avg_sentiment'] > 0
merged_data['5_day_avg_sentiment_norm'] = merged_data['5_day_avg_sentiment']/merged_data['Volume'].mean()


# In[216]:


# Flatten MultiIndex columns 
xly.columns = [
    '_'.join(col).strip() if isinstance(col, tuple) else col
    for col in xly.columns
]

# Calculate True Range
xly["prev_close"] = xly["Close_XLY"].shift(1)
xly["tr1"] = xly["High_XLY"] - xly["Low_XLY"]
xly["tr2"] = abs(xly["High_XLY"] - xly["prev_close"])
xly["tr3"] = abs(xly["Low_XLY"] - xly["prev_close"])

xly["true_range"] = xly[["tr1", "tr2", "tr3"]].max(axis=1)

# 10-day ATR
xly["ATR_10"] = xly["true_range"].rolling(window=10).mean()

# ---- STEP 4: Calculate ATR as a percentage of closing price ----
xly["atr_pct"] = xly["ATR_10"] / xly["Close_XLY"]

# allocating the capital

def position_size(row):
    if row["atr_pct"] < 0.03:  # < 3% volatility → low risk
        return 0.01  # allocate 1% of capital
    else:  # ≥ 3% volatility → high risk
        return 0.005  # allocate 0.5% of capital

xly["position_size"] = xly.apply(position_size, axis=1)

# ---- STEP 6: Optional - Capital allocation per trade ----
#capital = 100000 # Example: $100K total portfolio
#xly["allocation_dollars"] = xly["position_size"] * capital

# ---- Preview ----
print(xly[["Close_XLY", "ATR_10", "atr_pct", "position_size"]].tail(10))


# In[217]:


import plotly.express as px
fig = px.line(xly, x=xly.index, y="atr_pct", title="ATR% Over Time")
fig.add_hline(y=0.03, line_dash="dot", line_color="green", annotation_text="Low Volatility Cutoff")
fig.show()


# In[218]:


import plotly.express as px

# Filter only 2025 data
xly_2025 = xly[xly.index.year == 2025]

# Plot
fig = px.line(xly_2025, x=xly_2025.index, y="atr_pct", title="ATR% Over Time (2025 Only)")
fig.add_hline(y=0.03, line_dash="dot", line_color="green", annotation_text="Low Volatility Cutoff")
fig.show()


# In[219]:


merged_data = pd.merge(merged_data, xly, on='Date', how='left')


# In[220]:


# Calculate ATR percentage
merged_data['atr_pct'] = merged_data['ATR_10'] / merged_data['Close_XLY']

# Vortex crossover logic
merged_data['VI_Cross_Up'] = (merged_data['VI+_'] > merged_data['VI-_']) & (merged_data['VI+_'].shift(1) <= merged_data['VI-_'].shift(1))
merged_data['VI_Cross_Down'] = (merged_data['VI-_'] > merged_data['VI+_']) & (merged_data['VI-_'].shift(1) <= merged_data['VI+_'].shift(1))

# Initialize signal & state columns
merged_data['Buy_Signal'] = False
merged_data['Sell_Signal'] = False
merged_data['Position'] = 0
merged_data['Entry_Type'] = None  # aggressive/conservative

# Trailing stop logic variables
in_position = False
peak_price = 0

for i in range(1, len(merged_data)):
    row = merged_data.iloc[i]
    idx = merged_data.index[i]
    # Buy condition
    if not in_position or row['VI_Cross_Up'] or row['5_day_avg_sentiment_norm']>0:
        merged_data.at[idx, 'Buy_Signal'] = True
        merged_data.at[idx, 'Position'] = 1
        in_position = True
        peak_price = row['Close_XLY']

        # Entry Type: aggressive if ATR < 3%, else conservative
        if row['atr_pct'] < 0.03:
            merged_data.at[idx, 'Entry_Type'] = 'aggressive'
        else:
            merged_data.at[idx, 'Entry_Type'] = 'conservative'

    # While in position, check for trailing stop or VI cross down
    elif in_position:
        current_price = row['Close_XLY']
        peak_price = max(peak_price, current_price)
        drawdown = (peak_price - current_price) / peak_price

        if drawdown >= 0.03 or row['VI_Cross_Down']:
            merged_data.at[idx, 'Sell_Signal'] = True
            merged_data.at[idx, 'Position'] = 0
            in_position = False
        else:
            merged_data.at[idx, 'Position'] = 1

# Show result counts
print("Buy signals:", merged_data['Buy_Signal'].sum())
print("Sell signals:", merged_data['Sell_Signal'].sum())
print("Aggressive entries:", (merged_data['Entry_Type'] == 'aggressive').sum())
print("Conservative entries:", (merged_data['Entry_Type'] == 'conservative').sum())


# In[221]:


fig = go.Figure()

# Plot merged_data closing price
fig.add_trace(go.Scatter(
    x=merged_data.index, 
    y=merged_data['Close_XLY'], 
    mode='lines', 
    name='merged_data Price', 
    line=dict(color='blue')
))

# Aggressive buys
fig.add_trace(go.Scatter(
    x=merged_data[(merged_data['Buy_Signal']) & (merged_data['Entry_Type'] == 'aggressive')].index,
    y=merged_data[(merged_data['Buy_Signal']) & (merged_data['Entry_Type'] == 'aggressive')]['Close_XLY'],
    mode='markers',
    name='Buy (Aggressive)',
    marker=dict(symbol='triangle-up', color='limegreen', size=10)
))

# Conservative buys
fig.add_trace(go.Scatter(
    x=merged_data[(merged_data['Buy_Signal']) & (merged_data['Entry_Type'] == 'conservative')].index,
    y=merged_data[(merged_data['Buy_Signal']) & (merged_data['Entry_Type'] == 'conservative')]['Close_XLY'],
    mode='markers',
    name='Buy (Conservative)',
    marker=dict(symbol='triangle-up', color='green', size=10)
))

# Sells
fig.add_trace(go.Scatter(
    x=merged_data[merged_data['Sell_Signal']].index,
    y=merged_data[merged_data['Sell_Signal']]['Close_XLY'],
    mode='markers',
    name='Sell Signal',
    marker=dict(symbol='triangle-down', color='red', size=10)
))

fig.update_layout(
    title='merged_data Buy/Sell Signals Over Time',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    template='plotly_white',
    height=600
)

fig.show()


# In[222]:


# Ensure 'Date' is datetime and set as index if needed
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

fig = go.Figure()

# Plot 5-day Avg Sentiment
fig.add_trace(go.Scatter(
    x=merged_data['Date'],
    y=merged_data['5_day_avg_sentiment_norm'],
    mode='lines+markers',
    name='5-Day Avg Sentiment',
    line=dict(color='blue')
))

# Plot ATR %
fig.add_trace(go.Scatter(
    x=merged_data['Date'],
    y=merged_data['atr_pct'],
    mode='lines+markers',
    name='ATR %',
    yaxis='y2',
    line=dict(color='orange')
))

# Optional: Highlight Buy Signal Dates (even though there are none now)
fig.add_trace(go.Scatter(
    x=merged_data.loc[merged_data['Buy_Signal'], 'Date'],
    y=merged_data.loc[merged_data['Buy_Signal'], '5_day_avg_sentiment_norm'],
    mode='markers',
    marker=dict(color='green', size=10, symbol='star'),
    name='Buy Signal'
))

# Add dual axis layout
fig.update_layout(
    title="5-Day Sentiment vs ATR % (with Buy Signals)",
    xaxis_title='Date',
    yaxis=dict(title='5-Day Avg Sentiment'),
    yaxis2=dict(title='ATR %', overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99),
    height=500
)

fig.show()


# In[223]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=merged_data.index, y=merged_data['Close_XLY'], mode='lines', name='merged_data Price'))

# Buy markers
fig.add_trace(go.Scatter(
    x=merged_data[merged_data['Buy_Signal']].index,
    y=merged_data[merged_data['Buy_Signal']]['Close_XLY'],
    mode='markers',
    marker=dict(symbol='triangle-up', size=10, color='green'),
    name='Buy Signal'
))

# Sell markers
fig.add_trace(go.Scatter(
    x=merged_data[merged_data['Sell_Signal']].index,
    y=merged_data[merged_data['Sell_Signal']]['Close_XLY'],
    mode='markers',
    marker=dict(symbol='triangle-down', size=10, color='red'),
    name='Sell Signal'
))

fig.update_layout(title='XLY Buy & Sell Signals', template='plotly_white')
fig.show()


# In[224]:


capital = 100000
in_position = False
entry_price = 0
position_value = 0
cash = capital
returns = []

for i in range(len(merged_data)):
    row = merged_data.iloc[i]
    
    # Buy
    if row['Buy_Signal'] and not in_position:
        position_size = row['position_size']
        position_value = cash * position_size
        entry_price = row['Close_XLY']
        shares_bought = position_value / entry_price
        cash -= position_value
        in_position = True
        
    # Sell
    elif row['Sell_Signal'] and in_position:
        exit_price = row['Close_XLY']
        proceeds = shares_bought * exit_price
        profit = proceeds - position_value
        cash += proceeds
        returns.append(profit)
        in_position = False
        position_value = 0
        entry_price = 0

# Final capital
final_value = cash + (shares_bought * row['Close_XLY'] if in_position else 0)
total_return = final_value - capital

print(f"Final Capital: ${final_value:.2f}")
print(f"Total Return: ${total_return:.2f}")
print(f"Total Trades: {len(returns)}")
print(f"Average Profit per Trade: ${np.mean(returns):.2f}")


# ### Without sentiment code

# In[225]:


# Without sentiment score
xly_copy = xly.copy()
xly_copy['atr_pct'] = xly_copy['ATR_10'] / xly_copy['Close_XLY']

# Create Buy Signal (assuming VI_Cross_Up is defined elsewhere)
xly_copy['Buy_Signal'] = xly_copy['VI+_'] > xly_copy['VI-_']  # Vortex crossover
# + add any other buy conditions here...

# Create Sell Signal (basic)
xly_copy['Sell_Signal'] = xly_copy['VI-_'] > xly_copy['VI+_']

# Initialize position state
xly_copy['Position'] = 0
peak_price = 0

for i in range(1, len(xly_copy)):
    if xly_copy['Buy_Signal'].iloc[i]:
        xly_copy.at[xly_copy.index[i], 'Position'] = 1
        peak_price = xly_copy['Close_XLY'].iloc[i]
    elif xly_copy['Position'].iloc[i - 1] == 1:
        current_price = xly_copy['Close_XLY'].iloc[i]
        peak_price = max(peak_price, current_price)
        drawdown = (peak_price - current_price) / peak_price

        if drawdown >= 0.03:
            xly_copy.at[xly_copy.index[i], 'Sell_Signal'] = True  # trailing stop
            xly_copy.at[xly_copy.index[i], 'Position'] = 0
        else:
            xly_copy.at[xly_copy.index[i], 'Position'] = 1


# In[226]:


capital = 100000
in_position = False
entry_price = 0
position_value = 0
cash = capital
returns = []

for i in range(len(xly_copy)):
    row = xly_copy.iloc[i]
    
    # Buy
    if row['Buy_Signal'] and not in_position:
        position_size = row['position_size']
        position_value = cash * position_size
        entry_price = row['Close_XLY']
        shares_bought = position_value / entry_price
        cash -= position_value
        in_position = True
        
    # Sell
    elif row['Sell_Signal'] and in_position:
        exit_price = row['Close_XLY']
        proceeds = shares_bought * exit_price
        profit = proceeds - position_value
        cash += proceeds
        returns.append(profit)
        in_position = False
        position_value = 0
        entry_price = 0

# Final capital
final_value = cash + (shares_bought * row['Close_XLY'] if in_position else 0)
total_return = final_value - capital

print(f"Final Capital: ${final_value:.2f}")
print(f"Total Return: ${total_return:.2f}")
print(f"Total Trades: {len(returns)}")
print(f"Average Profit per Trade: ${np.mean(returns):.2f}")


# In[227]:


xly = xly_copy.dropna(subset=['Close_XLY'])
entries = xly_copy['Buy_Signal'].astype(bool)
exits = xly_copy['Sell_Signal'].astype(bool)

price = xly_copy['Close_XLY']
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=100_000,
    fees=0.001
)

print(portfolio.stats())
portfolio.plot().show()


# In[228]:


# Make sure index is datetime and 'Close_TSLA' exists
price = merged_data['Close_XLY']

# Generate entries and exits from your signals
entries = merged_data['Buy_Signal']
exits = merged_data['Sell_Signal']

# Create portfolio
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    size=np.nan,  # Let it auto-calculate position size if fixed capital
    init_cash=100_000,
    fees=0.001,  # 0.1% per trade
    slippage=0.0005  # Optional
)

# Plot portfolio value
portfolio.plot().show()


# In[229]:


# Summary stats
print(portfolio.stats())

# Equity curve
portfolio.plot().show()


# In[230]:


xly = merged_data.dropna(subset=['Close_XLY'])
entries = merged_data['Buy_Signal'].astype(bool)
exits = merged_data['Sell_Signal'].astype(bool)


# In[231]:


price = merged_data['Close_XLY']
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=100_000,
    fees=0.001
)

print(portfolio.stats())
portfolio.plot().show()


# ## SPY Analysis Results

# In[232]:


# Flatten MultiIndex columns 
spy.columns = [
    '_'.join(col).strip() if isinstance(col, tuple) else col
    for col in spy.columns
]

# Calculate True Range
spy["prev_close"] = spy["Close_SPY"].shift(1)
spy["tr1"] = spy["High_SPY"] - spy["Low_SPY"]
spy["tr2"] = abs(spy["High_SPY"] - spy["prev_close"])
spy["tr3"] = abs(spy["Low_SPY"] - spy["prev_close"])

spy["true_range"] = spy[["tr1", "tr2", "tr3"]].max(axis=1)

# 10-day ATR
spy["ATR_10"] = spy["true_range"].rolling(window=10).mean()

# ---- STEP 4: Calculate ATR as a percentage of closing price ----
spy["atr_pct"] = spy["ATR_10"] / spy["Close_SPY"]

# allocating the capital

def position_size(row):
    if row["atr_pct"] < 0.03:  # < 3% volatility → low risk
        return 0.01  # allocate 1% of capital
    else:  # ≥ 3% volatility → high risk
        return 0.005  # allocate 0.5% of capital

spy["position_size"] = spy.apply(position_size, axis=1)

# ---- STEP 6: Optional - Capital allocation per trade ----
#capital = 100000 # Example: $100K total portfolio
#spy["allocation_dollars"] = spy["position_size"] * capital

# ---- Preview ----
print(spy[["Close_SPY", "ATR_10", "atr_pct", "position_size"]].tail(10))


# In[233]:


fig = px.line(spy, x=spy.index, y="atr_pct", title="ATR% Over Time")
fig.add_hline(y=0.03, line_dash="dot", line_color="green", annotation_text="Low Volatility Cutoff")
fig.show()


# In[234]:


# Filter only 2025 data
spy_2025 = spy[spy.index.year == 2025]

# Plot
fig = px.line(spy_2025, x=spy_2025.index, y="atr_pct", title="ATR% Over Time (2025 Only)")
fig.add_hline(y=0.03, line_dash="dot", line_color="green", annotation_text="Low Volatility Cutoff")
fig.show()


# In[235]:


# Without sentiment score
spy_copy = spy.copy()
spy_copy['atr_pct'] = spy_copy['ATR_10'] / spy_copy['Close_SPY']

# Create Buy Signal (assuming VI_Cross_Up is defined elsewhere)
spy_copy['Buy_Signal'] = spy_copy['VI+_'] > spy_copy['VI-_']  # Vortex crossover

# Create Sell Signal (basic)
spy_copy['Sell_Signal'] = spy_copy['VI-_'] > spy_copy['VI+_']

# Initialize position state
spy_copy['Position'] = 0
peak_price = 0

for i in range(1, len(spy_copy)):
    if spy_copy['Buy_Signal'].iloc[i]:
        spy_copy.at[spy_copy.index[i], 'Position'] = 1
        peak_price = spy_copy['Close_SPY'].iloc[i]
    elif spy_copy['Position'].iloc[i - 1] == 1:
        current_price = spy_copy['Close_SPY'].iloc[i]
        peak_price = max(peak_price, current_price)
        drawdown = (peak_price - current_price) / peak_price

        if drawdown >= 0.03:
            spy_copy.at[spy_copy.index[i], 'Sell_Signal'] = True  # trailing stop
            spy_copy.at[spy_copy.index[i], 'Position'] = 0
        else:
            spy_copy.at[spy_copy.index[i], 'Position'] = 1


# In[236]:


capital = 100000
in_position = False
entry_price = 0
position_value = 0
cash = capital
returns = []

for i in range(len(spy_copy)):
    row = spy_copy.iloc[i]
    
    # Buy
    if row['Buy_Signal'] and not in_position:
        position_size = row['position_size']
        position_value = cash * position_size
        entry_price = row['Close_SPY']
        shares_bought = position_value / entry_price
        cash -= position_value
        in_position = True
        
    # Sell
    elif row['Sell_Signal'] and in_position:
        exit_price = row['Close_SPY']
        proceeds = shares_bought * exit_price
        profit = proceeds - position_value
        cash += proceeds
        returns.append(profit)
        in_position = False
        position_value = 0
        entry_price = 0

# Final capital
final_value = cash + (shares_bought * row['Close_SPY'] if in_position else 0)
total_return = final_value - capital

print(f"Final Capital: ${final_value:.2f}")
print(f"Total Return: ${total_return:.2f}")
print(f"Total Trades: {len(returns)}")
print(f"Average Profit per Trade: ${np.mean(returns):.2f}")


# In[237]:


import vectorbt as vbt

spy = spy_copy.dropna(subset=['Close_SPY'])
entries = spy_copy['Buy_Signal'].astype(bool)
exits = spy_copy['Sell_Signal'].astype(bool)

price = spy_copy['Close_SPY']
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=100_000,
    fees=0.001
)

print(portfolio.stats())
portfolio.plot().show()



# ## Optimization

# In[238]:


tsla = yf.download('TSLA', start='2019-01-01', end='2025-03-05')
xly = yf.download('XLY', start='2019-01-01', end='2025-03-05')
spy = yf.download('SPY', start='2019-01-01', end='2025-03-05')


# In[239]:


def calculate_vortex(df, value, n):
    """Calculate Vortex Indicator VI+ and VI-."""
    high = df[("High_"+value)]
    low = df[("Low_"+value)]
    close = df[("Close_"+value)]

    vm_plus = abs(high - low.shift(1))   # |Today's High - Yesterday's Low|
    vm_minus = abs(low - high.shift(1))  # |Today's Low - Yesterday's High|

    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    sum_vm_plus = vm_plus.rolling(window=n).sum()
    sum_vm_minus = vm_minus.rolling(window=n).sum()
    sum_tr = tr.rolling(window=n).sum()

    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr

    return vi_plus, vi_minus


# In[240]:


tsla.columns = [
    '_'.join(col).strip() if isinstance(col, tuple) else col
    for col in tsla.columns
]


# In[241]:


# Define a list of different smoothing periods to test for the Vortex Indicator
periods = [7, 14, 21, 30]
results = {}  # Dictionary to store performance metrics for each period

# Loop through each smoothing period
for n in periods:
    # === Compute Vortex Indicator for the given period ===
    tsla[f'VI+_{n}'], tsla[f'VI-_{n}'] = calculate_vortex(tsla, 'TSLA', n)

    # === Generate Buy/Sell signals based on crossover logic ===
    # Buy when VI+ crosses above VI-
    tsla[f'Buy_{n}'] = tsla[f'VI+_{n}'] > tsla[f'VI-_{n}']
    # Sell when VI- crosses above VI+
    tsla[f'Sell_{n}'] = tsla[f'VI-_{n}'] > tsla[f'VI+_{n}']

    # === Convert boolean signals to actual entry/exit Series ===
    entries = tsla[f'Buy_{n}']
    exits = tsla[f'Sell_{n}']

    # === Run a backtest using vectorbt Portfolio object ===
    portfolio = vbt.Portfolio.from_signals(
        close=tsla['Close_TSLA'],  # TSLA closing prices
        entries=entries,
        exits=exits,
        size=1,  # Assume buying 1 share per trade
        init_cash=10_000  # Initial capital for backtest
    )

    # === Store backtest performance metrics in results dict ===
    stats = portfolio.stats()
    results[n] = stats

# Identify the period with the highest total return
best_period = max(results, key=lambda x: results[x]['Total Return [%]'])
print(f"✅ Best Performing Period: {best_period} days")

# Rebuild portfolio using the best period to visualize it
portfolio = vbt.Portfolio.from_signals(
    close=tsla['Close_TSLA'],
    entries=tsla[f'VI+_{best_period}'] > tsla[f'VI-_{best_period}'],
    exits=tsla[f'VI-_{best_period}'] > tsla[f'VI+_{best_period}'],
    size=1,
    init_cash=10_000
)

# Plot the results of the best strategy
portfolio.plot().show()
print(portfolio.stats())


# ## Peer Comparison: Apple Analysis Results 

# In[242]:


appl = yf.download('AAPL', start='2019-01-01', end='2025-03-05')


# In[243]:


appl.columns = [
    '_'.join(col).strip() if isinstance(col, tuple) else col
    for col in appl.columns
]


# In[244]:


def calculate_vortex(df, value, n):
    """Calculate Vortex Indicator VI+ and VI-."""
    high = df[("High_"+value)]
    low = df[("Low_"+value)]
    close = df[("Close_"+value)]

    vm_plus = abs(high - low.shift(1))   # |Today's High - Yesterday's Low|
    vm_minus = abs(low - high.shift(1))  # |Today's Low - Yesterday's High|

    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    sum_vm_plus = vm_plus.rolling(window=n).sum()
    sum_vm_minus = vm_minus.rolling(window=n).sum()
    sum_tr = tr.rolling(window=n).sum()

    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr

    return vi_plus, vi_minus


# In[245]:


appl['VI+_'], appl['VI-_'] = calculate_vortex(appl, 'AAPL', 14)


# In[246]:


import json
import pandas as pd

# Load from file
with open("AAPL_sentiment_raw.json", "r") as f:
    sentiment_json = json.load(f)

# Extract feed
sentiment_feed = sentiment_json.get("feed", [])

# Extract useful fields
sentiment_data = []

for item in sentiment_feed:
    try:
        sentiment_data.append({
            "time_published": pd.to_datetime(item["time_published"]),
            "sentiment_score": float(item["overall_sentiment_score"]),
            "sentiment_label": item["overall_sentiment_label"],
        })
    except (KeyError, ValueError, TypeError):
        continue  # Skip malformed rows

# Convert to DataFrame
sentiment_df = pd.DataFrame(sentiment_data)
sentiment_df.set_index("time_published", inplace=True)

# View result
print(sentiment_df.head())
print(sentiment_df.info())


# In[247]:


sentiment_data = []
for news_item in sentiment_json.get("feed", []):
    sentiment_data.append({
            "time_published": pd.to_datetime(news_item["time_published"]),
            "sentiment_score": news_item["overall_sentiment_score"],
            "sentiment_label": news_item["overall_sentiment_label"],
    })
sentiment_data = pd.DataFrame(sentiment_data)
sentiment_data['time_published'] = sentiment_data['time_published'].dt.date


# In[248]:


sentiment_scores_filtered = sentiment_data[pd.to_datetime(sentiment_data['time_published']).isin(appl.index)]
sentiment_scores_filtered = sentiment_scores_filtered.groupby('time_published')['sentiment_score'].mean().reset_index()
print(len(sentiment_scores_filtered))


# In[249]:


appl_volume = appl[('Volume_AAPL')].reset_index()
appl_volume['Date'] = pd.to_datetime(appl_volume['Date'])

sentiment_scores_filtered['time_published'] = pd.to_datetime(sentiment_scores_filtered['time_published'])

merged_data = pd.merge(appl_volume, sentiment_scores_filtered, left_on='Date', right_on='time_published', how='inner')
merged_data['Weighted_Sentiment'] = merged_data['Volume_AAPL'] * merged_data['sentiment_score']
merged_data['5_day_avg_sentiment'] = merged_data['Weighted_Sentiment'].rolling(window=5).mean()
merged_data['Buy_Condition'] = merged_data['5_day_avg_sentiment'] > 0
merged_data['5_day_avg_sentiment_norm'] = merged_data['5_day_avg_sentiment']/merged_data['Volume_AAPL'].mean()

merged_data.head()


# In[250]:


# Calculate True Range
appl["prev_close"] = appl["Close_AAPL"].shift(1)
appl["tr1"] = appl["High_AAPL"] - appl["Low_AAPL"]
appl["tr2"] = abs(appl["High_AAPL"] - appl["prev_close"])
appl["tr3"] = abs(appl["Low_AAPL"] - appl["prev_close"])

appl["true_range"] = appl[["tr1", "tr2", "tr3"]].max(axis=1)

# 10-day ATR
appl["ATR_10"] = appl["true_range"].rolling(window=10).mean()

# ---- STEP 4: Calculate ATR as a percentage of closing price ----
appl["atr_pct"] = appl["ATR_10"] / appl["Close_AAPL"]

# allocating the capital

def position_size(row):
    if row["atr_pct"] < 0.03:  # < 3% volatility → low risk
        return 0.01  # allocate 1% of capital
    else:  # ≥ 3% volatility → high risk
        return 0.005  # allocate 0.5% of capital

appl["position_size"] = appl.apply(position_size, axis=1)

print(appl[["Close_AAPL", "ATR_10", "atr_pct", "position_size"]].tail(10))


# In[251]:


import plotly.express as px
fig = px.line(appl, x=appl.index, y="atr_pct", title="AAPL ATR% Over Time")
fig.add_hline(y=0.03, line_dash="dot", line_color="green", annotation_text="Low Volatility Cutoff")
fig.show()


# In[252]:


# Filter only 2025 data
appl_2025 = appl[appl.index.year == 2025]
fig = px.line(appl_2025, x=appl_2025.index, y="atr_pct", title="AAPL ATR% Over Time (2025 Only)")
fig.add_hline(y=0.03, line_dash="dot", line_color="green", annotation_text="Low Volatility Cutoff")
fig.show()


# In[253]:


merged_data = pd.merge(merged_data, appl, on='Date', how='left')


# In[254]:


# Calculate ATR percentage
merged_data['atr_pct'] = merged_data['ATR_10'] / merged_data['Close_AAPL']

# Vortex crossover logic
merged_data['VI_Cross_Up'] = (merged_data['VI+_'] > merged_data['VI-_']) & (merged_data['VI+_'].shift(1) <= merged_data['VI-_'].shift(1))
merged_data['VI_Cross_Down'] = (merged_data['VI-_'] > merged_data['VI+_']) & (merged_data['VI-_'].shift(1) <= merged_data['VI+_'].shift(1))

# Initialize signal & state columns
merged_data['Buy_Signal'] = False
merged_data['Sell_Signal'] = False
merged_data['Position'] = 0
merged_data['Entry_Type'] = None  # aggressive/conservative

# Trailing stop logic variables
in_position = False
peak_price = 0

for i in range(1, len(merged_data)):
    row = merged_data.iloc[i]
    idx = merged_data.index[i]
    # Buy condition
    if not in_position or row['VI_Cross_Up'] or row['5_day_avg_sentiment_norm']>0:
        merged_data.at[idx, 'Buy_Signal'] = True
        merged_data.at[idx, 'Position'] = 1
        in_position = True
        peak_price = row['Close_AAPL']

        # Entry Type: aggressive if ATR < 3%, else conservative
        if row['atr_pct'] < 0.03:
            merged_data.at[idx, 'Entry_Type'] = 'aggressive'
        else:
            merged_data.at[idx, 'Entry_Type'] = 'conservative'

    # While in position, check for trailing stop or VI cross down
    elif in_position:
        current_price = row['Close_AAPL']
        peak_price = max(peak_price, current_price)
        drawdown = (peak_price - current_price) / peak_price

        if drawdown >= 0.03 or row['VI_Cross_Down']:
            merged_data.at[idx, 'Sell_Signal'] = True
            merged_data.at[idx, 'Position'] = 0
            in_position = False
        else:
            merged_data.at[idx, 'Position'] = 1

# Show result counts
print("Buy signals:", merged_data['Buy_Signal'].sum())
print("Sell signals:", merged_data['Sell_Signal'].sum())
print("Aggressive entries:", (merged_data['Entry_Type'] == 'aggressive').sum())
print("Conservative entries:", (merged_data['Entry_Type'] == 'conservative').sum())


# In[255]:


import plotly.graph_objects as go

# Ensure 'Date' is datetime and set as index if needed
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

fig = go.Figure()

# Plot 5-day Avg Sentiment
fig.add_trace(go.Scatter(
    x=merged_data['Date'],
    y=merged_data['5_day_avg_sentiment_norm'],
    mode='lines+markers',
    name='5-Day Avg Sentiment',
    line=dict(color='blue')
))

# Plot ATR %
fig.add_trace(go.Scatter(
    x=merged_data['Date'],
    y=merged_data['atr_pct'],
    mode='lines+markers',
    name='ATR %',
    yaxis='y2',
    line=dict(color='orange')
))

# Optional: Highlight Buy Signal Dates (even though there are none now)
fig.add_trace(go.Scatter(
    x=merged_data.loc[merged_data['Buy_Signal'], 'Date'],
    y=merged_data.loc[merged_data['Buy_Signal'], '5_day_avg_sentiment_norm'],
    mode='markers',
    marker=dict(color='green', size=10, symbol='star'),
    name='Buy Signal'
))

# Add dual axis layout
fig.update_layout(
    title="5-Day Sentiment vs ATR % (with Buy Signals)",
    xaxis_title='Date',
    yaxis=dict(title='5-Day Avg Sentiment'),
    yaxis2=dict(title='ATR %', overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99),
    height=500
)

fig.show()


# In[256]:


capital = 100000
in_position = False
entry_price = 0
position_value = 0
cash = capital
returns = []

for i in range(len(merged_data)):
    row = merged_data.iloc[i]
    
    # Buy
    if row['Buy_Signal'] and not in_position:
        position_size = row['position_size']
        position_value = cash * position_size
        entry_price = row['Close_AAPL']
        shares_bought = position_value / entry_price
        cash -= position_value
        in_position = True
        
    # Sell
    elif row['Sell_Signal'] and in_position:
        exit_price = row['Close_AAPL']
        proceeds = shares_bought * exit_price
        profit = proceeds - position_value
        cash += proceeds
        returns.append(profit)
        in_position = False
        position_value = 0
        entry_price = 0

# Final capital
final_value = cash + (shares_bought * row['Close_AAPL'] if in_position else 0)
total_return = final_value - capital

print(f"Final Capital: ${final_value:.2f}")
print(f"Total Return: ${total_return:.2f}")
print(f"Total Trades: {len(returns)}")
print(f"Average Profit per Trade: ${np.mean(returns):.2f}")


# In[257]:


appl_ = merged_data.dropna(subset=['Close_AAPL'])
entries = merged_data['Buy_Signal'].astype(bool)
exits = merged_data['Sell_Signal'].astype(bool)

price = merged_data['Close_AAPL']
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=100_000,
    fees=0.001
)

print(portfolio.stats())
portfolio.plot().show()



# ### Without Sentiment

# In[258]:


# WITHOUT sentiment
appl_copy = appl.copy()
appl_copy['atr_pct'] = appl_copy['ATR_10'] / appl_copy['Close_AAPL']

# Create Buy Signal (assuming VI_Cross_Up is defined elsewhere)
appl_copy['Buy_Signal'] = appl_copy['VI+_'] > appl_copy['VI-_']  # Vortex crossover
# + add any other buy conditions here...

# Create Sell Signal (basic)
appl_copy['Sell_Signal'] = appl_copy['VI-_'] > appl_copy['VI+_']

# Initialize position state
appl_copy['Position'] = 0
peak_price = 0

for i in range(1, len(appl_copy)):
    if appl_copy['Buy_Signal'].iloc[i]:
        appl_copy.at[appl_copy.index[i], 'Position'] = 1
        peak_price = appl_copy['Close_AAPL'].iloc[i]
    elif appl_copy['Position'].iloc[i - 1] == 1:
        current_price = appl_copy['Close_AAPL'].iloc[i]
        peak_price = max(peak_price, current_price)
        drawdown = (peak_price - current_price) / peak_price

        if drawdown >= 0.03:
            appl_copy.at[appl_copy.index[i], 'Sell_Signal'] = True  # trailing stop
            appl_copy.at[appl_copy.index[i], 'Position'] = 0
        else:
            appl_copy.at[appl_copy.index[i], 'Position'] = 1


# In[259]:


capital = 100000
in_position = False
entry_price = 0
position_value = 0
cash = capital
returns = []

for i in range(len(appl_copy)):
    row = appl_copy.iloc[i]
    
    # Buy
    if row['Buy_Signal'] and not in_position:
        position_size = row['position_size']
        position_value = cash * position_size
        entry_price = row['Close_AAPL']
        shares_bought = position_value / entry_price
        cash -= position_value
        in_position = True
        
    # Sell
    elif row['Sell_Signal'] and in_position:
        exit_price = row['Close_AAPL']
        proceeds = shares_bought * exit_price
        profit = proceeds - position_value
        cash += proceeds
        returns.append(profit)
        in_position = False
        position_value = 0
        entry_price = 0

# Final capital
final_value = cash + (shares_bought * row['Close_AAPL'] if in_position else 0)
total_return = final_value - capital

print(f"Final Capital: ${final_value:.2f}")
print(f"Total Return: ${total_return:.2f}")
print(f"Total Trades: {len(returns)}")
print(f"Average Profit per Trade: ${np.mean(returns):.2f}")


# In[260]:


import vectorbt as vbt

appl = appl_copy.dropna(subset=['Close_AAPL'])
entries = appl_copy['Buy_Signal'].astype(bool)
exits = appl_copy['Sell_Signal'].astype(bool)

price = appl_copy['Close_AAPL']
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=100_000,
    fees=0.001
)

print(portfolio.stats())
portfolio.plot().show()


# Based on the results from applying the trading strategy to the Apple (AAPL) ticker, we can reasonably conclude that the strategy does work on peers like AAPL. The strategy delivered a total return of approximately 282% over the backtest period (2019–2025), compared to a benchmark return of about 526%, which indicates it captured a significant portion of the upward trend while actively managing trades. Although it underperformed the benchmark in absolute terms, this is typical of signal-driven strategies that trade in and out of the market. The profit factor of 2.11, expectancy of 4204, and a win rate of 45.5% suggest the strategy was profitable overall. Additionally, the drawdown was moderate (20.87%), reflecting a reasonable risk exposure relative to the potential reward.
# 
# The cumulative returns graph further supports this interpretation. The strategy closely follows the broader market trend, generating consistent gains and outperforming during certain periods. The trade PnL distribution shows a good number of winning trades with healthy profitability, and although there were losses, the downside was generally contained. Therefore, this peer comparison confirms that the strategy generalizes reasonably well beyond TSLA, making it a potentially viable approach for other high-liquidity technology stocks like AAPL.

# In[261]:


# Calculate ATR percentage
appl['atr_pct'] = appl['ATR_10'] / appl['Close_AAPL']
appl['Buy_Signal'] = appl['VI+_'] > appl['VI-_']  # Vortex crossover
appl['Sell_Signal'] = appl['VI-_'] > appl['VI+_']

# Initialize position state
appl['Position'] = 0
appl['Entry_Type'] = None
peak_price = 0

for i in range(1, len(appl)):
    if appl['Buy_Signal'].iloc[i]:
        appl.at[appl.index[i], 'Position'] = 1
        peak_price = appl['Close_AAPL'].iloc[i]
    elif appl['Position'].iloc[i - 1] == 1:
        current_price = appl['Close_AAPL'].iloc[i]
        peak_price = max(peak_price, current_price)
        drawdown = (peak_price - current_price) / peak_price

        if drawdown >= 0.03:
            appl.at[appl.index[i], 'Sell_Signal'] = True  # trailing stop
            appl.at[appl.index[i], 'Position'] = 0
        else:
            appl.at[appl.index[i], 'Position'] = 1

print("Buy signals:", appl['Buy_Signal'].sum())
print("Sell signals:", appl['Sell_Signal'].sum())


# In[262]:


import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=appl.index, y=appl['Close_AAPL'], mode='lines', name='AAPL Price'))

# Buy markers
fig.add_trace(go.Scatter(
    x=appl[appl['Buy_Signal']].index,
    y=appl[appl['Buy_Signal']]['Close_AAPL'],
    mode='markers',
    marker=dict(symbol='triangle-up', size=10, color='green'),
    name='Buy Signal'
))

# Sell markers
fig.add_trace(go.Scatter(
    x=appl[appl['Sell_Signal']].index,
    y=appl[appl['Sell_Signal']]['Close_AAPL'],
    mode='markers',
    marker=dict(symbol='triangle-down', size=10, color='red'),
    name='Sell Signal'
))

fig.update_layout(title='AAPL Buy & Sell Signals', template='plotly_white')
fig.show()


# ## VI Plots

# In[263]:


tsla = yf.download('TSLA', start='2019-01-01', end='2025-03-05')
xly = yf.download('XLY', start='2019-01-01', end='2025-03-05')
spy = yf.download('SPY', start='2019-01-01', end='2025-03-05')


# In[264]:


def calculate_vortex(df, value, n=14):
    high = df[("High", value)]
    low = df[("Low", value)]
    close = df[("Close", value)]

    # Calculate VM+ and VM-
    vm_plus = abs(high - low.shift(1))   # |Today's High - Yesterday's Low|
    vm_minus = abs(low - high.shift(1))  # |Today's Low - Yesterday's High|

    # Calculate True Range (TR)
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    # Rolling sum for lookback period
    sum_vm_plus = vm_plus.rolling(window=n).sum()
    sum_vm_minus = vm_minus.rolling(window=n).sum()
    sum_tr = tr.rolling(window=n).sum()

    # Compute VI+ and VI-
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr

    return vi_plus, vi_minus


# In[265]:


tsla['VI+'], tsla['VI-'] = calculate_vortex(tsla, 'TSLA')
xly['VI+'], xly['VI-'] = calculate_vortex(xly, 'XLY')
spy['VI+'], spy['VI-'] = calculate_vortex(spy, 'SPY')


# In[266]:


# Flatten MultiIndex columns 
tsla.columns = [
    '_'.join(col).strip() if isinstance(col, tuple) else col
    for col in tsla.columns
]

# Calculate True Range
tsla["prev_close"] = tsla["Close_TSLA"].shift(1)
tsla["tr1"] = tsla["High_TSLA"] - tsla["Low_TSLA"]
tsla["tr2"] = abs(tsla["High_TSLA"] - tsla["prev_close"])
tsla["tr3"] = abs(tsla["Low_TSLA"] - tsla["prev_close"])

tsla["true_range"] = tsla[["tr1", "tr2", "tr3"]].max(axis=1)

# 10-day ATR
tsla["ATR_10"] = tsla["true_range"].rolling(window=10).mean()

# ---- STEP 4: Calculate ATR as a percentage of closing price ----
tsla["atr_pct"] = tsla["ATR_10"] / tsla["Close_TSLA"]

# allocating the capital

def position_size(row):
    if row["atr_pct"] < 0.03:  # < 3% volatility → low risk
        return 0.01  # allocate 1% of capital
    else:  # ≥ 3% volatility → high risk
        return 0.005  # allocate 0.5% of capital

tsla["position_size"] = tsla.apply(position_size, axis=1)

# ---- STEP 6: Optional - Capital allocation per trade ----
#capital = 100000 # Example: $100K total portfolio
#tsla["allocation_dollars"] = tsla["position_size"] * capital

# ---- Preview ----
print(tsla[["Close_TSLA", "ATR_10", "atr_pct", "position_size"]].tail(10))


# In[267]:


# Create the line chart for ATR%
fig_atr_tsla = px.line(tsla, x=tsla.index, y="atr_pct", title="ATR% Over Time")

# Add horizontal line for low volatility threshold
fig_atr_tsla.add_hline(
    y=0.03,
    line_dash="dot",
    line_color="green",
    annotation_text="Low Volatility Cutoff"
)

# Display in Streamlit
st.subheader("ATR% Over Time for TSLA")
st.plotly_chart(fig_atr_tsla, use_container_width=True)


# In[268]:


# Create the figure for TSLA full period
fig_tsla_full = go.Figure()

# Add VI+ trace
fig_tsla_full.add_trace(go.Scatter(
    x=tsla.index,
    y=tsla["VI+_"],
    mode='lines',
    name='VI+_',
    line=dict(color='blue')
))

# Add VI- trace
fig_tsla_full.add_trace(go.Scatter(
    x=tsla.index,
    y=tsla["VI-_"],
    mode='lines',
    name='VI-_',
    line=dict(color='orange')
))

# Customize layout
fig_tsla_full.update_layout(
    title="Vortex Indicator (VI+ and VI−) for TSLA",
    xaxis_title="Date",
    yaxis_title="Value",
    legend=dict(x=0, y=1.1, orientation="h"),
    template="plotly_white"
)

# Display in Streamlit
st.subheader("Vortex Indicator for TSLA - Full Period")
st.plotly_chart(fig_tsla_full, use_container_width=True)


# In[269]:


# Filter TSLA data for 2025
tsla_2025 = tsla.loc["2025"]

# Create the figure
fig_tsla_2025 = go.Figure()

# Add VI+ trace
fig_tsla_2025.add_trace(go.Scatter(
    x=tsla_2025.index,
    y=tsla_2025["VI+_"],
    mode='lines',
    name='VI+_',
    line=dict(color='blue')
))

# Add VI- trace
fig_tsla_2025.add_trace(go.Scatter(
    x=tsla_2025.index,
    y=tsla_2025["VI-_"],
    mode='lines',
    name='VI-_',
    line=dict(color='orange')
))

# Update layout
fig_tsla_2025.update_layout(
    title="Vortex Indicator (VI+ and VI−) for TSLA – 2025",
    xaxis_title="Date",
    yaxis_title="Value",
    legend=dict(x=0, y=1.1, orientation="h"),
    template="plotly_white"
)

# Display in Streamlit
st.subheader("Vortex Indicator for TSLA - 2025")
st.plotly_chart(fig_tsla_2025, use_container_width=True)


# In[270]:


# Create the figure with a unique name
fig_spy_full = go.Figure()

# Add VI+ trace
fig_spy_full.add_trace(go.Scatter(
    x=spy.index,
    y=spy["VI+"],
    mode='lines',
    name='VI+',
    line=dict(color='blue')
))

# Add VI- trace
fig_spy_full.add_trace(go.Scatter(
    x=spy.index,
    y=spy["VI-"],
    mode='lines',
    name='VI-',
    line=dict(color='orange')
))

# Customize layout
fig_spy_full.update_layout(
    title="Vortex Indicator (VI+ and VI−) for SPY",
    xaxis_title="Date",
    yaxis_title="Value",
    legend=dict(x=0, y=1.1, orientation="h"),
    template="plotly_white"
)

# Display in Streamlit
st.subheader("Vortex Indicator for SPY - Full Period")
st.plotly_chart(fig_spy_full, use_container_width=True)


# In[271]:


# Filter SPY data for 2025
spy_2025 = spy.loc["2025"]

# Create the figure
fig_spy_2025 = go.Figure()

# Add VI+ trace
fig_spy_2025.add_trace(go.Scatter(
    x=spy_2025.index,
    y=spy_2025["VI+"],
    mode='lines',
    name='VI+',
    line=dict(color='blue')
))

# Add VI- trace
fig_spy_2025.add_trace(go.Scatter(
    x=spy_2025.index,
    y=spy_2025["VI-"],
    mode='lines',
    name='VI-',
    line=dict(color='orange')
))

# Update layout
fig_spy_2025.update_layout(
    title="Vortex Indicator (VI+ and VI−) for SPY – 2025",
    xaxis_title="Date",
    yaxis_title="Value",
    legend=dict(x=0, y=1.1, orientation="h"),
    template="plotly_white"
)

# Display in Streamlit
st.subheader("Vortex Indicator for SPY - 2025")
st.plotly_chart(fig_spy_2025, use_container_width=True)


# In[272]:


# Create the figure with a descriptive name
fig_xly_full = go.Figure()

# Add VI+ trace
fig_xly_full.add_trace(go.Scatter(
    x=xly.index,
    y=xly["VI+"],
    mode='lines',
    name='VI+',
    line=dict(color='blue')
))

# Add VI- trace
fig_xly_full.add_trace(go.Scatter(
    x=xly.index,
    y=xly["VI-"],
    mode='lines',
    name='VI-',
    line=dict(color='orange')
))

# Layout customization
fig_xly_full.update_layout(
    title="Vortex Indicator (VI+ and VI−) for XLY",
    xaxis_title="Date",
    yaxis_title="Value",
    legend=dict(x=0, y=1.1, orientation="h"),
    template="plotly_white"
)

# Render using Streamlit
st.subheader("Vortex Indicator for XLY - Full Period")
st.plotly_chart(fig_xly_full, use_container_width=True)


# In[273]:


# Filter the XLY data for 2025
xly_2025 = xly.loc["2025"]

# Create the figure with a unique name
fig_xly_2025 = go.Figure()

# Add VI+ line
fig_xly_2025.add_trace(go.Scatter(
    x=xly_2025.index,
    y=xly_2025["VI+"],
    mode='lines',
    name='VI+',
    line=dict(color='blue')
))

# Add VI- line
fig_xly_2025.add_trace(go.Scatter(
    x=xly_2025.index,
    y=xly_2025["VI-"],
    mode='lines',
    name='VI-',
    line=dict(color='orange')
))

# Layout styling
fig_xly_2025.update_layout(
    title="Vortex Indicator (VI+ and VI−) for XLY – 2025",
    xaxis_title="Date",
    yaxis_title="Value",
    legend=dict(x=0, y=1.1, orientation="h"),
    template="plotly_white"
)

# Display in Streamlit
st.subheader("Vortex Indicator for XLY – 2025")
st.plotly_chart(fig_xly_2025, use_container_width=True)


# In[274]:


from plotly.subplots import make_subplots

fig_2025 = make_subplots(
    rows=3, cols=1,
    subplot_titles=("SPY - 2025", "XLY - 2025", "TSLA - 2025")
)

fig_2025.add_trace(go.Scatter(
    x=spy_2025.index,
    y=spy_2025["VI+"],
    name="VI+ (SPY)",
    line=dict(color='blue'),
    showlegend=False
), row=1, col=1)

fig_2025.add_trace(go.Scatter(
    x=spy_2025.index,
    y=spy_2025["VI-"],
    name="VI- (SPY)",
    line=dict(color='orange'),
    showlegend=False
), row=1, col=1)

fig_2025.add_trace(go.Scatter(
    x=xly_2025.index,
    y=xly_2025["VI+"],
    name="VI+ (XLY)",
    line=dict(color='blue'),
    showlegend=False
), row=2, col=1)

fig_2025.add_trace(go.Scatter(
    x=xly_2025.index,
    y=xly_2025["VI-"],
    name="VI- (XLY)",
    line=dict(color='orange'),
    showlegend=False
), row=2, col=1)

fig_2025.add_trace(go.Scatter(
    x=tsla_2025.index,
    y=tsla_2025["VI+_"],
    name="VI+ (TSLA)",
    line=dict(color='blue'),
    showlegend=False
), row=3, col=1)

fig_2025.add_trace(go.Scatter(
    x=tsla_2025.index,
    y=tsla_2025["VI-_"],
    name="VI- (TSLA)",
    line=dict(color='orange'),
    showlegend=False
), row=3, col=1)

fig_2025.update_layout(
    height=500, width=1200,
    title_text="Vortex Indicator (VI+ and VI−) - 2025 Comparison",
    template="plotly_white"
)

st.plotly_chart(fig_2025)


# In[275]:


fig_full = make_subplots(
    rows=3, cols=1,
    subplot_titles=("SPY Year To Year", "XLY Year To Year", "TSLA Year To Year")
)

fig_full.add_trace(go.Scatter(
    x=spy.index,
    y=spy["VI+"],
    name="VI+ (SPY)",
    line=dict(color='blue'),
    showlegend=False
), row=1, col=1)

fig_full.add_trace(go.Scatter(
    x=spy.index,
    y=spy["VI-"],
    name="VI- (SPY)",
    line=dict(color='orange'),
    showlegend=False
), row=1, col=1)

fig_full.add_trace(go.Scatter(
    x=xly.index,
    y=xly["VI+"],
    name="VI+ (XLY)",
    line=dict(color='blue'),
    showlegend=False
), row=2, col=1)

fig_full.add_trace(go.Scatter(
    x=xly.index,
    y=xly["VI-"],
    name="VI- (XLY)",
    line=dict(color='orange'),
    showlegend=False
), row=2, col=1)

fig_full.add_trace(go.Scatter(
    x=tsla.index,
    y=tsla["VI+_"],
    name="VI+ (TSLA)",
    line=dict(color='blue'),
    showlegend=False
), row=3, col=1)

fig_full.add_trace(go.Scatter(
    x=tsla.index,
    y=tsla["VI-_"],
    name="VI- (TSLA)",
    line=dict(color='orange'),
    showlegend=False
), row=3, col=1)

fig_full.update_layout(
    height=900, width=1200,
    title_text="Vortex Indicator (VI+ and VI−) - Full Period Comparison",
    template="plotly_white"
)

st.plotly_chart(fig_full)


# ## Dashboard

# In[ ]:




