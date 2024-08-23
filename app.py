import dash
import dash_core_components as dcc
from dash import html
from dash.dependencies import Input, Output
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import plotly.graph_objs as go
from flask_caching import Cache
from concurrent.futures import ThreadPoolExecutor

# Initialize the Dash app
app = dash.Dash(__name__)

# Initialize the caching
cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})

# Initialize the ThreadPoolExecutor for parallel processing
executor = ThreadPoolExecutor()

# Enable TensorFlow's XLA (Accelerated Linear Algebra)
tf.config.optimizer.set_jit(True)  # Enable XLA for TensorFlow

# Define the layout of the app
app.layout = html.Div(
    style={
        'background-image': 'url("/assets/background.jpg")',  # Path to your background image
        'background-size': 'cover',
        'background-position': 'center',
        'min-height': '100vh',  # Ensure the background covers the full height
        'padding': '20px'
    },
    children=[
        html.H1(
            "Stock Price Prediction",
            style={
                'textAlign': 'center',  # Center the text
                'color': 'red'  # Ensure the text color is white for visibility against the background
            }
        ),
        html.Div([
            html.Label("Select Stock Ticker Symbol:", style={'color': 'white'}),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[
                    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
                    {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
                    {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                    {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
                    {'label': 'Facebook (META)', 'value': 'META'},
                    {'label': 'State Bank of India (SBI)', 'value': 'SBIN.NS'},  # SBI ticker
                    {'label': 'Reliance Industries (RELIANCE)', 'value': 'RELIANCE.NS'}  # Another NSE example
                ],
                value='AAPL'  # Default value
            ),
        ]),
        html.Div([
            html.Label("Enter Prediction Days:", style={'color': 'white'}),
            dcc.Input(id='days-input', value=1, type='number'),
        ]),
        html.Button(id='predict-button', n_clicks=0, children='Predict'),
        dcc.Graph(id='stock-price-graph'),
        html.Div(id='accuracy-output', style={'color': 'white'})  # New Div to display accuracy with white text color
    ]
)

# Cache the stock data to avoid repeated fetches
@cache.memoize(timeout=60*60)  # Cache the result for 1 hour
def fetch_live_stock_data(ticker):
    # Fetch the most recent data, e.g., last 1 day with 1-minute interval
    data = yf.download(ticker, period='1d', interval='1m')
    return data

# Simplified GRU model for faster predictions
def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to create sequences for the GRU model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Function to fetch data, train the model, and predict future prices
def fetch_and_predict(ticker, days):
    data = fetch_live_stock_data(ticker)
    data = data[['Close']].dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    seq_length = 60
    close_prices = data['Close'].values
    X, y = create_sequences(close_prices, seq_length)
    
    # Split the data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create and train the model
    model = create_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=0)  # Reduced epochs for faster training
    
    # Make predictions on the test set
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate RMSE for accuracy
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    accuracy = 100 - (rmse / np.mean(y_test)) * 100
    
    # Predict future prices
    last_sequence = close_prices[-seq_length:]
    future_predictions = []
    for _ in range(days):
        last_sequence = last_sequence.reshape(1, seq_length, 1)
        next_price = model.predict(last_sequence)
        future_predictions.append(next_price[0, 0])
        last_sequence = np.append(last_sequence[0, 1:], next_price)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return data.index, scaler.inverse_transform(data['Close'].values.reshape(-1, 1)), predictions.flatten(), future_predictions.flatten(), accuracy

# Callback to update the graph based on user inputs
@app.callback(
    [Output('stock-price-graph', 'figure'),
     Output('accuracy-output', 'children')],
    [Input('predict-button', 'n_clicks')],
    [Input('ticker-dropdown', 'value'), Input('days-input', 'value')],
    prevent_initial_call=True
)
def update_graph(n_clicks, ticker, days):
    future = executor.submit(fetch_and_predict, ticker, days)
    dates, historical_prices, predictions, future_predictions, accuracy = future.result()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=historical_prices.flatten(), mode='lines', name='Historical Data'))
    
    test_index = dates[-len(predictions):]
    fig.add_trace(go.Scatter(x=test_index, y=predictions, mode='lines', name='Model Predictions'))
    
    future_index = pd.date_range(start=test_index[-1], periods=days+1, freq='B')[1:]
    fig.add_trace(go.Scatter(x=future_index, y=future_predictions, mode='lines', name='Future Predictions'))
    
    fig.update_layout(title=f'{ticker} Stock Price Prediction', xaxis_title='Date', yaxis_title='Stock Price')
    
    accuracy_output = f'Prediction Accuracy: {accuracy:.2f}%'
    
    return fig, accuracy_output

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
