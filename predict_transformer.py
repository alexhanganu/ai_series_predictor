# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math

# --- 1. Data Generation and Preprocessing ---

def generate_stock_data(num_days=300):
    """
    Generates a synthetic time series dataset for five stocks.
    The primary stock's price (Stock_A) is influenced by others.
    
    Args:
        num_days (int): The number of days for which to generate data.
    
    Returns:
        pd.DataFrame: A DataFrame with the daily open prices of five stocks.
    """
    np.random.seed(42)
    
    # Base price for five stocks
    prices = np.zeros((num_days, 5))
    
    # Starting prices
    start_prices = np.random.uniform(50, 150, 5)
    prices[0, :] = start_prices
    
    # Generate prices with some random walk and dependencies
    for i in range(1, num_days):
        # A simple random walk for each stock
        prices[i, :] = prices[i - 1, :] + np.random.normal(0, 1, 5)
        
        # Introduce some correlation
        # Stock_A is correlated with others (Stock_B, Stock_C)
        prices[i, 0] += 0.5 * (prices[i, 1] - prices[i-1, 1]) + 0.3 * (prices[i, 2] - prices[i-1, 2])
        
        # Ensure prices don't go negative
        prices[i, :] = np.maximum(prices[i, :], 1.0)
    
    df = pd.DataFrame(prices, columns=['Stock_A', 'Stock_B', 'Stock_C', 'Stock_D', 'Stock_E'])
    return df

def create_sequences(data, sequence_length):
    """
    Creates input sequences and target labels from a time series dataset.
    
    Args:
        data (np.ndarray): The time series data.
        sequence_length (int): The length of the input sequence.
    
    Returns:
        tuple: Tensors for input sequences (X) and target values (y).
    """
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])  # Target is the next day's price of Stock_A
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- 2. Transformer Model Definition ---

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input sequence to help the model
    understand the order of the data points.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    """
    A simplified Transformer-based model for time series forecasting.
    Uses an encoder-only architecture.
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        # Linear layer to project the input features to the model dimension
        self.linear_in = nn.Linear(input_dim, d_model)
        
        # Positional encoding to capture the order of the time steps
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output linear layer to get the final prediction
        self.linear_out = nn.Linear(d_model, 1)

    def forward(self, src):
        # Apply the input linear projection
        src = self.linear_in(src)
        
        # Apply positional encoding
        src = self.pos_encoder(src)
        
        # Pass the data through the transformer encoder
        output = self.transformer_encoder(src)
        
        # Take the output of the last time step for prediction
        output = output[:, -1, :]
        
        # Pass through the output linear layer
        output = self.linear_out(output)
        
        return output

# --- 3. Main Execution Block ---

if __name__ == "__main__":
    # Define model parameters
    SEQUENCE_LENGTH = 15
    PREDICTION_HORIZON = 1  # We predict one step into the future
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # (1) Generate and load data
    print("Generating synthetic stock price data...")
    df = generate_stock_data()
    
    # Split data into primary and other series
    primary_stock = 'Stock_A'
    other_stocks = [col for col in df.columns if col != primary_stock]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Convert data to sequences for the model
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    
    # Prepare data for training
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # (2) Evaluate correlations
    print("\nEvaluating correlations between stock prices:")
    correlation_matrix = df.corr()
    print(correlation_matrix)
    
    # Select correlations with the primary stock
    primary_correlations = correlation_matrix[primary_stock]
    print(f"\nCorrelations of '{primary_stock}' with other stocks:")
    print(primary_correlations.drop(primary_stock))
    
    # --- Model Training ---
    
    input_dim = scaled_data.shape[1]
    model = TimeSeriesTransformer(input_dim=input_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting model training...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}")
    
    print("Training complete.")
    
    # --- Prediction ---
    
    # Use the last SEQUENCE_LENGTH days from the dataset for prediction
    last_sequence = scaled_data[-SEQUENCE_LENGTH:]
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        scaled_prediction = model(last_sequence_tensor).squeeze().item()
    
    # Inverse transform the prediction to get the actual price
    # The scaler was fitted on a 5-column array, so we need to create a dummy array for inversion
    dummy_array = np.zeros((1, scaled_data.shape[1]))
    dummy_array[0, 0] = scaled_prediction
    actual_prediction = scaler.inverse_transform(dummy_array)[0, 0]
    
    # (3) Print the prediction
    print("\n--- Prediction Results ---")
    print(f"The predicted next price for {primary_stock} is: ${actual_prediction:.2f}")






# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import os
import sys

# --- 1. Data Loading and Preprocessing ---

def load_stock_data(stock_files, column_name="Open"):
    """
    Loads stock price data from a list of local CSV files.
    
    This function assumes that each CSV file contains a time series for a single
    stock and has a column with the specified name (e.g., "Open"). The data
    is merged into a single DataFrame based on the date/time index.
    
    Args:
        stock_files (dict): A dictionary mapping stock tickers to their filenames.
        column_name (str): The name of the column to read from each file.
    
    Returns:
        pd.DataFrame: A DataFrame with the specified stock price data.
    """
    df_list = []
    
    # Check for the existence of all files before proceeding
    for stock, filename in stock_files.items():
        if not os.path.exists(filename):
            print(f"Error: The file '{filename}' was not found.")
            sys.exit(1)

    print("Loading stock data from local CSV files...")
    for stock, filename in stock_files.items():
        try:
            # Read the CSV file, assuming the first column is the date/index
            temp_df = pd.read_csv(filename, index_col=0, parse_dates=True)
            
            # Select the specified column and rename it to the stock ticker
            series = temp_df[column_name].rename(stock)
            df_list.append(series)
            
        except KeyError:
            print(f"Error: The column '{column_name}' was not found in '{filename}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred while reading '{filename}': {e}")
            sys.exit(1)

    # Concatenate all Series into a single DataFrame
    # Using 'outer' join to keep all dates from all files
    df = pd.concat(df_list, axis=1, join='outer').ffill()
    
    # Drop any rows with NaN values if there are any gaps after forward-fill
    df.dropna(inplace=True)
    
    print("Data loaded successfully.")
    return df

def create_sequences(data, sequence_length):
    """
    Creates input sequences and target labels from a time series dataset.
    
    Args:
        data (np.ndarray): The time series data.
        sequence_length (int): The length of the input sequence.
    
    Returns:
        tuple: Tensors for input sequences (X) and target values (y).
    """
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        # Target is the next day's price of the first stock (column index 0)
        y.append(data[i + sequence_length, 0])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- 2. Transformer Model Definition ---

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input sequence to help the model
    understand the order of the data points.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    """
    A simplified Transformer-based model for time series forecasting.
    Uses an encoder-only architecture.
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        # Linear layer to project the input features to the model dimension
        self.linear_in = nn.Linear(input_dim, d_model)
        
        # Positional encoding to capture the order of the time steps
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output linear layer to get the final prediction
        self.linear_out = nn.Linear(d_model, 1)

    def forward(self, src):
        # Apply the input linear projection
        src = self.linear_in(src)
        
        # Apply positional encoding
        src = self.pos_encoder(src)
        
        # Pass the data through the transformer encoder
        output = self.transformer_encoder(src)
        
        # Take the output of the last time step for prediction
        output = output[:, -1, :]
        
        # Pass through the output linear layer
        output = self.linear_out(output)
        
        return output

# --- 3. Main Execution Block ---

if __name__ == "__main__":
    # --- Define model parameters and file paths ---
    SEQUENCE_LENGTH = 15
    PREDICTION_HORIZON = 1  # We predict one step into the future
    EPOCHS = 20
    LEARNING_RATE = 0.001

    # --- IMPORTANT: Replace these with your actual file names ---
    stock_files = {
        'Stock_A': 'stock_a.csv',
        'Stock_B': 'stock_b.csv',
        'Stock_C': 'stock_c.csv',
        'Stock_D': 'stock_d.csv',
        'Stock_E': 'stock_e.csv'
    }
    primary_stock = 'Stock_A'

    # (1) Load data from CSV files
    try:
        df = load_stock_data(stock_files)
    except SystemExit:
        # Exit gracefully if an error occurs during file loading
        sys.exit()

    # Get the list of other stocks from the loaded DataFrame columns
    other_stocks = [col for col in df.columns if col != primary_stock]

    # --- Data Preprocessing ---
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Convert data to sequences for the model
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    # Prepare data for training
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # (2) Evaluate correlations
    print("\nEvaluating correlations between stock prices:")
    correlation_matrix = df.corr()
    print(correlation_matrix)

    # Select correlations with the primary stock
    primary_correlations = correlation_matrix[primary_stock]
    print(f"\nCorrelations of '{primary_stock}' with other stocks:")
    print(primary_correlations.drop(primary_stock))

    # --- Model Training ---
    input_dim = scaled_data.shape[1]
    model = TimeSeriesTransformer(input_dim=input_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting model training...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}")
    
    print("Training complete.")

    # --- Prediction ---
    
    # Use the last SEQUENCE_LENGTH days from the dataset for prediction
    last_sequence = scaled_data[-SEQUENCE_LENGTH:]
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        scaled_prediction = model(last_sequence_tensor).squeeze().item()
    
    # Inverse transform the prediction to get the actual price
    dummy_array = np.zeros((1, scaled_data.shape[1]))
    dummy_array[0, 0] = scaled_prediction
    actual_prediction = scaler.inverse_transform(dummy_array)[0, 0]

    # (3) Print the prediction
    print("\n--- Prediction Results ---")
    print(f"The predicted next price for {primary_stock} is: ${actual_prediction:.2f}")
