import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime, timedelta
import asyncio

class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerPricePredictor(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(TransformerPricePredictor, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        x = self.transformer(x)
        x = self.output_projection(x[:, -1, :])
        return x

class PricePredictionModel:
    def __init__(self, model_type='lstm', sequence_length=60, prediction_horizon=5):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_columns = [
            'price', 'volume', 'high', 'low', 'open', 'close',
            'rsi', 'macd', 'bollinger_upper', 'bollinger_lower',
            'volume_sma', 'price_sma_10', 'price_sma_50',
            'volatility', 'momentum', 'liquidity_depth'
        ]
        
    def prepare_features(self, df):
        df = df.copy()
        
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['momentum'] = df['price'] / df['price'].shift(10) - 1
        
        df['rsi'] = self.calculate_rsi(df['price'])
        df['macd'], df['macd_signal'] = self.calculate_macd(df['price'])
        
        df['bollinger_upper'], df['bollinger_lower'] = self.calculate_bollinger_bands(df['price'])
        
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['price_sma_10'] = df['price'].rolling(window=10).mean()
        df['price_sma_50'] = df['price'].rolling(window=50).mean()
        
        df['price_to_sma10_ratio'] = df['price'] / df['price_sma_10']
        df['price_to_sma50_ratio'] = df['price'] / df['price_sma_50']
        
        df['liquidity_depth'] = df.get('bid_depth', 0) + df.get('ask_depth', 0)
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def create_sequences(self, data, target_col='price'):
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            sequence = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon][target_col].values
            
            X.append(sequence[self.feature_columns].values)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_size):
        if self.model_type == 'lstm':
            model = LSTMPricePredictor(
                input_size=input_size,
                hidden_size=128,
                num_layers=3,
                output_size=self.prediction_horizon,
                dropout=0.3
            )
        elif self.model_type == 'transformer':
            model = TransformerPricePredictor(
                input_size=input_size,
                d_model=256,
                nhead=8,
                num_layers=6,
                output_size=self.prediction_horizon
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train(self, df, epochs=100, batch_size=32, learning_rate=0.001, validation_split=0.2):
        df_processed = self.prepare_features(df)
        df_processed = df_processed.dropna()
        
        scaled_data = self.scaler.fit_transform(df_processed[self.feature_columns])
        df_scaled = pd.DataFrame(scaled_data, columns=self.feature_columns, index=df_processed.index)
        df_scaled['price'] = df_processed['price']
        
        X, y = self.create_sequences(df_scaled)
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        self.model = self.build_model(len(self.feature_columns))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_price_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= 20:
                break
        
        self.model.load_state_dict(torch.load('best_price_model.pth'))
        
        return {
            'final_train_loss': total_loss / (len(X_train) // batch_size),
            'final_val_loss': val_loss,
            'epochs_trained': epoch + 1
        }
    
    def predict(self, df, steps_ahead=None):
        if steps_ahead is None:
            steps_ahead = self.prediction_horizon
        
        df_processed = self.prepare_features(df)
        df_processed = df_processed.dropna()
        
        if len(df_processed) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points")
        
        scaled_data = self.scaler.transform(df_processed[self.feature_columns])
        
        last_sequence = scaled_data[-self.sequence_length:]
        last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(last_sequence)
            prediction = prediction.cpu().numpy().flatten()
        
        return prediction[:steps_ahead]
    
    def predict_with_confidence(self, df, num_samples=100):
        predictions = []
        
        for _ in range(num_samples):
            pred = self.predict(df)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        confidence_intervals = {
            'lower_95': mean_pred - 1.96 * std_pred,
            'upper_95': mean_pred + 1.96 * std_pred,
            'lower_68': mean_pred - std_pred,
            'upper_68': mean_pred + std_pred
        }
        
        return mean_pred, std_pred, confidence_intervals
    
    def evaluate(self, df):
        df_processed = self.prepare_features(df)
        df_processed = df_processed.dropna()
        
        scaled_data = self.scaler.transform(df_processed[self.feature_columns])
        df_scaled = pd.DataFrame(scaled_data, columns=self.feature_columns, index=df_processed.index)
        df_scaled['price'] = df_processed['price']
        
        X, y_true = self.create_sequences(df_scaled)
        X = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), 32):
                batch_X = X[i:i + 32]
                batch_pred = self.model(batch_X)
                predictions.extend(batch_pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        mse = mean_squared_error(y_true.flatten(), predictions.flatten())
        mae = mean_absolute_error(y_true.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        
        directional_accuracy = np.mean(
            np.sign(y_true[:, 0] - df_processed['price'].iloc[-len(y_true):].values) == 
            np.sign(predictions[:, 0] - df_processed['price'].iloc[-len(y_true):].values)
        )
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.scaler = checkpoint['scaler']
        self.feature_columns = checkpoint['feature_columns']
        self.model_type = checkpoint['model_type']
        self.sequence_length = checkpoint['sequence_length']
        self.prediction_horizon = checkpoint['prediction_horizon']
        
        self.model = self.build_model(len(self.feature_columns))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

class EnsemblePricePredictor:
    def __init__(self, models=None):
        self.models = models or []
        self.weights = None
    
    def add_model(self, model):
        self.models.append(model)
    
    def fit_weights(self, validation_data):
        predictions = []
        for model in self.models:
            pred = model.predict(validation_data)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        from scipy.optimize import minimize
        
        def objective(weights):
            weighted_pred = np.average(predictions, axis=0, weights=weights)
            return mean_squared_error(validation_data['price'].values[-len(weighted_pred):], weighted_pred)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(objective, np.ones(len(self.models)) / len(self.models), 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.weights = result.x
    
    def predict(self, df):
        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        predictions = []
        for model in self.models:
            pred = model.predict(df)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return np.average(predictions, axis=0, weights=self.weights)