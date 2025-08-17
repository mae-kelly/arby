import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import lightning as L
from typing import Dict, List, Tuple
import time
from collections import deque
import pickle
import joblib
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import optuna

try:
    import cupy as cp
    import rapids.ai as rapids
    from cuml.ensemble import RandomForestRegressor as cuRF
    RAPIDS_AVAILABLE = True
except:
    RAPIDS_AVAILABLE = False

class AttentionPricePredictor(nn.Module):
    """Transformer-based price movement predictor"""
    
    def __init__(self, input_dim=128, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # [probability, direction, magnitude]
        )
        
    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global pooling
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
            
        output = self.output_layers(x)
        
        return {
            'probability': torch.sigmoid(output[:, 0]),
            'direction': torch.tanh(output[:, 1]),
            'magnitude': torch.relu(output[:, 2])
        }

class LiquidationPredictor(nn.Module):
    """Predict upcoming liquidations"""
    
    def __init__(self, num_features=50):
        super().__init__()
        
        self.lstm = nn.LSTM(
            num_features,
            256,
            num_layers=3,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # [liquidation_prob, estimated_size]
        )
        
    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Pool and classify
        pooled = attn_out.mean(dim=1)
        output = self.classifier(pooled)
        
        return {
            'liquidation_prob': torch.sigmoid(output[:, 0]),
            'estimated_size': torch.exp(output[:, 1])  # Log-normal size
        }

class MEVPredictor:
    """Predict MEV opportunities using ensemble methods"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.price_model = AttentionPricePredictor().to(device)
        self.liquidation_model = LiquidationPredictor().to(device)
        
        # XGBoost for arbitrage prediction
        self.xgb_model = None
        self.init_xgboost()
        
        # Feature memory
        self.feature_buffer = deque(maxlen=10000)
        self.prediction_cache = {}
        
        # Training setup
        self.scaler = GradScaler()
        self.optimizer_price = torch.optim.AdamW(self.price_model.parameters(), lr=1e-4)
        self.optimizer_liq = torch.optim.AdamW(self.liquidation_model.parameters(), lr=1e-4)
        
    def init_xgboost(self):
        """Initialize XGBoost with GPU support"""
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 10,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
            'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor',
            'gpu_id': 0,
            'n_jobs': -1
        }
        self.xgb_model = xgb.XGBRegressor(**params)
        
    def extract_features(self, market_data: Dict) -> np.ndarray:
        """Extract features from market data"""
        features = []
        
        # Price features
        if 'orderbook' in market_data:
            ob = market_data['orderbook']
            features.extend([
                ob['bid_ask_spread'],
                ob['mid_price'],
                ob['bid_volume'],
                ob['ask_volume'],
                ob['volume_imbalance'],
                np.log1p(ob['total_volume'])
            ])
            
        # Technical indicators
        if 'prices' in market_data:
            prices = np.array(market_data['prices'])
            features.extend([
                self.calculate_rsi(prices),
                self.calculate_volatility(prices),
                self.calculate_momentum(prices),
                self.calculate_vwap(prices, market_data.get('volumes', []))
            ])
            
        # Network features
        if 'gas_price' in market_data:
            features.append(np.log1p(market_data['gas_price']))
            
        if 'mempool_size' in market_data:
            features.append(np.log1p(market_data['mempool_size']))
            
        # Cross-exchange features
        if 'exchange_spreads' in market_data:
            spreads = market_data['exchange_spreads']
            features.extend([
                np.mean(spreads),
                np.std(spreads),
                np.max(spreads),
                np.min(spreads)
            ])
            
        return np.array(features, dtype=np.float32)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_volatility(self, prices, window=20):
        """Calculate volatility"""
        if len(prices) < 2:
            return 0.0
            
        returns = np.diff(np.log(prices + 1e-8))
        return np.std(returns[-window:]) * np.sqrt(252)
    
    def calculate_momentum(self, prices, period=10):
        """Calculate momentum"""
        if len(prices) < period:
            return 0.0
            
        return (prices[-1] - prices[-period]) / prices[-period]
    
    def calculate_vwap(self, prices, volumes):
        """Calculate VWAP"""
        if not volumes or len(volumes) != len(prices):
            return np.mean(prices) if prices else 0.0
            
        return np.sum(prices * volumes) / np.sum(volumes)
    
    @torch.no_grad()
    def predict_opportunity(self, market_data: Dict) -> Dict:
        """Predict arbitrage opportunity"""
        
        # Extract features
        features = self.extract_features(market_data)
        
        # Check cache
        feature_hash = hash(features.tobytes())
        if feature_hash in self.prediction_cache:
            cached = self.prediction_cache[feature_hash]
            if time.time() - cached['timestamp'] < 1.0:  # 1 second cache
                return cached['prediction']
                
        predictions = {}
        
        # Price movement prediction
        if len(features) > 0:
            price_input = torch.tensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
            price_pred = self.price_model(price_input)
            
            predictions['price_movement'] = {
                'probability': price_pred['probability'].cpu().item(),
                'direction': price_pred['direction'].cpu().item(),
                'magnitude': price_pred['magnitude'].cpu().item()
            }
            
        # Liquidation prediction
        if 'lending_positions' in market_data:
            liq_features = self.extract_liquidation_features(market_data['lending_positions'])
            liq_input = torch.tensor(liq_features).unsqueeze(0).to(self.device)
            liq_pred = self.liquidation_model(liq_input)
            
            predictions['liquidation'] = {
                'probability': liq_pred['liquidation_prob'].cpu().item(),
                'estimated_size': liq_pred['estimated_size'].cpu().item()
            }
            
        # Arbitrage prediction using XGBoost
        if self.xgb_model is not None and len(self.feature_buffer) > 100:
            try:
                arb_features = features.reshape(1, -1)
                arb_prob = self.xgb_model.predict_proba(arb_features)[0, 1]
                predictions['arbitrage_probability'] = float(arb_prob)
            except:
                predictions['arbitrage_probability'] = 0.5
                
        # Calculate overall opportunity score
        score = self.calculate_opportunity_score(predictions)
        predictions['opportunity_score'] = score
        
        # Cache result
        self.prediction_cache[feature_hash] = {
            'prediction': predictions,
            'timestamp': time.time()
        }
        
        # Clean old cache entries
        if len(self.prediction_cache) > 1000:
            self.prediction_cache.clear()
            
        return predictions
    
    def extract_liquidation_features(self, positions: List) -> np.ndarray:
        """Extract features from lending positions"""
        if not positions:
            return np.zeros(50)
            
        features = []
        
        for pos in positions[:10]:  # Top 10 positions
            features.extend([
                pos.get('health_factor', 1.0),
                pos.get('collateral_value', 0),
                pos.get('debt_value', 0),
                pos.get('liquidation_threshold', 0),
                pos.get('time_since_update', 0)
            ])
            
        # Pad if needed
        while len(features) < 50:
            features.append(0)
            
        return np.array(features[:50], dtype=np.float32)
    
    def calculate_opportunity_score(self, predictions: Dict) -> float:
        """Calculate overall opportunity score"""
        score = 0.0
        
        if 'price_movement' in predictions:
            pm = predictions['price_movement']
            score += pm['probability'] * abs(pm['magnitude']) * 0.3
            
        if 'liquidation' in predictions:
            liq = predictions['liquidation']
            score += liq['probability'] * np.log1p(liq['estimated_size']) * 0.3
            
        if 'arbitrage_probability' in predictions:
            score += predictions['arbitrage_probability'] * 0.4
            
        return min(score, 1.0)
    
    def train_on_batch(self, batch_data: List[Dict], labels: Dict):
        """Train models on a batch of data"""
        
        # Prepare data
        features = [self.extract_features(d) for d in batch_data]
        features_tensor = torch.tensor(features).to(self.device)
        
        # Train price model
        if 'price_labels' in labels:
            self.optimizer_price.zero_grad()
            
            with autocast():
                price_pred = self.price_model(features_tensor.unsqueeze(1))
                price_labels = torch.tensor(labels['price_labels']).to(self.device)
                
                loss = F.mse_loss(price_pred['magnitude'], price_labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_price)
            
        # Train XGBoost incrementally
        if 'arbitrage_labels' in labels and len(self.feature_buffer) > 1000:
            X = np.array([f for f, _ in self.feature_buffer])
            y = np.array([l for _, l in self.feature_buffer])
            
            self.xgb_model.fit(X, y)
            
        self.scaler.update()
    
    def save_models(self, path: str):
        """Save all models"""
        torch.save({
            'price_model': self.price_model.state_dict(),
            'liquidation_model': self.liquidation_model.state_dict(),
        }, f"{path}/torch_models.pt")
        
        if self.xgb_model:
            self.xgb_model.save_model(f"{path}/xgb_model.json")
            
    def load_models(self, path: str):
        """Load all models"""
        checkpoint = torch.load(f"{path}/torch_models.pt")
        self.price_model.load_state_dict(checkpoint['price_model'])
        self.liquidation_model.load_state_dict(checkpoint['liquidation_model'])
        
        if os.path.exists(f"{path}/xgb_model.json"):
            self.xgb_model.load_model(f"{path}/xgb_model.json")