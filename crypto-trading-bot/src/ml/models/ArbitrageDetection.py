import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import networkx as nx
from scipy.sparse import csr_matrix
from collections import defaultdict
import asyncio

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GraphNeuralNetwork, self).__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, adj_matrix):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.matmul(adj_matrix, x)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x)
        return x

class ArbitrageOpportunityDetector:
    def __init__(self, detection_threshold=0.0001, min_profit_bps=5):
        self.detection_threshold = detection_threshold
        self.min_profit_bps = min_profit_bps
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gnn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def extract_market_features(self, pools_data, prices_data):
        features = []
        
        for pool_id, pool in pools_data.items():
            pool_features = {
                'pool_id': pool_id,
                'reserve0': pool.get('reserve0', 0),
                'reserve1': pool.get('reserve1', 0),
                'fee': pool.get('fee', 0),
                'total_supply': pool.get('total_supply', 0),
                'token0_address': pool.get('token0', ''),
                'token1_address': pool.get('token1', ''),
                'pool_type': pool.get('type', ''),
                'chain': pool.get('chain', ''),
                'last_update': pool.get('timestamp', 0)
            }
            
            if pool['reserve0'] > 0 and pool['reserve1'] > 0:
                pool_features['price_ratio'] = pool['reserve1'] / pool['reserve0']
                pool_features['liquidity_usd'] = self.estimate_liquidity_usd(pool, prices_data)
                pool_features['utilization'] = self.calculate_utilization(pool)
                pool_features['volume_24h'] = pool.get('volume_24h', 0)
                pool_features['fees_24h'] = pool.get('fees_24h', 0)
                
                if pool_features['volume_24h'] > 0:
                    pool_features['fee_apr'] = (pool_features['fees_24h'] * 365) / pool_features['liquidity_usd']
                else:
                    pool_features['fee_apr'] = 0
                
                pool_features['price_impact_1k'] = self.estimate_price_impact(pool, 1000)
                pool_features['price_impact_10k'] = self.estimate_price_impact(pool, 10000)
            
            features.append(pool_features)
        
        return pd.DataFrame(features)
    
    def estimate_liquidity_usd(self, pool, prices):
        token0_price = prices.get(pool.get('token0', ''), 0)
        token1_price = prices.get(pool.get('token1', ''), 0)
        
        reserve0_usd = pool.get('reserve0', 0) * token0_price / 1e18
        reserve1_usd = pool.get('reserve1', 0) * token1_price / 1e18
        
        return reserve0_usd + reserve1_usd
    
    def calculate_utilization(self, pool):
        if pool.get('total_supply', 0) == 0:
            return 0
        
        return (pool.get('reserve0', 0) + pool.get('reserve1', 0)) / pool.get('total_supply', 1)
    
    def estimate_price_impact(self, pool, amount_usd):
        reserve0 = pool.get('reserve0', 0)
        reserve1 = pool.get('reserve1', 0)
        
        if reserve0 == 0 or reserve1 == 0:
            return 1.0
        
        amount_in = amount_usd * 1e18
        amount_in_with_fee = amount_in * 997
        
        amount_out = (amount_in_with_fee * reserve1) / (reserve0 * 1000 + amount_in_with_fee)
        
        old_price = reserve1 / reserve0
        new_reserve0 = reserve0 + amount_in
        new_reserve1 = reserve1 - amount_out
        new_price = new_reserve1 / new_reserve0
        
        price_impact = abs(new_price - old_price) / old_price
        return min(price_impact, 1.0)
    
    def detect_triangular_arbitrage(self, pools_df):
        arbitrage_opportunities = []
        
        token_pools = defaultdict(list)
        for _, pool in pools_df.iterrows():
            token0 = pool['token0_address']
            token1 = pool['token1_address']
            token_pools[token0].append(pool)
            token_pools[token1].append(pool)
        
        tokens = list(token_pools.keys())
        
        for i, token_a in enumerate(tokens):
            for j, token_b in enumerate(tokens[i+1:], i+1):
                for k, token_c in enumerate(tokens[j+1:], j+1):
                    
                    pools_ab = self.find_pools_for_pair(token_a, token_b, pools_df)
                    pools_bc = self.find_pools_for_pair(token_b, token_c, pools_df)
                    pools_ca = self.find_pools_for_pair(token_c, token_a, pools_df)
                    
                    if pools_ab and pools_bc and pools_ca:
                        for pool_ab in pools_ab:
                            for pool_bc in pools_bc:
                                for pool_ca in pools_ca:
                                    opportunity = self.calculate_triangular_profit(
                                        pool_ab, pool_bc, pool_ca, token_a, token_b, token_c
                                    )
                                    
                                    if opportunity['profit_bps'] > self.min_profit_bps:
                                        arbitrage_opportunities.append(opportunity)
        
        return arbitrage_opportunities
    
    def find_pools_for_pair(self, token_a, token_b, pools_df):
        return pools_df[
            ((pools_df['token0_address'] == token_a) & (pools_df['token1_address'] == token_b)) |
            ((pools_df['token0_address'] == token_b) & (pools_df['token1_address'] == token_a))
        ].to_dict('records')
    
    def calculate_triangular_profit(self, pool_ab, pool_bc, pool_ca, token_a, token_b, token_c):
        try:
            amount_start = 1e18
            
            amount_b = self.get_amount_out(amount_start, pool_ab, token_a, token_b)
            amount_c = self.get_amount_out(amount_b, pool_bc, token_b, token_c)
            amount_end = self.get_amount_out(amount_c, pool_ca, token_c, token_a)
            
            profit = amount_end - amount_start
            profit_bps = (profit / amount_start) * 10000
            
            total_gas = 300000
            gas_cost_bps = (total_gas * 20e9 * 3000 / 1e18) / (amount_start / 1e18) * 10000
            
            net_profit_bps = profit_bps - gas_cost_bps
            
            return {
                'tokens': [token_a, token_b, token_c],
                'pools': [pool_ab['pool_id'], pool_bc['pool_id'], pool_ca['pool_id']],
                'profit_bps': profit_bps,
                'net_profit_bps': net_profit_bps,
                'gas_cost_bps': gas_cost_bps,
                'amount_start': amount_start,
                'amount_end': amount_end,
                'confidence': self.calculate_confidence(pool_ab, pool_bc, pool_ca)
            }
        except:
            return {'profit_bps': 0, 'net_profit_bps': 0}
    
    def get_amount_out(self, amount_in, pool, token_in, token_out):
        if pool['token0_address'] == token_in:
            reserve_in = pool['reserve0']
            reserve_out = pool['reserve1']
        else:
            reserve_in = pool['reserve1']
            reserve_out = pool['reserve0']
        
        if reserve_in == 0 or reserve_out == 0:
            return 0
        
        fee = pool.get('fee', 30)
        amount_in_with_fee = amount_in * (10000 - fee)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in * 10000 + amount_in_with_fee
        
        return numerator / denominator
    
    def calculate_confidence(self, pool_ab, pool_bc, pool_ca):
        min_liquidity = min(
            pool_ab.get('liquidity_usd', 0),
            pool_bc.get('liquidity_usd', 0),
            pool_ca.get('liquidity_usd', 0)
        )
        
        max_impact = max(
            pool_ab.get('price_impact_1k', 1),
            pool_bc.get('price_impact_1k', 1),
            pool_ca.get('price_impact_1k', 1)
        )
        
        liquidity_score = min(min_liquidity / 100000, 1.0)
        impact_score = max(0, 1 - max_impact * 10)
        
        return (liquidity_score + impact_score) / 2
    
    def detect_cross_dex_arbitrage(self, pools_df):
        arbitrage_opportunities = []
        
        token_pairs = defaultdict(list)
        for _, pool in pools_df.iterrows():
            pair = tuple(sorted([pool['token0_address'], pool['token1_address']]))
            token_pairs[pair].append(pool)
        
        for pair, pools in token_pairs.items():
            if len(pools) < 2:
                continue
            
            for i in range(len(pools)):
                for j in range(i + 1, len(pools)):
                    pool1 = pools[i]
                    pool2 = pools[j]
                    
                    if pool1['pool_type'] != pool2['pool_type'] or pool1['chain'] != pool2['chain']:
                        opportunity = self.calculate_cross_dex_profit(pool1, pool2, pair)
                        
                        if opportunity['profit_bps'] > self.min_profit_bps:
                            arbitrage_opportunities.append(opportunity)
        
        return arbitrage_opportunities
    
    def calculate_cross_dex_profit(self, pool1, pool2, token_pair):
        try:
            amount_in = 1e18
            
            price1 = pool1['reserve1'] / pool1['reserve0']
            price2 = pool2['reserve1'] / pool2['reserve0']
            
            if price1 > price2:
                buy_pool = pool2
                sell_pool = pool1
                price_diff = (price1 - price2) / price2
            else:
                buy_pool = pool1
                sell_pool = pool2
                price_diff = (price2 - price1) / price1
            
            amount_out_buy = self.get_amount_out(amount_in, buy_pool, token_pair[0], token_pair[1])
            amount_out_sell = self.get_amount_out(amount_out_buy, sell_pool, token_pair[1], token_pair[0])
            
            profit = amount_out_sell - amount_in
            profit_bps = (profit / amount_in) * 10000
            
            gas_cost_bps = 200
            net_profit_bps = profit_bps - gas_cost_bps
            
            return {
                'token_pair': token_pair,
                'buy_pool': buy_pool['pool_id'],
                'sell_pool': sell_pool['pool_id'],
                'profit_bps': profit_bps,
                'net_profit_bps': net_profit_bps,
                'price_diff_pct': price_diff * 100,
                'confidence': min(
                    buy_pool.get('liquidity_usd', 0) / 50000,
                    sell_pool.get('liquidity_usd', 0) / 50000,
                    1.0
                )
            }
        except:
            return {'profit_bps': 0, 'net_profit_bps': 0}
    
    def build_market_graph(self, pools_df):
        G = nx.Graph()
        
        for _, pool in pools_df.iterrows():
            token0 = pool['token0_address']
            token1 = pool['token1_address']
            
            if not G.has_edge(token0, token1):
                G.add_edge(token0, token1, pools=[])
            
            G[token0][token1]['pools'].append({
                'pool_id': pool['pool_id'],
                'reserve0': pool['reserve0'],
                'reserve1': pool['reserve1'],
                'fee': pool['fee'],
                'liquidity_usd': pool.get('liquidity_usd', 0)
            })
        
        return G
    
    def find_arbitrage_paths(self, graph, max_hops=4):
        arbitrage_paths = []
        nodes = list(graph.nodes())
        
        for start_node in nodes:
            paths = self.dfs_arbitrage_paths(graph, start_node, start_node, [], max_hops)
            arbitrage_paths.extend(paths)
        
        return arbitrage_paths
    
    def dfs_arbitrage_paths(self, graph, current, target, path, max_hops):
        if len(path) > max_hops:
            return []
        
        if len(path) >= 2 and current == target:
            return [path.copy()]
        
        paths = []
        for neighbor in graph.neighbors(current):
            if neighbor not in path or (neighbor == target and len(path) >= 2):
                path.append(neighbor)
                paths.extend(self.dfs_arbitrage_paths(graph, neighbor, target, path, max_hops))
                path.pop()
        
        return paths
    
    def train_anomaly_detection(self, historical_features):
        X = historical_features.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        self.anomaly_detector.fit(X_scaled)
        
        return self.anomaly_detector
    
    def detect_anomalies(self, current_features):
        X = current_features.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = self.scaler.transform(X)
        
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        anomalies = self.anomaly_detector.predict(X_scaled)
        
        anomaly_indices = np.where(anomalies == -1)[0]
        
        return {
            'anomaly_indices': anomaly_indices,
            'anomaly_scores': anomaly_scores,
            'anomalous_pools': current_features.iloc[anomaly_indices]
        }
    
    def train_gnn_detector(self, graph_data, labels, epochs=100):
        features = []
        edges = []
        node_map = {}
        
        for i, node in enumerate(graph_data.nodes()):
            node_map[node] = i
            node_features = self.extract_node_features(graph_data, node)
            features.append(node_features)
        
        for edge in graph_data.edges():
            edges.append([node_map[edge[0]], node_map[edge[1]]])
            edges.append([node_map[edge[1]], node_map[edge[0]]])
        
        features = torch.FloatTensor(features).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        
        num_nodes = len(features)
        adj_matrix = torch.zeros(num_nodes, num_nodes).to(self.device)
        
        for edge in edges:
            adj_matrix[edge[0], edge[1]] = 1.0
        
        adj_matrix = adj_matrix + torch.eye(num_nodes).to(self.device)
        degree = torch.sum(adj_matrix, dim=1, keepdim=True)
        adj_matrix = adj_matrix / degree
        
        self.gnn_model = GraphNeuralNetwork(
            input_dim=features.shape[1],
            hidden_dim=64,
            output_dim=2
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.gnn_model.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            self.gnn_model.train()
            optimizer.zero_grad()
            
            outputs = self.gnn_model(features, adj_matrix)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
    
    def extract_node_features(self, graph, node):
        neighbors = list(graph.neighbors(node))
        degree = len(neighbors)
        
        total_liquidity = 0
        avg_fee = 0
        pool_count = 0
        
        for neighbor in neighbors:
            edge_data = graph[node][neighbor]
            for pool in edge_data.get('pools', []):
                total_liquidity += pool.get('liquidity_usd', 0)
                avg_fee += pool.get('fee', 0)
                pool_count += 1
        
        if pool_count > 0:
            avg_fee /= pool_count
        
        clustering_coeff = nx.clustering(graph, node) if degree > 1 else 0
        
        return [degree, total_liquidity, avg_fee, clustering_coeff, pool_count]
    
    def scan_for_opportunities(self, pools_data, prices_data):
        pools_df = self.extract_market_features(pools_data, prices_data)
        
        triangular_opportunities = self.detect_triangular_arbitrage(pools_df)
        cross_dex_opportunities = self.detect_cross_dex_arbitrage(pools_df)
        
        anomalies = self.detect_anomalies(pools_df)
        
        all_opportunities = triangular_opportunities + cross_dex_opportunities
        
        all_opportunities.sort(key=lambda x: x.get('net_profit_bps', 0), reverse=True)
        
        return {
            'triangular_arbitrage': triangular_opportunities,
            'cross_dex_arbitrage': cross_dex_opportunities,
            'anomalies': anomalies,
            'top_opportunities': all_opportunities[:10],
            'total_opportunities': len(all_opportunities),
            'scan_timestamp': pd.Timestamp.now()
        }