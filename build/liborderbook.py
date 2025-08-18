"""
Fallback Python implementation for M1 Mac
"""
class SimpleOrderbook:
    def __init__(self):
        self.bids = {}
        self.asks = {}
    
    def update(self, price, volume, is_bid):
        if is_bid:
            self.bids[price] = volume
        else:
            self.asks[price] = volume
    
    def get_best_bid(self):
        return max(self.bids.keys()) if self.bids else 0
    
    def get_best_ask(self):
        return min(self.asks.keys()) if self.asks else 0
