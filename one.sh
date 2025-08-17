# 1. Set up environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install ccxt python-dotenv aiohttp numpy

# 3. Create .env file with your API keys
cat > .env << 'EOF'
# Exchange API Keys (get from exchange websites)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here

COINBASE_API_KEY=your_coinbase_key
COINBASE_SECRET=your_coinbase_secret
COINBASE_PASSPHRASE=your_coinbase_passphrase

KRAKEN_API_KEY=your_kraken_key
KRAKEN_SECRET=your_kraken_secret
EOF

# 4. Run the simple bot
python src/python/simple_bot.py