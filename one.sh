#!/bin/bash

# Create the main project directory
mkdir -p crypto-trading-bot
cd crypto-trading-bot

# Create root level files
touch README.md
touch LICENSE
touch .gitignore
touch .env.example
touch package.json
touch requirements.txt
touch docker-compose.yml
touch Dockerfile

# Create config directory
mkdir -p config
touch config/default.json
touch config/production.json
touch config/development.json
touch config/exchanges.json
touch config/tokens.json
touch config/networks.json
touch config/strategies.json

# Create contracts directory
mkdir -p contracts/interfaces
mkdir -p contracts/libraries
touch contracts/FlashLoanArbitrage.sol
touch contracts/CrossDexArbitrage.sol
touch contracts/LiquidationBot.sol
touch contracts/interfaces/IFlashLoanReceiver.sol
touch contracts/interfaces/IAave.sol
touch contracts/interfaces/IUniswapV3.sol
touch contracts/interfaces/IBalancer.sol
touch contracts/libraries/SafeMath.sol
touch contracts/libraries/TransferHelper.sol

# Create src/core directory
mkdir -p src/core
touch src/core/bot.js
touch src/core/engine.js
touch src/core/executor.js
touch src/core/manager.js

# Create src/strategies directories
mkdir -p src/strategies/base
mkdir -p src/strategies/arbitrage
mkdir -p src/strategies/liquidation
mkdir -p src/strategies/mev
touch src/strategies/base/BaseStrategy.js
touch src/strategies/arbitrage/FlashLoanArbitrage.js
touch src/strategies/arbitrage/CrossExchangeArbitrage.js
touch src/strategies/arbitrage/TriangularArbitrage.js
touch src/strategies/arbitrage/CrossChainArbitrage.js
touch src/strategies/arbitrage/StablecoinArbitrage.js
touch src/strategies/liquidation/AaveLiquidator.js
touch src/strategies/liquidation/CompoundLiquidator.js
touch src/strategies/liquidation/MorphoLiquidator.js
touch src/strategies/mev/BackrunStrategy.js
touch src/strategies/mev/ArbitrageBundle.js

# Create src/connectors directories
mkdir -p src/connectors/exchanges
mkdir -p src/connectors/dex
mkdir -p src/connectors/blockchain
touch src/connectors/exchanges/BaseExchange.js
touch src/connectors/exchanges/Binance.js
touch src/connectors/exchanges/Coinbase.js
touch src/connectors/exchanges/Kraken.js
touch src/connectors/exchanges/FTX.js
touch src/connectors/dex/UniswapV3.js
touch src/connectors/dex/SushiSwap.js
touch src/connectors/dex/Curve.js
touch src/connectors/dex/Balancer.js
touch src/connectors/blockchain/EthereumConnector.js
touch src/connectors/blockchain/BSCConnector.js
touch src/connectors/blockchain/PolygonConnector.js
touch src/connectors/blockchain/SolanaConnector.js

# Create src/services directory
mkdir -p src/services
touch src/services/PriceService.js
touch src/services/GasService.js
touch src/services/MempoolService.js
touch src/services/FlashbotService.js
touch src/services/OracleService.js
touch src/services/NotificationService.js
touch src/services/MonitoringService.js

# Create src/analysis directory
mkdir -p src/analysis
touch src/analysis/OpportunityScanner.js
touch src/analysis/ProfitCalculator.js
touch src/analysis/RiskAnalyzer.js
touch src/analysis/MarketAnalyzer.js
touch src/analysis/SlippageCalculator.js

# Create src/ml directories
mkdir -p src/ml/models
mkdir -p src/ml/training
mkdir -p src/ml/reinforcement
touch src/ml/models/PricePrediction.py
touch src/ml/models/ArbitrageDetection.py
touch src/ml/models/RiskAssessment.py
touch src/ml/training/DataCollector.py
touch src/ml/training/FeatureEngineering.py
touch src/ml/training/ModelTrainer.py
touch src/ml/reinforcement/TradingAgent.py
touch src/ml/reinforcement/Environment.py
touch src/ml/reinforcement/RewardFunction.py

# Create src/utils directory
mkdir -p src/utils
touch src/utils/Logger.js
touch src/utils/Database.js
touch src/utils/Encryption.js
touch src/utils/RateLimiter.js
touch src/utils/RetryHandler.js
touch src/utils/WebsocketManager.js
touch src/utils/Constants.js

# Create src/api directories
mkdir -p src/api/routes
mkdir -p src/api/middleware
touch src/api/server.js
touch src/api/routes/health.js
touch src/api/routes/strategies.js
touch src/api/routes/positions.js
touch src/api/routes/analytics.js
touch src/api/middleware/auth.js
touch src/api/middleware/errorHandler.js

# Create scripts directories
mkdir -p scripts/deploy
mkdir -p scripts/maintenance
mkdir -p scripts/analysis
touch scripts/deploy/deployContracts.js
touch scripts/deploy/verifyContracts.js
touch scripts/deploy/setupAccounts.js
touch scripts/maintenance/rebalance.js
touch scripts/maintenance/withdraw.js
touch scripts/maintenance/emergencyStop.js
touch scripts/analysis/backtest.js
touch scripts/analysis/simulate.js
touch scripts/analysis/generateReport.js

# Create test directories
mkdir -p test/unit/strategies
mkdir -p test/unit/connectors
mkdir -p test/unit/services
mkdir -p test/integration
mkdir -p test/e2e
mkdir -p test/fixtures
touch test/integration/exchanges.test.js
touch test/integration/flashloans.test.js
touch test/integration/liquidations.test.js
touch test/e2e/fullCycle.test.js
touch test/fixtures/mockData.json
touch test/fixtures/testAccounts.json

# Create monitoring directories
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/prometheus
mkdir -p monitoring/alerts
touch monitoring/grafana/dashboards/performance.json
touch monitoring/grafana/dashboards/profits.json
touch monitoring/grafana/dashboards/errors.json
touch monitoring/prometheus/prometheus.yml
touch monitoring/alerts/telegram.js
touch monitoring/alerts/discord.js
touch monitoring/alerts/email.js

# Create data directories
mkdir -p data/cache
mkdir -p data/logs/trades
mkdir -p data/logs/errors
mkdir -p data/logs/performance
mkdir -p data/db/migrations
mkdir -p data/db/seeds

# Create docs directories
mkdir -p docs/architecture/diagrams
touch docs/API.md
touch docs/STRATEGIES.md
touch docs/DEPLOYMENT.md
touch docs/SECURITY.md
touch docs/architecture/system-design.md
touch docs/architecture/data-flow.md

# Create infrastructure directories
mkdir -p infrastructure/terraform/modules/ec2
mkdir -p infrastructure/terraform/modules/rds
mkdir -p infrastructure/terraform/modules/lambda
mkdir -p infrastructure/kubernetes
mkdir -p infrastructure/ansible/playbooks
mkdir -p infrastructure/ansible/inventory
touch infrastructure/terraform/main.tf
touch infrastructure/terraform/variables.tf
touch infrastructure/kubernetes/deployment.yaml
touch infrastructure/kubernetes/service.yaml
touch infrastructure/kubernetes/configmap.yaml

# Create security directories
mkdir -p security/audit/reports
mkdir -p security/keys
mkdir -p security/vault
touch security/keys/.gitkeep
touch security/vault/config.hcl

# Create tools directories
mkdir -p tools/cli/commands
mkdir -p tools/simulator
mkdir -p tools/performance
touch tools/cli/index.js
touch tools/cli/commands/start.js
touch tools/cli/commands/stop.js
touch tools/cli/commands/status.js
touch tools/cli/commands/config.js
touch tools/simulator/index.html
touch tools/simulator/simulator.js
touch tools/simulator/visualizer.js
touch tools/performance/profiler.js
touch tools/performance/optimizer.js

echo "✅ Crypto trading bot directory structure created successfully!"
echo "📁 Project created in: $(pwd)"
echo ""
echo "Next steps:"
echo "1. Initialize git: git init"
echo "2. Install Node.js dependencies: npm install"
echo "3. Install Python dependencies: pip install -r requirements.txt"
echo "4. Configure your .env file with API keys"
echo "5. Review and update configuration files in /config"