const EventEmitter = require('events');
const Redis = require('redis');
const { performance } = require('perf_hooks');

class BotManager extends EventEmitter {
   constructor(config) {
       super();
       this.config = config;
       this.strategies = new Map();
       this.activePositions = new Map();
       this.executionQueue = [];
       this.isRunning = false;
       this.redis = Redis.createClient(config.redis);
       this.performanceMetrics = {
           totalTrades: 0,
           totalProfit: 0,
           successRate: 0,
           avgExecutionTime: 0
       };
       this.riskLimits = {
           maxPositionSize: config.risk.maxPositionSize,
           maxDailyLoss: config.risk.maxDailyLoss,
           maxConcurrentTrades: config.risk.maxConcurrentTrades
       };
       this.emergencyStop = false;
   }

   async initialize() {
       await this.redis.connect();
       await this.loadPersistedData();
       this.setupEventHandlers();
       this.startPerformanceMonitoring();
   }

   async loadPersistedData() {
       const positions = await this.redis.hgetall('active_positions');
       for (const [id, data] of Object.entries(positions)) {
           this.activePositions.set(id, JSON.parse(data));
       }
       
       const metrics = await this.redis.get('performance_metrics');
       if (metrics) {
           this.performanceMetrics = JSON.parse(metrics);
       }
   }

   registerStrategy(strategy) {
       this.strategies.set(strategy.name, strategy);
       strategy.on('opportunity', this.handleOpportunity.bind(this));
       strategy.on('position_update', this.updatePosition.bind(this));
       strategy.on('trade_complete', this.handleTradeComplete.bind(this));
       strategy.on('error', this.handleStrategyError.bind(this));
   }

   async start() {
       if (this.isRunning) return;
       
       this.isRunning = true;
       this.emergencyStop = false;
       
       for (const strategy of this.strategies.values()) {
           await strategy.start();
       }
       
       this.processExecutionQueue();
       this.emit('manager_started');
   }

   async stop() {
       this.isRunning = false;
       
       for (const strategy of this.strategies.values()) {
           await strategy.stop();
       }
       
       await this.closeAllPositions();
       this.emit('manager_stopped');
   }

   async handleOpportunity(opportunity) {
       if (!this.isRunning || this.emergencyStop) return;
       
       const startTime = performance.now();
       
       if (!this.validateOpportunity(opportunity)) {
           this.emit('opportunity_rejected', opportunity);
           return;
       }

       if (!this.checkRiskLimits(opportunity)) {
           this.emit('risk_limit_exceeded', opportunity);
           return;
       }

       const execution = {
           id: this.generateExecutionId(),
           opportunity,
           timestamp: Date.now(),
           priority: this.calculatePriority(opportunity),
           status: 'queued'
       };

       this.executionQueue.push(execution);
       this.executionQueue.sort((a, b) => b.priority - a.priority);
       
       await this.redis.lpush('execution_queue', JSON.stringify(execution));
       
       const processingTime = performance.now() - startTime;
       this.updateMetrics('opportunity_processing_time', processingTime);
       
       this.emit('opportunity_queued', execution);
   }

   validateOpportunity(opportunity) {
       if (!opportunity.strategy || !opportunity.profit || !opportunity.confidence) {
           return false;
       }
       
       if (opportunity.profit < this.config.minProfitThreshold) {
           return false;
       }
       
       if (opportunity.confidence < this.config.minConfidenceThreshold) {
           return false;
       }
       
       return true;
   }

   checkRiskLimits(opportunity) {
       if (this.activePositions.size >= this.riskLimits.maxConcurrentTrades) {
           return false;
       }
       
       if (opportunity.size > this.riskLimits.maxPositionSize) {
           return false;
       }
       
       const dailyPnL = this.calculateDailyPnL();
       if (dailyPnL < -this.riskLimits.maxDailyLoss) {
           return false;
       }
       
       return true;
   }

   calculatePriority(opportunity) {
       const profitScore = Math.min(opportunity.profit / 1000, 100);
       const confidenceScore = opportunity.confidence * 50;
       const timeScore = Math.max(0, 50 - (Date.now() - opportunity.timestamp) / 1000);
       
       return profitScore + confidenceScore + timeScore;
   }

   async processExecutionQueue() {
       while (this.isRunning) {
           if (this.executionQueue.length === 0) {
               await new Promise(resolve => setTimeout(resolve, 10));
               continue;
           }
           
           const execution = this.executionQueue.shift();
           await this.executeOpportunity(execution);
       }
   }

   async executeOpportunity(execution) {
       const startTime = performance.now();
       execution.status = 'executing';
       
       try {
           const strategy = this.strategies.get(execution.opportunity.strategy);
           if (!strategy) {
               throw new Error(`Strategy ${execution.opportunity.strategy} not found`);
           }
           
           const result = await strategy.execute(execution.opportunity);
           
           if (result.success) {
               const position = {
                   id: execution.id,
                   strategy: execution.opportunity.strategy,
                   entry: result.entry,
                   size: result.size,
                   timestamp: Date.now(),
                   status: 'open'
               };
               
               this.activePositions.set(execution.id, position);
               await this.redis.hset('active_positions', execution.id, JSON.stringify(position));
               
               execution.status = 'completed';
               execution.result = result;
               
               this.emit('trade_executed', execution);
           } else {
               execution.status = 'failed';
               execution.error = result.error;
               this.emit('trade_failed', execution);
           }
           
       } catch (error) {
           execution.status = 'error';
           execution.error = error.message;
           this.emit('execution_error', execution);
       }
       
       const executionTime = performance.now() - startTime;
       this.updateMetrics('execution_time', executionTime);
       
       await this.redis.lpush('execution_history', JSON.stringify(execution));
   }

   async updatePosition(positionId, update) {
       const position = this.activePositions.get(positionId);
       if (!position) return;
       
       Object.assign(position, update);
       await this.redis.hset('active_positions', positionId, JSON.stringify(position));
       
       this.emit('position_updated', position);
   }

   async handleTradeComplete(trade) {
       const position = this.activePositions.get(trade.positionId);
       if (!position) return;
       
       position.status = 'closed';
       position.exit = trade.exit;
       position.profit = trade.profit;
       position.closeTime = Date.now();
       
       this.activePositions.delete(trade.positionId);
       await this.redis.hdel('active_positions', trade.positionId);
       await this.redis.lpush('completed_trades', JSON.stringify(position));
       
       this.performanceMetrics.totalTrades++;
       this.performanceMetrics.totalProfit += trade.profit;
       this.performanceMetrics.successRate = this.calculateSuccessRate();
       
       await this.redis.set('performance_metrics', JSON.stringify(this.performanceMetrics));
       
       this.emit('trade_completed', position);
   }

   handleStrategyError(error) {
       if (error.severity === 'critical') {
           this.emergencyStop = true;
           this.emit('emergency_stop', error);
       } else {
           this.emit('strategy_error', error);
       }
   }

   async closeAllPositions() {
       const closePromises = [];
       
       for (const [positionId, position] of this.activePositions) {
           const strategy = this.strategies.get(position.strategy);
           if (strategy && typeof strategy.closePosition === 'function') {
               closePromises.push(strategy.closePosition(positionId));
           }
       }
       
       await Promise.allSettled(closePromises);
       this.activePositions.clear();
       await this.redis.del('active_positions');
   }

   calculateDailyPnL() {
       const today = new Date().toDateString();
       let dailyPnL = 0;
       
       for (const position of this.activePositions.values()) {
           const positionDate = new Date(position.timestamp).toDateString();
           if (positionDate === today && position.profit) {
               dailyPnL += position.profit;
           }
       }
       
       return dailyPnL;
   }

   calculateSuccessRate() {
       if (this.performanceMetrics.totalTrades === 0) return 0;
       return (this.performanceMetrics.totalProfit > 0 ? 1 : 0) * 100;
   }

   updateMetrics(metric, value) {
       switch (metric) {
           case 'execution_time':
               this.performanceMetrics.avgExecutionTime = 
                   (this.performanceMetrics.avgExecutionTime + value) / 2;
               break;
           case 'opportunity_processing_time':
               break;
       }
   }

   generateExecutionId() {
       return `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
   }

   setupEventHandlers() {
       process.on('SIGINT', () => this.stop());
       process.on('SIGTERM', () => this.stop());
       
       this.on('emergency_stop', async () => {
           await this.stop();
       });
   }

   startPerformanceMonitoring() {
       setInterval(() => {
           const stats = {
               activePositions: this.activePositions.size,
               queueLength: this.executionQueue.length,
               totalProfit: this.performanceMetrics.totalProfit,
               successRate: this.performanceMetrics.successRate,
               avgExecutionTime: this.performanceMetrics.avgExecutionTime,
               dailyPnL: this.calculateDailyPnL(),
               timestamp: Date.now()
           };
           
           this.emit('performance_update', stats);
           this.redis.lpush('performance_history', JSON.stringify(stats));
       }, 5000);
   }

   getStatus() {
       return {
           isRunning: this.isRunning,
           emergencyStop: this.emergencyStop,
           activeStrategies: Array.from(this.strategies.keys()),
           activePositions: this.activePositions.size,
           queueLength: this.executionQueue.length,
           metrics: this.performanceMetrics,
           dailyPnL: this.calculateDailyPnL()
       };
   }
}

module.exports = BotManager;