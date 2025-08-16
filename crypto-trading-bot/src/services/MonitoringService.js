const WebSocket = require('ws');
const EventEmitter = require('events');

class MonitoringService extends EventEmitter {
   constructor(config) {
       super();
       this.config = config;
       this.metrics = new Map();
       this.alerts = [];
       this.wsServer = null;
       this.prometheusMetrics = new Map();
       this.lastHeartbeat = Date.now();
       this.healthStatus = 'starting';
       this.errorCounts = new Map();
       this.performanceMetrics = {
           latency: [],
           throughput: 0,
           errors: 0,
           uptime: Date.now()
       };
   }

   async start() {
       this.healthStatus = 'healthy';
       this.setupWebSocketServer();
       this.startMetricsCollection();
       this.startHealthChecks();
       this.emit('service_started');
   }

   setupWebSocketServer() {
       this.wsServer = new WebSocket.Server({ 
           port: this.config.monitoring.wsPort || 8080,
           perMessageDeflate: false
       });

       this.wsServer.on('connection', (ws) => {
           ws.on('message', (message) => {
               try {
                   const data = JSON.parse(message);
                   this.handleWebSocketMessage(ws, data);
               } catch (error) {
                   this.recordError('websocket_parse_error', error);
               }
           });

           ws.on('error', (error) => {
               this.recordError('websocket_connection_error', error);
           });

           this.sendInitialData(ws);
       });
   }

   handleWebSocketMessage(ws, data) {
       switch (data.type) {
           case 'subscribe':
               ws.subscriptions = data.channels || [];
               break;
           case 'get_metrics':
               this.sendMetrics(ws);
               break;
           case 'get_alerts':
               this.sendAlerts(ws);
               break;
           case 'clear_alerts':
               this.clearAlerts();
               break;
       }
   }

   sendInitialData(ws) {
       this.sendMetrics(ws);
       this.sendAlerts(ws);
       this.sendHealthStatus(ws);
   }

   sendMetrics(ws) {
       const metricsData = {
           type: 'metrics_update',
           timestamp: Date.now(),
           data: Object.fromEntries(this.metrics)
       };
       
       if (ws.readyState === WebSocket.OPEN) {
           ws.send(JSON.stringify(metricsData));
       }
   }

   sendAlerts(ws) {
       const alertsData = {
           type: 'alerts_update',
           timestamp: Date.now(),
           alerts: this.alerts
       };
       
       if (ws.readyState === WebSocket.OPEN) {
           ws.send(JSON.stringify(alertsData));
       }
   }

   sendHealthStatus(ws) {
       const healthData = {
           type: 'health_update',
           timestamp: Date.now(),
           status: this.healthStatus,
           uptime: Date.now() - this.performanceMetrics.uptime,
           lastHeartbeat: this.lastHeartbeat
       };
       
       if (ws.readyState === WebSocket.OPEN) {
           ws.send(JSON.stringify(healthData));
       }
   }

   broadcastToSubscribers(channel, data) {
       this.wsServer.clients.forEach((ws) => {
           if (ws.subscriptions && ws.subscriptions.includes(channel)) {
               if (ws.readyState === WebSocket.OPEN) {
                   ws.send(JSON.stringify({
                       type: 'channel_update',
                       channel: channel,
                       data: data,
                       timestamp: Date.now()
                   }));
               }
           }
       });
   }

   recordMetric(name, value, tags = {}) {
       const timestamp = Date.now();
       const metricKey = `${name}_${JSON.stringify(tags)}`;
       
       this.metrics.set(metricKey, {
           name,
           value,
           tags,
           timestamp
       });

       this.prometheusMetrics.set(name, {
           value,
           labels: tags,
           timestamp
       });

       this.broadcastToSubscribers('metrics', { name, value, tags, timestamp });
       this.emit('metric_recorded', { name, value, tags });
   }

   recordCounter(name, increment = 1, tags = {}) {
       const key = `${name}_${JSON.stringify(tags)}`;
       const current = this.metrics.get(key);
       const newValue = (current ? current.value : 0) + increment;
       this.recordMetric(name, newValue, tags);
   }

   recordGauge(name, value, tags = {}) {
       this.recordMetric(name, value, tags);
   }

   recordHistogram(name, value, tags = {}) {
       const key = `${name}_histogram_${JSON.stringify(tags)}`;
       const existing = this.metrics.get(key) || { values: [], count: 0, sum: 0 };
       
       existing.values.push(value);
       existing.count++;
       existing.sum += value;
       
       if (existing.values.length > 1000) {
           existing.values = existing.values.slice(-1000);
       }

       existing.avg = existing.sum / existing.count;
       existing.min = Math.min(...existing.values);
       existing.max = Math.max(...existing.values);
       
       const sorted = [...existing.values].sort((a, b) => a - b);
       existing.p50 = sorted[Math.floor(sorted.length * 0.5)];
       existing.p95 = sorted[Math.floor(sorted.length * 0.95)];
       existing.p99 = sorted[Math.floor(sorted.length * 0.99)];

       this.metrics.set(key, existing);
       this.broadcastToSubscribers('histogram', { name, stats: existing, tags });
   }

   recordLatency(operation, durationMs, tags = {}) {
       this.recordHistogram(`${operation}_latency_ms`, durationMs, tags);
       this.performanceMetrics.latency.push({
           operation,
           duration: durationMs,
           timestamp: Date.now(),
           tags
       });

       if (this.performanceMetrics.latency.length > 10000) {
           this.performanceMetrics.latency = this.performanceMetrics.latency.slice(-10000);
       }
   }

   recordError(type, error, context = {}) {
       const errorKey = `error_${type}`;
       const count = this.errorCounts.get(errorKey) || 0;
       this.errorCounts.set(errorKey, count + 1);

       this.recordCounter('errors_total', 1, { type });
       this.performanceMetrics.errors++;

       const alert = {
           id: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
           type: 'error',
           severity: this.getErrorSeverity(type),
           message: error.message || error.toString(),
           stack: error.stack,
           context,
           timestamp: Date.now(),
           acknowledged: false
       };

       this.alerts.push(alert);
       this.trimAlerts();
       
       this.broadcastToSubscribers('alerts', alert);
       this.emit('error_recorded', alert);

       if (alert.severity === 'critical') {
           this.healthStatus = 'degraded';
       }
   }

   getErrorSeverity(errorType) {
       const criticalErrors = ['flash_loan_failure', 'transaction_failure', 'fund_loss'];
       const warningErrors = ['rpc_timeout', 'gas_spike', 'slippage_high'];
       
       if (criticalErrors.includes(errorType)) return 'critical';
       if (warningErrors.includes(errorType)) return 'warning';
       return 'info';
   }

   createAlert(severity, message, context = {}) {
       const alert = {
           id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
           type: 'alert',
           severity,
           message,
           context,
           timestamp: Date.now(),
           acknowledged: false
       };

       this.alerts.push(alert);
       this.trimAlerts();
       
       this.broadcastToSubscribers('alerts', alert);
       this.emit('alert_created', alert);

       if (severity === 'critical') {
           this.healthStatus = 'critical';
       }
   }

   acknowledgeAlert(alertId) {
       const alert = this.alerts.find(a => a.id === alertId);
       if (alert) {
           alert.acknowledged = true;
           alert.acknowledgedAt = Date.now();
           this.broadcastToSubscribers('alerts', alert);
       }
   }

   clearAlerts() {
       this.alerts = [];
       this.broadcastToSubscribers('alerts', { type: 'cleared' });
       this.healthStatus = 'healthy';
   }

   trimAlerts() {
       if (this.alerts.length > 1000) {
           this.alerts = this.alerts.slice(-1000);
       }
   }

   startMetricsCollection() {
       setInterval(() => {
           this.collectSystemMetrics();
           this.cleanupOldMetrics();
       }, 5000);

       setInterval(() => {
           this.calculateThroughput();
       }, 1000);
   }

   collectSystemMetrics() {
       const memUsage = process.memoryUsage();
       this.recordGauge('memory_heap_used', memUsage.heapUsed);
       this.recordGauge('memory_heap_total', memUsage.heapTotal);
       this.recordGauge('memory_rss', memUsage.rss);
       this.recordGauge('memory_external', memUsage.external);

       this.recordGauge('uptime_seconds', process.uptime());
       this.recordGauge('websocket_connections', this.wsServer ? this.wsServer.clients.size : 0);
       this.recordGauge('alerts_count', this.alerts.length);
       this.recordGauge('metrics_count', this.metrics.size);
   }

   calculateThroughput() {
       const now = Date.now();
       const oneSecondAgo = now - 1000;
       
       const recentLatency = this.performanceMetrics.latency.filter(
           l => l.timestamp > oneSecondAgo
       );
       
       this.performanceMetrics.throughput = recentLatency.length;
       this.recordGauge('throughput_ops_per_second', this.performanceMetrics.throughput);
   }

   cleanupOldMetrics() {
       const cutoff = Date.now() - (24 * 60 * 60 * 1000);
       
       for (const [key, metric] of this.metrics.entries()) {
           if (metric.timestamp < cutoff) {
               this.metrics.delete(key);
           }
       }
   }

   startHealthChecks() {
       setInterval(() => {
           this.performHealthCheck();
       }, 10000);
   }

   performHealthCheck() {
       this.lastHeartbeat = Date.now();
       
       const checks = {
           memory: this.checkMemoryUsage(),
           errors: this.checkErrorRate(),
           throughput: this.checkThroughput(),
           websocket: this.checkWebSocketHealth()
       };

       const failedChecks = Object.entries(checks).filter(([_, status]) => !status);
       
       if (failedChecks.length === 0) {
           this.healthStatus = 'healthy';
       } else if (failedChecks.length <= 2) {
           this.healthStatus = 'degraded';
       } else {
           this.healthStatus = 'critical';
       }

       this.recordGauge('health_checks_passed', Object.values(checks).filter(Boolean).length);
       this.recordGauge('health_checks_total', Object.values(checks).length);

       this.wsServer.clients.forEach((ws) => {
           this.sendHealthStatus(ws);
       });
   }

   checkMemoryUsage() {
       const memUsage = process.memoryUsage();
       const heapUsedMB = memUsage.heapUsed / 1024 / 1024;
       return heapUsedMB < 512;
   }

   checkErrorRate() {
       const now = Date.now();
       const oneMinuteAgo = now - 60000;
       
       const recentErrors = this.performanceMetrics.latency.filter(
           l => l.timestamp > oneMinuteAgo && l.duration === -1
       );
       
       const recentOps = this.performanceMetrics.latency.filter(
           l => l.timestamp > oneMinuteAgo
       );

       if (recentOps.length === 0) return true;
       
       const errorRate = recentErrors.length / recentOps.length;
       return errorRate < 0.05;
   }

   checkThroughput() {
       return this.performanceMetrics.throughput >= 0;
   }

   checkWebSocketHealth() {
       return this.wsServer && this.wsServer.readyState === WebSocket.OPEN;
   }

   getMetrics() {
       return Object.fromEntries(this.metrics);
   }

   getPrometheusMetrics() {
       let output = '';
       
       for (const [name, metric] of this.prometheusMetrics.entries()) {
           const labels = Object.entries(metric.labels || {})
               .map(([k, v]) => `${k}="${v}"`)
               .join(',');
           
           const labelStr = labels ? `{${labels}}` : '';
           output += `${name}${labelStr} ${metric.value}\n`;
       }
       
       return output;
   }

   getAlerts() {
       return this.alerts;
   }

   getHealthStatus() {
       return {
           status: this.healthStatus,
           uptime: Date.now() - this.performanceMetrics.uptime,
           lastHeartbeat: this.lastHeartbeat,
           metrics: {
               throughput: this.performanceMetrics.throughput,
               errors: this.performanceMetrics.errors,
               memory: process.memoryUsage(),
               connections: this.wsServer ? this.wsServer.clients.size : 0
           }
       };
   }

   async stop() {
       if (this.wsServer) {
           this.wsServer.close();
       }
       this.healthStatus = 'stopped';
       this.emit('service_stopped');
   }
}

module.exports = MonitoringService;