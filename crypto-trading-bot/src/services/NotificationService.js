const EventEmitter = require('events');
const nodemailer = require('nodemailer');
const https = require('https');

class NotificationService extends EventEmitter {
    constructor(config) {
        super();
        this.config = config;
        this.emailTransporter = null;
        this.notificationQueue = [];
        this.rateLimiters = new Map();
        this.templates = new Map();
        this.channels = {
            email: false,
            telegram: false,
            discord: false,
            slack: false,
            webhook: false
        };
        this.priorityQueues = {
            urgent: [],
            high: [],
            medium: [],
            low: []
        };
    }

    async initialize() {
        await this.setupEmailService();
        await this.setupTelegramBot();
        await this.setupDiscordWebhook();
        await this.setupSlackWebhook();
        await this.loadNotificationTemplates();
        
        this.startQueueProcessor();
        this.emit('initialized');
    }

    async setupEmailService() {
        if (!this.config.email?.enabled) return;

        try {
            this.emailTransporter = nodemailer.createTransporter({
                host: this.config.email.smtp.host,
                port: this.config.email.smtp.port,
                secure: this.config.email.smtp.secure,
                auth: {
                    user: this.config.email.smtp.user,
                    pass: this.config.email.smtp.password
                }
            });

            await this.emailTransporter.verify();
            this.channels.email = true;
            this.emit('channelEnabled', 'email');
        } catch (error) {
            this.emit('error', `Failed to setup email service: ${error.message}`);
        }
    }

    async setupTelegramBot() {
        if (!this.config.telegram?.enabled) return;

        try {
            const response = await this.makeHttpRequest(
                `https://api.telegram.org/bot${this.config.telegram.botToken}/getMe`
            );

            if (response.ok) {
                this.channels.telegram = true;
                this.emit('channelEnabled', 'telegram');
            }
        } catch (error) {
            this.emit('error', `Failed to setup Telegram bot: ${error.message}`);
        }
    }

    async setupDiscordWebhook() {
        if (!this.config.discord?.enabled) return;

        try {
            const payload = {
                content: 'Bot initialization test',
                embeds: [{
                    title: 'Crypto Trading Bot',
                    description: 'Notification service initialized',
                    color: 0x00ff00,
                    timestamp: new Date().toISOString()
                }]
            };

            const response = await this.makeHttpRequest(
                this.config.discord.webhookUrl,
                'POST',
                payload
            );

            if (response.status < 400) {
                this.channels.discord = true;
                this.emit('channelEnabled', 'discord');
            }
        } catch (error) {
            this.emit('error', `Failed to setup Discord webhook: ${error.message}`);
        }
    }

    async setupSlackWebhook() {
        if (!this.config.slack?.enabled) return;

        try {
            const payload = {
                text: 'Crypto Trading Bot notification service initialized',
                attachments: [{
                    color: 'good',
                    fields: [{
                        title: 'Status',
                        value: 'Online',
                        short: true
                    }]
                }]
            };

            const response = await this.makeHttpRequest(
                this.config.slack.webhookUrl,
                'POST',
                payload
            );

            if (response.status < 400) {
                this.channels.slack = true;
                this.emit('channelEnabled', 'slack');
            }
        } catch (error) {
            this.emit('error', `Failed to setup Slack webhook: ${error.message}`);
        }
    }

    loadNotificationTemplates() {
        this.templates.set('trade_executed', {
            email: {
                subject: 'Trade Executed - {{strategy}}',
                html: `
                    <h2>Trade Executed</h2>
                    <p><strong>Strategy:</strong> {{strategy}}</p>
                    <p><strong>Type:</strong> {{type}}</p>
                    <p><strong>Profit:</strong> {{profit}} ETH</p>
                    <p><strong>Gas Cost:</strong> {{gasCost}} ETH</p>
                    <p><strong>Transaction:</strong> <a href="https://etherscan.io/tx/{{txHash}}">{{txHash}}</a></p>
                `
            },
            telegram: '🚀 Trade Executed\n\n📈 Strategy: {{strategy}}\n💰 Profit: {{profit}} ETH\n⛽ Gas: {{gasCost}} ETH\n🔗 TX: {{txHash}}',
            discord: {
                embeds: [{
                    title: '🚀 Trade Executed',
                    color: 0x00ff00,
                    fields: [
                        { name: 'Strategy', value: '{{strategy}}', inline: true },
                        { name: 'Profit', value: '{{profit}} ETH', inline: true },
                        { name: 'Gas Cost', value: '{{gasCost}} ETH', inline: true }
                    ],
                    footer: { text: 'TX: {{txHash}}' },
                    timestamp: '{{timestamp}}'
                }]
            }
        });

        this.templates.set('arbitrage_opportunity', {
            email: {
                subject: 'Arbitrage Opportunity - {{spread}}% spread',
                html: `
                    <h2>Arbitrage Opportunity Detected</h2>
                    <p><strong>Token:</strong> {{symbol}}</p>
                    <p><strong>Buy Source:</strong> {{buySource}} ({{buyPrice}})</p>
                    <p><strong>Sell Source:</strong> {{sellSource}} ({{sellPrice}})</p>
                    <p><strong>Spread:</strong> {{spread}}%</p>
                    <p><strong>Potential Profit:</strong> {{profit}}%</p>
                `
            },
            telegram: '💎 Arbitrage Opportunity\n\n🪙 {{symbol}}\n📉 Buy: {{buySource}} ({{buyPrice}})\n📈 Sell: {{sellSource}} ({{sellPrice}})\n💰 Spread: {{spread}}%',
            discord: {
                embeds: [{
                    title: '💎 Arbitrage Opportunity',
                    color: 0xffd700,
                    fields: [
                        { name: 'Token', value: '{{symbol}}', inline: true },
                        { name: 'Spread', value: '{{spread}}%', inline: true },
                        { name: 'Buy Price', value: '{{buyPrice}} ({{buySource}})', inline: false },
                        { name: 'Sell Price', value: '{{sellPrice}} ({{sellSource}})', inline: false }
                    ],
                    timestamp: '{{timestamp}}'
                }]
            }
        });

        this.templates.set('liquidation_opportunity', {
            email: {
                subject: 'Liquidation Opportunity - {{protocol}}',
                html: `
                    <h2>Liquidation Opportunity</h2>
                    <p><strong>Protocol:</strong> {{protocol}}</p>
                    <p><strong>User:</strong> {{user}}</p>
                    <p><strong>Health Factor:</strong> {{healthFactor}}</p>
                    <p><strong>Potential Reward:</strong> {{reward}} ETH</p>
                `
            },
            telegram: '⚡ Liquidation Alert\n\n🏛️ {{protocol}}\n👤 {{user}}\n💊 Health: {{healthFactor}}\n💰 Reward: {{reward}} ETH',
            discord: {
                embeds: [{
                    title: '⚡ Liquidation Opportunity',
                    color: 0xff4444,
                    fields: [
                        { name: 'Protocol', value: '{{protocol}}', inline: true },
                        { name: 'Health Factor', value: '{{healthFactor}}', inline: true },
                        { name: 'Reward', value: '{{reward}} ETH', inline: true }
                    ],
                    timestamp: '{{timestamp}}'
                }]
            }
        });

        this.templates.set('error_alert', {
            email: {
                subject: 'Error Alert - {{service}}',
                html: `
                    <h2 style="color: red;">Error Alert</h2>
                    <p><strong>Service:</strong> {{service}}</p>
                    <p><strong>Error:</strong> {{error}}</p>
                    <p><strong>Time:</strong> {{timestamp}}</p>
                    <p><strong>Severity:</strong> {{severity}}</p>
                `
            },
            telegram: '🚨 Error Alert\n\n⚙️ Service: {{service}}\n❌ Error: {{error}}\n🔴 Severity: {{severity}}',
            discord: {
                embeds: [{
                    title: '🚨 Error Alert',
                    color: 0xff0000,
                    fields: [
                        { name: 'Service', value: '{{service}}', inline: true },
                        { name: 'Severity', value: '{{severity}}', inline: true },
                        { name: 'Error', value: '{{error}}', inline: false }
                    ],
                    timestamp: '{{timestamp}}'
                }]
            }
        });

        this.templates.set('gas_price_alert', {
            telegram: '⛽ Gas Alert\n\n📊 Current: {{currentGas}} gwei\n🎯 Target: {{targetGas}} gwei\n📈 Trend: {{trend}}',
            discord: {
                embeds: [{
                    title: '⛽ Gas Price Alert',
                    color: 0xffaa00,
                    fields: [
                        { name: 'Current Gas', value: '{{currentGas}} gwei', inline: true },
                        { name: 'Target Gas', value: '{{targetGas}} gwei', inline: true },
                        { name: 'Trend', value: '{{trend}}', inline: true }
                    ],
                    timestamp: '{{timestamp}}'
                }]
            }
        });
    }

    async sendNotification(type, data, priority = 'medium', channels = null) {
        const notification = {
            id: `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type,
            data,
            priority,
            channels: channels || Object.keys(this.channels).filter(c => this.channels[c]),
            timestamp: new Date().toISOString(),
            attempts: 0,
            maxAttempts: 3
        };

        this.priorityQueues[priority].push(notification);
        this.emit('notificationQueued', notification);
    }

    startQueueProcessor() {
        setInterval(() => {
            this.processNotificationQueue();
        }, 1000);
    }

    async processNotificationQueue() {
        const priorities = ['urgent', 'high', 'medium', 'low'];
        
        for (const priority of priorities) {
            const queue = this.priorityQueues[priority];
            
            if (queue.length > 0) {
                const notification = queue.shift();
                await this.executeNotification(notification);
                
                if (priority === 'urgent') {
                    continue;
                } else {
                    break;
                }
            }
        }
    }

    async executeNotification(notification) {
        const { type, data, channels } = notification;
        const template = this.templates.get(type);
        
        if (!template) {
            this.emit('error', `No template found for notification type: ${type}`);
            return;
        }

        const results = [];

        for (const channel of channels) {
            if (!this.channels[channel]) continue;
            
            try {
                if (this.checkRateLimit(channel, type)) {
                    await this.sendToChannel(channel, template, data);
                    results.push({ channel, success: true });
                } else {
                    results.push({ channel, success: false, reason: 'Rate limited' });
                }
            } catch (error) {
                results.push({ channel, success: false, error: error.message });
                this.emit('error', `Failed to send to ${channel}: ${error.message}`);
            }
        }

        this.emit('notificationSent', {
            id: notification.id,
            type: notification.type,
            results
        });
    }

    async sendToChannel(channel, template, data) {
        const channelTemplate = template[channel];
        if (!channelTemplate) return;

        switch (channel) {
            case 'email':
                await this.sendEmail(channelTemplate, data);
                break;
            case 'telegram':
                await this.sendTelegram(channelTemplate, data);
                break;
            case 'discord':
                await this.sendDiscord(channelTemplate, data);
                break;
            case 'slack':
                await this.sendSlack(channelTemplate, data);
                break;
            case 'webhook':
                await this.sendWebhook(data);
                break;
        }
    }

    async sendEmail(template, data) {
        const subject = this.renderTemplate(template.subject, data);
        const html = this.renderTemplate(template.html, data);

        await this.emailTransporter.sendMail({
            from: this.config.email.from,
            to: this.config.email.to,
            subject,
            html
        });
    }

    async sendTelegram(template, data) {
        const message = this.renderTemplate(template, data);
        
        const payload = {
            chat_id: this.config.telegram.chatId,
            text: message,
            parse_mode: 'HTML',
            disable_web_page_preview: true
        };

        await this.makeHttpRequest(
            `https://api.telegram.org/bot${this.config.telegram.botToken}/sendMessage`,
            'POST',
            payload
        );
    }

    async sendDiscord(template, data) {
        let payload;
        
        if (typeof template === 'string') {
            payload = { content: this.renderTemplate(template, data) };
        } else {
            payload = JSON.parse(this.renderTemplate(JSON.stringify(template), data));
        }

        await this.makeHttpRequest(
            this.config.discord.webhookUrl,
            'POST',
            payload
        );
    }

    async sendSlack(template, data) {
        let payload;
        
        if (typeof template === 'string') {
            payload = { text: this.renderTemplate(template, data) };
        } else {
            payload = JSON.parse(this.renderTemplate(JSON.stringify(template), data));
        }

        await this.makeHttpRequest(
            this.config.slack.webhookUrl,
            'POST',
            payload
        );
    }

    async sendWebhook(data) {
        if (!this.config.webhook?.url) return;

        await this.makeHttpRequest(
            this.config.webhook.url,
            'POST',
            {
                timestamp: new Date().toISOString(),
                source: 'crypto-trading-bot',
                data
            }
        );
    }

    renderTemplate(template, data) {
        let rendered = template;
        
        for (const [key, value] of Object.entries(data)) {
            const placeholder = new RegExp(`{{${key}}}`, 'g');
            rendered = rendered.replace(placeholder, value || '');
        }
        
        return rendered;
    }

    checkRateLimit(channel, type) {
        const key = `${channel}-${type}`;
        const now = Date.now();
        const limits = this.config.rateLimits || {
            email: { count: 10, window: 3600000 },
            telegram: { count: 30, window: 60000 },
            discord: { count: 50, window: 60000 },
            slack: { count: 20, window: 60000 }
        };

        if (!this.rateLimiters.has(key)) {
            this.rateLimiters.set(key, []);
        }

        const requests = this.rateLimiters.get(key);
        const limit = limits[channel] || { count: 10, window: 60000 };
        
        const windowStart = now - limit.window;
        const recentRequests = requests.filter(time => time > windowStart);
        
        if (recentRequests.length >= limit.count) {
            return false;
        }

        recentRequests.push(now);
        this.rateLimiters.set(key, recentRequests);
        return true;
    }

    makeHttpRequest(url, method = 'GET', data = null) {
        return new Promise((resolve, reject) => {
            const urlObj = new URL(url);
            const options = {
                hostname: urlObj.hostname,
                port: urlObj.port || (urlObj.protocol === 'https:' ? 443 : 80),
                path: urlObj.pathname + urlObj.search,
                method,
                headers: {
                    'Content-Type': 'application/json',
                    'User-Agent': 'Crypto-Trading-Bot/1.0'
                }
            };

            if (data) {
                const payload = JSON.stringify(data);
                options.headers['Content-Length'] = Buffer.byteLength(payload);
            }

            const req = https.request(options, (res) => {
                let responseData = '';
                
                res.on('data', (chunk) => {
                    responseData += chunk;
                });
                
                res.on('end', () => {
                    try {
                        const parsedData = responseData ? JSON.parse(responseData) : {};
                        resolve({
                            status: res.statusCode,
                            data: parsedData,
                            ok: res.statusCode < 400
                        });
                    } catch (error) {
                        resolve({
                            status: res.statusCode,
                            data: responseData,
                            ok: res.statusCode < 400
                        });
                    }
                });
            });

            req.on('error', reject);
            
            if (data) {
                req.write(JSON.stringify(data));
            }
            
            req.end();
        });
    }

    async sendTradeAlert(tradeData) {
        await this.sendNotification('trade_executed', {
            strategy: tradeData.strategy,
            type: tradeData.type,
            profit: tradeData.profit,
            gasCost: tradeData.gasCost,
            txHash: tradeData.txHash,
            timestamp: new Date().toISOString()
        }, 'high');
    }

    async sendArbitrageAlert(opportunityData) {
        await this.sendNotification('arbitrage_opportunity', {
            symbol: opportunityData.symbol,
            buySource: opportunityData.buySource,
            sellSource: opportunityData.sellSource,
            buyPrice: opportunityData.buyPrice,
            sellPrice: opportunityData.sellPrice,
            spread: opportunityData.spread.toFixed(2),
            profit: opportunityData.profit.toFixed(2),
            timestamp: new Date().toISOString()
        }, 'medium');
    }

    async sendLiquidationAlert(liquidationData) {
        await this.sendNotification('liquidation_opportunity', {
            protocol: liquidationData.protocol,
            user: liquidationData.user,
            healthFactor: liquidationData.healthFactor,
            reward: liquidationData.reward,
            timestamp: new Date().toISOString()
        }, 'high');
    }

    async sendErrorAlert(errorData) {
        await this.sendNotification('error_alert', {
            service: errorData.service,
            error: errorData.error,
            severity: errorData.severity || 'medium',
            timestamp: new Date().toISOString()
        }, 'urgent');
    }

    async sendGasAlert(gasData) {
        await this.sendNotification('gas_price_alert', {
            currentGas: gasData.currentGas,
            targetGas: gasData.targetGas,
            trend: gasData.trend,
            timestamp: new Date().toISOString()
        }, 'low');
    }

    getChannelStatus() {
        return { ...this.channels };
    }

    getQueueStats() {
        return {
            urgent: this.priorityQueues.urgent.length,
            high: this.priorityQueues.high.length,
            medium: this.priorityQueues.medium.length,
            low: this.priorityQueues.low.length,
            total: Object.values(this.priorityQueues).reduce((sum, queue) => sum + queue.length, 0)
        };
    }

    cleanup() {
        this.priorityQueues = {
            urgent: [],
            high: [],
            medium: [],
            low: []
        };
        this.rateLimiters.clear();
    }
}

module.exports = NotificationService;