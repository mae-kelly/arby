#!/usr/bin/env python3
import asyncio
import json
import time
import psutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from typing import Dict, List
import websockets
import logging

class ArbitrageMonitor:
    def __init__(self):
        self.db_path = "monitoring/arbitrage_metrics.db"
        self.setup_database()
        self.opportunities_count = 0
        self.executed_trades = 0
        self.total_profit = 0.0
        self.error_count = 0
        self.start_time = time.time()
        
    def setup_database(self):
        """Setup SQLite database for metrics"""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp REAL,
                cpu_percent REAL,
                memory_percent REAL,
                disk_percent REAL,
                network_bytes_sent INTEGER,
                network_bytes_recv INTEGER,
                gpu_usage REAL,
                gpu_memory REAL
            );
            
            CREATE TABLE IF NOT EXISTS arbitrage_metrics (
                timestamp REAL,
                opportunities_found INTEGER,
                trades_executed INTEGER,
                profit_usd REAL,
                errors INTEGER,
                avg_latency_ms REAL,
                active_exchanges INTEGER,
                active_chains INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS exchange_metrics (
                timestamp REAL,
                exchange_name TEXT,
                symbols_count INTEGER,
                uptime_percent REAL,
                avg_latency_ms REAL,
                error_rate REAL,
                volume_24h REAL
            );
            
            CREATE TABLE IF NOT EXISTS alerts (
                timestamp REAL,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                acknowledged BOOLEAN DEFAULT FALSE
            );
        """)
        conn.close()
    
    def collect_system_metrics(self) -> Dict:
        """Collect comprehensive system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': (disk.used / disk.total) * 100,
            'network_bytes_sent': network.bytes_sent,
            'network_bytes_recv': network.bytes_recv,
            'memory_available_gb': memory.available / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            'process_count': len(psutil.pids())
        }
        
        # GPU metrics (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics.update({
                    'gpu_usage': gpu.load * 100,
                    'gpu_memory': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_temperature': gpu.temperature
                })
        except ImportError:
            metrics.update({
                'gpu_usage': 0,
                'gpu_memory': 0,
                'gpu_temperature': 0
            })
        
        # Process-specific metrics
        arbitrage_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            if any(keyword in proc.info['name'].lower() 
                   for keyword in ['arbitrage', 'orchestrator', 'python']):
                arbitrage_processes.append(proc.info)
        
        metrics['arbitrage_processes'] = arbitrage_processes
        metrics['arbitrage_process_count'] = len(arbitrage_processes)
        
        return metrics
    
    def collect_arbitrage_metrics(self) -> Dict:
        """Collect arbitrage-specific metrics"""
        runtime_hours = (time.time() - self.start_time) / 3600
        
        metrics = {
            'timestamp': time.time(),
            'opportunities_found': self.opportunities_count,
            'trades_executed': self.executed_trades,
            'profit_usd': self.total_profit,
            'errors': self.error_count,
            'runtime_hours': runtime_hours,
            'opportunities_per_hour': self.opportunities_count / max(runtime_hours, 0.1),
            'profit_per_hour': self.total_profit / max(runtime_hours, 0.1),
            'success_rate': (self.executed_trades / max(self.opportunities_count, 1)) * 100,
            'error_rate': (self.error_count / max(self.opportunities
