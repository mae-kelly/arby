#!/bin/bash
echo "📊 Starting CEX Arbitrage Bot"
echo "Make sure your .env file has exchange API keys!"
python3 src/orchestrator_m1_fixed.py
