#!/bin/bash
echo "⚡ Starting Flash Loan Arbitrage Bot"
echo "⚠️  WARNING: This trades on MAINNET with REAL money!"
echo "Make sure you have:"
echo "- Funded wallet with ETH for gas"
echo "- Deployed flash loan contract"
echo "- All RPC URLs configured"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm
if [ "$confirm" = "yes" ]; then
    python3 runner.py
else
    echo "Cancelled."
fi
