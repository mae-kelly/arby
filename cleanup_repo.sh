#!/bin/bash

echo "🧹 Cleaning up repository for production..."

# Remove test files
echo "Removing test files..."
rm -f test_*.py
rm -f *_test.py
rm -f test_all_components.py
rm -f test_build.py
rm -f test_m1.py

# Remove demo/example files
echo "Removing demo files..."
rm -f working_bot.py
rm -f simple_bot.py
rm -f hey.py
rm -f new.py
rm -f oneee.sh

# Remove backup files
echo "Removing backup files..."
rm -f *.backup
rm -f src/orchestrator_m1.py.backup

# Remove build artifacts (keep directories)
echo "Cleaning build artifacts..."
rm -rf target/debug/
rm -rf target/*/incremental/
find target/ -name "*.rlib" -delete 2>/dev/null || true
find target/ -name "*.d" -delete 2>/dev/null || true

# Remove temporary files
echo "Removing temporary files..."
rm -f *.tmp
rm -f *.log
rm -f .DS_Store
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove duplicate/redundant orchestrators (keep the best ones)
echo "Removing redundant orchestrators..."
rm -f fix_orchestrator.py
rm -f full_system.py
rm -f platform_bot.py
rm -f runner.py

# Remove redundant Python files in root
echo "Removing redundant root files..."
rm -f final_bot.py
rm -f orchestrator_m1_fixed.py

# Remove notebook duplicates (keep main.ipynb)
echo "Cleaning notebooks..."
rm -f universal_bot.ipynb

# Remove redundant benchmark files
echo "Removing redundant benchmarks..."
rm -f benches/pathfinding.rs

# Clean logs directory but keep structure
echo "Cleaning logs..."
mkdir -p logs
rm -f logs/*.log 2>/dev/null || true

# Remove empty directories
echo "Removing empty directories..."
find . -type d -empty -delete 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "📁 Production files kept:"
echo "   Core: src/lib.rs, src/engine.rs"
echo "   Python: src/python/orchestrator.py, src/python/ultra_orchestrator.py" 
echo "   Strategies: strategies/ directory"
echo "   Solidity: src/solidity/ directory"
echo "   GPU: src/gpu_kernel.cu, src/metal/"
echo "   Config: config/ directory"
echo ""
echo "🚀 Ready for production deployment!"