cmake_minimum_required(VERSION 3.18)
project(HyperArbitrageGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto")

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# CUDA architecture detection
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89) # A100, RTX 30/40xx, H100
endif()

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/include/python3.10)

# Main GPU arbitrage engine
add_library(hyperarbitrage_gpu SHARED
    src/gpu_arbitrage_engine.cu
    src/price_calculator.cu
    src/opportunity_detector.cu
    src/execution_engine.cpp
    src/market_data_processor.cu
    src/cross_chain_optimizer.cu
    src/flash_loan_calculator.cu
    src/mev_protection.cu
    src/risk_calculator.cu
    src/portfolio_optimizer.cu
)

target_link_libraries(hyperarbitrage_gpu
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CURAND_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
    Threads::Threads
    python3.10
)

# Set CUDA properties for maximum performance
set_target_properties(hyperarbitrage_gpu PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Python bindings
add_library(hyperarbitrage_py SHARED
    src/python_bindings.cpp
)

target_link_libraries(hyperarbitrage_py
    hyperarbitrage_gpu
    python3.10
)
