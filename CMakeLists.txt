cmake_minimum_required(VERSION 3.15)
project(ArbitrageBot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -arch arm64")

find_package(Threads REQUIRED)

# Orderbook scanner
if(EXISTS "${CMAKE_SOURCE_DIR}/src/cpp/orderbook_scanner.cpp")
    add_library(orderbook SHARED src/cpp/orderbook_scanner.cpp)
    target_link_libraries(orderbook Threads::Threads)
else()
    # Create dummy
    file(WRITE ${CMAKE_BINARY_DIR}/dummy_orderbook.cpp "void dummy_orderbook() {}")
    add_library(orderbook SHARED ${CMAKE_BINARY_DIR}/dummy_orderbook.cpp)
endif()

# Mempool monitor
if(EXISTS "${CMAKE_SOURCE_DIR}/src/cpp/mempool_monitor.cpp")
    add_library(mempool SHARED src/cpp/mempool_monitor.cpp)
    target_link_libraries(mempool Threads::Threads)
else()
    # Create dummy
    file(WRITE ${CMAKE_BINARY_DIR}/dummy_mempool.cpp "void dummy_mempool() {}")
    add_library(mempool SHARED ${CMAKE_BINARY_DIR}/dummy_mempool.cpp)
endif()

set_target_properties(orderbook mempool PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)
