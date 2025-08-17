cmake_minimum_required(VERSION 3.10)
project(CryptoArbBot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")

find_package(Threads REQUIRED)

# Create orderbook library
if(EXISTS "${CMAKE_SOURCE_DIR}/src/cpp/orderbook_scanner.cpp")
    add_library(orderbook SHARED src/cpp/orderbook_scanner.cpp)
elseif(EXISTS "${CMAKE_SOURCE_DIR}/src/cpp/orderbook.cpp")
    add_library(orderbook SHARED src/cpp/orderbook.cpp)
else()
    # Create dummy
    file(WRITE "${CMAKE_BINARY_DIR}/dummy_orderbook.cpp" "extern \"C\" { void dummy() {} }")
    add_library(orderbook SHARED ${CMAKE_BINARY_DIR}/dummy_orderbook.cpp)
endif()

# Create mempool library
if(EXISTS "${CMAKE_SOURCE_DIR}/src/cpp/mempool_monitor.cpp")
    add_library(mempool SHARED src/cpp/mempool_monitor.cpp)
elseif(EXISTS "${CMAKE_SOURCE_DIR}/src/cpp/mempool.cpp")
    add_library(mempool SHARED src/cpp/mempool.cpp)
else()
    # Create dummy
    file(WRITE "${CMAKE_BINARY_DIR}/dummy_mempool.cpp" "extern \"C\" { void dummy() {} }")
    add_library(mempool SHARED ${CMAKE_BINARY_DIR}/dummy_mempool.cpp)
endif()

target_link_libraries(orderbook Threads::Threads)
target_link_libraries(mempool Threads::Threads)

set_target_properties(orderbook mempool PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)
