#include <metal_stdlib>
using namespace metal;

kernel void find_arbitrage_metal(
    device float* prices [[buffer(0)]],
    device int* opportunities [[buffer(1)]],
    constant int& num_markets [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int idx = gid.y * num_markets + gid.x;
    
    if (gid.x < num_markets && gid.y < num_markets && gid.x != gid.y) {
        float buy_price = prices[gid.x * 2];
        float sell_price = prices[gid.y * 2 + 1];
        float profit = (sell_price - buy_price) / buy_price;
        
        if (profit > threshold) {
            opportunities[idx] = 1;
        }
    }
}
