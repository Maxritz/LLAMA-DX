/*
 * rms_norm.hlsl
 * PURPOSE: RMS Normalization: output = x / sqrt(mean(x^2) + eps) * weight
 * Uses wave-level reductions for the mean calculation.
 */

#include "common.hlsli"
struct Params { uint n; uint row_size; float eps; uint pad; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src : register(t0);
ByteAddressBuffer weight : register(t1);
RWByteAddressBuffer dst : register(u0);

half load_src(uint idx) { uint a = idx * 2; uint p = src.Load(a & ~2); uint16_t b = (a & 2) ? (uint16_t)(p >> 16) : (uint16_t)(p & 0xFFFF); return (half)f16_to_f32(b); }
half load_weight(uint idx) { uint a = idx * 2; uint p = weight.Load(a & ~2); uint16_t b = (a & 2) ? (uint16_t)(p >> 16) : (uint16_t)(p & 0xFFFF); return (half)f16_to_f32(b); }
void store_dst(uint idx, half v) { store_packed_f16(dst, idx, v); }

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint row = tid.x / params.row_size;
    uint lane = tid.x % params.row_size;
    uint base = row * params.row_size;
    if (base + lane >= params.n) return;

    // Step 1: Each thread loads its element and computes x^2
    half x = load_src(base + lane);
    float x_sq = (float)x * (float)x;

    // Step 2: Wave-level sum of squares
    float sum_sq = WaveActiveSum(x_sq);

    // Step 3: Compute RMS (only first lane has the full sum)
    float mean_sq = sum_sq / (float)params.row_size;
    float rms = sqrt(mean_sq + params.eps);
    float inv_rms = 1.0f / rms;

    // Step 4: Normalize and apply weight
    half w = load_weight(lane);
    float normalized = (float)x * inv_rms * (float)w;
    store_dst(base + lane, (half)normalized);
}
