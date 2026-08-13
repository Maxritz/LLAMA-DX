/*
 * wg_scale.hlsl
 * PURPOSE: Work Graph (Shader Model 6.8) probe node — elementwise scale.
 *
 * Broadcast-launch node. The CPU dispatch provides one input record holding
 * the launch grid (SV_DispatchGrid); the node runs NumThreads(64) per grid
 * cell. Root params mirror the mm signature pattern (root constants b0 +
 * root UAVs u0/u1) so the local root signature binds them without
 * descriptors.
 *
 * This is the OPTIONAL acceleration path. Every host that cannot run work
 * graphs (no tier, no env opt-in, state-object failure) falls back to the
 * classic per-dispatch path unchanged. See dx12_workgraph.cpp.
 *
 * NOTE: node shaders compile with the lib_6_8 profile (not cs_6_8).
 */

struct WGScaleParams {
    uint  nelems;
    float scale;
    uint  pad[2];
};

ConstantBuffer<WGScaleParams> cb : register(b0);
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

struct WGLaunchGrid {
    uint3 grid : SV_DispatchGrid;
};

[Shader("node")]
[NodeLaunch("broadcasting")]
[NodeMaxDispatchGrid(65535, 1, 1)]
[NumThreads(64, 1, 1)]
void wg_scale(DispatchNodeInputRecord<WGLaunchGrid> input, uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= cb.nelems) return;
    dst.Store(dtid.x * 4, asuint(asfloat(src.Load(dtid.x * 4)) * cb.scale));
}
