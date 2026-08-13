/*
 * ei_probe.hlsl — ExecuteIndirect probe kernel.
 * Each indirect command sets root constant myval (incrementing or absolute),
 * dispatches 1 group x 64 threads; thread 0 writes myval to out[myval].
 * Verifies: (a) ExecuteIndirect dispatches run, (b) the per-command root
 * constant arrives, (c) incrementing-constant semantics work on this driver.
 */

struct EIProbeParams {
    uint myval;
    uint pad[3];
};

ConstantBuffer<EIProbeParams> cb : register(b0);
RWByteAddressBuffer outbuf : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    if (gtid.x != 0) return;
    outbuf.Store(cb.myval * 4, cb.myval);
}
