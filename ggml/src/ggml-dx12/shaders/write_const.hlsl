// write_const.hlsl - minimal test shader: write constant value to output
// Compiles with: dxc /T cs_6_10 /E main /Fo write_const.cso write_const.hlsl

RWByteAddressBuffer output : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y * 8 + lid.y;
    uint col = gid.x * 8 + lid.x;
    // Only write to element (0,0) - thread (0,0,0) only
    if (row == 0 && col == 0) {
        uint addr = 0;
        uint val = 0x3C00; // F16 1.0
        output.Store(addr, val);
    }
}
