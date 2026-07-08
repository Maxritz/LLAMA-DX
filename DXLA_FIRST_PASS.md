DXLA Wave Matmul — First Verified Correct Result on AMD RX 9070 XT
===================================================================

Date: 2026-07-07

Hardware: AMD Radeon RX 9070 XT (RDNA4), driver 26.10.07.02 (Agility SDK preview)

Result: 4/4 tests pass. DXLA wave matmul computes 16x16x16 F16xF16->F32 correctly.
All 256 output elements = exactly 16.0 (K=16, identity matrix test).


Working Toolchain
-----------------

  DXC:         v1.10.2605.2 (ea53cb53) — April 27, 2026 SM 6.10 preview
               Downloaded from: https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.10.2605.2
  Agility SDK: 1.721.1 (1.721.1.0.20260617.1) or 1.721.2 (1.721.2.0.20260702.2)
               Both work for standard shaders. DXLA works with either when paired with correct DXC.
  Driver:      AMD Developer Preview 26.10.07.02
  GPU:         AMD Radeon RX 9070 XT
  OS:          Windows 11


DXC Regression — CRITICAL
--------------------------

  DXC v1.10.2605.2  (ea53cb53) → driver accepts DXIL, PSO creates, matmul computes
  DXC v1.10.2605.24 (c1e1fc784) → driver rejects with "Unknown DXIL LinalgMatrixLayout"
                                   reason=0x887A0020 (DRIVER_INTERNAL_ERROR)
                                   device removed at CreateComputePipelineState

  Root cause: The fix for DXC issue #8433 ("validator rejects LinAlgMatrix type in DXIL")
  changed the LinAlgMatrix type encoding in the DXIL metadata. DXC's own validator
  now accepts it, but the AMD driver validator rejects the new encoding.

  Both DXC versions produce valid DXIL per their respective validators.

  Also: DXC 1.9.2602.24 (dxc-older/) does NOT support cs_6_10. Unusable for LinAlg.


Shader Requirements (for AMD driver compatibility)
--------------------------------------------------

  1. MatrixLayout MUST be a compile-time constant.
     Runtime/dynamic layout values (e.g. params.transposed_b ? ColMajor : RowMajor)
     cause "Unknown DXIL LinalgMatrixLayout" even with the working DXC.

     WORKING:    MatrixLayout::ColMajor        (hardcoded, 5080-byte CSO)
     BROKEN:     params.transposed_b ? ...       (dynamic, 5092-byte CSO)

     Consequence: Non-transposed B (RowMajor) and transposed B (ColMajor) need
     separate shaders, each with hardcoded layout. Or offset-based workaround
     that avoids changing the layout parameter.

  2. Output buffer size: shader writes F32 (ComponentType::F32 accumulator).
     The Store function writes 4 bytes per element regardless of output buffer type.
     C++ side must allocate M * N * 4 bytes, not M * N * 2.
     This was masked by the PSO creation failure until fixed.

  3. CBV struct must match EXACTLY between C++ and HLSL.

     C++ (dx12_gemm.cpp, dxla_constants):
       uint32_t M, N, K;
       uint32_t stride_a, stride_b, stride_c;  // elements (not bytes)
       uint32_t transposed_b;
       uint32_t wave_size;                     // 32 or 64
       uint32_t reserved[9];                   // pad to 17 uints = 68 bytes

     HLSL (shader, DXLAWaveGEMMParams):
       Same layout: M, N, K, stride_a, stride_b, stride_c, transposed_b, wave_size, reserved[9]

     The OLD C++ struct was: {M, N, K, wave_size, reserved[12]}  (16 uints, 64 bytes)
     which shifted all fields after M by 4 bytes relative to the shader's layout.
     Result: stride_a = wave_size, stride_b/c = 0, transposed_b = 0.


Standard Shaders — SDK Compatibility
-------------------------------------

  Standard shaders (add, copy, mul_mat_f16_f16) work with Agility SDK 1.721.x
  paired with driver 26.10.07.02. Without the matching SDK, ALL PSO creation
  fails with E_INVALIDARG (0x80070057).

  The D3D12Core.dll from Agility SDK vendor must match the driver version.
  Mismatched SDK → everything breaks. Matching SDK → standard shaders work,
  DXLA works only with correct DXC (see above).


Bugs Found and Fixed
--------------------

  1. CBV struct mismatch (C++ vs HLSL) — fixed in dx12_gemm.cpp
  2. F32/F16 output buffer size — fixed in test_dx12_gemm.cpp (sz_c = M*N*4)
  3. Dynamic MatrixLayout rejected by driver — worked around via hardcoded ColMajor
  4. DXC 1.10.2605.24 regression — documented, using 1.10.2605.2 as workaround


Files Changed
-------------

  dx12_gemm.cpp:      dxla_constants struct + field initialization
  dx12_descriptor.cpp: GetDeviceRemovedReason() diagnostic on PSO failure
  dx12_device.cpp:     Re-enabled dxla_wave caps (was hardcoded false)
  shaders/mul_mat_dxla_wave_f16_f16.hlsl: hardcoded ColMajor for B matrix
  tests/test_dx12_gemm.cpp: F32 output buffer, float readback, pso_simple test


Outstanding
-----------

  - Non-transposed B case needs separate shader or offset workaround
  - DXC regression should be filed upstream (AMD driver rejects #8433 fix encoding)
  - Other DXLA shaders (q4_0, tg, attn) untested on this configuration
  - Buffer size mismatch in production path (not just test) — C++ side always
    allocates F16-sized output for DXLA dispatch
