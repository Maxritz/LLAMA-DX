# compile_shaders.ps1
# COMPONENT: 2 (HLSL Kernel Library)
# PURPOSE: Compile all .hlsl shaders to .cso and generate C++ registry
#
# USAGE: .\compile_shaders.ps1 [-Configuration Debug|Release]
#
# This script:
# 1. Finds the DXC compiler
# 2. Compiles each .hlsl file to .cso (compiled shader object)
# 3. Generates dx12_shader_registry.cpp/h (embedded CSO blobs)

param(
    [string]$Configuration = "Release",
    [string]$DxcPath = "",
    [string]$OutputDir = "",
    [string]$ShaderDir = ""
)

$ErrorActionPreference = "Stop"

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

if (-not $ShaderDir) {
    $ShaderDir = $PSScriptRoot
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $ShaderDir ".." "generated"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# ═══════════════════════════════════════════════════════════════════════════════
# Find DXC
# ═══════════════════════════════════════════════════════════════════════════════

$DxcExe = $DxcPath
if (-not $DxcExe) {
    # Try Windows SDK paths
    $SdkPaths = @(
        "${env:WindowsSdkDir}bin\${env:WindowsSDKVersion}x64\dxc.exe",
        "${env:WindowsSdkDir}bin\x64\dxc.exe",
        "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\dxc.exe",
        "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe",
        "C:\Program Files (x86)\Windows Kits\10\bin\x64\dxc.exe"
    )
    foreach ($path in $SdkPaths) {
        if (Test-Path $path) {
            $DxcExe = $path
            break
        }
    }
}

if (-not $DxcExe -or -not (Test-Path $DxcExe)) {
    Write-Error "DXC compiler not found. Install Windows SDK 22621+ or provide -DxcPath"
    exit 1
}

Write-Host "Using DXC: $DxcExe" -ForegroundColor Green

# Target a stable (non-experimental) shader model. cs_6_10 was previously used
# for all shaders, but that requires D3D12EnableExperimentalFeatures at runtime
# (unsupported/development mode), which has been observed to correlate with GPU
# hangs on preview drivers even for operations unrelated to any SM 6.10 feature.
# Only the DXLA (cooperative matrix) shaders actually need SM 6.10, and those
# are excluded below since the DXLA dispatch path is currently disabled.
$TargetProfile = "cs_6_6"

# ═══════════════════════════════════════════════════════════════════════════════
# Shader List
# ═══════════════════════════════════════════════════════════════════════════════

$Shaders = @(
    # Dequantization
    @{Name="dequant_q4_0"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="dequant_q8_0"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="dequant_q6_k"; Threads=@{X=64;Y=1;Z=1}},
    @{Name="dequant_q4_k"; Threads=@{X=64;Y=1;Z=1}},
    @{Name="dequant_q5_k"; Threads=@{X=64;Y=1;Z=1}},
    # Cast
    @{Name="cast_f16_f32"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="cast_bf16_f16"; Threads=@{X=256;Y=1;Z=1}},
    # GEMM
    @{Name="mul_mat_f16_f16"; Threads=@{X=32;Y=32;Z=1}},
    @{Name="mul_mat_f16_f32"; Threads=@{X=16;Y=16;Z=1}},
    @{Name="mul_mat_q4_0_f16"; Threads=@{X=16;Y=16;Z=1}},
    @{Name="mul_mat_q8_0_f16"; Threads=@{X=16;Y=16;Z=1}},
    @{Name="mul_mat_batched"; Threads=@{X=8;Y=8;Z=4}},
    @{Name="mul_mat_strided"; Threads=@{X=16;Y=16;Z=1}},
    # DXLA GEMM/Attention shaders excluded — need SM 6.10 (dx::linalg cooperative
    # matrix), but the DXLA dispatch path is currently disabled (dx12_gemm.cpp /
    # dx12_device.cpp), so these are unreachable dead weight. Re-add here (and
    # switch $TargetProfile back to cs_6_10 + re-enable experimental features in
    # dx12_device.cpp) if DXLA is ever revisited:
    #   mul_mat_dxla_wave_f16_f16, mul_mat_dxla_wave_q4_0_f16,
    #   mul_mat_dxla_tg_f16_f16, attn_qk_dxla, attn_ov_dxla
    # Activation
    @{Name="silu"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="gelu"; Threads=@{X=256;Y=1;Z=1}},
    # Elementwise
    @{Name="add"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="mul"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="scale"; Threads=@{X=256;Y=1;Z=1}},
    # Normalization
    @{Name="rms_norm"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="layer_norm"; Threads=@{X=256;Y=1;Z=1}},
    # Attention
    @{Name="soft_max"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="rope"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="diag_mask_inf"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="flash_attn"; Threads=@{X=64;Y=1;Z=1}},
    # FFN
    @{Name="ffn_fused"; Threads=@{X=256;Y=1;Z=1}},
    # Misc
    @{Name="get_rows"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="permute"; Threads=@{X=256;Y=1;Z=1}},
    @{Name="copy"; Threads=@{X=256;Y=1;Z=1}}
)

# ═══════════════════════════════════════════════════════════════════════════════
# Compile Shaders
# ═══════════════════════════════════════════════════════════════════════════════

$SuccessCount = 0
$FailCount = 0
$CsoFiles = @()

Write-Host ""
Write-Host "Compiling $($Shaders.Count) shaders..." -ForegroundColor Cyan
Write-Host ""

$DebugFlags = if ($Configuration -eq "Debug") { @("-Zi", "-Qembed_debug", "-Od") } else { @("-O3") }

foreach ($shader in $Shaders) {
    $Name = $shader.Name
    $HlslFile = Join-Path $ShaderDir "$Name.hlsl"
    $CsoFile = Join-Path $OutputDir "$Name.cso"

    if (-not (Test-Path $HlslFile)) {
        Write-Warning "Shader not found: $HlslFile"
        $FailCount++
        continue
    }

    $args = @(
        "-T", $TargetProfile,
        "-E", "main",
        "-enable-16bit-types",
        "-HV", "2021",
        "-Fo", $CsoFile,
        $HlslFile
    ) + $DebugFlags

    try {
        $output = & $DxcExe @args 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error "FAILED: $Name`n$output"
            $FailCount++
        } else {
            $size = (Get-Item $CsoFile).Length
            Write-Host "  OK  $Name.cso ($size bytes)" -ForegroundColor Green
            $SuccessCount++
            $CsoFiles += @{Name=$Name; File=$CsoFile; Threads=$shader.Threads}
        }
    } catch {
        Write-Error "FAILED: $Name - $_"
        $FailCount++
    }
}

Write-Host ""
Write-Host "Results: $SuccessCount succeeded, $FailCount failed" -ForegroundColor $(if ($FailCount -eq 0) { "Green" } else { "Red" })
Write-Host ""

if ($FailCount -gt 0) {
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════════════
# Generate Registry
# ═══════════════════════════════════════════════════════════════════════════════

Write-Host "Generating shader registry..." -ForegroundColor Cyan

# Header file
$HeaderFile = Join-Path $OutputDir "dx12_shader_registry.h"
$HeaderContent = @"
/* AUTO-GENERATED by compile_shaders.ps1 - DO NOT EDIT */
#pragma once
#include <cstdint>
#include <cstddef>

struct dx12_shader_entry {
    const char*     name;
    const uint8_t*  cso_data;
    size_t          cso_size;
    uint32_t        thread_group_x;
    uint32_t        thread_group_y;
    uint32_t        thread_group_z;
};

extern const dx12_shader_entry DX12_SHADER_REGISTRY[];
extern const size_t DX12_SHADER_COUNT;
"@

Set-Content -Path $HeaderFile -Value $HeaderContent -Encoding UTF8
Write-Host "  Generated: dx12_shader_registry.h" -ForegroundColor Green

# Source file
$SourceFile = Join-Path $OutputDir "dx12_shader_registry.cpp"
$SourceContent = New-Object System.Text.StringBuilder

[void]$SourceContent.AppendLine("/* AUTO-GENERATED by compile_shaders.ps1 - DO NOT EDIT */")
[void]$SourceContent.AppendLine('#include "dx12_shader_registry.h"')
[void]$SourceContent.AppendLine("")

# Embed each CSO as byte array
foreach ($cso in $CsoFiles) {
    $Bytes = [System.IO.File]::ReadAllBytes($cso.File)
    $ArrayName = "cso_$($cso.Name)"

    [void]$SourceContent.AppendLine("// $($cso.Name).cso ($($Bytes.Length) bytes)")
    [void]$SourceContent.AppendLine("static const uint8_t ${ArrayName}_data[] = {")

    $line = "    "
    for ($i = 0; $i -lt $Bytes.Length; $i++) {
        $line += "0x{0:X2}, " -f $Bytes[$i]
        if (($i + 1) % 16 -eq 0) {
            [void]$SourceContent.AppendLine($line)
            $line = "    "
        }
    }
    if ($line.Trim()) {
        [void]$SourceContent.AppendLine($line)
    }
    [void]$SourceContent.AppendLine("};")
    [void]$SourceContent.AppendLine("")
}

# Build registry array
[void]$SourceContent.AppendLine("const dx12_shader_entry DX12_SHADER_REGISTRY[] = {")
foreach ($cso in $CsoFiles) {
    $ArrayName = "cso_$($cso.Name)"
    $tg = $cso.Threads
    [void]$SourceContent.AppendLine("    { `"$($cso.Name)`", ${ArrayName}_data, sizeof(${ArrayName}_data), $($tg.X), $($tg.Y), $($tg.Z) },")
}
[void]$SourceContent.AppendLine("};")
[void]$SourceContent.AppendLine("")
[void]$SourceContent.AppendLine("const size_t DX12_SHADER_COUNT = sizeof(DX12_SHADER_REGISTRY) / sizeof(DX12_SHADER_REGISTRY[0]);")

Set-Content -Path $SourceFile -Value $SourceContent.ToString() -Encoding UTF8
Write-Host "  Generated: dx12_shader_registry.cpp ($($CsoFiles.Count) shaders)" -ForegroundColor Green
Write-Host ""
Write-Host "Shader compilation complete!" -ForegroundColor Green
