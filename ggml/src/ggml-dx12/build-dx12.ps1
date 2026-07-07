# build-dx12.ps1
# COMPONENT: 8 (Integration)
# PURPOSE: Full build script for ggml-backend-dx12
#
# USAGE:
#   .\build-dx12.ps1                     # Default: Release build
#   .\build-dx12.ps1 -Config Debug       # Debug build with validation layers
#   .\build-dx12.ps1 -Clean              # Clean and rebuild
#   .\build-dx12.ps1 -Test               # Build and run tests
#   .\build-dx12.ps1 -ShaderOnly         # Compile shaders only
#
# PREREQUISITES:
#   - Windows 11 23H2+
#   - Visual Studio 2022 17.8+ (with C++ workload)
#   - Windows SDK 22621+ (for dxc.exe)
#   - CMake 3.25+ and Ninja
#   - DirectX Agility SDK (auto-fetched via NuGet)

param(
    [string]$Config = "Release",
    [switch]$Clean,
    [switch]$Test,
    [switch]$ShaderOnly,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$buildDir = Join-Path $scriptDir "build"

# ═══════════════════════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ggml-backend-dx12 Build Script" -ForegroundColor Cyan
Write-Host "  Config: $Config" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ═══════════════════════════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════════════════════════
if ($Clean) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
}

# ═══════════════════════════════════════════════════════════════════════════════
# Compile Shaders
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host "[1/4] Compiling HLSL shaders..." -ForegroundColor Cyan
& "$scriptDir\shaders\compile_shaders.ps1" -Configuration $Config
if ($LASTEXITCODE -ne 0) {
    Write-Error "Shader compilation failed!"
    exit 1
}

if ($ShaderOnly) {
    Write-Host "Shader compilation complete!" -ForegroundColor Green
    exit 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# CMake Configure
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "[2/4] Configuring with CMake..." -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$cmakeArgs = @(
    "-S", $scriptDir,
    "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DGGML_DX12=ON",
    "-DGGML_DX12_BUILD_TESTS=$(if($Test){'ON'}else{'OFF'})"
)

if ($Verbose) { $cmakeArgs += "-DCMAKE_VERBOSE_MAKEFILE=ON" }

& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed!"
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "[3/4] Building..." -ForegroundColor Cyan
& cmake --build $buildDir --config $Config
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed!"
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════
if ($Test) {
    Write-Host ""
    Write-Host "[4/4] Running tests..." -ForegroundColor Cyan
    & ctest --test-dir $buildDir -C $Config --output-on-failure
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Some tests failed!"
    }
} else {
    Write-Host ""
    Write-Host "[4/4] Skipping tests (use -Test to run)" -ForegroundColor Gray
}

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "  Output: $buildDir" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "To run tests: .\build-dx12.ps1 -Test" -ForegroundColor Gray
Write-Host "To install:   cmake --install $buildDir --prefix <path>" -ForegroundColor Gray
Write-Host ""
