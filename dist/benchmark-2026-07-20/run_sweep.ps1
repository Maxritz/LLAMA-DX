$ErrorActionPreference = "Continue"
$binDir     = "E:\DXllama\OptimiseDX\build_dx12\bin\Release"
$modelsRoot = "E:\OLLAMA-Models\GGUF"
$outDir     = "E:\DXllama\OptimiseDX\dist\benchmark-2026-07-20"
$logDir     = "$outDir\logs"
$resultsCsv = "$outDir\results.csv"
$commandsMd = "$outDir\test-commands.md"
$doneFile   = "$outDir\DONE.txt"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Remove-Item -Force $doneFile -ErrorAction SilentlyContinue

if (-not (Test-Path $resultsCsv)) {
    "Category,Model,SizeMB,Backend,Config,NGL,MMAP,CtkCtv,PP_ts,TG_ts,TotalSec,Status,LogFile" | Out-File -FilePath $resultsCsv -Encoding utf8
}
if (-not (Test-Path $commandsMd)) {
    "# Exact commands run (chronological)`n" | Out-File -FilePath $commandsMd -Encoding utf8
}

# Build a set of already-completed (Model|Backend|Config) keys so a restart never redoes finished work.
$alreadyDone = New-Object System.Collections.Generic.HashSet[string]
if (Test-Path $resultsCsv) {
    Import-Csv $resultsCsv | ForEach-Object {
        if ($_.Status -eq "OK" -or $_.Status -match "^SKIPPED") {
            [void]$alreadyDone.Add("$($_.Model)|$($_.Backend)|$($_.Config)")
        }
    }
}

# Ascending by size: fast/small models first so problems surface early, big ones last.
$models = @(
    @{Name="LLAMAmobile";                          File="LLAMAmobile.gguf";                          Cat="TINY"; SizeMB=771;  Arch="dense"}
    @{Name="MiniCPM5-1B-Q8_0";                     File="MiniCPM5-1B-Q8_0.gguf";                     Cat="TINY"; SizeMB=1101; Arch="dense"}
    @{Name="Llama-3.2-1B-Instruct-Q8_0";           File="Llama-3.2-1B-Instruct-Q8_0.gguf";           Cat="TINY"; SizeMB=1260; Arch="dense"}
    @{Name="qwen3-1.7b-coder-distilled-sft-Q8_0";  File="qwen3-1.7b-coder-distilled-sft-Q8_0.gguf";  Cat="TINY"; SizeMB=2065; Arch="dense"}
    @{Name="VibeThinker-3B.Q6_K";                  File="VibeThinker-3B.Q6_K.gguf";                  Cat="TINY"; SizeMB=2421; Arch="dense"}
    @{Name="Samastam-2.5B-Q8_0";                   File="Samastam-2.5B-Q8_0.gguf";                   Cat="TINY"; SizeMB=2561; Arch="dense"}
    @{Name="GIGABATEMAN-7B.Q2_K";                  File="GIGABATEMAN-7B.Q2_K.gguf";                  Cat="TINY"; SizeMB=2594; Arch="dense"}

    @{Name="VibeThinker-3B.Q8_0";                       File="VibeThinker-3B.Q8_0.gguf";                       Cat="SMALL"; SizeMB=3134; Arch="dense"}
    @{Name="Qwen3-4B-Instruct-2507-Q6_K";                File="Qwen3-4B-Instruct-2507-Q6_K.gguf";                Cat="SMALL"; SizeMB=3154; Arch="dense"}
    @{Name="qwen2.5-coder-3b-instruct-q8_0";             File="qwen2.5-coder-3b-instruct-q8_0.gguf";             Cat="SMALL"; SizeMB=3449; Arch="dense"}
    @{Name="Bonsai-27B-Q1_0";                            File="Bonsai-27B-Q1_0.gguf";                            Cat="SMALL"; SizeMB=3628; Arch="dense"}
    @{Name="meeTARA-phi-3-mini-4k-instruct-spark-Q8_0";  File="meeTARA-phi-3-mini-4k-instruct-spark-Q8_0.gguf";  Cat="SMALL"; SizeMB=3874; Arch="dense"}
    @{Name="qwen3-1.7b-stem-proof-f16";                  File="qwen3-1.7b-stem-proof-f16.gguf";                  Cat="SMALL"; SizeMB=3882; Arch="dense"}
    @{Name="qwen3-1.7b-coder-distilled-sft-f16";         File="qwen3-1.7b-coder-distilled-sft-f16.gguf";         Cat="SMALL"; SizeMB=3882; Arch="dense"}

    @{Name="gemma-4-E4B-it-Q4_K_M";               File="gemma-4-E4B-it-Q4_K_M.gguf";               Cat="OTHER"; SizeMB=5089; Arch="dense"}
    @{Name="Q3.5-9B-GLM-5.1-DA.Q4_K_S";           File="Q3.5-9B-GLM-5.1-DA.Q4_K_S.gguf";           Cat="OTHER"; SizeMB=5104; Arch="dense"}
    @{Name="Worldsim-Hermes-7B.Q6_K";             File="Worldsim-Hermes-7B.Q6_K.gguf";             Cat="OTHER"; SizeMB=5667; Arch="dense"}
    @{Name="VibeThinker-3B.f16";                  File="VibeThinker-3B.f16.gguf";                  Cat="OTHER"; SizeMB=5893; Arch="dense"}
    @{Name="marco-o1-q6_k";                       File="marco-o1-q6_k.gguf";                       Cat="OTHER"; SizeMB=5965; Arch="dense"}
    @{Name="rocmforge-7b.Q6_K";                   File="rocmforge-7b.Q6_K.gguf";                   Cat="OTHER"; SizeMB=5965; Arch="dense"}
    @{Name="Ternary-Bonsai-27B-Q2_0";             File="Ternary-Bonsai-27B-Q2_0.gguf";             Cat="OTHER"; SizeMB=6834; Arch="dense"}
    @{Name="Phi-4-reasoning-plus-Q3_K_M";         File="Phi-4-reasoning-plus-Q3_K_M.gguf";         Cat="OTHER"; SizeMB=7023; Arch="dense"}
    @{Name="gemma4-coding-Q4_K_M";                File="gemma4-coding-Q4_K_M.gguf";                Cat="OTHER"; SizeMB=7040; Arch="dense"}
    @{Name="Gemma-4-12B-OBLITERATED.i1-Q4_K_M";   File="Gemma-4-12B-OBLITERATED.i1-Q4_K_M.gguf";   Cat="OTHER"; SizeMB=7040; Arch="dense"}
    @{Name="GIGABATEMAN-7B.Q8_0";                 File="GIGABATEMAN-7B.Q8_0.gguf";                 Cat="OTHER"; SizeMB=7340; Arch="dense"}
    @{Name="gemma4-toolcall-v02-Q8_0";            File="gemma4-toolcall-v02-Q8_0.gguf";            Cat="OTHER"; SizeMB=7579; Arch="dense"}
    @{Name="rocmforge-7b.Q8_0";                   File="rocmforge-7b.Q8_0.gguf";                   Cat="OTHER"; SizeMB=7724; Arch="dense"}

    @{Name="Llama-3-8B-16K-Q8_0";               File="Llama-3-8B-16K-Q8_0.gguf";               Cat="MEDIUM"; SizeMB=8146; Arch="dense"; DS=$true; KV=$true}
    @{Name="ornith-9b-Q8_0";                    File="ornith-9b-Q8_0.gguf";                    Cat="MEDIUM"; SizeMB=9087; Arch="dense"; DS=$true; KV=$true}
    @{Name="Qwythos-9B-Claude-Mythos-5-1M-Q8_0";File="Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf";Cat="MEDIUM"; SizeMB=9087; Arch="dense"; DS=$true; KV=$true}
    @{Name="gemma4-v2-Q6_K";                    File="gemma4-v2-Q6_K.gguf";                    Cat="MEDIUM"; SizeMB=9333; Arch="dense"; DS=$true; KV=$true}

    @{Name="Qwable-27b_Q4_K_M";    File="Qwable-27b_Q4_K_M.gguf";    Cat="VRAM16"; SizeMB=15781; Arch="dense"; DS=$true}
    @{Name="carwin-Q4_K_M";        File="carwin-Q4_K_M.gguf";        Cat="VRAM16"; SizeMB=16057; Arch="dense"; DS=$true}

    @{Name="Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-MTP-i1.Q5_K_M";        File="Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-MTP-i1.Q5_K_M.gguf";        Cat="LARGE";  SizeMB=18631; Arch="dense"; DS=$true}
    @{Name="L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q8_0";   File="L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q8_0.gguf";   Cat="LARGE";  SizeMB=18660; Arch="moe"}
    @{Name="Qwen3.5-35B-A3B-UD-Q4_K_XL";                                     File="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";                                     Cat="LARGE";  SizeMB=18776; Arch="moe"}
    @{Name="qwable-v1-mxfp4_moe";                                            File="qwable-v1-mxfp4_moe.gguf";                                            Cat="LARGE";  SizeMB=19323; Arch="moe"}
    @{Name="ornith-35b-Q8_0";                                                File="ornith-35b-Q8_0.gguf";                                                Cat="LARGE";  SizeMB=35194; Arch="moe"}
    @{Name="gpt-oss-120b-Q8_0";                                              File="gpt-oss-120b-Q8_0.gguf";                                              Cat="LARGE";  SizeMB=60451; Arch="moe"}

    # EXCLUDED: laguna-xs2-Q4_K_M.gguf, Laguna-XS.2-IQ4_XS.gguf - "unknown model architecture: 'laguna'"
    # EXCLUDED: ornith-1.0-35B-Q3_0_ROCMFPX.gguf - "failed to load model" on both backends; the
    #   "Q3_0_ROCMFPX" quant type name in the filename suggests a non-standard/custom quant this
    #   llama.cpp build doesn't support (same class of gap as laguna, different axis - quant type
    #   vs architecture).
)

$vramBudgetMB = 15000

function Sanitize($s) { return ($s -replace '[^a-zA-Z0-9_.-]', '_') }

function Run-One {
    param($model, $dev, [string[]]$extraArgs, [string]$configLabel, [string]$mmap, [string]$ctk)

    $key = "$($model.Name)|$dev|$configLabel"
    if ($alreadyDone.Contains($key)) {
        Write-Host "[$($model.Cat)] $($model.Name) | $dev | $configLabel -> already done, skipping"
        return @{Status="OK"; Ngl=""}
    }

    $safeName = Sanitize $model.Name
    $safeDev  = Sanitize $dev
    $safeCfg  = Sanitize $configLabel
    $logFile  = "$logDir\$safeName--$safeDev--$safeCfg.log"

    if ($dev -eq "DX120") { $env:DX12_ENABLE_FA = "1" } else { Remove-Item Env:\DX12_ENABLE_FA -ErrorAction SilentlyContinue }

    $modelPath = "$modelsRoot\$($model.File)"
    $argList = @("-m", $modelPath, "-dev", $dev, "-p", "512", "-n", "128", "-r", "2", "-fa", "auto", "-o", "md") + $extraArgs

    $cmdLine = "llama-bench.exe " + ($argList -join " ")
    if ($dev -eq "DX120") { $cmdLine = "DX12_ENABLE_FA=1 " + $cmdLine }
    Add-Content -Path $commandsMd -Value ("``````" + "`n" + $cmdLine + "`n" + "``````" + "`n(log: logs\$(Split-Path $logFile -Leaf))`n")

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath "$binDir\llama-bench.exe" -ArgumentList $argList -WorkingDirectory $binDir `
        -RedirectStandardOutput "$logFile.out" -RedirectStandardError "$logFile.err" -NoNewWindow -PassThru -Wait
    $sw.Stop()
    $exitCode = $proc.ExitCode

    # Read raw bytes as UTF-8 explicitly - Get-Content's default encoding on this PowerShell
    # mangles the multi-byte '+/-' character llama-bench prints, which silently broke every
    # regex match in the first version of this script.
    $rawOut = ""
    if (Test-Path "$logFile.out") { $rawOut = [System.IO.File]::ReadAllText("$logFile.out", [System.Text.Encoding]::UTF8) }
    $rawErr = ""
    if (Test-Path "$logFile.err") { $rawErr = [System.IO.File]::ReadAllText("$logFile.err", [System.Text.Encoding]::UTF8) }
    $combined = $rawOut + "`n--- STDERR ---`n" + $rawErr
    [System.IO.File]::WriteAllText($logFile, $combined, [System.Text.Encoding]::UTF8)
    Remove-Item "$logFile.out","$logFile.err" -ErrorAction SilentlyContinue

    $status = "OK"
    $ppTs = ""; $tgTs = ""
    if ($exitCode -ne 0) {
        $status = "CRASH(exit=$exitCode)"
    } else {
        if ($rawOut -match '\|\s*pp512\s*\|\s*([0-9.]+)\s*\xb1') { $ppTs = $matches[1] }
        if ($rawOut -match '\|\s*tg128\s*\|\s*([0-9.]+)\s*\xb1') { $tgTs = $matches[1] }
        if ([string]::IsNullOrEmpty($ppTs) -and [string]::IsNullOrEmpty($tgTs)) { $status = "NO_RESULT" }
    }

    $ngl = ""
    for ($i = 0; $i -lt $extraArgs.Count; $i++) { if ($extraArgs[$i] -eq "-ngl") { $ngl = $extraArgs[$i+1] } }

    $secs = [math]::Round($sw.Elapsed.TotalSeconds,1)
    $row = @($model.Cat, $model.Name, $model.SizeMB, $dev, $configLabel, $ngl, $mmap, $ctk, $ppTs, $tgTs, $secs, $status, (Split-Path $logFile -Leaf)) -join ","
    Add-Content -Path $resultsCsv -Value $row
    Write-Host "[$($model.Cat)] $($model.Name) | $dev | $configLabel -> $status pp=$ppTs tg=$tgTs ($($secs)s)"
    if ($status -eq "OK" -or $status -match "^SKIPPED") { [void]$alreadyDone.Add($key) }
    return @{Status=$status; Ngl=$ngl}
}

foreach ($model in $models) {
    $fitsFull = $model.SizeMB -le $vramBudgetMB

    # ---- Vulkan0 ----
    if ($fitsFull) {
        Run-One -model $model -dev "Vulkan0" -extraArgs @("-ngl","99") -configLabel "standard" -mmap "1" -ctk "f16" | Out-Null
    } else {
        Run-One -model $model -dev "Vulkan0" -extraArgs @("-fitt","1024") -configLabel "fitt" -mmap "1" -ctk "f16" | Out-Null
    }

    # ---- DX120 ----
    $workingNgl = $null
    if ($fitsFull) {
        Run-One -model $model -dev "DX120" -extraArgs @("-ngl","99") -configLabel "standard" -mmap "1" -ctk "f16" | Out-Null
        $workingNgl = "99"
    } elseif ($model.Arch -eq "moe") {
        $key = "$($model.Name)|DX120|standard"
        if (-not $alreadyDone.Contains($key)) {
            $row = @($model.Cat, $model.Name, $model.SizeMB, "DX120", "standard", "", "1", "f16", "", "", "0", "SKIPPED(known-moe-offload-bug)", "") -join ","
            Add-Content -Path $resultsCsv -Value $row
            Write-Host "[$($model.Cat)] $($model.Name) | DX120 -> SKIPPED (known MoE CPU-offload crash, see KNOWN-ISSUE-dx12-moe-cpu-offload-crash.md)"
            [void]$alreadyDone.Add($key)
        }
    } else {
        foreach ($n in @(60,48,40,32,24,16)) {
            $r = Run-One -model $model -dev "DX120" -extraArgs @("-ngl","$n") -configLabel "ngl$n" -mmap "1" -ctk "f16"
            if ($r.Status -eq "OK") { $workingNgl = "$n"; break }
        }
        if (-not $workingNgl) {
            Write-Host "[$($model.Cat)] $($model.Name) | DX120 -> all manual -ngl retries failed"
        }
    }

    # ---- DirectStorage comparison (DX120 only, DS-flagged models) ----
    if ($model.DS -and $workingNgl) {
        Run-One -model $model -dev "DX120" -extraArgs @("-ngl",$workingNgl,"-mmp","0") -configLabel "directstorage" -mmap "0" -ctk "f16" | Out-Null
    }

    # ---- KV cache quantization comparison (Vulkan only, KV-flagged models) ----
    if ($model.KV) {
        Run-One -model $model -dev "Vulkan0" -extraArgs @("-ngl","99","-ctk","q4_0","-ctv","q4_0") -configLabel "kvq4_0" -mmap "1" -ctk "q4_0" | Out-Null
    }
}

"DONE $(Get-Date -Format o)" | Out-File -FilePath $doneFile -Encoding utf8
Write-Host "=== SWEEP COMPLETE ==="
