param([string]$BenchExe,[string]$CompExe,[string[]]$Models,[string]$OutDir,[string]$BinDir)
foreach ($m in $Models) {
    $name = [IO.Path]::GetFileNameWithoutExtension($m)
    Write-Host "=== $name ==="
    & $BenchExe -m $m -ngl 99 -b 512 -p 128 -n 64 -fa off -r 2 -o md 2>$null | Select-String -Pattern "pp128|tg64" | ForEach-Object { $_ } | Out-File -Append "$OutDir\$name.bench.md"
    & $CompExe -m $m -ngl 99 -p "Once" -n 1 --perf -no-warmup 2>&1 | Select-String -Pattern "prompt eval time|eval time|pp512|sampling|total time|tokens" | ForEach-Object { $_ } | Out-File -Append "$OutDir\$name.ttft.txt"
}
Write-Host "DONE-CAMPAIGN"
