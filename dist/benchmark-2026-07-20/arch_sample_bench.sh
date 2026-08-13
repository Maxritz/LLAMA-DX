#!/bin/bash
set -uo pipefail
cd "/e/DXllama/OptimiseDX"
BIN="./build_dx12/bin/Release/llama-bench.exe"
GGUF="/e/OLLAMA-Models/GGUF"
OUT="dist/benchmark-2026-07-20/arch_sample"
mkdir -p "$OUT"
RESULTS="$OUT/results.txt"
: > "$RESULTS"

run() {
    local label="$1" model="$2" dev="$3" extra="$4"
    local log="$OUT/${label}--${dev}.log"
    echo "=== $label / $dev ===" | tee -a "$RESULTS"
    "$BIN" -m "$GGUF/$model" -dev "$dev" -p 512 -n 128 -r 1 -o md $extra > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        grep -E '\| *pp512 *\||\| *tg128 *\|' "$log" | tee -a "$RESULTS"
    else
        echo "FAILED rc=$rc - see $log" | tee -a "$RESULTS"
        tail -5 "$log" | tee -a "$RESULTS"
    fi
    echo "" | tee -a "$RESULTS"
}

# --- 1 of each architecture (all fit within the DX12 VRAM ceiling at ngl99, no CPU split needed) ---
run "llama"    "Llama-3-8B-16K-Q8_0.gguf"                                  "Vulkan0" "-ngl 99"
run "llama"    "Llama-3-8B-16K-Q8_0.gguf"                                  "DX120"   "-ngl 99"

run "gemma4"   "gemma4-v2-Q6_K.gguf"                                       "Vulkan0" "-ngl 99"
run "gemma4"   "gemma4-v2-Q6_K.gguf"                                       "DX120"   "-ngl 99"

run "qwen"     "Qwen3-4B-Q8_0.gguf"                                        "Vulkan0" "-ngl 99"
run "qwen"     "Qwen3-4B-Q8_0.gguf"                                        "DX120"   "-ngl 99"

run "deepseek" "DeepSeek-Coder-V2-Lite-Instruct.Q4_K_M.gguf"               "Vulkan0" "-ngl 99"
run "deepseek" "DeepSeek-Coder-V2-Lite-Instruct.Q4_K_M.gguf"               "DX120"   "-ngl 99"

# laguna exceeds the DX12 VRAM ceiling at ngl99; needs CPU-split to fit, which
# is the known-crashing DX12 path (root-caused earlier: fake 0x1000 sentinel
# pointer dereferenced by CPU backend). Vulkan-only for this run by design.
run "laguna"   "Laguna-XS.2-IQ4_XS.gguf"                                   "Vulkan0" "-fitt 1024"

# --- 2 large models (also exceed the DX12 ceiling, same known-crash path skipped on DX12) ---
run "large1_Qwen3.6-27B-AEON" "Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-MTP-i1.Q5_K_M.gguf" "Vulkan0" "-fitt 1024"
run "large2_Qwen3.5-35B-A3B"  "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"                               "Vulkan0" "-fitt 1024"

echo "DONE" | tee -a "$RESULTS"
