#include "models.h"

void llama_model_laguna::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // swa_type is assigned below, once the SWA-layer pattern is known.

    // Partial rotary is data-driven: rope.dimension_count (global layers) and
    // rope.dimension_count.swa (SWA layers) are written by the converter from each rope
    // config's partial_rotary_factor. Laguna-M uses full rotary (1.0) on every layer;
    // Laguna-XS uses half rotary (0.5) on global layers and full rotary on SWA layers.

    // MoE + SWA parameters
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,          hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,                hparams.n_expert_shared, false);

    // Laguna uses sigmoid gating by default
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,  hparams.n_swa);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,        hparams.rope_freq_base_train_swa, false);
    const bool has_swa_pattern =
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer(), false);

    // The fork converter writes sliding_window_pattern (bool array). The official Laguna
    // GGUFs ship attention.layer_types strings instead; derive the SWA mask from those when
    // the bool pattern is absent.
    if (!has_swa_pattern) {
        std::vector<std::string> layer_types;
        if (ml.get_arr(LLM_KV_ATTENTION_LAYER_TYPES, layer_types, false)) {
            for (uint32_t il = 0; il < hparams.n_layer() && il < (uint32_t) layer_types.size(); ++il) {
                hparams.is_swa_impl[il] = layer_types[il] == "sliding_attention";
            }
        }
    }

    // create_memory() requires swa_type != NONE iff is_swa_any(): the iswa KV cache is
    // only allocated when at least one layer is sliding-window. Laguna-M uses full
    // attention on every layer -> plain KV cache; Laguna-XS (mixed SWA) -> iswa cache.
    hparams.swa_type = hparams.is_swa_any() ? LLAMA_SWA_TYPE_STANDARD : LLAMA_SWA_TYPE_NONE;

    // Attention output gate mode: per-head (Laguna-XS) vs per-element (Laguna-M).
    ml.get_key(LLM_KV_ATTENTION_GATE_PER_HEAD, hparams.attn_gate_per_head, false);

    // Global layers use YaRN; load from GGUF (written by converter from full_attention rope config)
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR,  hparams.yarn_ext_factor,  false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, hparams.yarn_attn_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST,   hparams.yarn_beta_fast,   false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   hparams.yarn_beta_slow,   false);

    // SWA layers use plain RoPE — no YaRN
    hparams.yarn_ext_factor_swa  = 0.0f;
    hparams.yarn_attn_factor_swa = 1.0f;
    hparams.yarn_beta_fast_swa   = 32.0f;
    hparams.yarn_beta_slow_swa   =  1.0f;

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_33B_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_laguna::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    uint32_t n_rot_max = 0;
    for (int i = 0; i < n_layer; ++i) {
        n_rot_max = std::max(n_rot_max, hparams.n_rot(i));
    }
    if (n_rot_max == 0) {
        n_rot_max = n_rot;
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const uint32_t n_head_l      = hparams.n_head(i);
        const uint32_t n_embd_k_gqa  = hparams.n_embd_k_gqa(i);
        const uint32_t n_embd_v_gqa  = hparams.n_embd_v_gqa(i);

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);

        // optional rope factors
        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        } else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head_l, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_v * n_head_l, n_embd}, 0);

        // attention output gate: per-element on Laguna-M (g_proj output = n_head*head_dim),
        // per-head on Laguna-XS (g_proj output = n_head, broadcast across head_dim).
        const uint32_t n_attn_gate_out = hparams.attn_gate_per_head ? n_head_l : (n_head_l * n_embd_head_v);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_attn_gate_out}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        // dense MLP (leading dense blocks)
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);

        // MoE routed experts + selection bias
        const int64_t n_ff_exp = hparams.n_ff_exp;
        layer.ffn_gate_inp      = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_exps     = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp,   n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_exps       = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp,   n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_exp_probs_b   = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

        // shared expert MLP
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_laguna::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_laguna::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos     = build_inp_pos();
    // The iswa attention input requires swa_type != NONE (build_attn_inp_kv_iswa asserts
    // it); Laguna-M (all full attention) uses the plain KV input instead.
    const bool has_swa = hparams.is_swa_any();
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = has_swa ? build_attn_inp_kv_iswa() : nullptr;
    llm_graph_input_attn_kv      * inp_attn_full = has_swa ? nullptr : build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        const uint32_t n_head_l    = hparams.n_head(il);
        const uint32_t n_head_kv_l = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        cur = inpL;

        // self-attention
        {
            cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head_l,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv_l, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv_l, n_tokens);

            if (model.layers[il].attn_q_norm) {
                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);
            }
            if (model.layers[il].attn_k_norm) {
                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);
            }

            // RoPE (partial rotary: global layers half-rotated, SWA layers full-rotated)
            const bool is_swa = hparams.is_swa(il);
            ggml_tensor * rope_factors = is_swa ? nullptr : model.get_rope_factors(cparams, il);
            const int64_t n_rot_l = hparams.n_rot(il);
            const float ext_factor_l  = is_swa ? cparams.yarn_ext_factor_swa  : ext_factor;
            const float attn_factor_l = is_swa ? cparams.yarn_attn_factor_swa : attn_factor;
            const float beta_fast_l   = is_swa ? cparams.yarn_beta_fast_swa   : beta_fast;
            const float beta_slow_l   = is_swa ? cparams.yarn_beta_slow_swa   : beta_slow;
            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l
            );
            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l
            );
            cb(Qcur, "Qcur_pos", il);
            cb(Kcur, "Kcur_pos", il);

            const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));
            ggml_tensor * attn_out = has_swa
                    ? build_attn(inp_attn_iswa, nullptr, nullptr, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il)
                    : build_attn(inp_attn_full, nullptr, nullptr, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(attn_out, "attn_out", il);

            // attention output gate (per-element on Laguna-M, per-head on Laguna-XS),
            // applied as attn_out *= softplus(g_proj(x)) before o_proj.
            if (model.layers[il].wqkv_gate) {
                ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur);
                cb(gate, "attn_gate", il);

                gate = ggml_softplus(ctx0, gate);
                cb(gate, "attn_gate_softplus", il);

                if (hparams.attn_gate_per_head) {
                    // per-head: gate is [n_head_l] per token, broadcast across head_dim
                    ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, attn_out, n_embd_head_v, n_head_l, n_tokens);
                    ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,       1,          n_head_l, n_tokens);
                    cb(gate_3d, "attn_gate_3d", il);

                    attn_3d = ggml_mul(ctx0, attn_3d, gate_3d);
                    cb(attn_3d, "attn_gated_3d", il);

                    attn_out = ggml_reshape_2d(ctx0, attn_3d, n_embd_head_v * n_head_l, n_tokens);
                } else {
                    // per-element: gate is [n_embd_head_v * n_head_l] per token, element-wise
                    attn_out = ggml_mul(ctx0, attn_out, gate);
                }
                cb(attn_out, "attn_gated", il);
            }

            cur = build_lora_mm(model.layers[il].wo, attn_out, model.layers[il].wo_s);
            cb(cur, "attn_proj", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward: dense lead layers use MLP, rest use MoE
        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            // dense MLP
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   nullptr,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, nullptr,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE routed experts
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert MLP. Gate on tensor presence, not expert_shared_count: the
            // official Laguna GGUFs omit that KV (n_expert_shared stays 0) but still ship
            // the shexp tensors, which must be applied on every MoE layer.
            if (model.layers[il].ffn_gate_shexp) {
                ggml_tensor * sh_out = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                        model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                        model.layers[il].ffn_down_shexp, nullptr, nullptr,
                        nullptr,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(sh_out, "ffn_shared_out", il);

                moe_out = ggml_add(ctx0, moe_out, sh_out);
            }

            cur = moe_out;
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
