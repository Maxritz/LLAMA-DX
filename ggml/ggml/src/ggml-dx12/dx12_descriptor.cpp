/*
 * dx12_descriptor.cpp
 * COMPONENT: 1 (Backend Core)
 * PURPOSE: Descriptor heaps, root signatures, PSO cache
 */

#include "dx12_descriptor.h"
#include <d3d12.h>
#include <d3dx12/d3dx12.h>
#include <d3dcompiler.h>
#include <cstring>

// ═══════════════════════════════════════════════════════════════════════════════
// Descriptor Heap
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_descriptor_heap::init() {
    if (!dev || !dev->device) return false;

    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = type;
    desc.NumDescriptors = capacity;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    desc.NodeMask = 0;

    HRESULT hr = dev->device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) return false;

    descriptor_size = dev->device->GetDescriptorHandleIncrementSize(type);
    cpu_start = heap->GetCPUDescriptorHandleForHeapStart();
    gpu_start = heap->GetGPUDescriptorHandleForHeapStart();
    current = 0;

    return true;
}

void dx12_descriptor_heap::reset() {
    current = 0;
}

D3D12_CPU_DESCRIPTOR_HANDLE dx12_descriptor_heap::allocate_cpu() {
    std::lock_guard<std::mutex> lock(mutex);
    D3D12_CPU_DESCRIPTOR_HANDLE handle = cpu_handle(current);
    current = (current + 1) % capacity;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE dx12_descriptor_heap::allocate_gpu() {
    std::lock_guard<std::mutex> lock(mutex);
    D3D12_GPU_DESCRIPTOR_HANDLE handle = gpu_handle(current);
    current = (current + 1) % capacity;
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE dx12_descriptor_heap::cpu_handle(uint32_t index) {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = cpu_start;
    handle.ptr += static_cast<SIZE_T>(index) * descriptor_size;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE dx12_descriptor_heap::gpu_handle(uint32_t index) {
    D3D12_GPU_DESCRIPTOR_HANDLE handle = gpu_start;
    handle.ptr += static_cast<SIZE_T>(index) * descriptor_size;
    return handle;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Root Signature Creation
// ═══════════════════════════════════════════════════════════════════════════════

static uint32_t dx12_root_sig_hash(dx12_root_signature_type type) {
    return static_cast<uint32_t>(type);
}

static ComPtr<ID3D12RootSignature> dx12_build_root_signature(dx12_device* dev,
                                                              dx12_root_signature_type type) {
    if (!dev || !dev->device) return nullptr;

    std::vector<CD3DX12_ROOT_PARAMETER1> params;
    std::vector<CD3DX12_DESCRIPTOR_RANGE1> ranges;

    switch (type) {
        case dx12_root_signature_type::simple_2in_1out: {
            // Param 0: CBV (constants)
            CD3DX12_ROOT_PARAMETER1 cbv_param;
            cbv_param.InitAsConstants(8, 0); // b0, 8 uints
            cbv_param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(cbv_param);

            // Param 1: SRV (input A) - via root descriptor
            CD3DX12_ROOT_PARAMETER1 srv0_param;
            srv0_param.InitAsShaderResourceView(0); // t0
            srv0_param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(srv0_param);

            // Param 2: SRV (input B) - via root descriptor
            CD3DX12_ROOT_PARAMETER1 srv1_param;
            srv1_param.InitAsShaderResourceView(1); // t1
            srv1_param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(srv1_param);

            // Param 3: UAV (output) - via root descriptor
            CD3DX12_ROOT_PARAMETER1 uav_param;
            uav_param.InitAsUnorderedAccessView(0); // u0
            uav_param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(uav_param);
            break;
        }

        case dx12_root_signature_type::gemm: {
            // Param 0: CBV with M, N, K, stride_A, stride_B, stride_C
            CD3DX12_ROOT_PARAMETER1 cbv_param;
            cbv_param.InitAsConstants(16, 0); // b0, 16 uints for GEMM params
            params.push_back(cbv_param);

            // Param 1: SRV matrix A
            CD3DX12_ROOT_PARAMETER1 srv_a;
            srv_a.InitAsShaderResourceView(0); // t0
            params.push_back(srv_a);

            // Param 2: SRV matrix B
            CD3DX12_ROOT_PARAMETER1 srv_b;
            srv_b.InitAsShaderResourceView(1); // t1
            params.push_back(srv_b);

            // Param 3: UAV result
            CD3DX12_ROOT_PARAMETER1 uav_c;
            uav_c.InitAsUnorderedAccessView(0); // u0
            params.push_back(uav_c);
            break;
        }

        case dx12_root_signature_type::reduction: {
            // Param 0: CBV dimensions
            CD3DX12_ROOT_PARAMETER1 cbv_param;
            cbv_param.InitAsConstants(8, 0);
            params.push_back(cbv_param);

            // Param 1: SRV input
            CD3DX12_ROOT_PARAMETER1 srv_param;
            srv_param.InitAsShaderResourceView(0);
            params.push_back(srv_param);

            // Param 2: UAV output
            CD3DX12_ROOT_PARAMETER1 uav_param;
            uav_param.InitAsUnorderedAccessView(0);
            params.push_back(uav_param);
            break;
        }

        case dx12_root_signature_type::dequant_gemm: {
            // Param 0: CBV with dimensions + quant params
            CD3DX12_ROOT_PARAMETER1 cbv_param;
            cbv_param.InitAsConstants(16, 0);
            params.push_back(cbv_param);

            // Param 1: SRV quantized weights
            CD3DX12_ROOT_PARAMETER1 srv_weights;
            srv_weights.InitAsShaderResourceView(0);
            params.push_back(srv_weights);

            // Param 2: SRV scales (quant metadata)
            CD3DX12_ROOT_PARAMETER1 srv_scales;
            srv_scales.InitAsShaderResourceView(1);
            params.push_back(srv_scales);

            // Param 3: SRV input activations
            CD3DX12_ROOT_PARAMETER1 srv_input;
            srv_input.InitAsShaderResourceView(2);
            params.push_back(srv_input);

            // Param 4: UAV output
            CD3DX12_ROOT_PARAMETER1 uav_out;
            uav_out.InitAsUnorderedAccessView(0);
            params.push_back(uav_out);
            break;
        }

        case dx12_root_signature_type::attention: {
            // Param 0: CBV attention params (seq_len, head_dim, num_heads, scale)
            CD3DX12_ROOT_PARAMETER1 cbv_param;
            cbv_param.InitAsConstants(16, 0);
            params.push_back(cbv_param);

            // Param 1: SRV Q
            CD3DX12_ROOT_PARAMETER1 srv_q;
            srv_q.InitAsShaderResourceView(0);
            params.push_back(srv_q);

            // Param 2: SRV K
            CD3DX12_ROOT_PARAMETER1 srv_k;
            srv_k.InitAsShaderResourceView(1);
            params.push_back(srv_k);

            // Param 3: SRV V
            CD3DX12_ROOT_PARAMETER1 srv_v;
            srv_v.InitAsShaderResourceView(2);
            params.push_back(srv_v);

            // Param 4: UAV output
            CD3DX12_ROOT_PARAMETER1 uav_out;
            uav_out.InitAsUnorderedAccessView(0);
            params.push_back(uav_out);
            break;
        }

        case dx12_root_signature_type::custom: {
            // Minimal: just CBV + 2 SRV + 1 UAV
            CD3DX12_ROOT_PARAMETER1 cbv;
            cbv.InitAsConstants(8, 0);
            params.push_back(cbv);

            CD3DX12_ROOT_PARAMETER1 srv0;
            srv0.InitAsShaderResourceView(0);
            params.push_back(srv0);

            CD3DX12_ROOT_PARAMETER1 srv1;
            srv1.InitAsShaderResourceView(1);
            params.push_back(srv1);

            CD3DX12_ROOT_PARAMETER1 uav;
            uav.InitAsUnorderedAccessView(0);
            params.push_back(uav);
            break;
        }
    }

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC sig_desc;
    sig_desc.Init_1_1(
        static_cast<UINT>(params.size()), params.data(),
        0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> sig_blob;
    ComPtr<ID3DBlob> error_blob;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&sig_desc, &sig_blob, &error_blob);
    if (FAILED(hr)) {
        if (error_blob) {
            dx12_log(DX12_LOG_ERROR, "Root signature serialization failed: %s",
                static_cast<char*>(error_blob->GetBufferPointer()));
        }
        return nullptr;
    }

    ComPtr<ID3D12RootSignature> root_sig;
    hr = dev->device->CreateRootSignature(
        0,
        sig_blob->GetBufferPointer(),
        sig_blob->GetBufferSize(),
        IID_PPV_ARGS(&root_sig));

    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "CreateRootSignature failed: 0x%08X", hr);
        return nullptr;
    }

    return root_sig;
}

ID3D12RootSignature* dx12_create_root_signature(dx12_device* dev,
                                                 dx12_root_signature_type type) {
    auto sig = dx12_build_root_signature(dev, type);
    return sig.Get(); // Caller must AddRef if holding
}

// ═══════════════════════════════════════════════════════════════════════════════
// Root Signature Cache
// ═══════════════════════════════════════════════════════════════════════════════

ID3D12RootSignature* dx12_root_signature_cache::get_or_create(dx12_root_signature_type type) {
    uint32_t hash = dx12_root_sig_hash(type);

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(hash);
    if (it != cache.end()) {
        return it->second.Get();
    }

    auto sig = dx12_build_root_signature(dev, type);
    if (!sig) return nullptr;

    ID3D12RootSignature* raw = sig.Get();
    cache[hash] = std::move(sig);
    return raw;
}

void dx12_root_signature_cache::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    cache.clear();
}

// ═══════════════════════════════════════════════════════════════════════════════
// PSO Cache
// ═══════════════════════════════════════════════════════════════════════════════

dx12_pso* dx12_pso_cache::get_or_create(const char* shader_name,
                                         const void* cso_data, size_t cso_size,
                                         dx12_root_signature_type sig_type,
                                         std::array<uint32_t, 3> thread_group_size) {
    if (!shader_name || !cso_data || cso_size == 0) return nullptr;

    std::string key = shader_name;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second.get();
    }

    // Get root signature
    dx12_root_signature_cache sig_cache(dev);
    ID3D12RootSignature* root_sig = sig_cache.get_or_create(sig_type);
    if (!root_sig) return nullptr;

    // Create PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.CS.pShaderBytecode = cso_data;
    pso_desc.CS.BytecodeLength = cso_size;
    pso_desc.pRootSignature = root_sig;

    ComPtr<ID3D12PipelineState> pipeline;
    HRESULT hr = dev->device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pipeline));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "CreateComputePipelineState for %s failed: 0x%08X",
            shader_name, hr);
        return nullptr;
    }

    auto pso = std::make_unique<dx12_pso>();
    pso->pipeline_state = pipeline;
    pso->root_signature = root_sig;
    pso->sig_type = sig_type;
    pso->shader_name = key;
    pso->thread_group_size = thread_group_size;

    dx12_pso* result = pso.get();
    cache[key] = std::move(pso);
    return result;
}

void dx12_pso_cache::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    cache.clear();
}

// ═══════════════════════════════════════════════════════════════════════════════
// SRV / UAV Helpers
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_create_buffer_srv(dx12_device* dev,
                            D3D12_CPU_DESCRIPTOR_HANDLE handle,
                            ID3D12Resource* resource,
                            DXGI_FORMAT format,
                            uint32_t num_elements) {
    if (!dev || !resource) return;

    if (num_elements == 0) {
        D3D12_RESOURCE_DESC desc = resource->GetDesc();
        num_elements = static_cast<uint32_t>(desc.Width / 4); // assume 4-byte elements
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Format = format;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Buffer.NumElements = num_elements;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    dev->device->CreateShaderResourceView(resource, &srv_desc, handle);
}

void dx12_create_buffer_uav(dx12_device* dev,
                            D3D12_CPU_DESCRIPTOR_HANDLE handle,
                            ID3D12Resource* resource,
                            DXGI_FORMAT format,
                            uint32_t num_elements) {
    if (!dev || !resource) return;

    if (num_elements == 0) {
        D3D12_RESOURCE_DESC desc = resource->GetDesc();
        num_elements = static_cast<uint32_t>(desc.Width / 4);
    }

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
    uav_desc.Format = format;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.NumElements = num_elements;
    uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    dev->device->CreateUnorderedAccessView(resource, nullptr, &uav_desc, handle);
}
