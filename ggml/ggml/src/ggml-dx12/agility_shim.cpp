#include <windows.h>

extern "C" {
    __declspec(dllexport) extern const UINT D3D12SDKVersion = 721;
    __declspec(dllexport) extern const char* D3D12SDKPath = "E:\\DXllama\\agility_temp\\build\\D3D12\\";
}

BOOL APIENTRY DllMain(HMODULE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(GetModuleHandle(nullptr));
    }
    return TRUE;
}
