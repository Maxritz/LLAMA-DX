// Agility SDK exports — must be on the EXE for Windows D3D12 loader
extern "C" { __declspec(dllexport) extern const unsigned int D3D12SDKVersion = 721; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\"; }

int llama_cli(int argc, char ** argv);

int main(int argc, char ** argv) {
    return llama_cli(argc, argv);
}
