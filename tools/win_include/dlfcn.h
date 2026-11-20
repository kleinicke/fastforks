// Windows stand-in for the <dlfcn.h> subset the kernel sources use (fvm_r3: binding to the process's cuBLASLt),
// put on the include path by tools/build_kernels.py for Windows DLL builds only, so the shared .cu sources (and the
// Linux build ids) stay unchanged. Part of the Windows build id.
//
// dlopen maps a Linux soname to the Windows DLL name ("libcublasLt.so.13" -> "cublasLt64_13.dll"): RTLD_NOLOAD finds an
// already loaded copy (GetModuleHandle, e.g. PyTorch's torch/lib copy), otherwise the DLL is loaded from the
// application directory, System32 and the directories added with AddDllDirectory (PyTorch adds torch/lib), never from
// the current directory or PATH.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstring>
#include <string>

#define RTLD_LAZY 0x1
#define RTLD_NOW 0x2
#define RTLD_LOCAL 0x0
#define RTLD_GLOBAL 0x100
#define RTLD_NOLOAD 0x4

namespace ff_dlfcn {
// "lib<name>.so.<major>" -> "<name>64_<major>.dll" (the CUDA libraries' Windows naming); other names unchanged
inline std::string windows_name(const char* soname) {
    std::string s(soname);
    const std::string so = ".so.";
    const size_t at = s.rfind(so);
    if (s.rfind("lib", 0) != 0 || at == std::string::npos) return s;
    return s.substr(3, at - 3) + "64_" + s.substr(at + so.size()) + ".dll";
}
}  // namespace ff_dlfcn

inline void* dlopen(const char* name, int flags) {
    const std::string dll = ff_dlfcn::windows_name(name);
    HMODULE h = GetModuleHandleA(dll.c_str());
    if (h || (flags & RTLD_NOLOAD)) return reinterpret_cast<void*>(h);
    return reinterpret_cast<void*>(LoadLibraryExA(dll.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
}

inline void* dlsym(void* handle, const char* symbol) {
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol));
}
