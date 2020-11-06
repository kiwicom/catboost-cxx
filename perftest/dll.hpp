#pragma once

#include <stdexcept>

class FuncPtr {
public:
    FuncPtr() = delete;
    explicit FuncPtr(void* p) : ptr_(p) { }
    FuncPtr(const FuncPtr&) = default;
    FuncPtr(FuncPtr&&) = default;
    FuncPtr& operator = (const FuncPtr&) = default;
    FuncPtr& operator = (FuncPtr&&) = default;

    template <typename R, typename...Args>
    using Ptr = R(*)(Args...);

    template <typename R, typename...Args>
    operator Ptr<R, Args...> () {
        return (Ptr<R, Args...>)(ptr_);
    }

private:
    void* ptr_;
};

#ifdef WIN32
# include <windows.h>

class DLL {
public:
    explicit DLL(const std::string& filename) {
        lib_ = LoadLibraryA(filename.c_str());
    }

    ~DLL() {
        if (lib_) {
            FreeLibrary(lib_);
        }
    }

    DLL(const DLL&) = delete;
    DLL(DLL&&) = delete;
    DLL& operator=(const DLL&) = delete;
    DLL& operator=(DLL&&) = delete;

    FuncPtr sym(const std::string& name) {
        if (!lib_) {
            return nullptr;
        }

        return FuncPtr(static_cast<void*>(GetProcAddress(lib_, name.c_str())));
    }

    operator bool () const {
        return lib_ != nullptr;
    }

private:
    HMODULE lib_;
};

#else

#include <dlfcn.h>

class DLL {
public:
    explicit DLL(const std::string& filename) {
        lib_ = dlopen(filename.c_str(), RTLD_LOCAL | RTLD_NOW);
    }

    ~DLL() {
    }

    DLL(const DLL&) = delete;
    DLL(DLL&&) = delete;
    DLL& operator=(const DLL&) = delete;
    DLL& operator=(DLL&&) = delete;

    FuncPtr sym(const std::string& name) {
        if (!lib_) {
            return FuncPtr{nullptr};
        }

        return FuncPtr(dlsym(lib_, name.c_str()));
    }

    operator bool () const {
        return lib_ != nullptr;
    }

private:
    void* lib_;
};

#endif

