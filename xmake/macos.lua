-- brew gcc provides native OpenMP (libgomp); Apple Clang does not
local gcc_cc, gcc_cxx = nil, nil
if is_plat("macosx") then
    for _, prefix in ipairs({"/usr/local/opt/gcc", "/opt/homebrew/opt/gcc",
                              "/usr/local/bin", "/opt/homebrew/bin"}) do
        for major = 20, 10, -1 do
            local cc = path.join(prefix, "bin", "gcc-" .. major)
            if os.isfile(cc) then
                gcc_cc = cc
                gcc_cxx = path.join(prefix, "bin", "g++-" .. major)
                break
            end
        end
        if gcc_cc then
            break
        end
    end
    if not gcc_cc then
        raise("macOS build requires brew gcc for OpenMP, run: brew install gcc")
    end

    -- brew gcc defaults to CLT SDK (no headers); use a full SDK
    local sdk = nil
    for _, dir in ipairs({
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"}) do
        if os.isfile(path.join(dir, "usr", "include", "stdint.h")) then
            sdk = dir
            break
        end
    end
    if not sdk then
        raise("macOS build requires a full SDK with usr/include headers")
    end
    macos_sdk = sdk
end

toolchain("brew-gcc")
    set_kind("standalone")
    set_toolset("cc", gcc_cc or "gcc")
    set_toolset("cxx", gcc_cxx or "g++")
    set_toolset("ld", gcc_cxx or "g++")
    set_toolset("sh", gcc_cxx or "g++")
toolchain_end()
