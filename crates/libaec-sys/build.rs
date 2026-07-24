fn main() {
    if pkg_config::Config::new()
        .atleast_version("1.0")
        .probe("libaec")
        .is_ok()
    {
        return; // pkg-config found libaec and emitted the link directives
    }
    // Fallback: look for libaec.so / libaec.a in standard library paths.
    // libaec-dev on Debian/Ubuntu installs the library but omits the .pc file;
    // RPM-based distros (openSUSE, Fedora, RHEL) use /usr/lib64 instead of a
    // multiarch-tagged /usr/lib path and their libaec-devel package likewise
    // ships no .pc file, so both conventions need to be checked explicitly.
    let lib_dirs = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/lib64",
        "/usr/local/lib",
        "/usr/local/lib/x86_64-linux-gnu",
    ];
    for dir in &lib_dirs {
        let so = std::path::Path::new(dir).join("libaec.so");
        let a = std::path::Path::new(dir).join("libaec.a");
        if so.exists() || a.exists() {
            println!("cargo:rustc-link-search=native={dir}");
            println!("cargo:rustc-link-lib=aec");
            return;
        }
    }
    // libaec-sys is only ever pulled in as a dependency by clawhdf5-format's
    // optional `szip` feature, so if this build script is running at all, the
    // caller wants the szip filter linked. Silently continuing here doesn't
    // avoid a failure -- it just defers it to a much more confusing link-time
    // "undefined symbol: aec_buffer_decode" instead of a clear message here.
    panic!(
        "libaec-sys: could not find libaec via pkg-config or in any of {lib_dirs:?}. \
         Install libaec-devel (openSUSE/Fedora/RHEL) or libaec-dev (Debian/Ubuntu), \
         or point PKG_CONFIG_PATH at the directory containing libaec.pc."
    );
}
