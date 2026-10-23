use std::env;

/// Windows BLAS linking
fn main() {
    println!("cargo:rustc-check-cfg=cfg(win_blas_static)");

    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "windows" {
        return;
    }

    let has_openblas = env::var_os("CARGO_FEATURE_OPENBLAS").is_some();
    let has_openblas_system = env::var_os("CARGO_FEATURE_OPENBLAS_SYSTEM").is_some();
    let has_openblas_system_static = env::var_os("CARGO_FEATURE_OPENBLAS_SYSTEM_STATIC").is_some();

    let any_user_feature = has_openblas || has_openblas_system || has_openblas_system_static;

    if !any_user_feature {
        println!("cargo:rustc-cfg=win_blas_static");
    } else if has_openblas_system || has_openblas_system_static {
        println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
        println!("cargo:rerun-if-env-changed=VCPKGRS_TRIPLET");
        println!("cargo:rerun-if-env-changed=VCPKG_DEFAULT_TRIPLET");
        println!("cargo:rerun-if-env-changed=VCPKGRS_DYNAMIC");

        if let Err(err) = find_lapack_via_vcpkg() {
            panic!("Windows LAPACK system build failed: {err}");
        }
    }
}

// openblas-src's own build script already finds and links openblas via vcpkg.
// We only need to additionally link the LAPACK library, which vcpkg installs
// as a separate port providing the Fortran LAPACK routines (spotrf_, dpotrf_,
// dsyev_, etc.) that ndarray-linalg/lax requires.
fn find_lapack_via_vcpkg() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = vcpkg::Config::new();
    config.emit_includes(false);
    if let Some(triplet) = vcpkg_triplet_from_env() {
        config.target_triplet(triplet);
    }
    config.find_package("lapack").map_err(|err| format!("vcpkg failed to find lapack: {err}"))?;
    Ok(())
}

fn vcpkg_triplet_from_env() -> Option<String> {
    env::var("VCPKGRS_TRIPLET").or_else(|_| env::var("VCPKG_DEFAULT_TRIPLET")).ok()
}
