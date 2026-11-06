use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn load_key_hex() -> String {
    if let Ok(v) = env::var("TALKIT_MODEL_KEY") {
        let t = v.trim().to_string();
        if t.len() == 64 {
            return t;
        }
    }
    let home = env::var("HOME").unwrap_or_default();
    for rel in [
        ".config/talkit/r2.env",
        "Library/Application Support/Talkit/r2.env",
    ] {
        let p = PathBuf::from(&home).join(rel);
        println!("cargo:rerun-if-changed={}", p.display());
        if let Ok(txt) = fs::read_to_string(&p) {
            for line in txt.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("TALKIT_MODEL_KEY=") {
                    let v = rest.trim();
                    if v.len() == 64 {
                        return v.to_string();
                    }
                }
            }
        }
    }
    String::new()
}

fn xor_encode(bytes: &[u8], seed: u8) -> Vec<u8> {
    bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| b ^ seed.wrapping_add((i as u8).wrapping_mul(0x5D)))
        .collect()
}

fn fmt_c_array(name: &str, bytes: &[u8]) -> String {
    let body = bytes
        .iter()
        .map(|b| format!("0x{:02x}", b))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "static volatile const uint8_t {}[{}] = {{ {} }};\n",
        name,
        bytes.len(),
        body
    )
}

fn main() {
    println!("cargo:rerun-if-env-changed=TALKIT_MODEL_KEY");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.parent().expect("repo_root").to_path_buf();
    let hex = load_key_hex();
    let dest = out_dir.join("tk_key.rs");

    if hex.len() == 64 {
        let mut raw = [0u8; 32];
        for i in 0..32 {
            raw[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap_or(0);
        }
        // 3-way split obfuscation: raw[i] = (t0[perm[i]] ^ t1[i]).wrapping_sub(t2[i])
        let mut perm = [0usize; 32];
        for i in 0..32 {
            perm[i] = (i * 7 + 13) & 31;
        }
        let mut t0 = [0u8; 32];
        let mut t1 = [0u8; 32];
        let mut t2 = [0u8; 32];
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in 0..32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            t1[i] = (state & 0xFF) as u8;
            t2[i] = ((state >> 16) & 0xFF) as u8;
            t0[perm[i]] = raw[i].wrapping_add(t2[i]) ^ t1[i];
        }

        let magic_enc = xor_encode(b"TKOMEGA1", 0xA7);
        let salt_hdr_enc = xor_encode(b"tkomega1-hdr", 0x3C);
        let salt_body_enc = xor_encode(b"tkomega1-body", 0x69);

        let perm_u8: Vec<u8> = perm.iter().map(|&x| x as u8).collect();
        let c_src = format!(
            r#"#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonKeyDerivation.h>

{t0}
{t1}
{t2}
{perm}
{magic}
{shdr}
{sbody}

static int is_being_debugged(void) {{
    int mib[4] = {{ CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid() }};
    struct kinfo_proc info;
    size_t size = sizeof(info);
    memset(&info, 0, sizeof(info));
    if (sysctl(mib, 4, &info, &size, NULL, 0) == 0) {{
        if ((info.kp_proc.p_flag & P_TRACED) != 0) {{
            return 1;
        }}
    }}
    return 0;
}}

__attribute__((visibility("default")))
int tk_verify_tensor_buf(uint8_t *buf, size_t len) {{
    if (!buf || len < 16) return -1;
    if (is_being_debugged()) return -2;

    for (size_t i = 0; i < 8; i++) {{
        uint8_t m = MAGIC_ENC[i] ^ (uint8_t)(0xA7u + (uint8_t)(i * 0x5Du));
        if (buf[i] != m) return 1; /* Not TKOMEGA1 (plain safetensors) */
    }}

    uint8_t key[32];
    for (size_t i = 0; i < 32; i++) {{
        key[i] = (uint8_t)((T0[PERM[i]] ^ T1[i]) - T2[i]);
    }}

    uint8_t salt_hdr[12];
    for (size_t i = 0; i < 12; i++) {{
        salt_hdr[i] = SHDR_ENC[i] ^ (uint8_t)(0x3Cu + (uint8_t)(i * 0x5Du));
    }}

    uint8_t *mv = buf + 8;
    size_t mv_len = len - 8;

    uint8_t ks8[8];
    if (CCKeyDerivationPBKDF(kCCPBKDF2, (const char *)key, 32, salt_hdr, 12,
                             kCCPRFHmacAlgSHA256, 1, ks8, 8) != kCCSuccess) {{
        memset_s(key, sizeof(key), 0, sizeof(key));
        return -3;
    }}

    uint64_t hdr_len = 0;
    for (size_t i = 0; i < 8; i++) {{
        hdr_len |= ((uint64_t)(mv[i] ^ ks8[i])) << (i * 8);
    }}
    size_t hdr_total = 8 + (size_t)hdr_len;
    if (hdr_len > mv_len || hdr_total > mv_len) {{
        memset_s(key, sizeof(key), 0, sizeof(key));
        return -4;
    }}

    uint8_t *ks_hdr = (uint8_t *)malloc(hdr_total);
    if (!ks_hdr) {{
        memset_s(key, sizeof(key), 0, sizeof(key));
        return -5;
    }}
    if (CCKeyDerivationPBKDF(kCCPBKDF2, (const char *)key, 32, salt_hdr, 12,
                             kCCPRFHmacAlgSHA256, 1, ks_hdr, hdr_total) != kCCSuccess) {{
        free(ks_hdr);
        memset_s(key, sizeof(key), 0, sizeof(key));
        return -6;
    }}
    for (size_t i = 0; i < hdr_total; i++) {{
        mv[i] ^= ks_hdr[i];
    }}
    memset_s(ks_hdr, hdr_total, 0, hdr_total);
    free(ks_hdr);

    uint8_t *payload = mv + hdr_total;
    size_t payload_len = mv_len - hdr_total;
    const size_t stride = 1048576;
    const size_t block = 4096;
    size_t n_chunks = payload_len / stride;

    if (n_chunks > 0) {{
        uint8_t salt_body[13];
        for (size_t i = 0; i < 13; i++) {{
            salt_body[i] = SBODY_ENC[i] ^ (uint8_t)(0x69u + (uint8_t)(i * 0x5Du));
        }}
        size_t total_ks = n_chunks * block;
        uint8_t *ks_body = (uint8_t *)malloc(total_ks);
        if (!ks_body) {{
            memset_s(key, sizeof(key), 0, sizeof(key));
            return -7;
        }}
        if (CCKeyDerivationPBKDF(kCCPBKDF2, (const char *)key, 32, salt_body, 13,
                                 kCCPRFHmacAlgSHA256, 1, ks_body, total_ks) != kCCSuccess) {{
            free(ks_body);
            memset_s(key, sizeof(key), 0, sizeof(key));
            return -8;
        }}
        for (size_t c = 0; c < n_chunks; c++) {{
            uint64_t *dst64 = (uint64_t *)(payload + c * stride);
            const uint64_t *src64 = (const uint64_t *)(ks_body + c * block);
            for (size_t w = 0; w < (block / 8); w++) {{
                dst64[w] ^= src64[w];
            }}
        }}
        memset_s(ks_body, total_ks, 0, total_ks);
        free(ks_body);
        memset_s(salt_body, sizeof(salt_body), 0, sizeof(salt_body));
    }}

    memset_s(salt_hdr, sizeof(salt_hdr), 0, sizeof(salt_hdr));
    memset_s(key, sizeof(key), 0, sizeof(key));
    return 0;
}}
"#,
            t0 = fmt_c_array("T0", &t0),
            t1 = fmt_c_array("T1", &t1),
            t2 = fmt_c_array("T2", &t2),
            perm = fmt_c_array("PERM", &perm_u8),
            magic = fmt_c_array("MAGIC_ENC", &magic_enc),
            shdr = fmt_c_array("SHDR_ENC", &salt_hdr_enc),
            sbody = fmt_c_array("SBODY_ENC", &salt_body_enc),
        );

        let c_path = out_dir.join("tk_guard.c");
        fs::write(&c_path, c_src).expect("write tk_guard.c");
        let dylib_path = repo_root.join("libtk_guard.dylib");
        let status = Command::new("clang")
            .args([
                "-dynamiclib",
                "-O3",
                "-fvisibility=hidden",
                "-Wl,-x",
                "-arch",
                "arm64",
                "-mmacosx-version-min=14.0",
            ])
            .arg(&c_path)
            .arg("-o")
            .arg(&dylib_path)
            .status()
            .expect("compile libtk_guard.dylib");
        assert!(status.success(), "clang failed to compile libtk_guard.dylib");
        let _ = Command::new("strip").arg("-x").arg(&dylib_path).status();

        fs::write(&dest, "const TK_HAS_KEY: bool = true;\n").expect("write tk_key.rs");
    } else {
        fs::write(&dest, "const TK_HAS_KEY: bool = false;\n").expect("write tk_key.rs");
    }
}
