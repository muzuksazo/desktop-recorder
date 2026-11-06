use std::env;
use std::fs;
use std::path::PathBuf;

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

fn main() {
    println!("cargo:rerun-if-env-changed=TALKIT_MODEL_KEY");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let hex = load_key_hex();
    let dest = out_dir.join("tk_key.rs");

    if hex.len() == 64 {
        let mut raw = [0u8; 32];
        for i in 0..32 {
            raw[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap_or(0);
        }
        // Obfuscate with deterministic non-trivial mask so hex/raw key never appears in binary strings
        let mut mask = [0u8; 32];
        let mut state: u64 = 0x544B_4F4D_4547_4131;
        for i in 0..32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            mask[i] = (state & 0xFF) as u8 ^ ((i as u8).wrapping_mul(0x9E));
        }
        let mut enc = [0u8; 32];
        for i in 0..32 {
            enc[i] = raw[i] ^ mask[i];
        }
        let code = format!(
            "const TK_HAS_KEY: bool = true;\nconst TK_MASK: [u8; 32] = {:?};\nconst TK_ENC: [u8; 32] = {:?};\n",
            mask, enc
        );
        fs::write(dest, code).expect("write tk_key.rs");
    } else {
        let code = "const TK_HAS_KEY: bool = false;\nconst TK_MASK: [u8; 32] = [0u8; 32];\nconst TK_ENC: [u8; 32] = [0u8; 32];\n";
        fs::write(dest, code).expect("write tk_key.rs");
    }
}
