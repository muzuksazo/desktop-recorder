//! 모델 관리(내려받기 상태·삭제)와 앱 자동 업데이트. 화면(gpui)과 분리된 로직만 둔다.
//!
//! 자동 업데이트 흐름 (서버 없이 GitHub Releases 만 씀):
//!   1. GitHub API 로 최신 릴리스를 확인 → 지금 버전보다 높으면 알림
//!   2. DMG 를 받아 SHA-256(GitHub 가 주는 digest)과 대조
//!   3. DMG 안 앱이 '지금 앱과 같은 서명 요건(designated requirement)'을 만족하는지 codesign 으로 확인
//!      → 우리 인증서로 서명한 앱이 아니면 설치하지 않는다
//!   4. 앱 옆에 새 앱을 복사해 두고, 앱이 꺼지면 작은 스크립트가 바꿔 끼우고 다시 켠다
//!
//! 앱이 직접(curl) 받은 파일에는 '인터넷에서 받음' 표시(quarantine)가 붙지 않아서
//! 공증 전이어도 업데이트 때마다 '그래도 열기'를 다시 할 필요가 없다.

use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use serde::Deserialize;

// ---------------------------------------------------------------------------
// 모델
// ---------------------------------------------------------------------------

/// 다 받았다는 표시 파일 (mlx_quantized_asr.COMPLETE_MARKER 와 같아야 함)
const COMPLETE_MARKER: &str = ".talkit-complete";

/// 받을 수 있는 모델 하나 (backend.py 의 MODEL_REPOS 와 같은 id·폴더 이름)
pub struct ModelSpec {
    pub id: &'static str,
    /// models 폴더 아래 이름 (= Hugging Face 저장소 이름)
    pub dir: &'static str,
    pub label: &'static str,
    pub desc: &'static str,
    pub approx: &'static str,
    /// 개발용 Mac 의 예전 작업 폴더 이름 (외부 배포판은 보지 않음)
    dev_dir: Option<&'static str>,
}

pub const MODELS: [ModelSpec; 4] = [
    ModelSpec {
        id: "public",
        dir: "qwen3_asr_1.7b_ko-MLX-4bit",
        label: "기본 모델",
        desc: "Qwen3-ASR 1.7B 한국어 · 4-bit · 가장 가볍고 빠름",
        approx: "1.1GB",
        dev_dir: None,
    },
    ModelSpec {
        id: "rung-omega",
        dir: "qwen3_asr_1.7b_ko-RungOmega-MLX",
        label: "Rung Ω",
        desc: "정확도 우선 · 한국어 전용 출력층으로 빠르게",
        approx: "2.5GB",
        dev_dir: Some("qwen3_asr_1_7b_ko_rung_omega_g128_mlx"),
    },
    ModelSpec {
        id: "rung-omega-lite",
        dir: "qwen3_asr_1.7b_ko-RungOmega-MLX-lite",
        label: "Rung Ω 가벼운판",
        desc: "Rung Ω 4-bit · 메모리를 적게 씀",
        approx: "1.6GB",
        dev_dir: Some("qwen3_asr_1_7b_ko_rung_omega_mac_mlx"),
    },
    ModelSpec {
        id: "fa",
        dir: "qwen3_forced_aligner_0.6b-RungOmega-MLX",
        label: "문장 맞추기 (Forced Aligner)",
        desc: "Rung Ω 와 함께 켜면 단어 시각을 맞춰 문장 끝 잘림·유령 단어를 줄임",
        approx: "0.97GB",
        dev_dir: Some("qwen3_forced_aligner_0_6b_rung_omega_mac_mlx"),
    },
];

pub fn spec(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

/// 화면에서 고르는 항목: (id, 이름, 필요한 모델들)
pub const CHOICES: [(&str, &str, &[&str]); 4] = [
    ("public", "기본 모델", &["public"]),
    ("rung-omega", "Rung Ω", &["rung-omega"]),
    ("rung-omega-fa", "Rung Ω + 문장 맞추기", &["rung-omega", "fa"]),
    ("rung-omega-lite", "Rung Ω 가벼운판", &["rung-omega-lite"]),
];

/// 고른 항목에 필요한 모델 id 들 (모르는 값·빈 값이면 기본 모델)
pub fn required_models(choice: &str) -> &'static [&'static str] {
    CHOICES.iter().find(|c| c.0 == choice).map(|c| c.2).unwrap_or(&["public"])
}

/// 이 Mac 의 칩 이름과 통합 메모리(GB)를 바탕으로 알맞은 모델을 추천한다
#[derive(Clone, Debug)]
pub struct SystemInfo {
    pub chip: String,
    pub ram_gb: u64,
}

impl SystemInfo {
    /// 메모리에 따라 가장 알맞은 CHOICES 항목 id
    pub fn recommended_choice(&self) -> &'static str {
        if self.ram_gb <= 8 {
            "public"
        } else if self.ram_gb <= 18 {
            "rung-omega-lite"
        } else {
            "rung-omega-fa"
        }
    }

    /// 모델 카드에 '이 Mac 추천' 배지를 붙일 모델 id 들
    pub fn recommended_models(&self) -> &'static [&'static str] {
        if self.ram_gb <= 8 {
            &["public"]
        } else if self.ram_gb <= 18 {
            &["rung-omega-lite", "rung-omega"]
        } else {
            &["rung-omega", "fa"]
        }
    }

    pub fn is_recommended(&self, model_id: &str) -> bool {
        self.recommended_models().contains(&model_id)
    }

    /// 모델 화면 맨 위에 보여 줄 한 줄 안내
    pub fn summary_line(&self) -> String {
        if self.ram_gb <= 8 {
            format!(
                "이 Mac ({} · {}GB) → 가볍고 빠른 '기본 모델(1.1GB)'을 추천해요",
                self.chip, self.ram_gb
            )
        } else if self.ram_gb <= 18 {
            format!(
                "이 Mac ({} · {}GB) → 'Rung Ω 가벼운판(1.6GB)' 또는 'Rung Ω(2.5GB)'를 추천해요",
                self.chip, self.ram_gb
            )
        } else {
            format!(
                "이 Mac ({} · {}GB) → 가장 정확한 'Rung Ω + 문장 맞추기' 조합을 추천해요",
                self.chip, self.ram_gb
            )
        }
    }

    /// 메모리가 적은 Mac 에서 큰 모델을 볼 때 띄울 안내
    pub fn memory_warning(&self, model_id: &str) -> Option<&'static str> {
        if self.ram_gb <= 8 {
            match model_id {
                "rung-omega" => Some("8GB Mac 에서는 다른 앱과 함께 쓸 때 메모리가 빡빡할 수 있어요. '기본 모델'이나 'Rung Ω 가벼운판'을 권장합니다."),
                "fa" => Some("8GB Mac 에서는 Rung Ω 와 함께 켜면(총 3.5GB) 메모리가 부족할 수 있어요."),
                _ => None,
            }
        } else {
            None
        }
    }
}

fn sysctl_str(key: &str) -> Option<String> {
    let out = Command::new("/usr/sbin/sysctl").args(["-n", key]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

pub fn detect_system() -> SystemInfo {
    let chip = sysctl_str("machdep.cpu.brand_string").unwrap_or_else(|| "Apple Silicon".into());
    let ram_gb = sysctl_str("hw.memsize")
        .and_then(|s| s.parse::<u64>().ok())
        .map(|bytes| ((bytes + (1 << 29)) >> 30).max(1))
        .unwrap_or(16);
    SystemInfo { chip, ram_gb }
}

#[derive(Clone, PartialEq)]
pub enum ModelState {
    Installed { path: PathBuf, bytes: u64, legacy: bool },
    Missing { partial_bytes: u64 },
    Downloading { got: u64, total: Option<u64> },
    Failed { msg: String, partial_bytes: u64 },
}

fn home() -> PathBuf {
    env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp"))
}

/// 모델을 두는 곳: TALKIT_MODELS_DIR 또는 ~/Library/Application Support/Talkit/models
pub fn models_root() -> PathBuf {
    match env::var_os("TALKIT_MODELS_DIR") {
        Some(p) => PathBuf::from(p),
        None => home().join("Library/Application Support/Talkit/models"),
    }
}

pub fn model_path(spec: &ModelSpec) -> PathBuf {
    models_root().join(spec.dir)
}

fn is_public_edition() -> bool {
    env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|d| d.join("../Resources/edition.txt")))
        .and_then(|p| fs::read_to_string(p).ok())
        .is_some_and(|s| s.trim() == "public")
}

/// 개발용 Mac 에 이미 있는 같은 모델 (받지 않고 쓴다. 지우지 않는다)
fn legacy_path(spec: &ModelSpec) -> Option<PathBuf> {
    if spec.id == "public" {
        return Some(home().join("qwen3-asr-ko/qwen3_asr_1.7b_ko_mlx_4bit"));
    }
    if is_public_edition() {
        return None;
    }
    let sub = spec.dev_dir?;
    [
        home().join("Library/Application Support/Talkit/engines/rung-omega/artifacts"),
        home().join("qwen3-asr-ko/private_rung_omega_mac/artifacts"),
    ]
    .into_iter()
    .map(|a| a.join(sub))
    .find(|p| p.join("model.safetensors").exists())
}

fn has_incomplete(dir: &Path) -> bool {
    let Ok(rd) = fs::read_dir(dir) else { return false };
    for e in rd.flatten() {
        let p = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            if has_incomplete(&p) {
                return true;
            }
        } else if p.extension().is_some_and(|x| x == "incomplete") {
            return true;
        }
    }
    false
}

/// 다 받은 폴더인지 (Python 쪽 model_is_complete 와 같은 판정)
fn is_complete(dir: &Path) -> bool {
    if dir.join(COMPLETE_MARKER).exists() {
        return true;
    }
    dir.join("model.safetensors").exists()
        && dir.join("config.json").exists()
        && !has_incomplete(&dir.join(".cache"))
}

pub fn dir_size(dir: &Path) -> u64 {
    let Ok(rd) = fs::read_dir(dir) else { return 0 };
    rd.flatten()
        .map(|e| match e.file_type() {
            Ok(ft) if ft.is_dir() => dir_size(&e.path()),
            Ok(ft) if ft.is_file() => e.metadata().map(|m| m.len()).unwrap_or(0),
            _ => 0,
        })
        .sum()
}

pub fn scan_model(spec: &ModelSpec) -> ModelState {
    let dir = model_path(spec);
    if is_complete(&dir) {
        return ModelState::Installed { bytes: dir_size(&dir), path: dir, legacy: false };
    }
    if let Some(legacy) = legacy_path(spec).filter(|p| p.join("model.safetensors").exists()) {
        return ModelState::Installed { bytes: dir_size(&legacy), path: legacy, legacy: true };
    }
    ModelState::Missing { partial_bytes: dir_size(&dir) }
}

/// 받은 모델(또는 받다 만 조각)을 지운다. 개발용 예전 위치는 건드리지 않는다.
pub fn delete_model(spec: &ModelSpec) -> std::io::Result<()> {
    let dir = model_path(spec);
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    Ok(())
}

/// Finder 로 연다 (폴더가 없으면 만들어서 연다)
pub fn reveal(path: &Path) {
    let _ = fs::create_dir_all(path);
    let _ = Command::new("/usr/bin/open").arg(path).spawn();
}

pub fn fmt_mb(bytes: u64) -> String {
    let mb = bytes as f64 / (1024.0 * 1024.0);
    if mb >= 1024.0 {
        format!("{:.2}GB", mb / 1024.0)
    } else {
        format!("{:.0}MB", mb)
    }
}

// ---------------------------------------------------------------------------
// 자동 업데이트
// ---------------------------------------------------------------------------

const RELEASES_API: &str = "https://api.github.com/repos/muzuksazo/desktop-recorder/releases/latest";

#[derive(Clone, Debug)]
pub struct Release {
    pub version: String,
    pub page: String,
    pub notes: String,
    pub dmg_url: String,
    pub size: u64,
    pub sha256: Option<String>,
}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
    #[serde(default)]
    html_url: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    assets: Vec<GhAsset>,
}

#[derive(Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    digest: Option<String>,
}

/// 실행 중인 .app 경로 (소스에서 cargo run 하면 None → 업데이트 기능 꺼짐)
pub fn app_bundle_path() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let bundle = exe.parent()?.parent()?.parent()?.to_path_buf();
    bundle.extension().is_some_and(|x| x == "app").then_some(bundle)
}

fn plist_string(plist: &str, key: &str) -> Option<String> {
    let after = &plist[plist.find(&format!("<key>{key}</key>"))? + key.len() + 11..];
    let start = after.find("<string>")? + 8;
    let end = after[start..].find("</string>")? + start;
    Some(after[start..end].trim().to_string())
}

fn bundle_version(bundle: &Path) -> Option<String> {
    let text = fs::read_to_string(bundle.join("Contents/Info.plist")).ok()?;
    plist_string(&text, "CFBundleShortVersionString")
}

/// 지금 앱 버전 (Info.plist). 소스 실행이면 None
pub fn current_version() -> Option<String> {
    bundle_version(&app_bundle_path()?)
}

fn version_parts(v: &str) -> Vec<u64> {
    v.trim_start_matches('v')
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or(0))
        .collect()
}

/// a 가 b 보다 새 버전인가 (1.10.0 > 1.9.3)
pub fn is_newer(a: &str, b: &str) -> bool {
    let (mut x, mut y) = (version_parts(a), version_parts(b));
    let n = x.len().max(y.len());
    x.resize(n, 0);
    y.resize(n, 0);
    x > y
}

fn curl_text(url: &str) -> Result<String, String> {
    let out = Command::new("/usr/bin/curl")
        .args(["-fsSL", "--max-time", "20", "-H", "Accept: application/vnd.github+json", "-H", "User-Agent: Talkit-Updater"])
        .arg(url)
        .output()
        .map_err(|e| format!("curl 실행 실패: {e}"))?;
    if !out.status.success() {
        return Err(format!("업데이트 정보를 받지 못했어요 ({})", String::from_utf8_lossy(&out.stderr).trim()));
    }
    String::from_utf8(out.stdout).map_err(|e| e.to_string())
}

/// 최신 릴리스 정보. TALKIT_UPDATE_FEED(URL 또는 파일 경로)로 바꿀 수 있다 (시험용)
pub fn fetch_latest() -> Result<Release, String> {
    let text = match env::var("TALKIT_UPDATE_FEED") {
        Ok(feed) if feed.starts_with('/') => fs::read_to_string(&feed).map_err(|e| e.to_string())?,
        Ok(feed) => curl_text(&feed)?,
        Err(_) => curl_text(RELEASES_API)?,
    };
    let gh: GhRelease = serde_json::from_str(&text).map_err(|e| format!("업데이트 정보를 읽지 못했어요: {e}"))?;
    if gh.draft || gh.prerelease {
        return Err("정식 릴리스가 아닙니다".into());
    }
    let asset = gh
        .assets
        .into_iter()
        .find(|a| a.name.ends_with(".dmg"))
        .ok_or("릴리스에 DMG 파일이 없어요")?;
    Ok(Release {
        version: gh.tag_name.trim_start_matches('v').to_string(),
        page: gh.html_url,
        notes: gh.body.unwrap_or_default(),
        dmg_url: asset.browser_download_url,
        size: asset.size,
        sha256: asset
            .digest
            .and_then(|d| d.strip_prefix("sha256:").map(|s| s.to_ascii_lowercase())),
    })
}

/// 릴리스 노트의 '새로 바뀐 것' 첫 몇 줄 (배너에 짧게 보여 줄 용도)
pub fn short_notes(notes: &str, max: usize) -> Vec<String> {
    notes
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("- "))
        .map(|l| l.trim_start_matches("- ").replace("**", ""))
        .take(max)
        .collect()
}

fn updates_dir() -> PathBuf {
    home().join("Library/Caches/Talkit/updates")
}

/// 앱을 바꿔 끼울 수 있는 위치인가 (DMG 안에서 바로 실행·읽기 전용·격리 실행이면 안 됨)
pub fn install_blocker(bundle: &Path) -> Option<String> {
    let s = bundle.to_string_lossy();
    if s.starts_with("/Volumes/") || s.contains("/AppTranslocation/") {
        return Some("Talkit 을 '응용 프로그램' 폴더로 옮긴 뒤 다시 켜야 업데이트할 수 있어요.".into());
    }
    let parent = bundle.parent()?;
    let probe = parent.join(format!(".talkit-write-test-{}", std::process::id()));
    match fs::File::create(&probe) {
        Ok(_) => {
            let _ = fs::remove_file(&probe);
            None
        }
        Err(_) => Some(format!("'{}' 폴더에 쓸 권한이 없어 업데이트할 수 없어요.", parent.display())),
    }
}

fn run(cmd: &mut Command) -> Result<String, String> {
    let out = cmd.output().map_err(|e| e.to_string())?;
    let all = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if out.status.success() {
        Ok(all)
    } else {
        Err(all.trim().to_string())
    }
}

/// DMG 를 받는다. progress(받은 바이트) 를 0.3초마다 부른다.
pub fn download_dmg(rel: &Release, progress: impl Fn(u64)) -> Result<PathBuf, String> {
    let dir = updates_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let dest = dir.join(format!("Talkit-{}.dmg", rel.version));
    let part = dir.join(format!("Talkit-{}.dmg.part", rel.version));
    let _ = fs::remove_file(&part);
    let mut child = Command::new("/usr/bin/curl")
        .args(["-fsSL", "--retry", "2", "--connect-timeout", "20", "-H", "User-Agent: Talkit-Updater", "-o"])
        .arg(&part)
        .arg(&rel.dmg_url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("curl 실행 실패: {e}"))?;
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            if !status.success() {
                let mut msg = String::new();
                if let Some(mut err) = child.stderr.take() {
                    use std::io::Read;
                    let _ = err.read_to_string(&mut msg);
                }
                let _ = fs::remove_file(&part);
                return Err(format!("업데이트 파일을 받지 못했어요. 인터넷 연결을 확인해 주세요. ({})", msg.trim()));
            }
            break;
        }
        progress(fs::metadata(&part).map(|m| m.len()).unwrap_or(0));
        thread::sleep(Duration::from_millis(300));
    }
    fs::rename(&part, &dest).map_err(|e| e.to_string())?;
    progress(fs::metadata(&dest).map(|m| m.len()).unwrap_or(0));
    Ok(dest)
}

/// 받은 DMG 를 검증하고 새 앱을 지금 앱 옆에 복사해 둔다. 돌려주는 값: 복사해 둔 새 앱 경로
pub fn prepare_update(rel: &Release, dmg: &Path, bundle: &Path) -> Result<PathBuf, String> {
    // 1) 파일이 GitHub 에 올린 그 파일인지
    if let Some(want) = &rel.sha256 {
        let out = run(Command::new("/usr/bin/shasum").args(["-a", "256"]).arg(dmg))?;
        let got = out.split_whitespace().next().unwrap_or_default().to_ascii_lowercase();
        if &got != want {
            let _ = fs::remove_file(dmg);
            return Err("받은 파일이 손상됐거나 바뀌었어요 (SHA-256 불일치). 설치하지 않았습니다.".into());
        }
    }

    // 2) 지금 앱의 서명 요건 (같은 인증서·같은 앱 id 여야 함)
    let dr_out = run(Command::new("/usr/bin/codesign").args(["-d", "-r-"]).arg(bundle))?;
    let requirement = dr_out
        .lines()
        .find_map(|l| l.trim().strip_prefix("designated => "))
        .map(str::to_string)
        .ok_or("지금 앱의 서명 정보를 읽지 못했어요")?;

    // 3) DMG 열기 → 새 앱 확인 → 앱 옆에 복사
    let mnt = updates_dir().join(format!("mnt-{}", std::process::id()));
    let _ = fs::create_dir_all(&mnt);
    run(Command::new("/usr/bin/hdiutil")
        .args(["attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint"])
        .arg(&mnt)
        .arg(dmg))
    .map_err(|e| format!("업데이트 파일을 열지 못했어요: {e}"))?;
    let result = (|| {
        let src = mnt.join("Talkit.app");
        if !src.exists() {
            return Err("업데이트 파일 안에 Talkit.app 이 없어요".to_string());
        }
        run(Command::new("/usr/bin/codesign")
            .args(["--verify", "--deep", "--strict"])
            .arg(format!("-R={requirement}"))
            .arg(&src))
        .map_err(|e| format!("새 앱의 서명이 지금 앱과 달라 설치하지 않았어요: {e}"))?;
        match bundle_version(&src) {
            Some(v) if v == rel.version => {}
            other => return Err(format!("버전이 맞지 않아요 (기대 {}, 실제 {:?})", rel.version, other)),
        }
        let staged = bundle.with_file_name(".Talkit-update.app");
        if staged.exists() {
            fs::remove_dir_all(&staged).map_err(|e| e.to_string())?;
        }
        run(Command::new("/usr/bin/ditto").arg(&src).arg(&staged))
            .map_err(|e| format!("새 앱을 복사하지 못했어요: {e}"))?;
        Ok(staged)
    })();
    let _ = run(Command::new("/usr/bin/hdiutil").args(["detach", "-quiet"]).arg(&mnt));
    let _ = fs::remove_dir(&mnt);
    if result.is_ok() {
        let _ = fs::remove_file(dmg);
    }
    result
}

/// 앱이 꺼지길 기다렸다가 새 앱으로 바꿔 끼우고 다시 켜는 스크립트를 띄운다. 이 함수 뒤에 앱을 끈다.
pub fn launch_installer(bundle: &Path, staged: &Path) -> Result<(), String> {
    let dir = updates_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let script = dir.join("install.sh");
    let mut f = fs::File::create(&script).map_err(|e| e.to_string())?;
    // 스크립트 내용: 앱이 꺼지면 새 앱으로 바꾸고 다시 켠다. 바꾸다 실패하면 이전 앱을 되돌린다.
    f.write_all(
        br#"#!/bin/sh
# Talkit updater: wait for the app to quit, swap in the new app, relaunch. Roll back on failure.
PID="$1"; OLD="$2"; NEW="$3"
i=0
while kill -0 "$PID" 2>/dev/null && [ $i -lt 300 ]; do sleep 0.2; i=$((i+1)); done
BAK="$OLD.old-$$"
if mv "$OLD" "$BAK"; then
  if mv "$NEW" "$OLD"; then
    rm -rf "$BAK"
  else
    mv "$BAK" "$OLD"
  fi
fi
xattr -dr com.apple.quarantine "$OLD" 2>/dev/null
open "$OLD"
"#,
    )
    .map_err(|e| e.to_string())?;
    drop(f);
    use std::os::unix::process::CommandExt;
    Command::new("/bin/sh")
        .arg(&script)
        .arg(std::process::id().to_string())
        .arg(bundle)
        .arg(staged)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0) // 앱이 꺼져도 같이 죽지 않게
        .spawn()
        .map_err(|e| format!("설치 도우미를 띄우지 못했어요: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_compare() {
        assert!(is_newer("1.3.0", "1.2.1"));
        assert!(is_newer("v1.10.0", "1.9.9"));
        assert!(is_newer("1.2.1", "1.2"));
        assert!(!is_newer("1.2.1", "1.2.1"));
        assert!(!is_newer("1.2.0", "1.2.1"));
    }

    #[test]
    fn plist_parse() {
        let p = "<dict>\n  <key>CFBundleShortVersionString</key><string>1.2.1</string>\n</dict>";
        assert_eq!(plist_string(p, "CFBundleShortVersionString").as_deref(), Some("1.2.1"));
    }

    #[test]
    fn notes_bullets() {
        let n = "## T\n### 새로 바뀐 것\n- **고침** 하나\n- 둘\n### 설치\n1. x";
        assert_eq!(short_notes(n, 3), vec!["고침 하나".to_string(), "둘".to_string()]);
    }

    #[test]
    fn system_recommendations() {
        let m8 = SystemInfo { chip: "Apple M1".into(), ram_gb: 8 };
        assert_eq!(m8.recommended_choice(), "public");
        assert!(m8.is_recommended("public"));
        assert!(!m8.is_recommended("rung-omega"));
        assert!(m8.memory_warning("rung-omega").is_some());

        let m16 = SystemInfo { chip: "Apple M1 Pro".into(), ram_gb: 16 };
        assert_eq!(m16.recommended_choice(), "rung-omega-lite");
        assert!(m16.is_recommended("rung-omega-lite"));
        assert!(m16.is_recommended("rung-omega"));
        assert!(m16.memory_warning("rung-omega").is_none());

        let m32 = SystemInfo { chip: "Apple M3 Max".into(), ram_gb: 36 };
        assert_eq!(m32.recommended_choice(), "rung-omega-fa");
        assert!(m32.is_recommended("rung-omega"));
        assert!(m32.is_recommended("fa"));
    }
}

/// 실제 업데이트 교체 시험 (손으로만 실행):
///   TALKIT_TEST_BUNDLE=<옛 버전 Talkit.app> TALKIT_UPDATE_FEED=<feed.json> PATH=<가짜 open 있는 폴더>:$PATH \
///   cargo test --release -- --ignored update_end_to_end --nocapture
#[cfg(test)]
mod e2e {
    use super::*;

    #[test]
    #[ignore]
    fn update_end_to_end() {
        let bundle = PathBuf::from(env::var("TALKIT_TEST_BUNDLE").expect("TALKIT_TEST_BUNDLE"));
        let old = bundle_version(&bundle).unwrap();
        let rel = fetch_latest().unwrap();
        assert!(is_newer(&rel.version, &old), "{} > {}", rel.version, old);
        assert!(install_blocker(&bundle).is_none());

        // 해시가 틀린 파일은 거부해야 한다
        let mut bad = rel.clone();
        bad.sha256 = Some("00".repeat(32));
        let dmg = download_dmg(&bad, |_| {}).unwrap();
        let err = prepare_update(&bad, &dmg, &bundle).unwrap_err();
        assert!(err.contains("SHA-256"), "{err}");

        // 정상 파일: 받기 → 검증 → 옆에 복사
        let dmg = download_dmg(&rel, |_| {}).unwrap();
        let staged = prepare_update(&rel, &dmg, &bundle).unwrap();
        assert_eq!(bundle_version(&staged).as_deref(), Some(rel.version.as_str()));

        // 바꿔 끼우기: 이 테스트 프로세스가 끝나면 스크립트가 교체한다
        launch_installer(&bundle, &staged).unwrap();
        println!("교체 예약됨: {} → {}", old, rel.version);
    }
}
