//! TTS: Windows 기본 음성(SAPI)으로 문장을 읽는다. 한국어 음성(예: Microsoft Heami)이 있으면 그걸 쓴다.
//!
//! - 전용 스레드 하나가 SAPI 를 들고 차례로 읽는다 (UI 를 막지 않음).
//! - 읽은 시간 구간과 글을 기록해 둔다. 시스템 소리를 전사할 때 TTS 소리가 다시 전사되는 걸
//!   앱이 걸러 낼 수 있게 하기 위해서다 (`is_echo`).

use std::{
    collections::VecDeque,
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

enum Cmd {
    Speak(String),
    Stop,
}

/// 읽은 구간 (시작, 끝, 읽은 글) — 시각은 Unix 초, 읽는 중이면 끝 = 무한대.
type Spans = Arc<Mutex<Vec<(f64, f64, String)>>>;

/// 두 문장이 얼마나 비슷한가 (0~1, 공백 무시 글자 2개 묶음 기준)
fn similarity(a: &str, b: &str) -> f64 {
    fn bigrams(s: &str) -> Vec<(char, char)> {
        let cs: Vec<char> = s.chars().filter(|c| c.is_alphanumeric()).collect();
        cs.windows(2).map(|w| (w[0], w[1])).collect()
    }
    let (a, mut b) = (bigrams(a), bigrams(b));
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let total = a.len() + b.len();
    let mut common = 0;
    for x in &a {
        if let Some(i) = b.iter().position(|y| y == x) {
            b.swap_remove(i);
            common += 1;
        }
    }
    2.0 * common as f64 / total as f64
}

#[derive(Clone)]
pub struct Tts {
    tx: mpsc::Sender<Cmd>,
    spans: Spans,
}

pub fn now_secs() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

impl Tts {
    /// 음성 스레드를 띄운다. 준비 결과(사용할 음성 이름 또는 오류)는 `on_ready` 로 한 번 알려 준다.
    pub fn start(on_ready: impl FnOnce(Result<String, String>) + Send + 'static) -> Self {
        let (tx, rx) = mpsc::channel();
        let spans: Spans = Arc::default();
        let thread_spans = spans.clone();
        thread::spawn(move || imp::run(rx, thread_spans, on_ready));
        Self { tx, spans }
    }

    pub fn speak(&self, text: impl Into<String>) {
        let _ = self.tx.send(Cmd::Speak(text.into()));
    }

    pub fn stop(&self) {
        let _ = self.tx.send(Cmd::Stop);
    }

    /// 이 전사가 TTS 소리가 다시 녹음된 것(메아리)인가.
    /// [start, end] 동안 TTS 가 읽고 있었고, 전사된 글이 그때 읽은 글과 비슷하면 메아리로 본다.
    /// 시간만 겹치고 내용이 다르면(그동안 강사가 한 말) 메아리가 아니다.
    pub fn is_echo(&self, text: &str, start: f64, end: f64, margin: f64) -> bool {
        let mut spans = self.spans.lock().unwrap();
        let cutoff = now_secs() - 120.0;
        spans.retain(|(_, e, _)| *e > cutoff); // 2분 지난 기록은 버림
        let spoken: String = spans
            .iter()
            .filter(|(s, e, _)| start < e + margin && end > s - margin)
            .map(|(_, _, t)| t.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        !spoken.is_empty() && (similarity(text, &spoken) >= 0.5 || spoken.contains(text.trim()))
    }
}

/// 명령을 받아 대기열에 반영한다. Stop 이면 true.
fn apply(cmd: Cmd, queue: &mut VecDeque<String>) -> bool {
    match cmd {
        Cmd::Speak(t) => {
            queue.push_back(t);
            false
        }
        Cmd::Stop => {
            queue.clear();
            true
        }
    }
}

#[cfg(windows)]
mod imp {
    use super::*;
    use windows::{
        core::{w, Interface, HSTRING, PCWSTR},
        Win32::{
            Media::Speech::{
                IEnumSpObjectTokens, ISpObjectToken, ISpObjectTokenCategory, ISpVoice, SpObjectTokenCategory,
                SpVoice, SPCAT_VOICES, SPF_ASYNC, SPF_PURGEBEFORESPEAK,
            },
            System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED},
        },
    };

    /// 설치된 음성 중 한국어(LCID 412)를 찾는다.
    unsafe fn korean_voice() -> Option<ISpObjectToken> {
        let cat: ISpObjectTokenCategory = CoCreateInstance(&SpObjectTokenCategory, None, CLSCTX_ALL).ok()?;
        cat.SetId(SPCAT_VOICES, false).ok()?;
        let tokens: IEnumSpObjectTokens = cat.EnumTokens(w!("Language=412"), PCWSTR::null()).ok()?;
        let mut token = None;
        tokens.Next(1, &mut token, None).ok()?;
        token
    }

    unsafe fn voice_name(token: &ISpObjectToken) -> String {
        token
            .GetStringValue(PCWSTR::null())
            .ok()
            .and_then(|p| p.to_string().ok())
            .unwrap_or_else(|| "한국어 음성".into())
    }

    /// 끝날 때까지 기다린다. 기다리는 동안 Stop 이 오면 끊고 true.
    unsafe fn wait_done(voice: &ISpVoice, rx: &mpsc::Receiver<Cmd>, queue: &mut VecDeque<String>) -> bool {
        loop {
            // WaitUntilDone: S_OK(0) = 끝남, S_FALSE(1) = 아직 (windows-rs 의 Result 로는 구분이 안 돼 직접 부른다)
            let hr = (Interface::vtable(voice).WaitUntilDone)(Interface::as_raw(voice), 50);
            if hr.0 != 1 {
                return false;
            }
            while let Ok(cmd) = rx.try_recv() {
                if apply(cmd, queue) {
                    let _ = voice.Speak(w!(""), (SPF_ASYNC.0 | SPF_PURGEBEFORESPEAK.0) as u32, None);
                    return true;
                }
            }
        }
    }

    pub fn run(rx: mpsc::Receiver<Cmd>, spans: Spans, on_ready: impl FnOnce(Result<String, String>)) {
        let voice: ISpVoice = unsafe {
            if let Err(e) = CoInitializeEx(None, COINIT_MULTITHREADED).ok() {
                return on_ready(Err(format!("COM 초기화 실패: {e}")));
            }
            match CoCreateInstance(&SpVoice, None, CLSCTX_ALL) {
                Ok(v) => v,
                Err(e) => return on_ready(Err(format!("음성 엔진을 열 수 없음: {e}"))),
            }
        };
        let name = unsafe {
            match korean_voice() {
                Some(token) => {
                    let name = voice_name(&token);
                    let _ = voice.SetVoice(&token);
                    name
                }
                None => "기본 음성 (한국어 음성 없음)".into(),
            }
        };
        on_ready(Ok(name));

        let mut queue = VecDeque::new();
        loop {
            if queue.is_empty() {
                match rx.recv() {
                    Ok(cmd) => {
                        apply(cmd, &mut queue);
                    }
                    Err(_) => return, // 앱 종료
                }
            }
            let Some(text) = queue.pop_front() else { continue };

            let idx = {
                let mut s = spans.lock().unwrap();
                s.push((now_secs(), f64::INFINITY, text.clone()));
                s.len() - 1
            };
            unsafe {
                if voice.Speak(&HSTRING::from(text.as_str()), SPF_ASYNC.0 as u32, None).is_ok() {
                    wait_done(&voice, &rx, &mut queue);
                }
            }
            if let Some(span) = spans.lock().unwrap().get_mut(idx) {
                span.1 = now_secs();
            }
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use super::*;
    pub fn run(rx: mpsc::Receiver<Cmd>, _spans: Spans, on_ready: impl FnOnce(Result<String, String>)) {
        on_ready(Err("TTS 는 지금 Windows 에서만 지원해요".into()));
        let mut q = VecDeque::new();
        while let Ok(cmd) = rx.recv() {
            apply(cmd, &mut q);
            q.clear();
        }
    }
}
