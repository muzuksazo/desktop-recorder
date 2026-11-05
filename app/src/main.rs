//! 실시간 전사 화면 (GPUI)
//!
//! Python 백엔드(`../backend.py`)를 실행하고, stdout 으로 나오는 JSON 줄을 읽어 화면에 표시한다.
//! 백엔드와의 약속(줄 형식)은 backend.py 맨 위 설명을 본다.
//! 문장을 누르면 TTS(Windows 기본 한국어 음성)로 읽어 준다. "자동 읽기"를 켜면 새 문장마다 읽는다.
//!
//! 실행 (app 폴더에서)
//!     cargo run -- --model C:\경로\qwen3_asr_1.7b_ko --source both
//! - `--` 뒤의 옵션은 그대로 backend.py 에 넘어간다 (--source, --context, --pause, --model ...).
//! - 쓸 Python: 환경 변수 RECORDER_PYTHON (기본 "python"). torch·qwen-asr·silero-vad 가 깔린 것이어야 한다.
//! - 백엔드 스크립트 위치를 바꾸려면 환경 변수 RECORDER_BACKEND.

mod tts;

use std::{
    env,
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
};

use futures::{channel::mpsc, StreamExt};
use gpui::{
    div, prelude::*, px, rgb, size, App, Application, Bounds, Context, ScrollHandle, SharedString,
    TitlebarOptions, Window, WindowBounds, WindowOptions,
};
use serde::Deserialize;
use tts::Tts;

/// TTS 가 소리를 낸 구간과 겹친 전사는 TTS 소리가 다시 잡힌 것으로 보고 버린다 (앞뒤 여유, 초)
const TTS_ECHO_MARGIN: f64 = 0.5;

/// 백엔드가 stdout 으로 보내는 한 줄
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum BackendMsg {
    Status {
        msg: String,
    },
    Text {
        text: String,
        clock: Option<String>,
        start: Option<f64>,
        end: Option<f64>,
        infer_sec: Option<f64>,
    },
    Error {
        msg: String,
    },
}

enum Event {
    Msg(BackendMsg),
    Exited,
    TtsReady(Result<String, String>),
}

/// Python 백엔드를 실행하고, stdout 을 읽는 스레드를 띄운다.
fn spawn_backend(tx: mpsc::UnboundedSender<Event>) -> std::io::Result<Child> {
    let python = env::var("RECORDER_PYTHON").unwrap_or_else(|_| "python".into());
    let script = env::var_os("RECORDER_BACKEND")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("backend.py"));

    let mut child = Command::new(python)
        .args(["-X", "utf8"])
        .arg(&script)
        .args(env::args().skip(1)) // 앱 옵션을 그대로 백엔드에
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // 백엔드 로그는 터미널에
        .spawn()?;

    let stdout = child.stdout.take().expect("stdout 을 piped 로 열었음");
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            // 약속과 다른 줄이 와도 버리지 않고 상태로 보여 준다
            let msg = serde_json::from_str(&line).unwrap_or(BackendMsg::Status { msg: line });
            if tx.unbounded_send(Event::Msg(msg)).is_err() {
                return; // 창이 닫힘
            }
        }
        let _ = tx.unbounded_send(Event::Exited);
    });
    Ok(child)
}

struct Line {
    text: SharedString,
    is_error: bool,
}

struct TranscriptView {
    status: Vec<SharedString>,
    lines: Vec<Line>,
    scroll: ScrollHandle,
    backend: Option<Child>,
    tts: Tts,
    tts_voice: SharedString,
    auto_read: bool,
    /// TTS 소리가 다시 잡혀서 버린 문장 수
    echo_dropped: usize,
    /// 마지막 문장의 전사 시간 (초)
    last_infer: Option<f64>,
}

impl Drop for TranscriptView {
    /// 창이 닫히면 백엔드도 끈다
    fn drop(&mut self) {
        self.tts.stop();
        if let Some(mut child) = self.backend.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl TranscriptView {
    fn new(cx: &mut Context<Self>) -> Self {
        let (tx, mut rx) = mpsc::unbounded();
        let tts_tx = tx.clone();
        let tts = Tts::start(move |r| {
            let _ = tts_tx.unbounded_send(Event::TtsReady(r));
        });

        let mut view = Self {
            status: vec!["백엔드 시작 중…".into()],
            lines: Vec::new(),
            scroll: ScrollHandle::new(),
            backend: None,
            tts,
            tts_voice: "음성 준비 중…".into(),
            auto_read: false,
            echo_dropped: 0,
            last_infer: None,
        };
        match spawn_backend(tx) {
            Ok(child) => view.backend = Some(child),
            Err(e) => view.push_line(format!("⚠ 백엔드 실행 실패: {e} (RECORDER_PYTHON 확인)"), true),
        }

        // 백엔드·TTS 스레드에서 온 일을 UI 스레드에서 반영한다
        cx.spawn(async move |this, cx| {
            while let Some(event) = rx.next().await {
                let alive = this.update(cx, |view, cx| {
                    view.apply(event);
                    cx.notify();
                });
                if alive.is_err() {
                    break;
                }
            }
        })
        .detach();
        view
    }

    fn push_line(&mut self, text: impl Into<SharedString>, is_error: bool) {
        self.lines.push(Line { text: text.into(), is_error });
        self.scroll.scroll_to_bottom();
    }

    fn apply(&mut self, event: Event) {
        match event {
            Event::Msg(BackendMsg::Status { msg }) => self.status.push(msg.into()),
            Event::Msg(BackendMsg::Text { text, clock, start, end, infer_sec }) => {
                if let (Some(s), Some(e)) = (start, end) {
                    if self.tts.overlaps(s, e, TTS_ECHO_MARGIN) {
                        self.echo_dropped += 1;
                        return;
                    }
                }
                self.last_infer = infer_sec;
                if self.auto_read {
                    self.tts.speak(text.clone());
                }
                let shown = match clock {
                    Some(c) => format!("[{c}] {text}"),
                    None => text,
                };
                self.push_line(shown, false);
            }
            Event::Msg(BackendMsg::Error { msg }) => self.push_line(format!("⚠ {msg}"), true),
            Event::Exited => self.push_line("⚠ 백엔드가 종료됐습니다 (터미널 로그 확인)", true),
            Event::TtsReady(Ok(name)) => self.tts_voice = format!("음성: {name}").into(),
            Event::TtsReady(Err(e)) => self.tts_voice = format!("TTS 사용 불가: {e}").into(),
        }
    }

    /// 화면의 "[14:31:05] 문장" 에서 시각을 빼고 읽는다
    fn read_line(&self, ix: usize) {
        if let Some(line) = self.lines.get(ix).filter(|l| !l.is_error) {
            let text = line.text.as_ref();
            let body = text.split_once("] ").map_or(text, |(_, b)| b);
            self.tts.stop();
            self.tts.speak(body.to_string());
        }
    }
}

fn button(id: &'static str, label: impl Into<SharedString>, active: bool) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .px_3()
        .py_1()
        .rounded_md()
        .cursor_pointer()
        .bg(if active { rgb(0x2e7d32) } else { rgb(0x333333) })
        .hover(|s| s.bg(rgb(0x444444)))
        .child(label.into())
}

impl Render for TranscriptView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let recent_status = self.status.iter().skip(self.status.len().saturating_sub(3)).cloned();
        let mut info = vec![self.tts_voice.to_string()];
        if let Some(t) = self.last_infer {
            info.push(format!("마지막 전사 {t:.1}초"));
        }
        if self.echo_dropped > 0 {
            info.push(format!("TTS 소리라서 건너뜀 {}개", self.echo_dropped));
        }

        div()
            .flex()
            .flex_col()
            .size_full()
            .font_family("Malgun Gothic")
            .bg(rgb(0x1e1e1e))
            .text_color(rgb(0xe0e0e0))
            // 상태 (최근 3줄)
            .child(
                div()
                    .flex_none()
                    .px_3()
                    .pt_2()
                    .text_sm()
                    .text_color(rgb(0x9e9e9e))
                    .children(recent_status),
            )
            // TTS 도구 줄
            .child(
                div()
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap_2()
                    .px_3()
                    .py_2()
                    .border_b_1()
                    .border_color(rgb(0x333333))
                    .text_sm()
                    .child(
                        button("auto-read", if self.auto_read { "자동 읽기 켜짐" } else { "자동 읽기 꺼짐" }, self.auto_read)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.auto_read = !this.auto_read;
                                cx.notify();
                            })),
                    )
                    .child(button("stop-read", "읽기 멈춤", false).on_click(cx.listener(|this, _, _, _| this.tts.stop())))
                    .child(div().text_color(rgb(0x9e9e9e)).child(format!("문장을 누르면 읽어 줘요 · {}", info.join(" · ")))),
            )
            // 전사 결과
            .child(
                div()
                    .id("transcript")
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_h(px(0.))
                    .gap_1()
                    .p_3()
                    .text_lg()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .children(self.lines.iter().enumerate().map(|(ix, line)| {
                        div()
                            .id(("line", ix))
                            .px_1()
                            .rounded_sm()
                            .when(line.is_error, |d| d.text_color(rgb(0xff8a80)))
                            .when(!line.is_error, |d| {
                                d.cursor_pointer()
                                    .hover(|s| s.bg(rgb(0x2a2a2a)))
                                    .on_click(cx.listener(move |this, _, _, _| this.read_line(ix)))
                            })
                            .child(line.text.clone())
                    })),
            )
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(760.), px(520.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some("실시간 전사 — qwen3_asr_1.7b_ko".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            |_, cx| cx.new(TranscriptView::new),
        )
        .expect("창 열기 실패");

        // 창을 닫으면 앱도 끝낸다
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();
        cx.activate(true);
    });
}
