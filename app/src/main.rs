//! 실시간 전사 화면 (GPUI)
//!
//! Python 백엔드(`../backend.py`)를 실행하고, stdout 으로 나오는 JSON 줄을 읽어 화면에 표시한다.
//! 백엔드와의 약속(줄 형식)은 backend.py 맨 위 설명을 본다.
//!
//! 실행 (app 폴더에서)
//!     cargo run -- --model C:\경로\qwen3_asr_1.7b_ko --source both
//! - `--` 뒤의 옵션은 그대로 backend.py 에 넘어간다 (--source, --chunk, --model ...).
//! - 쓸 Python: 환경 변수 RECORDER_PYTHON (기본 "python"). torch·qwen-asr 가 깔린 것이어야 한다.
//! - 백엔드 스크립트 위치를 바꾸려면 환경 변수 RECORDER_BACKEND.

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

/// 백엔드가 stdout 으로 보내는 한 줄
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum BackendMsg {
    Status { msg: String },
    Text { text: String },
    Error { msg: String },
}

enum Event {
    Msg(BackendMsg),
    Exited,
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

struct TranscriptView {
    status: Vec<SharedString>,
    /// (내용, 오류 여부)
    lines: Vec<(SharedString, bool)>,
    scroll: ScrollHandle,
    backend: Option<Child>,
}

impl Drop for TranscriptView {
    /// 창이 닫히면 백엔드도 끈다
    fn drop(&mut self) {
        if let Some(mut child) = self.backend.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl TranscriptView {
    fn new(cx: &mut Context<Self>) -> Self {
        let (tx, mut rx) = mpsc::unbounded();
        let mut view = Self {
            status: vec!["백엔드 시작 중…".into()],
            lines: Vec::new(),
            scroll: ScrollHandle::new(),
            backend: None,
        };
        match spawn_backend(tx) {
            Ok(child) => view.backend = Some(child),
            Err(e) => view.push_line(format!("⚠ 백엔드 실행 실패: {e} (RECORDER_PYTHON 확인)"), true),
        }

        // 백엔드 스레드에서 온 줄을 UI 스레드에서 반영한다
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
        self.lines.push((text.into(), is_error));
        self.scroll.scroll_to_bottom();
    }

    fn apply(&mut self, event: Event) {
        match event {
            Event::Msg(BackendMsg::Status { msg }) => self.status.push(msg.into()),
            Event::Msg(BackendMsg::Text { text }) => self.push_line(text, false),
            Event::Msg(BackendMsg::Error { msg }) => self.push_line(format!("⚠ {msg}"), true),
            Event::Exited => self.push_line("⚠ 백엔드가 종료됐습니다 (터미널 로그 확인)", true),
        }
    }
}

impl Render for TranscriptView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let recent_status = self.status.iter().skip(self.status.len().saturating_sub(4)).cloned();

        div()
            .flex()
            .flex_col()
            .size_full()
            .font_family("Malgun Gothic")
            .bg(rgb(0x1e1e1e))
            .text_color(rgb(0xe0e0e0))
            // 상태 (최근 4줄)
            .child(
                div()
                    .flex_none()
                    .p_3()
                    .border_b_1()
                    .border_color(rgb(0x333333))
                    .text_sm()
                    .text_color(rgb(0x9e9e9e))
                    .children(recent_status),
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
                    .children(self.lines.iter().map(|(text, is_error)| {
                        div()
                            .when(*is_error, |d| d.text_color(rgb(0xff8a80)))
                            .child(text.clone())
                    })),
            )
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(720.), px(480.)), cx);
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
