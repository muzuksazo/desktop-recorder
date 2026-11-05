// qwen-live 화면(프론트엔드)
//
// 저장소 루트의 backend.py(전사 백엔드)를 실행하고, 출력되는 JSON 줄을 화면에 표시한다.
// 규칙은 docs/backend-protocol.md 참고 (backend.py 맨 위 설명과 같음).
//
// 흐름:
//   1. Backend::start()가 Python을 자식 프로세스로 실행
//   2. 별도 스레드가 Python 출력(stdout)을 한 줄씩 읽어 BackendMsg로 바꾼 뒤 채널로 보냄
//   3. 화면(MainView)이 채널에서 메시지를 받아 목록에 추가하고 다시 그림
//   4. 창을 닫으면 Python 프로세스를 끔
//
// 실행 방법 (ui 폴더에서):
//   cargo run                          진짜 백엔드 (backend.py)
//   cargo run -- --source mic          뒤에 붙인 옵션은 backend.py에 그대로 넘어감
//   $env:QWEN_BACKEND="fake"; cargo run   연습용 가짜 백엔드 (모델 없이 화면만 확인)
//   $env:QWEN_PYTHON="..\.venv\Scripts\python.exe"   쓸 Python 지정 (기본: python)

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

use gpui::{
    App, Application, Bounds, Context, ScrollHandle, Subscription, Task, TitlebarOptions, Window,
    WindowBackgroundAppearance, WindowBounds, WindowOptions, div, prelude::*, px, rgba, size,
};
use serde::Deserialize;

// ------------------------------------------------------------ 백엔드와 주고받는 메시지

/// Python이 보내는 JSON 한 줄. "type" 값에 따라 종류가 나뉜다.
/// 모르는 필드(start, end, infer_sec 등)는 자동으로 무시된다.
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
enum BackendMsg {
    Status {
        msg: String,
    },
    Text {
        /// backend.py는 "clock", 가짜 백엔드 초안은 "time" — 둘 다 받음
        #[serde(default, alias = "clock")]
        time: String,
        /// backend.py는 "text", 초안은 "msg" — 둘 다 받음
        #[serde(alias = "text")]
        msg: String,
    },
    Error {
        msg: String,
    },
    /// 모르는 type은 여기로 와서 무시된다
    #[serde(other)]
    Unknown,
}

// ------------------------------------------------------------ Python 프로세스 관리

struct Backend {
    child: Child,
}

impl Backend {
    /// Python을 실행하고, 출력 줄을 읽는 스레드를 시작한다.
    fn start(tx: smol::channel::Sender<BackendMsg>) -> Result<Backend, String> {
        let script = if std::env::var("QWEN_BACKEND").as_deref() == Ok("fake") {
            concat!(env!("CARGO_MANIFEST_DIR"), "/fake_backend.py")
        } else {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../backend.py")
        };
        // 쓸 Python: QWEN_PYTHON (또는 팀원 앱과 같은 RECORDER_PYTHON) 환경변수, 없으면 기본 python
        let python = std::env::var("QWEN_PYTHON")
            .or_else(|_| std::env::var("RECORDER_PYTHON"))
            .unwrap_or_else(|_| default_python().into());
        // cargo run -- 뒤에 쓴 옵션은 백엔드에 그대로 넘긴다
        let extra_args: Vec<String> = std::env::args().skip(1).collect();

        let mut child = Command::new(&python)
            .args(["-X", "utf8", "-u"]) // 한글 UTF-8로, 출력 버퍼링 끄기
            .arg(script)
            .args(&extra_args)
            // stdin은 쓰지 않음: Windows에서 stdin을 기다리면 torch import가 멈춘다 (backend.py 설명)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Python 실행 실패 ({python}): {e}"))?;

        let stdout = child.stdout.take().expect("stdout을 piped로 열었음");

        // 출력 읽기는 시간이 걸리므로(기다려야 하므로) 별도 스레드에서
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                match serde_json::from_str::<BackendMsg>(&line) {
                    Ok(msg) => {
                        if tx.send_blocking(msg).is_err() {
                            break; // 화면이 닫힘
                        }
                    }
                    Err(_) => eprintln!("[JSON 아님, 무시] {line}"),
                }
            }
            // 여기까지 오면 Python 출력이 끝난 것 → tx가 사라지면서 화면 쪽 recv()가 끝남
        });

        Ok(Backend { child })
    }

    /// 앱이 끝날 때 백엔드를 끈다 (txt는 문장마다 바로 저장되므로 그냥 꺼도 안전).
    fn shutdown(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn default_python() -> &'static str {
    if cfg!(windows) { "python" } else { "python3" }
}

// ------------------------------------------------------------ 화면

/// 화면에 표시할 한 줄
struct Line {
    time: String,
    msg: String,
    is_error: bool,
}

struct MainView {
    status: String,
    listening: bool, // 듣는 중인지 (상태 점 색깔용)
    theme: &'static Theme, // 지금 쓰는 디자인
    lines: Vec<Line>,
    scroll: ScrollHandle,
    backend: Option<Backend>,
    _reader_task: Option<Task<()>>,
    _quit_sub: Subscription,
}

impl MainView {
    fn new(cx: &mut Context<Self>) -> Self {
        let (tx, rx) = smol::channel::unbounded::<BackendMsg>();

        let (backend, status, reader_task) = match Backend::start(tx) {
            Ok(backend) => {
                // 채널에서 메시지를 하나씩 받아 화면에 반영하는 작업
                let task = cx.spawn(async move |this, cx| {
                    while let Ok(msg) = rx.recv().await {
                        let alive = this.update(cx, |view, cx| {
                            view.handle(msg);
                            cx.notify(); // "다시 그려줘"
                        });
                        if alive.is_err() {
                            return; // 화면이 없어짐
                        }
                    }
                    let _ = this.update(cx, |view, cx| {
                        view.status = "백엔드가 종료되었습니다 (원인은 cargo run 한 터미널의 오류 메시지를 확인하세요)".into();
                        view.listening = false;
                        cx.notify();
                    });
                });
                (Some(backend), "백엔드 시작 중…".to_string(), Some(task))
            }
            Err(e) => (None, e, None),
        };

        // 앱이 끝날 때 Python도 정리
        let quit_sub = cx.on_app_quit(|view: &mut Self, _cx| {
            if let Some(backend) = view.backend.take() {
                backend.shutdown();
            }
            async {}
        });

        MainView {
            status,
            listening: false,
            theme: start_theme(),
            lines: Vec::new(),
            scroll: ScrollHandle::new(),
            backend,
            _reader_task: reader_task,
            _quit_sub: quit_sub,
        }
    }

    /// 받은 메시지 종류에 따라 화면 상태를 바꾼다.
    fn handle(&mut self, msg: BackendMsg) {
        match msg {
            BackendMsg::Status { msg } => {
                // backend.py는 듣기 시작할 때 "듣는 중…"으로 시작하는 상태를 보낸다
                if msg.starts_with("듣는 중") {
                    self.listening = true;
                }
                self.status = msg;
            }
            BackendMsg::Text { time, msg } => {
                self.listening = true;
                self.push(time, msg, false);
            }
            BackendMsg::Error { msg } => self.push(String::new(), format!("[오류] {msg}"), true),
            BackendMsg::Unknown => {}
        }
    }

    fn push(&mut self, time: String, msg: String, is_error: bool) {
        self.lines.push(Line { time, msg, is_error });
        self.scroll.scroll_to_bottom(); // 새 줄이 오면 맨 아래로
    }
}

// ------------------------------------------------------------ 디자인 테마
// 테마 하나 = 색과 모양 값 묶음. 값만 바꿔도 분위기가 바뀐다.
// 색은 0xRRGGBBAA: 마지막 두 자리(AA)가 불투명도. 00=완전 투명, ff=완전 불투명

struct Theme {
    name: &'static str,
    window_bg: WindowBackgroundAppearance, // 창 뒤 비침 여부
    root_bg: u32,             // 창 전체 바탕
    root_padding: f32,        // 창 가장자리 여백
    status_bg: u32,           // 상태 줄 바탕
    status_border: u32,       // 상태 줄 테두리 (00이면 안 보임)
    status_radius: f32,       // 상태 줄 모서리 둥글기
    status_text: u32,
    panel_bg: u32,            // 문장 판 바탕
    panel_border: u32,
    panel_radius: f32,
    panel_margin: f32,        // 문장 판 바깥 여백
    text: u32,                // 문장 글자
    time: u32,                // 시간 글자
    error: u32,
    dot_ready: u32,           // 상태 점: 듣는 중
    dot_wait: u32,            // 상태 점: 준비 중
    button_bg: u32,           // 테마 바꾸기 버튼
    button_text: u32,
}

/// 애플 리퀴드 글래스 느낌: 뒤가 흐릿하게 비치는 유리 + 글자 판은 거의 불투명
const GLASS: Theme = Theme {
    name: "Glass",
    window_bg: WindowBackgroundAppearance::Blurred,
    root_bg: 0x10101450,
    root_padding: 12.,
    status_bg: 0xffffff14,
    status_border: 0xffffff2e,
    status_radius: 12.,
    status_text: 0xe5e5eaff,
    panel_bg: 0x1c1c20e6,
    panel_border: 0xffffff1a,
    panel_radius: 16.,
    panel_margin: 0.,
    text: 0xf5f5f7ff,
    time: 0x8e8e93ff,
    error: 0xff6b6bff,
    dot_ready: 0x30d158ff,
    dot_wait: 0xffd60aff,
    button_bg: 0xffffff1f,
    button_text: 0xf5f5f7ff,
};

/// 구글 머티리얼 디자인 3 (다크): 불투명한 면, 테두리 대신 색의 밝기 차이로 층을 구분
const MATERIAL: Theme = Theme {
    name: "Material",
    window_bg: WindowBackgroundAppearance::Opaque,
    root_bg: 0x141218ff,      // surface
    root_padding: 16.,        // 창 테두리(제목 막대)와 띄우기
    status_bg: 0x2b2930ff,    // surface container high (상단 바: 문장 카드보다 한 단계 밝게)
    status_border: 0x00000000,
    status_radius: 12.,
    status_text: 0xe6e0e9ff,  // on-surface
    panel_bg: 0x211f26ff,     // surface container (카드)
    panel_border: 0x00000000,
    panel_radius: 12.,
    panel_margin: 0.,
    text: 0xe6e0e9ff,
    time: 0xd0bcffff,         // primary (보라)
    error: 0xf2b8b5ff,        // error
    dot_ready: 0x7bd88fff,
    dot_wait: 0xffd479ff,
    button_bg: 0xd0bcffff,    // primary 채운 버튼
    button_text: 0x381e72ff,  // on-primary
};

impl Render for MainView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = self.theme;

        div()
            .size_full()
            .flex()
            .flex_col()
            .p(px(t.root_padding))
            .gap(px(t.root_padding))
            .bg(rgba(t.root_bg))
            .text_color(rgba(t.text))
            // 위쪽 상태 줄
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .px_4()
                    .py_2()
                    .rounded(px(t.status_radius))
                    .bg(rgba(t.status_bg))
                    .border_1()
                    .border_color(rgba(t.status_border))
                    .child(
                        // 상태 점: 듣는 중이면 초록, 아니면 노랑
                        div()
                            .size_2()
                            .rounded_full()
                            .bg(rgba(if self.listening { t.dot_ready } else { t.dot_wait })),
                    )
                    .child(
                        div()
                            .flex_1()
                            .text_sm()
                            .text_color(rgba(t.status_text))
                            .child(self.status.clone()),
                    )
                    // 테마 바꾸기 버튼 (비교용)
                    .child(
                        div()
                            .id("theme-toggle")
                            .px_3()
                            .py_1()
                            .rounded_full()
                            .bg(rgba(t.button_bg))
                            .text_color(rgba(t.button_text))
                            .text_xs()
                            .cursor_pointer()
                            .child(format!("디자인: {} ⇄", t.name))
                            .on_click(cx.listener(|view, _event, window, cx| {
                                view.theme = if view.theme.name == GLASS.name { &MATERIAL } else { &GLASS };
                                window.set_background_appearance(view.theme.window_bg);
                                cx.notify();
                            })),
                    ),
            )
            // 전사된 문장 판
            .child(
                div()
                    .id("lines")
                    .flex_1()
                    .min_h(px(0.))
                    .mx(px(t.panel_margin))
                    .mb(px(t.panel_margin))
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .rounded(px(t.panel_radius))
                    .bg(rgba(t.panel_bg))
                    .border_1()
                    .border_color(rgba(t.panel_border))
                    .shadow_lg()
                    .px_5()
                    .py_4()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .children(self.lines.iter().map(|line| {
                        div()
                            .flex()
                            .items_baseline()
                            .gap_3()
                            .child(div().w(px(64.)).text_sm().text_color(rgba(t.time)).child(line.time.clone()))
                            .child(
                                div()
                                    .flex_1()
                                    .text_lg()
                                    .text_color(rgba(if line.is_error { t.error } else { t.text }))
                                    .child(line.msg.clone()),
                            )
                    })),
            )
    }
}

// ------------------------------------------------------------ 시작

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(800.), px(600.)), cx);

        // 마지막 창을 닫으면 앱도 종료
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some("qwen-live".into()),
                    ..Default::default()
                }),
                // 시작 테마의 창 배경 (Glass면 뒤가 흐릿하게 비침)
                window_background: start_theme().window_bg,
                ..Default::default()
            },
            |_, cx| cx.new(MainView::new),
        )
        .unwrap();

        cx.activate(true);
    });
}

/// 앱을 켰을 때 처음 보이는 테마. QWEN_THEME=material 로 실행하면 머티리얼로 시작.
fn start_theme() -> &'static Theme {
    match std::env::var("QWEN_THEME").as_deref() {
        Ok("material") => &MATERIAL,
        _ => &GLASS,
    }
}
