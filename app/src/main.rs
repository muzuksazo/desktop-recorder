//! Qwen3-ASR 1.7B KO — Liquid Glass Studio & Picture-in-Picture (PiP) Overlay
//!
//! Includes:
//! 1. **Bug-free PiP ↔ Studio Transition (`macos_glass`)**:
//!    - Synchronously resizes `NSWindow`, `NSGlassEffectView`, and GPUI's `native_view`
//!      (calling `setFrameSize:` directly on `native_view` so GPUI updates its Metal
//!      drawable and fires `resize_callback`), plus `window.resize(...)` and `window.refresh()`.
//!    - Restores previous Studio window frame when expanding back from PiP.
//! 2. **Unified Outer Capsule Container (PiP & Studio)**:
//!    - Both PiP (`480x142`) and Studio (`860x580`) wrap their entire UI inside a single
//!      concentric rounded `26px` Liquid Glass shell with generous inner padding (`px_5`, `py_4`)
//!      so bottom streaming text never collides with the window's outer corner radius.
//! 3. **Long Text Multi-Line Wrapping**:
//!    - Removed single-line `.truncate()` on transcript cards and live streaming captions so
//!      long Korean sentences wrap naturally across multiple lines.
//! 4. **Drag-to-Select & Copy (`MouseDown` + `MouseMove` + `MouseUp` + `Cmd+C`)**:
//!    - Dragging across one or more transcript cards highlights the range and copies all
//!      selected sentences to the system clipboard automatically on mouse release (or `Cmd+C`).
//! 5. **100% Vector SVG Icons (`MemoryAssetSource`)**:
//!    - Replaced all emojis with crisp embedded SVG icons rendered via GPUI's `svg()`.

#[allow(dead_code)]
mod tts;
mod updates;

use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    env, fs,
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use futures::{channel::mpsc, StreamExt};
use gpui::{
    div, relative, AnyElement, Div, Stateful, hsla, linear_color_stop, linear_gradient, point, prelude::*, px, rgba, size, svg, App,
    Application, AssetSource, Bounds, BoxShadow, ClipboardItem, Context, IntoElement, KeyDownEvent,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Render, Result as GpuiResult,
    ScrollHandle, ScrollWheelEvent, SharedString, TitlebarOptions, Window, WindowBackgroundAppearance, WindowBounds,
    WindowOptions,
};
use serde::Deserialize;
use updates::{fmt_mb, ModelSpec, ModelState, Release};

/// Maximum number of committed transcript cards retained in UI memory.
const MAX_TRANSCRIPT_ITEMS: usize = 300;
/// Maximum number of status log entries retained.
const MAX_STATUS_ENTRIES: usize = 16;

// ---------------------------------------------------------------------------
// Embedded Vector SVG Icons (Zero Emojis)
// ---------------------------------------------------------------------------

struct EmbeddedSvgAssets;

impl AssetSource for EmbeddedSvgAssets {
    fn load(&self, path: &str) -> GpuiResult<Option<Cow<'static, [u8]>>> {
        let svg_str: &'static str = match path {
            "icons/download.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><polyline points="7 10 12 15 17 10"/><path d="M5 21h14"/></svg>"#
            }
            "icons/folder.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg>"#
            }
            "icons/cpu.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>"#
            }
            "icons/pip.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="3"/><rect x="12" y="11" width="8" height="7" rx="1.5" fill="currentColor"/></svg>"#
            }
            "icons/expand.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>"#
            }
            "icons/lock.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>"#
            }
            "icons/unlock.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></svg>"#
            }
            "icons/copy.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>"#
            }
            "icons/check.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>"#
            }
            "icons/pause.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16" rx="1.5"/><rect x="14" y="4" width="4" height="16" rx="1.5"/></svg>"#
            }
            "icons/play.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>"#
            }
            "icons/trash.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>"#
            }
            "icons/mic.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>"#
            }
            "icons/sparkles.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>"#
            }
            "icons/talkit_logo.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><rect x="1.2" y="8.8" width="3.2" height="6.4" rx="1.6"/><rect x="5.8" y="5.2" width="3.2" height="13.6" rx="1.6"/><rect x="10.4" y="2.0" width="3.2" height="20.0" rx="1.6"/><rect x="15.0" y="5.2" width="3.2" height="13.6" rx="1.6"/><rect x="19.6" y="8.8" width="3.2" height="6.4" rx="1.6"/></svg>"#
            }
            "icons/history.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>"#
            }
            "icons/arrow_down.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v15"/><path d="m6 13 6 6 6-6"/></svg>"#
            }
            "icons/back.svg" => {
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>"#
            }
            _ => return Ok(None),
        };
        Ok(Some(Cow::Borrowed(svg_str.as_bytes())))
    }

    fn list(&self, _path: &str) -> GpuiResult<Vec<SharedString>> {
        Ok(vec![])
    }
}

fn icon_svg(path: &'static str, size_px: f32, color_rgba: u32) -> impl IntoElement {
    svg()
        .path(path)
        .size(px(size_px))
        .text_color(rgba(color_rgba))
        .flex_none()
}

// ---------------------------------------------------------------------------
// Talkit Official Brand Identity Palette & Theme System
// - Talkit Blue : #4F6BFF
// - Text Black  : #1A1A1A
// - Gray        : #E9EBF7
// - Slogan      : "Talk it, Get it." / "말하는 순간, 바로 텍스트로."
// ---------------------------------------------------------------------------

pub const TALKIT_BLUE: u32 = 0x4f6bffff;
pub const TALKIT_BLACK: u32 = 0x1a1a1aff;
pub const TALKIT_GRAY: u32 = 0xe9ebf7ff;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GlassPreset {
    TalkitLight,
    TalkitDark,
    TalkitBlue,
    ClearGlass,
}

#[derive(Clone, Copy)]
struct BrandTheme {
    is_light: bool,
    shell_bg_top: u32,
    shell_bg_bottom: u32,
    shell_border: u32,
    shell_active_border: u32,
    header_bg_top: u32,
    header_bg_bottom: u32,
    header_border: u32,
    strip_bg: u32,
    strip_border: u32,
    card_bg_top: u32,
    card_bg_bottom: u32,
    card_border: u32,
    card_hover_bg: u32,
    text_primary: u32,
    text_secondary: u32,
    text_muted: u32,
    pill_bg: u32,
    pill_border: u32,
    pill_hover_bg: u32,
    wave_idle: u32,
}

impl GlassPreset {
    fn all() -> [GlassPreset; 4] {
        [
            GlassPreset::TalkitLight,
            GlassPreset::TalkitDark,
            GlassPreset::TalkitBlue,
            GlassPreset::ClearGlass,
        ]
    }

    fn label(&self) -> &'static str {
        match self {
            GlassPreset::TalkitLight => "White",
            GlassPreset::TalkitDark => "Black",
            GlassPreset::TalkitBlue => "Blue",
            GlassPreset::ClearGlass => "Glass",
        }
    }

    fn theme(&self) -> BrandTheme {
        match self {
            // Matches the main Brand Identity Sheet & bottom-right white pill widget (#FFFFFF / #E9EBF7 / #1A1A1A / #4F6BFF)
            GlassPreset::TalkitLight => BrandTheme {
                is_light: true,
                shell_bg_top: 0xf9fafee8,
                shell_bg_bottom: 0xe9ebf7eb,
                shell_border: 0xd8ddf2ff,
                shell_active_border: 0x4f6bffaa,
                header_bg_top: 0xffffffff,
                header_bg_bottom: 0xf4f6fcff,
                header_border: 0xdce1f5ff,
                strip_bg: 0xe9ebf7b8,
                strip_border: 0xd5dbf2ff,
                card_bg_top: 0xffffffff,
                card_bg_bottom: 0xf7f8fdff,
                card_border: 0xdce1f5ff,
                card_hover_bg: 0xe9ebf7cc,
                text_primary: TALKIT_BLACK,
                text_secondary: 0x1a1a1ab8,
                text_muted: 0x1a1a1a78,
                pill_bg: TALKIT_GRAY,
                pill_border: 0xd2d8f0ff,
                pill_hover_bg: 0xdbe0f6ff,
                wave_idle: 0xb8c4f8ff,
            },
            // Matches the #1A1A1A Text Black Logo Variation in the Brand Identity Guide
            GlassPreset::TalkitDark => BrandTheme {
                is_light: false,
                shell_bg_top: 0x1a1a1ae8,
                shell_bg_bottom: 0x111217f2,
                shell_border: 0xe9ebf726,
                shell_active_border: 0x4f6bffaa,
                header_bg_top: 0x262730ee,
                header_bg_bottom: 0x1a1a1aee,
                header_border: 0xe9ebf729,
                strip_bg: 0x0f101588,
                strip_border: 0xe9ebf71a,
                card_bg_top: 0x252730cc,
                card_bg_bottom: 0x1a1a1acc,
                card_border: 0xe9ebf724,
                card_hover_bg: 0x2f3240dd,
                text_primary: 0xffffffff,
                text_secondary: 0xe9ebf7cc,
                text_muted: 0xe9ebf788,
                pill_bg: 0xe9ebf718,
                pill_border: 0xe9ebf72b,
                pill_hover_bg: 0xe9ebf72c,
                wave_idle: 0xe9ebf755,
            },
            // Talkit Blue (#4F6BFF) Brand Tinted Liquid Glass
            GlassPreset::TalkitBlue => BrandTheme {
                is_light: false,
                shell_bg_top: 0x1e295ee0,
                shell_bg_bottom: 0x121738eb,
                shell_border: 0x4f6bff77,
                shell_active_border: 0x8094ffff,
                header_bg_top: 0x4f6bff38,
                header_bg_bottom: 0x1a1a1a88,
                header_border: 0x4f6bff66,
                strip_bg: 0x11163377,
                strip_border: 0xe9ebf722,
                card_bg_top: 0x4f6bff28,
                card_bg_bottom: 0x1a1a1a88,
                card_border: 0xe9ebf728,
                card_hover_bg: 0x4f6bff3d,
                text_primary: 0xffffffff,
                text_secondary: 0xe9ebf7d8,
                text_muted: 0xe9ebf790,
                pill_bg: 0xe9ebf71c,
                pill_border: 0xe9ebf733,
                pill_hover_bg: 0x4f6bff44,
                wave_idle: 0xe9ebf766,
            },
            // Pure translucent Liquid Glass with Talkit Blue & Gray accents
            GlassPreset::ClearGlass => BrandTheme {
                is_light: false,
                shell_bg_top: 0x1a1a1a66,
                shell_bg_bottom: 0x1a1a1a88,
                shell_border: 0xe9ebf733,
                shell_active_border: 0x4f6bffaa,
                header_bg_top: 0xe9ebf724,
                header_bg_bottom: 0xe9ebf70e,
                header_border: 0xe9ebf730,
                strip_bg: 0x1a1a1a44,
                strip_border: 0xe9ebf71e,
                card_bg_top: 0xe9ebf722,
                card_bg_bottom: 0xe9ebf70c,
                card_border: 0xe9ebf72b,
                card_hover_bg: 0xe9ebf72e,
                text_primary: 0xffffffff,
                text_secondary: 0xe9ebf7cc,
                text_muted: 0xe9ebf788,
                pill_bg: 0xe9ebf71a,
                pill_border: 0xe9ebf72e,
                pill_hover_bg: 0xe9ebf72e,
                wave_idle: 0xe9ebf755,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// macOS 26 Native NSGlassEffectView + Bug-Free PiP Window Transition Bridge
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[allow(unexpected_cfgs)]
mod macos_glass {
    use super::GlassPreset;
    use cocoa::{
        appkit::{NSColor, NSView, NSViewHeightSizable, NSViewWidthSizable},
        base::{id, nil, BOOL, NO, YES},
        foundation::{NSPoint, NSRect, NSSize, NSString},
    };
    use gpui::{Window, WindowBackgroundAppearance};
    use objc::{
        class, msg_send,
        runtime::{Class, Object},
        sel, sel_impl,
    };
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use std::cell::RefCell;
    use std::collections::HashMap;

    pub const STUDIO_WIDTH: f32 = 860.0;
    pub const STUDIO_HEIGHT: f32 = 580.0;
    pub const PIP_WIDTH: f32 = 500.0;
    pub const PIP_HEIGHT: f32 = 148.0;

    thread_local! {
        /// Maps `native_window` pointer -> installed `NSGlassEffectView` pointer
        static INSTALLED_GLASS: RefCell<HashMap<usize, usize>> = RefCell::new(HashMap::new());
        /// Maps `native_window` pointer -> saved Studio `NSRect` before entering PiP
        static SAVED_STUDIO_FRAME: RefCell<HashMap<usize, NSRect>> = RefCell::new(HashMap::new());
    }

    pub fn resolve_native_ptrs(window: &mut Window) -> Option<(usize, usize)> {
        unsafe {
            let handle = HasWindowHandle::window_handle(window).ok()?;
            let RawWindowHandle::AppKit(appkit_handle) = handle.as_raw() else {
                return None;
            };
            let native_view: id = appkit_handle.ns_view.as_ptr() as id;
            if native_view == nil {
                return None;
            }
            let native_window: id = msg_send![native_view, window];
            if native_window == nil {
                return None;
            }
            Some((native_window as usize, native_view as usize))
        }
    }

    /// Initiates a native macOS window drag from a dedicated header handle area
    /// (only when dragging is allowed — disabled when PiP is locked).
    pub fn start_window_drag(window: &mut Window, can_move: bool) {
        if !can_move {
            return;
        }
        unsafe {
            let Some((win_ptr, _)) = resolve_native_ptrs(window) else {
                return;
            };
            let native_window = win_ptr as id;
            let app: id = msg_send![class!(NSApplication), sharedApplication];
            if app == nil {
                return;
            }
            let current_event: id = msg_send![app, currentEvent];
            if current_event != nil {
                let _: () = msg_send![native_window, performWindowDragWithEvent: current_event];
            }
        }
    }

    /// Updates window level and movability:
    /// - Studio mode (`pip_mode == false`): Normal window level (`0`), movable via header/titlebar,
    ///   `movableByWindowBackground = NO` so dragging inside content always selects/copies text.
    /// - PiP mode (`pip_mode == true`): Floating overlay level (`25`), `movable = !pip_locked`,
    ///   `movableByWindowBackground = NO` so locked PiP never moves and dragging text always copies.
    pub unsafe fn sync_window_behavior(native_window: id, pip_mode: bool, pip_locked: bool) {
        let level: isize = if pip_mode { 25 } else { 0 };
        let _: () = msg_send![native_window, setLevel: level];

        let movable: BOOL = if pip_mode {
            if pip_locked { NO } else { YES }
        } else {
            YES
        };
        let _: () = msg_send![native_window, setMovable: movable];
        // Never let window background hijack mouse drags over transcript cards or subtitles!
        let _: () = msg_send![native_window, setMovableByWindowBackground: NO];

        if pip_mode {
            let behavior: usize = (1 << 0) | (1 << 8);
            let _: () = msg_send![native_window, setCollectionBehavior: behavior];
        }
    }

    pub fn apply_window_glass(
        window: &mut Window,
        preset: GlassPreset,
        pip_mode: bool,
        pip_locked: bool,
    ) {
        unsafe {
            let Some((win_ptr, view_ptr)) = resolve_native_ptrs(window) else {
                return;
            };
            let native_window = win_ptr as id;
            let native_view = view_ptr as id;

            sync_window_behavior(native_window, pip_mode, pip_locked);

            let Some(glass_class) = Class::get("NSGlassEffectView") else {
                window.set_background_appearance(WindowBackgroundAppearance::Blurred);
                return;
            };

            let win_key = native_window as usize;
            let glass_view: id = INSTALLED_GLASS.with(|map| {
                if let Some(&existing) = map.borrow().get(&win_key) {
                    return existing as id;
                }

                // Keep GPUI's `native_window.contentView()` and `native_view` intact!
                // Insert `NSGlassEffectView` behind `native_view` (`NSWindowBelow = -1`),
                // matching GPUI's own `BlurredView` subview architecture.
                let content_view: id = msg_send![native_window, contentView];
                let bounds = NSView::bounds(content_view);

                let mut gv: id = msg_send![glass_class, alloc];
                gv = NSView::initWithFrame_(gv, bounds);
                gv.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable);

                let _: () = msg_send![
                    content_view,
                    addSubview: gv
                    positioned: -1isize
                    relativeTo: native_view
                ];
                let _: BOOL = msg_send![native_window, makeFirstResponder: native_view];

                let _: () = msg_send![native_window, setOpaque: NO];
                let bg_color =
                    NSColor::colorWithSRGBRed_green_blue_alpha_(nil, 0.0, 0.0, 0.0, 0.0001);
                let _: () = msg_send![native_window, setBackgroundColor: bg_color];

                if let Some(app_cls) = Class::get("NSAppearance") {
                    let dark_str: id = NSString::alloc(nil).init_str("NSAppearanceNameDarkAqua");
                    let appearance: id = msg_send![app_cls, appearanceNamed: dark_str];
                    let _: () = msg_send![gv, setAppearance: appearance];
                }

                let _ = std::ptr::null::<Object>();
                map.borrow_mut().insert(win_key, gv as usize);
                gv
            });

            if let Some(app_cls) = Class::get("NSAppearance") {
                let appearance_name = match preset {
                    GlassPreset::TalkitLight => "NSAppearanceNameAqua",
                    _ => "NSAppearanceNameDarkAqua",
                };
                let app_str: id = NSString::alloc(nil).init_str(appearance_name);
                let appearance: id = msg_send![app_cls, appearanceNamed: app_str];
                let _: () = msg_send![glass_view, setAppearance: appearance];
            }

            // Style: 0 = Regular, 1 = Clear
            let style: isize = match preset {
                GlassPreset::TalkitLight | GlassPreset::TalkitDark | GlassPreset::TalkitBlue => 0,
                GlassPreset::ClearGlass => 1,
            };
            // Talkit Brand Colors: Talkit Blue (#4F6BFF -> R=0.31, G=0.42, B=1.0), Gray (#E9EBF7)
            let tint_color: id = match preset {
                GlassPreset::TalkitLight => {
                    NSColor::colorWithSRGBRed_green_blue_alpha_(nil, 0.914, 0.922, 0.969, 0.45)
                }
                GlassPreset::TalkitDark => {
                    NSColor::colorWithSRGBRed_green_blue_alpha_(nil, 0.102, 0.102, 0.102, 0.55)
                }
                GlassPreset::TalkitBlue => {
                    NSColor::colorWithSRGBRed_green_blue_alpha_(nil, 0.310, 0.420, 1.000, 0.36)
                }
                GlassPreset::ClearGlass => nil,
            };

            let _: () = msg_send![glass_view, setStyle: style];
            let _: () = msg_send![glass_view, setTintColor: tint_color];
            let _: () = msg_send![glass_view, setCornerRadius: 0.0f64];
        }
    }

    /// Must be called on the main thread when GPUI's `Window` is NOT borrowed (e.g. inside
    /// `cx.foreground_executor().spawn(...)`), so that when AppKit invokes `setFrameSize:` on
    /// `native_view` (`GPUIView`), GPUI's `resize_callback` (`handle.update` -> `window.bounds_changed`)
    /// succeeds without any lease conflict.
    pub fn perform_pip_transition_unlocked(
        win_ptr: usize,
        view_ptr: usize,
        pip_mode: bool,
        pip_locked: bool,
    ) {
        let (target_w, target_h) = if pip_mode {
            (PIP_WIDTH as f64, PIP_HEIGHT as f64)
        } else {
            (STUDIO_WIDTH as f64, STUDIO_HEIGHT as f64)
        };

        unsafe {
            let native_window = win_ptr as id;
            let native_view = view_ptr as id;
            if native_window == nil || native_view == nil {
                return;
            }

            let cur_frame: NSRect = msg_send![native_window, frame];
            let new_frame = if pip_mode {
                if cur_frame.size.width > 540.0 && cur_frame.size.height > 240.0 {
                    SAVED_STUDIO_FRAME.with(|m| {
                        m.borrow_mut().insert(win_ptr, cur_frame);
                    });
                }
                let top_y = cur_frame.origin.y + cur_frame.size.height;
                let right_x = cur_frame.origin.x + cur_frame.size.width;
                let new_x = (right_x - target_w).max(24.0);
                let new_y = (top_y - target_h).max(24.0);
                NSRect::new(NSPoint::new(new_x, new_y), NSSize::new(target_w, target_h))
            } else {
                let saved = SAVED_STUDIO_FRAME.with(|m| m.borrow().get(&win_ptr).copied());
                if let Some(prev) = saved {
                    let w = prev.size.width.max(target_w);
                    let h = prev.size.height.max(target_h);
                    NSRect::new(prev.origin, NSSize::new(w, h))
                } else {
                    let top_y = cur_frame.origin.y + cur_frame.size.height;
                    let right_x = cur_frame.origin.x + cur_frame.size.width;
                    let new_x = (right_x - target_w).max(24.0);
                    let new_y = (top_y - target_h).max(24.0);
                    NSRect::new(NSPoint::new(new_x, new_y), NSSize::new(target_w, target_h))
                }
            };

            // Show/hide standard macOS traffic lights (0=Close, 1=Miniaturize, 2=Zoom)
            for btn_idx in 0usize..=2usize {
                let btn: id = msg_send![native_window, standardWindowButton: btn_idx];
                if btn != nil {
                    let hidden: BOOL = if pip_mode { YES } else { NO };
                    let _: () = msg_send![btn, setHidden: hidden];
                }
            }

            sync_window_behavior(native_window, pip_mode, pip_locked);

            // Resize NSWindow synchronously (`animate: NO`) while GPUI's `Window` is unlocked
            let _: () = msg_send![native_window, setFrame: new_frame display: YES animate: NO];

            // Guarantee `GPUIView::set_frame_size` sees a size change while `content_view` has `new_frame.size`
            let final_size = new_frame.size;
            let content_view: id = msg_send![native_window, contentView];
            if content_view != nil {
                let cv_rect = NSRect::new(NSPoint::new(0.0, 0.0), final_size);
                let _: () = msg_send![content_view, setFrame: cv_rect];
            }
            INSTALLED_GLASS.with(|map| {
                if let Some(&gv_ptr) = map.borrow().get(&win_ptr) {
                    let gv = gv_ptr as id;
                    let gv_rect = NSRect::new(NSPoint::new(0.0, 0.0), final_size);
                    let _: () = msg_send![gv, setFrame: gv_rect];
                }
            });

            // Force GPUI's `set_frame_size` to fire `resize_callback` (`window.bounds_changed`)
            let cur_view_frame: NSRect = msg_send![native_view, frame];
            if (cur_view_frame.size.width - final_size.width).abs() < 0.5
                && (cur_view_frame.size.height - final_size.height).abs() < 0.5
            {
                let nudge = NSSize::new(final_size.width - 1.0, final_size.height);
                let _: () = msg_send![native_view, setFrameSize: nudge];
            }
            let _: () = msg_send![native_view, setFrameSize: final_size];
            let _: BOOL = msg_send![native_window, makeFirstResponder: native_view];
        }
    }
}

// ---------------------------------------------------------------------------
// Specular Rim & Concentric Capsule Shadows (`sxhxliang/gpui-liquid-glass`)
// ---------------------------------------------------------------------------

fn glass_card_shadows() -> Vec<BoxShadow> {
    vec![
        BoxShadow {
            color: hsla(0.0, 0.0, 0.0, 0.24),
            offset: point(px(0.0), px(10.0)),
            blur_radius: px(24.0),
            spread_radius: px(-3.0),
        },
        BoxShadow {
            color: hsla(0.0, 0.0, 1.0, 0.20),
            offset: point(px(0.0), px(1.0)),
            blur_radius: px(0.5),
            spread_radius: px(0.0),
        },
        BoxShadow {
            color: hsla(0.0, 0.0, 0.0, 0.14),
            offset: point(px(0.0), px(-1.0)),
            blur_radius: px(1.0),
            spread_radius: px(0.0),
        },
    ]
}

fn glass_pill_shadows() -> Vec<BoxShadow> {
    vec![
        BoxShadow {
            color: hsla(0.0, 0.0, 0.0, 0.20),
            offset: point(px(0.0), px(4.0)),
            blur_radius: px(12.0),
            spread_radius: px(-1.0),
        },
        BoxShadow {
            color: hsla(0.0, 0.0, 1.0, 0.26),
            offset: point(px(0.0), px(1.0)),
            blur_radius: px(0.5),
            spread_radius: px(0.0),
        },
    ]
}

// ---------------------------------------------------------------------------
// Python Backend Subprocess Protocol
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[allow(dead_code)]
enum BackendMsg {
    Status {
        msg: String,
    },
    Partial {
        text: String,
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
    Level {
        rms: f32,
    },
    Telemetry {
        rms: f32,
        mem_mb: f32,
    },
    /// 이 Mac 에서 쓸 수 있는 모델 목록과 지금 쓰는 모델 (백엔드가 시작할 때 보냄)
    Models {
        items: Vec<ModelItem>,
        current: String,
    },
    /// 모델 내려받기 진행 (backend.py --download-model)
    Download {
        got: u64,
        total: Option<u64>,
    },
    #[serde(rename = "download_done")]
    DownloadDone {
        path: String,
    },
}

/// 자동 업데이트 쪽 일꾼 스레드가 화면에 보내는 소식
enum UpdEvent {
    Found(Release),
    Progress(u64),
    Ready(PathBuf),
    Failed(String),
}

/// 화면에 보이는 업데이트 상태
#[derive(Clone)]
enum UpdateState {
    None,
    Available(Release),
    Downloading { rel: Release, got: u64 },
    Ready { rel: Release, staged: PathBuf },
    Failed { rel: Option<Release>, msg: String },
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)] // 백엔드가 보내는 목록. 고르기는 앱의 updates::CHOICES 로 한다
struct ModelItem {
    id: String,
    label: String,
}

enum Event {
    Msg(BackendMsg),
    Exited,
}

/// 백엔드를 띄울 Python 을 고른다. 어느 Mac 에서든 동작하도록 사용자 이름이 박힌 경로는 쓰지 않는다.
/// 1) RECORDER_PYTHON (개발용)
/// 2) 비공개 엔진 애드온 폴더의 python-path.txt (그 엔진은 torch 가 필요해 자기 Python 을 쓴다)
/// 3) 앱 안에 넣은 Python (Contents/Resources/python) — 배포된 앱의 기본
/// 4) 소스에서 실행할 때의 .venv
fn resolve_python_bin() -> PathBuf {
    if let Some(p) = env::var_os("RECORDER_PYTHON") {
        return PathBuf::from(p);
    }
    // 외부 배포판(Resources/edition.txt = public)은 비공개 엔진 애드온을 쓰지 않는다
    let public_edition = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|d| d.join("../Resources/edition.txt")))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| s.trim() == "public")
        .unwrap_or(false);
    if let (false, Some(home)) = (public_edition, env::var_os("HOME")) {
        let home = PathBuf::from(home);
        let addon_dirs = [
            home.join("Library/Application Support/Talkit/engines/rung-omega"),
            home.join("qwen3-asr-ko/private_rung_omega_mac"),
        ];
        for dir in addon_dirs {
            if let Ok(text) = std::fs::read_to_string(dir.join("python-path.txt")) {
                let p = PathBuf::from(text.trim());
                if p.exists() {
                    return p;
                }
            }
        }
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(macos_dir) = exe.parent() {
            let bundled = macos_dir.join("../Resources/python/bin/python3");
            if bundled.exists() {
                return bundled;
            }
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let candidates = [
        root.join(".venv").join("bin").join("python"),
        root.join(".venv").join("Scripts").join("python.exe"),
        PathBuf::from("/opt/homebrew/bin/python3"),
        PathBuf::from("python3"),
        PathBuf::from("python"),
    ];
    for c in candidates {
        if c.exists() {
            return c;
        }
    }
    PathBuf::from("python3")
}

fn resolve_backend_script() -> PathBuf {
    if let Some(s) = env::var_os("RECORDER_BACKEND") {
        return PathBuf::from(s);
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(macos_dir) = exe.parent() {
            let bundled = macos_dir.join("..").join("Resources").join("backend.py");
            if bundled.exists() {
                return bundled;
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("backend.py")
}

fn spawn_backend(
    tx: mpsc::UnboundedSender<Event>,
    source: SourceMode,
    model_choice: &str,
    out_path: Option<&PathBuf>,
) -> std::io::Result<Child> {
    let python = resolve_python_bin();
    let script = resolve_backend_script();

    let mut cmd = Command::new(&python);
    cmd.args(["-X", "utf8"])
        .arg(&script)
        .args(env::args().skip(1))
        // 뒤에 붙인 값이 이긴다 (argparse 는 마지막 값을 씀)
        .args(["--source", source.arg()]);
    // 입력 전환으로 백엔드를 다시 띄울 때도 같은 세션 파일에 이어 쓴다
    if let Some(p) = out_path {
        cmd.arg("--out").arg(p);
    }
    // 설정에서 고른 모델 (비어 있으면 백엔드가 그 Mac 에 맞는 기본값을 고른다)
    if !model_choice.is_empty() {
        cmd.args(["--model-choice", model_choice]);
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdout = child.stdout.take().expect("stdout 을 piped 로 열었음");
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            let msg = serde_json::from_str(&line).unwrap_or(BackendMsg::Status { msg: line });
            if tx.unbounded_send(Event::Msg(msg)).is_err() {
                return;
            }
        }
        let _ = tx.unbounded_send(Event::Exited);
    });
    Ok(child)
}

/// 모델 하나만 내려받는 백엔드를 띄운다 (모델 관리 화면용)
fn scan_all_models() -> HashMap<&'static str, ModelState> {
    updates::MODELS.iter().map(|m| (m.id, updates::scan_model(m))).collect()
}

fn spawn_model_download(tx: mpsc::UnboundedSender<Event>, model_id: &str) -> std::io::Result<Child> {
    let mut cmd = Command::new(resolve_python_bin());
    cmd.args(["-X", "utf8"])
        .arg(resolve_backend_script())
        .args(["--download-model", model_id])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = cmd.spawn()?;
    let stdout = child.stdout.take().expect("stdout 을 piped 로 열었음");
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let Ok(msg) = serde_json::from_str::<BackendMsg>(&line) else { continue };
            if tx.unbounded_send(Event::Msg(msg)).is_err() {
                return;
            }
        }
        let _ = tx.unbounded_send(Event::Exited);
    });
    Ok(child)
}

// ---------------------------------------------------------------------------
// Main Mobbin-Inspired Liquid Glass & PiP View
// ---------------------------------------------------------------------------

/// 무엇을 녹음할지: 마이크 / 시스템 소리 / 둘 다
#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceMode {
    Mic,
    System,
    Both,
}

impl SourceMode {
    fn all() -> [SourceMode; 3] {
        [SourceMode::Mic, SourceMode::System, SourceMode::Both]
    }
    fn arg(&self) -> &'static str {
        match self {
            SourceMode::Mic => "mic",
            SourceMode::System => "system",
            SourceMode::Both => "both",
        }
    }
    fn label(&self) -> &'static str {
        match self {
            SourceMode::Mic => "마이크",
            SourceMode::System => "시스템",
            SourceMode::Both => "둘 다",
        }
    }
    fn from_arg(s: &str) -> Option<Self> {
        match s.trim() {
            "mic" => Some(SourceMode::Mic),
            "system" => Some(SourceMode::System),
            "both" => Some(SourceMode::Both),
            _ => None,
        }
    }
}

fn settings_path() -> Option<PathBuf> {
    env::var_os("HOME").map(|h| {
        PathBuf::from(h)
            .join("Library")
            .join("Application Support")
            .join("Talkit")
            .join("settings.txt")
    })
}

fn load_source_mode() -> SourceMode {
    // 명령줄에 --source 를 줬으면 그것을 시작값으로 쓴다
    let args: Vec<String> = env::args().collect();
    if let Some(m) = args
        .windows(2)
        .find(|w| w[0] == "--source")
        .and_then(|w| SourceMode::from_arg(&w[1]))
    {
        return m;
    }
    read_setting("source")
        .and_then(|v| SourceMode::from_arg(&v))
        .unwrap_or(SourceMode::Both)
}

fn save_source_mode(mode: SourceMode) {
    write_setting("source", mode.arg());
}

/// settings.txt 의 key=value 한 줄을 읽는다
fn read_setting(key: &str) -> Option<String> {
    let text = fs::read_to_string(settings_path()?).ok()?;
    let prefix = format!("{key}=");
    text.lines()
        .find_map(|l| l.strip_prefix(prefix.as_str()).map(|v| v.trim().to_string()))
}

/// 다른 설정은 그대로 두고 key 한 줄만 바꾼다
fn write_setting(key: &str, value: &str) {
    let Some(p) = settings_path() else { return };
    if let Some(dir) = p.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let prefix = format!("{key}=");
    let mut lines: Vec<String> = fs::read_to_string(&p)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.starts_with(prefix.as_str()) && !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect();
    lines.push(format!("{key}={value}"));
    let _ = fs::write(p, lines.join("\n") + "\n");
}

// ---------------------------------------------------------------------------
// 이전 녹음 기록 (transcript_YYYY-MM-DD_HHMM.txt)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Live,
    HistoryList,
    HistoryDetail,
    /// 음성 인식 모델 관리 (내려받기·삭제)
    Models,
}

#[derive(Clone)]
struct HistoryEntry {
    path: PathBuf,
    title: SharedString,
    lines: usize,
    preview: SharedString,
}

#[derive(Clone)]
struct HistoryLine {
    clock: Option<SharedString>,
    source: Option<SharedString>,
    body: SharedString,
}

fn parse_history_line(line: &str) -> Option<HistoryLine> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let (clock, rest) = if line.starts_with('[') {
        match line.find("] ") {
            Some(idx) if !matches!(&line[1..idx], "시스템" | "마이크") => {
                (Some(SharedString::from(line[1..idx].to_string())), &line[idx + 2..])
            }
            _ => (None, line),
        }
    } else {
        (None, line)
    };
    let (source, body) = parse_source_prefix(rest);
    let body = body.trim();
    if body.is_empty() {
        return None;
    }
    Some(HistoryLine {
        clock,
        source,
        body: SharedString::from(body.to_string()),
    })
}

/// transcript_2026-11-05_1740.txt → "2026-11-05 17:40"
fn history_title(path: &PathBuf) -> SharedString {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let core = stem.strip_prefix("transcript_").unwrap_or(stem);
    if let Some((date, hm)) = core.split_once('_') {
        if hm.len() == 4 && hm.chars().all(|c| c.is_ascii_digit()) {
            return SharedString::from(format!("{} {}:{}", date, &hm[..2], &hm[2..]));
        }
    }
    SharedString::from(core.to_string())
}

fn scan_history(extra_dir: Option<PathBuf>) -> Vec<HistoryEntry> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(h) = env::var_os("HOME") {
        dirs.push(PathBuf::from(h).join("Documents").join("Talkit"));
    }
    if let Some(d) = extra_dir {
        dirs.push(d);
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for dir in dirs {
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        for ent in rd.flatten() {
            let path = ent.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !(name.starts_with("transcript_") && name.ends_with(".txt")) {
                continue;
            }
            let key = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            if !seen.insert(key) {
                continue;
            }
            let content = fs::read_to_string(&path).unwrap_or_default();
            let parsed: Vec<HistoryLine> = content.lines().filter_map(parse_history_line).collect();
            let preview = parsed
                .first()
                .map(|l| {
                    let s: String = l.body.chars().take(60).collect();
                    SharedString::from(s)
                })
                .unwrap_or_else(|| "(내용 없음)".into());
            out.push(HistoryEntry {
                title: history_title(&path),
                lines: parsed.len(),
                preview,
                path,
            });
        }
    }
    // 파일 이름이 시간순이라 이름 역순 = 최신순
    out.sort_by(|a, b| b.path.file_name().cmp(&a.path.file_name()));
    out
}

#[derive(Clone)]
struct TranscriptItem {
    index: usize,
    timestamp: Option<SharedString>,
    source: Option<SharedString>,
    body: SharedString,
    sentence_complete: bool,
    is_error: bool,
}

fn is_korean_sentence_end_word(word: &str, word_count_in_sentence: usize) -> bool {
    let trimmed = word.trim_end_matches(|c: char| matches!(c, '"' | '\'' | ')' | ']' | '”' | '’'));
    if trimmed.is_empty() {
        return false;
    }
    if let Some(last_ch) = trimmed.chars().last() {
        if matches!(last_ch, '?' | '!' | '…') {
            return true;
        }
        if last_ch == '.' {
            let stem = trimmed.trim_end_matches('.');
            let is_number_dot = !stem.is_empty() && stem.chars().all(|c| c.is_ascii_digit());
            if !is_number_dot {
                return true;
            }
        }
    }
    if word_count_in_sentence >= 3 {
        let clean = trimmed.trim_end_matches(|c: char| c.is_ascii_punctuation());
        const ENDINGS: &[&str] = &[
            "습니다", "ㅂ니다", "입니다", "였습니다", "겠습니다", "랍니다",
            "네요", "군요", "어요", "아요", "해요", "예요", "이에요", "거예요",
            "까요", "나요", "가요", "지요", "죠", "세", "십시오",
            "했다", "됐다", "한다", "된다", "있다", "없다", "이다", "였다", "겠다",
        ];
        if ENDINGS.iter().any(|e| clean.ends_with(e)) {
            return true;
        }
    }
    false
}

/// 연속 전사된 텍스트를 문장 단위로 깔끔하게 분리한다.
/// 반환값: `Vec<(문장 텍스트, 문장 종결 여부)>`
fn split_into_sentences(text: &str) -> Vec<(String, bool)> {
    let mut results = Vec::new();
    let mut current_words: Vec<&str> = Vec::new();

    for word in text.split_whitespace() {
        current_words.push(word);
        if is_korean_sentence_end_word(word, current_words.len()) {
            results.push((current_words.join(" "), true));
            current_words.clear();
        }
    }

    if !current_words.is_empty() {
        results.push((current_words.join(" "), false));
    }

    results
}

fn parse_source_prefix(s: &str) -> (Option<SharedString>, &str) {
    let trimmed = s.trim_start();
    if let Some(rest) = trimmed.strip_prefix("[시스템]") {
        return (Some(SharedString::from("시스템")), rest.trim_start());
    }
    if let Some(rest) = trimmed.strip_prefix("[마이크]") {
        return (Some(SharedString::from("마이크")), rest.trim_start());
    }
    (None, trimmed)
}

struct TranscriptView {
    preset: GlassPreset,
    glass_initialized: bool,
    pip_mode: bool,
    pip_locked: bool,
    paused: bool,
    started_at: Instant,
    total_committed: usize,
    copied_index: Option<usize>, // Some(0) = all, Some(usize::MAX - 2) = drag range, Some(idx) = card #idx
    // Drag-range selection state across sentence cards
    drag_anchor: Option<usize>,
    drag_current: Option<usize>,
    is_dragging_selection: bool,
    audio_level: f32,
    wave_history: VecDeque<f32>,
    mem_mb: f32,
    status: VecDeque<SharedString>,
    partial: SharedString,
    partial_sys: SharedString,
    partial_mic: SharedString,
    items: VecDeque<TranscriptItem>,
    scroll: ScrollHandle,
    /// 전사 따라가기: 켜져 있을 때만 새 전사가 오면 맨 아래로 내려간다
    follow_tail: bool,
    source_mode: SourceMode,
    /// 이 Mac 에서 쓸 수 있는 모델들 (백엔드가 알려 줌) 과 고른 모델 id
    models: Vec<ModelItem>,
    model_choice: String,
    /// 백엔드를 다시 띄울 때마다 증가. 이전 프로세스의 이벤트(종료 알림 등)는 무시한다
    backend_gen: u64,
    save_path: Option<PathBuf>,
    view_mode: ViewMode,
    history: Vec<HistoryEntry>,
    history_detail: Option<(HistoryEntry, Vec<HistoryLine>)>,
    history_copied: Option<usize>,
    history_scroll: ScrollHandle,
    /// 공개 모델 상태와 내려받는 프로세스
    sys_info: updates::SystemInfo,
    model_states: HashMap<&'static str, ModelState>,
    /// 지금 받고 있는 모델 id (한 번에 하나)
    downloading: Option<&'static str>,
    dl_gen: u64,
    _downloader: Option<Child>,
    /// 자동 업데이트
    update: UpdateState,
    upd_gen: u64,
    _backend: Option<Child>,
}

impl Drop for TranscriptView {
    fn drop(&mut self) {
        for mut child in [self._backend.take(), self._downloader.take()].into_iter().flatten() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl TranscriptView {
    fn new(cx: &mut Context<Self>) -> Self {
        let mut status = VecDeque::with_capacity(MAX_STATUS_ENTRIES);
        status.push_back("MLX 4-bit (1.07GB) 실시간 스트리밍 준비 중…".into());

        let mut wave_history = VecDeque::with_capacity(12);
        for _ in 0..12 {
            wave_history.push_back(0.08);
        }

        let sys_info = updates::detect_system();
        let model_states = scan_all_models();
        let saved_model = read_setting("model").unwrap_or_default();
        let saved_visible = !saved_model.is_empty()
            && updates::required_models(&saved_model)
                .iter()
                .all(|id| updates::spec(id).is_some_and(updates::is_model_visible));
        let model_choice = if saved_visible {
            saved_model
        } else {
            sys_info.recommended_choice().to_string()
        };

        let mut view = Self {
            preset: GlassPreset::TalkitLight,
            glass_initialized: false,
            pip_mode: false,
            pip_locked: false,
            paused: false,
            started_at: Instant::now(),
            total_committed: 0,
            copied_index: None,
            drag_anchor: None,
            drag_current: None,
            is_dragging_selection: false,
            audio_level: 0.0,
            wave_history,
            mem_mb: 1105.5,
            status,
            partial: "".into(),
            partial_sys: "".into(),
            partial_mic: "".into(),
            items: VecDeque::with_capacity(MAX_TRANSCRIPT_ITEMS),
            scroll: ScrollHandle::new(),
            follow_tail: true,
            source_mode: load_source_mode(),
            models: Vec::new(),
            model_choice,
            backend_gen: 0,
            save_path: None,
            view_mode: ViewMode::Live,
            history: Vec::new(),
            history_detail: None,
            history_copied: None,
            history_scroll: ScrollHandle::new(),
            sys_info,
            model_states,
            downloading: None,
            dl_gen: 0,
            _downloader: None,
            update: UpdateState::None,
            upd_gen: 0,
            _backend: None,
        };
        view.start_backend(cx);
        view.start_update_checks(cx);
        view
    }

    fn start_backend(&mut self, cx: &mut Context<Self>) {
        // 고른 모델이 아직 없으면 백엔드 대신 '모델 내려받기'부터 한다 (끝나면 자동으로 백엔드를 띄움)
        if !self.chosen_ready() {
            if let Some(id) = self.next_missing() {
                self.start_model_download(id, cx);
            }
            return;
        }
        let (tx, mut rx) = mpsc::unbounded();
        self.backend_gen += 1;
        let gen = self.backend_gen;
        match spawn_backend(tx, self.source_mode, &self.model_choice, self.save_path.as_ref()) {
            Ok(child) => self._backend = Some(child),
            Err(e) => self.push_line(format!("백엔드 실행 실패: {e} (RECORDER_PYTHON 확인)"), true),
        }

        cx.spawn(async move |this, cx| {
            while let Some(event) = rx.next().await {
                let alive = this.update(cx, |view, cx| {
                    if view.backend_gen != gen {
                        return; // 이미 교체된 백엔드
                    }
                    view.apply(event);
                    cx.notify();
                });
                if alive.is_err() {
                    break;
                }
            }
        })
        .detach();
    }

    fn stop_backend(&mut self) {
        if let Some(mut child) = self._backend.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn set_source(&mut self, mode: SourceMode, cx: &mut Context<Self>) {
        if self.source_mode == mode {
            return;
        }
        self.source_mode = mode;
        save_source_mode(mode);
        self.restart_backend(format!("입력을 '{}'(으)로 바꾸는 중… 모델 다시 불러오는 중", mode.label()), cx);
    }

    fn set_model(&mut self, id: String, cx: &mut Context<Self>) {
        if self.model_choice == id {
            return;
        }
        let label = updates::CHOICES
            .iter()
            .find(|c| c.0 == id)
            .map(|c| c.1)
            .unwrap_or("기본 모델");
        self.model_choice = id;
        write_setting("model", &self.model_choice);
        let msg = if self.chosen_ready() {
            format!("모델을 '{label}'(으)로 바꾸는 중… 불러오는 중")
        } else {
            format!("'{label}' 을(를) 쓰려면 먼저 내려받아야 해요. 받는 중…")
        };
        self.restart_backend(msg, cx);
    }

    /// 바뀐 설정으로 백엔드를 다시 띄운다 (모델을 다시 불러오므로 몇 초 걸림)
    fn restart_backend(&mut self, status: String, cx: &mut Context<Self>) {
        self.backend_gen += 1; // 이전 백엔드의 '종료됨' 알림이 오류 카드로 뜨지 않게 먼저 끊는다
        self.stop_backend();
        self.partial = "".into();
        self.partial_sys = "".into();
        self.partial_mic = "".into();
        self.audio_level = 0.0;
        self.push_status(status);
        self.start_backend(cx);
        cx.notify();
    }

    // ----------------------------------------------------------- 모델 관리

    fn state_of(&self, id: &str) -> ModelState {
        self.model_states
            .get(id)
            .cloned()
            .unwrap_or(ModelState::Missing { partial_bytes: 0 })
    }

    fn is_installed(&self, id: &str) -> bool {
        matches!(self.model_states.get(id), Some(ModelState::Installed { .. }))
    }

    /// 지금 고른 항목(기본 / Rung Ω / Rung Ω + 문장 맞추기 / 가벼운판)에 필요한 모델이 다 있나
    fn chosen_ready(&self) -> bool {
        updates::required_models(&self.model_choice).iter().all(|id| self.is_installed(id))
    }

    /// 고른 항목에 필요한데 아직 없는 첫 모델
    fn next_missing(&self) -> Option<&'static str> {
        updates::required_models(&self.model_choice)
            .iter()
            .copied()
            .find(|id| !self.is_installed(id))
    }

    fn choice_label(&self) -> &'static str {
        updates::CHOICES
            .iter()
            .find(|c| c.0 == self.model_choice)
            .map(|c| c.1)
            .unwrap_or("기본 모델")
    }

    fn rescan(&mut self, id: &'static str) {
        if let Some(spec) = updates::spec(id) {
            self.model_states.insert(id, updates::scan_model(spec));
        }
    }

    fn kill_downloader(&mut self) {
        if let Some(mut child) = self._downloader.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn start_model_download(&mut self, id: &'static str, cx: &mut Context<Self>) {
        if self.downloading.is_some() {
            return; // 한 번에 하나씩 (끝나면 다음 것을 이어서 받음)
        }
        let Some(spec) = updates::spec(id) else { return };
        let partial = updates::dir_size(&updates::model_path(spec));
        self.dl_gen += 1;
        let gen = self.dl_gen;
        let (tx, mut rx) = mpsc::unbounded();
        match spawn_model_download(tx, id) {
            Ok(child) => {
                self._downloader = Some(child);
                self.downloading = Some(id);
                self.model_states.insert(id, ModelState::Downloading { got: partial, total: None });
                self.push_status(format!("'{}' 내려받는 중…", spec.label));
            }
            Err(e) => {
                self.model_states.insert(
                    id,
                    ModelState::Failed { msg: format!("내려받기를 시작하지 못했어요: {e}"), partial_bytes: partial },
                );
                cx.notify();
                return;
            }
        }
        cx.spawn(async move |this, cx| {
            while let Some(event) = rx.next().await {
                let alive = this.update(cx, |view, cx| {
                    if view.dl_gen != gen {
                        return; // 취소했거나 이미 끝난 내려받기
                    }
                    view.apply_download(id, event, cx);
                    cx.notify();
                });
                if alive.is_err() {
                    break;
                }
            }
        })
        .detach();
        cx.notify();
    }

    fn apply_download(&mut self, id: &'static str, event: Event, cx: &mut Context<Self>) {
        let partial = || updates::spec(id).map(|s| updates::dir_size(&updates::model_path(s))).unwrap_or(0);
        match event {
            Event::Msg(BackendMsg::Download { got, total }) => {
                self.model_states.insert(id, ModelState::Downloading { got, total });
            }
            Event::Msg(BackendMsg::DownloadDone { .. }) => {
                self.dl_gen += 1; // 뒤따르는 '종료' 알림은 무시
                self.kill_downloader();
                self.downloading = None;
                self.rescan(id);
                if !self.is_installed(id) {
                    self.model_states.insert(
                        id,
                        ModelState::Failed {
                            msg: "받은 모델 파일이 완전하지 않아요. 다시 시도해 주세요.".into(),
                            partial_bytes: partial(),
                        },
                    );
                } else if self.chosen_ready() {
                    self.push_status(format!("'{}' 준비 완료 · 듣기를 시작합니다", self.choice_label()));
                    if self._backend.is_none() {
                        self.start_backend(cx);
                    }
                } else if let Some(next) = self.next_missing() {
                    // 고른 항목에 필요한 모델이 더 있으면 이어서 받는다 (예: Rung Ω 다음 문장 맞추기)
                    self.start_model_download(next, cx);
                }
            }
            Event::Msg(BackendMsg::Error { msg }) => {
                self.dl_gen += 1;
                self.kill_downloader();
                self.downloading = None;
                self.model_states.insert(id, ModelState::Failed { msg, partial_bytes: partial() });
            }
            Event::Exited => {
                if self.downloading == Some(id) {
                    self._downloader = None;
                    self.downloading = None;
                    self.model_states.insert(
                        id,
                        ModelState::Failed {
                            msg: "내려받기가 중간에 멈췄어요. 다시 시도하면 받은 부분부터 이어서 받습니다.".into(),
                            partial_bytes: partial(),
                        },
                    );
                }
            }
            _ => {}
        }
    }

    fn cancel_model_download(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.downloading.take() else { return };
        self.dl_gen += 1;
        self.kill_downloader();
        self.rescan(id);
        self.push_status("내려받기를 멈췄어요. 다시 받으면 이어서 받습니다.");
        cx.notify();
    }

    fn delete_model(&mut self, id: &'static str, cx: &mut Context<Self>) {
        let Some(spec) = updates::spec(id) else { return };
        if matches!(self.model_states.get(id), Some(ModelState::Installed { legacy: true, .. })) {
            return; // 개발용 예전 위치는 지우지 않는다
        }
        if self.downloading == Some(id) {
            self.dl_gen += 1;
            self.kill_downloader();
            self.downloading = None;
        }
        if updates::required_models(&self.model_choice).contains(&id) {
            self.backend_gen += 1;
            self.stop_backend();
        }
        match updates::delete_model(spec) {
            Ok(()) => self.push_status(format!("'{}' 을(를) 지웠어요.", spec.label)),
            Err(e) => self.push_line(format!("모델을 지우지 못했어요: {e}"), true),
        }
        self.rescan(id);
        cx.notify();
    }

    fn open_models(&mut self, cx: &mut Context<Self>) {
        for m in updates::MODELS.iter() {
            if self.downloading != Some(m.id) {
                self.model_states.insert(m.id, updates::scan_model(m));
            }
        }
        self.view_mode = ViewMode::Models;
        cx.notify();
    }

    // ----------------------------------------------------------- 자동 업데이트

    /// 켤 때와 6시간마다 GitHub Releases 에서 새 버전을 확인한다 (.app 으로 실행할 때만)
    fn start_update_checks(&mut self, cx: &mut Context<Self>) {
        if updates::app_bundle_path().is_none() || env::var_os("TALKIT_NO_UPDATE").is_some() {
            return;
        }
        let Some(current) = updates::current_version() else { return };
        let (tx, mut rx) = mpsc::unbounded();
        thread::spawn(move || loop {
            match updates::fetch_latest() {
                Ok(rel) if updates::is_newer(&rel.version, &current) => {
                    if tx.unbounded_send(UpdEvent::Found(rel)).is_err() {
                        return;
                    }
                }
                Ok(_) => {}
                Err(e) => eprintln!("[Talkit] 업데이트 확인 실패: {e}"),
            }
            thread::sleep(Duration::from_secs(6 * 3600));
            if tx.is_closed() {
                return;
            }
        });
        cx.spawn(async move |this, cx| {
            while let Some(ev) = rx.next().await {
                let alive = this.update(cx, |view, cx| {
                    view.apply_update(ev);
                    cx.notify();
                });
                if alive.is_err() {
                    break;
                }
            }
        })
        .detach();
    }

    fn apply_update(&mut self, ev: UpdEvent) {
        let current_rel = match &self.update {
            UpdateState::Available(r) | UpdateState::Downloading { rel: r, .. } | UpdateState::Ready { rel: r, .. } => {
                Some(r.clone())
            }
            UpdateState::Failed { rel, .. } => rel.clone(),
            UpdateState::None => None,
        };
        match ev {
            UpdEvent::Found(rel) => {
                // 받는 중이거나 설치 준비가 끝났으면 건드리지 않는다
                if matches!(self.update, UpdateState::None | UpdateState::Available(_) | UpdateState::Failed { .. }) {
                    self.update = UpdateState::Available(rel);
                }
            }
            UpdEvent::Progress(got) => {
                if let UpdateState::Downloading { rel, .. } = &self.update {
                    self.update = UpdateState::Downloading { rel: rel.clone(), got };
                }
            }
            UpdEvent::Ready(staged) => {
                if let Some(rel) = current_rel {
                    self.update = UpdateState::Ready { rel, staged };
                }
            }
            UpdEvent::Failed(msg) => self.update = UpdateState::Failed { rel: current_rel, msg },
        }
    }

    fn start_update_download(&mut self, rel: Release, cx: &mut Context<Self>) {
        let Some(bundle) = updates::app_bundle_path() else { return };
        if let Some(msg) = updates::install_blocker(&bundle) {
            self.update = UpdateState::Failed { rel: Some(rel), msg };
            cx.notify();
            return;
        }
        self.upd_gen += 1;
        let gen = self.upd_gen;
        self.update = UpdateState::Downloading { rel: rel.clone(), got: 0 };
        let (tx, mut rx) = mpsc::unbounded();
        thread::spawn(move || {
            let ptx = tx.clone();
            let res = updates::download_dmg(&rel, |got| {
                let _ = ptx.unbounded_send(UpdEvent::Progress(got));
            })
            .and_then(|dmg| updates::prepare_update(&rel, &dmg, &bundle));
            let _ = tx.unbounded_send(match res {
                Ok(staged) => UpdEvent::Ready(staged),
                Err(e) => UpdEvent::Failed(e),
            });
        });
        cx.spawn(async move |this, cx| {
            while let Some(ev) = rx.next().await {
                let alive = this.update(cx, |view, cx| {
                    if view.upd_gen != gen {
                        return;
                    }
                    view.apply_update(ev);
                    cx.notify();
                });
                if alive.is_err() {
                    break;
                }
            }
        })
        .detach();
        cx.notify();
    }

    /// 새 앱으로 바꾸고 다시 켠다 (앱은 여기서 꺼짐)
    fn install_update(&mut self, cx: &mut Context<Self>) {
        let UpdateState::Ready { staged, .. } = &self.update else { return };
        let Some(bundle) = updates::app_bundle_path() else { return };
        match updates::launch_installer(&bundle, staged) {
            Ok(()) => {
                self.backend_gen += 1;
                self.stop_backend();
                self.kill_downloader();
                cx.quit();
            }
            Err(msg) => {
                let rel = match &self.update {
                    UpdateState::Ready { rel, .. } => Some(rel.clone()),
                    _ => None,
                };
                self.update = UpdateState::Failed { rel, msg };
                cx.notify();
            }
        }
    }

    /// 전사 따라가기가 켜져 있을 때만 맨 아래로 내린다
    fn autoscroll(&self) {
        if self.follow_tail {
            self.scroll.scroll_to_bottom();
        }
    }

    fn toggle_follow(&mut self, cx: &mut Context<Self>) {
        self.follow_tail = !self.follow_tail;
        if self.follow_tail {
            self.scroll.scroll_to_bottom();
        }
        cx.notify();
    }

    fn open_history(&mut self, cx: &mut Context<Self>) {
        let extra = self.save_path.as_ref().and_then(|p| p.parent().map(|d| d.to_path_buf()));
        self.history = scan_history(extra);
        self.history_detail = None;
        self.history_copied = None;
        self.view_mode = ViewMode::HistoryList;
        self.history_scroll.set_offset(point(px(0.0), px(0.0)));
        cx.notify();
    }

    fn open_history_entry(&mut self, idx: usize, cx: &mut Context<Self>) {
        let Some(entry) = self.history.get(idx).cloned() else { return };
        let content = fs::read_to_string(&entry.path).unwrap_or_default();
        let lines: Vec<HistoryLine> = content.lines().filter_map(parse_history_line).collect();
        self.history_detail = Some((entry, lines));
        self.history_copied = None;
        self.view_mode = ViewMode::HistoryDetail;
        self.history_scroll.set_offset(point(px(0.0), px(0.0)));
        cx.notify();
    }

    fn close_history(&mut self, cx: &mut Context<Self>) {
        self.view_mode = ViewMode::Live;
        self.history_detail = None;
        if self.follow_tail {
            self.scroll.scroll_to_bottom();
        }
        cx.notify();
    }

    fn copy_history(&mut self, idx: usize, text: String, cx: &mut Context<Self>) {
        if text.trim().is_empty() {
            return;
        }
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        self.history_copied = Some(idx);
        cx.notify();
    }

    fn push_status(&mut self, msg: impl Into<SharedString>) {
        if self.status.len() >= MAX_STATUS_ENTRIES {
            self.status.pop_front();
        }
        self.status.push_back(msg.into());
    }

    fn push_wave(&mut self, rms: f32) {
        self.audio_level = rms.clamp(0.0, 1.0);
        if self.wave_history.len() >= 12 {
            self.wave_history.pop_front();
        }
        self.wave_history.push_back(self.audio_level.max(0.06));
    }

    fn has_any_partial(&self) -> bool {
        !self.partial_sys.trim().is_empty()
            || !self.partial_mic.trim().is_empty()
            || !self.partial.trim().is_empty()
    }

    fn partial_for_source(&self, source: &Option<SharedString>) -> SharedString {
        match source.as_ref().map(|s| s.as_ref()) {
            Some("시스템") => self.partial_sys.clone(),
            Some("마이크") => self.partial_mic.clone(),
            _ => self.partial.clone(),
        }
    }

    fn push_line(&mut self, raw: impl Into<String>, is_error: bool) {
        if self.paused && !is_error {
            return;
        }
        let s: String = raw.into();
        let (timestamp, after_ts) = if s.starts_with('[') {
            if let Some(idx) = s.find("] ") {
                let tag = &s[1..idx];
                if tag == "시스템" || tag == "마이크" {
                    (None, s.as_str())
                } else {
                    (Some(SharedString::from(tag.to_string())), &s[idx + 2..])
                }
            } else {
                (None, s.as_str())
            }
        } else {
            (None, s.as_str())
        };

        let (source, rest) = parse_source_prefix(after_ts);
        let clean_body = rest.trim();
        if clean_body.is_empty() {
            return;
        }

        if is_error {
            self.total_committed += 1;
            if self.items.len() >= MAX_TRANSCRIPT_ITEMS {
                self.items.pop_front();
            }
            self.items.push_back(TranscriptItem {
                index: self.total_committed,
                timestamp,
                source,
                body: SharedString::from(clean_body.to_string()),
                sentence_complete: true,
                is_error: true,
            });
            self.autoscroll();
            return;
        }

        // 연속 전사된 텍스트를 문장 단위로 끊어서 처리하되,
        // 직전 동일 채널(시스템/마이크) 카드가 아직 미완성 문장이면 자연스럽게 이어붙인 뒤 문장 완결 시 다음 카드로 분리한다.
        let segments = split_into_sentences(clean_body);
        for (seg_text, is_complete) in segments {
            if seg_text.is_empty() {
                continue;
            }

            let mut merged_into_open = false;
            if let Some(open_item) = self
                .items
                .iter_mut()
                .rev()
                .find(|it| !it.is_error && it.source == source)
            {
                if !open_item.sentence_complete {
                    let combined = if open_item.body.is_empty() {
                        seg_text.clone()
                    } else {
                        format!("{} {}", open_item.body.trim_end(), seg_text)
                    };
                    let total_words = combined.split_whitespace().count();
                    let last_w = combined.split_whitespace().last().unwrap_or("");
                    let now_complete =
                        is_complete || is_korean_sentence_end_word(last_w, total_words) || total_words >= 22;
                    open_item.body = SharedString::from(combined);
                    open_item.sentence_complete = now_complete;
                    if open_item.timestamp.is_none() {
                        open_item.timestamp = timestamp.clone();
                    }
                    merged_into_open = true;
                }
            }

            if !merged_into_open {
                self.total_committed += 1;
                if self.items.len() >= MAX_TRANSCRIPT_ITEMS {
                    self.items.pop_front();
                }
                let word_cnt = seg_text.split_whitespace().count();
                self.items.push_back(TranscriptItem {
                    index: self.total_committed,
                    timestamp: timestamp.clone(),
                    source: source.clone(),
                    body: SharedString::from(seg_text),
                    sentence_complete: is_complete || word_cnt >= 22,
                    is_error: false,
                });
            }
        }
        self.autoscroll();
    }

    fn full_continuous_text(&self) -> String {
        let mut parts: Vec<String> = self
            .items
            .iter()
            .filter(|it| !it.is_error && !it.body.trim().is_empty())
            .map(|it| {
                if let Some(ref src) = it.source {
                    format!("[{}] {}", src, it.body.trim())
                } else {
                    it.body.trim().to_string()
                }
            })
            .collect();
        if !self.partial_sys.trim().is_empty() {
            parts.push(format!("[시스템] {}", self.partial_sys.trim()));
        }
        if !self.partial_mic.trim().is_empty() {
            parts.push(format!("[마이크] {}", self.partial_mic.trim()));
        }
        if !self.partial.trim().is_empty() {
            parts.push(self.partial.trim().to_string());
        }
        parts.join("\n")
    }

    fn continuous_tail(&self, max_chars: usize) -> Option<SharedString> {
        let mut stream_parts: Vec<String> = self
            .items
            .iter()
            .filter(|it| !it.is_error && !it.body.trim().is_empty())
            .map(|it| it.body.trim().to_string())
            .collect();
        if !self.partial_sys.trim().is_empty() {
            stream_parts.push(self.partial_sys.trim().to_string());
        }
        if !self.partial_mic.trim().is_empty() {
            stream_parts.push(self.partial_mic.trim().to_string());
        }
        if !self.partial.trim().is_empty() {
            stream_parts.push(self.partial.trim().to_string());
        }
        let full = stream_parts.join(" ");
        if full.is_empty() {
            return None;
        }
        let total_chars = full.chars().count();
        if total_chars <= max_chars {
            return Some(SharedString::from(full));
        }
        let skip = total_chars.saturating_sub(max_chars);
        let tail: String = full.chars().skip(skip).collect();
        let clean_tail = if let Some((pos, _)) = tail.char_indices().take(12).find(|(_, c)| c.is_whitespace()) {
            tail[pos..].trim_start()
        } else {
            tail.as_str()
        };
        Some(SharedString::from(format!("… {}", clean_tail)))
    }

    fn apply(&mut self, event: Event) {
        match event {
            Event::Msg(BackendMsg::Status { msg }) => {
                // 지금 세션 파일 경로를 기억해 둔다 (입력 전환 시 같은 파일에 이어 쓰기, 기록 목록 위치)
                if let Some(p) = msg.strip_prefix("저장 파일: ") {
                    self.save_path = Some(PathBuf::from(p.trim()));
                }
                self.push_status(msg)
            }
            Event::Msg(BackendMsg::Partial { text }) => {
                if !self.paused {
                    let (src, rest) = parse_source_prefix(&text);
                    match src.as_ref().map(|s| s.as_ref()) {
                        Some("시스템") => self.partial_sys = SharedString::from(rest.to_string()),
                        Some("마이크") => self.partial_mic = SharedString::from(rest.to_string()),
                        _ => self.partial = SharedString::from(rest.to_string()),
                    }
                    self.autoscroll();
                }
            }
            Event::Msg(BackendMsg::Text {
                text,
                clock,
                ..
            }) => {
                let (_, after_clock) = if text.starts_with('[') {
                    if let Some(idx) = text.find("] ") {
                        let tag = &text[1..idx];
                        if tag == "시스템" || tag == "마이크" {
                            (None, text.as_str())
                        } else {
                            (Some(tag), &text[idx + 2..])
                        }
                    } else {
                        (None, text.as_str())
                    }
                } else {
                    (None, text.as_str())
                };
                let (src, _) = parse_source_prefix(after_clock);
                match src.as_ref().map(|s| s.as_ref()) {
                    Some("시스템") => self.partial_sys = "".into(),
                    Some("마이크") => self.partial_mic = "".into(),
                    _ => self.partial = "".into(),
                }
                let formatted = if let Some(c) = clock {
                    if text.starts_with('[') && !text.starts_with("[시스템]") && !text.starts_with("[마이크]") {
                        text
                    } else {
                        format!("[{c}] {text}")
                    }
                } else {
                    text
                };
                self.push_line(formatted, false);
            }
            Event::Msg(BackendMsg::Error { msg }) => self.push_line(msg, true),
            Event::Msg(BackendMsg::Level { rms }) => {
                if !self.paused {
                    self.push_wave(rms);
                }
            }
            Event::Msg(BackendMsg::Models { items, current }) => {
                self.models = items;
                self.model_choice = current; // 실제로 올라간 모델로 맞춘다 (없는 모델을 골랐던 경우 대비)
            }
            Event::Msg(BackendMsg::Telemetry { rms, mem_mb }) => {
                if !self.paused {
                    self.push_wave(rms);
                }
                self.mem_mb = mem_mb;
            }
            // 모델 내려받기 소식은 내려받기 전용 프로세스에서만 오고 apply_download 가 처리한다
            Event::Msg(BackendMsg::Download { .. }) | Event::Msg(BackendMsg::DownloadDone { .. }) => {}
            Event::Exited => self.push_line("백엔드가 종료됐습니다 (터미널 로그 확인)", true),
        }
    }

    fn select_preset(&mut self, preset: GlassPreset, window: &mut Window, cx: &mut Context<Self>) {
        self.preset = preset;
        #[cfg(target_os = "macos")]
        macos_glass::apply_window_glass(window, preset, self.pip_mode, self.pip_locked);
        cx.notify();
    }

    fn toggle_pip(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.pip_mode = !self.pip_mode;
        #[cfg(target_os = "macos")]
        {
            let pip_mode = self.pip_mode;
            let pip_locked = self.pip_locked;
            if let Some((win_ptr, view_ptr)) = macos_glass::resolve_native_ptrs(window) {
                cx.foreground_executor()
                    .spawn(async move {
                        macos_glass::perform_pip_transition_unlocked(
                            win_ptr,
                            view_ptr,
                            pip_mode,
                            pip_locked,
                        );
                    })
                    .detach();
            }
        }
        cx.notify();
    }

    fn toggle_pip_lock(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.pip_locked = !self.pip_locked;
        #[cfg(target_os = "macos")]
        macos_glass::apply_window_glass(window, self.preset, self.pip_mode, self.pip_locked);
        cx.notify();
    }

    fn toggle_pause(&mut self, cx: &mut Context<Self>) {
        self.paused = !self.paused;
        if self.paused {
            self.partial = "".into();
            self.partial_sys = "".into();
            self.partial_mic = "".into();
            self.audio_level = 0.0;
        }
        cx.notify();
    }

    fn clear_history(&mut self, cx: &mut Context<Self>) {
        self.items.clear();
        self.partial = "".into();
        self.partial_sys = "".into();
        self.partial_mic = "".into();
        self.copied_index = None;
        self.drag_anchor = None;
        self.drag_current = None;
        self.is_dragging_selection = false;
        cx.notify();
    }

    fn copy_text(&mut self, idx: usize, text: String, cx: &mut Context<Self>) {
        if text.trim().is_empty() {
            return;
        }
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        self.copied_index = Some(idx);
        cx.notify();
    }

    fn copy_all(&mut self, cx: &mut Context<Self>) {
        let full = self.full_continuous_text();
        if full.is_empty() {
            return;
        }
        cx.write_to_clipboard(ClipboardItem::new_string(full));
        self.copied_index = Some(0);
        cx.notify();
    }

    // -----------------------------------------------------------------------
    // Sentence-Card Range Drag-to-Select & Copy
    // -----------------------------------------------------------------------

    fn selected_range(&self) -> Option<(usize, usize)> {
        let a = self.drag_anchor?;
        let b = self.drag_current.unwrap_or(a);
        Some((a.min(b), a.max(b)))
    }

    fn is_in_drag_selection(&self, item_idx: usize) -> bool {
        if let Some((lo, hi)) = self.selected_range() {
            item_idx >= lo && item_idx <= hi
        } else {
            false
        }
    }

    fn start_card_drag(&mut self, item_idx: usize, cx: &mut Context<Self>) {
        self.drag_anchor = Some(item_idx);
        self.drag_current = Some(item_idx);
        self.is_dragging_selection = true;
        self.copy_selected_range(cx);
    }

    fn update_card_drag(&mut self, item_idx: usize, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        if self.is_dragging_selection && event.dragging() {
            if self.drag_current != Some(item_idx) {
                self.drag_current = Some(item_idx);
            }
            self.copy_selected_range(cx);
        }
    }

    fn finish_card_drag(&mut self, cx: &mut Context<Self>) {
        if !self.is_dragging_selection {
            return;
        }
        self.is_dragging_selection = false;
        self.copy_selected_range(cx);
    }

    fn copy_selected_range(&mut self, cx: &mut Context<Self>) {
        let Some((lo, hi)) = self.selected_range() else {
            return;
        };
        let lines: Vec<String> = self
            .items
            .iter()
            .filter(|it| !it.is_error && it.index >= lo && it.index <= hi)
            .map(|it| it.body.to_string())
            .collect();
        if lines.is_empty() {
            return;
        }
        let joined = lines.join("\n");
        cx.write_to_clipboard(ClipboardItem::new_string(joined));
        if lo == hi {
            self.copied_index = Some(lo);
        } else {
            self.copied_index = Some(usize::MAX - 2);
        }
        cx.notify();
    }

    fn format_elapsed(&self) -> SharedString {
        let secs = self.started_at.elapsed().as_secs();
        let mins = secs / 60;
        let rem = secs % 60;
        SharedString::from(format!("{:02}:{:02}", mins, rem))
    }

    fn render_waveform_bars(&self, bar_max_h: f32, max_bars: usize) -> impl IntoElement {
        let theme = self.preset.theme();
        let is_active = self.has_any_partial() || self.audio_level > 0.08;
        let skip = self.wave_history.len().saturating_sub(max_bars);
        div()
            .flex_none()
            .flex()
            .items_center()
            .gap(px(2.5))
            .h(px(bar_max_h))
            .children(self.wave_history.iter().skip(skip).enumerate().map(|(i, &val)| {
                let mid = (max_bars.saturating_sub(1)) as f32 / 2.0;
                let span = (max_bars as f32 / 1.4).max(1.0);
                // Diamond silhouette inspired by Talkit's 5-bar logo
                let center_weight = (1.0 - ((i as f32 - mid).abs() / span) * 0.55).max(0.35);
                let norm = if is_active {
                    (val * center_weight).clamp(0.22, 1.0)
                } else {
                    (0.32 + center_weight * 0.42).clamp(0.24, 0.78)
                };
                let h = (bar_max_h * norm).max(4.0);
                let color = if self.paused {
                    0xf59e0bcc
                } else if is_active {
                    TALKIT_BLUE
                } else {
                    theme.wave_idle
                };
                div()
                    .w(px(3.0))
                    .h(px(h))
                    .rounded(px(999.0))
                    .bg(rgba(color))
            }))
    }

    fn render_preset_pill(
        &self,
        preset: GlassPreset,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = self.preset.theme();
        let selected = self.preset == preset;
        div()
            .id(SharedString::from(format!("glass-preset-{:?}", preset)))
            .flex_none()
            .h(px(24.0))
            .px_2p5()
            .flex()
            .items_center()
            .justify_center()
            .rounded(px(999.0))
            .cursor_pointer()
            .text_size(px(11.0))
            .text_color(rgba(if selected {
                0xffffffff
            } else {
                theme.text_secondary
            }))
            .border_1()
            .border_color(rgba(if selected {
                TALKIT_BLUE
            } else {
                theme.pill_border
            }))
            .when(selected, |d| {
                d.bg(rgba(TALKIT_BLUE)).shadow(glass_pill_shadows())
            })
            .when(!selected, |d| d.bg(rgba(theme.pill_bg)))
            .hover(|d| d.bg(rgba(if selected { 0x3b57f5ff } else { theme.pill_hover_bg })))
            .active(|d| d.opacity(0.8))
            .child(preset.label())
            .on_click(cx.listener(move |this, _, window, cx| {
                this.select_preset(preset, window, cx);
            }))
    }

    // -----------------------------------------------------------------------
    // 1. Compact Picture-in-Picture (PiP) Floating Overlay HUD (Talkit Brand Guide)
    // -----------------------------------------------------------------------
    fn render_pip_overlay(&self, vw: f32, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = self.preset.theme();
        let is_xs = vw < 430.0;
        let has_partial = self.has_any_partial();
        let elapsed = self.format_elapsed();
        let mem_badge = SharedString::from(format!("{:.2} GB", self.mem_mb / 1024.0));

        let tail_caption = self.continuous_tail(85);
        let full_text = self.full_continuous_text();
        let copyable_pip: Option<String> = if !full_text.is_empty() {
            Some(full_text)
        } else {
            None
        };
        let pip_copied = self.copied_index == Some(usize::MAX);

        let primary_caption: SharedString = if self.paused {
            "일시정지됨 — 재개 버튼을 누르면 다시 실시간 전사를 시작합니다".into()
        } else if let Some(ref tail) = tail_caption {
            tail.clone()
        } else {
            "Talk it, Get it. 말하는 순간, 바로 텍스트로.".into()
        };

        let secondary_caption: Option<SharedString> = None;

        div()
            .size_full()
            .p_2()
            .flex()
            .flex_col()
            // Outer Concentric Talkit PiP Capsule (matches bottom-right floating pill in Brand Guide)
            .child(
                div()
                    .size_full()
                    .px(px(if is_xs { 12.0 } else { 16.0 }))
                    .py_2p5()
                    .rounded(px(26.0))
                    .bg(linear_gradient(
                        180.0,
                        linear_color_stop(rgba(theme.shell_bg_top), 0.0),
                        linear_color_stop(rgba(theme.shell_bg_bottom), 1.0),
                    ))
                    .border_1()
                    .border_color(rgba(if has_partial {
                        theme.shell_active_border
                    } else {
                        theme.shell_border
                    }))
                    .shadow(glass_card_shadows())
                    .flex()
                    .flex_col()
                    .justify_between()
                    .gap_1p5()
                    // Top Row: Talkit 5-Bar Logo + Brand Wordmark + Waveform + Timer + Lock + Quick Controls
                    .child(
                        div()
                            .flex_none()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_between()
                            .gap_1p5()
                            .child(
                                div()
                                    .id("pip-drag-handle")
                                    .flex_1()
                                    .min_w(px(140.0))
                                    .when(!self.pip_locked, |d| d.cursor_pointer())
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _: &MouseDownEvent, window, _| {
                                            #[cfg(target_os = "macos")]
                                            macos_glass::start_window_drag(window, !this.pip_locked);
                                        }),
                                    )
                                    // Talkit 5-Bar Soundwave Logo + "Talkit" Wordmark
                                    .child(
                                        div()
                                            .flex_none()
                                            .flex()
                                            .items_center()
                                            .gap_1p5()
                                            .child(icon_svg(
                                                "icons/talkit_logo.svg",
                                                18.0,
                                                TALKIT_BLUE,
                                            ))
                                            .child(
                                                div()
                                                    .text_size(px(13.5))
                                                    .text_color(rgba(theme.text_primary))
                                                    .child("Talkit"),
                                            ),
                                    )
                                    .child(self.render_waveform_bars(
                                        16.0,
                                        if is_xs { 5 } else { 9 },
                                    ))
                                    .child(
                                        div()
                                            .flex_none()
                                            .px_2()
                                            .py(px(1.5))
                                            .rounded(px(999.0))
                                            .bg(rgba(theme.pill_bg))
                                            .border_1()
                                            .border_color(rgba(theme.pill_border))
                                            .text_size(px(10.5))
                                            .text_color(rgba(theme.text_secondary))
                                            .child(elapsed),
                                    )
                                    .when(!is_xs, |d| {
                                        d.child(
                                            div()
                                                .flex_none()
                                                .px_2()
                                                .py(px(1.5))
                                                .rounded(px(999.0))
                                                .bg(rgba(0x4f6bff1a))
                                                .border_1()
                                                .border_color(rgba(0x4f6bff44))
                                                .text_size(px(10.0))
                                                .text_color(rgba(TALKIT_BLUE))
                                                .child(mem_badge),
                                        )
                                    }),
                            )
                            // Right Controls: Lock (Padlock SVG) + Copy + Pause/Resume + Expand back to Studio
                            .child(
                                div()
                                    .flex_none()
                                    .flex()
                                    .items_center()
                                    .gap_1()
                                    .child(
                                        div()
                                            .id("pip-lock-btn")
                                            .flex_none()
                                            .h(px(24.0))
                                            .px_2()
                                            .rounded(px(999.0))
                                            .cursor_pointer()
                                            .bg(rgba(if self.pip_locked {
                                                TALKIT_BLUE
                                            } else {
                                                theme.pill_bg
                                            }))
                                            .border_1()
                                            .border_color(rgba(if self.pip_locked {
                                                TALKIT_BLUE
                                            } else {
                                                theme.pill_border
                                            }))
                                            .hover(|d| {
                                                d.bg(rgba(if self.pip_locked {
                                                    0x3b57f5ff
                                                } else {
                                                    theme.pill_hover_bg
                                                }))
                                            })
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .text_size(px(10.5))
                                            .text_color(rgba(if self.pip_locked {
                                                0xffffffff
                                            } else {
                                                theme.text_primary
                                            }))
                                            .child(if self.pip_locked {
                                                icon_svg("icons/lock.svg", 11.5, 0xffffffff)
                                            } else {
                                                icon_svg(
                                                    "icons/unlock.svg",
                                                    11.5,
                                                    theme.text_primary,
                                                )
                                            })
                                            .when(!is_xs, |d| {
                                                d.child(if self.pip_locked { "잠금됨" } else { "잠금" })
                                            })
                                            .on_click(cx.listener(|this, _, window, cx| {
                                                this.toggle_pip_lock(window, cx);
                                            })),
                                    )
                                    .when_some(copyable_pip.clone(), |d, txt| {
                                        d.child(
                                            div()
                                                .id("pip-copy-btn")
                                                .flex_none()
                                                .h(px(24.0))
                                                .px_2()
                                                .rounded(px(999.0))
                                                .cursor_pointer()
                                                .bg(rgba(if pip_copied {
                                                    TALKIT_BLUE
                                                } else {
                                                    theme.pill_bg
                                                }))
                                                .border_1()
                                                .border_color(rgba(if pip_copied {
                                                    TALKIT_BLUE
                                                } else {
                                                    theme.pill_border
                                                }))
                                                .hover(|d| d.bg(rgba(theme.pill_hover_bg)))
                                                .flex()
                                                .items_center()
                                                .gap_1()
                                                .text_size(px(10.5))
                                                .text_color(rgba(if pip_copied {
                                                    0xffffffff
                                                } else {
                                                    theme.text_primary
                                                }))
                                                .child(if pip_copied {
                                                    icon_svg("icons/check.svg", 11.5, 0xffffffff)
                                                } else {
                                                    icon_svg(
                                                        "icons/copy.svg",
                                                        11.5,
                                                        theme.text_primary,
                                                    )
                                                })
                                                .when(!is_xs, |d| {
                                                    d.child(if pip_copied { "복사됨" } else { "복사" })
                                                })
                                                .on_click(cx.listener(move |this, _, _, cx| {
                                                    this.copy_text(usize::MAX, txt.clone(), cx);
                                                })),
                                        )
                                    })
                                    .child(
                                        div()
                                            .id("pip-pause-btn")
                                            .flex_none()
                                            .h(px(24.0))
                                            .px_2()
                                            .rounded(px(999.0))
                                            .cursor_pointer()
                                            .bg(rgba(theme.pill_bg))
                                            .border_1()
                                            .border_color(rgba(theme.pill_border))
                                            .hover(|d| d.bg(rgba(theme.pill_hover_bg)))
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .text_size(px(10.5))
                                            .text_color(rgba(theme.text_primary))
                                            .child(if self.paused {
                                                icon_svg("icons/play.svg", 11.0, TALKIT_BLUE)
                                            } else {
                                                icon_svg("icons/pause.svg", 11.0, theme.text_primary)
                                            })
                                            .when(!is_xs, |d| {
                                                d.child(if self.paused { "재개" } else { "정지" })
                                            })
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.toggle_pause(cx);
                                            })),
                                    )
                                    .child(
                                        div()
                                            .id("pip-expand-btn")
                                            .flex_none()
                                            .h(px(24.0))
                                            .px_2p5()
                                            .rounded(px(999.0))
                                            .cursor_pointer()
                                            .bg(rgba(TALKIT_BLUE))
                                            .border_1()
                                            .border_color(rgba(TALKIT_BLUE))
                                            .hover(|d| d.bg(rgba(0x3b57f5ff)))
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .text_size(px(10.5))
                                            .text_color(rgba(0xffffffff))
                                            .child(icon_svg("icons/expand.svg", 11.5, 0xffffffff))
                                            .child("큰창")
                                            .on_click(cx.listener(|this, _, window, cx| {
                                                this.toggle_pip(window, cx);
                                            })),
                                    ),
                            ),
                    )
                    // Bottom Live Subtitle Area (drag or click to copy directly without moving the window)
                    .child(
                        div()
                            .id("pip-caption-area")
                            .cursor_pointer()
                            .flex_1()
                            .min_h(px(0.0))
                            .px_2p5()
                            .py_1()
                            .rounded(px(14.0))
                            .bg(rgba(if pip_copied {
                                0x4f6bff22
                            } else if theme.is_light {
                                0xffffffff
                            } else {
                                0xffffff0e
                            }))
                            .border_1()
                            .border_color(rgba(if pip_copied || has_partial {
                                TALKIT_BLUE
                            } else {
                                theme.card_border
                            }))
                            .hover(|d| d.bg(rgba(theme.card_hover_bg)))
                            .flex()
                            .flex_col()
                            .justify_center()
                            .gap(px(2.0))
                            .overflow_hidden()
                            .when_some(secondary_caption, |d, prev| {
                                d.child(
                                    div()
                                        .w_full()
                                        .truncate()
                                        .text_size(px(11.5))
                                        .text_color(rgba(theme.text_muted))
                                        .child(prev),
                                )
                            })
                            .child(
                                div()
                                    .w_full()
                                    .text_size(px(if is_xs { 13.5 } else { 14.5 }))
                                    .text_color(rgba(if has_partial {
                                        TALKIT_BLUE
                                    } else {
                                        theme.text_primary
                                    }))
                                    .child(primary_caption),
                            )
                            .when_some(copyable_pip, |d, txt| {
                                let txt_down = txt.clone();
                                let txt_drag = txt.clone();
                                let txt_click = txt;
                                d.on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _: &MouseDownEvent, _, cx| {
                                        this.copy_text(usize::MAX, txt_down.clone(), cx);
                                    }),
                                )
                                .on_mouse_move(cx.listener(
                                    move |this, ev: &MouseMoveEvent, _, cx| {
                                        if ev.dragging() {
                                            this.copy_text(usize::MAX, txt_drag.clone(), cx);
                                        }
                                    },
                                ))
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.copy_text(usize::MAX, txt_click.clone(), cx);
                                }))
                            }),
                    ),
            )
    }

    // -----------------------------------------------------------------------
    // 2. Full Talkit Studio Mode (Talkit Brand Identity Guide)
    // -----------------------------------------------------------------------
    /// 둥근 알약 버튼 (primary 면 Talkit 파랑)
    fn pill_btn(&self, id: impl Into<SharedString>, primary: bool) -> Stateful<Div> {
        let theme = self.preset.theme();
        div()
            .id(id.into())
            .flex_none()
            .h(px(26.0))
            .px_3()
            .rounded(px(999.0))
            .cursor_pointer()
            .flex()
            .items_center()
            .gap_1()
            .text_size(px(11.5))
            .bg(rgba(if primary { TALKIT_BLUE } else { theme.pill_bg }))
            .border_1()
            .border_color(rgba(if primary { TALKIT_BLUE } else { theme.pill_border }))
            .text_color(rgba(if primary { 0xffffffff } else { theme.text_primary }))
            .hover(move |d| d.bg(rgba(if primary { 0x3b57f5ff } else { theme.pill_hover_bg })))
    }

    fn progress_bar(&self, frac: f32) -> Div {
        let theme = self.preset.theme();
        div()
            .w_full()
            .h(px(6.0))
            .rounded(px(999.0))
            .bg(rgba(theme.pill_bg))
            .border_1()
            .border_color(rgba(theme.pill_border))
            .child(
                div()
                    .h_full()
                    .w(relative(frac.clamp(0.02, 1.0)))
                    .rounded(px(999.0))
                    .bg(rgba(TALKIT_BLUE)),
            )
    }

    /// 모델 한 장: 상태 · 진행 막대 · 쓰기/내려받기/멈추기/다시 시도/지우기/폴더 열기
    fn render_model_card(&self, spec: &'static ModelSpec, is_compact: bool, cx: &mut Context<Self>) -> Div {
        let theme = self.preset.theme();
        let id = spec.id;
        let state = self.state_of(id);
        let dir = updates::model_path(spec);
        let in_use = updates::required_models(&self.model_choice).contains(&id) && self.chosen_ready();
        let busy_other = self.downloading.is_some() && self.downloading != Some(id);

        let (state_line, state_color): (String, u32) = match &state {
            ModelState::Installed { bytes, legacy, .. } => (
                format!("설치됨 · {}{}", fmt_mb(*bytes), if *legacy { " · 개발용 위치" } else { "" }),
                0x10b981ff,
            ),
            ModelState::Missing { partial_bytes } if *partial_bytes > 0 => (
                format!("받다 멈춤 · {} 받음 · 이어서 받을 수 있어요", fmt_mb(*partial_bytes)),
                theme.text_secondary,
            ),
            ModelState::Missing { .. } => (format!("아직 없어요 · 약 {}", spec.approx), theme.text_secondary),
            ModelState::Downloading { got, total } => match total {
                Some(t) if *t > 0 => (
                    format!("내려받는 중 · {}% ({} / {})", (*got * 100 / *t).min(99), fmt_mb(*got), fmt_mb(*t)),
                    TALKIT_BLUE,
                ),
                _ => (format!("내려받는 중 · {}", fmt_mb(*got)), TALKIT_BLUE),
            },
            ModelState::Failed { msg, .. } => (msg.clone(), 0xfb7185ff),
        };

        let mut buttons = div().flex().flex_wrap().items_center().gap_2();
        // 쓰기 버튼: 음성 인식 모델은 '이 모델 쓰기', 문장 맞추기는 Rung Ω 에 켜고 끄기
        if let ModelState::Installed { .. } = state {
            if id == "fa" {
                let on = self.model_choice == "rung-omega-fa";
                let rung_ready = self.is_installed("rung-omega");
                if rung_ready {
                    buttons = buttons.child(
                        self.pill_btn(format!("{id}-toggle"), !on)
                            .child(if on { "문장 맞추기 끄기" } else { "정밀 모델에 문장 맞추기 켜기" })
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.set_model(if on { "rung-omega".into() } else { "rung-omega-fa".into() }, cx)
                            })),
                    );
                }
            } else if !in_use || (id == "rung-omega" && self.model_choice == "rung-omega-fa") {
                if !in_use {
                    buttons = buttons.child(
                        self.pill_btn(format!("{id}-use"), true)
                            .child(icon_svg("icons/check.svg", 12.0, 0xffffffff))
                            .child("이 모델 쓰기")
                            .on_click(cx.listener(move |this, _, _, cx| this.set_model(id.to_string(), cx))),
                    );
                }
            }
        }
        match &state {
            ModelState::Installed { legacy, path, .. } => {
                let path = path.clone();
                buttons = buttons.child(
                    self.pill_btn(format!("{id}-folder"), false)
                        .child(icon_svg("icons/folder.svg", 12.0, theme.text_primary))
                        .child("폴더 열기")
                        .on_click(move |_, _, _| updates::reveal(&path)),
                );
                if !legacy {
                    buttons = buttons.child(
                        self.pill_btn(format!("{id}-delete"), false)
                            .border_color(rgba(0xfb718566))
                            .text_color(rgba(0xfb7185ff))
                            .hover(|d| d.bg(rgba(0xf43f5e24)))
                            .child(icon_svg("icons/trash.svg", 12.0, 0xfb7185ff))
                            .child("모델 제거")
                            .on_click(cx.listener(move |this, _, _, cx| this.delete_model(id, cx))),
                    );
                }
            }
            ModelState::Downloading { .. } => {
                buttons = buttons.child(
                    self.pill_btn(format!("{id}-cancel"), false)
                        .child("내려받기 멈추기")
                        .on_click(cx.listener(|this, _, _, cx| this.cancel_model_download(cx))),
                );
            }
            ModelState::Missing { partial_bytes } | ModelState::Failed { partial_bytes, .. } => {
                let label = if matches!(state, ModelState::Failed { .. }) {
                    "다시 시도".to_string()
                } else if *partial_bytes > 0 {
                    "이어서 받기".to_string()
                } else {
                    format!("내려받기 ({})", spec.approx)
                };
                if busy_other {
                    buttons = buttons.child(
                        div()
                            .text_size(px(11.0))
                            .text_color(rgba(theme.text_muted))
                            .child("다른 모델을 받는 중이에요. 끝나면 받을 수 있어요."),
                    );
                } else {
                    buttons = buttons.child(
                        self.pill_btn(format!("{id}-download"), true)
                            .child(icon_svg("icons/download.svg", 12.0, 0xffffffff))
                            .child(SharedString::from(label))
                            .on_click(cx.listener(move |this, _, _, cx| this.start_model_download(id, cx))),
                    );
                }
                if *partial_bytes > 0 {
                    buttons = buttons.child(
                        self.pill_btn(format!("{id}-delete"), false)
                            .child(icon_svg("icons/trash.svg", 12.0, theme.text_primary))
                            .child("받은 조각 지우기")
                            .on_click(cx.listener(move |this, _, _, cx| this.delete_model(id, cx))),
                    );
                }
            }
        }

        let progress = match &state {
            ModelState::Downloading { got, total: Some(t) } if *t > 0 => Some(self.progress_bar(*got as f32 / *t as f32)),
            ModelState::Downloading { .. } => Some(self.progress_bar(0.02)),
            _ => None,
        };

        let is_rec = self.sys_info.is_recommended(id);
        let mem_warn = self.sys_info.memory_warning(id);

        div()
            .flex()
            .flex_col()
            .gap_2()
            .px(px(if is_compact { 14.0 } else { 18.0 }))
            .py_3()
            .rounded(px(18.0))
            .bg(linear_gradient(
                180.0,
                linear_color_stop(rgba(theme.card_bg_top), 0.0),
                linear_color_stop(rgba(theme.card_bg_bottom), 1.0),
            ))
            .border_1()
            .border_color(rgba(if in_use { TALKIT_BLUE } else { theme.card_border }))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(icon_svg("icons/cpu.svg", 15.0, TALKIT_BLUE))
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .flex()
                            .flex_col()
                            .child(
                                div()
                                    .flex()
                                    .flex_wrap()
                                    .items_center()
                                    .gap_2()
                                    .text_size(px(13.5))
                                    .text_color(rgba(theme.text_primary))
                                    .child(spec.label)
                                    .when(in_use, |d| {
                                        d.child(
                                            div()
                                                .px_2()
                                                .rounded(px(999.0))
                                                .bg(rgba(TALKIT_BLUE))
                                                .text_size(px(10.0))
                                                .text_color(rgba(0xffffffff))
                                                .child("사용 중"),
                                        )
                                    })
                                    .when(is_rec, |d| {
                                        d.child(
                                            div()
                                                .px_2()
                                                .rounded(px(999.0))
                                                .bg(rgba(0x10b98122))
                                                .border_1()
                                                .border_color(rgba(0x10b98166))
                                                .text_size(px(10.0))
                                                .text_color(rgba(0x10b981ff))
                                                .child("이 Mac 추천"),
                                        )
                                    }),
                            )
                            .child(
                                div()
                                    .text_size(px(11.0))
                                    .text_color(rgba(theme.text_muted))
                                    .child(spec.desc),
                            )
                            .when_some(mem_warn, |d, warn| {
                                d.child(
                                    div()
                                        .text_size(px(10.5))
                                        .text_color(rgba(0xd97706ff))
                                        .child(warn),
                                )
                            }),
                    ),
            )
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(rgba(state_color))
                    .child(SharedString::from(state_line)),
            )
            .children(progress)
            .child(buttons)
            .when(!is_compact, |d| {
                d.child(
                    div()
                        .text_size(px(10.0))
                        .text_color(rgba(theme.text_muted))
                        .truncate()
                        .child(SharedString::from(format!("저장 위치: {}", dir.display()))),
                )
            })
    }

    /// 음성 인식 모델 관리 화면: 모델마다 카드 한 장
    fn render_model_panel(&self, is_compact: bool, cx: &mut Context<Self>) -> AnyElement {
        let theme = self.preset.theme();
        let rec_choice = self.sys_info.recommended_choice();
        let rec_label = updates::CHOICES
            .iter()
            .find(|c| c.0 == rec_choice)
            .map(|c| c.1)
            .unwrap_or("기본 모델");
        let rec_ready = updates::required_models(rec_choice)
            .iter()
            .all(|id| self.is_installed(id));
        let rec_active = self.model_choice == rec_choice && rec_ready;

        let rec_banner = div()
            .flex_none()
            .flex()
            .flex_wrap()
            .items_center()
            .gap_2()
            .px_3()
            .py_2()
            .rounded(px(14.0))
            .bg(rgba(0x4f6bff1a))
            .border_1()
            .border_color(rgba(0x4f6bff4d))
            .child(icon_svg("icons/sparkles.svg", 13.0, TALKIT_BLUE))
            .child(
                div()
                    .flex_1()
                    .min_w(px(160.0))
                    .text_size(px(11.5))
                    .text_color(rgba(theme.text_primary))
                    .child(SharedString::from(self.sys_info.summary_line())),
            )
            .when(rec_active, |d| {
                d.child(
                    div()
                        .px_2p5()
                        .py(px(2.0))
                        .rounded(px(999.0))
                        .bg(rgba(0x10b98122))
                        .border_1()
                        .border_color(rgba(0x10b98166))
                        .text_size(px(10.5))
                        .text_color(rgba(0x10b981ff))
                        .child("추천 조합 사용 중"),
                )
            })
            .when(!rec_active && (rec_ready || self.downloading.is_none()), |d| {
                let btn_label = if rec_ready {
                    format!("추천({rec_label}) 바로 쓰기")
                } else {
                    format!("추천({rec_label}) 받기")
                };
                d.child(
                    self.pill_btn("use-recommended-model", true)
                        .child(icon_svg(
                            if rec_ready { "icons/check.svg" } else { "icons/download.svg" },
                            11.5,
                            0xffffffff,
                        ))
                        .child(SharedString::from(btn_label))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.set_model(rec_choice.to_string(), cx);
                        })),
                )
            });

        let header = div()
            .flex_none()
            .flex()
            .items_center()
            .gap_2()
            .px_1()
            .when(self.view_mode == ViewMode::Models, |d| {
                d.child(
                    self.pill_btn("models-back-btn", false)
                        .child(icon_svg("icons/back.svg", 11.5, theme.text_primary))
                        .child("실시간")
                        .on_click(cx.listener(|this, _, _, cx| this.close_history(cx))),
                )
            })
            .child(
                div()
                    .flex_1()
                    .text_size(px(13.0))
                    .text_color(rgba(theme.text_primary))
                    .child(SharedString::from(format!("음성 인식 모델 · 지금: {}", self.choice_label()))),
            );
        div()
            .id("models-scroll")
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .gap_2()
            .overflow_y_scroll()
            .child(header)
            .child(rec_banner)
            .children(
                updates::MODELS
                    .iter()
                    .filter(|spec| updates::is_model_visible(spec))
                    .map(|spec| self.render_model_card(spec, is_compact, cx)),
            )
            .child(
                div()
                    .px_1()
                    .text_size(px(10.5))
                    .text_color(rgba(theme.text_muted))
                    .child("모든 모델은 로그인 없이 받을 수 있어요. 인터넷이 없는 Mac 은 다른 Mac 에서 받은 모델 폴더를 같은 위치에 넣어도 됩니다."),
            )
            .into_any_element()
    }

    /// 새 버전 알림 줄 (없으면 None)
    fn render_update_banner(&self, is_compact: bool, cx: &mut Context<Self>) -> Option<AnyElement> {
        let theme = self.preset.theme();
        let row = |id: &'static str| {
            div()
                .id(id)
                .flex_none()
                .flex()
                .flex_wrap()
                .items_center()
                .gap_2()
                .px_3()
                .py_2()
                .rounded(px(14.0))
                .bg(rgba(0x4f6bff1f))
                .border_1()
                .border_color(rgba(0x4f6bff59))
                .text_size(px(12.0))
                .text_color(rgba(theme.text_primary))
        };
        let el = match &self.update {
            UpdateState::None => return None,
            UpdateState::Available(rel) => {
                let notes = updates::short_notes(&rel.notes, 1);
                let page = rel.page.clone();
                let rel2 = rel.clone();
                row("update-banner")
                    .child(icon_svg("icons/sparkles.svg", 13.0, TALKIT_BLUE))
                    .child(SharedString::from(format!("Talkit {} 이 나왔어요", rel.version)))
                    .when(!is_compact && !notes.is_empty(), |d| {
                        d.child(
                            div()
                                .flex_1()
                                .min_w(px(0.0))
                                .truncate()
                                .text_size(px(11.0))
                                .text_color(rgba(theme.text_secondary))
                                .child(SharedString::from(notes[0].clone())),
                        )
                    })
                    .when(is_compact || notes.is_empty(), |d| d.child(div().flex_1()))
                    .child(
                        self.pill_btn("update-notes", false)
                            .child("바뀐 내용")
                            .on_click(move |_, _, _| {
                                let _ = Command::new("/usr/bin/open").arg(&page).spawn();
                            }),
                    )
                    .child(
                        self.pill_btn("update-start", true)
                            .child(icon_svg("icons/download.svg", 12.0, 0xffffffff))
                            .child("업데이트")
                            .on_click(cx.listener(move |this, _, _, cx| this.start_update_download(rel2.clone(), cx))),
                    )
                    .child(
                        self.pill_btn("update-later", false)
                            .child("나중에")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.update = UpdateState::None;
                                cx.notify();
                            })),
                    )
            }
            UpdateState::Downloading { rel, got } => {
                let frac = if rel.size > 0 { *got as f32 / rel.size as f32 } else { 0.0 };
                row("update-banner")
                    .child(icon_svg("icons/download.svg", 13.0, TALKIT_BLUE))
                    .child(SharedString::from(format!(
                        "Talkit {} 받는 중 · {}%",
                        rel.version,
                        ((frac * 100.0) as u32).min(99)
                    )))
                    .child(div().flex_1().min_w(px(80.0)).child(self.progress_bar(frac)))
            }
            UpdateState::Ready { rel, .. } => row("update-banner")
                .child(icon_svg("icons/check.svg", 13.0, TALKIT_BLUE))
                .child(SharedString::from(format!("Talkit {} 설치 준비 완료 · 다시 켜면 바뀝니다", rel.version)))
                .child(div().flex_1())
                .child(
                    self.pill_btn("update-install", true)
                        .child("다시 켜서 설치")
                        .on_click(cx.listener(|this, _, _, cx| this.install_update(cx))),
                ),
            UpdateState::Failed { rel, msg } => {
                let retry = rel.clone();
                row("update-banner")
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .text_color(rgba(0xfb7185ff))
                            .child(SharedString::from(format!("업데이트 실패: {msg}"))),
                    )
                    .when_some(retry, |d, rel| {
                        d.child(
                            self.pill_btn("update-retry", false)
                                .child("다시 시도")
                                .on_click(cx.listener(move |this, _, _, cx| this.start_update_download(rel.clone(), cx))),
                        )
                    })
                    .child(
                        self.pill_btn("update-close", false)
                            .child("닫기")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.update = UpdateState::None;
                                cx.notify();
                            })),
                    )
            }
        };
        Some(el.into_any_element())
    }

    fn render_history_panel(
        &self,
        is_compact: bool,
        body_font: f32,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = self.preset.theme();
        let current = self
            .save_path
            .as_ref()
            .map(|p| fs::canonicalize(p).unwrap_or_else(|_| p.clone()));

        let small_btn = |id: &'static str| {
            div()
                .id(id)
                .flex_none()
                .h(px(24.0))
                .px_2p5()
                .rounded(px(999.0))
                .cursor_pointer()
                .bg(rgba(theme.pill_bg))
                .border_1()
                .border_color(rgba(theme.pill_border))
                .hover(|d| d.bg(rgba(theme.pill_hover_bg)))
                .flex()
                .items_center()
                .gap_1()
                .text_size(px(11.0))
                .text_color(rgba(theme.text_primary))
        };

        let card = |id: SharedString| {
            div()
                .id(id)
                .px(px(if is_compact { 12.0 } else { 16.0 }))
                .py_2p5()
                .rounded(px(18.0))
                .cursor_pointer()
                .bg(linear_gradient(
                    180.0,
                    linear_color_stop(rgba(theme.card_bg_top), 0.0),
                    linear_color_stop(rgba(theme.card_bg_bottom), 1.0),
                ))
                .border_1()
                .border_color(rgba(theme.card_border))
                .hover(|d| d.bg(rgba(theme.card_hover_bg)))
                .flex()
                .items_start()
                .gap(px(if is_compact { 8.0 } else { 12.0 }))
        };

        // 위쪽 제목 줄
        let header = match (&self.view_mode, &self.history_detail) {
            (ViewMode::HistoryDetail, Some((entry, lines))) => {
                let all_text = lines
                    .iter()
                    .map(|l| match &l.source {
                        Some(s) => format!("[{}] {}", s, l.body),
                        None => l.body.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let all_copied = self.history_copied == Some(usize::MAX);
                div()
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap_2()
                    .px_1()
                    .child(
                        small_btn("history-back-btn")
                            .child(icon_svg("icons/back.svg", 11.5, theme.text_primary))
                            .child("목록")
                            .on_click(cx.listener(|this, _, _, cx| this.open_history(cx))),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .truncate()
                            .text_size(px(13.0))
                            .text_color(rgba(theme.text_primary))
                            .child(SharedString::from(format!(
                                "{} · {}문장",
                                entry.title,
                                lines.len()
                            ))),
                    )
                    .child(
                        small_btn("history-copy-all-btn")
                            .child(icon_svg(
                                if all_copied { "icons/check.svg" } else { "icons/copy.svg" },
                                11.5,
                                theme.text_primary,
                            ))
                            .child(if all_copied { "복사됨" } else { "전체복사" })
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.copy_history(usize::MAX, all_text.clone(), cx)
                            })),
                    )
            }
            _ => div()
                .flex_none()
                .flex()
                .items_center()
                .gap_2()
                .px_1()
                .child(
                    div()
                        .flex_1()
                        .text_size(px(13.0))
                        .text_color(rgba(theme.text_primary))
                        .child(SharedString::from(format!(
                            "이전 녹음 {}개",
                            self.history.len()
                        ))),
                )
                .child(
                    small_btn("history-refresh-btn")
                        .child("새로고침")
                        .on_click(cx.listener(|this, _, _, cx| this.open_history(cx))),
                ),
        };

        let mut list = div()
            .id("history-scroll")
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .gap_2()
            .px_1()
            .py_1()
            .overflow_y_scroll()
            .track_scroll(&self.history_scroll);

        match (&self.view_mode, &self.history_detail) {
            (ViewMode::HistoryDetail, Some((_, lines))) => {
                if lines.is_empty() {
                    list = list.child(
                        div()
                            .m_auto()
                            .text_size(px(12.5))
                            .text_color(rgba(theme.text_muted))
                            .child("이 녹음에는 전사된 문장이 없습니다."),
                    );
                }
                list = list.children(lines.iter().enumerate().map(|(i, line)| {
                    let is_mic = line.source.as_ref().map(|s| s.as_ref()) == Some("마이크");
                    let accent = if is_mic { 0x10b981ff } else { TALKIT_BLUE };
                    let copied = self.history_copied == Some(i);
                    let text = line.body.to_string();
                    let label = SharedString::from(match (&line.source, &line.clock) {
                        (Some(s), Some(c)) if !is_compact => format!("{} {}", s, c),
                        (Some(s), _) => s.to_string(),
                        (None, Some(c)) => c.to_string(),
                        (None, None) => "문장".to_string(),
                    });
                    card(SharedString::from(format!("history-line-{}", i)))
                        .when(copied, |d| d.border_color(rgba(TALKIT_BLUE)))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.copy_history(i, text.clone(), cx)
                        }))
                        .child(
                            div()
                                .flex_none()
                                .mt(px(3.0))
                                .w(px(3.5))
                                .h(px(20.0))
                                .rounded(px(999.0))
                                .bg(rgba(accent)),
                        )
                        .child(
                            div()
                                .flex_none()
                                .mt(px(1.0))
                                .px_2()
                                .py(px(2.0))
                                .rounded(px(999.0))
                                .bg(rgba(theme.pill_bg))
                                .border_1()
                                .border_color(rgba(theme.pill_border))
                                .text_size(px(11.0))
                                .text_color(rgba(accent))
                                .child(label),
                        )
                        .child(
                            div()
                                .flex_1()
                                .min_w(px(0.0))
                                .text_size(px(body_font))
                                .text_color(rgba(theme.text_primary))
                                .child(line.body.clone()),
                        )
                        .child(
                            div()
                                .flex_none()
                                .text_size(px(11.0))
                                .text_color(rgba(if copied { TALKIT_BLUE } else { theme.text_muted }))
                                .child(if copied { "복사됨" } else { "" }),
                        )
                }));
            }
            _ => {
                if self.history.is_empty() {
                    list = list.child(
                        div()
                            .m_auto()
                            .text_size(px(12.5))
                            .text_color(rgba(theme.text_muted))
                            .child("아직 저장된 녹음이 없습니다. (~/Documents/Talkit)"),
                    );
                }
                list = list.children(self.history.iter().enumerate().map(|(i, entry)| {
                    let is_current = current.as_ref().map_or(false, |c| {
                        fs::canonicalize(&entry.path).unwrap_or_else(|_| entry.path.clone()) == *c
                    });
                    card(SharedString::from(format!("history-entry-{}", i)))
                        .items_center()
                        .on_click(cx.listener(move |this, _, _, cx| this.open_history_entry(i, cx)))
                        .child(icon_svg("icons/history.svg", 15.0, TALKIT_BLUE))
                        .child(
                            div()
                                .flex_1()
                                .min_w(px(0.0))
                                .flex()
                                .flex_col()
                                .gap(px(2.0))
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap_2()
                                        .child(
                                            div()
                                                .text_size(px(14.0))
                                                .text_color(rgba(theme.text_primary))
                                                .child(entry.title.clone()),
                                        )
                                        .when(is_current, |d| {
                                            d.child(
                                                div()
                                                    .px_1p5()
                                                    .rounded(px(999.0))
                                                    .bg(rgba(TALKIT_BLUE))
                                                    .text_size(px(10.0))
                                                    .text_color(rgba(0xffffffff))
                                                    .child("지금 녹음 중"),
                                            )
                                        }),
                                )
                                .child(
                                    div()
                                        .truncate()
                                        .text_size(px(12.0))
                                        .text_color(rgba(theme.text_muted))
                                        .child(entry.preview.clone()),
                                ),
                        )
                        .child(
                            div()
                                .flex_none()
                                .text_size(px(11.0))
                                .text_color(rgba(theme.text_secondary))
                                .child(SharedString::from(format!("{}문장", entry.lines))),
                        )
                }));
            }
        }

        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .gap_2()
            .child(header)
            .child(list)
    }

    fn render_studio_mode(&self, vw: f32, vh: f32, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = self.preset.theme();
        let is_compact = vw < 680.0;
        let is_medium = vw < 860.0;
        let is_short = vh < 440.0;

        let has_partial = self.has_any_partial();
        let last_sys_open_idx = self
            .items
            .iter()
            .rev()
            .find(|it| !it.is_error && it.source.as_ref().map(|s| s.as_ref()) == Some("시스템"))
            .and_then(|it| if !it.sentence_complete { Some(it.index) } else { None });
        let last_mic_open_idx = self
            .items
            .iter()
            .rev()
            .find(|it| !it.is_error && it.source.as_ref().map(|s| s.as_ref()) == Some("마이크"))
            .and_then(|it| if !it.sentence_complete { Some(it.index) } else { None });
        let last_default_open_idx = self
            .items
            .iter()
            .rev()
            .find(|it| !it.is_error && it.source.is_none())
            .and_then(|it| if !it.sentence_complete { Some(it.index) } else { None });

        let mut pending_live_cards: Vec<(Option<SharedString>, SharedString)> = Vec::new();
        if !self.partial_sys.trim().is_empty() && last_sys_open_idx.is_none() {
            pending_live_cards.push((
                Some(SharedString::from("시스템")),
                SharedString::from(self.partial_sys.trim().to_string()),
            ));
        }
        if !self.partial_mic.trim().is_empty() && last_mic_open_idx.is_none() {
            pending_live_cards.push((
                Some(SharedString::from("마이크")),
                SharedString::from(self.partial_mic.trim().to_string()),
            ));
        }
        if !self.partial.trim().is_empty() && last_default_open_idx.is_none() {
            pending_live_cards.push((None, SharedString::from(self.partial.trim().to_string())));
        }

        let elapsed = self.format_elapsed();
        let mem_badge = SharedString::from(if is_compact {
            format!("{:.1}GB", self.mem_mb / 1024.0)
        } else {
            format!("RAM {:.2} GB", self.mem_mb / 1024.0)
        });
        let count_badge = SharedString::from(format!("{} 문장", self.total_committed));
        let all_copied = self.copied_index == Some(0);
        let range_copied = self.copied_index == Some(usize::MAX - 2);

        let selection_count = self
            .selected_range()
            .map(|(lo, hi)| hi.saturating_sub(lo) + 1)
            .unwrap_or(0);

        let latest_status = self
            .status
            .back()
            .cloned()
            .unwrap_or_else(|| "Talk it, Get it. 말하는 순간, 바로 텍스트로.".into());
        let source_summary: SharedString = self
            .status
            .iter()
            .rev()
            .find(|s| s.starts_with("마이크:") || s.starts_with("시스템:"))
            .cloned()
            .unwrap_or_else(|| "Talkit · Qwen3-ASR 4-bit".into());

        let outer_pad = if is_compact { 8.0 } else { 12.0 };
        let shell_px = if is_compact { 12.0 } else { 20.0 };
        let shell_pb = if is_short { 10.0 } else { 16.0 };
        let header_bars = if is_compact {
            6
        } else if is_medium {
            9
        } else {
            12
        };
        let body_font = if is_compact { 14.5 } else { 15.5 };

        div()
            .size_full()
            .p(px(outer_pad))
            .flex()
            .flex_col()
            // Release drag selection anywhere inside the studio window
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    this.finish_card_drag(cx);
                }),
            )
            // Outer Concentric Talkit Studio Capsule Shell
            .child(
                div()
                    .size_full()
                    .px(px(shell_px))
                    .pt_3()
                    .pb(px(shell_pb))
                    .rounded(px(if is_compact { 22.0 } else { 26.0 }))
                    .bg(linear_gradient(
                        180.0,
                        linear_color_stop(rgba(theme.shell_bg_top), 0.0),
                        linear_color_stop(rgba(theme.shell_bg_bottom), 1.0),
                    ))
                    .border_1()
                    .border_color(rgba(if has_partial {
                        theme.shell_active_border
                    } else {
                        theme.shell_border
                    }))
                    .shadow(glass_card_shadows())
                    .flex()
                    .flex_col()
                    .gap(px(if is_short { 8.0 } else { 12.0 }))
                    // 1. Top Talkit Brand Header Bar
                    .child(
                        div()
                            .flex_none()
                            .ml(px(if is_compact { 62.0 } else { 68.0 }))
                            .min_h(px(44.0))
                            .px(px(if is_compact { 12.0 } else { 18.0 }))
                            .py(px(5.0))
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_between()
                            .gap_2()
                            .rounded(px(999.0))
                            .bg(linear_gradient(
                                180.0,
                                linear_color_stop(rgba(theme.header_bg_top), 0.0),
                                linear_color_stop(rgba(theme.header_bg_bottom), 1.0),
                            ))
                            .border_1()
                            .border_color(rgba(theme.header_border))
                            .shadow(glass_pill_shadows())
                            // Left: Talkit App Squircle Icon + "Talkit" Wordmark + Slogan + Live Waveform + Timer + RAM
                            .child(
                                div()
                                    .id("studio-drag-handle")
                                    .flex_1()
                                    .cursor_pointer()
                                    .flex()
                                    .items_center()
                                    .gap_2p5()
                                    .min_w(px(0.0))
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|_, _: &MouseDownEvent, window, _| {
                                            #[cfg(target_os = "macos")]
                                            macos_glass::start_window_drag(window, true);
                                        }),
                                    )
                                    // Talkit Squircle App Icon (#4F6BFF background + white 5-bar wave)
                                    .child(
                                        div()
                                            .flex_none()
                                            .size(px(28.0))
                                            .rounded(px(8.5))
                                            .bg(rgba(TALKIT_BLUE))
                                            .flex()
                                            .items_center()
                                            .justify_center()
                                            .shadow(glass_pill_shadows())
                                            .child(icon_svg(
                                                "icons/talkit_logo.svg",
                                                16.5,
                                                0xffffffff,
                                            )),
                                    )
                                    .child(
                                        div()
                                            .flex_none()
                                            .text_size(px(if is_compact { 14.5 } else { 16.0 }))
                                            .text_color(rgba(theme.text_primary))
                                            .child("Talkit"),
                                    )
                                    .when(!is_compact, |d| {
                                        d.child(
                                            div()
                                                .flex_none()
                                                .px_2()
                                                .py(px(2.0))
                                                .rounded(px(999.0))
                                                .bg(rgba(theme.pill_bg))
                                                .border_1()
                                                .border_color(rgba(theme.pill_border))
                                                .text_size(px(10.5))
                                                .text_color(rgba(TALKIT_BLUE))
                                                .child("Talk it, Get it."),
                                        )
                                    })
                                    .child(self.render_waveform_bars(16.0, header_bars))
                                    .child(
                                        div()
                                            .flex_none()
                                            .px_2()
                                            .py(px(2.0))
                                            .rounded(px(999.0))
                                            .bg(rgba(theme.pill_bg))
                                            .border_1()
                                            .border_color(rgba(theme.pill_border))
                                            .text_size(px(11.0))
                                            .text_color(rgba(theme.text_secondary))
                                            .child(elapsed),
                                    )
                                    .child(
                                        div()
                                            .flex_none()
                                            .px_2()
                                            .py(px(2.0))
                                            .rounded(px(999.0))
                                            .bg(rgba(0x4f6bff1c))
                                            .border_1()
                                            .border_color(rgba(0x4f6bff4d))
                                            .text_size(px(11.0))
                                            .text_color(rgba(TALKIT_BLUE))
                                            .child(mem_badge),
                                    ),
                            )
                            // Right: PiP Mode Toggle Button in Talkit Blue (#4F6BFF)
                            .child(
                                div()
                                    .flex_none()
                                    .flex()
                                    .items_center()
                                    .gap_1p5()
                                    // 모델 관리 버튼: 받는 중이면 %, 없으면 '모델 필요'
                                    .child({
                                        let in_models = self.view_mode == ViewMode::Models;
                                        let dl_state = self.downloading.map(|id| self.state_of(id));
                                        let (label, alert): (SharedString, bool) = match dl_state {
                                            Some(ModelState::Downloading { got, total: Some(t) }) if t > 0 => {
                                                (format!("모델 {}%", (got * 100 / t).min(99)).into(), true)
                                            }
                                            Some(_) => ("모델 받는 중".into(), true),
                                            None if self.chosen_ready() => (self.choice_label().into(), false),
                                            None => ("모델 필요".into(), true),
                                        };
                                        let hi = in_models || alert;
                                        div()
                                            .id("models-btn")
                                            .flex_none()
                                            .h(px(30.0))
                                            .px_3()
                                            .rounded(px(999.0))
                                            .cursor_pointer()
                                            .bg(rgba(if hi { 0x4f6bff24 } else { theme.pill_bg }))
                                            .border_1()
                                            .border_color(rgba(if hi { TALKIT_BLUE } else { theme.pill_border }))
                                            .hover(|d| d.bg(rgba(theme.pill_hover_bg)))
                                            .flex()
                                            .items_center()
                                            .gap_1p5()
                                            .text_size(px(11.5))
                                            .text_color(rgba(if hi { TALKIT_BLUE } else { theme.text_primary }))
                                            .child(icon_svg(
                                                "icons/cpu.svg",
                                                13.0,
                                                if hi { TALKIT_BLUE } else { theme.text_primary },
                                            ))
                                            .when(!is_compact || alert, |d| d.child(label))
                                            .on_click(cx.listener(move |this, _, _, cx| {
                                                if in_models {
                                                    this.close_history(cx);
                                                } else {
                                                    this.open_models(cx);
                                                }
                                            }))
                                    })
                                    .child({
                                        let in_history = matches!(
                                            self.view_mode,
                                            ViewMode::HistoryList | ViewMode::HistoryDetail
                                        );
                                        div()
                                            .id("history-btn")
                                            .flex_none()
                                            .h(px(30.0))
                                            .px_3()
                                            .rounded(px(999.0))
                                            .cursor_pointer()
                                            .bg(rgba(if in_history { 0x4f6bff24 } else { theme.pill_bg }))
                                            .border_1()
                                            .border_color(rgba(if in_history {
                                                TALKIT_BLUE
                                            } else {
                                                theme.pill_border
                                            }))
                                            .hover(|d| d.bg(rgba(theme.pill_hover_bg)))
                                            .flex()
                                            .items_center()
                                            .gap_1p5()
                                            .text_size(px(11.5))
                                            .text_color(rgba(if in_history {
                                                TALKIT_BLUE
                                            } else {
                                                theme.text_primary
                                            }))
                                            .child(icon_svg(
                                                if in_history { "icons/back.svg" } else { "icons/history.svg" },
                                                13.0,
                                                if in_history { TALKIT_BLUE } else { theme.text_primary },
                                            ))
                                            .child(if in_history {
                                                if is_compact { "실시간" } else { "실시간으로" }
                                            } else if is_compact {
                                                "기록"
                                            } else {
                                                "녹음 기록"
                                            })
                                            .on_click(cx.listener(move |this, _, _, cx| {
                                                if in_history {
                                                    this.close_history(cx);
                                                } else {
                                                    this.open_history(cx);
                                                }
                                            }))
                                    })
                                    .child(
                                        div()
                                            .id("pip-mode-btn")
                                            .flex_none()
                                            .h(px(30.0))
                                            .px_3p5()
                                            .rounded(px(999.0))
                                            .cursor_pointer()
                                            .bg(rgba(TALKIT_BLUE))
                                            .border_1()
                                            .border_color(rgba(TALKIT_BLUE))
                                            .shadow(glass_pill_shadows())
                                            .hover(|d| d.bg(rgba(0x3b57f5ff)))
                                            .flex()
                                            .items_center()
                                            .gap_1p5()
                                            .text_size(px(11.5))
                                            .text_color(rgba(0xffffffff))
                                            .child(icon_svg("icons/pip.svg", 13.0, 0xffffffff))
                                            .child(if is_compact { "PiP" } else { "PiP 미니창" })
                                            .on_click(cx.listener(|this, _, window, cx| {
                                                this.toggle_pip(window, cx);
                                            })),
                                    ),
                            ),
                    )
                    // 2. Context Telemetry, Brand Theme Presets & Session Action Strip
                    .child(
                        div()
                            .flex_none()
                            .px_3p5()
                            .py(px(6.0))
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_between()
                            .gap_2()
                            .rounded(px(16.0))
                            .bg(rgba(theme.strip_bg))
                            .border_1()
                            .border_color(rgba(theme.strip_border))
                            .text_size(px(11.5))
                            // Left: Sentence count + drag selection badge + status + mic source
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(200.0))
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .overflow_hidden()
                                    .child(
                                        div()
                                            .flex_none()
                                            .px_2()
                                            .py(px(1.5))
                                            .rounded(px(999.0))
                                            .bg(rgba(theme.pill_bg))
                                            .border_1()
                                            .border_color(rgba(theme.pill_border))
                                            .text_size(px(11.0))
                                            .text_color(rgba(theme.text_primary))
                                            .child(count_badge),
                                    )
                                    .when(selection_count > 1, |d| {
                                        let sel_label = SharedString::from(if range_copied {
                                            format!("{}문장 복사됨", selection_count)
                                        } else {
                                            format!("{}문장 선택", selection_count)
                                        });
                                        d.child(
                                            div()
                                                .flex_none()
                                                .px_2()
                                                .py(px(1.5))
                                                .rounded(px(999.0))
                                                .bg(rgba(TALKIT_BLUE))
                                                .flex()
                                                .items_center()
                                                .gap_1()
                                                .text_size(px(10.5))
                                                .text_color(rgba(0xffffffff))
                                                .child(icon_svg("icons/check.svg", 10.5, 0xffffffff))
                                                .child(sel_label),
                                        )
                                    })
                                    .child(
                                        div()
                                            .flex_1()
                                            .min_w(px(0.0))
                                            .truncate()
                                            .text_color(rgba(theme.text_secondary))
                                            .child(latest_status),
                                    )
                                    .when(!is_medium, |d| {
                                        d.child(
                                            div()
                                                .flex_none()
                                                .max_w(px(160.0))
                                                .truncate()
                                                .text_size(px(11.0))
                                                .text_color(rgba(theme.text_muted))
                                                .child(source_summary),
                                        )
                                    }),
                            )
                            // Right: Brand Theme Presets (White / Black / Blue / Glass) + Copy All + Pause + Clear
                            .child(
                                div()
                                    .flex_none()
                                    .flex()
                                    .flex_wrap()
                                    .items_center()
                                    .gap_1()
                                    // 녹음 입력 선택: 마이크 / 시스템 / 둘 다
                                    .child(
                                        div()
                                            .flex_none()
                                            .flex()
                                            .items_center()
                                            .p(px(2.0))
                                            .gap(px(2.0))
                                            .rounded(px(999.0))
                                            .bg(rgba(theme.pill_bg))
                                            .border_1()
                                            .border_color(rgba(theme.pill_border))
                                            .children(SourceMode::all().into_iter().map(|m| {
                                                let active = self.source_mode == m;
                                                div()
                                                    .id(SharedString::from(format!("source-{}", m.arg())))
                                                    .flex_none()
                                                    .h(px(20.0))
                                                    .px_2()
                                                    .rounded(px(999.0))
                                                    .cursor_pointer()
                                                    .flex()
                                                    .items_center()
                                                    .gap_1()
                                                    .text_size(px(11.0))
                                                    .bg(rgba(if active { TALKIT_BLUE } else { 0x00000000 }))
                                                    .text_color(rgba(if active {
                                                        0xffffffff
                                                    } else {
                                                        theme.text_secondary
                                                    }))
                                                    .when(!active, |d| d.hover(|d| d.bg(rgba(theme.pill_hover_bg))))
                                                    .when(m == SourceMode::Mic, |d| {
                                                        d.child(icon_svg(
                                                            "icons/mic.svg",
                                                            10.5,
                                                            if active { 0xffffffff } else { theme.text_secondary },
                                                        ))
                                                    })
                                                    .child(m.label())
                                                    .on_click(cx.listener(move |this, _, _, cx| {
                                                        this.set_source(m, cx);
                                                    }))
                                            })),
                                    )
                                    // 모델 고르기는 '모델' 화면에서 한다 (머리줄 버튼에 지금 모델 이름이 보임)
                                    // 전사 따라가기 토글
                                    .child(
                                        div()
                                            .id("follow-tail-btn")
                                            .flex_none()
                                            .h(px(24.0))
                                            .px_2p5()
                                            .rounded(px(999.0))
                                            .cursor_pointer()
                                            .bg(rgba(if self.follow_tail { TALKIT_BLUE } else { theme.pill_bg }))
                                            .border_1()
                                            .border_color(rgba(if self.follow_tail {
                                                TALKIT_BLUE
                                            } else {
                                                theme.pill_border
                                            }))
                                            .when(!self.follow_tail, |d| d.hover(|d| d.bg(rgba(theme.pill_hover_bg))))
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .text_size(px(11.0))
                                            .text_color(rgba(if self.follow_tail {
                                                0xffffffff
                                            } else {
                                                theme.text_primary
                                            }))
                                            .child(icon_svg(
                                                "icons/arrow_down.svg",
                                                11.0,
                                                if self.follow_tail { 0xffffffff } else { theme.text_primary },
                                            ))
                                            .child(if is_compact {
                                                "따라가기"
                                            } else if self.follow_tail {
                                                "전사 따라가기 켬"
                                            } else {
                                                "전사 따라가기 끔"
                                            })
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.toggle_follow(cx);
                                            })),
                                    )
                                    .child(
                                        div()
                                            .w(px(1.0))
                                            .h(px(14.0))
                                            .mx_1()
                                            .bg(rgba(theme.pill_border)),
                                    )
                                    .children(
                                        GlassPreset::all()
                                            .into_iter()
                                            .map(|p| self.render_preset_pill(p, cx)),
                                    )
                                    .child(
                                        div()
                                            .w(px(1.0))
                                            .h(px(14.0))
                                            .mx_1()
                                            .bg(rgba(theme.pill_border)),
                                    )
                                    .child(
                                        div()
                                            .id("studio-copy-all-btn")
                                            .flex_none()
                                            .h(px(24.0))
                                            .px_2p5()
                                            .rounded(px(999.0))
                                            .cursor_pointer()
                                            .bg(rgba(if all_copied {
                                                TALKIT_BLUE
                                            } else {
                                                theme.pill_bg
                                            }))
                                            .border_1()
                                            .border_color(rgba(if all_copied {
                                                TALKIT_BLUE
                                            } else {
                                                theme.pill_border
                                            }))
                                            .hover(|d| d.bg(rgba(theme.pill_hover_bg)))
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .text_size(px(11.0))
                                            .text_color(rgba(if all_copied {
                                                0xffffffff
                                            } else {
                                                theme.text_primary
                                            }))
                                            .child(if all_copied {
                                                icon_svg("icons/check.svg", 11.5, 0xffffffff)
                                            } else {
                                                icon_svg("icons/copy.svg", 11.5, theme.text_primary)
                                            })
                                            .child(if is_compact {
                                                if all_copied { "복사됨" } else { "전체" }
                                            } else if all_copied {
                                                "전체복사됨"
                                            } else {
                                                "전체복사"
                                            })
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.copy_all(cx);
                                            })),
                                    )
                                    .child(
                                        div()
                                            .id("studio-pause-btn")
                                            .flex_none()
                                            .h(px(24.0))
                                            .px_2p5()
                                            .rounded(px(999.0))
                                            .cursor_pointer()
                                            .bg(rgba(theme.pill_bg))
                                            .border_1()
                                            .border_color(rgba(theme.pill_border))
                                            .hover(|d| d.bg(rgba(theme.pill_hover_bg)))
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .text_size(px(11.0))
                                            .text_color(rgba(theme.text_primary))
                                            .child(if self.paused {
                                                icon_svg("icons/play.svg", 11.0, TALKIT_BLUE)
                                            } else {
                                                icon_svg("icons/pause.svg", 11.0, theme.text_primary)
                                            })
                                            .child(if is_compact {
                                                if self.paused { "재개" } else { "정지" }
                                            } else if self.paused {
                                                "재개"
                                            } else {
                                                "일시정지"
                                            })
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.toggle_pause(cx);
                                            })),
                                    )
                                    .child(
                                        div()
                                            .id("studio-clear-btn")
                                            .flex_none()
                                            .h(px(24.0))
                                            .px_2p5()
                                            .rounded(px(999.0))
                                            .cursor_pointer()
                                            .bg(rgba(theme.pill_bg))
                                            .border_1()
                                            .border_color(rgba(theme.pill_border))
                                            .hover(|d| d.bg(rgba(0xf43f5e28)))
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .text_size(px(11.0))
                                            .text_color(rgba(theme.text_secondary))
                                            .child(icon_svg(
                                                "icons/trash.svg",
                                                11.5,
                                                theme.text_secondary,
                                            ))
                                            .when(!is_compact, |d| d.child("비우기"))
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.clear_history(cx);
                                            })),
                                    ),
                            ),
                    )
                    // 3. Scrollable Transcript Feed (Brand Concept Card when empty + Drag-to-Select Cards)
                    // 새 버전 알림
                    .children(self.render_update_banner(is_compact, cx))
                    // 모델 관리 화면 (직접 열었거나, 기본 모델이 아직 없어서 녹음을 못 할 때)
                    .child(if self.view_mode == ViewMode::Models
                        || (self.view_mode == ViewMode::Live && !self.chosen_ready())
                    {
                        self.render_model_panel(is_compact, cx)
                    } else if self.view_mode != ViewMode::Live {
                        self.render_history_panel(is_compact, body_font, cx).into_any_element()
                    } else {
                        div()
                            .id("transcript")
                            .flex()
                            .flex_col()
                            .flex_1()
                            .min_h(px(0.0))
                            .gap_2p5()
                            .px_1()
                            .py_1()
                            .overflow_y_scroll()
                            .track_scroll(&self.scroll)
                            // 위로 스크롤하면 따라가기를 끈다 (읽는 중에 끌려 내려가지 않게)
                            .on_scroll_wheel(cx.listener(|this, ev: &ScrollWheelEvent, _, cx| {
                                if this.follow_tail && ev.delta.pixel_delta(px(16.0)).y > px(0.0) {
                                    this.follow_tail = false;
                                    cx.notify();
                                }
                            }))
                            .when(self.items.is_empty() && !has_partial, |d| {
                                // Matches the "Brand Concept: Talk it, Get it. / 말하는 순간, 바로 텍스트로." card in the Guide
                                d.child(
                                    div()
                                        .m_auto()
                                        .px(px(if is_compact { 22.0 } else { 36.0 }))
                                        .py(px(if is_short { 18.0 } else { 28.0 }))
                                        .rounded(px(24.0))
                                        .bg(linear_gradient(
                                            180.0,
                                            linear_color_stop(rgba(theme.card_bg_top), 0.0),
                                            linear_color_stop(rgba(theme.card_bg_bottom), 1.0),
                                        ))
                                        .border_1()
                                        .border_color(rgba(theme.card_border))
                                        .shadow(glass_card_shadows())
                                        .flex()
                                        .flex_col()
                                        .items_center()
                                        .gap_2p5()
                                        // Talkit Squircle App Icon (#4F6BFF)
                                        .child(
                                            div()
                                                .size(px(52.0))
                                                .rounded(px(16.0))
                                                .bg(rgba(TALKIT_BLUE))
                                                .flex()
                                                .items_center()
                                                .justify_center()
                                                .shadow(glass_card_shadows())
                                                .child(icon_svg(
                                                    "icons/talkit_logo.svg",
                                                    30.0,
                                                    0xffffffff,
                                                )),
                                        )
                                        .child(
                                            div()
                                                .text_size(px(if is_compact { 20.0 } else { 24.0 }))
                                                .text_color(rgba(theme.text_primary))
                                                .child("Talk it, Get it."),
                                        )
                                        .child(
                                            div()
                                                .text_size(px(body_font))
                                                .text_color(rgba(TALKIT_BLUE))
                                                .child("말하는 순간, 바로 텍스트로."),
                                        )
                                        .child(
                                            div()
                                                .text_size(px(12.0))
                                                .text_color(rgba(theme.text_muted))
                                                .child("연속 스트리밍 전사 후 문장 단위로 자동 분리되며, 문장을 클릭하거나 드래그해 즉시 복사할 수 있습니다."),
                                        ),
                                )
                            })
                            .children(self.items.iter().map(|item| {
                                let is_err = item.is_error;
                                let item_idx = item.index;
                                let is_copied = self.copied_index == Some(item_idx)
                                    || (!is_err && self.copied_index == Some(0));
                                let is_drag_selected = self.is_in_drag_selection(item_idx);

                                let is_mic =
                                    item.source.as_ref().map(|s| s.as_ref()) == Some("마이크");
                                let accent_color = if is_err {
                                    0xfb7185ff
                                } else if is_mic {
                                    0x10b981ff
                                } else {
                                    TALKIT_BLUE
                                };

                                let idx_label = if is_err {
                                    SharedString::from("오류")
                                } else if let Some(ref src) = item.source {
                                    SharedString::from(format!("{} #{}", src, item_idx))
                                } else {
                                    SharedString::from(format!("문장 #{}", item_idx))
                                };
                                let ts = item.timestamp.clone();

                                let is_open_for_source = match item.source.as_ref().map(|s| s.as_ref()) {
                                    Some("시스템") => Some(item_idx) == last_sys_open_idx,
                                    Some("마이크") => Some(item_idx) == last_mic_open_idx,
                                    _ => Some(item_idx) == last_default_open_idx,
                                };
                                let src_partial = self.partial_for_source(&item.source);
                                let combined_body = if !is_err
                                    && is_open_for_source
                                    && !src_partial.trim().is_empty()
                                {
                                    SharedString::from(format!(
                                        "{} {}",
                                        item.body.trim_end(),
                                        src_partial.trim()
                                    ))
                                } else {
                                    item.body.clone()
                                };
                                let body = combined_body.clone();
                                let copy_str = combined_body.to_string();

                                div()
                                    .id(SharedString::from(format!("transcript-card-{}", item_idx)))
                                    .px(px(if is_compact { 12.0 } else { 16.0 }))
                                    .py_2p5()
                                    .rounded(px(18.0))
                                    .cursor_pointer()
                                    .bg(if is_err {
                                        linear_gradient(
                                            180.0,
                                            linear_color_stop(rgba(0xf43f5e28), 0.0),
                                            linear_color_stop(rgba(0x9f123918), 1.0),
                                        )
                                    } else if is_drag_selected || is_copied {
                                        linear_gradient(
                                            180.0,
                                            linear_color_stop(rgba(0x4f6bff24), 0.0),
                                            linear_color_stop(rgba(0x4f6bff12), 1.0),
                                        )
                                    } else {
                                        linear_gradient(
                                            180.0,
                                            linear_color_stop(rgba(theme.card_bg_top), 0.0),
                                            linear_color_stop(rgba(theme.card_bg_bottom), 1.0),
                                        )
                                    })
                                    .border_1()
                                    .border_color(rgba(if is_err {
                                        0xfb718566
                                    } else if is_drag_selected || is_copied {
                                        TALKIT_BLUE
                                    } else {
                                        theme.card_border
                                    }))
                                    .shadow(glass_card_shadows())
                                    .hover(|d| d.bg(rgba(theme.card_hover_bg)))
                                    .flex()
                                    .items_start()
                                    .gap(px(if is_compact { 8.0 } else { 12.0 }))
                                    // Sentence card click / drag-to-select handlers
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _: &MouseDownEvent, _, cx| {
                                            this.start_card_drag(item_idx, cx);
                                        }),
                                    )
                                    .on_mouse_move(cx.listener(
                                        move |this, ev: &MouseMoveEvent, _, cx| {
                                            this.update_card_drag(item_idx, ev, cx);
                                        },
                                    ))
                                    .on_click(cx.listener(move |this, _, _, cx| {
                                        if this.selected_range().map(|(a, b)| a != b).unwrap_or(false)
                                        {
                                            this.copy_selected_range(cx);
                                        } else {
                                            this.copy_text(item_idx, copy_str.clone(), cx);
                                        }
                                    }))
                                    // Left vertical source accent pill (Blue = 시스템, Emerald = 마이크)
                                    .child(
                                        div()
                                            .flex_none()
                                            .mt(px(3.0))
                                            .w(px(3.5))
                                            .h(px(20.0))
                                            .rounded(px(999.0))
                                            .bg(rgba(accent_color)),
                                    )
                                    // Source + Sentence Index + Timestamp pill
                                    .child(
                                        div()
                                            .flex_none()
                                            .mt(px(1.0))
                                            .flex()
                                            .items_center()
                                            .gap_1p5()
                                            .px_2()
                                            .py(px(2.0))
                                            .rounded(px(999.0))
                                            .bg(rgba(theme.pill_bg))
                                            .border_1()
                                            .border_color(rgba(theme.pill_border))
                                            .text_size(px(11.0))
                                            .child(
                                                div()
                                                    .text_color(rgba(accent_color))
                                                    .child(idx_label),
                                            )
                                            .when(!is_compact, |d| {
                                                d.when_some(ts, |d, stamp| {
                                                    d.child(
                                                        div()
                                                            .text_color(rgba(theme.text_muted))
                                                            .child(stamp),
                                                    )
                                                })
                                            }),
                                    )
                                    // Sentence text body
                                    .child(
                                        div()
                                            .flex_1()
                                            .min_w(px(0.0))
                                            .text_size(px(body_font))
                                            .text_color(rgba(if is_err {
                                                0xe11d48ff
                                            } else {
                                                theme.text_primary
                                            }))
                                            .child(body),
                                    )
                                    // Copy action pill button
                                    .child(
                                        div()
                                            .flex_none()
                                            .h(px(24.0))
                                            .px_2()
                                            .rounded(px(999.0))
                                            .bg(rgba(if is_copied || is_drag_selected {
                                                TALKIT_BLUE
                                            } else {
                                                theme.pill_bg
                                            }))
                                            .border_1()
                                            .border_color(rgba(if is_copied || is_drag_selected {
                                                TALKIT_BLUE
                                            } else {
                                                theme.pill_border
                                            }))
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .text_size(px(11.0))
                                            .text_color(rgba(if is_copied || is_drag_selected {
                                                0xffffffff
                                            } else {
                                                theme.text_secondary
                                            }))
                                            .child(if is_copied || is_drag_selected {
                                                icon_svg("icons/check.svg", 11.0, 0xffffffff)
                                            } else {
                                                icon_svg(
                                                    "icons/copy.svg",
                                                    11.0,
                                                    theme.text_secondary,
                                                )
                                            })
                                            .when(!is_compact, |d| {
                                                d.child(if is_copied {
                                                    "복사됨"
                                                } else if is_drag_selected {
                                                    "선택됨"
                                                } else {
                                                    "복사"
                                                })
                                            }),
                                    )
                            }))
                            // Live emerging sentence cards for any channel (`시스템` / `마이크`) starting a new sentence
                            .children(pending_live_cards.into_iter().enumerate().map(
                                |(live_i, (src_opt, live_txt))| {
                                    let live_card_id = usize::MAX - 10 - live_i;
                                    let is_mic =
                                        src_opt.as_ref().map(|s| s.as_ref()) == Some("마이크");
                                    let accent_color = if is_mic { 0x10b981ff } else { TALKIT_BLUE };
                                    let badge_label = match src_opt.as_ref().map(|s| s.as_ref()) {
                                        Some("시스템") => SharedString::from("시스템 실시간"),
                                        Some("마이크") => SharedString::from("마이크 실시간"),
                                        _ => SharedString::from("실시간 전사"),
                                    };
                                    let copy_live_str = live_txt.to_string();
                                    let is_copied = self.copied_index == Some(live_card_id);
                                    div()
                                        .id(SharedString::from(format!(
                                            "transcript-card-live-{}",
                                            live_i
                                        )))
                                        .px(px(if is_compact { 12.0 } else { 16.0 }))
                                        .py_2p5()
                                        .rounded(px(18.0))
                                        .cursor_pointer()
                                        .bg(linear_gradient(
                                            180.0,
                                            linear_color_stop(rgba(theme.card_bg_top), 0.0),
                                            linear_color_stop(rgba(theme.card_bg_bottom), 1.0),
                                        ))
                                        .border_1()
                                        .border_color(rgba(accent_color))
                                        .shadow(glass_card_shadows())
                                        .flex()
                                        .items_start()
                                        .gap(px(if is_compact { 8.0 } else { 12.0 }))
                                        .on_click(cx.listener(move |this, _, _, cx| {
                                            this.copy_text(
                                                live_card_id,
                                                copy_live_str.clone(),
                                                cx,
                                            );
                                        }))
                                        .child(
                                            div()
                                                .flex_none()
                                                .mt(px(3.0))
                                                .w(px(3.5))
                                                .h(px(20.0))
                                                .rounded(px(999.0))
                                                .bg(rgba(accent_color)),
                                        )
                                        .child(
                                            div()
                                                .flex_none()
                                                .mt(px(1.0))
                                                .px_2()
                                                .py(px(2.0))
                                                .rounded(px(999.0))
                                                .bg(rgba(theme.pill_bg))
                                                .border_1()
                                                .border_color(rgba(theme.pill_border))
                                                .text_size(px(11.0))
                                                .text_color(rgba(accent_color))
                                                .child(badge_label),
                                        )
                                        .child(
                                            div()
                                                .flex_1()
                                                .min_w(px(0.0))
                                                .text_size(px(body_font))
                                                .text_color(rgba(theme.text_primary))
                                                .child(live_txt),
                                        )
                                        .child(
                                            div()
                                                .flex_none()
                                                .h(px(24.0))
                                                .px_2()
                                                .rounded(px(999.0))
                                                .bg(rgba(if is_copied {
                                                    TALKIT_BLUE
                                                } else {
                                                    theme.pill_bg
                                                }))
                                                .border_1()
                                                .border_color(rgba(if is_copied {
                                                    TALKIT_BLUE
                                                } else {
                                                    theme.pill_border
                                                }))
                                                .flex()
                                                .items_center()
                                                .gap_1()
                                                .text_size(px(11.0))
                                                .text_color(rgba(if is_copied {
                                                    0xffffffff
                                                } else {
                                                    theme.text_secondary
                                                }))
                                                .child(icon_svg(
                                                    if is_copied {
                                                        "icons/check.svg"
                                                    } else {
                                                        "icons/copy.svg"
                                                    },
                                                    11.0,
                                                    if is_copied {
                                                        0xffffffff
                                                    } else {
                                                        theme.text_secondary
                                                    },
                                                ))
                                                .when(!is_compact, |d| {
                                                    d.child(if is_copied { "복사됨" } else { "복사" })
                                                }),
                                        )
                                },
                            ))
                            .into_any_element()
                    })
                    // 4. Inset Live Streaming Subtitle Bar (Talkit Brand Pill)
                    .child({
                        let tail_caption = self.continuous_tail(92);
                        let dock_has_text = tail_caption.is_some();
                        let partial_for_copy = self.full_continuous_text();
                        let live_copied = self.copied_index == Some(usize::MAX - 1);
                        div()
                            .id("bottom-live-dock")
                            .flex_none()
                            .mx_1()
                            .px(px(if is_compact { 12.0 } else { 16.0 }))
                            .py(px(if is_short { 8.0 } else { 12.0 }))
                            .rounded(px(18.0))
                            .when(dock_has_text, |d| d.cursor_pointer())
                            .bg(if has_partial {
                                linear_gradient(
                                    180.0,
                                    linear_color_stop(rgba(0x4f6bff22), 0.0),
                                    linear_color_stop(rgba(0x4f6bff10), 1.0),
                                )
                            } else {
                                linear_gradient(
                                    180.0,
                                    linear_color_stop(rgba(theme.card_bg_top), 0.0),
                                    linear_color_stop(rgba(theme.card_bg_bottom), 1.0),
                                )
                            })
                            .border_1()
                            .border_color(rgba(if has_partial {
                                TALKIT_BLUE
                            } else {
                                theme.card_border
                            }))
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_2p5()
                            .child(
                                div()
                                    .flex()
                                    .items_start()
                                    .gap_2p5()
                                    .flex_1()
                                    .min_w(px(0.0))
                                    .child(
                                        div()
                                            .flex_none()
                                            .mt(px(1.5))
                                            .px_2p5()
                                            .py(px(2.5))
                                            .rounded(px(999.0))
                                            .bg(rgba(if self.paused {
                                                0xf59e0b33
                                            } else if has_partial {
                                                TALKIT_BLUE
                                            } else {
                                                theme.pill_bg
                                            }))
                                            .border_1()
                                            .border_color(rgba(if self.paused {
                                                0xfbbf2488
                                            } else if has_partial {
                                                TALKIT_BLUE
                                            } else {
                                                theme.pill_border
                                            }))
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .text_size(px(10.5))
                                            .text_color(rgba(if self.paused {
                                                0xd97706ff
                                            } else if has_partial {
                                                0xffffffff
                                            } else {
                                                TALKIT_BLUE
                                            }))
                                            .child(icon_svg(
                                                "icons/talkit_logo.svg",
                                                12.0,
                                                if has_partial { 0xffffffff } else { TALKIT_BLUE },
                                            ))
                                            .child(if self.paused {
                                                "PAUSED"
                                            } else if has_partial {
                                                if is_compact { "LIVE" } else { "TALKIT LIVE" }
                                            } else if is_compact {
                                                "READY"
                                            } else {
                                                "TALK IT, GET IT"
                                            }),
                                    )
                                    // Multi-line wrapping live caption (retains latest committed sentence when idle)
                                    .child(
                                        div()
                                            .flex_1()
                                            .min_w(px(0.0))
                                            .text_size(px(body_font))
                                            .text_color(rgba(if dock_has_text {
                                                theme.text_primary
                                            } else {
                                                theme.text_muted
                                            }))
                                            .child(if self.paused {
                                                "전사가 일시정지되었습니다.".into()
                                            } else if let Some(tail) = tail_caption {
                                                tail
                                            } else {
                                                "말하는 순간, 바로 텍스트로 — 실시간 스트리밍 대기 중…".into()
                                            }),
                                    ),
                            )
                            .when(dock_has_text, |d| {
                                let txt_down = partial_for_copy.clone();
                                let txt_drag = partial_for_copy.clone();
                                let txt_click = partial_for_copy;
                                d.child(
                                    div()
                                        .flex_none()
                                        .px_2()
                                        .py(px(2.0))
                                        .rounded(px(999.0))
                                        .bg(rgba(if live_copied {
                                            TALKIT_BLUE
                                        } else {
                                            theme.pill_bg
                                        }))
                                        .flex()
                                        .items_center()
                                        .gap_1()
                                        .text_size(px(11.0))
                                        .text_color(rgba(if live_copied {
                                            0xffffffff
                                        } else {
                                            TALKIT_BLUE
                                        }))
                                        .child(if live_copied {
                                            icon_svg("icons/check.svg", 11.0, 0xffffffff)
                                        } else {
                                            icon_svg("icons/copy.svg", 11.0, TALKIT_BLUE)
                                        })
                                        .when(!is_compact, |d| {
                                            d.child(if live_copied { "복사됨" } else { "복사" })
                                        }),
                                )
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _: &MouseDownEvent, _, cx| {
                                        this.copy_text(usize::MAX - 1, txt_down.clone(), cx);
                                    }),
                                )
                                .on_mouse_move(cx.listener(
                                    move |this, ev: &MouseMoveEvent, _, cx| {
                                        if ev.dragging() {
                                            this.copy_text(usize::MAX - 1, txt_drag.clone(), cx);
                                        }
                                    },
                                ))
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.copy_text(usize::MAX - 1, txt_click.clone(), cx);
                                }))
                            })
                            .child(self.render_waveform_bars(
                                18.0,
                                if is_compact { 6 } else { 12 },
                            ))
                    }),
            )
    }
}

impl Render for TranscriptView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if !self.glass_initialized {
            self.glass_initialized = true;
            #[cfg(target_os = "macos")]
            macos_glass::apply_window_glass(window, self.preset, self.pip_mode, self.pip_locked);
        }

        let vp = window.viewport_size();
        let raw_vw: f32 = vp.width.into();
        let raw_vh: f32 = vp.height.into();

        // Use `self.pip_mode` as the single source of truth so transitional viewport sizes
        // never flip the mode back or cause a layout flash.
        let (vw, vh) = if self.pip_mode {
            (raw_vw.min(520.0), raw_vh.min(180.0))
        } else if raw_vw <= 520.0 && raw_vh <= 200.0 {
            (860.0, 580.0)
        } else {
            (raw_vw, raw_vh)
        };

        let font_name = if cfg!(target_os = "macos") {
            ".AppleSystemUIFont"
        } else {
            "Malgun Gothic"
        };

        div()
            .id("root-transcript-view")
            .size_full()
            .font_family(font_name)
            .text_color(rgba(0xffffffff))
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _, cx| {
                if ev.keystroke.modifiers.platform && ev.keystroke.key == "c" {
                    if this.selected_range().is_some() {
                        this.copy_selected_range(cx);
                    } else {
                        this.copy_all(cx);
                    }
                }
            }))
            .child(if self.pip_mode {
                self.render_pip_overlay(vw, cx).into_any_element()
            } else {
                self.render_studio_mode(vw, vh, cx).into_any_element()
            })
    }
}

fn main() {
    Application::new()
        .with_assets(EmbeddedSvgAssets)
        .run(|cx: &mut App| {
            let bounds = Bounds::centered(None, size(px(860.0), px(580.0)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    window_min_size: Some(size(px(360.0), px(136.0))),
                    window_background: WindowBackgroundAppearance::Transparent,
                    titlebar: Some(TitlebarOptions {
                        title: Some("Talkit".into()),
                        appears_transparent: true,
                        traffic_light_position: Some(point(px(26.0), px(26.0))),
                    }),
                    ..Default::default()
                },
                |_, cx| cx.new(TranscriptView::new),
            )
            .expect("창 열기 실패");

            cx.on_window_closed(|cx| {
                if cx.windows().is_empty() {
                    cx.quit();
                }
            })
            .detach();
            cx.activate(true);
        });
}
