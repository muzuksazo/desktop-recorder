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

use std::{
    borrow::Cow,
    collections::VecDeque,
    env,
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::Instant,
};

use futures::{channel::mpsc, StreamExt};
use gpui::{
    div, hsla, linear_color_stop, linear_gradient, point, prelude::*, px, rgba, size, svg, App,
    Application, AssetSource, Bounds, BoxShadow, ClipboardItem, Context, IntoElement, KeyDownEvent,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Render, Result as GpuiResult,
    ScrollHandle, SharedString, TitlebarOptions, Window, WindowBackgroundAppearance, WindowBounds,
    WindowOptions,
};
use serde::Deserialize;

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
}

enum Event {
    Msg(BackendMsg),
    Exited,
}

fn resolve_python_bin() -> PathBuf {
    if let Some(p) = env::var_os("RECORDER_PYTHON") {
        return PathBuf::from(p);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let candidates = [
        root.join(".venv").join("bin").join("python"),
        root.join(".venv").join("Scripts").join("python.exe"),
        PathBuf::from("/Users/yun-yeowon/qwen3-asr-ko/.venv/bin/python"),
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

fn spawn_backend(tx: mpsc::UnboundedSender<Event>) -> std::io::Result<Child> {
    let python = resolve_python_bin();
    let script = env::var_os("RECORDER_BACKEND")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("backend.py"));

    let mut child = Command::new(&python)
        .args(["-X", "utf8"])
        .arg(&script)
        .args(env::args().skip(1))
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

// ---------------------------------------------------------------------------
// Main Mobbin-Inspired Liquid Glass & PiP View
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TranscriptItem {
    index: usize,
    timestamp: Option<SharedString>,
    body: SharedString,
    is_error: bool,
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
    // Drag-range selection state across transcript cards
    drag_anchor: Option<usize>,
    drag_current: Option<usize>,
    is_dragging_selection: bool,
    audio_level: f32,
    wave_history: VecDeque<f32>,
    mem_mb: f32,
    status: VecDeque<SharedString>,
    partial: SharedString,
    items: VecDeque<TranscriptItem>,
    scroll: ScrollHandle,
    _backend: Option<Child>,
}

impl Drop for TranscriptView {
    fn drop(&mut self) {
        if let Some(mut child) = self._backend.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl TranscriptView {
    fn new(cx: &mut Context<Self>) -> Self {
        let (tx, mut rx) = mpsc::unbounded();
        let mut status = VecDeque::with_capacity(MAX_STATUS_ENTRIES);
        status.push_back("MLX 4-bit (1.07GB) 실시간 스트리밍 준비 중…".into());

        let mut wave_history = VecDeque::with_capacity(12);
        for _ in 0..12 {
            wave_history.push_back(0.08);
        }

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
            items: VecDeque::with_capacity(MAX_TRANSCRIPT_ITEMS),
            scroll: ScrollHandle::new(),
            _backend: None,
        };
        match spawn_backend(tx) {
            Ok(child) => view._backend = Some(child),
            Err(e) => view.push_line(format!("백엔드 실행 실패: {e} (RECORDER_PYTHON 확인)"), true),
        }

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

    fn push_line(&mut self, raw: impl Into<String>, is_error: bool) {
        if self.paused && !is_error {
            return;
        }
        let s: String = raw.into();
        let (timestamp, body) = if s.starts_with('[') {
            if let Some(idx) = s.find("] ") {
                (
                    Some(SharedString::from(s[1..idx].to_string())),
                    SharedString::from(s[idx + 2..].to_string()),
                )
            } else {
                (None, SharedString::from(s))
            }
        } else {
            (None, SharedString::from(s))
        };

        self.total_committed += 1;
        if self.items.len() >= MAX_TRANSCRIPT_ITEMS {
            self.items.pop_front();
        }
        self.items.push_back(TranscriptItem {
            index: self.total_committed,
            timestamp,
            body,
            is_error,
        });
        self.scroll.scroll_to_bottom();
    }

    fn apply(&mut self, event: Event) {
        match event {
            Event::Msg(BackendMsg::Status { msg }) => self.push_status(msg),
            Event::Msg(BackendMsg::Partial { text }) => {
                if !self.paused {
                    self.partial = text.into();
                    self.scroll.scroll_to_bottom();
                }
            }
            Event::Msg(BackendMsg::Text {
                text,
                clock,
                ..
            }) => {
                self.partial = "".into();
                let formatted = if let Some(c) = clock {
                    if text.starts_with('[') {
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
            Event::Msg(BackendMsg::Telemetry { rms, mem_mb }) => {
                if !self.paused {
                    self.push_wave(rms);
                }
                self.mem_mb = mem_mb;
            }
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
            self.audio_level = 0.0;
        }
        cx.notify();
    }

    fn clear_history(&mut self, cx: &mut Context<Self>) {
        self.items.clear();
        self.partial = "".into();
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
        let mut lines: Vec<String> = self
            .items
            .iter()
            .filter(|it| !it.is_error)
            .map(|it| match &it.timestamp {
                Some(ts) => format!("[{}] {}", ts, it.body),
                None => it.body.to_string(),
            })
            .collect();
        if !self.partial.is_empty() {
            lines.push(self.partial.to_string());
        }
        if lines.is_empty() {
            return;
        }
        cx.write_to_clipboard(ClipboardItem::new_string(lines.join("\n")));
        self.copied_index = Some(0);
        cx.notify();
    }

    // -----------------------------------------------------------------------
    // Drag-to-Select & Copy Range Logic
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
        let is_active = !self.partial.is_empty() || self.audio_level > 0.08;
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
        let has_partial = !self.partial.is_empty();
        let elapsed = self.format_elapsed();
        let mem_badge = SharedString::from(format!("{:.2} GB", self.mem_mb / 1024.0));

        let latest_committed: Option<SharedString> =
            self.items.back().map(|it| it.body.clone());

        let copyable_pip: Option<String> = if has_partial {
            Some(self.partial.to_string())
        } else {
            latest_committed.as_ref().map(|s| s.to_string())
        };
        let pip_copied = self.copied_index == Some(usize::MAX);

        let primary_caption: SharedString = if self.paused {
            "일시정지됨 — 재개 버튼을 누르면 다시 실시간 전사를 시작합니다".into()
        } else if has_partial {
            self.partial.clone()
        } else if let Some(ref last) = latest_committed {
            last.clone()
        } else {
            "Talk it, Get it. 말하는 순간, 바로 텍스트로.".into()
        };

        let secondary_caption: Option<SharedString> = if has_partial {
            latest_committed
        } else if self.items.len() >= 2 {
            self.items
                .get(self.items.len() - 2)
                .map(|it| it.body.clone())
        } else {
            None
        };

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
    fn render_studio_mode(&self, vw: f32, vh: f32, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = self.preset.theme();
        let is_compact = vw < 680.0;
        let is_medium = vw < 860.0;
        let is_short = vh < 440.0;

        let has_partial = !self.partial.is_empty();
        let partial_text: SharedString = self.partial.clone();
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
                    .child(
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
                            .when(self.items.is_empty(), |d| {
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
                                                .child("문장 카드를 클릭하거나 마우스로 드래그하면 창 이동 없이 즉시 복사됩니다."),
                                        ),
                                )
                            })
                            .children(self.items.iter().map(|item| {
                                let is_err = item.is_error;
                                let item_idx = item.index;
                                let is_copied = self.copied_index == Some(item_idx);
                                let is_drag_selected = self.is_in_drag_selection(item_idx);
                                let idx_label = SharedString::from(format!("#{:02}", item_idx));
                                let ts = item.timestamp.clone();
                                let body = item.body.clone();
                                let copy_str = item.body.to_string();

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
                                    // Drag-to-select handlers
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
                                    // Left vertical Talkit Blue accent pill
                                    .child(
                                        div()
                                            .flex_none()
                                            .mt(px(3.0))
                                            .w(px(3.5))
                                            .h(px(20.0))
                                            .rounded(px(999.0))
                                            .bg(rgba(if is_err {
                                                0xfb7185ff
                                            } else {
                                                TALKIT_BLUE
                                            })),
                                    )
                                    // Sequence + Timestamp pill in Talkit Gray (#E9EBF7)
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
                                                    .text_color(rgba(TALKIT_BLUE))
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
                                    // Multi-line wrapping transcript utterance body
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
                                    // Copy action pill with SVG icon
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
                            })),
                    )
                    // 4. Inset Live Streaming Subtitle Bar (Talkit Brand Pill)
                    .child({
                        let partial_for_copy = self.partial.to_string();
                        let live_copied = self.copied_index == Some(usize::MAX - 1);
                        div()
                            .id("bottom-live-dock")
                            .flex_none()
                            .mx_1()
                            .px(px(if is_compact { 12.0 } else { 16.0 }))
                            .py(px(if is_short { 8.0 } else { 12.0 }))
                            .rounded(px(18.0))
                            .when(has_partial, |d| d.cursor_pointer())
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
                                    // Multi-line wrapping live caption
                                    .child(
                                        div()
                                            .flex_1()
                                            .min_w(px(0.0))
                                            .text_size(px(body_font))
                                            .text_color(rgba(if has_partial {
                                                theme.text_primary
                                            } else {
                                                theme.text_muted
                                            }))
                                            .child(if self.paused {
                                                "전사가 일시정지되었습니다.".into()
                                            } else if has_partial {
                                                partial_text
                                            } else {
                                                "말하는 순간, 바로 텍스트로 — 실시간 스트리밍 대기 중…".into()
                                            }),
                                    ),
                            )
                            .when(has_partial, |d| {
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
