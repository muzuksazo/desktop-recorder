#!/usr/bin/env bash
# Talkit.app 과 DMG 를 만든다. 어느 Apple Silicon Mac 에서나 동작하도록:
#   - Python 을 앱 안에 넣는다 (uv 가 관리하는 이식 가능한 CPython + 고정된 패키지)
#   - 모델은 넣지 않는다. 첫 실행 때 ~/Library/Application Support/Talkit/models 로 내려받는다(약 1.1GB)
#   - 사용자 이름이 들어간 경로를 쓰지 않는다
#
# 사용:  packaging/package_macos.sh                 (버전 1.2.0)
#        VERSION=1.2.1 BUILD=7 packaging/package_macos.sh
#        SIGN_ID="-" packaging/package_macos.sh      (서명 인증서 없이 ad-hoc)
#
# 비공개 Rung Ω 엔진은 여기서 넣지 않는다. 그 엔진이 있는 Mac 에서는 애드온 폴더
# (~/Library/Application Support/Talkit/engines/rung-omega 또는 ~/qwen3-asr-ko/private_rung_omega_mac)
# 의 python-path.txt 가 가리키는 Python 으로 앱이 백엔드를 띄운다(app/src/main.rs).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${VERSION:-1.2.0}"
BUILD="${BUILD:-6}"
PY_VER="${PY_VER:-3.11}"
BUNDLE_ID="com.talkit.app"
STAGE="$ROOT/dist/stage-$VERSION"
APP="$STAGE/Talkit.app"
C="$APP/Contents"
DMG="$ROOT/Talkit-$VERSION.dmg"

# 서명: 지정이 없으면 이 Mac 의 첫 'Apple Development' 인증서, 없으면 ad-hoc.
# 같은 인증서로 계속 서명해야 macOS 가 녹음 권한을 앱을 다시 만들어도 기억한다.
if [ -z "${SIGN_ID:-}" ]; then
  SIGN_ID="$(security find-identity -v -p codesigning 2>/dev/null | awk -F'"' '/Apple Development/{print $2; exit}')"
  SIGN_ID="${SIGN_ID:--}"
fi
echo "▶ Talkit $VERSION ($BUILD) · 서명: $SIGN_ID"

rm -rf "$STAGE"
mkdir -p "$C/MacOS" "$C/Resources"

echo "▶ 앱 빌드"
(cd "$ROOT/app" && cargo build --release)
cp "$ROOT/app/target/release/recorder-app" "$C/MacOS/Talkit"

echo "▶ 시스템 오디오 도우미 빌드"
swiftc -O -parse-as-library "$ROOT/macos_audio_tap.swift" -o "$C/Resources/macos_audio_tap"

cp "$ROOT/backend.py" "$ROOT/live_transcribe.py" "$ROOT/mlx_quantized_asr.py" "$C/Resources/"
cp "$ROOT/packaging/AppIcon.icns" "$C/Resources/AppIcon.icns"

echo "▶ Python $PY_VER 넣기"
uv python install "$PY_VER" >/dev/null
# 꼭 uv 가 관리하는 독립 CPython 을 쓴다. 그냥 'uv python find' 는 PATH 의 anaconda 등을 먼저 골라
# 이 Mac 의 경로를 가리키는 링크가 앱에 들어간다(2026-11-06 실제로 발생).
PY_EXE="$(uv python find --managed-python "$PY_VER")"
PY_HOME="$(python3 -c 'import os,sys; print(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[1]))))' "$PY_EXE")"
echo "  Python 원본: $PY_HOME"
cp -R "$PY_HOME" "$C/Resources/python"
# 앱 밖을 가리키는 링크가 하나라도 있으면 중단 (다른 Mac 에서 깨진다)
if find "$C/Resources/python" -type l -exec readlink {} \; | grep -q '^/'; then
  echo "✗ 앱 밖을 가리키는 링크가 있어요:" >&2
  find "$C/Resources/python" -type l -exec sh -c 'readlink "$1" | grep -q "^/" && echo "  $1 -> $(readlink "$1")"' _ {} \; >&2
  exit 1
fi
BUNDLED_PY="$C/Resources/python/bin/python3"
uv pip install --quiet --python "$BUNDLED_PY" --break-system-packages \
  -r "$ROOT/packaging/requirements-macos.txt"

echo "▶ 필요 없는 파일 정리"
LIB="$C/Resources/python/lib/python$PY_VER"
rm -rf "$LIB/test" "$LIB/idlelib" "$LIB/tkinter" "$LIB/turtledemo" "$LIB/ensurepip" \
       "$C/Resources/python/lib/"tcl* "$C/Resources/python/lib/"tk* "$C/Resources/python/lib/itcl"* \
       "$C/Resources/python/share" "$C/Resources/python/include"
find "$C/Resources/python" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$C/Resources/python" -type d -name "tests" -path "*site-packages*" -prune -exec rm -rf {} +

echo "▶ 앱이 쓰는 모듈이 실제로 불러와지는지 확인"
# -E -s: 이 Mac 의 환경 변수·사용자 패키지를 무시하고 앱 안의 것만으로 불러온다
"$BUNDLED_PY" -E -s -c "import sys; sys.path.insert(0, '$C/Resources'); \
import mlx.core, numpy, soundcard, soundfile, huggingface_hub, qwen3_asr_mlx, mlx_quantized_asr, live_transcribe; \
print('  imports OK · huggingface_hub', huggingface_hub.__version__)"

cat > "$C/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Talkit</string>
  <key>CFBundleDisplayName</key><string>Talkit</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key><string>Talkit</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$BUILD</string>
  <key>CFBundleDevelopmentRegion</key><string>ko</string>
  <key>LSMinimumSystemVersion</key><string>14.4</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>Talkit이 실시간 한국어 음성 인식을 위해 마이크에 접근합니다.</string>
  <key>NSAudioCaptureUsageDescription</key><string>Talkit이 실시간 시스템 소리 전사를 위해 오디오 캡처에 접근합니다.</string>
</dict>
</plist>
PLIST

echo "▶ 서명"
# 앱 안의 모든 실행 파일·라이브러리를 먼저 서명하고, 마지막에 앱 전체를 서명한다
find "$C/Resources" -type f \( -name "*.so" -o -name "*.dylib" -o -perm -u+x \) -print0 |
  while IFS= read -r -d '' f; do
    if file -b "$f" | grep -q "Mach-O"; then
      codesign --force --timestamp=none -s "$SIGN_ID" "$f" >/dev/null 2>&1
    fi
  done
codesign --force --timestamp=none -s "$SIGN_ID" --entitlements "$ROOT/packaging/Entitlements.plist" "$APP"
codesign --verify --deep --strict "$APP"
echo "  서명 확인 OK"

echo "▶ DMG"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "Talkit" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
hdiutil verify "$DMG" >/dev/null
# 스테이지의 앱이 LaunchServices 에 '또 하나의 Talkit' 으로 남지 않게 지운다(권한 기록 꼬임 방지)
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -u "$APP" 2>/dev/null || true
rm -rf "$STAGE"

echo "✅ $DMG ($(du -h "$DMG" | cut -f1))"
