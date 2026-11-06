# Talkit.exe (Windows) 를 만든다.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#
# - exe 안에는 화면(windows_app.py, talkit_win.py)과 엔진 스크립트(backend.py 등, payload\)만 넣는다 (약 15MB).
# - torch·모델은 넣지 않는다. 처음 실행할 때 앱의 '설정 → 엔진 설치 / 모델 받기'로 %LOCALAPPDATA%\Talkit 에 받는다.
# - 버전은 talkit_win.py 의 APP_VERSION. 릴리스 태그는 win-v<버전>, 첨부 파일 이름은 Talkit.exe (자동 업데이트가 이걸 찾음).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root "build\windows"
$Venv = Join-Path $Build "venv"
New-Item -ItemType Directory -Force $Build | Out-Null

$Version = (Select-String -Path (Join-Path $Root "talkit_win.py") -Pattern '^APP_VERSION = "(.+)"').Matches[0].Groups[1].Value
Write-Host "▶ Talkit for Windows $Version"

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Write-Host "▶ 빌드용 가상환경 만들기"
    python -m venv $Venv
}
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --quiet --disable-pip-version-check pyinstaller==6.22.3 pillow==12.3.0
if ($LASTEXITCODE -ne 0) { throw "pip 실패" }

Write-Host "▶ 아이콘 (AppIcon.icns → ico)"
$Ico = Join-Path $Build "Talkit.ico"
& $Py -c "from PIL import Image; im = Image.open(r'$Root\packaging\AppIcon.icns'); im.save(r'$Ico', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])"
if ($LASTEXITCODE -ne 0) { throw "아이콘 변환 실패" }

Write-Host "▶ PyInstaller"
$Data = @("backend.py", "live_transcribe.py", "mlx_quantized_asr.py", "model_tool.py") | ForEach-Object {
    "--add-data"; "$Root\$_;payload"
}
& $Py -m PyInstaller --noconfirm --clean --onefile --windowed --name Talkit --icon $Ico `
    --distpath (Join-Path $Root "dist") --workpath (Join-Path $Build "work") --specpath $Build `
    --add-data "$Root\packaging\requirements-windows.txt;." @Data `
    --exclude-module numpy --exclude-module torch `
    (Join-Path $Root "windows_app.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 실패" }

$Exe = Join-Path $Root "dist\Talkit.exe"
$Hash = (Get-FileHash $Exe -Algorithm SHA256).Hash.ToLower()
$MB = [math]::Round((Get-Item $Exe).Length / 1MB, 1)
Write-Host "✅ $Exe ($MB MB)"
Write-Host "   SHA-256: $Hash"
Write-Host "   릴리스: gh release create win-v$Version dist\Talkit.exe --title `"Talkit for Windows $Version`""
