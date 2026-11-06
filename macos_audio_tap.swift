// Talkit 시스템 오디오 도우미 — Core Audio Process Tap (macOS 14.2+)
//
// 왜 ScreenCaptureKit 이 아닌가:
//   ScreenCaptureKit 은 '화면 및 시스템 오디오 녹음' 권한이 필요하고, 도우미 프로그램이 요청하면
//   macOS 가 허용 창 없이 거절한다(does not allow prompting). 권한을 켜도 앱을 강제로 다시 열어야 한다.
//   Process Tap 은 '시스템 오디오 녹음만' 권한을 쓰고, 처음 한 번 일반 '허용' 창을 띄우며 바로 동작한다.
//
// 출력: stdout 으로 16kHz mono float32 (little-endian) 를 계속 흘린다. 로그는 stderr.

import AudioToolbox
import CoreAudio
import Foundation

let TARGET_SR: Double = 16000

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write("[macos_audio_tap] ERROR: \(msg)\n".data(using: .utf8)!)
    exit(1)
}

func log(_ msg: String) {
    FileHandle.standardError.write("[macos_audio_tap] \(msg)\n".data(using: .utf8)!)
}

/// 입력 샘플레이트 → 16kHz 선형 보간 리샘플러 (블록 사이 연속성 유지)
final class Resampler {
    let step: Double
    var pos: Double = 0
    var carry: Float = 0
    var hasCarry = false

    init(inputRate: Double) { step = inputRate / TARGET_SR }

    func process(_ input: [Float]) -> [Float] {
        if input.isEmpty { return [] }
        var src = input
        if hasCarry { src.insert(carry, at: 0) } else { pos = 0 }
        var out: [Float] = []
        out.reserveCapacity(Int(Double(src.count) / step) + 1)
        while pos + 1 < Double(src.count) {
            let i = Int(pos)
            let f = Float(pos - Double(i))
            out.append(src[i] * (1 - f) + src[i + 1] * f)
            pos += step
        }
        pos -= Double(src.count - 1)
        carry = src[src.count - 1]
        hasCarry = true
        return out
    }
}

@main
struct Main {
    static func main() {
        // 1. 시스템 전체 출력(모든 앱 소리)을 듣는 탭. 소리는 그대로 스피커로 나간다.
        let tapDesc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        tapDesc.uuid = UUID()
        tapDesc.muteBehavior = .unmuted
        tapDesc.isPrivate = true
        tapDesc.name = "Talkit System Audio"

        var tapID = AudioObjectID(kAudioObjectUnknown)
        var err = AudioHardwareCreateProcessTap(tapDesc, &tapID)
        if err != noErr { fail("시스템 오디오 탭을 만들 수 없습니다 (OSStatus \(err))") }

        // 2. 탭 형식(샘플레이트·채널) 읽기
        var asbd = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        err = AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &asbd)
        if err != noErr { fail("탭 형식을 읽을 수 없습니다 (OSStatus \(err))") }
        let inRate = asbd.mSampleRate > 0 ? asbd.mSampleRate : 48000
        let isNonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0

        // 3. 탭을 담는 비공개 집합 장치를 만들어 IOProc 으로 받는다
        let aggDesc: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Talkit Tap",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapDesc.uuid.uuidString,
                    kAudioSubTapDriftCompensationKey: true,
                ]
            ],
        ]
        var aggID = AudioObjectID(kAudioObjectUnknown)
        err = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID)
        if err != noErr {
            AudioHardwareDestroyProcessTap(tapID)
            fail("집합 장치를 만들 수 없습니다 (OSStatus \(err))")
        }

        let out = FileHandle.standardOutput
        let resampler = Resampler(inputRate: inRate)
        let queue = DispatchQueue(label: "ai.talkit.process-tap", qos: .userInteractive)
        var procID: AudioDeviceIOProcID?

        err = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, queue) { _, inInputData, _, _, _ in
            let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
            var mono: [Float] = []
            if isNonInterleaved {
                let chans = abl.count
                guard chans > 0, let first = abl.first else { return }
                let frames = Int(first.mDataByteSize) / MemoryLayout<Float>.size
                mono = [Float](repeating: 0, count: frames)
                for b in abl {
                    guard let p = b.mData else { continue }
                    let fp = p.assumingMemoryBound(to: Float.self)
                    for i in 0..<frames { mono[i] += fp[i] }
                }
                let inv = 1 / Float(chans)
                for i in 0..<frames { mono[i] *= inv }
            } else {
                guard let b = abl.first, let p = b.mData else { return }
                let ch = max(1, Int(b.mNumberChannels))
                let frames = Int(b.mDataByteSize) / MemoryLayout<Float>.size / ch
                let fp = p.assumingMemoryBound(to: Float.self)
                mono = [Float](repeating: 0, count: frames)
                for i in 0..<frames {
                    var s: Float = 0
                    for c in 0..<ch { s += fp[i * ch + c] }
                    mono[i] = s / Float(ch)
                }
            }
            let down = resampler.process(mono)
            if down.isEmpty { return }
            down.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                do {
                    try out.write(contentsOf: Data(bytes: base, count: raw.count))
                } catch {
                    exit(0)  // 부모(백엔드)가 사라짐
                }
            }
        }
        if err != noErr || procID == nil { fail("오디오 콜백을 만들 수 없습니다 (OSStatus \(err))") }

        err = AudioDeviceStart(aggID, procID)
        if err != noErr { fail("시스템 오디오 탭을 시작할 수 없습니다 (OSStatus \(err))") }
        log("READY (Core Audio Process Tap, \(Int(inRate))Hz -> 16kHz mono float32)")

        signal(SIGTERM) { _ in exit(0) }
        signal(SIGPIPE) { _ in exit(0) }
        dispatchMain()
    }
}
