import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

final class SystemAudioTap: NSObject, SCStreamOutput, SCStreamDelegate {
    private let stdoutHandle = FileHandle.standardOutput
    private let lock = NSLock()

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        guard outputType == .audio else { return }
        guard CMSampleBufferDataIsReady(sampleBuffer) else { return }

        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else {
            return
        }
        let asbd = asbdPtr.pointee
        let numSamples = CMSampleBufferGetNumSamples(sampleBuffer)
        if numSamples <= 0 { return }

        var blockBuffer: CMBlockBuffer?
        var audioBufferList = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(mNumberChannels: 0, mDataByteSize: 0, mData: nil)
        )

        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: &audioBufferList,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &blockBuffer
        )

        if status == noErr {
            let buf = audioBufferList.mBuffers
            guard let dataPtr = buf.mData, buf.mDataByteSize > 0 else { return }

            if asbd.mFormatID == kAudioFormatLinearPCM && (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0 && asbd.mBitsPerChannel == 32 {
                let floatCount = Int(buf.mDataByteSize) / MemoryLayout<Float>.size
                let channels = max(1, Int(buf.mNumberChannels))
                if channels == 1 {
                    let data = Data(bytes: dataPtr, count: floatCount * MemoryLayout<Float>.size)
                    lock.lock()
                    try? stdoutHandle.write(contentsOf: data)
                    lock.unlock()
                } else {
                    let floats = dataPtr.bindMemory(to: Float.self, capacity: floatCount)
                    let frames = floatCount / channels
                    var mono = [Float](repeating: 0, count: frames)
                    for i in 0..<frames {
                        var sum: Float = 0
                        for c in 0..<channels {
                            sum += floats[i * channels + c]
                        }
                        mono[i] = sum / Float(channels)
                    }
                    mono.withUnsafeBytes { rawBuf in
                        if let base = rawBuf.baseAddress {
                            let data = Data(bytes: base, count: rawBuf.count)
                            lock.lock()
                            try? stdoutHandle.write(contentsOf: data)
                            lock.unlock()
                        }
                    }
                }
            }
        } else if let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) {
            let length = CMBlockBufferGetDataLength(dataBuffer)
            guard length > 0 else { return }
            var data = Data(count: length)
            let copyStatus = data.withUnsafeMutableBytes { rawBuf -> OSStatus in
                guard let base = rawBuf.baseAddress else { return -1 }
                return CMBlockBufferCopyDataBytes(dataBuffer, atOffset: 0, dataLength: length, destination: base)
            }
            if copyStatus == kCMBlockBufferNoErr {
                lock.lock()
                try? stdoutHandle.write(contentsOf: data)
                lock.unlock()
            }
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("[macos_audio_tap] stream stopped: \(error.localizedDescription)\n", stderr)
        exit(1)
    }
}

@main
struct Main {
    static func main() async {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            guard let display = content.displays.first else {
                fputs("[macos_audio_tap] No display found\n", stderr)
                exit(1)
            }

            let filter = SCContentFilter(display: display, excludingWindows: [])
            let config = SCStreamConfiguration()
            config.capturesAudio = true
            config.excludesCurrentProcessAudio = false
            config.sampleRate = 16000
            config.channelCount = 1
            // Minimize video overhead
            config.width = 2
            config.height = 2
            config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
            config.showsCursor = false

            let tap = SystemAudioTap()
            let stream = SCStream(filter: filter, configuration: config, delegate: tap)
            let queue = DispatchQueue(label: "ai.talkit.system-audio-tap", qos: .userInteractive)
            try stream.addStreamOutput(tap, type: .audio, sampleHandlerQueue: queue)
            try await stream.startCapture()
            fputs("[macos_audio_tap] READY (ScreenCaptureKit 16kHz mono float32)\n", stderr)

            // Keep alive until parent closes or terminates
            while true {
                try await Task.sleep(nanoseconds: 1_000_000_000)
            }
        } catch {
            fputs("[macos_audio_tap] ERROR: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
