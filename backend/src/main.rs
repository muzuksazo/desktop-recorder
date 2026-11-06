//! 전사 백엔드 (WSL2 Linux에서 실행)
//!
//! Windows의 Python GUI(`gui_client.py`)가 TCP로 오디오를 보내면,
//! CHUNK 초씩 모아서 Python 전사 워커(`asr_worker.py`, GPU)에 넘기고 결과를 돌려준다.
//!
//! GUI → 백엔드: [u32 LE 바이트 수][f32 LE 샘플...]  (16kHz mono, 길이 자유)
//! 백엔드 → GUI: JSON 한 줄씩  {"type":"text"|"status"|"error", ...}
//! 백엔드 → 워커: [u32 LE 샘플 수][f32 LE 샘플...]
//! 워커 → 백엔드: JSON 한 줄  {"text":..., "secs":...} 또는 {"error":...}

use std::{io::ErrorKind, path::PathBuf, process::Stdio, sync::Arc, time::Instant};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{tcp::OwnedWriteHalf, TcpListener, TcpStream},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{mpsc, Mutex},
};

const SAMPLE_RATE: usize = 16_000;
/// 한 프레임 최대 크기 (오디오 약 4분). 잘못된 길이로 메모리를 다 쓰는 것을 막는다.
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Parser, Debug)]
#[command(about = "Qwen3-ASR 전사 백엔드 (WSL2)")]
struct Config {
    /// 들을 주소. Windows에서는 localhost:7700 으로 접속된다 (WSL2 localhost 전달).
    #[arg(long, default_value = "127.0.0.1:7700")]
    listen: String,
    /// 워커를 실행할 Python (torch + qwen-asr 가 설치된 것)
    #[arg(long, default_value = "python3")]
    python: String,
    /// 워커 스크립트 경로
    #[arg(long, default_value = "asr_worker.py")]
    worker: PathBuf,
    /// 모델 폴더 경로 또는 Hugging Face 모델 ID
    #[arg(long)]
    model: String,
    /// 워커에 넘길 장치 (cuda:0 / cpu)
    #[arg(long, default_value = "cuda:0")]
    device: String,
    /// 몇 초 단위로 전사할지
    #[arg(long, default_value_t = 4.0)]
    chunk: f32,
    /// 이 RMS 보다 조용한 구간은 건너뛴다
    #[arg(long, default_value_t = 0.005)]
    silence: f32,
}

// ---------------------------------------------------------------- 워커
#[derive(Deserialize)]
struct WorkerReply {
    text: Option<String>,
    error: Option<String>,
}

struct Worker {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    async fn spawn(cfg: &Config) -> Result<Self> {
        let mut child = Command::new(&cfg.python)
            .arg(&cfg.worker)
            .args(["--model", &cfg.model, "--device", &cfg.device])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // 워커 로그는 그대로 터미널에
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("워커 실행 실패: {} {}", cfg.python, cfg.worker.display()))?;
        let stdin = child.stdin.take().context("워커 stdin 없음")?;
        let mut stdout = BufReader::new(child.stdout.take().context("워커 stdout 없음")?);

        // 모델 로딩이 끝나면 워커가 {"ready": true, ...} 한 줄을 보낸다
        let mut line = String::new();
        if stdout.read_line(&mut line).await? == 0 {
            bail!("워커가 준비 전에 종료됨 (위 로그 확인)");
        }
        let ready: Value = serde_json::from_str(&line).context("워커 준비 메시지 형식 오류")?;
        eprintln!("[backend] 워커 준비 완료: {ready}");
        Ok(Self { _child: child, stdin, stdout })
    }

    async fn transcribe(&mut self, audio: &[f32]) -> Result<WorkerReply> {
        let mut buf = Vec::with_capacity(4 + audio.len() * 4);
        buf.extend_from_slice(&(audio.len() as u32).to_le_bytes());
        for s in audio {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        self.stdin.write_all(&buf).await?;
        self.stdin.flush().await?;

        let mut line = String::new();
        if self.stdout.read_line(&mut line).await? == 0 {
            bail!("워커가 종료됨");
        }
        Ok(serde_json::from_str(&line)?)
    }
}

// ---------------------------------------------------------------- 전사 한 덩어리
async fn process_chunk(mut audio: Vec<f32>, worker: &Mutex<Worker>, cfg: &Config) -> Option<Value> {
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    if rms < cfg.silence {
        return None;
    }
    // 너무 작은 소리는 키운다 (클리핑 방지)
    let peak = audio.iter().fold(0f32, |m, s| m.max(s.abs()));
    if peak > 0.0 && peak < 0.5 {
        let gain = 0.5 / peak;
        audio.iter_mut().for_each(|s| *s *= gain);
    }

    let started = Instant::now();
    let reply = worker.lock().await.transcribe(&audio).await;
    match reply {
        Err(e) => Some(json!({ "type": "error", "msg": format!("워커 오류: {e:#}") })),
        Ok(WorkerReply { error: Some(e), .. }) => Some(json!({ "type": "error", "msg": e })),
        Ok(WorkerReply { text, .. }) => {
            let text = text.unwrap_or_default().trim().to_string();
            (!text.is_empty()).then(|| {
                json!({
                    "type": "text",
                    "text": text,
                    "audio_sec": audio.len() as f64 / SAMPLE_RATE as f64,
                    "infer_sec": started.elapsed().as_secs_f64(),
                })
            })
        }
    }
}

// ---------------------------------------------------------------- 접속 하나
async fn send(wr: &Mutex<OwnedWriteHalf>, msg: Value) -> std::io::Result<()> {
    let mut line = msg.to_string();
    line.push('\n');
    wr.lock().await.write_all(line.as_bytes()).await
}

async fn handle(stream: TcpStream, worker: Arc<Mutex<Worker>>, cfg: Arc<Config>) -> Result<()> {
    let (mut rd, wr) = stream.into_split();
    let wr = Arc::new(Mutex::new(wr));
    send(&wr, json!({ "type": "status", "msg": format!("백엔드 연결됨 · {}초 단위 전사", cfg.chunk) })).await?;

    // 받기와 전사를 나눈다: 전사가 밀려도 소켓은 계속 읽는다
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<f32>>();
    let processor = {
        let (wr, cfg) = (wr.clone(), cfg.clone());
        tokio::spawn(async move {
            while let Some(chunk) = rx.recv().await {
                if let Some(msg) = process_chunk(chunk, &worker, &cfg).await {
                    if send(&wr, msg).await.is_err() {
                        break;
                    }
                }
            }
        })
    };

    let chunk_len = (cfg.chunk * SAMPLE_RATE as f32) as usize;
    let mut pending: Vec<f32> = Vec::with_capacity(chunk_len * 2);
    loop {
        let mut len = [0u8; 4];
        match rd.read_exact(&mut len).await {
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break, // 정상 종료
            Err(e) => return Err(e.into()),
        }
        let n = u32::from_le_bytes(len) as usize;
        if n % 4 != 0 || n > MAX_FRAME_BYTES {
            bail!("잘못된 프레임 길이: {n}");
        }
        let mut bytes = vec![0u8; n];
        rd.read_exact(&mut bytes).await?;
        pending.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));

        while pending.len() >= chunk_len {
            let rest = pending.split_off(chunk_len);
            let chunk = std::mem::replace(&mut pending, rest);
            if tx.send(chunk).is_err() {
                return Ok(()); // 전사 쪽이 끝남 (GUI 연결 끊김)
            }
        }
    }
    drop(tx);
    let _ = processor.await;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Arc::new(Config::parse());
    if cfg.chunk <= 0.0 {
        bail!("--chunk 는 0보다 커야 합니다");
    }
    eprintln!("[backend] 모델 로딩 중: {}", cfg.model);
    let worker = Arc::new(Mutex::new(Worker::spawn(&cfg).await?));

    let listener = TcpListener::bind(&cfg.listen)
        .await
        .with_context(|| format!("{} 에서 대기 실패", cfg.listen))?;
    eprintln!("[backend] {} 에서 GUI 접속 대기 중", cfg.listen);

    loop {
        let (stream, peer) = listener.accept().await?;
        stream.set_nodelay(true)?;
        let (worker, cfg) = (worker.clone(), cfg.clone());
        tokio::spawn(async move {
            eprintln!("[backend] 접속: {peer}");
            if let Err(e) = handle(stream, worker, cfg).await {
                eprintln!("[backend] {peer} 오류: {e:#}");
            }
            eprintln!("[backend] 접속 종료: {peer}");
        });
    }
}
