# coding=utf-8
"""
Talkit Rung Ω 추론 엔진 (Apple Silicon MLX 전용, torch 불필요).

공개된 Rung Ω 가중치(MLX 양자화 형식)를 불러와 실시간 한국어 전사를 한다.
가중치를 만드는 양자화 과정은 이 파일에 없다 — 여기는 불러오기와 추론만 있다.
  - 합쳐진 QKV / Gate-Up 레이어로 토큰당 Metal 호출 수를 줄임
  - 한국어·ASCII 토큰 25,696개만 읽는 출력층 (전체 151,936개 대신)
  - (선택) Forced Aligner 로 단어 시각을 맞춰 문장 끝 잘림·유령 단어를 줄임
"""
from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from qwen3_asr_mlx.audio import log_mel_spectrogram
from qwen3_asr_mlx.config import ModelConfig, TextDecoderConfig
from qwen3_asr_mlx.decoder import KVCache, create_additive_causal_mask
from qwen3_asr_mlx.encoder import AudioEncoder, load_encoder_weights
from qwen3_asr_mlx.tokenizer import AUDIO_PAD_TOKEN_ID, EOS_TOKEN_IDS, Tokenizer, parse_language_and_output

SAMPLE_RATE = 16000
_TK_MAGIC = b"TKOMEGA1"
_TK_STRIDE = 1048576
_TK_BLOCK = 4096
_CACHED_TK_KEY: Optional[bytes] = None


def _get_tkomega_key() -> bytes:
    global _CACHED_TK_KEY
    if _CACHED_TK_KEY is not None:
        return _CACHED_TK_KEY
    raw_hex = os.environ.pop("TALKIT_MODEL_KEY", "").strip()
    if not raw_hex:
        for p in (
            Path.home() / ".config" / "talkit" / "r2.env",
            Path.home() / "Library" / "Application Support" / "Talkit" / "r2.env",
        ):
            if p.is_file():
                try:
                    for line in p.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line.startswith("TALKIT_MODEL_KEY="):
                            raw_hex = line.split("=", 1)[1].strip()
                            break
                except OSError:
                    pass
            if raw_hex:
                break
    if len(raw_hex) != 64:
        raise RuntimeError(
            "이 모델은 Talkit 전용 암호화(TKOMEGA1) 포맷으로 보호되어 있습니다. 정식 Talkit 앱에서 실행해 주세요."
        )
    _CACHED_TK_KEY = bytes.fromhex(raw_hex)
    return _CACHED_TK_KEY


def _load_safetensors_auto(path: str | Path) -> Dict[str, mx.array]:
    """평문 safetensors 와 Talkit 전용 암호화(TKOMEGA1) 포맷을 모두 메모리에서 불러온다."""
    p = Path(path)
    with open(p, "rb") as f:
        head8 = f.read(8)
        if head8 != _TK_MAGIC:
            return mx.load(str(p))
        f.seek(0)
        buf = bytearray(f.read())

    key = _get_tkomega_key()
    mv = memoryview(buf)[8:]
    ks8 = hashlib.pbkdf2_hmac("sha256", key, b"tkomega1-hdr", 1, dklen=8)
    hdr_len = struct.unpack("<Q", bytes(a ^ b for a, b in zip(mv[:8], ks8)))[0]
    hdr_total = 8 + hdr_len
    if hdr_total > len(mv):
        raise RuntimeError("TKOMEGA1 모델 복호화 키가 올바르지 않거나 파일이 손상되었습니다.")

    ks_hdr = hashlib.pbkdf2_hmac("sha256", key, b"tkomega1-hdr", 1, dklen=hdr_total)
    hdr_arr = np.frombuffer(mv[:hdr_total], dtype=np.uint8)
    hdr_arr ^= np.frombuffer(ks_hdr, dtype=np.uint8)

    payload_mv = mv[hdr_total:]
    n_chunks = len(payload_mv) // _TK_STRIDE
    if n_chunks > 0:
        ks_body = hashlib.pbkdf2_hmac(
            "sha256", key, b"tkomega1-body", 1, dklen=n_chunks * _TK_BLOCK
        )
        arr = np.frombuffer(payload_mv, dtype=np.uint8)[: n_chunks * _TK_STRIDE].reshape(
            n_chunks, _TK_STRIDE
        )
        ks = np.frombuffer(ks_body, dtype=np.uint8).reshape(n_chunks, _TK_BLOCK)
        arr[:, :_TK_BLOCK] ^= ks

    return mx.load(io.BytesIO(mv), format="safetensors")


def _bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def build_lossless_ko_ascii_active_vocab(snap_dir: Path) -> np.ndarray:
    """
    Constructs the exact ~25,696-token subset of Qwen3's 151,936 vocabulary covering:
      - 100.0% of all 11,172 Korean Hangul syllables (spaced & unspaced) + Hangul Jamo
      - All partial UTF-8 Hangul byte-BPE prefixes/suffixes
      - All ASCII digits, punctuation, spaces, short subwords + ASR control tokens.
    """
    from tokenizers import Tokenizer as HFTokenizer

    hf_tok = HFTokenizer.from_file(str(snap_dir / "tokenizer.json"))
    vocab = hf_tok.get_vocab()
    byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}

    ko_compact = set(range(151600, 151936))
    for w in ["language", "Korean", "English", "None", "null", " language", " Korean", " English", "<asr_text>"]:
        for tid in hf_tok.encode(w, add_special_tokens=False).ids:
            ko_compact.add(tid)

    allowed_lead = {0xE1, 0xE2, 0xE3, 0xEA, 0xEB, 0xEC, 0xED}
    for token_str, tid in vocab.items():
        if tid >= 151600:
            continue
        raw_bytes = [byte_decoder[c] for c in token_str if c in byte_decoder]
        s = hf_tok.decode([tid], skip_special_tokens=False)
        has_hangul = any(
            (0x1100 <= ord(c) <= 0x11FF) or (0x3130 <= ord(c) <= 0x318F) or (0xAC00 <= ord(c) <= 0xD7A3)
            for c in s
        )
        if has_hangul:
            # Always include every token containing Hangul, even merged Byte-BPE tokens with leading/trailing '\ufffd'
            # (e.g., 62618 '\ufffd이' in '님이', 89363 '\ufffd습니다', 48458 '\ufffd니다', 19946 '\ufffd다', 50362 '\ufffd이터')
            ko_compact.add(tid)
            continue
        if "\ufffd" not in s and s:
            s_strip = s.strip()
            is_ascii_basic = all(ord(c) < 0x0080 for c in s) and (
                not s_strip or all(not c.isalpha() for c in s_strip) or len(s_strip) <= 3
            )
            if is_ascii_basic or any(c in "‘’“”·…—–₩°±×÷％" for c in s):
                ko_compact.add(tid)
        else:
            non_ascii = [b for b in raw_bytes if b >= 0x80]
            ascii_prefix = [b for b in raw_bytes if b < 0x80]
            if non_ascii and all(b in (0x20, 0x0A, 0x09) for b in ascii_prefix):
                if all((0x80 <= b <= 0xBF) or (b in allowed_lead) for b in non_ascii):
                    ko_compact.add(tid)

    active_ids_list = sorted(ko_compact)
    while len(active_ids_list) % 64 != 0:
        active_ids_list.append(active_ids_list[-1])
    return np.array(active_ids_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# 4. Mac TheoremFlow-Ω Metal Fast Decoder (Fused QKV + Fused GateUp + Active Vocab)
# ---------------------------------------------------------------------------
class TheoremFlowOmegaMLP(nn.Module):
    """Fused Gate+Up SwiGLU MLP on Apple Silicon Metal."""

    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gu = self.gate_up_proj(x)
        g, u = mx.split(gu, 2, axis=-1)
        return self.down_proj(nn.silu(g) * u)


class TheoremFlowOmegaAttention(nn.Module):
    """Fused QKV Grouped-Query Attention with QK-norm, RoPE, and SDPA on Apple Silicon Metal."""

    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads  # 16
        self.n_kv_heads = config.num_key_value_heads  # 8
        self.head_dim = config.head_dim  # 128
        self.q_dim = self.n_heads * self.head_dim  # 2048
        self.kv_dim = self.n_kv_heads * self.head_dim  # 1024
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(config.hidden_size, self.q_dim + 2 * self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, config.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array, cache: KVCache | None, layer_idx: int) -> mx.array:
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)

        q = self.q_norm(q.reshape(B, T, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, T, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        mask = create_additive_causal_mask(T, offset=offset).astype(q.dtype) if T > 1 else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.q_dim)
        return self.o_proj(out)


class TheoremFlowOmegaDecoderLayer(nn.Module):
    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TheoremFlowOmegaAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TheoremFlowOmegaMLP(config)

    def __call__(self, x: mx.array, cache: KVCache | None, layer_idx: int) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cache, layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MacTheoremFlowOmegaDecoder(nn.Module):
    """
    Mac TheoremFlow-Ω Decoder:
      - 28 Fused QKV + Fused GateUp quantized layers
      - Active-Vocab Readout (`active_lm_head [25696, 2048]`) reducing lm_head GEMV by 5.91x
    """

    def __init__(self, config: TextDecoderConfig, active_vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.active_vocab_size = active_vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TheoremFlowOmegaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.active_lm_head = nn.Linear(config.hidden_size, active_vocab_size, bias=False)

    def forward_active_logits(
        self,
        inputs: mx.array,
        cache: KVCache | None = None,
        is_embeds: bool = False,
    ) -> mx.array:
        h = inputs if is_embeds else self.embed_tokens(inputs)
        for i, layer in enumerate(self.layers):
            h = layer(h, cache, layer_idx=i)
        h_last = self.norm(h[:, -1:, :])
        return self.active_lm_head(h_last)

    def forward_timestamp_logits(
        self,
        inputs: mx.array,
        ts_positions: mx.array,
    ) -> mx.array:
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(h, None, layer_idx=i)
        h_ts = self.norm(h[:, ts_positions, :])
        return self.active_lm_head(h_ts)


class MacTheoremFlowOmegaForcedAligner:
    """
    Mac Rung Ω Fused QKV/GateUp Forced Aligner (MLX)
    (`Qwen/Qwen3-ForcedAligner-0.6B` quantized to Group-64 4-bit on Apple Silicon Metal).
    Provides:
      1) Single-pass non-autoregressive word-level timestamp alignment (80ms bin resolution)
      2) Monotonic LIS timestamp solver + per-word acoustic/alignment confidence scoring
      3) Exact inter-word boundary splitting (`find_word_aligned_split`) so long streaming utterances
         never cut inside a Korean word
      4) Trailing silence hallucination filter (`filter_phantom_tail_words`).
    """

    AUDIO_START_TOKEN_ID = 151669
    AUDIO_PAD_TOKEN_ID = 151676
    AUDIO_END_TOKEN_ID = 151670
    TIMESTAMP_TOKEN_ID = 151705
    TIMESTAMP_SEGMENT_MS = 80.0

    def __init__(
        self,
        config: ModelConfig,
        encoder: AudioEncoder,
        decoder: MacTheoremFlowOmegaDecoder,
        hf_tokenizer: Any,
    ) -> None:
        self._config = config
        self._encoder = encoder
        self._decoder = decoder
        self._hf_tok = hf_tokenizer

    @classmethod
    def from_omega_dir(cls, fa_omega_dir: str | Path) -> "MacTheoremFlowOmegaForcedAligner":
        from tokenizers import Tokenizer as HFTokenizer

        fa_omega_dir = Path(fa_omega_dir)
        config = ModelConfig.from_pretrained(fa_omega_dir)
        q_cfg = json.loads((fa_omega_dir / "omega_config.json").read_text(encoding="utf-8"))
        classify_num = int(q_cfg.get("classify_num", 5000))

        encoder = AudioEncoder(config.audio_encoder)
        decoder = MacTheoremFlowOmegaDecoder(config.text_decoder, active_vocab_size=classify_num)
        nn.quantize(decoder, group_size=q_cfg["group_size"], bits=q_cfg["bits"])

        raw = _load_safetensors_auto(fa_omega_dir / "model.safetensors")
        enc_w = []
        for k, v in raw.items():
            if k.startswith("audio_tower."):
                lk = k[len("audio_tower.") :]
                if lk.startswith("conv2d") and lk.endswith(".weight") and v.ndim == 4 and v.shape[1] != v.shape[2]:
                    v = v.transpose(0, 2, 3, 1)
                enc_w.append((lk, v))
        dec_w = [(k[len("model.") :], v) for k, v in raw.items() if k.startswith("model.")]

        encoder.load_weights(enc_w)
        # Quantize AudioEncoder Linear layers to 8-bit on Metal for fast, low-memory alignment
        nn.quantize(
            encoder,
            group_size=64,
            bits=8,
            class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0,
        )
        decoder.load_weights(dec_w)
        mx.eval(encoder.parameters(), decoder.parameters())
        hf_tok = HFTokenizer.from_file(str(fa_omega_dir / "tokenizer.json"))
        return cls(config, encoder, decoder, hf_tok)

    def encode_audio(self, audio_np: np.ndarray) -> mx.array:
        from qwen3_asr_mlx.audio import _get_mel_filterbank

        x = np.asarray(audio_np, dtype=np.float32)
        window = np.hanning(401)[:-1].astype(np.float32)
        padded = np.pad(x, 200, mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160] * window
        stft = np.fft.rfft(frames, n=400, axis=-1).T
        magnitudes = (np.abs(stft[:, :-1]) ** 2).astype(np.float32)
        mel_filters = _get_mel_filterbank(400, 128, 16000, 0.0, 8000.0)
        mel_spec = mel_filters @ magnitudes
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        mel_batched = mx.array(log_spec)[None, :, :]
        audio_features = self._encoder(mel_batched)
        mx.eval(audio_features)
        return audio_features

    @staticmethod
    def _fix_timestamp_lis(data: List[float]) -> Tuple[List[int], List[bool]]:
        n = len(data)
        if n == 0:
            return [], []
        dp = [1] * n
        parent = [-1] * n
        for i in range(1, n):
            for j in range(i):
                if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        max_len = max(dp)
        idx = dp.index(max_len)
        lis_indices = []
        while idx != -1:
            lis_indices.append(idx)
            idx = parent[idx]
        lis_indices.reverse()

        is_normal = [False] * n
        for k in lis_indices:
            is_normal[k] = True

        result = list(data)
        i = 0
        while i < n:
            if not is_normal[i]:
                j = i
                while j < n and not is_normal[j]:
                    j += 1
                anomaly_count = j - i
                left_val = next((result[k] for k in range(i - 1, -1, -1) if is_normal[k]), None)
                right_val = next((result[k] for k in range(j, n) if is_normal[k]), None)
                if anomaly_count <= 2:
                    for k in range(i, j):
                        if left_val is None:
                            result[k] = right_val if right_val is not None else 0.0
                        elif right_val is None:
                            result[k] = left_val
                        else:
                            result[k] = left_val if (k - (i - 1)) <= (j - k) else right_val
                else:
                    if left_val is not None and right_val is not None:
                        step = (right_val - left_val) / (anomaly_count + 1)
                        for k in range(i, j):
                            result[k] = left_val + step * (k - i + 1)
                    elif left_val is not None:
                        for k in range(i, j):
                            result[k] = left_val
                    elif right_val is not None:
                        for k in range(i, j):
                            result[k] = right_val
                i = j
            else:
                i += 1
        return [int(round(r)) for r in result], is_normal

    def align_words(self, audio_np: np.ndarray, text: str) -> List[Dict[str, Any]]:
        """
        Aligns whitespace-separated Korean words (preserving punctuation on each word token)
        against `audio_np` (16kHz float32) in a single non-autoregressive Metal pass.
        """
        raw_words = [w for w in (text or "").strip().split() if w]
        if not raw_words or audio_np is None or audio_np.size < 1600:
            return []

        clean_words = []
        kept_raw_words = []
        for rw in raw_words:
            cw = "".join(ch for ch in rw if ch.isalnum() or ch == "'")
            if cw:
                clean_words.append(cw)
                kept_raw_words.append(rw)
        if not clean_words:
            return []

        audio_features = self.encode_audio(audio_np)
        n_audio = int(audio_features.shape[1])

        input_ids: List[int] = (
            [self.AUDIO_START_TOKEN_ID]
            + [self.AUDIO_PAD_TOKEN_ID] * n_audio
            + [self.AUDIO_END_TOKEN_ID]
        )
        ts_positions: List[int] = []
        for cw in clean_words:
            w_ids = self._hf_tok.encode(cw, add_special_tokens=False).ids
            if not w_ids:
                continue
            input_ids.extend(w_ids)
            ts_positions.append(len(input_ids))
            input_ids.append(self.TIMESTAMP_TOKEN_ID)
            ts_positions.append(len(input_ids))
            input_ids.append(self.TIMESTAMP_TOKEN_ID)

        if not ts_positions:
            return []

        prompt_mx = mx.array([input_ids])
        token_embeds = self._decoder.embed_tokens(prompt_mx)
        start_pos = 1
        end_pos = 1 + n_audio
        combined = mx.concatenate(
            [token_embeds[:, :start_pos, :], audio_features, token_embeds[:, end_pos:, :]],
            axis=1,
        )
        ts_pos_mx = mx.array(ts_positions)
        logits_ts = self._decoder.forward_timestamp_logits(combined, ts_pos_mx)
        probs_ts = mx.softmax(logits_ts, axis=-1)
        best_bins = mx.argmax(logits_ts, axis=-1)
        best_probs = mx.max(probs_ts, axis=-1)
        mx.eval(best_bins, best_probs)

        bins_np = np.array(best_bins[0]).tolist()
        probs_np = np.array(best_probs[0]).tolist()
        raw_ms = [b * self.TIMESTAMP_SEGMENT_MS for b in bins_np]
        fixed_ms, is_normal = self._fix_timestamp_lis(raw_ms)

        total_dur_sec = audio_np.shape[0] / float(SAMPLE_RATE)
        items: List[Dict[str, Any]] = []
        n_aligned = len(ts_positions) // 2
        for i in range(n_aligned):
            s_sec = min(total_dur_sec, max(0.0, fixed_ms[2 * i] / 1000.0))
            e_sec = min(total_dur_sec, max(s_sec, fixed_ms[2 * i + 1] / 1000.0))
            conf = 0.5 * (float(probs_np[2 * i]) + float(probs_np[2 * i + 1]))
            mono = bool(is_normal[2 * i] and is_normal[2 * i + 1])
            items.append(
                {
                    "word": kept_raw_words[i],
                    "clean_word": clean_words[i],
                    "start_time": round(s_sec, 3),
                    "end_time": round(e_sec, 3),
                    "confidence": round(conf, 4),
                    "is_monotonic": mono,
                }
            )
        del combined, token_embeds, audio_features, logits_ts, probs_ts
        mx.clear_cache()
        return items

    def filter_phantom_tail_words(self, audio_np: np.ndarray, text: str) -> str:
        """
        Strips trailing hallucinated words at the end of an utterance if ForcedAligner shows
        they have collapsed 0ms duration at the very end of the audio AND near-zero acoustic energy.
        """
        words = [w for w in (text or "").strip().split() if w]
        if len(words) <= 2 or audio_np is None or audio_np.size < 8000:
            return text
        try:
            items = self.align_words(audio_np, text)
            if len(items) != len(words):
                return text
            keep_n = len(items)
            total_samples = audio_np.shape[0]
            while keep_n > 1:
                last = items[keep_n - 1]
                dur = last["end_time"] - last["start_time"]
                s_idx = max(0, min(total_samples - 320, int(last["start_time"] * SAMPLE_RATE)))
                e_idx = max(s_idx + 320, min(total_samples, int((last["end_time"] + 0.08) * SAMPLE_RATE)))
                seg_rms = float(np.sqrt(np.mean(audio_np[s_idx:e_idx] ** 2)))
                if dur <= 0.001 and not last["is_monotonic"] and last["confidence"] < 0.15 and seg_rms < 0.0018:
                    keep_n -= 1
                else:
                    break
            if keep_n < len(words):
                return " ".join(words[:keep_n]).strip()
        except Exception:
            pass
        return text

    def find_word_aligned_split(
        self,
        audio_np: np.ndarray,
        text: str,
        min_sec: float,
        max_sec: float,
        sr: int = SAMPLE_RATE,
    ) -> Optional[Tuple[str, int]]:
        """
        Uses Rung Ω ForcedAligner word timestamps to find the exact inter-word pause boundary
        inside [min_sec, max_sec] so long utterances split cleanly between words without cutting
        a word in half or needing a lossy re-decode of the first half.
        Returns (prefix_text_up_to_split, split_sample_index) or None.
        """
        if not text or audio_np is None or audio_np.size < int(min_sec * sr):
            return None
        try:
            items = self.align_words(audio_np, text)
            if len(items) < 3:
                return None
            total_samples = int(audio_np.shape[0])
            best_score = -1e9
            best_idx = -1
            best_split_sample = -1

            for i in range(1, len(items) - 1):
                w_cur = items[i]
                w_nxt = items[i + 1]
                gap_start = w_cur["end_time"]
                gap_end = max(gap_start, w_nxt["start_time"])
                mid_sec = 0.5 * (gap_start + gap_end)
                if mid_sec < min_sec or mid_sec > max_sec:
                    continue

                gap_dur = gap_end - gap_start
                s1 = max(0, min(total_samples - 160, int(gap_start * sr) - 400))
                s2 = max(s1 + 160, min(total_samples, int(gap_end * sr) + 400))
                local = np.abs(audio_np[s1:s2])
                min_rel = int(np.argmin(local))
                cand_sample = s1 + min_rel
                local_rms = float(np.sqrt(np.mean(audio_np[s1:s2] ** 2)))

                # Prefer sentence-ending words, longer inter-word gaps, and quieter valleys
                punct_bonus = 0.25 if w_cur["word"].endswith((".", "?", "!", "다", "요", "죠", "네")) else 0.0
                score = gap_dur * 2.0 + punct_bonus - local_rms * 8.0
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_split_sample = cand_sample

            if best_idx >= 1 and 0 < best_split_sample < total_samples:
                prefix_text = " ".join(it["word"] for it in items[: best_idx + 1]).strip()
                return prefix_text, best_split_sample
        except Exception:
            pass
        return None


class MacTheoremFlowOmegaASREngine:
    """
    End-to-end Apple Silicon Metal ASR Engine for Rung Ω weights
    and TheoremFlow-Ω Fused QKV/GateUp + 25,696 Active Korean Vocab Readout.
    """

    def __init__(
        self,
        config: ModelConfig,
        encoder: AudioEncoder,
        decoder: MacTheoremFlowOmegaDecoder,
        tokenizer: Tokenizer,
        active_ids: np.ndarray,
        forced_aligner: Optional[MacTheoremFlowOmegaForcedAligner] = None,
    ) -> None:
        self._config = config
        self._encoder = encoder
        self._decoder = decoder
        self._tokenizer = tokenizer
        self.active_ids_np = active_ids
        self.active_ids_mx = mx.array(active_ids)
        self.forced_aligner = forced_aligner

    @classmethod
    def from_omega_dir(cls, omega_dir: str | Path, fa_dir: str | Path | None = None) -> "MacTheoremFlowOmegaASREngine":
        omega_dir = Path(omega_dir)
        config = ModelConfig.from_pretrained(omega_dir)
        active_ids = np.load(str(omega_dir / "active_ids.npy"))
        q_cfg = json.loads((omega_dir / "omega_config.json").read_text(encoding="utf-8"))

        encoder = AudioEncoder(config.audio_encoder)
        decoder = MacTheoremFlowOmegaDecoder(config.text_decoder, active_vocab_size=len(active_ids))
        nn.quantize(decoder, group_size=q_cfg["group_size"], bits=q_cfg["bits"])

        raw = _load_safetensors_auto(omega_dir / "model.safetensors")
        enc_w = []
        for k, v in raw.items():
            if k.startswith("audio_tower."):
                lk = k[len("audio_tower.") :]
                if lk.startswith("conv2d") and lk.endswith(".weight") and v.ndim == 4 and v.shape[1] != v.shape[2]:
                    v = v.transpose(0, 2, 3, 1)
                enc_w.append((lk, v))
        dec_w = [(k[len("model.") :], v) for k, v in raw.items() if k.startswith("model.")]

        encoder.load_weights(enc_w)
        decoder.load_weights(dec_w)

        # Dynamically ensure 100% of merged Hangul Byte-BPE tokens (e.g. 62618 in '님이', 89363 '습니다')
        # are present in active_lm_head by slicing directly from tied quantized embed_tokens
        full_active_ids = build_lossless_ko_ascii_active_vocab(omega_dir)
        if not np.array_equal(full_active_ids, active_ids):
            full_active_mx = mx.array(full_active_ids)
            decoder.active_lm_head.weight = decoder.embed_tokens.weight[full_active_mx]
            decoder.active_lm_head.scales = decoder.embed_tokens.scales[full_active_mx]
            if getattr(decoder.embed_tokens, "biases", None) is not None:
                decoder.active_lm_head.biases = decoder.embed_tokens.biases[full_active_mx]
            decoder.active_vocab_size = int(len(full_active_ids))
            active_ids = full_active_ids

        mx.eval(encoder.parameters(), decoder.parameters())
        tokenizer = Tokenizer(omega_dir)

        fa_engine = None
        if fa_dir is not None and (Path(fa_dir) / "model.safetensors").exists():
            try:
                fa_engine = MacTheoremFlowOmegaForcedAligner.from_omega_dir(fa_dir)
            except Exception as e:
                print(f"[RungOmega] ForcedAligner 로드 건너뜀: {e}")

        return cls(config, encoder, decoder, tokenizer, active_ids, forced_aligner=fa_engine)

    def encode_audio(self, audio_np: np.ndarray) -> mx.array:
        from qwen3_asr_mlx.audio import _get_mel_filterbank

        x = np.asarray(audio_np, dtype=np.float32)
        window = np.hanning(401)[:-1].astype(np.float32)
        padded = np.pad(x, 200, mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160] * window
        stft = np.fft.rfft(frames, n=400, axis=-1).T
        magnitudes = (np.abs(stft[:, :-1]) ** 2).astype(np.float32)
        mel_filters = _get_mel_filterbank(400, 128, 16000, 0.0, 8000.0)
        mel_spec = mel_filters @ magnitudes
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        mel_batched = mx.array(log_spec)[None, :, :]
        audio_features = self._encoder(mel_batched)
        mx.eval(audio_features)
        return audio_features

    def init_streaming_state(
        self,
        chunk_size_sec: float = 1.0,
        unfixed_chunk_num: int = 4,
        unfixed_token_num: int = 12,
        language: str = "Korean",
        context: str = "",
    ) -> Any:
        return self.create_streaming_state(
            chunk_size_sec=chunk_size_sec,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            language=language,
            context=context,
        )

    def create_streaming_state(
        self,
        chunk_size_sec: float = 1.0,
        unfixed_chunk_num: int = 4,
        unfixed_token_num: int = 12,
        language: str = "Korean",
        context: str = "",
    ) -> Any:
        import sys
        sys.path.insert(0, "/Users/yun-yeowon/qwen3-asr-ko")
        from mlx_quantized_asr import MLXStreamingState

        return MLXStreamingState(
            chunk_size_sec=chunk_size_sec,
            chunk_size_samples=int(round(chunk_size_sec * 16000)),
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            language=language,
            context=context,
            chunk_id=0,
            buffer=np.zeros((0,), dtype=np.float32),
            audio_accum=np.zeros((0,), dtype=np.float32),
            text="",
            decoded_tokens=[],
        )

    @staticmethod
    def _prepare_eval_audio(audio_accum: np.ndarray, pad_sec: float = 0.0) -> np.ndarray:
        """
        Smooth whole-utterance peak normalization (protecting quiet speakers from log-mel floor clipping)
        + MIN_ASR_INPUT_SECONDS (0.5s) minimum length guard without artificial boundary silence cliffs.
        """
        x = np.asarray(audio_accum, dtype=np.float32)
        if x.size == 0:
            return x
        peak = float(np.max(np.abs(x)))
        if 0.001 < peak < 0.45:
            x = x * min(8.0, 0.45 / peak)
        elif peak > 1.0:
            x = x / peak
        x = np.clip(x, -1.0, 1.0)
        min_samples = 8000  # MIN_ASR_INPUT_SECONDS = 0.5s at 16kHz
        if x.shape[0] < min_samples:
            x = np.pad(x, (0, min_samples - x.shape[0]), mode="constant", constant_values=0.0).astype(np.float32)
        elif pad_sec > 0.0:
            pad_samples = int(round(pad_sec * 16000))
            if pad_samples > 0:
                x = np.concatenate([x, np.zeros((pad_samples,), dtype=np.float32)], axis=0)
        return x

    @staticmethod
    def _detect_and_fix_repetitions(text: str, threshold: int = 5) -> str:
        """Official qwen_asr repetition cleaner adapted for Korean syllable/phrase loops."""
        if not text:
            return text
        # 1) Collapse character repeats
        res = []
        i = 0
        n = len(text)
        while i < n:
            count = 1
            while i + count < n and text[i + count] == text[i]:
                count += 1
            if count > threshold:
                res.append(text[i])
            else:
                res.append(text[i : i + count])
            i += count
        s = "".join(res)

        # 2) Collapse pattern repeats (e.g. "아야지 아야지 아야지 ")
        def _fix_patterns(inp: str, thresh: int = 4, max_len: int = 16) -> str:
            m = len(inp)
            if m < thresh * 2:
                return inp
            idx = 0
            out = []
            while idx <= m - thresh * 2:
                found = False
                for k in range(1, max_len + 1):
                    if idx + k * thresh > m:
                        break
                    pat = inp[idx : idx + k]
                    if all(inp[idx + r * k : idx + (r + 1) * k] == pat for r in range(1, thresh)):
                        end_idx = idx + thresh * k
                        while end_idx + k <= m and inp[end_idx : end_idx + k] == pat:
                            end_idx += k
                        out.append(pat)
                        out.append(_fix_patterns(inp[end_idx:], thresh, max_len))
                        return "".join(out)
                out.append(inp[idx])
                idx += 1
            out.append(inp[idx:])
            return "".join(out)

        return _fix_patterns(s, thresh=4, max_len=16).strip()

    def _build_rollback_prefix_tokens(self, state: Any) -> List[int]:
        """
        Official `qwen_asr/inference/qwen3_asr.py` prefix rollback strategy:
        - If chunk_id < unfixed_chunk_num: prefix_tokens = []
        - Else: roll back last `unfixed_token_num` tokens from accumulated decoded_tokens,
          advancing k if partial UTF-8 ('\\ufffd') is encountered.
        """
        if state.chunk_id < state.unfixed_chunk_num or not state.decoded_tokens:
            return []
        cur_ids = list(state.decoded_tokens)
        k = int(state.unfixed_token_num)
        while True:
            end_idx = max(0, len(cur_ids) - k)
            if end_idx == 0:
                return []
            cand = cur_ids[:end_idx]
            prefix_str = self._tokenizer.decode(cand)
            if "\ufffd" not in prefix_str:
                return cand
            k += 1

    def _decode_utterance_clean(
        self,
        audio_accum: np.ndarray,
        language: str,
        context: str,
        prefix_tokens: List[int],
        max_new_tokens: int = 96,
        pad_sec: float = 0.0,
    ) -> tuple[List[int], str]:
        eval_audio = self._prepare_eval_audio(audio_accum, pad_sec=pad_sec)
        audio_features = self.encode_audio(eval_audio)
        n_audio = int(audio_features.shape[1])

        prompt_ids = self._tokenizer.build_prompt(
            n_audio, language=language, context=context or None
        )
        full_prompt_ids = prompt_ids + prefix_tokens
        prompt_mx = mx.array([full_prompt_ids])
        token_embeds = self._decoder.embed_tokens(prompt_mx)

        positions = [i for i, t in enumerate(full_prompt_ids) if t == AUDIO_PAD_TOKEN_ID]
        start_pos = positions[0]
        end_pos = positions[-1] + 1
        combined = mx.concatenate(
            [token_embeds[:, :start_pos, :], audio_features, token_embeds[:, end_pos:, :]],
            axis=1,
        )

        cache = KVCache()
        logits_active = self._decoder.forward_active_logits(combined, cache=cache, is_embeds=True)
        cache.offset = combined.shape[1]
        best_idx = mx.argmax(logits_active[:, -1, :], axis=-1)
        next_tok = self.active_ids_mx[best_idx]
        mx.eval(next_tok)

        generated: List[int] = []
        tok_id = int(next_tok.item())

        while tok_id not in EOS_TOKEN_IDS and len(generated) < max_new_tokens:
            generated.append(tok_id)
            # Only break on >= 6 consecutive repeats so "1000" (three '0' tokens) or "알아 알아 알아" never abort mid-sentence
            loop_cut = 0
            n_gen = len(generated)
            for k_len in range(1, 7):
                if n_gen >= 6 * k_len:
                    pat = generated[-k_len:]
                    if all(generated[-r * k_len : -(r - 1) * k_len] == pat for r in range(2, 7)):
                        loop_cut = 5 * k_len
                        break
            if loop_cut > 0:
                generated = generated[:-loop_cut]
                break

            tok_input = mx.array([[tok_id]])
            logits_active = self._decoder.forward_active_logits(tok_input, cache=cache, is_embeds=False)
            cache.offset += 1
            best_idx = mx.argmax(logits_active[:, -1, :], axis=-1)
            next_tok = self.active_ids_mx[best_idx]
            mx.eval(next_tok)
            tok_id = int(next_tok.item())

        all_tokens = prefix_tokens + generated
        raw_text = self._tokenizer.decode(all_tokens)
        _, parsed = parse_language_and_output(raw_text)
        cleaned = self._detect_and_fix_repetitions(parsed.strip(), threshold=6)

        del cache, combined, token_embeds, audio_features, logits_active
        mx.clear_cache()
        return all_tokens, cleaned

    def _step_streaming(self, state: Any) -> None:
        MAX_AUDIO_WINDOW_SAMPLES = 30 * 16000
        if state.audio_accum.shape[0] > MAX_AUDIO_WINDOW_SAMPLES:
            state.audio_accum = state.audio_accum[-MAX_AUDIO_WINDOW_SAMPLES:]
            state.decoded_tokens = []

        if state.audio_accum.shape[0] < 3200:
            return

        utt_rms = float(np.sqrt(np.mean(state.audio_accum**2)))
        if utt_rms < 0.0008:
            return

        prev_text = state.text
        prev_tokens = list(state.decoded_tokens)

        # Official qwen_asr prefix rollback (chunk_id < unfixed_chunk_num -> [], else rollback unfixed_token_num)
        use_prefix = self._build_rollback_prefix_tokens(state)

        all_tokens, cleaned = self._decode_utterance_clean(
            audio_accum=state.audio_accum,
            language=state.language,
            context=state.context,
            prefix_tokens=use_prefix,
            max_new_tokens=96,
            pad_sec=0.0,
        )
        setattr(state, "_last_decoded_samples", int(state.audio_accum.shape[0]))

        if cleaned:
            state.decoded_tokens = all_tokens
            state.text = cleaned
        else:
            state.decoded_tokens = prev_tokens
            state.text = prev_text

        state.chunk_id += 1

    def streaming_transcribe(self, pcm16k: np.ndarray, state: Any) -> Any:
        if pcm16k is not None and pcm16k.size > 0:
            state.buffer = np.concatenate([state.buffer, pcm16k.astype(np.float32, copy=False)])
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]
            state.audio_accum = np.concatenate([state.audio_accum, chunk])
            self._step_streaming(state)
        return state

    def finish_streaming_transcribe(self, state: Any) -> Any:
        valley_truncated = getattr(state, "_valley_truncated", False)
        prev_best = "" if valley_truncated else (state.text or "")
        last_decoded_samples = getattr(state, "_last_decoded_samples", 0)

        if state.buffer is not None and state.buffer.shape[0] > 0:
            state.audio_accum = np.concatenate([state.audio_accum, state.buffer])
            state.buffer = np.zeros((0,), dtype=np.float32)

        needs_final_decode = (
            valley_truncated
            or (state.audio_accum.shape[0] != last_decoded_samples)
            or (not prev_best)
        )
        if needs_final_decode and state.audio_accum.shape[0] >= 3200:
            # Always perform a clean full-utterance decode with prefix_tokens=[] on utterance finish
            # so any early-chunk boundary errors are 100% corrected by full bidirectional audio context.
            all_tokens, cleaned = self._decode_utterance_clean(
                audio_accum=state.audio_accum,
                language=state.language,
                context=state.context,
                prefix_tokens=[],
                max_new_tokens=224,
                pad_sec=0.0,
            )
            if cleaned and self.forced_aligner is not None:
                cleaned = self.forced_aligner.filter_phantom_tail_words(state.audio_accum, cleaned)
            if cleaned:
                state.decoded_tokens = all_tokens
                state.text = cleaned
            elif prev_best:
                state.text = prev_best
            state.chunk_id += 1
        elif prev_best and not state.text:
            state.text = prev_best

        state.audio_accum = np.zeros((0,), dtype=np.float32)
        state.buffer = np.zeros((0,), dtype=np.float32)
        setattr(state, "_valley_truncated", False)
        mx.clear_cache()
        return state

    def split_at_word_boundary(
        self,
        state: Any,
        min_sec: float,
        max_sec: float,
        sr: int = SAMPLE_RATE,
    ) -> Optional[Tuple[str, np.ndarray]]:
        """
        Uses a full-context clean decode + Rung Ω ForcedAligner word timestamps to split
        a long streaming utterance (`hard_ceiling_sec`) at the exact inter-word boundary
        `[word[i].end_time, word[i+1].start_time]`, returning `(committed_prefix_text, carryover_audio)`.
        Eliminates mid-word truncation ("시키겠" / "습니다") and double-decoding drift.
        """
        if self.forced_aligner is None or state.audio_accum.shape[0] < int(min_sec * sr):
            return None
        try:
            _, full_cleaned = self._decode_utterance_clean(
                audio_accum=state.audio_accum,
                language=state.language,
                context=state.context,
                prefix_tokens=[],
                max_new_tokens=224,
                pad_sec=0.0,
            )
            if not full_cleaned:
                return None
            split_res = self.forced_aligner.find_word_aligned_split(
                state.audio_accum,
                full_cleaned,
                min_sec=min_sec,
                max_sec=max_sec,
                sr=sr,
            )
            if split_res is None:
                return None
            prefix_text, split_sample = split_res
            carryover = state.audio_accum[split_sample:].copy()
            state.audio_accum = np.zeros((0,), dtype=np.float32)
            state.buffer = np.zeros((0,), dtype=np.float32)
            state.text = prefix_text
            mx.clear_cache()
            return prefix_text, carryover
        except Exception:
            return None

    def transcribe_with_metrics(
        self,
        audio_np: np.ndarray,
        language: str = "Korean",
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        audio_features = self.encode_audio(audio_np)
        num_audio_tokens = audio_features.shape[1]

        prompt_ids = self._tokenizer.build_prompt(
            num_audio_tokens, language=language, context=None
        )
        prompt_mx = mx.array([prompt_ids])
        token_embeds = self._decoder.embed_tokens(prompt_mx)

        positions = [i for i, t in enumerate(prompt_ids) if t == AUDIO_PAD_TOKEN_ID]
        start_pos = positions[0]
        end_pos = positions[-1] + 1
        combined = mx.concatenate(
            [token_embeds[:, :start_pos, :], audio_features, token_embeds[:, end_pos:, :]],
            axis=1,
        )

        cache = KVCache()
        logits_active = self._decoder.forward_active_logits(combined, cache=cache, is_embeds=True)
        cache.offset = combined.shape[1]
        best_idx = mx.argmax(logits_active[:, -1, :], axis=-1)
        next_tok = self.active_ids_mx[best_idx]
        mx.eval(next_tok)
        t_prefill = time.perf_counter()

        generated: List[int] = []
        tok_id = int(next_tok.item())

        while tok_id not in EOS_TOKEN_IDS and len(generated) < max_tokens:
            generated.append(tok_id)
            tok_input = mx.array([[tok_id]])
            logits_active = self._decoder.forward_active_logits(tok_input, cache=cache, is_embeds=False)
            cache.offset += 1
            best_idx = mx.argmax(logits_active[:, -1, :], axis=-1)
            next_tok = self.active_ids_mx[best_idx]
            mx.eval(next_tok)
            tok_id = int(next_tok.item())

        t_end = time.perf_counter()
        raw_output = self._tokenizer.decode(generated)
        _, text = parse_language_and_output(raw_output)
        prefill_ms = (t_prefill - t0) * 1000.0
        decode_ms = (t_end - t_prefill) * 1000.0
        total_ms = (t_end - t0) * 1000.0
        n_tok = max(len(generated), 1)
        return {
            "text": text.strip(),
            "generated_ids": generated,
            "latency_ms": round(total_ms, 2),
            "prefill_ms": round(prefill_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "ms_per_token": round(decode_ms / n_tok, 2),
            "tokens_per_sec": round(n_tok / max((t_end - t_prefill), 1e-6), 1),
        }


# ---------------------------------------------------------------------------
# 5. Korean CER & Evaluation Utilities
# ---------------------------------------------------------------------------
