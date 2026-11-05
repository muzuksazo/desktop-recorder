# coding=utf-8
"""
Apple Silicon MLX 4-bit / 8-bit Quantized & Real-Time Streaming Engine
for `yeowon/qwen3_asr_1.7b_ko-MLX-4bit` and `TeamUNIVA/qwen3_asr_1.7b_ko`.
"""
from __future__ import annotations

import gc
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
from huggingface_hub import snapshot_download

from qwen3_asr_mlx.audio import log_mel_spectrogram
from qwen3_asr_mlx.config import ModelConfig
from qwen3_asr_mlx.decoder import KVCache, TextDecoder
from qwen3_asr_mlx.encoder import AudioEncoder
from qwen3_asr_mlx.generate import prepare_inputs, sample
from qwen3_asr_mlx.model import Qwen3ASR, TranscriptionResult
from qwen3_asr_mlx.tokenizer import EOS_TOKEN_IDS, Tokenizer

DEFAULT_MLX_4BIT_MODEL = "yeowon/qwen3_asr_1.7b_ko-MLX-4bit"
# Cap MLX Metal buffer pool cache to 128 MB so long-running streaming never balloons Unified Memory
MLX_CACHE_LIMIT_BYTES = 128 * 1024 * 1024
# Maximum rolling audio window (seconds) fed to the encoder per streaming utterance
MAX_STREAMING_AUDIO_WINDOW_SEC = 24.0


class QuantizableTextDecoder(TextDecoder):
    """Uses `self.embed_tokens.as_linear(h)` when `embed_tokens` is a `QuantizedEmbedding`."""

    def __call__(
        self,
        inputs: mx.array,
        cache: KVCache | None = None,
        is_embeds: bool = False,
    ) -> mx.array:
        h = inputs if is_embeds else self.embed_tokens(inputs)
        for i, layer in enumerate(self.layers):
            h = layer(h, cache, layer_idx=i)
        h = self.norm(h)
        if hasattr(self.embed_tokens, "as_linear"):
            return self.embed_tokens.as_linear(h)
        return h @ self.embed_tokens.weight.T


def load_hf_or_mlx_weights(
    encoder: AudioEncoder,
    decoder: TextDecoder,
    model_dir: Path,
    dtype: mx.Dtype = mx.bfloat16,
) -> None:
    weights_file = model_dir / "model.safetensors"
    raw = cast(dict[str, mx.array], mx.load(str(weights_file)))

    enc_mapped: list[tuple[str, mx.array]] = []
    dec_mapped: list[tuple[str, mx.array]] = []

    for key, val in raw.items():
        v = val.astype(dtype) if val.dtype in (mx.float32, mx.float16, mx.bfloat16) else val
        if key.startswith("thinker.audio_tower."):
            local_k = key[len("thinker.audio_tower.") :]
            if local_k.startswith("conv2d") and local_k.endswith(".weight") and v.ndim == 4:
                v = v.transpose(0, 2, 3, 1)
            enc_mapped.append((local_k, v))
        elif key.startswith("audio_tower."):
            local_k = key[len("audio_tower.") :]
            enc_mapped.append((local_k, v))
        elif key.startswith("thinker.model."):
            local_k = key[len("thinker.model.") :]
            dec_mapped.append((local_k, v))
        elif key.startswith("model."):
            local_k = key[len("model.") :]
            dec_mapped.append((local_k, v))

    encoder.load_weights(enc_mapped)
    decoder.load_weights(dec_mapped)
    mx.eval(encoder.parameters(), decoder.parameters())


def count_model_bytes(module: nn.Module) -> int:
    total = 0
    for _, v in mlx.utils.tree_flatten(module.parameters()):
        if isinstance(v, mx.array):
            total += v.nbytes
    return total


@dataclass
class MLXStreamingState:
    chunk_size_sec: float
    chunk_size_samples: int
    unfixed_chunk_num: int
    unfixed_token_num: int
    language: str
    context: str
    chunk_id: int
    buffer: np.ndarray
    audio_accum: np.ndarray
    text: str
    decoded_tokens: list[int]


class QuantizedMLXQwen3ASR:
    def __init__(
        self,
        model_id_or_path: str | Path = DEFAULT_MLX_4BIT_MODEL,
        bits: Optional[int] = 4,
        group_size: int = 64,
        quantize_encoder: bool = True,
    ) -> None:
        t0 = time.time()
        path = Path(model_id_or_path)
        local_cached = Path("/Users/yun-yeowon/qwen3-asr-ko/qwen3_asr_1.7b_ko_mlx_4bit")
        if not path.is_dir():
            if str(model_id_or_path) == DEFAULT_MLX_4BIT_MODEL and (local_cached / "model.safetensors").exists():
                path = local_cached
            else:
                path = Path(snapshot_download(repo_id=str(model_id_or_path)))

        self.model_path = path
        self.bits = bits
        self.group_size = group_size
        self.config = ModelConfig.from_pretrained(path)

        raw_cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        pre_quant = raw_cfg.get("quantization")

        self.encoder = AudioEncoder(self.config.audio_encoder)
        self.decoder = QuantizableTextDecoder(self.config.text_decoder)

        if pre_quant:
            q_bits = int(pre_quant.get("bits", 4))
            q_enc_bits = pre_quant.get("encoder_bits")
            q_group = int(pre_quant.get("group_size", 64))
            self.bits = q_bits
            self.group_size = q_group
            nn.quantize(self.decoder, group_size=q_group, bits=q_bits)
            if q_enc_bits in (4, 8):
                nn.quantize(
                    self.encoder,
                    group_size=q_group,
                    bits=int(q_enc_bits),
                    class_predicate=lambda _, m: isinstance(m, nn.Linear)
                    and m.weight.shape[-1] % q_group == 0,
                )
            load_hf_or_mlx_weights(self.encoder, self.decoder, path, dtype=mx.bfloat16)
        else:
            load_hf_or_mlx_weights(self.encoder, self.decoder, path, dtype=mx.bfloat16)
            if bits in (4, 8):
                nn.quantize(self.decoder, group_size=group_size, bits=bits)
                if quantize_encoder:
                    nn.quantize(
                        self.encoder,
                        group_size=group_size,
                        bits=bits,
                        class_predicate=lambda _, m: isinstance(m, nn.Linear)
                        and m.weight.shape[-1] % group_size == 0,
                    )
                mx.eval(self.encoder.parameters(), self.decoder.parameters())

        self.tokenizer = Tokenizer(path)
        self.asr = Qwen3ASR(self.config, self.encoder, self.decoder, self.tokenizer)
        self.lock = threading.Lock()

        # Restrict Metal cache pool and flush initial weight-loading staging buffers
        mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        mx.clear_cache()
        gc.collect()

        self.enc_mb = count_model_bytes(self.encoder) / (1024 * 1024)
        self.dec_mb = count_model_bytes(self.decoder) / (1024 * 1024)
        self.total_mb = self.enc_mb + self.dec_mb
        self.load_time_sec = time.time() - t0

    def warm_up(self) -> None:
        self.asr.warm_up()
        mx.clear_cache()

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        language: str | None = "Korean",
        max_tokens: int | None = 256,
    ) -> TranscriptionResult:
        try:
            return self.asr.transcribe(audio=audio, language=language, max_tokens=max_tokens)
        finally:
            mx.clear_cache()

    def init_streaming_state(
        self,
        language: str = "Korean",
        context: str = "",
        chunk_size_sec: float = 1.0,
        unfixed_chunk_num: int = 4,
        unfixed_token_num: int = 12,
    ) -> MLXStreamingState:
        chunk_size_samples = max(1, int(round(chunk_size_sec * 16000)))
        return MLXStreamingState(
            chunk_size_sec=float(chunk_size_sec),
            chunk_size_samples=chunk_size_samples,
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            language=language,
            context=context,
            chunk_id=0,
            buffer=np.zeros((0,), dtype=np.float32),
            audio_accum=np.zeros((0,), dtype=np.float32),
            text="",
            decoded_tokens=[],
        )

    def _decode_step_with_prefix(
        self,
        audio_samples: np.ndarray,
        language: str,
        context: str,
        prefix_tokens: list[int],
        max_new_tokens: int = 48,
    ) -> list[int]:
        with self.lock:
            mel = log_mel_spectrogram(audio_samples)
            encoder_output = self.encoder(mel)
            mx.eval(encoder_output)
            del mel

            n_audio_tokens = encoder_output.shape[1]
            base_input_ids = self.tokenizer.build_prompt(
                n_audio_tokens=n_audio_tokens,
                language=language,
                context=context or None,
            )
            full_input_ids = base_input_ids + prefix_tokens

            embeddings = prepare_inputs(encoder_output, full_input_ids, self.decoder.embed_tokens)
            del encoder_output
            cache = KVCache()
            seq_len = embeddings.shape[1]
            logits = self.decoder(embeddings, cache=cache, is_embeds=True)
            mx.eval(logits)
            del embeddings
            cache.offset = seq_len

            next_token = sample(logits[:, -1, :], temperature=0.0, repetition_penalty=1.1)
            new_tokens: list[int] = []

            for _ in range(max_new_tokens - 1):
                if next_token in EOS_TOKEN_IDS:
                    break
                new_tokens.append(next_token)
                loop_cut = 0
                n_gen = len(new_tokens)
                for k_len in range(1, 7):
                    if n_gen >= 6 * k_len:
                        pat = new_tokens[-k_len:]
                        if all(new_tokens[-r * k_len : -(r - 1) * k_len] == pat for r in range(2, 7)):
                            loop_cut = 5 * k_len
                            break
                if loop_cut > 0:
                    new_tokens = new_tokens[:-loop_cut]
                    break
                token_embed = self.decoder.embed_tokens(mx.array([[next_token]]))
                logits = self.decoder(token_embed, cache=cache, is_embeds=True)
                mx.eval(logits)
                cache.offset += 1
                recent = (prefix_tokens + new_tokens)[-64:]
                next_token = sample(
                    logits[:, -1, :],
                    temperature=0.0,
                    repetition_penalty=1.15,
                    recent_tokens=recent,
                )

            del cache, logits
            mx.clear_cache()
            return prefix_tokens + new_tokens

    def _clean_prefix_tokens(self, state: MLXStreamingState) -> list[int]:
        if state.chunk_id < state.unfixed_chunk_num or not state.decoded_tokens:
            return []
        k = state.unfixed_token_num
        while True:
            end_idx = max(0, len(state.decoded_tokens) - k)
            if end_idx == 0:
                return []
            candidate = state.decoded_tokens[:end_idx]
            text = self.tokenizer.decode(candidate, skip_special_tokens=True)
            if "\ufffd" not in text:
                return candidate
            k += 1

    @staticmethod
    def _prepare_eval_audio(audio_accum: np.ndarray, pad_sec: float = 0.0) -> np.ndarray:
        x = np.asarray(audio_accum, dtype=np.float32)
        if x.size == 0:
            return x
        peak = float(np.max(np.abs(x)))
        if 0.001 < peak < 0.45:
            x = x * min(8.0, 0.45 / peak)
        elif peak > 1.0:
            x = x / peak
        x = np.clip(x, -1.0, 1.0)
        min_samples = 8000  # MIN_ASR_INPUT_SECONDS = 0.5s
        if x.shape[0] < min_samples:
            x = np.pad(x, (0, min_samples - x.shape[0]), mode="constant", constant_values=0.0).astype(np.float32)
        elif pad_sec > 0.0:
            pad_samples = int(round(pad_sec * 16000))
            if pad_samples > 0:
                x = np.concatenate([x, np.zeros((pad_samples,), dtype=np.float32)], axis=0)
        return x

    def streaming_transcribe(self, pcm16k: np.ndarray, state: MLXStreamingState) -> MLXStreamingState:
        x = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        max_window_samples = int(MAX_STREAMING_AUDIO_WINDOW_SEC * 16000)

        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]

            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

            if state.audio_accum.shape[0] > max_window_samples:
                state.audio_accum = state.audio_accum[-max_window_samples:]

            if float(np.sqrt(np.mean(state.audio_accum**2))) < 0.0008:
                continue

            prev_text = state.text
            prev_tokens = list(state.decoded_tokens)

            # Official qwen_asr prefix rollback
            use_prefix = self._clean_prefix_tokens(state)

            eval_audio = self._prepare_eval_audio(state.audio_accum, pad_sec=0.0)
            state.decoded_tokens = self._decode_step_with_prefix(
                audio_samples=eval_audio,
                language=state.language,
                context=state.context,
                prefix_tokens=use_prefix,
                max_new_tokens=96,
            )
            setattr(state, "_last_decoded_samples", int(state.audio_accum.shape[0]))
            new_text = self.tokenizer.decode(state.decoded_tokens, skip_special_tokens=True).strip()
            if new_text:
                state.text = new_text
            elif prev_text:
                state.text = prev_text
                state.decoded_tokens = prev_tokens
            state.chunk_id += 1

        return state

    def finish_streaming_transcribe(self, state: MLXStreamingState) -> MLXStreamingState:
        valley_truncated = getattr(state, "_valley_truncated", False)
        prev_best = "" if valley_truncated else (state.text or "")
        last_decoded_samples = getattr(state, "_last_decoded_samples", 0)
        has_tail = state.buffer is not None and state.buffer.shape[0] > 0
        if has_tail:
            tail = state.buffer
            state.buffer = np.zeros((0,), dtype=np.float32)
            if state.audio_accum.shape[0] == 0:
                state.audio_accum = tail
            else:
                state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

        needs_final_decode = (
            valley_truncated
            or (state.audio_accum.shape[0] != last_decoded_samples)
            or (not prev_best)
        )
        if state.audio_accum.shape[0] < 3200 or not needs_final_decode:
            if prev_best and not state.text:
                state.text = prev_best
            state.audio_accum = np.zeros((0,), dtype=np.float32)
            state.buffer = np.zeros((0,), dtype=np.float32)
            setattr(state, "_valley_truncated", False)
            mx.clear_cache()
            gc.collect()
            return state

        max_window_samples = int(MAX_STREAMING_AUDIO_WINDOW_SEC * 16000)
        if state.audio_accum.shape[0] > max_window_samples:
            state.audio_accum = state.audio_accum[-max_window_samples:]

        eval_audio = self._prepare_eval_audio(state.audio_accum, pad_sec=0.0)
        state.decoded_tokens = self._decode_step_with_prefix(
            audio_samples=eval_audio,
            language=state.language,
            context=state.context,
            prefix_tokens=[],
            max_new_tokens=224,
        )
        new_text = self.tokenizer.decode(state.decoded_tokens, skip_special_tokens=True).strip()
        if new_text:
            state.text = new_text
        elif prev_best:
            state.text = prev_best
        state.chunk_id += 1
        state.audio_accum = np.zeros((0,), dtype=np.float32)
        state.buffer = np.zeros((0,), dtype=np.float32)
        setattr(state, "_valley_truncated", False)
        mx.clear_cache()
        gc.collect()
        return state
