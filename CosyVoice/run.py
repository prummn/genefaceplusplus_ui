#!/usr/bin/env python3
"""
Simple offline inference helper.

Reads a reference wav from ./io/input/audio, target text from ./io/input/text (or other),
and writes the synthesized wav to ./io/output using the local CosyVoice2-0.5B model by default.
"""
import argparse
import importlib
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
import torchaudio


# Quiet noisy warnings (keep progress bar visible).
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "4"  # hide CUDA provider load errors (falls back to CPU)
os.environ.setdefault("ORT_DISABLE_ALL_WARNINGS", "1")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchaudio._backend.utils")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")
warnings.filterwarnings("ignore", message=r"We detected that you are passing `past_key_values`.*")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent
THIRD_PARTY = ROOT_DIR / "third_party" / "Matcha-TTS"
for p in (ROOT_DIR, THIRD_PARTY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
try:
    import cosyvoice  # noqa: F401
except ImportError:
    cv_pkg = importlib.import_module("CosyVoice")
    sys.modules["cosyvoice"] = cv_pkg

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav, logging


def _read_text(text_path: Path) -> str:
    # 强制解析为绝对路径以便调试
    abs_path = text_path.resolve()

    text = abs_path.read_text(encoding="utf-8").strip()
    if text == "":
        raise ValueError(f"Text file {abs_path} is empty")
    return text


def _normalize_audio(audio: torch.Tensor, max_val: float = 0.8) -> torch.Tensor:
    peak = audio.abs().max()
    if peak > max_val:
        audio = audio / peak * max_val
    return audio


def synthesize(model: CosyVoice2, tts_text: str, prompt_wav: Path, output_wav: Path,
               prompt_text: Optional[str], stream: bool, speed: float) -> Path:
    logging.info("Loading prompt wav %s", prompt_wav)
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Reference wav not found: {prompt_wav}")
    
    prompt_audio_input = str(prompt_wav.resolve())

    logging.info("Running inference (stream=%s, speed=%.2f)", stream, speed)
    if prompt_text:
        logging.info("Using zero-shot mode with prompt text: %s", prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text)
        # 传入路径字符串，而不是 Tensor
        generator = model.inference_zero_shot(tts_text, prompt_text, prompt_audio_input, stream=stream, speed=speed)
    else:
        logging.info("Using cross-lingual mode (no prompt text provided)")
        generator = model.inference_cross_lingual(tts_text, prompt_audio_input, stream=stream, speed=speed)

    chunks = []
    for out in generator:
        chunks.append(out["tts_speech"])
    if len(chunks) == 0:
        raise RuntimeError("Model returned no audio chunks")
    speech = torch.cat(chunks, dim=1)
    speech = _normalize_audio(speech)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_wav), speech.cpu(), sample_rate=model.sample_rate)
    logging.info("Saved synthesized audio to %s", output_wav)
    return output_wav


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize speech using CosyVoice")
    
    # --- IO 路径基准设为上级目录 (ROOT_DIR.parent) ---
    IO_ROOT = ROOT_DIR.parent
    
    default_ref_wav = IO_ROOT / "io/input/audio/nahida.wav"
    default_text_path = IO_ROOT / "io/history/latest_ai_response.txt"
    default_output_wav = IO_ROOT / "io/output/output.wav"
    default_prompt_text_path = IO_ROOT / "io/input/text/nahida.txt"

    parser.add_argument("--ref_wav", type=Path, default=default_ref_wav,
                        help="Reference wav used as timbre (16 kHz or higher)")
    parser.add_argument("--text_path", type=Path, default=default_text_path,
                        help="Text file containing the target speech content (TTS input)")
    parser.add_argument("--output_wav", type=Path, default=default_output_wav,
                        help="Where to save the synthesized wav")
    parser.add_argument("--model_dir", type=str, default="CosyVoice2-0.5B",
                        help="Local path to the model directory (or repo id)")
    # 注意: argparse 默认值在这里只作为字符串传递，后续逻辑会处理
    parser.add_argument("--prompt_text", type=str, default=str(default_prompt_text_path),
                        help="(Optional) transcript of the reference audio.")
    
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--stream", action="store_true", help="Enable streaming synthesis")
    parser.add_argument("--speed", type=float, default=0.95, help="Playback speed multiplier (non-streaming only)")
    parser.add_argument("--fp16", action="store_true", help="Load model weights in fp16 when GPU is available")
    args = parser.parse_args()

    # --- 自动加载参考文本逻辑 ---
    final_prompt_text = None
    
    # 1. 检查是否传入的是文件路径
    potential_path = Path(args.prompt_text)
    if potential_path.exists() and potential_path.is_file():
         final_prompt_text = potential_path.read_text(encoding="utf-8").strip()
    elif len(args.prompt_text) > 0 and not args.prompt_text.endswith(".txt"):
        # 用户可能直接在命令行输入了提示文本，而不是路径
        final_prompt_text = args.prompt_text
    
    # 2. 如果以上都没找到，尝试自动推断
    if final_prompt_text is None and args.ref_wav:
        ref_dir = args.ref_wav.parent 
        if ref_dir.name == 'audio':
            auto_text_path = ref_dir.parent / "text" / f"{args.ref_wav.stem}.txt"
        else:
            auto_text_path = args.ref_wav.with_suffix(".txt")
            
        if auto_text_path.exists():
            try:
                final_prompt_text = auto_text_path.read_text(encoding="utf-8").strip()
                logging.info("Auto-loaded prompt text from file: %s", auto_text_path)
            except Exception as e:
                logging.warning("Failed to read auto-detected prompt text file %s: %s", auto_text_path, e)

    # ---------------------------

    set_all_random_seed(args.seed)
    model = CosyVoice2(args.model_dir, fp16=args.fp16)
    logging.info("Text frontend: %s", "ttsfrd" if getattr(model.frontend, "use_ttsfrd", False) else "wetext/identity")

    # 读取 TTS 文本
    print(f"DEBUG: 正在尝试读取文本文件: {args.text_path}")
    tts_text = _read_text(args.text_path)
    print(f"DEBUG: 读取成功，文本内容 ({len(tts_text)} 字符): {tts_text[:20]}...")

    synthesize(model=model,
               tts_text=tts_text,
               prompt_wav=args.ref_wav,
               output_wav=args.output_wav,
               prompt_text=final_prompt_text,
               stream=args.stream,
               speed=args.speed)


if __name__ == "__main__":
    main()