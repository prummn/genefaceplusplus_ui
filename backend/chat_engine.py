import os
import sys
import subprocess
import time
import shutil
from datetime import datetime
import re
from pydub import AudioSegment

# 引入视频生成模块
# 注意：需要在 backend 目录下有 __init__.py，或者确保 pythonpath 正确
try:
    from backend.video_generator import generate_video
except ImportError:
    # 如果直接运行此脚本可能报错，但在 Flask 上下文中通常没问题
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from backend.video_generator import generate_video

# --- 1. 基础路径配置 ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# RVC 相关目录
RVC_DIR = os.path.join(BASE_DIR, "RVC")
RVC_IO_DIR = os.path.join(BASE_DIR, "io")
RVC_MODELS_DIR = os.path.join(RVC_DIR, "models_zh")
HISTORY_FILE_PATH = os.path.join(RVC_DIR, "chat_history.json")

# 输出目录
AUDIO_DIR = os.path.join(BASE_DIR, "io", "output")
os.makedirs(AUDIO_DIR, exist_ok=True)

# GeneFace 音频输入目录
GENEFACE_DIR = os.path.join(BASE_DIR, "GeneFace")
GENEFACE_AUDIO_INPUT_DIR = os.path.join(GENEFACE_DIR, "data", "raw", "val_wavs")
os.makedirs(GENEFACE_AUDIO_INPUT_DIR, exist_ok=True)

# 临时文件与脚本
INTERMEDIATE_TEXT_FILE = os.path.join(RVC_DIR, "latest_ai_response.txt")
HOST_TEMP_TEXT_FILE = os.path.join(RVC_IO_DIR, "input", "temp_rvc_text.txt")
CHINA_PIPELINE_SCRIPT = os.path.join(RVC_DIR, "china_pipeline.py")
INPUT_AUDIO_PATH = os.path.join(BASE_DIR, "SyncTalk", "audio", "aud.wav")

FINAL_AUDIO_NAME = "cloned_output.wav"
FINAL_AUDIO_PATH_SERVER = os.path.join(AUDIO_DIR, FINAL_AUDIO_NAME)
FINAL_AUDIO_PATH_WEB = f"static/audios/{FINAL_AUDIO_NAME}"  # 修改为相对 static 的路径，不带前导 /

PYTHON_EXECUTABLE = sys.executable


def simple_splitter(text, max_len=45):
    """简单文本切分器"""
    # ... (保持不变) ...
    print(f"[Splitter] 正在切分总长 {len(text)} 的文本...")
    segments = re.findall(r"([^。！？，、,!?]+[。！？，、,!?]?)", text, re.UNICODE)
    if not segments:
        return [text[i:i + max_len] for i in range(0, len(text), max_len)]
    chunks = []
    current_chunk = ""
    for seg in segments:
        if len(current_chunk) + len(seg) > max_len:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = seg
            while len(current_chunk) > max_len:
                chunks.append(current_chunk[:max_len])
                current_chunk = current_chunk[max_len:]
        else:
            current_chunk += seg
    if current_chunk: chunks.append(current_chunk)
    return [c for c in chunks if c.strip()]


def clear_chat_history():
    # ... (保持不变) ...
    try:
        if os.path.exists(HISTORY_FILE_PATH):
            os.remove(HISTORY_FILE_PATH)
            print("[History] 已删除对话历史文件")
        return True, "对话记忆已清除"
    except Exception as e:
        print(f"[History] 清除失败: {e}")
        return False, f"清除失败: {e}"


def chat_response(data):
    """
    处理对话请求: ASR -> LLM (Multi-turn) -> RVC -> [Optional: GeneFace]
    """
    print("[chat_engine] 收到请求，开始处理...")

    # -----------------------------------------------
    # 1. 解析参数
    # -----------------------------------------------
    voice_choice = data.get('voice_clone', 'zhb')
    model_name = data.get('model_name', 'SyncTalk')

    host_ref_audio = os.path.join(RVC_IO_DIR, "input", f"{voice_choice}.wav")
    if not os.path.exists(host_ref_audio):
        print(f"[chat_engine] 警告: 音色 {host_ref_audio} 不存在，使用默认 zhb")
        host_ref_audio = os.path.join(RVC_IO_DIR, "input", "zhb.wav")

    llm_model = data.get('api_choice', 'glm-4.5-flash')
    print(f"[chat_engine] 音色: {os.path.basename(host_ref_audio)} | LLM: {llm_model} | 数字人: {model_name}")

    # -----------------------------------------------
    # 2. 检查环境 (保持不变)
    # -----------------------------------------------
    if not os.path.exists(INPUT_AUDIO_PATH):
        raise FileNotFoundError(f"用户录音未找到: {INPUT_AUDIO_PATH}")

    if not os.path.exists(RVC_MODELS_DIR):
        raise FileNotFoundError(f"RVC 模型目录未找到: {RVC_MODELS_DIR}")

    chunk_audio_files = []

    try:
        # -----------------------------------------------
        # 步骤 1: 运行 china_pipeline.py (集成 ASR, Gemini/GLM, 历史记录)
        # -----------------------------------------------
        print(f"[chat_engine] 1. 调用 Pipeline (多轮对话)...")

        cmd_pipeline = [
            PYTHON_EXECUTABLE,
            CHINA_PIPELINE_SCRIPT,
            "--input", INPUT_AUDIO_PATH,
            "--model", llm_model
        ]

        pipeline_result = subprocess.run(cmd_pipeline, check=True, cwd=RVC_DIR, capture_output=True, text=True,
                                         encoding='gbk')

        if pipeline_result.stdout:
            print("============= Pipeline Logs Start =============")
            print(pipeline_result.stdout)
            print("============= Pipeline Logs End ===============")

        # --- 解析用户提问 (保持不变) ---
        user_question = "（未能解析用户提问）"
        if pipeline_result.stdout:
            for line in pipeline_result.stdout.splitlines():
                if "[ASR] 识别结果:" in line:
                    parts = line.split("[ASR] 识别结果:", 1)
                    if len(parts) > 1:
                        user_question = parts[1].strip()
                    break
        print(f"[chat_engine] 用户提问: {user_question}")

        if not os.path.exists(INTERMEDIATE_TEXT_FILE):
            raise FileNotFoundError("Pipeline 未能生成回复文本")

        with open(INTERMEDIATE_TEXT_FILE, 'r', encoding='utf-8') as f:
            ai_long_text = f.read().strip()

        print(f"[chat_engine] AI 回复: {ai_long_text[:50]}...")

        # -----------------------------------------------
        # 步骤 2: RVC 语音克隆 (保持不变)
        # -----------------------------------------------
        text_chunks = simple_splitter(ai_long_text)
        print(f"[chat_engine] 2. RVC 克隆 ({len(text_chunks)} 块)...")

        for i, chunk in enumerate(text_chunks):
            if not chunk.strip(): continue

            with open(HOST_TEMP_TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write(chunk)

            ref_name = os.path.basename(host_ref_audio)
            docker_ref_path = f"/io/input/{ref_name}"
            temp_text_name = os.path.basename(HOST_TEMP_TEXT_FILE)
            docker_text_path = f"/io/input/{temp_text_name}"
            chunk_filename = f"chunk_{i}.wav"
            docker_out_path = f"/output/{chunk_filename}"
            host_chunk_path = os.path.join(AUDIO_DIR, chunk_filename)

            cmd_docker_rvc = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{RVC_IO_DIR}:/io",
                "-v", f"{RVC_MODELS_DIR}:/app/models_zh",
                "-v", f"{AUDIO_DIR}:/output",
                "rvc-app",
                "--ref", docker_ref_path,
                "--text-file", docker_text_path,
                "--out", docker_out_path
            ]

            chunk_start_time = time.time()
            subprocess.run(cmd_docker_rvc, check=True, cwd=BASE_DIR, capture_output=True, text=True, encoding='utf-8')
            duration = time.time() - chunk_start_time

            if os.path.exists(host_chunk_path):
                chunk_audio_files.append(host_chunk_path)
                print(f"[chat_engine] ✅ 第 {i + 1}/{len(text_chunks)} 块完成 | {duration:.2f}s")
            else:
                print(f"[chat_engine] ❌ 第 {i + 1}/{len(text_chunks)} 块失败 | {duration:.2f}s")

        # -----------------------------------------------
        # 步骤 3: 拼接音频
        # -----------------------------------------------
        if not chunk_audio_files:
            raise Exception("RVC 未生成任何音频块")

        print(f"[chat_engine] 3. 拼接音频...")
        final_audio = AudioSegment.empty()
        for audio_file in chunk_audio_files:
            try:
                segment = AudioSegment.from_wav(audio_file)
                final_audio += segment
            except Exception as e:
                print(f"拼接警告: {e}")

        final_audio.export(FINAL_AUDIO_PATH_SERVER, format="wav")
        print(f"[chat_engine] 音频已生成: {FINAL_AUDIO_PATH_SERVER}")

        # -----------------------------------------------
        # 步骤 4: 根据模型选择执行视频生成
        # -----------------------------------------------
        if model_name == "GeneFace++":
            print(f"[chat_engine] 4. 触发 GeneFace++ 推理...")

            # 4.1 移动/复制音频到 GeneFace 输入目录
            # 生成唯一文件名防止冲突
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gf_audio_filename = f"chat_{timestamp}.wav"
            gf_audio_path_abs = os.path.join(GENEFACE_AUDIO_INPUT_DIR, gf_audio_filename)

            shutil.copy(FINAL_AUDIO_PATH_SERVER, gf_audio_path_abs)
            print(f"[chat_engine] 音频已复制到: {gf_audio_path_abs}")

            # 4.2 构造参数调用 generate_video
            # 注意：backend/video_generator.py 需要的是相对于 WORK_DIR 的音频路径
            # 或者是它可以识别的路径。根据之前逻辑，传 'data/raw/val_wavs/filename.wav'
            gf_audio_path_rel = f"data/raw/val_wavs/{gf_audio_filename}"

            gen_data = {
                "model_name": "GeneFace++",
                "gf_head_ckpt": data.get('gf_head_ckpt'),
                "gf_torso_ckpt": data.get('gf_torso_ckpt'),
                "gf_audio_path": gf_audio_path_rel,
                "gpu_choice": "GPU0"  # 默认
            }

            video_path = generate_video(gen_data)
            print(f"[chat_engine] 视频生成完成: {video_path}")
            return video_path

        else:
            # 如果是其他模型，或者是 SyncTalk (这里暂时只返回音频，或你可以按需调用其他生成器)
            # 目前前端 chatVideo 兼容音频播放，所以返回音频路径即可
            return FINAL_AUDIO_PATH_WEB

    except subprocess.CalledProcessError as e:
        print(f"!!! Pipeline 执行失败 !!! \nCommand: {e.cmd}\nStderr: {e.stderr}")
        raise e
    except Exception as e:
        print(f"!!! 未知错误: {e}")
        raise e
    finally:
        if os.path.exists(HOST_TEMP_TEXT_FILE):
            try:
                os.remove(HOST_TEMP_TEXT_FILE)
            except:
                pass
        for f in chunk_audio_files:
            try:
                os.remove(f)
            except:
                pass