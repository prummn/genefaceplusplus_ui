import os
import sys
import subprocess
import time
import shutil
import re
from pydub import AudioSegment

try:
    from backend.video_generator import generate_video
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from backend.video_generator import generate_video

# --- 1. 路径配置 ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR) 

# 核心文件夹
IO_DIR = os.path.join(BASE_DIR, "io")
RVC_DIR = os.path.join(BASE_DIR, "RVC")
COSYVOICE_DIR = os.path.join(BASE_DIR, "CosyVoice")

# IO 子目录
IO_INPUT_AUDIO = os.path.join(IO_DIR, "input", "audio")
IO_INPUT_TEXT = os.path.join(IO_DIR, "input", "text")
IO_HISTORY = os.path.join(IO_DIR, "history")
IO_TEMP = os.path.join(IO_DIR, "temp")
IO_OUTPUT = os.path.join(IO_DIR, "output")

# 确保文件夹存在
for path in [IO_INPUT_AUDIO, IO_INPUT_TEXT, IO_HISTORY, IO_TEMP, IO_OUTPUT]:
    os.makedirs(path, exist_ok=True)

# 模型路径
RVC_MODELS_DIR = os.path.join(RVC_DIR, "models_zh")
COSYVOICE_MODELS_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models") 

# 关键文件路径
PIPELINE_SCRIPT = os.path.join(BACKEND_DIR, "llm_asr_pipeline.py")
LATEST_RESPONSE_FILE = os.path.join(IO_HISTORY, "latest_ai_response.txt")
CHAT_HISTORY_FILE = os.path.join(IO_HISTORY, "chat_history.json")

# 用户录音文件搜索路径
POSSIBLE_USER_AUDIO_PATHS = [
    os.path.join(IO_HISTORY, "latest_user_aud.wav"), 
    os.path.join(BASE_DIR, "SyncTalk", "audio", "aud.wav")
]

FINAL_AUDIO_NAME = "cloned_output.wav"
FINAL_AUDIO_PATH_SERVER = os.path.join(IO_OUTPUT, FINAL_AUDIO_NAME)
FINAL_AUDIO_PATH_WEB = f"static/audios/{FINAL_AUDIO_NAME}"

PYTHON_EXECUTABLE = sys.executable

def get_actual_user_audio_path():
    for path in POSSIBLE_USER_AUDIO_PATHS:
        if os.path.exists(path):
            return path
    return None

def simple_splitter(text, max_len=45):
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
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
            print("[History] 已删除对话历史文件")
        return True, "对话记忆已清除"
    except Exception as e:
        print(f"[History] 清除失败: {e}")
        return False, f"清除失败: {e}"

def chat_response(data):
    print("[chat_engine] 收到请求...")

    user_audio_path = get_actual_user_audio_path()
    if not user_audio_path:
        raise FileNotFoundError(f"找不到用户录音文件，请检查 app.py 保存路径。搜索路径: {POSSIBLE_USER_AUDIO_PATHS}")
    
    print(f"[chat_engine] 使用录音文件: {user_audio_path}")

    # --- 参数解析 ---
    voice_choice = data.get('voice_clone', 'zhb')
    voice_model_type = data.get('voice_clone_model', 'RVC')
    llm_model = data.get('api_choice', 'glm-4.5-flash')
    model_name = data.get('model_name', 'SyncTalk')

    # 修复：处理 voice_choice 可能包含或不包含扩展名的情况
    if os.path.splitext(voice_choice)[1]:
        ref_audio_path_host = os.path.join(IO_INPUT_AUDIO, voice_choice)
    else:
        ref_audio_path_host = os.path.join(IO_INPUT_AUDIO, f"{voice_choice}.wav")

    # 对应的文本文件 (假设同名 .txt)
    ref_text_filename = os.path.splitext(os.path.basename(ref_audio_path_host))[0] + ".txt"
    ref_text_path_host = os.path.join(IO_INPUT_TEXT, ref_text_filename)

    if not os.path.exists(ref_audio_path_host):
        print(f"[Warn] 参考音频不存在: {ref_audio_path_host}")
    
    # -----------------------------------------------
    # 【新增】检测并生成参考音频文本
    # -----------------------------------------------
    if not os.path.exists(ref_text_path_host):
        print(f"[Info] 检测到参考音频文本缺失: {ref_text_path_host}")
        if os.path.exists(ref_audio_path_host):
            print(f"[Info] 正在调用 ASR 为参考音频生成字幕...")
            try:
                cmd_asr = [
                    PYTHON_EXECUTABLE,
                    PIPELINE_SCRIPT,
                    "--mode", "asr",
                    "--input", ref_audio_path_host,
                    "--output_file", ref_text_path_host
                ]
                # 运行 ASR 模式
                subprocess.run(cmd_asr, check=True, cwd=BACKEND_DIR, capture_output=True, text=True, encoding='gbk', errors='replace')
                print(f"[Info] 参考音频字幕生成成功。")
            except subprocess.CalledProcessError as e:
                print(f"[Warn] 参考音频 ASR 失败: {e.stderr}")
        else:
            print(f"[Warn] 无法生成字幕，因为参考音频文件也不存在。")

    # 读取参考文本
    ref_text_content = ""
    if os.path.exists(ref_text_path_host):
        with open(ref_text_path_host, 'r', encoding='utf-8') as f:
            ref_text_content = f.read().strip()
    
    print(f"[Info] 语音模型: {voice_model_type} | 参考音频: {os.path.basename(ref_audio_path_host)}")


    # -----------------------------------------------
    # 步骤 1: 运行 Pipeline (ASR + LLM)
    # -----------------------------------------------
    try:
        # 清理旧的回复文件，防止读取到历史数据
        if os.path.exists(LATEST_RESPONSE_FILE):
            try:
                os.remove(LATEST_RESPONSE_FILE)
            except Exception:
                pass

        cmd_pipeline = [
            PYTHON_EXECUTABLE,
            PIPELINE_SCRIPT,
            "--input", user_audio_path,
            "--model", llm_model
        ]
        
        # 设置环境变量，强制 Python 子进程输出 UTF-8，避免 Windows GBK 问题
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        print(f"[chat_engine] 启动 Pipeline: {' '.join(cmd_pipeline)}")

        # 使用 utf-8 解码，配合 PYTHONIOENCODING
        result = subprocess.run(
            cmd_pipeline,
            check=True,
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )

        # 调试：打印 Pipeline 的完整输出
        if result.stdout:
            print("============= ASR/LLM Pipeline Log =============")
            print(result.stdout)
            
        # 提取用户提问
        user_q = "未知"
        if result.stdout:
            for line in result.stdout.splitlines():
                if "[ASR] 识别结果:" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        user_q = parts[1].strip()
        
        # -----------------------------------------------
        # 步骤 2: 检查生成的音频
        # -----------------------------------------------
        if not os.path.exists(FINAL_AUDIO_PATH_SERVER):
            raise FileNotFoundError(f"TTS 生成失败，未找到音频文件: {FINAL_AUDIO_PATH_SERVER}")

        print(f"[chat_engine] 最终音频已保存: {FINAL_AUDIO_PATH_SERVER}")

        # -----------------------------------------------
        # 步骤 3: (新增) 视频生成逻辑
        # -----------------------------------------------
        # 检查是否启用了 GeneFace++ 并且提供了必要的 Checkpoints
        gf_torso_ckpt = data.get('gf_torso_ckpt')
        gf_head_ckpt = data.get('gf_head_ckpt')

        if model_name == "GeneFace++" and gf_torso_ckpt:
            print("[chat_engine] 检测到 GeneFace++ 配置，开始生成视频...")

            # 3.1 复制音频到 GeneFace 数据目录
            # GeneFace 需要音频在 data/raw/val_wavs/ 下
            gf_audio_dir = os.path.join(BASE_DIR, "GeneFace", "data", "raw", "val_wavs")
            os.makedirs(gf_audio_dir, exist_ok=True)

            # 使用时间戳命名防止冲突
            timestamp = int(time.time())
            gf_audio_name = f"chat_{timestamp}.wav"
            gf_audio_path = os.path.join(gf_audio_dir, gf_audio_name)

            shutil.copy2(FINAL_AUDIO_PATH_SERVER, gf_audio_path)
            print(f"[chat_engine] 音频已复制到: {gf_audio_path}")

            # 3.2 构造视频生成参数
            # 注意：GeneFace API 期望的 audio_path 是相对于 GeneFace 根目录或 data 目录的
            # 这里我们传递相对路径 data/raw/val_wavs/xxx.wav
            video_gen_data = {
                "model_name": "GeneFace++",
                "gf_head_ckpt": gf_head_ckpt,
                "gf_torso_ckpt": gf_torso_ckpt,
                "gf_audio_path": f"data/raw/val_wavs/{gf_audio_name}",
                "gpu_choice": data.get('gpu_choice', 'GPU0'),
                # 不需要 target_text 和 voice_clone，因为我们直接提供音频
            }

            # 3.3 调用视频生成
            # generate_video 返回的是 web 路径 (static/videos/...)
            video_web_path = generate_video(video_gen_data)

            if video_web_path:
                print(f"[chat_engine] 视频生成成功: {video_web_path}")
                return video_web_path
            else:
                print("[chat_engine] 视频生成失败，回退到仅音频")

        # 如果没有生成视频，返回音频路径 (Web 路径)
        return FINAL_AUDIO_PATH_WEB

    except subprocess.CalledProcessError as e:
        print(f"[Error] Pipeline 运行失败: {e}")
        if e.stderr:
            print(f"[Error Log] {e.stderr}")
        raise e
    except Exception as e:
        print(f"[Error] 未知错误: {e}")
        raise e
