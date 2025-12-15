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

    ref_audio_path_host = os.path.join(IO_INPUT_AUDIO, f"{voice_choice}.wav")
    ref_text_path_host = os.path.join(IO_INPUT_TEXT, f"{voice_choice}.txt")

    if not os.path.exists(ref_audio_path_host):
        print(f"[Warn] 参考音频不存在: {ref_audio_path_host}")
    
    ref_text_content = ""
    if os.path.exists(ref_text_path_host):
        with open(ref_text_path_host, 'r', encoding='utf-8') as f:
            ref_text_content = f.read().strip()
    
    print(f"[Info] 语音模型: {voice_model_type} | 参考音频: {os.path.basename(ref_audio_path_host)}")

    # -----------------------------------------------
    # 步骤 1: 运行 Pipeline (ASR + LLM)
    # -----------------------------------------------
    try:
        cmd_pipeline = [
            PYTHON_EXECUTABLE,
            PIPELINE_SCRIPT,
            "--input", user_audio_path,
            "--model", llm_model
        ]
        
        # 【修改】开启 errors='replace' 防止编码错误，并打印输出以便调试
        result = subprocess.run(cmd_pipeline, check=True, cwd=BACKEND_DIR, capture_output=True, text=True, encoding='gbk', errors='replace')
        
        # 调试：打印 Pipeline 的完整输出，方便定位 ASR 错误
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
        

        # 读取 LLM 回复
        if not os.path.exists(LATEST_RESPONSE_FILE):
            raise FileNotFoundError("LLM 回复文件未生成，Pipeline 可能运行失败")
        
        with open(LATEST_RESPONSE_FILE, 'r', encoding='utf-8') as f:
            ai_text = f.read().strip()

    except subprocess.CalledProcessError as e:
        print(f"Pipeline Error (Exit Code {e.returncode}):")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise e

    # -----------------------------------------------
    # 步骤 2: 语音克隆
    # -----------------------------------------------
    chunks = simple_splitter(ai_text)
    chunk_audio_files = []

    print(f"[TTS] 开始语音合成，共 {len(chunks)} 块...")

    for i, chunk_text in enumerate(chunks):
        if not chunk_text.strip(): continue

        chunk_text_filename = f"chunk_{i}.txt"
        chunk_wav_filename = f"chunk_{i}.wav"
        
        host_chunk_text_path = os.path.join(IO_TEMP, chunk_text_filename)
        host_chunk_wav_path = os.path.join(IO_TEMP, chunk_wav_filename)
        
        with open(host_chunk_text_path, 'w', encoding='utf-8') as f:
            f.write(chunk_text)

        docker_ref_audio = f"/io/input/audio/{os.path.basename(ref_audio_path_host)}"
        docker_target_text = f"/io/temp/{chunk_text_filename}"
        docker_out_wav = f"/io/temp/{chunk_wav_filename}"

        if voice_model_type == "RVC":
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{IO_DIR}:/io",
                "-v", f"{RVC_MODELS_DIR}:/app/models_zh",
                "rvc-app",
                "--ref", docker_ref_audio,
                "--text-file", docker_target_text,
                "--out", docker_out_wav
            ]
        
        elif voice_model_type == "CosyVoice":
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{IO_DIR}:/io",
                "-v", f"{COSYVOICE_MODELS_DIR}:/app/pretrained_models",
                "cosyvoice-app",
                "--ref-audio", docker_ref_audio,
                "--ref-text", ref_text_content,
                "--target-text-file", docker_target_text,
                "--out", docker_out_wav
            ]
        else:
            raise ValueError(f"未知的语音模型: {voice_model_type}")

        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, cwd=BASE_DIR, capture_output=True, text=True, encoding='utf-8')
            if os.path.exists(host_chunk_wav_path):
                chunk_audio_files.append(host_chunk_wav_path)
                print(f"  [Chunk {i+1}] ✅ ({time.time()-t0:.2f}s)")
            else:
                print(f"  [Chunk {i+1}] ❌ 生成失败")
        except subprocess.CalledProcessError as e:
            print(f"  [Chunk {i+1}] Docker Error: {e.stderr}")

    # -----------------------------------------------
    # 步骤 3: 拼接音频
    # -----------------------------------------------
    if not chunk_audio_files:
        raise Exception("语音合成失败: 没有生成任何音频块")

    final_audio = AudioSegment.empty()
    for f in chunk_audio_files:
        try:
            final_audio += AudioSegment.from_wav(f)
        except Exception:
            pass

    final_audio.export(FINAL_AUDIO_PATH_SERVER, format="wav")
    print(f"[chat_engine] 最终音频已保存: {FINAL_AUDIO_PATH_SERVER}")

    for f in os.listdir(IO_TEMP):
        try:
            os.remove(os.path.join(IO_TEMP, f))
        except: pass

    # -----------------------------------------------
    # 步骤 4: 视频生成 (GeneFace++)
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