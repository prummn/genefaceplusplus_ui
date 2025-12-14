import os
import sys
import subprocess
import time
import shutil
import re
from pydub import AudioSegment
from datetime import datetime

# 尝试引入视频生成模块 (可选)
try:
    from backend.video_generator import generate_video
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from backend.video_generator import generate_video

# --- 1. 路径配置 ---
# 此时位于 backend/ 目录
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)  # 项目根目录

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
# 假设 CosyVoice 模型在 CosyVoice/pretrained_models (根据实际情况调整)
COSYVOICE_MODELS_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models") 

# 关键文件路径
PIPELINE_SCRIPT = os.path.join(BACKEND_DIR, "llm_asr_pipeline.py") # 【改名后的脚本】
LATEST_RESPONSE_FILE = os.path.join(IO_HISTORY, "latest_ai_response.txt")
CHAT_HISTORY_FILE = os.path.join(IO_HISTORY, "chat_history.json")
INPUT_AUDIO_PATH = os.path.join(BASE_DIR, "io", "history", "aud.wav")

# 最终输出配置
FINAL_AUDIO_NAME = "cloned_output.wav"
FINAL_AUDIO_PATH_SERVER = os.path.join(IO_OUTPUT, FINAL_AUDIO_NAME)
FINAL_AUDIO_PATH_WEB = f"static/audios/{FINAL_AUDIO_NAME}"

PYTHON_EXECUTABLE = sys.executable

def simple_splitter(text, max_len=45):
    """简单文本切分器"""
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
    """清除历史记录"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
            print("[History] 已删除对话历史文件")
        return True, "对话记忆已清除"
    except Exception as e:
        print(f"[History] 清除失败: {e}")
        return False, f"清除失败: {e}"

def chat_response(data):
    """
    流程: ASR -> LLM -> (RVC or CosyVoice) -> [GeneFace]
    """
    print("[chat_engine] 收到请求...")

    # --- 参数解析 ---
    voice_choice = data.get('voice_clone', 'zhb')  # 例如: "zhb"
    voice_model_type = data.get('voice_clone_model', 'RVC') # "RVC" 或 "CosyVoice" (需前端传参)
    llm_model = data.get('api_choice', 'glm-4.5-flash')
    model_name = data.get('model_name', 'SyncTalk')

    # 1. 确定参考音频和参考文本路径
    # 假设前端传来的 voice_choice 是文件名(不含扩展名)，如 "zhb"
    # 音频: io/input/audio/zhb.wav
    ref_audio_path_host = os.path.join(IO_INPUT_AUDIO, f"{voice_choice}.wav")
    # 文本: io/input/text/zhb.txt (CosyVoice 可能需要参考文本)
    ref_text_path_host = os.path.join(IO_INPUT_TEXT, f"{voice_choice}.txt")

    if not os.path.exists(ref_audio_path_host):
        print(f"[Warn] 参考音频不存在: {ref_audio_path_host}，回退默认")
        # 这里你可以设置一个默认的回退逻辑
    
    # CosyVoice 需要参考文本内容，如果不存在则可能需要处理
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
            "--input", INPUT_AUDIO_PATH,
            "--model", llm_model
        ]
        
        # 运行并捕获输出
        result = subprocess.run(cmd_pipeline, check=True, cwd=BACKEND_DIR, capture_output=True, text=True, encoding='gbk')
        
        # 提取用户提问用于日志显示
        user_q = "未知"
        if result.stdout:
            for line in result.stdout.splitlines():
                if "[ASR] 识别结果:" in line:
                    user_q = line.split(":", 1)[1].strip()
        print(f"[chat_engine] 用户提问: {user_q}")

        # 读取 LLM 回复
        if not os.path.exists(LATEST_RESPONSE_FILE):
            raise FileNotFoundError("LLM 回复文件未生成")
        
        with open(LATEST_RESPONSE_FILE, 'r', encoding='utf-8') as f:
            ai_text = f.read().strip()
        print(f"[chat_engine] AI 回复: {ai_text[:30]}...")

    except subprocess.CalledProcessError as e:
        print(f"Pipeline Error: {e.stderr}")
        raise e

    # -----------------------------------------------
    # 步骤 2: 语音克隆 (分块处理)
    # -----------------------------------------------
    chunks = simple_splitter(ai_text)
    chunk_audio_files = []

    print(f"[TTS] 开始语音合成，共 {len(chunks)} 块...")

    for i, chunk_text in enumerate(chunks):
        if not chunk_text.strip(): continue

        # 2.1 准备临时文件 (Host 路径)
        chunk_text_filename = f"chunk_{i}.txt"
        chunk_wav_filename = f"chunk_{i}.wav"
        
        host_chunk_text_path = os.path.join(IO_TEMP, chunk_text_filename)
        host_chunk_wav_path = os.path.join(IO_TEMP, chunk_wav_filename)
        
        # 写入当前块的文本
        with open(host_chunk_text_path, 'w', encoding='utf-8') as f:
            f.write(chunk_text)

        # 2.2 构造 Docker 命令
        # 注意：我们将宿主机的 IO_DIR 挂载到容器的 /io
        # 容器内路径映射:
        # 参考音频: /io/input/audio/{name}.wav
        # 待合成文本: /io/temp/chunk_{i}.txt
        # 输出音频: /io/temp/chunk_{i}.wav
        
        docker_ref_audio = f"/io/input/audio/{os.path.basename(ref_audio_path_host)}"
        docker_target_text = f"/io/temp/{chunk_text_filename}"
        docker_out_wav = f"/io/temp/{chunk_wav_filename}"

        if voice_model_type == "RVC":
            # --- RVC Docker 命令 ---
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{IO_DIR}:/io",                   # 挂载整个 io 目录
                "-v", f"{RVC_MODELS_DIR}:/app/models_zh",# RVC 模型
                "rvc-app",                               # RVC 镜像名
                "--ref", docker_ref_audio,
                "--text-file", docker_target_text,
                "--out", docker_out_wav
            ]
        
        elif voice_model_type == "CosyVoice":
            # --- CosyVoice Docker 命令 (假设) ---
            # 这里的 ref_text_content 通常作为 prompt 传递，或者也写入文件
            # 假设 CosyVoice 脚本接收 --ref-text 参数
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{IO_DIR}:/io",
                "-v", f"{COSYVOICE_MODELS_DIR}:/app/pretrained_models",
                "cosyvoice-app",                         # CosyVoice 镜像名
                "--ref-audio", docker_ref_audio,
                "--ref-text", ref_text_content,          # 传递参考文本内容
                "--target-text-file", docker_target_text,
                "--out", docker_out_wav
            ]
        else:
            raise ValueError(f"未知的语音模型: {voice_model_type}")

        # 2.3 执行 Docker
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

    # 清理 temp 目录下的文件
    for f in os.listdir(IO_TEMP):
        try:
            os.remove(os.path.join(IO_TEMP, f))
        except: pass

    # -----------------------------------------------
    # 步骤 4: 视频生成 (GeneFace++) 或直接返回
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