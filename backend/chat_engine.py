import os
import sys
import subprocess
from datetime import datetime
import re
from pydub import AudioSegment

# --- 1. 基础路径配置 ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# RVC 相关目录
RVC_DIR = os.path.join(BASE_DIR, "RVC")
RVC_IO_DIR = os.path.join(RVC_DIR, "io")
RVC_MODELS_DIR = os.path.join(RVC_DIR, "models_zh")

# 输出目录
STATIC_AUDIO_DIR = os.path.join(BASE_DIR, "static", "audios")
os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)

# 临时文件与脚本
INTERMEDIATE_TEXT_FILE = os.path.join(RVC_DIR, "latest_ai_response.txt")
HOST_TEMP_TEXT_FILE = os.path.join(RVC_IO_DIR, "input", "temp_rvc_text.txt")
CHINA_PIPELINE_SCRIPT = os.path.join(RVC_DIR, "china_pipeline.py")
INPUT_AUDIO_PATH = os.path.join(BASE_DIR, "SyncTalk", "audio", "aud.wav")

FINAL_AUDIO_NAME = "cloned_output.wav"
FINAL_AUDIO_PATH_SERVER = os.path.join(STATIC_AUDIO_DIR, FINAL_AUDIO_NAME)
FINAL_AUDIO_PATH_WEB = f"/static/audios/{FINAL_AUDIO_NAME}"

PYTHON_EXECUTABLE = sys.executable 


def simple_splitter(text, max_len=50):
    """简单文本切分器"""
    print(f"[Splitter] 正在切分总长 {len(text)} 的文本...")
    segments = re.findall(r"([^。！？，、,!?]+[。！？，、,!?]?)", text, re.UNICODE)
    if not segments:
        return [text[i:i+max_len] for i in range(0, len(text), max_len)]
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


def chat_response(data):
    """
    参数 data 包含前端传来的 'voice_clone' (音色) 和 'api_choice' (模型)
    """
    print("[chat_engine] 收到请求，开始处理...")
    print(f"[chat_engine] 参数数据: {data}")

    # -----------------------------------------------
    # 1. 解析前端参数
    # -----------------------------------------------
    # 获取音色选择 (默认为 zhb)
    voice_choice = data.get('voice_clone', 'zhb') 
    # 构造参考音频路径
    # 假设文件都在 RVC/io/input/ 目录下，且命名为 {voice_choice}.wav
    host_ref_audio = os.path.join(RVC_IO_DIR, "input", f"{voice_choice}.wav")
    
    if not os.path.exists(host_ref_audio):
        print(f"[chat_engine] 警告: 请求的音色文件 {host_ref_audio} 不存在，回退到 zhb.wav")
        host_ref_audio = os.path.join(RVC_IO_DIR, "input", "zhb.wav")

    # 获取 LLM 模型选择 (默认为 glm-4-flash)
    llm_model = data.get('api_choice', 'glm-4-flash')

    print(f"[chat_engine] 使用参考音色: {os.path.basename(host_ref_audio)}")
    print(f"[chat_engine] 使用 LLM 模型: {llm_model}")

    # -----------------------------------------------
    # 2. 检查基础环境
    # -----------------------------------------------
    if not os.path.exists(INPUT_AUDIO_PATH):
        raise FileNotFoundError(f"用户录音未找到: {INPUT_AUDIO_PATH}")
    
    if not os.path.exists(RVC_MODELS_DIR):
        raise FileNotFoundError(f"RVC 模型目录未找到: {RVC_MODELS_DIR}")

    chunk_audio_files = [] 
    
    try:
        # -----------------------------------------------
        # 步骤 1: 运行 ASR + LLM (传递 --model 参数)
        # -----------------------------------------------
        print(f"[chat_engine] 1. 运行 ASR+LLM (Model: {llm_model})...")
        
        cmd_pipeline = [
            PYTHON_EXECUTABLE, 
            CHINA_PIPELINE_SCRIPT, 
            "--input", INPUT_AUDIO_PATH,
            "--model", llm_model  # 【新】传递模型参数
        ]
        
        # 运行并捕获输出
        subprocess.run(cmd_pipeline, check=True, cwd=RVC_DIR, capture_output=True, text=True, encoding='gbk')

        # 读取生成的文本
        if not os.path.exists(INTERMEDIATE_TEXT_FILE):
            raise FileNotFoundError("china_pipeline.py 未能生成回复文本")

        with open(INTERMEDIATE_TEXT_FILE, 'r', encoding='utf-8') as f:
            ai_long_text = f.read().strip()
        print(f"[chat_engine] AI 回复内容: {ai_long_text[:30]}...")

        # -----------------------------------------------
        # 步骤 2: 切分文本 & Docker RVC 循环
        # -----------------------------------------------
        text_chunks = simple_splitter(ai_long_text)
        print(f"[chat_engine] 2. 启动 Docker RVC 克隆 ({len(text_chunks)} 块)...")

        for i, chunk in enumerate(text_chunks):
            if not chunk.strip(): continue
            
            # a. 写入临时文本
            with open(HOST_TEMP_TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            # b. 定义 Docker 路径映射
            # Host: RVC/io -> Docker: /io
            # 参考音频路径映射:
            ref_name = os.path.basename(host_ref_audio)
            docker_ref_path = f"/io/input/{ref_name}" 
            
            # 文本路径映射:
            temp_text_name = os.path.basename(HOST_TEMP_TEXT_FILE)
            docker_text_path = f"/io/input/{temp_text_name}"
            
            # 输出路径映射:
            chunk_filename = f"chunk_{i}.wav"
            docker_out_path = f"/output/{chunk_filename}"
            host_chunk_path = os.path.join(STATIC_AUDIO_DIR, chunk_filename)

            # c. 构造 Docker 命令
            cmd_docker_rvc = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{RVC_IO_DIR}:/io",
                "-v", f"{RVC_MODELS_DIR}:/app/models_zh",
                "-v", f"{STATIC_AUDIO_DIR}:/output",
                "rvc-app", 
                "--ref", docker_ref_path,     # 使用动态选择的参考音频
                "--text-file", docker_text_path,
                "--out", docker_out_path
            ]
            
            # d. 执行
            subprocess.run(cmd_docker_rvc, check=True, cwd=BASE_DIR, capture_output=True, text=True, encoding='utf-8')
            
            if os.path.exists(host_chunk_path):
                chunk_audio_files.append(host_chunk_path)
            else:
                print(f"[chat_engine] 警告: Docker 未能生成 {host_chunk_path}")

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
                print(f"[chat_engine] 拼接错误: {e}")

        final_audio.export(FINAL_AUDIO_PATH_SERVER, format="wav")
        print(f"[chat_engine] 完成! 音频路径: {FINAL_AUDIO_PATH_SERVER}")

        return FINAL_AUDIO_PATH_WEB

    except subprocess.CalledProcessError as e:
        print(f"!!! 脚本执行失败 !!! \nCommand: {' '.join(e.cmd)}\nStderr: {e.stderr}")
        raise e
    except Exception as e:
        print(f"!!! 未知错误: {e}")
        raise e
    finally:
        # 清理
        if os.path.exists(HOST_TEMP_TEXT_FILE):
            os.remove(HOST_TEMP_TEXT_FILE)
        for f in chunk_audio_files:
            if os.path.exists(f):
                os.remove(f)