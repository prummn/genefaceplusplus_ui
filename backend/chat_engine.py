import os
import sys
import subprocess
from datetime import datetime
import re
from pydub import AudioSegment

# --- 1. 基础路径配置 (基于您的项目结构) ---
# 假设 chat_engine.py 位于 /genefaceplusplus_ui/backend/
# BASE_DIR 将会是 /genefaceplusplus_ui/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- 2. 关键目录定义 ---
# RVC 相关目录
RVC_DIR = os.path.join(BASE_DIR, "RVC")
RVC_IO_DIR = os.path.join(RVC_DIR, "io")
RVC_MODELS_DIR = os.path.join(RVC_DIR, "models_zh")

# 输入文件路径 (Host)
# 您的描述: RVC脚本的输入音频和文本都在input文件夹内
HOST_REF_AUDIO = os.path.join(RVC_IO_DIR, "input", "zhb.wav") # 或 nahida.wav，请根据实际文件名修改

# 输出目录 (Host)
# 您的描述: 把音频输出在根目录下的static中的audios文件夹内
STATIC_AUDIO_DIR = os.path.join(BASE_DIR, "static", "audios")

# 确保输出目录存在
os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)

# 中间文件
INTERMEDIATE_TEXT_FILE = os.path.join(RVC_DIR, "latest_ai_response.txt")

# 最终拼接后的文件路径
FINAL_AUDIO_NAME = "cloned_output.wav"
FINAL_AUDIO_PATH_SERVER = os.path.join(STATIC_AUDIO_DIR, FINAL_AUDIO_NAME)
FINAL_AUDIO_PATH_WEB = f"/static/audios/{FINAL_AUDIO_NAME}"

# 临时文本文件 (Host)
# 技巧: 我们把临时文本放在 RVC/io/input 下，这样它可以通过 /io 挂载被 Docker 读取
HOST_TEMP_TEXT_FILE = os.path.join(RVC_IO_DIR, "input", "temp_rvc_text.txt")

# 脚本路径 (用于 ASR/LLM)
CHINA_PIPELINE_SCRIPT = os.path.join(RVC_DIR, "china_pipeline.py")
# 输入音频 (用于 ASR) - 这里假设是 SyncTalk 的录音
INPUT_AUDIO_PATH = os.path.join(BASE_DIR, "SyncTalk", "audio", "aud.wav")
PYTHON_EXECUTABLE = sys.executable 


def simple_splitter(text, max_len=50):
    """简单文本切分器 (保持不变)"""
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
    运行 china_pipeline (本地) -> Docker RVC (容器) -> 拼接 (本地)
    """
    print("[chat_engine] 收到请求，开始处理...")

    # 检查必要的输入
    if not os.path.exists(INPUT_AUDIO_PATH):
        raise FileNotFoundError(f"Input audio not found at {INPUT_AUDIO_PATH}")
    
    # 检查 models_zh 是否存在 (Docker 运行的关键)
    if not os.path.exists(RVC_MODELS_DIR):
        raise FileNotFoundError(f"RVC Models dir not found at {RVC_MODELS_DIR}")

    chunk_audio_files = [] 
    
    try:
        # -----------------------------------------------
        # 步骤 1: 运行 ASR + LLM (使用本地环境运行 china_pipeline.py)
        # -----------------------------------------------
        print(f"[chat_engine] 1. 运行 ASR+LLM...")
        cmd_pipeline = [PYTHON_EXECUTABLE, CHINA_PIPELINE_SCRIPT, "--input", INPUT_AUDIO_PATH]
        subprocess.run(cmd_pipeline, check=True, cwd=RVC_DIR, capture_output=True, text=True, encoding='gbk') # 注意 cwd 设为 RVC_DIR

        # 读取生成的文本
        if not os.path.exists(INTERMEDIATE_TEXT_FILE):
            raise FileNotFoundError("china_pipeline.py failed to create output text")

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
            
            print(f"  Processing chunk {i+1}/{len(text_chunks)}...")
            
            # a. 写入临时文本 (Host路径: RVC/io/input/temp_rvc_text.txt)
            with open(HOST_TEMP_TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            # b. 定义 Docker 内部路径 (Internal Paths)
            # 挂载关系: 
            # Host: RVC/io         -> Docker: /io
            # Host: static/audio   -> Docker: /output
            
            # 参考音频在 Host 的 RVC/io/input/zhb.wav -> Docker 的 /io/input/zhb.wav
            ref_name = os.path.basename(HOST_REF_AUDIO)
            docker_ref_path = f"/io/input/{ref_name}" 
            
            # 文本文件在 Host 的 RVC/io/input/temp_rvc_text.txt -> Docker 的 /io/input/temp_rvc_text.txt
            temp_text_name = os.path.basename(HOST_TEMP_TEXT_FILE)
            docker_text_path = f"/io/input/{temp_text_name}"
            
            # 输出文件 -> Docker 的 /output/chunk_i.wav
            chunk_filename = f"chunk_{i}.wav"
            docker_out_path = f"/output/{chunk_filename}"
            
            # Host 对应的输出文件路径 (用于后续拼接检查)
            host_chunk_path = os.path.join(STATIC_AUDIO_DIR, chunk_filename)

            # c. 构造 Docker 命令
            cmd_docker_rvc = [
                "docker", "run", "--rm", "--gpus", "all",
                # 挂载 IO (输入)
                "-v", f"{RVC_IO_DIR}:/io",
                # 挂载 Models (模型) - 修正为您要求的 /app/models_zh
                "-v", f"{RVC_MODELS_DIR}:/app/models_zh",
                # 挂载 Static Audio (输出) - 映射到容器的 /output
                "-v", f"{STATIC_AUDIO_DIR}:/output",
                "rvc-app", # 镜像名
                "--ref", docker_ref_path,
                "--text-file", docker_text_path,
                "--out", docker_out_path
            ]
            
            # d. 执行 Docker 命令
            subprocess.run(cmd_docker_rvc, check=True, cwd=BASE_DIR, capture_output=True, text=True, encoding='utf-8')
            
            # e. 验证生成
            if os.path.exists(host_chunk_path):
                chunk_audio_files.append(host_chunk_path)
            else:
                print(f"[chat_engine] 警告: Docker 未能生成 {host_chunk_path}")

        # -----------------------------------------------
        # 步骤 3: 拼接音频 (本地 Pydub)
        # -----------------------------------------------
        if not chunk_audio_files:
            raise Exception("RVC 未生成任何音频块")

        print(f"[chat_engine] 3. 拼接 {len(chunk_audio_files)} 个音频块...")
        final_audio = AudioSegment.empty()
        for audio_file in chunk_audio_files:
            try:
                segment = AudioSegment.from_wav(audio_file)
                final_audio += segment
            except Exception as e:
                print(f"[chat_engine] 拼接错误: {e}")

        # 导出到 static/audios/cloned_output.wav
        final_audio.export(FINAL_AUDIO_PATH_SERVER, format="wav")
        print(f"[chat_engine] 完成! 音频路径: {FINAL_AUDIO_PATH_SERVER}")

        return FINAL_AUDIO_PATH_WEB

    except subprocess.CalledProcessError as e:
        print(f"!!! 脚本执行失败 !!!")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Stderr: {e.stderr}")
        raise e
    except Exception as e:
        print(f"!!! 未知错误: {e}")
        raise e
    finally:
        # 清理临时文件
        print("[chat_engine] 清理临时文件...")
        if os.path.exists(HOST_TEMP_TEXT_FILE):
            os.remove(HOST_TEMP_TEXT_FILE)
        for f in chunk_audio_files:
            if os.path.exists(f):
                os.remove(f)