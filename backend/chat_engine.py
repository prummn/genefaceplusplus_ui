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
# RVC_IO_DIR = os.path.join(RVC_DIR, "io")
RVC_IO_DIR = os.path.join(BASE_DIR, "io")
RVC_MODELS_DIR = os.path.join(RVC_DIR, "models_zh")
# 历史记录文件
HISTORY_FILE_PATH = os.path.join(RVC_DIR, "chat_history.json")

# 输出目录
AUDIO_DIR = os.path.join(BASE_DIR, "io", "output")
os.makedirs(AUDIO_DIR, exist_ok=True)

# 临时文件与脚本
INTERMEDIATE_TEXT_FILE = os.path.join(RVC_DIR, "latest_ai_response.txt")
HOST_TEMP_TEXT_FILE = os.path.join(RVC_IO_DIR, "input", "temp_rvc_text.txt")
CHINA_PIPELINE_SCRIPT = os.path.join(RVC_DIR, "china_pipeline.py")
INPUT_AUDIO_PATH = os.path.join(BASE_DIR, "SyncTalk", "audio", "aud.wav")

FINAL_AUDIO_NAME = "cloned_output.wav"
FINAL_AUDIO_PATH_SERVER = os.path.join(AUDIO_DIR, FINAL_AUDIO_NAME)
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

def clear_chat_history():
    """清除多轮对话历史文件"""
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
    处理对话请求: ASR -> LLM (Multi-turn) -> RVC
    """
    print("[chat_engine] 收到请求，开始处理...")
    
    # -----------------------------------------------
    # 1. 解析参数
    # -----------------------------------------------
    voice_choice = data.get('voice_clone', 'zhb') 
    host_ref_audio = os.path.join(RVC_IO_DIR, "input", f"{voice_choice}.wav")
    
    if not os.path.exists(host_ref_audio):
        print(f"[chat_engine] 警告: 音色 {host_ref_audio} 不存在，使用默认 zhb")
        host_ref_audio = os.path.join(RVC_IO_DIR, "input", "zhb.wav")

    # 获取前端传来的具体模型 (gemini-2.5-pro, glm-4.5 等)
    llm_model = data.get('api_choice', 'glm-4.5-flash')

    print(f"[chat_engine] 音色: {os.path.basename(host_ref_audio)} | 模型: {llm_model}")

    # -----------------------------------------------
    # 2. 检查环境
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
        
        # 运行管线
        # 修改: 将执行结果赋值给变量，以便后续解析 stdout
        pipeline_result = subprocess.run(cmd_pipeline, check=True, cwd=RVC_DIR, capture_output=True, text=True, encoding='gbk')

        # 【新增】打印子进程的完整输出，以便在控制台看到详细的 Gemini 报错
        if pipeline_result.stdout:
            print("============= Pipeline Logs Start =============")
            print(pipeline_result.stdout)
            print("============= Pipeline Logs End ===============")

        # --- 解析用户提问 ---
        user_question = "（未能解析用户提问）"
        if pipeline_result.stdout:
            for line in pipeline_result.stdout.splitlines():
                if "[ASR] 识别结果:" in line:
                    # 提取冒号后面的文本
                    parts = line.split("[ASR] 识别结果:", 1)
                    if len(parts) > 1:
                        user_question = parts[1].strip()
                    break
        
        print(f"[chat_engine] 用户提问: {user_question}")
        # ---------------------------------------------------

        # 读取 Pipeline 生成的回复文本
        if not os.path.exists(INTERMEDIATE_TEXT_FILE):
            raise FileNotFoundError("Pipeline 未能生成回复文本")

        with open(INTERMEDIATE_TEXT_FILE, 'r', encoding='utf-8') as f:
            ai_long_text = f.read().strip()
            
        print(f"[chat_engine] AI 回复: {ai_long_text[:50]}...")

        # -----------------------------------------------
        # 步骤 2: RVC 语音克隆 (保持原逻辑不变)
        # -----------------------------------------------
        text_chunks = simple_splitter(ai_long_text)
        print(f"[chat_engine] 2. RVC 克隆 ({len(text_chunks)} 块)...")

        for i, chunk in enumerate(text_chunks):
            if not chunk.strip(): continue
            
            with open(HOST_TEMP_TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            # Docker 路径映射
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
            
            subprocess.run(cmd_docker_rvc, check=True, cwd=BASE_DIR, capture_output=True, text=True, encoding='utf-8')
            
            if os.path.exists(host_chunk_path):
                chunk_audio_files.append(host_chunk_path)

        # -----------------------------------------------
        # 步骤 3: 拼接
        # -----------------------------------------------
        if not chunk_audio_files:
            # 如果没有音频生成（可能是文本为空），生成一段静音或报错
            raise Exception("RVC 未生成任何音频")

        print(f"[chat_engine] 3. 拼接音频...")
        final_audio = AudioSegment.empty()
        for audio_file in chunk_audio_files:
            try:
                segment = AudioSegment.from_wav(audio_file)
                final_audio += segment
            except Exception as e:
                print(f"拼接警告: {e}")

        final_audio.export(FINAL_AUDIO_PATH_SERVER, format="wav")
        
        return FINAL_AUDIO_PATH_WEB

    except subprocess.CalledProcessError as e:
        print(f"!!! Pipeline 执行失败 !!! \nCommand: {e.cmd}\nStderr: {e.stderr}")
        raise e
    except Exception as e:
        print(f"!!! 未知错误: {e}")
        raise e
    finally:
        if os.path.exists(HOST_TEMP_TEXT_FILE):
            try: os.remove(HOST_TEMP_TEXT_FILE)
            except: pass
        for f in chunk_audio_files:
            try: os.remove(f)
            except: pass