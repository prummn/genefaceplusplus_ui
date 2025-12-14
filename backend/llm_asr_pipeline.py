import argparse
import os
import time
import json
import requests
from zhipuai import ZhipuAI
from pydub import AudioSegment 
from dotenv import load_dotenv

# --- 1. 路径配置 ---
# 当前脚本位于 backend/ 目录
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR) # 项目根目录

# IO 目录结构
IO_DIR = os.path.join(BASE_DIR, "io")
HISTORY_DIR = os.path.join(IO_DIR, "history")

# 确保 history 目录存在
os.makedirs(HISTORY_DIR, exist_ok=True)

# 文件路径定义
LATEST_RESPONSE_FILE_PATH = os.path.join(HISTORY_DIR, "latest_ai_response.txt")
CHAT_HISTORY_FILE_PATH = os.path.join(HISTORY_DIR, "chat_history.json")
LOG_FILE_PATH = os.path.join(HISTORY_DIR, "conversation_log.txt")

# 加载根目录下的 .env 文件
load_dotenv(os.path.join(BASE_DIR, ".env")) 

# --- 2. API 配置 ---
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://142790.xyz")

SYSTEM_PROMPT = """你是一名实时语音对话助手。你的所有输出必须遵守以下对话策略：
1. 输出长度控制：单次回答不得超过 100 字。禁止长篇大段输出。
2. 对话风格：语气自然、口语化，贴近人类交谈方式。回答应简洁、清晰、直接。
3. 普通聊天场景：面对日常问题，用短句、轻松自然的方式回答。
4. 专业性/解释性问题：用“分段式讲解”，每轮只解释一个概念。
5. 多轮互动优先：始终优先保持对话节奏。"""

# 初始化客户端
zhipu_client = None
if ZHIPU_API_KEY:
    try:
        zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY)
    except Exception as e:
        print(f"[Config] Zhipu Client 初始化失败: {e}")

# --- 3. 功能函数 ---

def load_history():
    """从 io/history/chat_history.json 加载历史"""
    if os.path.exists(CHAT_HISTORY_FILE_PATH):
        try:
            with open(CHAT_HISTORY_FILE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    """保存历史到 io/history/chat_history.json"""
    try:
        if len(history) > 40: 
            history = history[-40:]
        with open(CHAT_HISTORY_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[History] 保存失败: {e}")

def append_to_log(user_text, assistant_reply, model_name):
    """追加日志到 io/history/conversation_log.txt"""
    try:
        user_text_val = user_text if user_text else "N/A (ASR失败)"
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(f"【{time.ctime()} | Model: {model_name}】\n")
            f.write(f"你: {user_text_val}\n")
            f.write(f"AI: {assistant_reply}\n\n")
    except Exception as e:
        print(f"[Log] 追加日志失败: {e}")

def convert_audio_if_needed(audio_file_path):
    """转换音频格式"""
    file_name, _ = os.path.splitext(audio_file_path)
    if not os.path.exists(audio_file_path):
        return None
    output_path = file_name + "_mono.mp3"
    try:
        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_channels(1)
        audio.export(output_path, format="mp3")
        return output_path
    except Exception as e:
        print(f"[Converter] 转换失败: {e}")
        return None

def transcribe_audio_zhipu(audio_file_path):
    """ASR 识别"""
    if not audio_file_path or not zhipu_client:
        return None
    print(f"[ASR] 正在识别: {os.path.basename(audio_file_path)}...")
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = zhipu_client.audio.transcriptions.create(
                model="glm-asr",
                file=audio_file,
            )
        print(f"[ASR] 识别结果: {response.text}")
        return response.text
    except Exception as e:
        print(f"[ASR] 识别失败: {e}")
        return None

def get_llm_response_zhipu(user_text, model_name, history):
    """GLM 调用"""
    if not zhipu_client: return "错误: 智谱 API 未配置。"
    print(f"[LLM] 调用 Zhipu ({model_name})...")
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    
    try:
        response = zhipu_client.chat.completions.create(model=model_name, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Zhipu Error] {e}")
        return f"智谱API出错: {e}"

def get_llm_response_gemini(user_text, model_name, history):
    """Gemini 调用"""
    if not GEMINI_API_KEY: return "错误: Gemini API 未配置。"
    print(f"[LLM] 调用 Gemini ({model_name})...")
    
    url = f"{GEMINI_BASE_URL}/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    
    contents = []
    for item in history:
        role = "user" if item["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": item["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192}
    }
    
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=60)
        if response.status_code == 200:
            result = response.json()
            try:
                return result['candidates'][0]['content']['parts'][0]['text']
            except:
                return "Gemini 返回解析失败。"
        else:
            return f"Gemini Error: {response.status_code}"
    except Exception as e:
        return f"Gemini 连接失败: {e}"

# --- 主入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="输入音频路径")
    parser.add_argument("--model", "-m", default="glm-4.5-flash", help="LLM模型名称")
    args = parser.parse_args()
    
    # 1. 加载上下文
    history = load_history()
    
    # 2. ASR
    if os.path.exists(args.input):
        converted_path = convert_audio_if_needed(args.input)
        user_text = transcribe_audio_zhipu(converted_path)
    else:
        user_text = None

    # 3. LLM
    assistant_reply = ""
    if user_text:
        if "gemini" in args.model.lower():
            assistant_reply = get_llm_response_gemini(user_text, args.model, history)
        else:
            assistant_reply = get_llm_response_zhipu(user_text, args.model, history)
    else:
        assistant_reply = "抱歉，我没有听清。"

    print(f"[Main] AI 回答: {assistant_reply}")
    
    # 4. 保存最新回复 (供 chat_engine 读取)
    with open(LATEST_RESPONSE_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(assistant_reply)
        
    # 5. 更新历史和日志
    if user_text:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_reply})
        save_history(history)
        append_to_log(user_text, assistant_reply, args.model)