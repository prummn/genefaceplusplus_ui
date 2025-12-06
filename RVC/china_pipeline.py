import argparse
import os
import time
import json
import requests
from zhipuai import ZhipuAI
from pydub import AudioSegment 
from dotenv import load_dotenv

# --- 文件路径 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载 .env 文件
load_dotenv(os.path.join(BASE_DIR, ".env")) 
load_dotenv() 

# --- 配置区域 ---
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://142790.xyz")

# --- 【新增】统一的系统提示词 (System Prompt) ---
# 核心指令：一段话回答 + 场景区分
SYSTEM_PROMPT = "你是一个智能助手。请务必用一段话回答（不要分段）。在日常对话时回复尽量简短，而在面对专业问题时再提供长内容回答。"

if not ZHIPU_API_KEY:
    print("[Config] 警告: 未找到 ZHIPU_API_KEY，请在 .env 文件中配置")
if not GEMINI_API_KEY:
    print("[Config] 警告: 未找到 GEMINI_API_KEY，请在 .env 文件中配置")

LATEST_RESPONSE_FILE = "latest_ai_response.txt"
LATEST_RESPONSE_FILE_PATH = os.path.join(BASE_DIR, LATEST_RESPONSE_FILE)
HISTORY_FILE_PATH = os.path.join(BASE_DIR, "chat_history.json")
LOG_FILE_PATH = os.path.join(BASE_DIR, "conversation_log.txt")

# 初始化 ZhipuAI 客户端
try:
    if ZHIPU_API_KEY:
        zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY)
    else:
        zhipu_client = None
        print("[Config] Zhipu Client 未初始化 (缺少 Key)")
except Exception as e:
    print(f"Zhipu API Key 初始化失败: {e}")
    zhipu_client = None

# --- 多轮对话管理 ---
def load_history():
    """从 JSON 文件加载历史记录"""
    if os.path.exists(HISTORY_FILE_PATH):
        try:
            with open(HISTORY_FILE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    """保存历史记录到 JSON 文件"""
    try:
        if len(history) > 40: 
            history = history[-40:]
        with open(HISTORY_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[History] 保存失败: {e}")

def append_to_log(user_text, assistant_reply, model_name):
    """追加到纯文本日志文件"""
    try:
        user_text_val = user_text if user_text else "N/A (ASR失败)"
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(f"【{time.ctime()} | Model: {model_name}】\n")
            f.write(f"你: {user_text_val}\n")
            f.write(f"AI: {assistant_reply}\n\n")
    except Exception as e:
        print(f"[Log] 追加日志失败: {e}")

# --- 音频处理 ---
def convert_audio_if_needed(audio_file_path):
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
    if not audio_file_path or not zhipu_client:
        return None
    print(f"[ASR] 正在识别 (Zhipu): {audio_file_path}...")
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

# --- LLM 调用逻辑 ---

def get_llm_response_zhipu(user_text, model_name, history):
    if not zhipu_client:
        return "错误: 智谱 API 未配置。"
        
    print(f"[LLM] 调用 Zhipu ({model_name})...")
    
    # 【应用系统提示词】
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for item in history:
        messages.append(item)
    messages.append({"role": "user", "content": user_text})
    
    try:
        response = zhipu_client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        reply = response.choices[0].message.content
        return reply
    except Exception as e:
        print(f"[Zhipu Error] {e}")
        return f"智谱API调用出错: {e}"

def get_llm_response_gemini(user_text, model_name, history):
    if not GEMINI_API_KEY:
        return "错误: Gemini API 未配置。"

    print(f"[LLM] 调用 Gemini ({model_name})...")
    
    api_model_name = model_name
    
    url = f"{GEMINI_BASE_URL}/v1beta/models/{api_model_name}:generateContent?key={GEMINI_API_KEY}"
    
    # 构造 Gemini 格式的内容
    contents = []
    for item in history:
        role = "user" if item["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": item["content"]}]
        })
    
    contents.append({
        "role": "user",
        "parts": [{"text": user_text}]
    })

    payload = {
        # 【应用系统提示词】Gemini 使用 system_instruction
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192, 
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        print(f"[Gemini Debug] Request URL: {url}")
        
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        
        print(f"[Gemini Debug] HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            try:
                if "promptFeedback" in result:
                    pf = result["promptFeedback"]
                    if "blockReason" in pf:
                        print(f"[Gemini Debug] PromptFeedback Blocked: {pf}")
                        return f"Gemini 拒绝回答 (Prompt Blocked: {pf.get('blockReason')})"

                if "candidates" not in result or not result["candidates"]:
                    return "Gemini API 未返回结果 (可能被安全策略拦截)。"

                candidate = result['candidates'][0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                if not parts:
                    finish_reason = candidate.get("finishReason", "UNKNOWN")
                    print(f"[Gemini Debug] Finish Reason: {finish_reason}")
                    if finish_reason == "MAX_TOKENS":
                        return "回答被截断 (Token上限不足)。"
                    elif finish_reason == "SAFETY":
                        return "回答被安全策略拦截。"
                    else:
                        return f"Gemini 生成内容为空 (原因: {finish_reason})"

                text = parts[0].get('text', '')
                if not text:
                     return "Gemini 返回了空的文本内容。"
                     
                return text
                
            except (KeyError, IndexError) as e:
                print(f"[Gemini Error] 解析 JSON 失败: {e}")
                return "Gemini 返回格式异常 (解析失败)。"
        else:
            print(f"[Gemini Error] API 错误: {response.text}")
            return f"Gemini API 错误: {response.status_code}"
            
    except Exception as e:
        print(f"[Gemini Error] 连接异常: {e}")
        return "无法连接到 Gemini API。"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR + LLM 管道 (Gemini/Zhipu)")
    parser.add_argument("--input", "-i", required=True, help="输入的音频文件路径")
    parser.add_argument("--model", "-m", default="glm-4.5-flash", help="LLM模型名称")
    
    args = parser.parse_args()
    
    audio_path = args.input
    model_choice = args.model
    
    history = load_history()
    
    if os.path.exists(audio_path):
        converted_path = convert_audio_if_needed(audio_path)
        user_input_text = transcribe_audio_zhipu(converted_path)
        
        assistant_reply = ""
        
        if user_input_text:
            if "gemini" in model_choice.lower():
                assistant_reply = get_llm_response_gemini(user_input_text, model_choice, history)
            else:
                assistant_reply = get_llm_response_zhipu(user_input_text, model_choice, history)
        else:
            assistant_reply = "抱歉，我没有听清您说什么。"

        print(f"[Main] AI 回答: {assistant_reply}")
        
        with open(LATEST_RESPONSE_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(assistant_reply)
            
        if user_input_text:
            history.append({"role": "user", "content": user_input_text})
            history.append({"role": "assistant", "content": assistant_reply})
            save_history(history)
            
        append_to_log(user_input_text, assistant_reply, model_choice)