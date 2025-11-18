import argparse
import os
import time
from zhipuai import ZhipuAI
from pydub import AudioSegment 

# --- 文件名常量 ---
LATEST_RESPONSE_FILE = "latest_ai_response.txt"
LOG_FILE = "conversation_log.txt"
LATEST_RESPONSE_FILE_PATH = os.path.join(os.path.dirname(__file__), LATEST_RESPONSE_FILE)
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), LOG_FILE)

# 1. 设置API Key
api_key = "c4f0ad328f1248b2a0840c78c7b54ab4.6VT7eLDZ9T7ZagY7"

# 2. 初始化 ZhipuAI 客户端
try:
    client = ZhipuAI(api_key=api_key)
except Exception as e:
    print(f"API Key 初始化失败: {e}")
    exit()

# 3. 维护一个对话历史列表
conversation_history = [
    {"role": "system", "content": "你是一个乐于助人的对话助手，请用简洁明了的中文回答。"}
]

def convert_audio_if_needed(audio_file_path):
    """
    检查音频文件格式，如果不是 .mp3 或 .wav, 则转换为 .mp3
    同时，强制将所有音频文件转换为“单声道”
    """
    file_name, file_extension = os.path.splitext(audio_file_path)
    
    if not os.path.exists(audio_file_path):
        print(f"[Converter] 错误：输入文件 {audio_file_path} 未找到。")
        return None

    # 定义输出文件名 (统一转为 .mp3)
    output_path = file_name + "_mono.mp3"
    
    try:
        # 加载音频文件
        audio = AudioSegment.from_file(audio_file_path)
        # 强制设置为单声道
        audio = audio.set_channels(1)
        # 导出为 .mp3
        audio.export(output_path, format="mp3")
        return output_path
    except Exception as e:
        print(f"[Converter] 转换失败: {e}")
        return None

def transcribe_audio_zhipu(audio_file_path):
    """
    步骤 A: 语音转文字 (ASR)
    """
    if not audio_file_path:
        return None
    print(f"[ASR] 正在识别 (Zhipu): {audio_file_path}...")
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="glm-asr",
                file=audio_file,
            )
        print(f"[ASR] 识别结果: {response.text}")
        return response.text
    except Exception as e:
        print(f"[ASR] 识别失败: {e}")
        return None

def save_response_to_file(user_text, assistant_reply):
    """保存AI回答到文件"""
    try:
        with open(LATEST_RESPONSE_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(assistant_reply)
    except Exception as e:
        print(f"[Save] 保存最新回答失败: {e}")

    try:
        user_text_to_save = user_text if user_text else "N/A (ASR失败)"
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(f"【{time.ctime()}】\n")
            f.write(f"你: {user_text_to_save}\n")
            f.write(f"AI: {assistant_reply}\n\n")
    except Exception as e:
        print(f"[Save] 追加日志失败: {e}")

def get_llm_response_zhipu(user_text, model_name="glm-4-flash"):
    """
    步骤 B: 获取大语言模型应答 (LLM)
    """
    if not user_text:
        assistant_reply = "我没有听清，你能再说一遍吗？"
        save_response_to_file(user_text, assistant_reply)
        return assistant_reply
        
    print(f"[LLM] 正在思考 (Model: {model_name})...")
    try:
        conversation_history.append({"role": "user", "content": user_text})
        
        # 使用传入的模型参数
        response = client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
        )
        
        assistant_reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        print(f"[LLM] AI 回答: {assistant_reply}")

        save_response_to_file(user_text, assistant_reply)
        return assistant_reply
        
    except Exception as e:
        print(f"[LLM] 思考失败: {e}")
        return "抱歉，我好像出错了。"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR + LLM 管道")
    parser.add_argument("--input", "-i", required=True, help="输入的音频文件路径")
    # 【新】添加 --model 参数
    parser.add_argument("--model", "-m", default="glm-4-flash", help="LLM模型名称 (e.g., glm-4-flash, glm-4)")
    
    args = parser.parse_args()

    my_audio_file = args.input 
    llm_model_name = args.model

    if os.path.exists(my_audio_file):
        path_to_upload = convert_audio_if_needed(my_audio_file)
        
        user_input_text = transcribe_audio_zhipu(path_to_upload)
        
        # 传递模型参数
        ai_response_text = get_llm_response_zhipu(user_input_text, model_name=llm_model_name)