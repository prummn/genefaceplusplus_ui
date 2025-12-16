from flask import Flask, render_template, request, jsonify, send_file
from pydub import AudioSegment
import os
import uuid
import re
import time
from werkzeug.utils import secure_filename
from datetime import datetime

# 引入后端模块
from backend.video_generator import generate_video
from backend.model_trainer import (
    train_model, 
    stop_train_remote,
    geneface_health,
    geneface_api_call,
    geneface_list_videos
)
from backend.chat_engine import chat_response, clear_chat_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# --- 页面路由 ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            "ref_audio": request.form.get('ref_audio'),
            "gpu_choice": request.form.get('gpu_choice'),
            "target_text": request.form.get('target_text'),
            "voice_clone": request.form.get('voice_clone'),
            # GeneFace++ 参数
            "gf_head_ckpt": request.form.get('gf_head_ckpt'),
            "gf_torso_ckpt": request.form.get('gf_torso_ckpt'),
            "gf_audio_path": request.form.get('gf_audio_path'),
        }

        video_path = generate_video(data)
        return jsonify({'status': 'success', 'video_path': video_path})

    default_video = 'videos/sample.mp4'
    latest_video_path = os.path.join('static', 'videos', 'geneface_latest.mp4')
    if os.path.exists(os.path.join(BASE_DIR, latest_video_path)):
        default_video = 'videos/geneface_latest.mp4'

    return render_template('video_generation.html', default_video=default_video)

@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        data = {
            "model_choice": request.form.get('model_choice'),
            "ref_video": request.form.get('ref_video'),
            "gpu_choice": request.form.get('gpu_choice'),
            "epoch": request.form.get('epoch'),
            "custom_params": request.form.get('custom_params'),
            # GeneFace++ 专用参数
            "video_id": request.form.get('video_id'),
            "train_stage": request.form.get('train_stage'),
            "max_updates_head": request.form.get('max_updates_head'),
            "max_updates_torso": request.form.get('max_updates_torso'),
            "head_ckpt": request.form.get('head_ckpt'),
        }

        result = train_model(data)

        if isinstance(result, dict):
            return jsonify(result)
        else:
            return jsonify({'status': 'success', 'video_path': result})

    return render_template('model_training.html')

@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            "voice_clone": request.form.get('voice_clone'),
            "api_choice": request.form.get('api_choice'),
            "gpu_choice": request.form.get('gpu_choice'),
            "voice_clone_model": request.form.get('voice_clone_model'),
            "gf_torso_ckpt": request.form.get('gf_torso_ckpt'),
            "gf_head_ckpt": request.form.get('gf_head_ckpt'),
        }

        try:
            result_path = chat_response(data)
            # 统一路径格式为 Web 路径
            if result_path:
                result_path = "/" + result_path.replace("\\", "/") if not result_path.startswith("/") else result_path
            return jsonify({'status': 'success', 'video_path': result_path})
        except Exception as e:
            print(f"Chat System Error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    return render_template('chat_system.html')

# --- 功能接口 ---

@app.route('/geneface/stop/<task_id>', methods=['POST'])
def geneface_stop(task_id):
    """停止指定 ID 的训练任务"""
    cmd = request.form.get('cmd', type=int)
    # 注意：这里假设 stop_train_remote 接受 task_id 或 cmd，根据原逻辑调整
    # 如果原逻辑是用 cmd 参数控制，这里保留原样
    result = stop_train_remote(cmd)
    
    if isinstance(result, dict):
        return jsonify(result)
    else:
        return jsonify({'status': 'success', 'message': str(result)})

@app.route('/save_audio', methods=['POST'])
def save_audio():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})

    save_dir = os.path.join(BASE_DIR, 'io', "history")
    os.makedirs(save_dir, exist_ok=True)

    raw_file_path = os.path.join(save_dir, 'aud_raw_from_browser')
    audio_file.save(raw_file_path)

    final_wav_path = os.path.join(save_dir, 'latest_user_aud.wav')

    try:
        print(f"[app.py] 收到原始音频，正在从 {raw_file_path} 加载...")
        audio = AudioSegment.from_file(raw_file_path)
        
        print(f"[app.py] 正在重新导出为标准 WAV: {final_wav_path}")
        # 强制转为单声道和16k采样率以兼容 ASR/RVC
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(final_wav_path, format="wav")
        print(f"[app.py] 音频保存并修复成功。")

    except Exception as e:
        print(f"!!!!!!!!!!!!!! [app.py] 严重错误: 修复音频文件失败 !!!!!!!!!!!!!!")
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': f'后台处理音频文件失败 (请检查ffmpeg): {e}'}), 500
    finally:
        try:
            if os.path.exists(raw_file_path):
                os.remove(raw_file_path)
        except Exception:
            pass

    # 返回给前端的路径可以不包含盘符，用于展示或调试
    return jsonify({'status': 'success', 'message': '音频保存并修复成功', 'file_path': 'io/history/latest_user_aud.wav'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """清除对话历史"""
    success, msg = clear_chat_history()
    if success:
        return jsonify({'status': 'success', 'message': msg})
    else:
        return jsonify({'status': 'error', 'message': msg})

@app.route('/list_ref_audios', methods=['GET'])
def list_ref_audios():
    """列出 io/input/audio 下的参考音频"""
    audio_dir = os.path.join(BASE_DIR, "io", "input", "audio")
    audios = []
    if os.path.exists(audio_dir):
        for f in os.listdir(audio_dir):
            if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                audios.append(f)
    return jsonify(audios)

@app.route('/upload_ref_audio', methods=['POST'])
def upload_ref_audio():
    """上传参考音频 (增强版: 支持自动转码和UUID命名)"""
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '请求中没有音频文件'})
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '未选择文件'})
        
    try:
        # 1. 定义路径
        temp_dir = os.path.join(BASE_DIR, 'io', 'temp')
        final_dir = os.path.join(BASE_DIR, 'io', 'input', 'audio')
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        
        # 2. 保存原始文件到 temp (使用UUID防止中文名问题)
        original_ext = os.path.splitext(file.filename)[1].lower()
        if not original_ext: original_ext = ".wav"
            
        temp_filename = f"upload_temp_{uuid.uuid4().hex}{original_ext}"
        temp_path = os.path.join(temp_dir, temp_filename)
        file.save(temp_path)
        
        # 3. 准备最终文件名
        original_stem = os.path.splitext(file.filename)[0]
        # 尝试保留原始文件名主体，如果 secure_filename 变为空(如纯中文)，则生成随机名
        safe_stem = secure_filename(original_stem)
        if not safe_stem: 
            safe_stem = f"ref_{uuid.uuid4().hex[:8]}"
            
        final_filename = f"{safe_stem}.wav"
        final_path = os.path.join(final_dir, final_filename)
        
        # 避免文件名冲突
        counter = 1
        while os.path.exists(final_path):
            final_filename = f"{safe_stem}_{counter}.wav"
            final_path = os.path.join(final_dir, final_filename)
            counter += 1
        
        print(f"[Upload] 正在转换: {temp_path} -> {final_path}")
        
        # 4. 转换格式
        try:
            audio = AudioSegment.from_file(temp_path)
            audio = audio.set_channels(1) # 转单声道
            audio.export(final_path, format="wav")
        except FileNotFoundError:
            raise Exception("服务器未安装 ffmpeg，无法转换音频格式。")
        except Exception as trans_e:
            raise Exception(f"音频转换失败: {trans_e}")
        
        # 5. 清理
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        print(f"[Upload] 完成: {final_path}")
        return jsonify({'status': 'success', 'filename': final_filename})
        
    except Exception as e:
        print(f"[Upload] Error: {e}")
        # 尝试清理
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass
        return jsonify({'status': 'error', 'message': str(e)})

# --- GeneFace++ 专用接口 ---

@app.route('/geneface/health', methods=['GET'])
def geneface_health_check():
    """检查 GeneFace++ 服务"""
    is_ok = geneface_health()
    return jsonify({
        "status": "ok" if is_ok else "unavailable",
        "message": "服务正常" if is_ok else "请启动 Docker: docker start geneface"
    })

@app.route('/geneface/videos', methods=['GET'])
def geneface_videos():
    """列出可用视频"""
    return jsonify(geneface_api_call("/videos"))

@app.route('/geneface/upload', methods=['POST'])
def geneface_upload():
    """上传视频到 GeneFace++"""
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "没有视频文件"}), 400

    video_file = request.files['video']
    video_id = request.form.get('video_id')

    if not video_id:
        video_id = os.path.splitext(secure_filename(video_file.filename))[0]

    video_id = re.sub(r'[^a-zA-Z0-9_]', '_', video_id)

    geneface_video_dir = os.path.join(BASE_DIR, "GeneFace", "data", "raw", "videos")
    os.makedirs(geneface_video_dir, exist_ok=True)

    save_path = os.path.join(geneface_video_dir, f"{video_id}.mp4")
    video_file.save(save_path)

    return jsonify({
        "status": "success",
        "video_id": video_id,
        "message": "视频上传成功"
    })

@app.route('/geneface/task/<task_id>', methods=['GET'])
def geneface_task(task_id):
    return jsonify(geneface_api_call(f"/task/{task_id}"))

@app.route('/geneface/tasks', methods=['GET'])
def geneface_tasks():
    return jsonify(geneface_api_call("/tasks"))

@app.route('/geneface/models', methods=['GET'])
def geneface_models():
    """列出已训练模型"""
    models = []
    ckpt_base = os.path.join(BASE_DIR, "GeneFace", "checkpoints", "motion2video_nerf")

    if os.path.exists(ckpt_base):
        video_models = {}
        for name in os.listdir(ckpt_base):
            full_path = os.path.join(ckpt_base, name)
            if os.path.isdir(full_path):
                if name.endswith('_head'):
                    video_id = name[:-5]
                    if video_id not in video_models:
                        video_models[video_id] = {"video_id": video_id, "head_checkpoints": [], "torso_checkpoints": []}
                    for root, dirs, files in os.walk(full_path):
                        for f in files:
                            if f.endswith(('.ckpt', '.pt')):
                                rel_path = os.path.relpath(os.path.join(root, f), os.path.join(BASE_DIR, "GeneFace"))
                                rel_path = rel_path.replace("\\", "/")
                                video_models[video_id]["head_checkpoints"].append(rel_path)
                elif name.endswith('_torso'):
                    video_id = name[:-6]
                    if video_id not in video_models:
                        video_models[video_id] = {"video_id": video_id, "head_checkpoints": [], "torso_checkpoints": []}
                    for root, dirs, files in os.walk(full_path):
                        for f in files:
                            if f.endswith(('.ckpt', '.pt')):
                                rel_path = os.path.relpath(os.path.join(root, f), os.path.join(BASE_DIR, "GeneFace"))
                                rel_path = rel_path.replace("\\", "/")
                                video_models[video_id]["torso_checkpoints"].append(rel_path)

        for vid, data in video_models.items():
            data["head_checkpoints"].sort(key=lambda x: os.path.getmtime(os.path.join(BASE_DIR, "GeneFace", x)) if os.path.exists(os.path.join(BASE_DIR, "GeneFace", x)) else 0, reverse=True)
            data["torso_checkpoints"].sort(key=lambda x: os.path.getmtime(os.path.join(BASE_DIR, "GeneFace", x)) if os.path.exists(os.path.join(BASE_DIR, "GeneFace", x)) else 0, reverse=True)

        models = list(video_models.values())

    return jsonify(models)

@app.route('/geneface/video/<video_id>')
def geneface_video_file(video_id):
    video_id = re.sub(r'[^a-zA-Z0-9_-]', '', video_id)
    video_path = os.path.join(BASE_DIR, "GeneFace", "data", "raw", "videos", f"{video_id}.mp4")
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "视频不存在"}), 404

@app.route('/geneface/audios', methods=['GET'])
def geneface_audios():
    return jsonify(geneface_api_call("/audios"))

@app.route('/geneface/upload_audio', methods=['POST'])
def geneface_upload_audio():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "没有音频文件"}), 400

    audio_file = request.files['audio']
    audio_name = request.form.get('audio_name')

    if not audio_name:
        audio_name = os.path.splitext(secure_filename(audio_file.filename))[0]

    audio_name = re.sub(r'[^a-zA-Z0-9_-]', '_', audio_name)
    audio_dir = os.path.join(BASE_DIR, "GeneFace", "data", "raw", "val_wavs")
    os.makedirs(audio_dir, exist_ok=True)

    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in ['.wav', '.mp3', '.m4a']:
        ext = '.wav'

    save_path = os.path.join(audio_dir, f"{audio_name}{ext}")
    audio_file.save(save_path)
    rel_path = f"data/raw/val_wavs/{audio_name}{ext}"

    return jsonify({
        "status": "success",
        "audio_name": f"{audio_name}{ext}",
        "audio_path": rel_path
    })

@app.route('/geneface/infer', methods=['POST'])
def geneface_infer():
    data = request.json
    return jsonify(geneface_api_call("/infer", "POST", data))

@app.route('/geneface/output/<path:filename>')
def geneface_output_file(filename):
    filename = secure_filename(filename)
    video_path = os.path.join(BASE_DIR, "GeneFace", "infer_out", filename)
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "视频不存在"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000, use_reloader=False)