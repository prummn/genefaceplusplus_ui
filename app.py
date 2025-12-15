from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment
import os
from backend.video_generator import generate_video
from backend.model_trainer import train_model,stop_train_remote
from backend.chat_engine import chat_response, clear_chat_history
from werkzeug.utils import secure_filename
from datetime import datetime
from backend.model_trainer import (
    # geneface_check_health,
    # geneface_get_task_status,
    # geneface_list_tasks,
    geneface_health,
    geneface_api_call,
    geneface_list_videos
)
from werkzeug.utils import secure_filename
from flask import send_file
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


# 首页
@app.route('/')
def index():
    return render_template('index.html')


# 视频生成界面
@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    if request.method == 'POST':
        # 修改这里：添加 voice_clone 参数提取
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            "ref_audio": request.form.get('ref_audio'),
            "gpu_choice": request.form.get('gpu_choice'),

            # 核心修改：接收文本和音色
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



# 模型训练界面
# app.py 中检查这个路由

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

        # 确保返回 JSON
        if isinstance(result, dict):
            return jsonify(result)
        else:
            return jsonify({'status': 'success', 'video_path': result})

    return render_template('model_training.html')

# 新增：停止训练任务路由
@app.route('/geneface/stop/<task_id>', methods=['POST'])
def geneface_stop(task_id):
    """停止指定 ID 的训练任务"""
    if request.method == 'POST':
        cmd = request.form.get('cmd', type=int)
        result = stop_train_remote(cmd)

        # 确保返回 JSON
        if isinstance(result, dict):
            return jsonify(result)
        else:
            return jsonify({'status': 'success', 'video_path': result})

    return render_template('model_training.html')



# 实时对话系统界面
@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            "voice_clone": request.form.get('voice_clone'),
            "api_choice": request.form.get('api_choice'),
            # 补充缺失的参数
            "gpu_choice": request.form.get('gpu_choice'),
            "voice_clone_model": request.form.get('voice_clone_model'),
            # 新增：GeneFace++ 参数
            "gf_torso_ckpt": request.form.get('gf_torso_ckpt'),
            "gf_head_ckpt": request.form.get('gf_head_ckpt'),
        }

        # chat_response 现在会根据模型选择返回视频路径或音频路径
        result_path = chat_response(data)

        # 统一路径格式
        result_path = "/" + result_path.replace("\\", "/") if not result_path.startswith("/") else result_path

        return jsonify({'status': 'success', 'video_path': result_path})

    return render_template('chat_system.html')


@app.route('/save_audio', methods=['POST'])
def save_audio():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})

    # --- 修复流程开始 ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, 'io', "history")
    os.makedirs(save_dir, exist_ok=True)

    # 1. 先将浏览器发送的(可能损坏的)Blob保存到一个临时文件
    # (我们不关心它的扩展名，因为 pydub 会自动检测)
    raw_file_path = os.path.join(save_dir, 'aud_raw_from_browser')
    audio_file.save(raw_file_path)

    # 2. 定义我们最终想要的、修复后的、固定的 WAV 路径
    final_wav_path = os.path.join(save_dir, 'latest_user_aud.wav')

    # 3. 使用 pydub 加载这个(可能损坏的)文件，并重新导出
    #    pydub (和 ffmpeg) 擅长猜测原始格式
    try:
        print(f"[app.py] 收到原始音频。正在从 {raw_file_path} 加载...")
        # 加载文件 (pydub 会自动检测格式, 无论是 ogg, webm, 还是无头的 wav)
        audio = AudioSegment.from_file(raw_file_path)

        print(f"[app.py] 正在重新导出为标准 WAV: {final_wav_path}")

        # 重新导出为标准的、带头的 WAV 文件
        # 这时你可以强制设置参数，以确保 ASR 兼容：
        # audio.set_channels(1) 设为单声道
        # audio.set_frame_rate(16000) 设为 16kHz

        audio.export(final_wav_path, format="wav")

        print(f"[app.py] 音频保存并修复成功。")

    except Exception as e:
        print(f"!!!!!!!!!!!!!! [app.py] 严重错误: 修复音频文件失败 !!!!!!!!!!!!!!")
        print(f"Error: {e}")
        print("请确保 'ffmpeg' 已正确安装并_INCLUDED_在您 'RVC_cuda' 环境的 PATH 中。")
        # 强制确保文件已写入磁盘，避免后续读取时的竞争条件
        import time
        time.sleep(0.1)

        # 返回一个错误，停止流程
        return jsonify({'status': 'error', 'message': f'后台处理音频文件失败: {e}'}), 500
    finally:
        # 4. (可选) 清理临时的原始文件
        try:
            if os.path.exists(raw_file_path):
                os.remove(raw_file_path)
        except Exception as e:
            print(f"[app.py] 清理临时文件 {raw_file_path} 失败: {e}")
            pass  # 即使清理失败也不是大问题

    # --- 修复流程结束 ---

    # 5. 返回成功信息
    web_path = '/io/history/latest_usr_aud.wav'
    return jsonify({'status': 'success', 'message': '音频保存并修复成功', 'file_path': web_path})


@app.route('/clear_history', methods=['POST'])
def clear_history():
    success, msg = clear_chat_history()
    if success:
        return jsonify({'status': 'success', 'message': msg})
    else:
        return jsonify({'status': 'error', 'message': msg})


# GeneFace++ 状态查询路由
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

    # 清理 video_id
    video_id = re.sub(r'[^a-zA-Z0-9_]', '_', video_id)

    # 保存到 GeneFace 目录
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
    """查询任务状态"""
    return jsonify(geneface_api_call(f"/task/{task_id}"))


@app.route('/geneface/tasks', methods=['GET'])
def geneface_tasks():
    """列出所有任务"""
    return jsonify(geneface_api_call("/tasks"))


@app.route('/geneface/models', methods=['GET'])
def geneface_models():
    """列出已训练模型 (直接扫描本地文件系统)"""
    models = []
    # 假设 GeneFace 目录在 BASE_DIR 下
    ckpt_base = os.path.join(BASE_DIR, "GeneFace", "checkpoints", "motion2video_nerf")

    if os.path.exists(ckpt_base):
        video_models = {}

        for name in os.listdir(ckpt_base):
            full_path = os.path.join(ckpt_base, name)
            if os.path.isdir(full_path):
                if name.endswith('_head'):
                    video_id = name[:-5]
                    if video_id not in video_models:
                        video_models[video_id] = {
                            "video_id": video_id,
                            "head_checkpoints": [],
                            "torso_checkpoints": []
                        }

                    # 扫描 .ckpt 和 .pt 文件
                    for root, dirs, files in os.walk(full_path):
                        for f in files:
                            if f.endswith(('.ckpt', '.pt')):
                                # 构造相对路径，保持与 Docker 内部路径结构一致 (checkpoints/motion2video_nerf/...)
                                # 这里我们需要返回相对于 GeneFace 根目录的路径
                                rel_path = os.path.relpath(os.path.join(root, f), os.path.join(BASE_DIR, "GeneFace"))
                                rel_path = rel_path.replace("\\", "/") # 统一为正斜杠
                                video_models[video_id]["head_checkpoints"].append(rel_path)

                elif name.endswith('_torso'):
                    video_id = name[:-6]
                    if video_id not in video_models:
                        video_models[video_id] = {
                            "video_id": video_id,
                            "head_checkpoints": [],
                            "torso_checkpoints": []
                        }

                    for root, dirs, files in os.walk(full_path):
                        for f in files:
                            if f.endswith(('.ckpt', '.pt')):
                                rel_path = os.path.relpath(os.path.join(root, f), os.path.join(BASE_DIR, "GeneFace"))
                                rel_path = rel_path.replace("\\", "/")
                                video_models[video_id]["torso_checkpoints"].append(rel_path)

        # 排序
        for vid, data in video_models.items():
            # 按修改时间排序
            data["head_checkpoints"].sort(key=lambda x: os.path.getmtime(os.path.join(BASE_DIR, "GeneFace", x)) if os.path.exists(os.path.join(BASE_DIR, "GeneFace", x)) else 0, reverse=True)
            data["torso_checkpoints"].sort(key=lambda x: os.path.getmtime(os.path.join(BASE_DIR, "GeneFace", x)) if os.path.exists(os.path.join(BASE_DIR, "GeneFace", x)) else 0, reverse=True)

        models = list(video_models.values())

    return jsonify(models)


@app.route('/geneface/video/<video_id>')
def geneface_video_file(video_id):
    """提供 GeneFace 视频文件访问"""
    import re
    # 清理 video_id 防止路径遍历攻击
    video_id = re.sub(r'[^a-zA-Z0-9_-]', '', video_id)

    video_path = os.path.join(BASE_DIR, "GeneFace", "data", "raw", "videos", f"{video_id}.mp4")

    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "视频不存在"}), 404


# app.py 中添加以下路由

@app.route('/geneface/audios', methods=['GET'])
def geneface_audios():
    """列出可用音频"""
    return jsonify(geneface_api_call("/audios"))


@app.route('/geneface/upload_audio', methods=['POST'])
def geneface_upload_audio():
    """上传音频到 GeneFace++"""
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "没有音频文件"}), 400

    audio_file = request.files['audio']
    audio_name = request.form.get('audio_name')

    if not audio_name:
        audio_name = os.path.splitext(secure_filename(audio_file.filename))[0]

    # 清理名称
    audio_name = re.sub(r'[^a-zA-Z0-9_-]', '_', audio_name)

    # 保存到 GeneFace 目录
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
    """GeneFace++ 推理"""
    data = request.json
    return jsonify(geneface_api_call("/infer", "POST", data))


@app.route('/geneface/output/<path:filename>')
def geneface_output_file(filename):
    """提供 GeneFace 输出视频文件访问"""
    # 清理路径防止遍历攻击
    filename = secure_filename(filename)
    video_path = os.path.join(BASE_DIR, "GeneFace", "infer_out", filename)

    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "视频不存在"}), 404


@app.route('/list_ref_audios', methods=['GET'])
def list_ref_audios():
    """列出 io/input/audio 下的参考音频"""
    audio_dir = os.path.join(BASE_DIR, "io", "input", "audio")
    audios = []
    if os.path.exists(audio_dir):
        for f in os.listdir(audio_dir):
            if f.lower().endswith(('.wav', '.mp3', '.m4a')):
                audios.append(f)
    return jsonify(audios)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)