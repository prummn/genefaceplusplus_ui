# GeneFace/api_server.py
"""
GeneFace++ API 服务
"""

from flask import Flask, request, jsonify
import subprocess
import os
import threading
import time
import uuid
import json
import shutil
import re
import sys
import logging

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(message, level="INFO"):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}", flush=True)


app = Flask(__name__)

# 禁用 Flask 请求日志
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

WORK_DIR = "/data/geneface"
TASKS_FILE = os.path.join(WORK_DIR, "tasks_state.json")
TEMPLATE_CONFIG_DIR = os.path.join(WORK_DIR, "egs", "datasets", "May")

tasks = {}


def load_tasks():
    global tasks
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            log(f"已加载 {len(tasks)} 个任务")
        except:
            tasks = {}


def save_tasks():
    try:
        with open(TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"保存任务失败: {e}", "ERROR")


def run_command(cmd, cwd=None, env=None):
    """执行命令并实时输出"""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    work_dir = cwd or WORK_DIR
    log(f"执行命令: {cmd}")
    log(f"工作目录: {work_dir}")

    stdout_lines = []
    stderr_lines = []

    try:
        # 使用 Popen 实现实时输出
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=work_dir,
            env=full_env,
            bufsize=1,  # 行缓冲
        )

        # 实时读取并打印输出
        def read_stdout():
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.rstrip()
                    print(f"[STDOUT] {line}", flush=True)
                    stdout_lines.append(line)

        def read_stderr():
            for line in iter(process.stderr.readline, ''):
                if line:
                    line = line.rstrip()
                    print(f"[STDERR] {line}", flush=True)
                    stderr_lines.append(line)

        # 启动读取线程
        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # 等待进程完成
        process.wait()

        # 等待读取线程完成
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        returncode = process.returncode

        if returncode != 0:
            log(f"命令失败，返回码: {returncode}", "ERROR")

        return {
            "returncode": returncode,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines)
        }

    except Exception as e:
        log(f"执行命令异常: {e}", "ERROR")
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }


def update_task(task_id, **kwargs):
    if task_id in tasks:
        tasks[task_id].update(kwargs)
        tasks[task_id]["updated_at"] = time.time()
        save_tasks()
        log(f"任务 {task_id} 更新: {kwargs.get('current_step', kwargs.get('status', ''))}")


def generate_config_files(video_id, max_updates_head=40000, max_updates_torso=40000):
    """从 May 模板复制并修改配置文件"""
    log(f"生成配置文件: video_id={video_id}")

    target_config_dir = os.path.join(WORK_DIR, "egs", "datasets", video_id)

    if os.path.exists(target_config_dir):
        shutil.rmtree(target_config_dir)

    if not os.path.exists(TEMPLATE_CONFIG_DIR):
        log(f"模板目录不存在: {TEMPLATE_CONFIG_DIR}", "ERROR")
        return False

    shutil.copytree(TEMPLATE_CONFIG_DIR, target_config_dir)
    log(f"已复制模板到: {target_config_dir}")

    config_files = [
        ("lm3d_radnerf.yaml", False),
        ("lm3d_radnerf_sr.yaml", False),
        ("lm3d_radnerf_torso.yaml", True),
        ("lm3d_radnerf_torso_sr.yaml", True),
    ]

    for filename, is_torso in config_files:
        config_path = os.path.join(target_config_dir, filename)
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            content = re.sub(r'video_id:\s*May', f'video_id: {video_id}', content)

            if is_torso:
                content = re.sub(
                    r'head_model_dir:\s*checkpoints/[^\s]+',
                    f'head_model_dir: checkpoints/motion2video_nerf/{video_id}_head',
                    content
                )
                content = re.sub(r'max_updates:\s*\d+', f'max_updates: {max_updates_torso}', content)
            else:
                content = re.sub(r'max_updates:\s*\d+', f'max_updates: {max_updates_head}', content)

            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            log(f"已更新配置: {filename}")

    return True


# ==================== 预处理 ====================
def preprocessing_worker(task_id, video_id, gpu_id="0", max_updates_head=40000, max_updates_torso=40000):
    log(f"========== 开始预处理任务 {task_id} ==========")
    update_task(task_id, status="running", current_step="初始化")

    env = {"VIDEO_ID": video_id, "CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": "./"}

    try:
        update_task(task_id, current_step="1/10 视频缩放到512x512")
        log("步骤 1/10: 视频缩放")

        original_video = f"data/raw/videos/{video_id}.mp4"
        full_path = os.path.join(WORK_DIR, original_video)
        if not os.path.exists(full_path):
            raise Exception(f"视频文件不存在: {full_path}")

        cmd = f'ffmpeg -y -i data/raw/videos/{video_id}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 data/raw/videos/{video_id}_512.mp4'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"视频缩放失败: {result['stderr']}")

        run_command(f'mv data/raw/videos/{video_id}.mp4 data/raw/videos/{video_id}_original.mp4', WORK_DIR, env)
        run_command(f'mv data/raw/videos/{video_id}_512.mp4 data/raw/videos/{video_id}.mp4', WORK_DIR, env)

        update_task(task_id, current_step="2/10 提取音频")
        log("步骤 2/10: 提取音频")
        run_command(f'mkdir -p data/processed/videos/{video_id}', WORK_DIR, env)
        cmd = f'ffmpeg -y -i data/raw/videos/{video_id}.mp4 -f wav -ar 16000 data/processed/videos/{video_id}/aud.wav'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"音频提取失败: {result['stderr']}")

        update_task(task_id, current_step="3/10 提取HuBERT特征")
        log("步骤 3/10: 提取HuBERT特征")
        cmd = f'python data_gen/utils/process_audio/extract_hubert.py --video_id={video_id}'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"HuBERT提取失败: {result['stderr']}")

        update_task(task_id, current_step="4/10 提取Mel和F0特征")
        log("步骤 4/10: 提取Mel和F0")
        cmd = f'python data_gen/utils/process_audio/extract_mel_f0.py --video_id={video_id}'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"Mel/F0提取失败: {result['stderr']}")

        update_task(task_id, current_step="5/10 提取视频帧")
        log("步骤 5/10: 提取视频帧")
        run_command(f'mkdir -p data/processed/videos/{video_id}/gt_imgs', WORK_DIR, env)
        cmd = f'ffmpeg -y -i data/raw/videos/{video_id}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 data/processed/videos/{video_id}/gt_imgs/%08d.jpg'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"视频帧提取失败: {result['stderr']}")

        update_task(task_id, current_step="6/10 提取分割图")
        log("步骤 6/10: 提取分割图")
        cmd = f'python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/{video_id}.mp4'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"分割图提取失败: {result['stderr']}")

        update_task(task_id, current_step="7/10 提取2D关键点")
        log("步骤 7/10: 提取2D关键点")
        cmd = f'python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/{video_id}.mp4'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"2D关键点提取失败: {result['stderr']}")

        update_task(task_id, current_step="8/10 拟合3DMM")
        log("步骤 8/10: 拟合3DMM")
        cmd = f'python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/{video_id}.mp4 --reset --id_mode=global'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"3DMM拟合失败: {result['stderr']}")

        update_task(task_id, current_step="9/10 二值化数据")
        log("步骤 9/10: 二值化数据")
        cmd = f'python data_gen/runs/binarizer_nerf.py --video_id={video_id}'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"二值化失败: {result['stderr']}")

        update_task(task_id, current_step="10/10 生成配置文件")
        log("步骤 10/10: 生成配置文件")
        generate_config_files(video_id, max_updates_head, max_updates_torso)

        update_task(task_id, status="completed", current_step="预处理完成",
                    message="数据预处理完成，可以开始训练头部模型", next_stage="head")
        log(f"========== 任务 {task_id} 完成 ==========")

    except Exception as e:
        log(f"任务 {task_id} 失败: {e}", "ERROR")
        update_task(task_id, status="failed", error=str(e))


# ==================== 头部训练 ====================
def head_training_worker(task_id, video_id, gpu_id="0"):
    log(f"========== 开始头部训练 {task_id} ==========")
    update_task(task_id, status="running", current_step="训练头部模型")

    env = {"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": "./"}

    try:
        config_path = f"egs/datasets/{video_id}/lm3d_radnerf_sr.yaml"
        if not os.path.exists(os.path.join(WORK_DIR, config_path)):
            config_path = f"egs/datasets/{video_id}/lm3d_radnerf.yaml"
            if not os.path.exists(os.path.join(WORK_DIR, config_path)):
                raise Exception(f"配置文件不存在，请先进行数据预处理")

        exp_name = f"motion2video_nerf/{video_id}_head"

        cmd = f'python tasks/run.py --config={config_path} --exp_name={exp_name} --reset'
        result = run_command(cmd, WORK_DIR, env)

        if result["returncode"] != 0:
            raise Exception(f"头部模型训练失败: {result['stderr']}")

        head_ckpt = f"checkpoints/motion2video_nerf/{video_id}_head"
        update_task(task_id, status="completed", current_step="头部训练完成",
                    head_ckpt=head_ckpt, message="可以开始训练身体模型", next_stage="torso")
        log(f"========== 头部训练 {task_id} 完成 ==========")

    except Exception as e:
        log(f"头部训练失败: {e}", "ERROR")
        update_task(task_id, status="failed", error=str(e))


# ==================== 身体训练 ====================
def torso_training_worker(task_id, video_id, head_ckpt, gpu_id="0"):
    log(f"========== 开始身体训练 {task_id} ==========")
    log(f"video_id={video_id}, head_ckpt={head_ckpt}, gpu_id={gpu_id}")
    update_task(task_id, status="running", current_step="训练身体模型")

    env = {"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": "./"}

    try:
        config_path = f"egs/datasets/{video_id}/lm3d_radnerf_torso_sr.yaml"
        if not os.path.exists(os.path.join(WORK_DIR, config_path)):
            config_path = f"egs/datasets/{video_id}/lm3d_radnerf_torso.yaml"
            if not os.path.exists(os.path.join(WORK_DIR, config_path)):
                raise Exception(f"配置文件不存在，请先进行数据预处理")

        exp_name = f"motion2video_nerf/{video_id}_torso"

        cmd = f'python tasks/run.py --config={config_path} --exp_name={exp_name} --hparams=head_model_dir={head_ckpt} --reset'
        result = run_command(cmd, WORK_DIR, env)

        if result["returncode"] != 0:
            raise Exception(f"身体模型训练失败: {result['stderr']}")

        torso_ckpt = f"checkpoints/motion2video_nerf/{video_id}_torso"
        update_task(task_id, status="completed", current_step="全部训练完成",
                    torso_ckpt=torso_ckpt, head_ckpt=head_ckpt, message="训练完成！")
        log(f"========== 身体训练 {task_id} 完成 ==========")

    except Exception as e:
        log(f"身体训练失败: {e}", "ERROR")
        update_task(task_id, status="failed", error=str(e))


# ==================== API 路由 ====================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "GeneFace++ API", "work_dir": WORK_DIR})


@app.route('/videos', methods=['GET'])
def list_videos():
    video_dir = os.path.join(WORK_DIR, "data", "raw", "videos")
    videos = []
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith('.mp4') and not f.endswith('_original.mp4') and not f.endswith('_512.mp4'):
                video_path = os.path.join(video_dir, f)
                video_id = os.path.splitext(f)[0]
                stat = os.stat(video_path)
                videos.append({
                    "video_id": video_id,
                    "filename": f,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                })
    return jsonify(videos)


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "没有视频文件"}), 400

    video_file = request.files['video']
    video_id = request.form.get('video_id') or os.path.splitext(video_file.filename)[0]
    video_id = re.sub(r'[^a-zA-Z0-9_]', '_', video_id)

    video_dir = os.path.join(WORK_DIR, "data", "raw", "videos")
    os.makedirs(video_dir, exist_ok=True)

    save_path = os.path.join(video_dir, f"{video_id}.mp4")
    video_file.save(save_path)
    log(f"视频已上传: {save_path}")

    return jsonify({"status": "success", "video_id": video_id})


@app.route('/preprocess', methods=['POST'])
def start_preprocessing():
    data = request.json
    video_id = data.get('video_id')
    gpu_id = data.get('gpu_id', '0')
    max_updates_head = int(data.get('max_updates_head', 40000))
    max_updates_torso = int(data.get('max_updates_torso', 40000))

    log(f"收到预处理请求: video_id={video_id}, gpu_id={gpu_id}")

    if not video_id:
        return jsonify({"status": "error", "message": "缺少 video_id"}), 400

    video_path = os.path.join(WORK_DIR, f"data/raw/videos/{video_id}.mp4")
    if not os.path.exists(video_path):
        return jsonify({"status": "error", "message": f"视频不存在: {video_id}.mp4"}), 404

    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "task_id": task_id, "type": "preprocess", "video_id": video_id,
        "gpu_id": gpu_id, "status": "pending", "current_step": "等待开始",
        "created_at": time.time()
    }
    save_tasks()

    log(f"创建预处理任务: {task_id}")

    thread = threading.Thread(target=preprocessing_worker,
                              args=(task_id, video_id, gpu_id, max_updates_head, max_updates_torso))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "success", "task_id": task_id, "video_id": video_id})


@app.route('/train/head', methods=['POST'])
def start_head_training():
    data = request.json
    video_id = data.get('video_id')
    gpu_id = data.get('gpu_id', '0')

    log(f"收到头部训练请求: video_id={video_id}, gpu_id={gpu_id}")

    if not video_id:
        log("错误: 缺少 video_id", "ERROR")
        return jsonify({"status": "error", "message": "缺少 video_id"}), 400

    processed_dir = os.path.join(WORK_DIR, f"data/processed/videos/{video_id}")
    if not os.path.exists(processed_dir):
        log(f"错误: 预处理数据不存在: {processed_dir}", "ERROR")
        return jsonify({"status": "error", "message": "预处理数据不存在，请先进行数据预处理"}), 400

    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "task_id": task_id, "type": "head_training", "video_id": video_id,
        "status": "pending", "current_step": "等待开始", "created_at": time.time()
    }
    save_tasks()

    log(f"创建头部训练任务: {task_id}")

    thread = threading.Thread(target=head_training_worker, args=(task_id, video_id, gpu_id))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "success", "task_id": task_id})


@app.route('/train/torso', methods=['POST'])
def start_torso_training():
    data = request.json
    video_id = data.get('video_id')
    head_ckpt = data.get('head_ckpt')
    gpu_id = data.get('gpu_id', '0')

    log(f"收到身体训练请求: video_id={video_id}, head_ckpt={head_ckpt}, gpu_id={gpu_id}")

    if not video_id:
        log("错误: 缺少 video_id", "ERROR")
        return jsonify({"status": "error", "message": "缺少 video_id"}), 400

    if not head_ckpt:
        head_ckpt = f"checkpoints/motion2video_nerf/{video_id}_head"
        log(f"使用默认 head_ckpt: {head_ckpt}")

    head_ckpt_full = os.path.join(WORK_DIR, head_ckpt)
    log(f"检查头部模型路径: {head_ckpt_full}")

    if not os.path.exists(head_ckpt_full):
        log(f"错误: 头部模型不存在: {head_ckpt_full}", "ERROR")
        return jsonify({
            "status": "error",
            "message": f"头部模型不存在: {head_ckpt}，请先训练头部模型"
        }), 400

    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "task_id": task_id, "type": "torso_training", "video_id": video_id,
        "head_ckpt": head_ckpt, "gpu_id": gpu_id,
        "status": "pending", "current_step": "等待开始", "created_at": time.time()
    }
    save_tasks()

    log(f"创建身体训练任务: {task_id}")

    thread = threading.Thread(target=torso_training_worker, args=(task_id, video_id, head_ckpt, gpu_id))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "success", "task_id": task_id, "message": "身体模型训练已启动"})


@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    if task_id not in tasks:
        return jsonify({"status": "error", "message": "任务不存在"}), 404
    return jsonify(tasks[task_id])


@app.route('/tasks', methods=['GET'])
def list_tasks():
    return jsonify(sorted(tasks.values(), key=lambda x: x.get('created_at', 0), reverse=True))


@app.route('/models', methods=['GET'])
def list_models():
    """列出所有已训练的模型及其具体的checkpoint文件"""
    models = []
    ckpt_base = os.path.join(WORK_DIR, "checkpoints", "motion2video_nerf")

    log(f"扫描模型目录: {ckpt_base}")

    if os.path.exists(ckpt_base):
        video_models = {}

        for name in os.listdir(ckpt_base):
            full_path = os.path.join(ckpt_base, name)
            if os.path.isdir(full_path):
                if name.endswith('_head'):
                    video_id = name[:-5]
                    if video_id not in video_models:
                        # 初始化结构
                        video_models[video_id] = {
                            "video_id": video_id,
                            "head_checkpoints": [],
                            "torso_checkpoints": []
                        }

                    # 扫描具体的 .ckpt 和 .pt 文件
                    for root, dirs, files in os.walk(full_path):
                        for f in files:
                            if f.endswith(('.ckpt', '.pt')):
                                # 保存相对于 WORK_DIR 的路径
                                rel_path = os.path.relpath(os.path.join(root, f), WORK_DIR)
                                video_models[video_id]["head_checkpoints"].append(rel_path)

                elif name.endswith('_torso'):
                    video_id = name[:-6]
                    if video_id not in video_models:
                        video_models[video_id] = {
                            "video_id": video_id,
                            "head_checkpoints": [],
                            "torso_checkpoints": []
                        }

                    # 扫描具体的 .ckpt 和 .pt 文件
                    for root, dirs, files in os.walk(full_path):
                        for f in files:
                            if f.endswith(('.ckpt', '.pt')):
                                rel_path = os.path.relpath(os.path.join(root, f), WORK_DIR)
                                video_models[video_id]["torso_checkpoints"].append(rel_path)

        # 对checkpoint进行排序，让最新的在前面（如果可能的话）
        # 这里简单按照文件名排序，或者你可以按修改时间排序
        for vid, data in video_models.items():
            data["head_checkpoints"].sort(key=lambda x: os.path.getmtime(os.path.join(WORK_DIR, x)) if os.path.exists(
                os.path.join(WORK_DIR, x)) else 0, reverse=True)
            data["torso_checkpoints"].sort(key=lambda x: os.path.getmtime(os.path.join(WORK_DIR, x)) if os.path.exists(
                os.path.join(WORK_DIR, x)) else 0, reverse=True)

        models = list(video_models.values())

    log(f"找到 {len(models)} 个模型组")
    return jsonify(models)


@app.route('/audios', methods=['GET'])
def list_audios():
    """列出可用的音频文件"""
    audios = []

    # 扫描多个目录
    audio_dirs = [
        os.path.join(WORK_DIR, "data", "raw", "val_wavs"),
        os.path.join(WORK_DIR, "data", "raw", "audios"),
        os.path.join(WORK_DIR, "audios"),
    ]

    for audio_dir in audio_dirs:
        if os.path.exists(audio_dir):
            for f in os.listdir(audio_dir):
                if f.endswith(('.wav', '.mp3', '.m4a')):
                    audio_path = os.path.join(audio_dir, f)
                    rel_path = os.path.relpath(audio_path, WORK_DIR)

                    # 获取音频时长（可选）
                    duration = None
                    try:
                        import wave
                        if f.endswith('.wav'):
                            with wave.open(audio_path, 'r') as w:
                                frames = w.getnframes()
                                rate = w.getframerate()
                                duration = round(frames / rate, 1)
                    except:
                        pass

                    audios.append({
                        "name": f,
                        "path": rel_path,
                        "duration": duration,
                        "dir": os.path.basename(audio_dir)
                    })

    log(f"找到 {len(audios)} 个音频文件")
    return jsonify(audios)


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """上传音频文件"""
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "没有音频文件"}), 400

    audio_file = request.files['audio']
    audio_name = request.form.get('audio_name')

    if not audio_name:
        audio_name = os.path.splitext(audio_file.filename)[0]

    # 清理文件名
    audio_name = re.sub(r'[^a-zA-Z0-9_-]', '_', audio_name)

    # 保存到音频目录
    audio_dir = os.path.join(WORK_DIR, "data", "raw", "val_wavs")
    os.makedirs(audio_dir, exist_ok=True)

    # 获取扩展名
    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in ['.wav', '.mp3', '.m4a']:
        ext = '.wav'

    save_path = os.path.join(audio_dir, f"{audio_name}{ext}")
    audio_file.save(save_path)

    rel_path = os.path.relpath(save_path, WORK_DIR)
    log(f"音频已上传: {save_path}")

    return jsonify({
        "status": "success",
        "audio_name": f"{audio_name}{ext}",
        "audio_path": rel_path
    })


@app.route('/infer', methods=['POST'])
def inference():
    """推理生成视频"""
    data = request.json
    video_id = data.get('video_id')
    head_ckpt = data.get('head_ckpt')
    torso_ckpt = data.get('torso_ckpt')
    audio_path = data.get('audio_path')
    gpu_id = data.get('gpu_id', '0')

    log(f"收到推理请求: head={head_ckpt}, torso={torso_ckpt}, audio={audio_path}")

    # 参数验证
    if not head_ckpt or not torso_ckpt:
        return jsonify({"status": "error", "message": "缺少模型路径"}), 400

    if not audio_path:
        return jsonify({"status": "error", "message": "缺少音频路径"}), 400

    # 检查文件是否存在
    audio_full = os.path.join(WORK_DIR, audio_path)
    if not os.path.exists(audio_full):
        return jsonify({"status": "error", "message": f"音频文件不存在: {audio_path}"}), 404

    # ================= 修改开始 =================
    # 修复：不要去掉 checkpoints/ 前缀
    # 确保路径是相对于工作目录的正确路径
    head_exp = head_ckpt
    torso_exp = torso_ckpt

    # 如果路径不包含 checkpoints 但文件确实在那里，可以尝试自动补全（可选）
    if not head_exp.startswith('checkpoints/') and not os.path.exists(os.path.join(WORK_DIR, head_exp)):
        if os.path.exists(os.path.join(WORK_DIR, 'checkpoints', head_exp)):
            head_exp = os.path.join('checkpoints', head_exp)

    if not torso_exp.startswith('checkpoints/') and not os.path.exists(os.path.join(WORK_DIR, torso_exp)):
        if os.path.exists(os.path.join(WORK_DIR, 'checkpoints', torso_exp)):
            torso_exp = os.path.join('checkpoints', torso_exp)
    # ================= 修改结束 =================

    env = {"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": "./"}

    # 生成输出文件名
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    model_name = os.path.basename(torso_ckpt).replace('_torso', '')
    output_name = f"{model_name}_{audio_name}"

    cmd = f'python inference/genefacepp_infer.py --head_ckpt={head_exp} --torso_ckpt={torso_exp} --drv_aud={audio_path}'

    log(f"执行推理命令: {cmd}")
    result = run_command(cmd, WORK_DIR, env)

    if result["returncode"] != 0:
        return jsonify({
            "status": "error",
            "message": "推理失败",
            "error": result["stderr"][-500:] if result["stderr"] else "未知错误"
        }), 500

    # 查找生成的视频文件
    output_dir = os.path.join(WORK_DIR, "infer_out")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 检查是否生成了 tmp.mp4 (某些版本的推理脚本默认输出这个)
    tmp_video_path = os.path.join(WORK_DIR, "tmp.mp4")
    if os.path.exists(tmp_video_path):
        # 生成一个优雅的文件名: 模型名_音频名_时间戳.mp4
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        elegant_name = f"{output_name}_{timestamp}.mp4"
        target_path = os.path.join(output_dir, elegant_name)

        shutil.move(tmp_video_path, target_path)
        log(f"检测到 tmp.mp4，已移动并重命名为: {target_path}")

    # 2. 查找 infer_out 中最新的视频 (包含刚才移动进去的)
    output_video = None
    if os.path.exists(output_dir):
        # 查找最新生成的视频
        mp4_files = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith('.mp4'):
                    full_path = os.path.join(root, f)
                    mp4_files.append((full_path, os.path.getmtime(full_path)))

        if mp4_files:
            # 按修改时间排序，取最新的
            mp4_files.sort(key=lambda x: x[1], reverse=True)
            output_video = mp4_files[0][0]
            log(f"找到输出视频: {output_video}")

    if output_video:
        rel_path = os.path.relpath(output_video, WORK_DIR)
        return jsonify({
            "status": "success",
            "message": "推理完成",
            "video_path": rel_path,
            "video_name": os.path.basename(output_video)
        })
    else:
        return jsonify({
            "status": "success",
            "message": "推理完成，但未找到输出视频",
            "stdout": result["stdout"][-500:] if result["stdout"] else ""
        })


if __name__ == '__main__':
    log("=" * 60)
    log("GeneFace++ API Server 启动")
    log(f"工作目录: {WORK_DIR}")
    log("=" * 60)

    if os.path.exists(WORK_DIR):
        log(f"工作目录内容: {os.listdir(WORK_DIR)[:10]}")

    load_tasks()
    app.run(host='0.0.0.0', port=7860, debug=False, threaded=True)

