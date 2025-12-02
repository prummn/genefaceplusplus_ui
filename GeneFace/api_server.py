# GeneFace/api_server.py
"""
GeneFace++ API 服务
"""

from flask import Flask, request, jsonify
import subprocess
import os
import select
import functools
import sys
import threading
import time
import uuid
import json
import shutil
import re

print = functools.partial(print, flush=True)

app = Flask(__name__)

# ==================== 配置 ====================
WORK_DIR = "/data/geneface"
TASKS_FILE = os.path.join(WORK_DIR, "tasks_state.json")
TEMPLATE_CONFIG_DIR = os.path.join(WORK_DIR, "egs", "datasets", "May")  # 模板配置目录

# 任务状态存储
tasks = {}


def load_tasks():
    """从文件加载任务状态"""
    global tasks
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
        except:
            tasks = {}


def save_tasks():
    """保存任务状态到文件"""
    try:
        with open(TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[API] 保存任务状态失败: {e}")


def run_command(cmd, cwd=None, env=None, timeout=None):
    """执行命令并实时输出"""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    # 让子进程认为它在终端中运行（显示进度条）
    full_env['PYTHONUNBUFFERED'] = '1'

    print(f"[CMD] 执行: {cmd}")

    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd or WORK_DIR,
            env=full_env,
            bufsize=1,  # 行缓冲
        )

        stdout_lines = []
        stderr_lines = []

        # 实时读取输出
        while True:
            # 使用 select 同时监控 stdout 和 stderr
            reads = [process.stdout, process.stderr]
            readable, _, _ = select.select(reads, [], [], 0.1)

            for stream in readable:
                line = stream.readline()
                if line:
                    if stream == process.stdout:
                        stdout_lines.append(line)
                        print(f"[OUT] {line.rstrip()}")
                    else:
                        stderr_lines.append(line)
                        # ffmpeg 进度信息在 stderr
                        print(f"[ERR] {line.rstrip()}")

            # 检查进程是否结束
            if process.poll() is not None:
                # 读取剩余输出
                for line in process.stdout:
                    stdout_lines.append(line)
                    print(f"[OUT] {line.rstrip()}")
                for line in process.stderr:
                    stderr_lines.append(line)
                    print(f"[ERR] {line.rstrip()}")
                break

        returncode = process.returncode
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)

        if returncode != 0:
            print(f"[CMD] 命令失败，返回码: {returncode}")

        return {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }

    except Exception as e:
        print(f"[CMD] 异常: {e}")
        return {"returncode": -1, "stdout": "", "stderr": str(e)}

def update_task(task_id, **kwargs):
    """更新任务状态"""
    if task_id in tasks:
        tasks[task_id].update(kwargs)
        tasks[task_id]["updated_at"] = time.time()
        save_tasks()


def generate_config_files(video_id, max_updates_head=40000, max_updates_torso=40000):
    """
    从 May 模板复制并修改配置文件

    Args:
        video_id: 视频标识符
        max_updates_head: 头部模型训练步数
        max_updates_torso: 身体模型训练步数
    """
    target_config_dir = os.path.join(WORK_DIR, "egs", "datasets", video_id)

    # 如果目标目录已存在，先删除
    if os.path.exists(target_config_dir):
        shutil.rmtree(target_config_dir)

    # 复制整个模板目录
    shutil.copytree(TEMPLATE_CONFIG_DIR, target_config_dir)
    print(f"[Config] 已复制模板配置到: {target_config_dir}")

    # 修改 lm3d_radnerf.yaml（头部模型配置）
    head_config_path = os.path.join(target_config_dir, "lm3d_radnerf.yaml")
    if os.path.exists(head_config_path):
        with open(head_config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换 video_id
        content = re.sub(r'video_id:\s*May', f'video_id: {video_id}', content)
        # 替换 max_updates（如果存在）
        content = re.sub(r'max_updates:\s*\d+', f'max_updates: {max_updates_head}', content)

        with open(head_config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[Config] 已更新: lm3d_radnerf.yaml")

    # 修改 lm3d_radnerf_torso.yaml（身体模型配置）
    torso_config_path = os.path.join(target_config_dir, "lm3d_radnerf_torso.yaml")
    if os.path.exists(torso_config_path):
        with open(torso_config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换 video_id
        content = re.sub(r'video_id:\s*May', f'video_id: {video_id}', content)
        # 替换 head_model_dir 路径
        content = re.sub(
            r'head_model_dir:\s*checkpoints/May/lm3d_radnerf',
            f'head_model_dir: checkpoints/{video_id}/lm3d_radnerf',
            content
        )
        # 替换 max_updates
        content = re.sub(r'max_updates:\s*\d+', f'max_updates: {max_updates_torso}', content)

        with open(torso_config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[Config] 已更新: lm3d_radnerf_torso.yaml")

    # 修改 lm3d_radnerf_sr.yaml（如果存在）
    sr_config_path = os.path.join(target_config_dir, "lm3d_radnerf_sr.yaml")
    if os.path.exists(sr_config_path):
        with open(sr_config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'video_id:\s*May', f'video_id: {video_id}', content)
        content = re.sub(r'max_updates:\s*\d+', f'max_updates: {max_updates_head}', content)
        with open(sr_config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[Config] 已更新: lm3d_radnerf_sr.yaml")

    # 修改 lm3d_radnerf_torso_sr.yaml（如果存在）
    torso_sr_config_path = os.path.join(target_config_dir, "lm3d_radnerf_torso_sr.yaml")
    if os.path.exists(torso_sr_config_path):
        with open(torso_sr_config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'video_id:\s*May', f'video_id: {video_id}', content)
        content = re.sub(
            r'head_model_dir:\s*checkpoints/May/lm3d_radnerf',
            f'head_model_dir: checkpoints/{video_id}/lm3d_radnerf',
            content
        )
        content = re.sub(r'max_updates:\s*\d+', f'max_updates: {max_updates_torso}', content)
        with open(torso_sr_config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[Config] 已更新: lm3d_radnerf_torso_sr.yaml")

    print(f"[Config] 配置文件生成完成: {video_id}")
    return True


# ==================== 预处理工作线程 ====================
def preprocessing_worker(task_id, video_id, gpu_id="0", max_updates_head=40000, max_updates_torso=40000):
    """数据预处理后台任务"""
    update_task(task_id, status="running", current_step="初始化")

    env = {
        "VIDEO_ID": video_id,
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "PYTHONPATH": "./"
    }

    try:
        # ============ 步骤 1: 视频缩放 ============
        update_task(task_id, current_step="1/10 视频缩放到512x512")

        original_video = f"data/raw/videos/{video_id}.mp4"
        if not os.path.exists(os.path.join(WORK_DIR, original_video)):
            raise Exception(f"视频文件不存在: {original_video}")

        cmd = f'ffmpeg -y -i data/raw/videos/{video_id}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 data/raw/videos/{video_id}_512.mp4'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"视频缩放失败: {result['stderr']}")

        run_command(f'mv data/raw/videos/{video_id}.mp4 data/raw/videos/{video_id}_original.mp4', WORK_DIR, env)
        run_command(f'mv data/raw/videos/{video_id}_512.mp4 data/raw/videos/{video_id}.mp4', WORK_DIR, env)

        # ============ 步骤 2: 提取音频 ============
        update_task(task_id, current_step="2/10 提取音频")

        run_command(f'mkdir -p data/processed/videos/{video_id}', WORK_DIR, env)
        cmd = f'ffmpeg -y -i data/raw/videos/{video_id}.mp4 -f wav -ar 16000 data/processed/videos/{video_id}/aud.wav'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"音频提取失败: {result['stderr']}")

        # ============ 步骤 3: 提取 HuBERT 特征 ============
        update_task(task_id, current_step="3/10 提取HuBERT特征")

        cmd = f'python data_gen/utils/process_audio/extract_hubert.py --video_id={video_id}'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"HuBERT提取失败: {result['stderr']}")

        # ============ 步骤 4: 提取 Mel 和 F0 ============
        update_task(task_id, current_step="4/10 提取Mel和F0特征")

        cmd = f'python data_gen/utils/process_audio/extract_mel_f0.py --video_id={video_id}'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"Mel/F0提取失败: {result['stderr']}")

        # ============ 步骤 5: 提取图片帧 ============
        update_task(task_id, current_step="5/10 提取视频帧")

        run_command(f'mkdir -p data/processed/videos/{video_id}/gt_imgs', WORK_DIR, env)
        cmd = f'ffmpeg -y -i data/raw/videos/{video_id}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 data/processed/videos/{video_id}/gt_imgs/%08d.jpg'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"图片帧提取失败: {result['stderr']}")

        # ============ 步骤 6: 提取分割图和背景 ============
        update_task(task_id, current_step="6/10 提取分割图和背景")

        cmd = f'python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/{video_id}.mp4'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"分割图提取失败: {result['stderr']}")

        # ============ 步骤 7: 提取 2D 关键点 ============
        update_task(task_id, current_step="7/10 提取2D面部关键点")

        cmd = f'python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/{video_id}.mp4'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"2D关键点提取失败: {result['stderr']}")

        # ============ 步骤 8: 拟合 3DMM ============
        update_task(task_id, current_step="8/10 拟合3DMM人脸模型")

        cmd = f'python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/{video_id}.mp4 --reset --id_mode=global'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"3DMM拟合失败: {result['stderr']}")

        # ============ 步骤 9: 二值化数据 ============
        update_task(task_id, current_step="9/10 二值化处理数据")

        cmd = f'python data_gen/runs/binarizer_nerf.py --video_id={video_id}'
        result = run_command(cmd, WORK_DIR, env)
        if result["returncode"] != 0:
            raise Exception(f"二值化失败: {result['stderr']}")

        # ============ 步骤 10: 生成配置文件 ============
        update_task(task_id, current_step="10/10 生成训练配置文件")
        generate_config_files(video_id, max_updates_head, max_updates_torso)

        # ============ 完成 ============
        update_task(
            task_id,
            status="completed",
            current_step="预处理完成",
            message="数据预处理完成，可以开始训练头部模型",
            next_stage="head"
        )
        print(f"[Task {task_id}] 预处理完成")

    except Exception as e:
        error_msg = str(e)
        print(f"[Task {task_id}] 失败: {error_msg}")
        update_task(task_id, status="failed", error=error_msg)


# ==================== 头部训练工作线程 ====================
def head_training_worker(task_id, video_id, gpu_id="0"):
    """头部模型训练"""
    update_task(task_id, status="running", current_step="训练头部模型")

    env = {
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "PYTHONPATH": "./"
    }

    try:
        # 使用 sr 版本的配置（如果存在）
        config_path = f"egs/datasets/{video_id}/lm3d_radnerf_sr.yaml"
        if not os.path.exists(os.path.join(WORK_DIR, config_path)):
            config_path = f"egs/datasets/{video_id}/lm3d_radnerf.yaml"

        cmd = f'python tasks/run.py --config={config_path} --exp_name={video_id}/lm3d_radnerf --reset'
        result = run_command(cmd, WORK_DIR, env)

        if result["returncode"] != 0:
            raise Exception(f"头部模型训练失败: {result['stderr']}")

        head_ckpt = f"checkpoints/{video_id}/lm3d_radnerf"

        update_task(
            task_id,
            status="completed",
            current_step="头部模型训练完成",
            head_ckpt=head_ckpt,
            message="头部模型训练完成，可以开始训练身体模型",
            next_stage="torso"
        )
        print(f"[Task {task_id}] 头部训练完成")

    except Exception as e:
        error_msg = str(e)
        print(f"[Task {task_id}] 头部训练失败: {error_msg}")
        update_task(task_id, status="failed", error=error_msg)


# ==================== 身体训练工作线程 ====================
def torso_training_worker(task_id, video_id, head_ckpt, gpu_id="0"):
    """身体模型训练"""
    update_task(task_id, status="running", current_step="训练身体模型")

    env = {
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "PYTHONPATH": "./"
    }

    try:
        # 使用 sr 版本的配置（如果存在）
        config_path = f"egs/datasets/{video_id}/lm3d_radnerf_torso_sr.yaml"
        if not os.path.exists(os.path.join(WORK_DIR, config_path)):
            config_path = f"egs/datasets/{video_id}/lm3d_radnerf_torso.yaml"

        cmd = f'python tasks/run.py --config={config_path} --exp_name={video_id}/lm3d_radnerf_torso --hparams=head_model_dir={head_ckpt} --reset'
        result = run_command(cmd, WORK_DIR, env)

        if result["returncode"] != 0:
            raise Exception(f"身体模型训练失败: {result['stderr']}")

        torso_ckpt = f"checkpoints/{video_id}/lm3d_radnerf_torso"

        update_task(
            task_id,
            status="completed",
            current_step="全部训练完成",
            torso_ckpt=torso_ckpt,
            head_ckpt=head_ckpt,
            message="训练完成！可以进行推理生成视频了"
        )
        print(f"[Task {task_id}] 身体训练完成")

    except Exception as e:
        error_msg = str(e)
        print(f"[Task {task_id}] 身体训练失败: {error_msg}")
        update_task(task_id, status="failed", error=error_msg)


# ==================== API 路由 ====================

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "service": "GeneFace++ API",
        "work_dir": WORK_DIR
    })


@app.route('/videos', methods=['GET'])
def list_videos():
    """列出可用的视频文件"""
    video_dir = os.path.join(WORK_DIR, "data", "raw", "videos")
    videos = []

    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith('.mp4') and not f.endswith('_original.mp4') and not f.endswith('_512.mp4'):
                video_path = os.path.join(video_dir, f)
                video_id = os.path.splitext(f)[0]

                # 获取文件信息
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
    """上传视频文件"""
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "没有视频文件"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"status": "error", "message": "没有选择文件"}), 400

    # 获取 video_id（可选，默认使用文件名）
    video_id = request.form.get('video_id')
    if not video_id:
        video_id = os.path.splitext(video_file.filename)[0]

    # 清理 video_id（只保留字母数字和下划线）
    video_id = re.sub(r'[^a-zA-Z0-9_]', '_', video_id)

    # 保存文件
    video_dir = os.path.join(WORK_DIR, "data", "raw", "videos")
    os.makedirs(video_dir, exist_ok=True)

    save_path = os.path.join(video_dir, f"{video_id}.mp4")
    video_file.save(save_path)

    print(f"[Upload] 视频已保存: {save_path}")

    return jsonify({
        "status": "success",
        "message": "视频上传成功",
        "video_id": video_id,
        "path": save_path
    })


@app.route('/preprocess', methods=['POST'])
def start_preprocessing():
    """
    启动数据预处理
    请求体: {
        "video_id": "xxx",
        "gpu_id": "0",
        "max_updates_head": 40000,
        "max_updates_torso": 40000
    }
    """
    data = request.json
    video_id = data.get('video_id')
    gpu_id = data.get('gpu_id', '0')
    max_updates_head = int(data.get('max_updates_head', 40000))
    max_updates_torso = int(data.get('max_updates_torso', 40000))

    if not video_id:
        return jsonify({"status": "error", "message": "缺少 video_id"}), 400

    # 检查视频文件是否存在
    video_path = os.path.join(WORK_DIR, f"data/raw/videos/{video_id}.mp4")
    if not os.path.exists(video_path):
        return jsonify({
            "status": "error",
            "message": f"视频文件不存在: data/raw/videos/{video_id}.mp4"
        }), 404

    # 创建任务
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "task_id": task_id,
        "type": "preprocess",
        "video_id": video_id,
        "gpu_id": gpu_id,
        "max_updates_head": max_updates_head,
        "max_updates_torso": max_updates_torso,
        "status": "pending",
        "current_step": "等待开始",
        "created_at": time.time()
    }
    save_tasks()

    # 启动后台线程
    thread = threading.Thread(
        target=preprocessing_worker,
        args=(task_id, video_id, gpu_id, max_updates_head, max_updates_torso)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        "status": "success",
        "task_id": task_id,
        "message": "预处理任务已启动",
        "video_id": video_id
    })


@app.route('/train/head', methods=['POST'])
def start_head_training():
    """启动头部模型训练"""
    data = request.json
    video_id = data.get('video_id')
    gpu_id = data.get('gpu_id', '0')

    if not video_id:
        return jsonify({"status": "error", "message": "缺少 video_id"}), 400

    # 检查预处理数据是否存在
    processed_dir = os.path.join(WORK_DIR, f"data/processed/videos/{video_id}")
    if not os.path.exists(processed_dir):
        return jsonify({
            "status": "error",
            "message": "预处理数据不存在，请先进行数据预处理"
        }), 400

    # 检查配置文件是否存在
    config_dir = os.path.join(WORK_DIR, f"egs/datasets/{video_id}")
    if not os.path.exists(config_dir):
        return jsonify({
            "status": "error",
            "message": "配置文件不存在，请先进行数据预处理"
        }), 400

    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "task_id": task_id,
        "type": "head_training",
        "video_id": video_id,
        "gpu_id": gpu_id,
        "status": "pending",
        "current_step": "等待开始",
        "created_at": time.time()
    }
    save_tasks()

    thread = threading.Thread(target=head_training_worker, args=(task_id, video_id, gpu_id))
    thread.daemon = True
    thread.start()

    return jsonify({
        "status": "success",
        "task_id": task_id,
        "message": "头部模型训练已启动",
        "video_id": video_id
    })


@app.route('/train/torso', methods=['POST'])
def start_torso_training():
    """启动身体模型训练"""
    data = request.json
    video_id = data.get('video_id')
    head_ckpt = data.get('head_ckpt')
    gpu_id = data.get('gpu_id', '0')

    if not video_id:
        return jsonify({"status": "error", "message": "缺少 video_id"}), 400

    if not head_ckpt:
        head_ckpt = f"checkpoints/{video_id}/lm3d_radnerf"

    # 检查头部模型是否存在
    head_ckpt_dir = os.path.join(WORK_DIR, head_ckpt)
    if not os.path.exists(head_ckpt_dir):
        return jsonify({
            "status": "error",
            "message": f"头部模型不存在: {head_ckpt}，请先训练头部模型"
        }), 400

    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "task_id": task_id,
        "type": "torso_training",
        "video_id": video_id,
        "head_ckpt": head_ckpt,
        "gpu_id": gpu_id,
        "status": "pending",
        "current_step": "等待开始",
        "created_at": time.time()
    }
    save_tasks()

    thread = threading.Thread(target=torso_training_worker, args=(task_id, video_id, head_ckpt, gpu_id))
    thread.daemon = True
    thread.start()

    return jsonify({
        "status": "success",
        "task_id": task_id,
        "message": "身体模型训练已启动",
        "video_id": video_id
    })


@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """查询任务状态"""
    if task_id not in tasks:
        return jsonify({"status": "error", "message": "任务不存在"}), 404
    return jsonify(tasks[task_id])


@app.route('/tasks', methods=['GET'])
def list_tasks():
    """列出所有任务"""
    sorted_tasks = sorted(tasks.values(), key=lambda x: x.get('created_at', 0), reverse=True)
    return jsonify(sorted_tasks)


@app.route('/tasks/clear', methods=['POST'])
def clear_tasks():
    """清除所有已完成/失败的任务"""
    global tasks
    tasks = {k: v for k, v in tasks.items() if v.get('status') in ['pending', 'running']}
    save_tasks()
    return jsonify({"status": "success", "message": "已清除完成的任务"})


@app.route('/models', methods=['GET'])
def list_models():
    """列出已训练的模型"""
    models = []
    ckpt_dir = os.path.join(WORK_DIR, "checkpoints")

    if os.path.exists(ckpt_dir):
        for video_id in os.listdir(ckpt_dir):
            video_ckpt_dir = os.path.join(ckpt_dir, video_id)
            if os.path.isdir(video_ckpt_dir):
                model_info = {
                    "video_id": video_id,
                    "head_model": None,
                    "torso_model": None
                }

                # 检查头部模型
                head_dir = os.path.join(video_ckpt_dir, "lm3d_radnerf")
                if os.path.exists(head_dir):
                    model_info["head_model"] = f"checkpoints/{video_id}/lm3d_radnerf"

                # 检查身体模型
                torso_dir = os.path.join(video_ckpt_dir, "lm3d_radnerf_torso")
                if os.path.exists(torso_dir):
                    model_info["torso_model"] = f"checkpoints/{video_id}/lm3d_radnerf_torso"

                if model_info["head_model"] or model_info["torso_model"]:
                    models.append(model_info)

    return jsonify(models)


@app.route('/infer', methods=['POST'])
def inference():
    """推理生成视频"""
    data = request.json
    video_id = data.get('video_id')
    head_ckpt = data.get('head_ckpt')
    torso_ckpt = data.get('torso_ckpt')
    audio_path = data.get('audio_path')
    gpu_id = data.get('gpu_id', '0')

    if video_id and not head_ckpt:
        head_ckpt = f"{video_id}/lm3d_radnerf"
    if video_id and not torso_ckpt:
        torso_ckpt = f"{video_id}/lm3d_radnerf_torso"

    if not all([head_ckpt, torso_ckpt, audio_path]):
        return jsonify({
            "status": "error",
            "message": "缺少必要参数: head_ckpt, torso_ckpt, audio_path"
        }), 400

    env = {
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "PYTHONPATH": "./"
    }

    cmd = f'python inference/genefacepp_infer.py --head_ckpt={head_ckpt} --torso_ckpt={torso_ckpt} --drv_aud={audio_path}'
    result = run_command(cmd, WORK_DIR, env)

    if result["returncode"] != 0:
        return jsonify({
            "status": "error",
            "error": result["stderr"],
            "stdout": result["stdout"]
        }), 500

    return jsonify({
        "status": "success",
        "message": "推理完成",
        "stdout": result["stdout"]
    })


# ==================== 启动 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("  GeneFace++ API Server")
    print("=" * 60)
    print(f"  工作目录: {WORK_DIR}")
    print(f"  模板配置: {TEMPLATE_CONFIG_DIR}")
    print(f"  端口: 7860")
    print("=" * 60)

    load_tasks()

    app.run(host='0.0.0.0', port=7860, debug=False, threaded=True)