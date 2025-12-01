# backend/model_trainer.py
import subprocess
import os
import time
import requests
import shutil

# ==================== 路径配置 ====================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# GeneFace++ 配置
GENEFACE_API_URL = os.environ.get("GENEFACE_API_URL", "http://localhost:7869")
GENEFACE_DIR = os.path.join(BASE_DIR, "GeneFace")
GENEFACE_VIDEO_DIR = os.path.join(GENEFACE_DIR, "data", "raw", "videos")

os.makedirs(GENEFACE_VIDEO_DIR, exist_ok=True)


def train_model(data):
    """模型训练入口"""
    print("[model_trainer] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    model_choice = data.get('model_choice')

    if model_choice == "GeneFace++":
        return train_geneface(data)
    elif model_choice == "SyncTalk":
        return train_synctalk(data)
    else:
        return {"status": "error", "message": f"不支持的模型: {model_choice}"}


# ==================== SyncTalk ====================

def train_synctalk(data):
    """SyncTalk 模型训练"""
    try:
        cmd = [
            "./SyncTalk/run_synctalk.sh", "train",
            "--video_path", data['ref_video'],
            "--gpu", data['gpu_choice'],
            "--epochs", data['epoch']
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"status": "success", "message": "SyncTalk 训练完成"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"训练失败: {e.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==================== GeneFace++ ====================

def geneface_health():
    """检查 GeneFace++ 服务"""
    try:
        r = requests.get(f"{GENEFACE_API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False


def geneface_list_videos():
    """获取可用视频列表"""
    try:
        r = requests.get(f"{GENEFACE_API_URL}/videos", timeout=10)
        return r.json()
    except:
        return []


def geneface_upload_video(local_path, video_id):
    """上传视频到 GeneFace++"""
    target = os.path.join(GENEFACE_VIDEO_DIR, f"{video_id}.mp4")
    try:
        shutil.copy2(local_path, target)
        return True
    except:
        return False


def geneface_api_call(endpoint, method="GET", data=None):
    """调用 GeneFace++ API"""
    url = f"{GENEFACE_API_URL}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=30)
        else:
            r = requests.post(url, json=data, timeout=30)
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def train_geneface(data):
    """
    GeneFace++ 训练入口

    train_stage:
    - preprocess: 数据预处理
    - head: 头部模型训练
    - torso: 身体模型训练
    - status: 查询任务状态
    - list: 列出任务
    - videos: 列出可用视频
    - models: 列出已训练模型
    """
    if not geneface_health():
        return {
            "status": "error",
            "message": "GeneFace++ 服务不可用，请运行: ./start_geneface.sh"
        }

    video_path = data.get('ref_video')
    video_id = data.get('video_id')
    gpu_id = data.get('gpu_choice', '0').replace('GPU', '')
    train_stage = data.get('train_stage', 'preprocess')
    task_id = data.get('task_id')
    head_ckpt = data.get('head_ckpt')
    max_updates_head = int(data.get('max_updates_head', 40000))
    max_updates_torso = int(data.get('max_updates_torso', 40000))

    if not video_id and video_path:
        video_id = os.path.splitext(os.path.basename(video_path))[0]

    print(f"[GeneFace] video_id={video_id}, stage={train_stage}, gpu={gpu_id}")

    # 查询任务状态
    if train_stage == "status":
        if not task_id:
            return {"status": "error", "message": "需要 task_id"}
        return geneface_api_call(f"/task/{task_id}")

    # 列出任务
    if train_stage == "list":
        return {"status": "success", "tasks": geneface_api_call("/tasks")}

    # 列出可用视频
    if train_stage == "videos":
        return {"status": "success", "videos": geneface_api_call("/videos")}

    # 列出模型
    if train_stage == "models":
        return {"status": "success", "models": geneface_api_call("/models")}

    # 数据预处理
    if train_stage == "preprocess":
        if video_path and os.path.exists(video_path):
            if not geneface_upload_video(video_path, video_id):
                return {"status": "error", "message": "视频上传失败"}

        return geneface_api_call("/preprocess", "POST", {
            "video_id": video_id,
            "gpu_id": gpu_id,
            "max_updates_head": max_updates_head,
            "max_updates_torso": max_updates_torso
        })

    # 头部训练
    if train_stage == "head":
        return geneface_api_call("/train/head", "POST", {
            "video_id": video_id,
            "gpu_id": gpu_id
        })

    # 身体训练
    if train_stage == "torso":
        if not head_ckpt:
            head_ckpt = f"checkpoints/{video_id}/lm3d_radnerf"
        return geneface_api_call("/train/torso", "POST", {
            "video_id": video_id,
            "head_ckpt": head_ckpt,
            "gpu_id": gpu_id
        })

    return {"status": "error", "message": f"未知阶段: {train_stage}"}