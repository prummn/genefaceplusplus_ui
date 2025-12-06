# backend/video_generator.py
import os
import time
import subprocess
import shutil
import requests

GENEFACE_API_URL = os.environ.get("GENEFACE_API_URL", "http://localhost:7869")


def generate_video(data):
    """
    视频生成逻辑
    """
    print("[backend.video_generator] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    model_name = data.get('model_name')

    if model_name == "GeneFace++":
        return generate_geneface_video(data)
    elif model_name == "SyncTalk":
        return generate_synctalk_video(data)
    else:
        print(f"[backend.video_generator] 未知模型: {model_name}")
        return os.path.join("static", "videos", "out.mp4")


def generate_geneface_video(data):
    """GeneFace++ 推理"""
    head_ckpt = data.get('gf_head_ckpt')
    torso_ckpt = data.get('gf_torso_ckpt')
    audio_path = data.get('gf_audio_path')
    gpu_choice = data.get('gpu_choice', 'GPU0')
    gpu_id = gpu_choice.replace('GPU', '')

    print(f"[GeneFace++] head={head_ckpt}, torso={torso_ckpt}, audio={audio_path}")

    if not head_ckpt or not torso_ckpt or not audio_path:
        print("[GeneFace++] 缺少参数")
        return os.path.join("static", "videos", "out.mp4")

    try:
        # 调用 GeneFace++ API
        response = requests.post(
            f"{GENEFACE_API_URL}/infer",
            json={
                "head_ckpt": head_ckpt,
                "torso_ckpt": torso_ckpt,
                "audio_path": audio_path,
                "gpu_id": gpu_id
            },
            timeout=600  # 推理可能需要较长时间
        )

        result = response.json()
        print(f"[GeneFace++] API 响应: {result}")

        if result.get('status') == 'success' and result.get('video_path'):
            # 复制视频到 static 目录
            video_rel_path = result['video_path']
            source_path = os.path.join(os.path.dirname(__file__), '..', 'GeneFace', video_rel_path)

            # 生成目标文件名 (保留历史记录)
            video_name = result.get('video_name', 'geneface_output.mp4')
            dest_path = os.path.join("static", "videos", video_name)

            # ================= 修改开始 =================
            if os.path.exists(source_path):
                # 1. 复制为唯一文件名 (用于下载或历史)
                shutil.copy(source_path, dest_path)
                print(f"[GeneFace++] 视频已复制到: {dest_path}")

                # 2. 复制为 "latest" 文件 (用于页面默认加载)
                latest_path = os.path.join("static", "videos", "geneface_latest.mp4")
                shutil.copy(source_path, latest_path)
                print(f"[GeneFace++] 更新了最新视频缓存: {latest_path}")

                # 这里的返回值决定了当前这次请求前端播放哪个视频
                # 返回 dest_path 或 latest_path 都可以，这里返回 dest_path 确保文件名不重复
                return dest_path
            else:
                print(f"[GeneFace++] 源视频不存在: {source_path}")
                return os.path.join("static", "videos", "out.mp4")
            # ================= 修改结束 =================
        else:
            print(f"[GeneFace++] 推理失败: {result.get('message', '未知错误')}")
            return os.path.join("static", "videos", "out.mp4")

    except requests.exceptions.Timeout:
        print("[GeneFace++] 请求超时")
        return os.path.join("static", "videos", "out.mp4")
    except Exception as e:
        print(f"[GeneFace++] 错误: {e}")
        return os.path.join("static", "videos", "out.mp4")


def generate_synctalk_video(data):
    """SyncTalk 推理"""
    try:
        cmd = [
            './SyncTalk/run_synctalk.sh', 'infer',
            '--model_dir', data['model_param'],
            '--audio_path', data['ref_audio'],
            '--gpu', data['gpu_choice']
        ]

        print(f"[backend.video_generator] 执行命令: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        print("命令标准输出:", result.stdout)
        if result.stderr:
            print("命令标准错误:", result.stderr)

        # 查找输出视频
        model_dir_name = os.path.basename(data['model_param'])
        source_path = os.path.join("SyncTalk", "model", model_dir_name, "results", "test_audio.mp4")
        audio_name = os.path.splitext(os.path.basename(data['ref_audio']))[0]
        video_filename = f"{model_dir_name}_{audio_name}.mp4"
        destination_path = os.path.join("static", "videos", video_filename)

        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
            print(f"[backend.video_generator] 视频生成完成: {destination_path}")
            return destination_path
        else:
            print(f"[backend.video_generator] 视频文件不存在: {source_path}")
            # 尝试查找最新的 mp4
            results_dir = os.path.join("SyncTalk", "model", model_dir_name, "results")
            if os.path.exists(results_dir):
                mp4_files = [f for f in os.listdir(results_dir) if f.endswith('.mp4')]
                if mp4_files:
                    latest_file = max(mp4_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
                    source_path = os.path.join(results_dir, latest_file)
                    shutil.copy(source_path, destination_path)
                    return destination_path

            return os.path.join("static", "videos", "out.mp4")

    except subprocess.CalledProcessError as e:
        print(f"[backend.video_generator] 命令执行失败: {e}")
        return os.path.join("static", "videos", "out.mp4")
    except Exception as e:
        print(f"[backend.video_generator] 其他错误: {e}")
        return os.path.join("static", "videos", "out.mp4")