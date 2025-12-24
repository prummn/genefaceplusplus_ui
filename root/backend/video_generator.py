import os
import time
import subprocess
import shutil
import requests
import uuid
from pydub import AudioSegment

# 基础路径
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENEFACE_API_URL = os.environ.get("GENEFACE_API_URL", "http://localhost:7869")

# RVC 配置
RVC_DIR = os.path.join(BASE_DIR, "RVC")
RVC_IO_DIR = os.path.join(BASE_DIR, "io")
RVC_MODELS_DIR = os.path.join(RVC_DIR, "models_zh")
AUDIO_OUTPUT_DIR = os.path.join(BASE_DIR, "io", "output")
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


def simple_splitter(text, max_len=45):
    """简单的文本切分器 (从 chat_engine 复制)"""
    import re
    segments = re.findall(r"([^。！？，、,!?]+[。！？，、,!?]?)", text, re.UNICODE)
    if not segments:
        return [text[i:i + max_len] for i in range(0, len(text), max_len)]
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


def generate_audio_from_text(text, voice_choice='zhb'):
    """
    使用 RVC Docker 生成音频
    """
    print(f"[video_generator] 开始文本生成语音: {text[:20]}... (Voice: {voice_choice})")

    # 准备参考音频路径 (修正为 io/input/audio)
    audio_input_dir = os.path.join(RVC_IO_DIR, "input", "audio")

    # 处理 voice_choice 可能包含或不包含扩展名的情况
    if os.path.splitext(voice_choice)[1]:
        host_ref_audio = os.path.join(audio_input_dir, voice_choice)
    else:
        host_ref_audio = os.path.join(audio_input_dir, f"{voice_choice}.wav")

    if not os.path.exists(host_ref_audio):
        print(f"[video_generator] 警告: 音色 {host_ref_audio} 不存在，使用默认 zhb")
        host_ref_audio = os.path.join(audio_input_dir, "zhb.wav")

    chunk_audio_files = []
    text_chunks = simple_splitter(text)

    # 临时文本文件路径
    host_temp_text_file = os.path.join(RVC_IO_DIR, "input", f"temp_gen_{uuid.uuid4()}.txt")

    try:
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip(): continue

            # 写入文本块
            with open(host_temp_text_file, 'w', encoding='utf-8') as f:
                f.write(chunk)

            # Docker 路径映射
            ref_name = os.path.basename(host_ref_audio)
            # 修正 Docker 内路径，匹配 io/input/audio
            docker_ref_path = f"/io/input/audio/{ref_name}"
            temp_text_name = os.path.basename(host_temp_text_file)
            docker_text_path = f"/io/input/{temp_text_name}"

            chunk_filename = f"gen_chunk_{uuid.uuid4()}.wav"
            docker_out_path = f"/output/{chunk_filename}"
            host_chunk_path = os.path.join(AUDIO_OUTPUT_DIR, chunk_filename)

            # 调用 Docker
            cmd_docker_rvc = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{RVC_IO_DIR}:/io",
                "-v", f"{RVC_MODELS_DIR}:/app/models_zh",
                "-v", f"{AUDIO_OUTPUT_DIR}:/output",
                "rvc-app",
                "--ref", docker_ref_path,
                "--text-file", docker_text_path,
                "--out", docker_out_path
            ]

            print(f"[video_generator] 生成第 {i + 1}/{len(text_chunks)} 块...")
            subprocess.run(cmd_docker_rvc, check=True, cwd=BASE_DIR, capture_output=True, text=True, encoding='utf-8')

            if os.path.exists(host_chunk_path):
                chunk_audio_files.append(host_chunk_path)
            else:
                print(f"[video_generator] 错误: 第 {i + 1} 块生成失败")

        # 拼接音频
        if not chunk_audio_files:
            raise Exception("RVC 未生成任何音频")

        print("[video_generator] 拼接音频...")
        final_audio = AudioSegment.empty()
        for audio_file in chunk_audio_files:
            try:
                segment = AudioSegment.from_wav(audio_file)
                final_audio += segment
            except Exception as e:
                print(f"拼接警告: {e}")

        # 保存最终文件
        final_filename = f"generated_{int(time.time())}.wav"
        final_path = os.path.join(AUDIO_OUTPUT_DIR, final_filename)
        final_audio.export(final_path, format="wav")
        print(f"[video_generator] 最终音频已保存: {final_path}")

        return final_path

    except Exception as e:
        print(f"[video_generator] 音频生成异常: {e}")
        return None
    finally:
        # 清理临时文件
        if os.path.exists(host_temp_text_file):
            try:
                os.remove(host_temp_text_file)
            except:
                pass
        for f in chunk_audio_files:
            try:
                os.remove(f)
            except:
                pass


def generate_video(data):
    """
    视频生成逻辑
    """
    print("[backend.video_generator] 收到数据：")
    for k, v in data.items():
        if k != 'target_text':  # 文本太长不打印
            print(f"  {k}: {v}")

    # ==================== 核心修改：处理文本生成语音 ====================
    target_text = data.get('target_text')
    voice_clone = data.get('voice_clone')

    if target_text and target_text.strip():
        print("[backend.video_generator] 检测到文本输入，开始生成语音...")
        generated_audio_path = generate_audio_from_text(target_text, voice_clone)

        if generated_audio_path and os.path.exists(generated_audio_path):
            # 更新 data 中的音频路径，供后续模型使用

            # 1. 为 SyncTalk 准备绝对路径
            data['ref_audio'] = generated_audio_path

            # 2. 为 GeneFace 准备相对路径 (需要先复制到 GeneFace 目录)
            gf_audio_dir = os.path.join(BASE_DIR, "GeneFace", "data", "raw", "val_wavs")
            os.makedirs(gf_audio_dir, exist_ok=True)
            gf_filename = os.path.basename(generated_audio_path)
            gf_target_path = os.path.join(gf_audio_dir, gf_filename)

            try:
                shutil.copy(generated_audio_path, gf_target_path)
                # GeneFace API 需要的是相对于 WORK_DIR 的路径
                data['gf_audio_path'] = f"data/raw/val_wavs/{gf_filename}"
                print(f"[backend.video_generator] 已将生成音频复制到 GeneFace 目录: {data['gf_audio_path']}")
            except Exception as e:
                print(f"[backend.video_generator] 复制音频失败: {e}")
        else:
            print("[backend.video_generator] 语音生成失败，尝试使用原有音频路径(如果有)")
    else:
        print("[backend.video_generator] 未检测到文本输入，使用上传/选择的音频")
    # ==================== 修改结束 ====================

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
        print(f"[GeneFace++] 开始调用 API: {GENEFACE_API_URL}/infer")
        response = requests.post(
            f"{GENEFACE_API_URL}/infer",
            json={
                "head_ckpt": head_ckpt,
                "torso_ckpt": torso_ckpt,
                "audio_path": audio_path,
                "gpu_id": gpu_id
            },
            timeout=600
        )
        result = response.json()
        print(f"[GeneFace++] API 响应: {result}")

    except Exception as e:
        print(f"[GeneFace++] API 调用异常: {e}")

    # 强制检查 GeneFace/tmp.mp4 是否存在，并以此为准
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    geneface_tmp_path = os.path.join(project_root, "GeneFace", "tmp.mp4")

    # 检查 infer_out 里的规范文件 (API server 可能会移动文件)
    geneface_infer_out = os.path.join(project_root, "GeneFace", "infer_out")

    source_path = None

    # 优先检查 tmp.mp4
    if os.path.exists(geneface_tmp_path):
        source_path = geneface_tmp_path
    # 其次检查 API 返回的路径
    elif 'result' in locals() and result.get('video_path'):
        api_path = os.path.join(project_root, "GeneFace", result['video_path'])
        if os.path.exists(api_path):
            source_path = api_path

    if source_path:
        try:
            # 生成目标文件名
            video_name = 'geneface_output.mp4'
            dest_path = os.path.join(project_root, "static", "videos", video_name)

            # 1. 复制为唯一文件名 (用于下载或历史)
            shutil.copy(source_path, dest_path)
            print(f"[GeneFace++] 视频已复制到: {dest_path}")

            # 2. 复制为 "latest" 文件 (用于页面默认加载)
            latest_path = os.path.join(project_root, "static", "videos", "geneface_latest.mp4")
            shutil.copy(source_path, latest_path)
            print(f"[GeneFace++] 更新了最新视频缓存: {latest_path}")

            return os.path.join("static", "videos", video_name)
        except Exception as e:
            print(f"[GeneFace++] 复制视频文件失败: {e}")
    else:
        print(f"[GeneFace++] 未找到输出视频")

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
            return destination_path
        else:
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

    except Exception as e:
        print(f"[backend.video_generator] 错误: {e}")
        return os.path.join("static", "videos", "out.mp4")