import os

os.environ["OMP_NUM_THREADS"] = "1"
import sys
import glob
import numpy as np
from tqdm import tqdm
import traceback

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data_gen.utils.mp_feature_extractors.face_landmarker import MediapipeLandmarker

"""
单进程版本的 extract_lm2d.py
解决 MediaPipe + multiprocessing 的兼容性问题
用法与原脚本相同:
    python data_gen/utils/process_video/extract_lm2d_single.py --ds_name=nerf --vid_dir=data/raw/videos/xxx.mp4
"""


def extract_landmark_job(face_landmarker, video_name, nerf=False):
    """提取单个视频的 landmarks"""
    try:
        if nerf:
            out_name = video_name.replace("/raw/", "/processed/").replace(".mp4", "/lms_2d.npy")
        else:
            out_name = video_name.replace("/video/", "/lms_2d/").replace(".mp4", "_lms.npy")

        if os.path.exists(out_name):
            print(f"Output already exists, skipping: {out_name}")
            return True

        os.makedirs(os.path.dirname(out_name), exist_ok=True)

        img_lm478, vid_lm478 = face_landmarker.extract_lm478_from_video_name(video_name)
        lm478 = face_landmarker.combine_vid_img_lm478_to_lm478(img_lm478, vid_lm478)
        np.save(out_name, lm478)
        return True
    except Exception as e:
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", default='data/raw/videos/May.mp4')
    parser.add_argument("--ds_name", default='nerf')
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    args = parser.parse_args()

    vid_dir = args.vid_dir
    ds_name = args.ds_name

    # 构建视频列表
    if ds_name.lower() == 'nerf':  # 处理单个视频
        vid_names = [vid_dir]
    else:  # 处理整个数据集
        if ds_name in ['lrs3_trainval']:
            vid_name_pattern = os.path.join(vid_dir, "*/*.mp4")
        elif ds_name in ['TH1KH_512', 'CelebV-HQ']:
            vid_name_pattern = os.path.join(vid_dir, "*.mp4")
        elif ds_name in ['lrs2', 'lrs3', 'voxceleb2', 'CMLR']:
            vid_name_pattern = os.path.join(vid_dir, "*/*/*.mp4")
        elif ds_name in ["RAVDESS", 'VFHQ']:
            vid_name_pattern = os.path.join(vid_dir, "*/*/*/*.mp4")
        else:
            raise NotImplementedError(f"Unknown dataset: {ds_name}")
        vid_names = sorted(glob.glob(vid_name_pattern))

    # 多进程分片处理
    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process - 1
        num_samples_per_process = len(vid_names) // total_process
        if process_id == total_process - 1:
            vid_names = vid_names[process_id * num_samples_per_process:]
        else:
            vid_names = vid_names[process_id * num_samples_per_process: (process_id + 1) * num_samples_per_process]

    print(f"Todo videos number: {len(vid_names)}")

    if len(vid_names) == 0:
        print("No videos to process!")
        return

    # 验证视频文件存在
    for vid_name in vid_names:
        if not os.path.exists(vid_name):
            print(f"ERROR: Video file not found: {vid_name}")
            return

    # 在主进程中初始化 MediapipeLandmarker (关键！)
    print("Initializing MediapipeLandmarker...")
    face_landmarker = MediapipeLandmarker()
    print("MediapipeLandmarker initialized successfully!")

    # 单进程顺序处理
    is_nerf = (ds_name.lower() == 'nerf')
    fail_cnt = 0

    for i, vid_name in enumerate(tqdm(vid_names, desc="Extracting landmarks")):
        res = extract_landmark_job(face_landmarker, vid_name, nerf=is_nerf)
        if res is False:
            fail_cnt += 1
        print(
            f"Finished {i + 1} / {len(vid_names)} = {(i + 1) / len(vid_names):.4f}, failed {fail_cnt} / {i + 1} = {fail_cnt / (i + 1) if i > 0 else 0:.4f}")
        sys.stdout.flush()

    print(f"\nDone! Total: {len(vid_names)}, Failed: {fail_cnt}")


if __name__ == '__main__':
    main()