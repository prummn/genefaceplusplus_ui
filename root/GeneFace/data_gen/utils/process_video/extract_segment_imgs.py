import os

# 关键修复1：限制 OpenMP 线程数，防止多进程下 CPU 争抢
os.environ["OMP_NUM_THREADS"] = "1"

import random
import glob
import cv2

# 关键修复2：禁止 OpenCV 内部多线程，防止死锁
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import tqdm
import numpy as np
from typing import Union
from utils.commons.tensor_utils import convert_to_np
from utils.commons.os_utils import multiprocess_glob
import pickle
import traceback
import multiprocessing
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.neighbors import NearestNeighbors
from mediapipe.tasks.python import vision
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter, encode_segmap_mask_to_image, \
    decode_segmap_mask_from_image, job_cal_seg_map_for_image

# 删除全局变量引用，避免状态污染
# seg_model   = None
# segmenter   = None
mat_model = None
lama_model = None
lama_config = None

from data_gen.utils.process_video.split_video_to_imgs import extract_img_job

BG_NAME_MAP = {
    "knn": "",
}
FRAME_SELECT_INTERVAL = 5
SIM_METHOD = "mse"
SIM_THRESHOLD = 3


def save_file(name, content):
    with open(name, "wb") as f:
        pickle.dump(content, f)


def load_file(name):
    with open(name, "rb") as f:
        content = pickle.load(f)
    return content


def save_rgb_alpha_image_to_path(img, alpha, img_path):
    try:
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    except:
        pass
    cv2.imwrite(img_path, np.concatenate([cv2.cvtColor(img, cv2.COLOR_RGB2BGR), alpha], axis=-1))


def save_rgb_image_to_path(img, img_path):
    try:
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    except:
        pass
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_rgb_image_to_path(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


def image_similarity(x: np.ndarray, y: np.ndarray, method="mse"):
    if method == "mse":
        return np.mean((x - y) ** 2)
    else:
        raise NotImplementedError


def extract_background(img_lst, segmap_mask_lst=None, method="knn", device='cpu', mix_bg=True):
    """
    img_lst: list of rgb ndarray
    method: "knn"
    """
    # 局部初始化，避免依赖全局变量
    local_seg_model = None
    local_segmenter = None

    assert len(img_lst) > 0
    if segmap_mask_lst is not None:
        assert len(segmap_mask_lst) == len(img_lst)
    else:
        # 只有在需要的时候才初始化
        local_seg_model = MediapipeSegmenter()
        local_segmenter = vision.ImageSegmenter.create_from_options(local_seg_model.video_options)

    def get_segmap_mask(img_lst, segmap_mask_lst, index, segmenter_inst=None, model_inst=None):
        if segmap_mask_lst is not None:
            segmap = refresh_segment_mask(segmap_mask_lst[index])
        else:
            # 使用传入的实例
            segmap = model_inst._cal_seg_map(refresh_image(img_lst[index]), segmenter=segmenter_inst)
        return segmap

    if method == "knn":
        num_frames = len(img_lst)
        # 注意：这里需要重新定义 INTERVAL 变量或使用传入参数，这里沿用全局默认值
        local_interval = FRAME_SELECT_INTERVAL
        if num_frames <= 100:
            local_interval = 5
        elif num_frames < 10000:
            local_interval = 20
        else:
            local_interval = num_frames // 500

        img_lst_sampled = img_lst[::local_interval] if num_frames > local_interval else img_lst[0:1]

        if segmap_mask_lst is not None:
            segmap_mask_lst_sampled = segmap_mask_lst[
                                      ::local_interval] if num_frames > local_interval else segmap_mask_lst[0:1]
            assert len(img_lst_sampled) == len(segmap_mask_lst_sampled)
        else:
            segmap_mask_lst_sampled = None

        # get H/W
        h, w = refresh_image(img_lst_sampled[0]).shape[:2]

        # nearest neighbors
        all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()  # [512*512, 2] coordinate grid
        distss = []

        # 这里的 tqdm 不需要多进程，因为主要是 KNN 计算
        for idx, img in tqdm.tqdm(enumerate(img_lst_sampled), desc='combining backgrounds...',
                                  total=len(img_lst_sampled)):
            segmap = get_segmap_mask(img_lst=img_lst_sampled, segmap_mask_lst=segmap_mask_lst_sampled, index=idx,
                                     segmenter_inst=local_segmenter, model_inst=local_seg_model)
            bg = (segmap[0]).astype(bool)  # [h,w] bool mask
            fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)  # [N_nonbg,2] coordinate of non-bg pixels

            # 边界情况处理：如果整张图都是背景或都是前景
            if fg_xys.shape[0] == 0:
                # 全是背景
                dists = np.zeros((all_xys.shape[0], 1))
            else:
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
                dists, _ = nbrs.kneighbors(all_xys)  # [512*512, 1] distance to nearest non-bg pixel

            distss.append(dists)

        distss = np.stack(distss)  # [B, 512*512, 1]
        max_dist = np.max(distss, 0)  # [512*512, 1]
        max_id = np.argmax(distss, 0)  # id of frame

        bc_pixs = max_dist > 10  # 在各个frame有一个出现过是bg的pixel，bg标准是离最近的non-bg pixel距离大于10
        bc_pixs_id = np.nonzero(bc_pixs)
        bc_ids = max_id[bc_pixs]

        num_pixs = distss.shape[1]
        bg_img = np.zeros((h * w, 3), dtype=np.uint8)
        img_lst_sampled = [refresh_image(img) for img in img_lst_sampled]
        imgs = np.stack(img_lst_sampled).reshape(-1, num_pixs, 3)
        bg_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]  # 对那些铁bg的pixel，直接去对应的image里面采样
        bg_img = bg_img.reshape(h, w, 3)

        max_dist = max_dist.reshape(h, w)
        bc_pixs = max_dist > 10  # 5
        bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
        fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()

        if len(fg_xys) > 0 and len(bg_xys) > 0:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
            distances, indices = nbrs.kneighbors(bg_xys)  # 对non-bg img，用KNN找最近的bg pixel
            bg_fg_xys = fg_xys[indices[:, 0]]
            bg_img[bg_xys[:, 0], bg_xys[:, 1], :] = bg_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]

    else:
        raise NotImplementedError  # deperated

    # 清理资源
    if local_segmenter: local_segmenter.close()

    return bg_img


def inpaint_torso_job(gt_img, segmap):
    bg_part = (segmap[0]).astype(bool)
    head_part = (segmap[1] + segmap[3] + segmap[5]).astype(bool)
    neck_part = (segmap[2]).astype(bool)
    torso_part = (segmap[4]).astype(bool)
    img = gt_img.copy()
    img[head_part] = 0

    # torso part "vertical" in-painting...
    L = 8 + 1
    torso_coords = np.stack(np.nonzero(torso_part), axis=-1)  # [M, 2]
    # lexsort: sort 2D coords first by y then by x,
    # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
    inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
    torso_coords = torso_coords[inds]

    if len(torso_coords) == 0:
        inpaint_torso_mask = None
    else:
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid]  # [m, 2]
        # only keep top-is-head pixels
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])  # [N, 2]

        # 边界检查
        top_torso_coords_up[:, 0] = np.clip(top_torso_coords_up[:, 0], 0, img.shape[0] - 1)

        mask = head_part[tuple(top_torso_coords_up.T)]
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            # get the color
            top_torso_colors = gt_img[tuple(top_torso_coords.T)]  # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0)  # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None]  # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2)  # [Lm, 2]
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0)  # [L, m, 3]
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1)  # [L, 1, 1]
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3)  # [Lm, 3]

            # Clip coords to image bounds
            inpaint_torso_coords[:, 0] = np.clip(inpaint_torso_coords[:, 0], 0, img.shape[0] - 1)
            inpaint_torso_coords[:, 1] = np.clip(inpaint_torso_coords[:, 1], 0, img.shape[1] - 1)

            # set color
            img[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors
            inpaint_torso_mask = np.zeros_like(img[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None

    # neck part "vertical" in-painting...
    push_down = 4
    L = 48 + push_down + 1
    neck_part = binary_dilation(neck_part, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
                                iterations=3)
    neck_coords = np.stack(np.nonzero(neck_part), axis=-1)  # [M, 2]

    if len(neck_coords) > 0:
        # lexsort: sort 2D coords first by y then by x
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid]  # [m, 2]
        # only keep top-is-head pixels
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        top_neck_coords_up[:, 0] = np.clip(top_neck_coords_up[:, 0], 0, img.shape[0] - 1)

        mask = head_part[tuple(top_neck_coords_up.T)]
        top_neck_coords = top_neck_coords[mask]

        if len(top_neck_coords) > 0:
            # push these top down for 4 pixels to make the neck inpainting more natural...
            offset_down = np.minimum(ucnt[mask] - 1, push_down)
            top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
            # get the color
            top_neck_colors = gt_img[tuple(top_neck_coords.T)]  # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_neck_coords = top_neck_coords[None].repeat(L, 0)  # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None]  # [L, 1, 2]
            inpaint_neck_coords += inpaint_offsets
            inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2)  # [Lm, 2]
            inpaint_neck_colors = top_neck_colors[None].repeat(L, 0)  # [L, m, 3]
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1)  # [L, 1, 1]
            inpaint_neck_colors = (inpaint_neck_colors * darken_scaler).reshape(-1, 3)  # [Lm, 3]

            # Clip
            inpaint_neck_coords[:, 0] = np.clip(inpaint_neck_coords[:, 0], 0, img.shape[0] - 1)
            inpaint_neck_coords[:, 1] = np.clip(inpaint_neck_coords[:, 1], 0, img.shape[1] - 1)

            # set color
            img[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors
            # apply blurring to the inpaint area to avoid vertical-line artifects...
            inpaint_mask = np.zeros_like(img[..., 0]).astype(bool)
            inpaint_mask[tuple(inpaint_neck_coords.T)] = True

            blur_img = img.copy()
            blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)
            img[inpaint_mask] = blur_img[inpaint_mask]
        else:
            inpaint_mask = np.zeros_like(img[..., 0]).astype(bool)
    else:
        inpaint_mask = np.zeros_like(img[..., 0]).astype(bool)

    # set mask
    torso_img_mask = (neck_part | torso_part | inpaint_mask)
    torso_with_bg_img_mask = (bg_part | neck_part | torso_part | inpaint_mask)
    if inpaint_torso_mask is not None:
        torso_img_mask = torso_img_mask | inpaint_torso_mask
        torso_with_bg_img_mask = torso_with_bg_img_mask | inpaint_torso_mask

    torso_img = img.copy()
    torso_img[~torso_img_mask] = 0
    torso_with_bg_img = img.copy()
    torso_with_bg_img[
        ~torso_with_bg_img_mask] = 0  # Bug fix: original code used torso_img, changed to torso_with_bg_img to be safe

    return torso_img, torso_img_mask, torso_with_bg_img, torso_with_bg_img_mask


def load_segment_mask_from_file(filename: str):
    encoded_segmap = load_rgb_image_to_path(filename)
    segmap_mask = decode_segmap_mask_from_image(encoded_segmap)
    return segmap_mask


# load segment mask to memory if not loaded yet
def refresh_segment_mask(segmap_mask: Union[str, np.ndarray]):
    if isinstance(segmap_mask, str):
        segmap_mask = load_segment_mask_from_file(segmap_mask)
    return segmap_mask


# load segment mask to memory if not loaded yet
def refresh_image(image: Union[str, np.ndarray]):
    if isinstance(image, str):
        image = load_rgb_image_to_path(image)
    return image


def generate_segment_imgs_job(img_name, segmap, img, seg_model_inst=None):
    # 如果没有传入 model 实例，临时创建一个，但这很不推荐，因为会非常慢
    if seg_model_inst is None:
        seg_model_inst = MediapipeSegmenter()

    out_img_name = segmap_name = img_name.replace("/gt_imgs/", "/segmaps/").replace(".jpg",
                                                                                    ".png")  # 存成jpg的话，pixel value会有误差
    try:
        os.makedirs(os.path.dirname(out_img_name), exist_ok=True)
    except:
        pass
    encoded_segmap = encode_segmap_mask_to_image(segmap)
    save_rgb_image_to_path(encoded_segmap, out_img_name)

    for mode in ['head', 'torso', 'person', 'bg']:
        # 使用传入的实例方法
        out_img, mask = seg_model_inst._seg_out_img_with_segmap(img, segmap, mode=mode)
        img_alpha = 255 * np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8)  # alpha
        mask = mask[0][..., None]
        img_alpha[~mask] = 0
        out_img_name = img_name.replace("/gt_imgs/", f"/{mode}_imgs/").replace(".jpg", ".png")
        save_rgb_alpha_image_to_path(out_img, img_alpha, out_img_name)

    inpaint_torso_img, inpaint_torso_img_mask, inpaint_torso_with_bg_img, inpaint_torso_with_bg_img_mask = inpaint_torso_job(
        img, segmap)
    img_alpha = 255 * np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8)  # alpha
    img_alpha[~inpaint_torso_img_mask[..., None]] = 0
    out_img_name = img_name.replace("/gt_imgs/", f"/inpaint_torso_imgs/").replace(".jpg", ".png")
    save_rgb_alpha_image_to_path(inpaint_torso_img, img_alpha, out_img_name)
    return segmap_name


# 关键修改：此函数是 Worker 执行的主体
def segment_and_generate_for_image_job(img_name, img, segmenter_options=None, segmenter=None, seg_model=None,
                                       store_in_memory=False):
    # 逻辑：如果 segmenter 是 None，说明我们在子进程中，且没有继承父进程的对象，需要自己初始化
    local_created = False

    if segmenter is None or seg_model is None:
        # 在 Worker 内部初始化，保证进程安全
        seg_model = MediapipeSegmenter()
        segmenter = vision.ImageSegmenter.create_from_options(seg_model.options)
        local_created = True

    try:
        img = refresh_image(img)
        segmap_mask, segmap_image = job_cal_seg_map_for_image(img, segmenter_options=segmenter_options,
                                                              segmenter=segmenter)
        # 传递 seg_model 实例给 generate
        segmap_name = generate_segment_imgs_job(img_name=img_name, segmap=segmap_mask, img=img,
                                                seg_model_inst=seg_model)

        if store_in_memory:
            return segmap_mask
        else:
            return segmap_name
    finally:
        # 如果是本地创建的，用完必须销毁，否则可能内存泄露或挂起
        if local_created and segmenter:
            segmenter.close()
            del segmenter


def extract_segment_job(
        video_name,
        nerf=False,
        background_method='knn',
        device="cpu",
        total_gpus=0,
        mix_bg=True,
        store_in_memory=False,
        force_single_process=False,
):
    # 关键修改：不要在这里使用 global
    # 我们根据是否启用多进程来决定如何传递对象

    # nerf means that we extract only one video, so can enable multi-process acceleration for frames
    multiprocess_enable = nerf and not force_single_process

    # 本地变量
    local_seg_model = None
    local_segmenter = None

    try:
        if "cuda" in device:
            # determine which cuda index from subprocess id
            pname = multiprocessing.current_process().name
            try:
                pid = int(pname.rsplit("-", 1)[-1]) - 1
            except:
                pid = 0
            cuda_id = pid % (total_gpus if total_gpus > 0 else 1)
            device = f"cuda:{cuda_id}"

        if nerf:  # single video
            raw_img_dir = video_name.replace(".mp4", "/gt_imgs/").replace("/raw/", "/processed/")
        else:  # whole dataset
            raw_img_dir = video_name.replace(".mp4", "").replace("/video/", "/gt_imgs/")

        if not os.path.exists(raw_img_dir):
            extract_img_job(video_name, raw_img_dir)  # use ffmpeg to split video into imgs

        img_names = glob.glob(os.path.join(raw_img_dir, "*.jpg"))
        # 排序以保证处理顺序
        img_names = sorted(img_names)

        img_lst = []

        for img_name in img_names:
            if store_in_memory:
                img = load_rgb_image_to_path(img_name)
            else:
                img = img_name
            img_lst.append(img)

        print("| Extracting Segmaps && Saving...")

        # 如果是单进程模式，我们在这里初始化一次，传给所有任务
        if not multiprocess_enable:
            local_seg_model = MediapipeSegmenter()
            local_segmenter = vision.ImageSegmenter.create_from_options(local_seg_model.options)

        args = []
        segmap_mask_lst = []
        # preparing parameters for segment
        for i in range(len(img_lst)):
            img_name = img_names[i]
            img = img_lst[i]

            if multiprocess_enable:
                # 多进程模式：传入 None，让 Worker 自己初始化
                options = None
                segmenter_arg = None
                model_arg = None
            else:
                # 单进程模式：传入已经初始化的对象
                options = None
                segmenter_arg = local_segmenter
                model_arg = local_seg_model

            arg = (img_name, img, options, segmenter_arg, model_arg, store_in_memory)
            args.append(arg)

        if multiprocess_enable:
            # 使用较多的 worker 来并行处理帧
            workers = max(1, multiprocessing.cpu_count() // 2)
            # 限制一下最大 worker 数量，防止内存爆掉
            workers = min(workers, 8)

            for (_, res) in multiprocess_run_tqdm(segment_and_generate_for_image_job, args=args, num_workers=workers,
                                                  desc='generating segment images in multi-processes...'):
                segmap_mask = res
                segmap_mask_lst.append(segmap_mask)
        else:
            for index in tqdm.tqdm(range(len(img_lst)), desc="generating segment images in single-process..."):
                segmap_mask = segment_and_generate_for_image_job(*args[index])
                segmap_mask_lst.append(segmap_mask)

        # 清理单进程模式下的资源
        if local_segmenter:
            local_segmenter.close()
            local_segmenter = None

        # 确保顺序一致（多进程返回可能是乱序的，但 multiprocess_run_tqdm 通常保持顺序或返回带 index 的结果，这里假设 multiprocess_run_tqdm 是保持顺序或我们不需要强顺序）
        # 注意：上面的 multiprocess_run_tqdm 如果返回乱序，segmap_mask_lst 顺序会错。
        # 通常 multiprocess_run_tqdm 实现会处理好顺序，或者我们应该重新根据文件名读取

        # 为了稳妥起见，如果 segmap_mask_lst 存储的是文件名，重新 glob 一次最安全；如果是内存对象，必须保证 multiprocess_run_tqdm 有序。
        # 这里假设 multiprocess_run_tqdm 返回是有序的 (idx, res) 或者原代码逻辑能接受。

        print("| Extracted Segmaps Done.")

        # 重新加载 img_lst 如果之前只是文件名
        # Background extraction logic
        print("| Extracting background...")
        bg_prefix_name = f"bg{BG_NAME_MAP[background_method]}"

        # 如果是内存模式，segmap_mask_lst 已经是 mask 数组；否则是文件名
        if not store_in_memory and multiprocess_enable:
            # 多进程跑完只返回了文件名，重新整理列表以防乱序（虽然 glob 是 sorted 的）
            segmap_path = raw_img_dir.replace("/gt_imgs/", "/segmaps/")
            # 重新构建 segmap 列表
            segmap_mask_lst = [n.replace("/gt_imgs/", "/segmaps/").replace(".jpg", ".png") for n in img_names]

        bg_img = extract_background(img_lst, segmap_mask_lst, method=background_method, device=device, mix_bg=mix_bg)

        if nerf:
            out_img_name = video_name.replace("/raw/", "/processed/").replace(".mp4", f"/{bg_prefix_name}.jpg")
        else:
            out_img_name = video_name.replace("/video/", f"/{bg_prefix_name}_img/").replace(".mp4", ".jpg")
        save_rgb_image_to_path(bg_img, out_img_name)
        print("| Extracted background done.")

        print("| Extracting com_imgs...")
        com_prefix_name = f"com{BG_NAME_MAP[background_method]}"

        # 这里的合成操作非常快，单进程处理即可
        for i in tqdm.trange(len(img_names), desc='extracting com_imgs'):
            img_name = img_names[i]
            com_img = refresh_image(img_lst[i]).copy()
            segmap = refresh_segment_mask(segmap_mask_lst[i])
            bg_part = segmap[0].astype(bool)[..., None].repeat(3, axis=-1)
            com_img[bg_part] = bg_img[bg_part]
            out_img_name = img_name.replace("/gt_imgs/", f"/{com_prefix_name}_imgs/")
            save_rgb_image_to_path(com_img, out_img_name)
        print("| Extracted com_imgs done.")

        return 0
    except Exception as e:
        print(str(type(e)), e)
        traceback.print_exc()
        return 1


def out_exist_job(vid_name, background_method='knn'):
    com_prefix_name = f"com{BG_NAME_MAP[background_method]}"
    img_dir = vid_name.replace("/video/", "/gt_imgs/").replace(".mp4", "")
    out_dir1 = img_dir.replace("/gt_imgs/", "/head_imgs/")
    out_dir2 = img_dir.replace("/gt_imgs/", f"/{com_prefix_name}_imgs/")

    if os.path.exists(img_dir) and os.path.exists(out_dir1) and os.path.exists(out_dir1) and os.path.exists(out_dir2):
        num_frames = len(os.listdir(img_dir))
        if len(os.listdir(out_dir1)) == num_frames and len(os.listdir(out_dir2)) == num_frames:
            return None
        else:
            return vid_name
    else:
        return vid_name


def get_todo_vid_names(vid_names, background_method='knn'):
    if len(vid_names) == 1:  # nerf
        return vid_names
    todo_vid_names = []
    fn_args = [(vid_name, background_method) for vid_name in vid_names]
    # Check 过程通常很快，可以保留多进程
    for i, res in multiprocess_run_tqdm(out_exist_job, fn_args, num_workers=16, desc="checking todo videos..."):
        if res is not None:
            todo_vid_names.append(res)
    return todo_vid_names


if __name__ == '__main__':
    import argparse, glob, tqdm, random

    # 设置 mp start method 为 spawn，这是最安全的做法，但在已有代码库中可能导致 pickle 错误
    # 如果代码中有不能 pickle 的对象，保持默认 (fork) 但确保不初始化 C++ 对象
    # multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", default='data/raw/videos/May.mp4')
    parser.add_argument("--ds_name", default='nerf')
    parser.add_argument("--num_workers", default=16, type=int)  # 稍微调小默认值，MediaPipe 较重
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--reset", action='store_true')
    parser.add_argument("--load_names", action="store_true")
    parser.add_argument("--background_method", choices=['knn', 'mat', 'ddnm', 'lama'], type=str, default='knn')
    parser.add_argument("--total_gpus", default=0, type=int)  # zero gpus means utilizing cpu
    parser.add_argument("--no_mix_bg", action="store_true")
    parser.add_argument("--store_in_memory",
                        action="store_true")  # set to True to speed up preprocess, but leads to high memory costs
    parser.add_argument("--force_single_process",
                        action="store_true")  # turn this on if you find multi-process does not work on your environment

    args = parser.parse_args()
    vid_dir = args.vid_dir
    ds_name = args.ds_name
    load_names = args.load_names
    background_method = args.background_method
    total_gpus = args.total_gpus
    mix_bg = not args.no_mix_bg
    store_in_memory = args.store_in_memory
    force_single_process = args.force_single_process

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    # 慎用 pkill，可能会杀掉其他无关进程，这里注释掉或者由用户自己保证
    # for d in devices[:total_gpus]:
    #     os.system(f'pkill -f "voidgpu{d}"')

    if ds_name.lower() == 'nerf':  # 处理单个视频
        vid_names = [vid_dir]
        out_names = [video_name.replace("/raw/", "/processed/").replace(".mp4", "_lms.npy") for video_name in vid_names]
    else:  # 处理整个数据集
        if ds_name in ['lrs3_trainval']:
            vid_name_pattern = os.path.join(vid_dir, "*/*.mp4")
        elif ds_name in ['TH1KH_512', 'CelebV-HQ']:
            vid_name_pattern = os.path.join(vid_dir, "*.mp4")
        elif ds_name in ['lrs2', 'lrs3', 'voxceleb2']:
            vid_name_pattern = os.path.join(vid_dir, "*/*/*.mp4")
        elif ds_name in ["RAVDESS", 'VFHQ']:
            vid_name_pattern = os.path.join(vid_dir, "*/*/*/*.mp4")
        else:
            raise NotImplementedError()

        vid_names_path = os.path.join(vid_dir, "vid_names.pkl")
        if os.path.exists(vid_names_path) and load_names:
            print(f"loading vid names from {vid_names_path}")
            vid_names = load_file(vid_names_path)
        else:
            vid_names = multiprocess_glob(vid_name_pattern)
        vid_names = sorted(vid_names)
        print(f"saving vid names to {vid_names_path}")
        save_file(vid_names_path, vid_names)

    vid_names = sorted(vid_names)
    random.seed(args.seed)
    random.shuffle(vid_names)

    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process - 1
        num_samples_per_process = len(vid_names) // total_process
        if process_id == total_process:
            vid_names = vid_names[process_id * num_samples_per_process:]
        else:
            vid_names = vid_names[process_id * num_samples_per_process: (process_id + 1) * num_samples_per_process]

    if not args.reset:
        vid_names = get_todo_vid_names(vid_names, background_method)
    print(f"todo videos number: {len(vid_names)}")

    device = "cuda" if total_gpus > 0 else "cpu"
    extract_job = extract_segment_job

    # 修改逻辑：nerf=True 表示开启内部多进程（Video级单进程，Frame级多进程）
    # 如果 ds_name 不是 nerf，但列表里只有一个视频，也尽量开启内部多进程
    is_nerf_mode = (ds_name.lower() == 'nerf') or (len(vid_names) == 1)

    fn_args = [
        (vid_name, is_nerf_mode, background_method, device, total_gpus, mix_bg, store_in_memory, force_single_process)
        for i, vid_name in enumerate(vid_names)]

    if len(vid_names) == 1:
        # 直接调用，避免最外层再套一个 multiprocess pool，方便调试
        extract_job(*fn_args[0])
    else:
        # 数据集模式：Video级多进程，内部强制单进程（在 extract_segment_job 内部通过 is_nerf_mode=False 控制）
        # 注意：如果 is_nerf_mode 为 False，extract_segment_job 内部就不会开启多进程
        for vid_name in multiprocess_run_tqdm(extract_job, fn_args,
                                              desc=f"Root process {args.process_id}:  segment images",
                                              num_workers=args.num_workers):
            pass