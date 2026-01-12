import os
ROOT = os.path.dirname(os.path.dirname(__file__))

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(top_dir)
sys.path.append(os.path.join(top_dir, 'models', 'sam_3d_body'))
sys.path.append(os.path.join(top_dir, 'models', 'diffusion_vas'))

import uuid
from datetime import datetime

def gen_id():
    t = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    u = uuid.uuid4().hex[:8]
    return f"{t}_{u}"

import argparse
import time
import cv2
import glob
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from utils import draw_point_marker, mask_painter, images_to_mp4, DAVIS_PALETTE, jpg_folder_to_mp4, is_super_long_or_wide, keep_largest_component, is_skinny_mask, bbox_from_mask, gpu_profile, resize_mask_with_unique_label

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import process_image_with_mask, save_mesh_results
from models.sam_3d_body.tools.vis_utils import visualize_sample_together, visualize_sample
from models.diffusion_vas.demo import init_amodal_segmentation_model, init_rgb_model, init_depth_model, load_and_transform_masks, load_and_transform_rgbs, rgb_to_depth

import torch
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def build_sam3_from_config(cfg):
    """
    Construct and return your SAM-3 model from config.
    You replace this with your real init code.
    """
    from models.sam3.sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model(checkpoint_path=cfg.sam3['ckpt_path'])
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    return sam3_model, predictor


def read_frame_at(path: str, idx: int):
    """Read a specific frame (by index) from a video file."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def build_sam3_3d_body_config(cfg):
    mhr_path = cfg.sam_3d_body['mhr_path']
    fov_path = cfg.sam_3d_body['fov_path']
    detector_path = cfg.sam_3d_body['detector_path']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body['ckpt_path'], device=device, mhr_path=mhr_path
    )
    
    human_detector, human_segmentor, fov_estimator = None, None, None
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator
    fov_estimator = FOVEstimator(name='moge2', device=device, path=fov_path)
    from models.sam_3d_body.tools.build_detector import HumanDetector
    human_detector = HumanDetector(name="vitdet", device=device, path=detector_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    return estimator


def smooth_camera_parameters(outputs_list, id_batch_list, smoothing_factor=0.5):
    """
    Apply temporal smoothing to camera parameters across frames to prevent floating.
    Tracks persons by their ID across frames and smooths their camera parameters.
    
    Args:
        outputs_list: List of lists, where each inner list contains dicts with camera params per person
        id_batch_list: List of lists, where each inner list contains person IDs for that frame
        smoothing_factor: Strength of smoothing (0.0 = no smoothing, 1.0 = maximum smoothing)
    
    Returns:
        Smoothed outputs_list with modified pred_cam_t and focal_length
    """
    if len(outputs_list) == 0:
        return outputs_list
    
    # Create a copy to avoid modifying original
    smoothed_outputs = []
    for frame_outputs in outputs_list:
        smoothed_outputs.append([out.copy() for out in frame_outputs])
    
    num_frames = len(smoothed_outputs)
    if num_frames < 2:
        return smoothed_outputs
    
    # Track camera parameters per person ID across frames
    # person_id -> list of (frame_idx, person_idx_in_frame, pred_cam_t, focal_length)
    person_tracks = {}  # person_id -> list of tuples
    
    # Collect all camera parameters, tracking by person ID
    for frame_idx, (frame_outputs, frame_ids) in enumerate(zip(smoothed_outputs, id_batch_list)):
        # Match IDs to outputs - they should correspond by index, but handle mismatches
        num_outputs = len(frame_outputs)
        num_ids = len(frame_ids)
        
        if num_outputs != num_ids:
            print(f"Warning: Frame {frame_idx} has {num_outputs} outputs but {num_ids} IDs. Using minimum.")
        
        # Use the minimum to avoid index errors
        num_people = min(num_outputs, num_ids)
        
        for person_idx_in_frame in range(num_people):
            # Double-check bounds (shouldn't be needed but safety first)
            if person_idx_in_frame >= len(frame_ids) or person_idx_in_frame >= len(frame_outputs):
                break
            
            person_id = frame_ids[person_idx_in_frame]
            
            if person_id not in person_tracks:
                person_tracks[person_id] = []
            
            person_output = frame_outputs[person_idx_in_frame]
            pred_cam_t = person_output.get('pred_cam_t', np.zeros(3))
            focal_length = person_output.get('focal_length', 5000.0)
            
            person_tracks[person_id].append((
                frame_idx,
                person_idx_in_frame,
                pred_cam_t.copy(),
                focal_length
            ))
    
    # Apply smoothing for each person track
    for person_id, track in person_tracks.items():
        if len(track) < 2:
            continue  # Need at least 2 frames to smooth
        
        # Sort by frame_idx to ensure correct order
        track.sort(key=lambda x: x[0])
        
        # Extract camera parameters
        cam_t_list = [t[2] for t in track]
        focal_list = [t[3] for t in track]
        
        # Convert to numpy arrays
        cam_t_array = np.array(cam_t_list)
        focal_array = np.array(focal_list)
        
        # Apply exponential moving average (bidirectional)
        smoothed_cam_t = cam_t_array.copy()
        smoothed_focal = focal_array.copy()
        
        # Forward pass
        for i in range(1, len(track)):
            alpha = smoothing_factor
            smoothed_cam_t[i] = (1 - alpha) * smoothed_cam_t[i] + alpha * smoothed_cam_t[i-1]
            smoothed_focal[i] = (1 - alpha) * smoothed_focal[i] + alpha * smoothed_focal[i-1]
        
        # Backward pass for bidirectional smoothing
        for i in range(len(track) - 2, -1, -1):
            alpha = smoothing_factor
            smoothed_cam_t[i] = (1 - alpha) * smoothed_cam_t[i] + alpha * smoothed_cam_t[i+1]
            smoothed_focal[i] = (1 - alpha) * smoothed_focal[i] + alpha * smoothed_focal[i+1]
        
        # Apply smoothed values back to outputs
        for track_idx, (frame_idx, person_idx_in_frame, _, _) in enumerate(track):
            # Verify bounds before accessing
            if frame_idx < len(smoothed_outputs) and person_idx_in_frame < len(smoothed_outputs[frame_idx]):
                smoothed_outputs[frame_idx][person_idx_in_frame]['pred_cam_t'] = smoothed_cam_t[track_idx]
                smoothed_outputs[frame_idx][person_idx_in_frame]['focal_length'] = smoothed_focal[track_idx]
            else:
                print(f"Warning: Skipping smoothing update for frame {frame_idx}, person {person_idx_in_frame} (out of bounds)")
    
    return smoothed_outputs


def build_diffusion_vas_config(cfg):
    model_path_mask = cfg.completion['model_path_mask']
    model_path_rgb = cfg.completion['model_path_rgb']
    depth_encoder = cfg.completion['depth_encoder']
    model_path_depth = cfg.completion['model_path_depth']
    max_occ_len = min(cfg.completion['max_occ_len'], cfg.sam_3d_body['batch_size'])
    enable_cpu_offload = cfg.completion.get('enable_cpu_offload', False)

    generator = torch.manual_seed(23)

    pipeline_mask = init_amodal_segmentation_model(model_path_mask)
    if enable_cpu_offload:
        pipeline_mask.enable_model_cpu_offload()
        print("Enabled CPU offloading for pipeline_mask to save GPU memory")
    
    pipeline_rgb = init_rgb_model(model_path_rgb)
    # pipeline_rgb already has CPU offload enabled in init_rgb_model
    
    depth_model = init_depth_model(model_path_depth, depth_encoder)

    return pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator


class OfflineApp:
    def __init__(self, config_path: str = os.path.join(ROOT, "configs", "body4d.yaml")):
        """Initialize CONFIG, SAM3_MODEL, and global RUNTIME dict."""
        self.CONFIG = OmegaConf.load(config_path)
        self.sam3_model, self.predictor = build_sam3_from_config(self.CONFIG)
        self.sam3_3d_body_model = build_sam3_3d_body_config(self.CONFIG)

        if self.CONFIG.completion.get('enable', False):
            self.pipeline_mask, self.pipeline_rgb, self.depth_model, self.max_occ_len, self.generator = build_diffusion_vas_config(self.CONFIG)
        else:
            self.pipeline_mask, self.pipeline_rgb, self.depth_model, self.max_occ_len, self.generator = None, None, None, None, None
        
        self.RUNTIME = {}  # clear any old state
        self.OUTPUT_DIR = os.path.join(self.CONFIG.runtime['output_dir'], gen_id())
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.RUNTIME['batch_size'] = self.CONFIG.sam_3d_body.get('batch_size', 1)
        self.RUNTIME['detection_resolution'] = self.CONFIG.completion.get('detection_resolution', [256, 512])
        self.RUNTIME['completion_resolution'] = self.CONFIG.completion.get('completion_resolution', [512, 1024])
        self.RUNTIME['smpl_export'] = self.CONFIG.runtime.get('smpl_export', False)
        self.RUNTIME['bboxes'] = None

    def on_mask_generation(self, video_path: str=None, start_frame_idx: int = 0, max_frame_num_to_track: int = 1800):
        """
        Mask generation across the video.
        Currently runs SAM-3 propagation and renders a mask video.
        """
        print("[DEBUG] Mask Generation button clicked.")

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores, iou_scores in self.predictor.propagate_in_video(
            self.RUNTIME['inference_state'],
            start_frame_idx=0,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
            propagate_preflight=True,
        ):
            video_segments[frame_idx] = {
                out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(self.RUNTIME['out_obj_ids'])
            } 

        # render the segmentation results every few frames
        vis_frame_stride = 1
        out_h = self.RUNTIME['inference_state']['video_height']
        out_w = self.RUNTIME['inference_state']['video_width']
        # img_to_video = []

        IMAGE_PATH = os.path.join(self.OUTPUT_DIR, 'images') # for sam3-3d-body
        MASKS_PATH = os.path.join(self.OUTPUT_DIR, 'masks')  # for sam3-3d-body
        os.makedirs(IMAGE_PATH, exist_ok=True)
        os.makedirs(MASKS_PATH, exist_ok=True)

        for out_frame_idx in range(0, len(video_segments), vis_frame_stride):
            # Handle both Sam3VideoInference (input_batch) and Sam3TrackerPredictor (images)
            inference_state = self.RUNTIME['inference_state']
            if 'input_batch' in inference_state:
                images = inference_state['input_batch'].img_batch
            elif 'images' in inference_state:
                images = inference_state['images']
            else:
                raise KeyError("Could not find images in inference_state. Expected 'input_batch' or 'images'")
            
            # Handle both tensor and batch loader
            img = images[out_frame_idx]
            if isinstance(img, torch.Tensor):
                img = img.detach().float().cpu()
            else:
                img = img.float().cpu()
            
            # Denormalize: images are normalized with mean=0.5, std=0.5, so we need to reverse that
            # Original normalization: (img - 0.5) / 0.5, which maps [0, 1] -> [-1, 1]
            # Reverse: (img + 1) / 2, which maps [-1, 1] -> [0, 1]
            img = (img + 1) / 2
            img = img.clamp(0, 1)
            img = F.interpolate(
                img.unsqueeze(0),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            img = img.permute(1, 2, 0)
            img = (img.numpy() * 255).astype("uint8")
            img_pil = Image.fromarray(img).convert('RGB')
            msk = np.zeros_like(img[:, :, 0])
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                mask = (out_mask[0] > 0).astype(np.uint8) * 255
                # img = mask_painter(img, mask, mask_color=4 + out_obj_id)
                msk[mask == 255] = out_obj_id
            # img_to_video.append(img)

            msk_pil = Image.fromarray(msk).convert('P')
            msk_pil.putpalette(DAVIS_PALETTE)
            img_pil.save(os.path.join(IMAGE_PATH, f"{out_frame_idx+start_frame_idx:08d}.jpg"))
            msk_pil.save(os.path.join(MASKS_PATH, f"{out_frame_idx+start_frame_idx:08d}.png"))

        out_video_path = os.path.join(self.OUTPUT_DIR, f"video_mask_{time.time():.0f}.mp4")
        # images_to_mp4(img_to_video, out_video_path, fps=self.RUNTIME['video_fps'])

        return out_video_path

    def on_4d_generation(self, video_path: str=None):
        """
        Placeholder for 4D generation.
        Later:
        - run sam3_3d_body_model on per-frame images + masks
        - render 4D visualization video
        For now, just log and return None.
        """
        print("[DEBUG] 4D Generation button clicked.")

        IMAGE_PATH = os.path.join(self.OUTPUT_DIR, 'images') # for sam3-3d-body
        MASKS_PATH = os.path.join(self.OUTPUT_DIR, 'masks')  # for sam3-3d-body
        image_extensions = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.bmp",
            "*.tiff",
            "*.webp",
        ]
        images_list = sorted(
            [
                image
                for ext in image_extensions
                for image in glob.glob(os.path.join(IMAGE_PATH, ext))
            ]
        )
        masks_list = sorted(
            [
                image
                for ext in image_extensions
                for image in glob.glob(os.path.join(MASKS_PATH, ext))
            ]
        )

        os.makedirs(f"{self.OUTPUT_DIR}/rendered_frames", exist_ok=True)
        for obj_id in self.RUNTIME['out_obj_ids']:
            os.makedirs(f"{self.OUTPUT_DIR}/mesh_4d_individual/{obj_id}", exist_ok=True)
            os.makedirs(f"{self.OUTPUT_DIR}/focal_4d_individual/{obj_id}", exist_ok=True)
            os.makedirs(f"{self.OUTPUT_DIR}/rendered_frames_individual/{obj_id}", exist_ok=True)

        batch_size = self.RUNTIME['batch_size']
        n = len(images_list)
        
        # Optional, detect occlusions
        pred_res = self.RUNTIME['detection_resolution']
        pred_res_hi = self.RUNTIME['completion_resolution']
        modal_pixels_list = []
        if self.pipeline_mask is not None:
            for obj_id in self.RUNTIME['out_obj_ids']:
                modal_pixels, ori_shape = load_and_transform_masks(self.OUTPUT_DIR + "/masks", resolution=pred_res, obj_id=obj_id)
                modal_pixels_list.append(modal_pixels)
            rgb_pixels, _, raw_rgb_pixels = load_and_transform_rgbs(self.OUTPUT_DIR + "/images", resolution=pred_res)
            depth_pixels = rgb_to_depth(rgb_pixels, self.depth_model)

        mhr_shape_scale_dict = {}   # each element is a list storing input parameters for mhr_forward
        obj_ratio_dict = {}         # avoid fake completion by obj ratio on the first frame

        # Get camera smoothing config
        camera_smoothing_factor = self.CONFIG.runtime.get('camera_smoothing_factor', 0.5)
        enable_camera_smoothing = self.CONFIG.runtime.get('enable_camera_smoothing', True)
        
        # Accumulate all outputs for smoothing across batches
        all_mask_outputs = []  # List of lists: [batch][frame][person]
        all_id_batches = []    # List of lists: [batch][frame][person_ids]
        all_empty_frame_lists = []  # List of lists: [batch][empty_frame_indices]
        all_batch_images_list = []  # List of lists: [batch][image_paths]
        all_batch_masks_list = []   # List of lists: [batch][mask_paths]
        all_idx_paths = []          # List of dicts: [batch][obj_id] -> paths
        all_idx_dicts = []          # List of dicts: [batch][obj_id] -> (start, end)

        # First pass: Process all batches and accumulate outputs
        for i in tqdm(range(0, n, batch_size), desc="Processing frames"):
            batch_images = images_list[i:i + batch_size]
            batch_masks  = masks_list[i:i + batch_size]

            W, H = Image.open(batch_masks[0]).size

            # Optional, detect occlusions
            idx_dict = {}
            idx_path = {}
            occ_dict = {}
            if len(modal_pixels_list) > 0:
                print("detect occlusions ...")
                pred_amodal_masks_dict = {}
                
                # Memory management: clear GPU cache if enabled
                if self.CONFIG.completion.get('clear_gpu_cache', True):
                    torch.cuda.empty_cache()
                
                decode_chunk_size = self.CONFIG.completion.get('decode_chunk_size', 8)
                
                for (modal_pixels, obj_id) in zip(modal_pixels_list, self.RUNTIME['out_obj_ids']):
                    # Clear GPU cache before processing each object to free up memory
                    if self.CONFIG.completion.get('clear_gpu_cache', True):
                        torch.cuda.empty_cache()
                    
                    # detect occlusions for each object
                    # predict amodal masks (amodal segmentation)
                    try:
                        pred_amodal_masks = self.pipeline_mask(
                            modal_pixels[:, i:i + batch_size, :, :, :],
                            depth_pixels[:, i:i + batch_size, :, :, :],
                            height=pred_res[0],
                            width=pred_res[1],
                            num_frames=modal_pixels[:, i:i + batch_size, :, :, :].shape[1],
                            decode_chunk_size=decode_chunk_size,
                            motion_bucket_id=127,
                            fps=8,
                            noise_aug_strength=0.02,
                            min_guidance_scale=1.5,
                            max_guidance_scale=1.5,
                            generator=self.generator,
                        ).frames[0]
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"GPU out of memory for object {obj_id}. Trying with smaller decode_chunk_size...")
                        torch.cuda.empty_cache()
                        # Retry with smaller chunk size
                        pred_amodal_masks = self.pipeline_mask(
                            modal_pixels[:, i:i + batch_size, :, :, :],
                            depth_pixels[:, i:i + batch_size, :, :, :],
                            height=pred_res[0],
                            width=pred_res[1],
                            num_frames=modal_pixels[:, i:i + batch_size, :, :, :].shape[1],
                            decode_chunk_size=max(1, decode_chunk_size // 2),  # Reduce chunk size
                            motion_bucket_id=127,
                            fps=8,
                            noise_aug_strength=0.02,
                            min_guidance_scale=1.5,
                            max_guidance_scale=1.5,
                            generator=self.generator,
                        ).frames[0]

                    # for completion
                    pred_amodal_masks_com = [np.array(img.resize((pred_res_hi[1], pred_res_hi[0]))) for img in pred_amodal_masks]
                    pred_amodal_masks_com = np.array(pred_amodal_masks_com).astype('uint8')
                    pred_amodal_masks_com = (pred_amodal_masks_com.sum(axis=-1) > 600).astype('uint8')
                    pred_amodal_masks_com = [keep_largest_component(pamc) for pamc in pred_amodal_masks_com]
                    # for iou
                    pred_amodal_masks = [np.array(img.resize((W, H))) for img in pred_amodal_masks]
                    pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
                    pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
                    pred_amodal_masks = [keep_largest_component(pamc) for pamc in pred_amodal_masks]    # avoid small noisy masks
                    # compute iou
                    masks = [(np.array(Image.open(bm).convert('P'))==obj_id).astype('uint8') for bm in batch_masks]
                    ious = []
                    masks_margin_shrink = [bm.copy() for bm in masks]
                    mask_H, mask_W = masks_margin_shrink[0].shape
                    for bi, (a, b) in enumerate(zip(masks, pred_amodal_masks)):
                        # mute objects near margin
                        zero_mask_cp = np.zeros_like(masks_margin_shrink[bi])
                        zero_mask_cp[masks_margin_shrink[bi]==1] = 255
                        mask_binary_cp = zero_mask_cp.astype(np.uint8)
                        mask_binary_cp[:int(mask_H*0.05), :] = mask_binary_cp[-int(mask_H*0.05):, :] = mask_binary_cp[:, :int(mask_W*0.05)] = mask_binary_cp[:, -int(mask_W*0.05):] = 0
                        if mask_binary_cp.max() == 0:   # margin objects
                            ious.append(1.0)
                            continue
                        area_a = (a > 0).sum()
                        area_b = (b > 0).sum()
                        if area_a == 0 and area_b == 0:
                            ious.append(1.0)
                        elif area_a > area_b:
                            ious.append(1.0)
                        else:
                            inter = np.logical_and(a > 0, b > 0).sum()
                            uni = np.logical_or(a > 0, b > 0).sum()
                            obj_iou = inter / (uni + 1e-6)
                            ious.append(obj_iou)

                        if i == 0 and bi == 0:
                            if ious[0] < 0.7:
                                obj_ratio_dict[obj_id] = bbox_from_mask(b)
                            else:
                                obj_ratio_dict[obj_id] = bbox_from_mask(a)

                    # remove fake completions (empty or from MARGINs)
                    for pi, pamc in enumerate(pred_amodal_masks_com):
                        # zero predictions, back to original masks
                        if masks[pi].sum() > pred_amodal_masks[pi].sum():
                            ious[pi] = 1.0
                            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
                        # elif len(obj_ratio_dict)>0 and not are_bboxes_similar(bbox_from_mask(pred_amodal_masks[pi]), obj_ratio_dict[obj_id]):
                        #     ious[pi] = 1.0
                        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
                        elif is_super_long_or_wide(pred_amodal_masks[pi], obj_id):
                            ious[pi] = 1.0
                            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
                        elif is_skinny_mask(pred_amodal_masks[pi]):
                            ious[pi] = 1.0
                            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
                        # elif masks[pi].sum() == 0: # TODO: recover empty masks in future versions (to avoid severe fake completion)
                        #     ious[pi] = 1.0
                        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)

                    pred_amodal_masks_dict[obj_id] = pred_amodal_masks_com

                    # confirm occlusions & save masks (for HMR)
                    start, end = (idxs := [ix for ix,x in enumerate(ious) if x < 0.7]) and (idxs[0], idxs[-1]) or (None, None)

                    occ_dict[obj_id] = [1 if ix > 0.7 else 0 for ix in ious]

                    if start is not None and end is not None:
                        start = max(0, start-2)
                        end = min(modal_pixels[:, i:i + batch_size, :, :, :].shape[1]-1, end+2)
                        idx_dict[obj_id] = (start, end)
                        completion_path = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
                        completion_image_path = f'{self.OUTPUT_DIR}/completion/{completion_path}/images'
                        completion_masks_path = f'{self.OUTPUT_DIR}/completion/{completion_path}/masks'
                        os.makedirs(completion_image_path, exist_ok=True)
                        os.makedirs(completion_masks_path, exist_ok=True)
                        idx_path[obj_id] = {'images': completion_image_path, 'masks': completion_masks_path}
                        # save completion masks
                        for idx_ in range(start, end):
                            mask_idx_ = pred_amodal_masks[idx_].copy()
                            mask_idx_[mask_idx_ > 0] = obj_id
                            mask_idx_ = Image.fromarray(mask_idx_).convert('P')
                            mask_idx_.putpalette(DAVIS_PALETTE)
                            mask_idx_.save(os.path.join(completion_masks_path, f"{idx_:08d}.png"))

                # completion
                for obj_id, (start, end) in idx_dict.items(): 
                    completion_image_path = idx_path[obj_id]['images']
                    # prepare inputs
                    modal_pixels_current, ori_shape = load_and_transform_masks(self.OUTPUT_DIR + "/masks", resolution=pred_res_hi, obj_id=obj_id)
                    rgb_pixels_current, _, raw_rgb_pixels_current = load_and_transform_rgbs(self.OUTPUT_DIR + "/images", resolution=pred_res_hi)
                    modal_pixels_current = modal_pixels_current[:, i:i + batch_size, :, :, :]
                    modal_pixels_current = modal_pixels_current[:, start:end]
                    pred_amodal_masks_current = pred_amodal_masks_dict[obj_id][start:end]
                    modal_mask_union = (modal_pixels_current[0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
                    pred_amodal_masks_current = np.logical_or(pred_amodal_masks_current, modal_mask_union).astype('uint8')
                    pred_amodal_masks_tensor = torch.from_numpy(np.where(pred_amodal_masks_current == 0, -1, 1)).float().unsqueeze(0).unsqueeze(
                        2).repeat(1, 1, 3, 1, 1)

                    rgb_pixels_current = rgb_pixels_current[:, i:i + batch_size, :, :, :][:, start:end]
                    modal_obj_mask = (modal_pixels_current > 0).float()
                    modal_background = 1 - modal_obj_mask
                    rgb_pixels_current = (rgb_pixels_current + 1) / 2
                    modal_rgb_pixels = rgb_pixels_current * modal_obj_mask + modal_background
                    modal_rgb_pixels = modal_rgb_pixels * 2 - 1

                    print("content completion by diffusion-vas ...")
                    # predict amodal rgb (content completion)
                    pred_amodal_rgb = self.pipeline_rgb(
                        modal_rgb_pixels,
                        pred_amodal_masks_tensor,
                        height=pred_res_hi[0], # my_res[0]
                        width=pred_res_hi[1],  # my_res[1]
                        num_frames=end-start,
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=8,
                        noise_aug_strength=0.02,
                        min_guidance_scale=1.5,
                        max_guidance_scale=1.5,
                        generator=self.generator,
                    ).frames[0]

                    pred_amodal_rgb = [np.array(img) for img in pred_amodal_rgb]

                    # save pred_amodal_rgb
                    pred_amodal_rgb = np.array(pred_amodal_rgb).astype('uint8')
                    pred_amodal_rgb_save = np.array([cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                                                    for frame in pred_amodal_rgb])
                    idx_ = start
                    for img in pred_amodal_rgb_save:
                        cv2.imwrite(os.path.join(completion_image_path, f"{idx_:08d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        idx_ += 1

            else:
                for obj_id in self.RUNTIME['out_obj_ids']:
                    occ_dict[obj_id] = [1] * len(batch_masks)

            # Process with external mask
            mask_outputs, id_batch, empty_frame_list = process_image_with_mask(self.sam3_3d_body_model, batch_images, batch_masks, idx_path, idx_dict, mhr_shape_scale_dict, occ_dict)
            
            # Accumulate outputs for smoothing
            all_mask_outputs.append(mask_outputs)
            all_id_batches.append(id_batch)
            all_empty_frame_lists.append(empty_frame_list)
            all_batch_images_list.append(batch_images)
            all_batch_masks_list.append(batch_masks)
            all_idx_paths.append(idx_path)
            all_idx_dicts.append(idx_dict)
        
        # Flatten accumulated outputs across all batches
        flattened_mask_outputs = []
        flattened_id_batches = []
        frame_to_batch_map = []  # Map global frame index to batch index and local frame index
        
        global_frame_idx = 0
        for batch_idx, (batch_outputs, batch_ids, empty_frames) in enumerate(zip(all_mask_outputs, all_id_batches, all_empty_frame_lists)):
            batch_images = all_batch_images_list[batch_idx]
            for local_frame_idx in range(len(batch_images)):
                if local_frame_idx in empty_frames:
                    # Skip empty frames in accumulation
                    continue
                # Calculate the index in mask_outputs (accounting for empty frames before this one)
                num_empty_before = sum(1 for ef in empty_frames if ef < local_frame_idx)
                output_idx = local_frame_idx - num_empty_before
                if output_idx < len(batch_outputs):
                    flattened_mask_outputs.append(batch_outputs[output_idx])
                    flattened_id_batches.append(batch_ids[output_idx])
                    frame_to_batch_map.append((batch_idx, local_frame_idx))
                    global_frame_idx += 1
        
        # Apply camera smoothing if enabled
        if enable_camera_smoothing and len(flattened_mask_outputs) > 1:
            print(f"Applying camera smoothing (factor={camera_smoothing_factor})...")
            flattened_mask_outputs = smooth_camera_parameters(
                flattened_mask_outputs, 
                flattened_id_batches, 
                smoothing_factor=camera_smoothing_factor
            )
            print("Camera smoothing completed.")
        
        # Second pass: Render using smoothed outputs
        global_output_idx = 0
        for batch_idx in tqdm(range(len(all_batch_images_list)), desc="Rendering frames"):
            batch_images = all_batch_images_list[batch_idx]
            batch_masks = all_batch_masks_list[batch_idx]
            empty_frame_list = all_empty_frame_lists[batch_idx]
            idx_path = all_idx_paths[batch_idx]
            idx_dict = all_idx_dicts[batch_idx]
            
            num_empth_ids = 0
            for frame_id in range(len(batch_images)):
                image_path = batch_images[frame_id]
                if frame_id in empty_frame_list:
                    mask_output = None
                    id_current = None
                    num_empth_ids += 1
                else:
                    # Use smoothed outputs
                    if global_output_idx < len(flattened_mask_outputs):
                        mask_output = flattened_mask_outputs[global_output_idx]
                        id_current = flattened_id_batches[global_output_idx]
                        global_output_idx += 1
                    else:
                        # Fallback to original processing if smoothing didn't work
                        num_empty_before = sum(1 for ef in empty_frame_list if ef < frame_id)
                        output_idx = frame_id - num_empty_before
                        if output_idx < len(all_mask_outputs[batch_idx]):
                            mask_output = all_mask_outputs[batch_idx][output_idx]
                            id_current = all_id_batches[batch_idx][output_idx]
                        else:
                            mask_output = None
                            id_current = None
                img = cv2.imread(image_path)
                
                # Check if rendering is enabled
                enable_rendering = self.CONFIG.runtime.get('enable_rendering', True)
                skip_on_error = self.CONFIG.runtime.get('skip_rendering_on_error', True)
                
                if enable_rendering:
                    try:
                        rend_img = visualize_sample_together(img, mask_output, self.sam3_3d_body_model.faces, id_current)
                        cv2.imwrite(
                            f"{self.OUTPUT_DIR}/rendered_frames/{os.path.basename(image_path)[:-4]}.jpg",
                            rend_img.astype(np.uint8),
                        )

                        # save rendered frames for individual person
                        rend_img_list = visualize_sample(img, mask_output, self.sam3_3d_body_model.faces, id_current)
                        for ri, rend_img in enumerate(rend_img_list):
                            cv2.imwrite(
                                f"{self.OUTPUT_DIR}/rendered_frames_individual/{ri+1}/{os.path.basename(image_path)[:-4]}_{ri+1}.jpg",
                                rend_img.astype(np.uint8),
                            )
                    except Exception as e:
                        if skip_on_error:
                            print(f"Warning: Rendering failed for frame {frame_id}: {e}")
                            print("Saving original image instead of rendered version.")
                            # Save original image as fallback
                            cv2.imwrite(
                                f"{self.OUTPUT_DIR}/rendered_frames/{os.path.basename(image_path)[:-4]}.jpg",
                                img,
                            )
                            # Save original for individual frames too
                            cv2.imwrite(
                                f"{self.OUTPUT_DIR}/rendered_frames_individual/1/{os.path.basename(image_path)[:-4]}_1.jpg",
                                img,
                            )
                        else:
                            raise
                else:
                    # Rendering disabled, just save original images
                    cv2.imwrite(
                        f"{self.OUTPUT_DIR}/rendered_frames/{os.path.basename(image_path)[:-4]}.jpg",
                        img,
                    )
                    cv2.imwrite(
                        f"{self.OUTPUT_DIR}/rendered_frames_individual/1/{os.path.basename(image_path)[:-4]}_1.jpg",
                        img,
                    )
                # save mesh for individual person
                save_mesh_results(
                    outputs=mask_output, 
                    faces=self.sam3_3d_body_model.faces, 
                    save_dir=f"{self.OUTPUT_DIR}/mesh_4d_individual",
                    focal_dir = f"{self.OUTPUT_DIR}/focal_4d_individual",
                    image_path=image_path,
                    id_current=id_current,
                )

        out_4d_path = os.path.join(self.OUTPUT_DIR, f"4d_{time.time():.0f}.mp4")
        jpg_folder_to_mp4(f"{self.OUTPUT_DIR}/rendered_frames", out_4d_path)

        return out_4d_path


def inference(args):
    # init configs and cover with cmd options
    predictor = OfflineApp()
    if args.output_dir is not None:
        predictor.OUTPUT_DIR = args.output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)

    # human detection on the frame where human FIRST appear
    if os.path.isfile(args.input_video) and args.input_video.endswith(".mp4"):
        input_type = "video"
        image = read_frame_at(args.input_video, 0)
        width, height = image.size
        for starting_frame_idx in range(10, 100):
            image = np.array(read_frame_at(args.input_video, starting_frame_idx))
            outputs = predictor.sam3_3d_body_model.process_one_image(image, bbox_thr=0.6,)
            if len(outputs) > 0:
                break
        
        frame_batch_size = predictor.CONFIG.video_loading.get('frame_batch_size', None)
        offload_to_cpu = predictor.CONFIG.video_loading.get('offload_to_cpu', False)
        inference_state = predictor.predictor.init_state(
            video_path=args.input_video,
            offload_video_to_cpu=offload_to_cpu,
            batch_size=frame_batch_size,
        )
        predictor.predictor.clear_all_points_in_video(inference_state)
        predictor.RUNTIME['inference_state'] = inference_state
        predictor.RUNTIME['out_obj_ids'] = []

        # 1. load bbox (first frame)
        for obj_id, output in enumerate(outputs):
            # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
            xmin, ymin, xmax, ymax = output['bbox']
            rel_box = [[xmin / width, ymin / height, xmax / width, ymax / height]]
            rel_box = np.array(rel_box, dtype=np.float32)
            _, predictor.RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.predictor.add_new_points_or_box(
                inference_state=predictor.RUNTIME['inference_state'],
                frame_idx=starting_frame_idx,
                obj_id=obj_id+1,
                box=rel_box,
            )

    elif os.path.isdir(args.input_video):
        input_type = "images"
        image_list = glob.glob(os.path.join(args.input_video, '*.jpg'))
        image_list.sort()
        image = Image.open(image_list[0]).convert('RGB')
        width, height = image.size
        starting_frame_idx = 0
        for image_path in image_list:
            outputs = predictor.sam3_3d_body_model.process_one_image(image_path, bbox_thr=0.6,)
            if len(outputs) > 0:
                break
            starting_frame_idx += 1

        frame_batch_size = predictor.CONFIG.video_loading.get('frame_batch_size', None)
        offload_to_cpu = predictor.CONFIG.video_loading.get('offload_to_cpu', False)
        inference_state = predictor.predictor.init_state(
            video_path=image_list,
            offload_video_to_cpu=offload_to_cpu,
            batch_size=frame_batch_size,
        )
        predictor.predictor.clear_all_points_in_video(inference_state)
        predictor.RUNTIME['inference_state'] = inference_state
        predictor.RUNTIME['out_obj_ids'] = []

        # 1. load bbox (first frame)
        for obj_id, output in enumerate(outputs):
            # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
            xmin, ymin, xmax, ymax = output['bbox']
            rel_box = [[xmin / width, ymin / height, xmax / width, ymax / height]]
            rel_box = np.array(rel_box, dtype=np.float32)
            _, predictor.RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.predictor.add_new_points_or_box(
                inference_state=predictor.RUNTIME['inference_state'],
                frame_idx=starting_frame_idx,
                obj_id=obj_id+1,
                box=rel_box,
            )

    # 2. tracking
    predictor.on_mask_generation(start_frame_idx=0)
    # 3. hmr upon masks
    with torch.autocast("cuda", enabled=False):
        predictor.on_4d_generation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline 4D Body Generation from Videos")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video (either *.mp4 or a directory containing image sequences)")
    args = parser.parse_args()

    input_path = args.input_video
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"--input_video does not exist: {input_path}")
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".mp4"):
            raise ValueError(
                f"--input_video must be an .mp4 file or a directory, got file: {input_path}"
            )
    elif os.path.isdir(input_path):
        # Optional: check whether the directory contains images
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        images = [
            f for f in os.listdir(input_path)
            if f.lower().endswith(valid_ext)
        ]
        if len(images) == 0:
            raise ValueError(
                f"--input_video directory contains no image files: {input_path}"
            )
    else:
        raise ValueError(
            f"--input_video must be an .mp4 file or a directory: {input_path}"
        )

    inference(args)
