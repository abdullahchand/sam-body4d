import torch
import numpy as np

def kalman_smooth_constant_velocity_safe(Y, q_pos=1e-4, q_vel=1e-6, r_obs=1e-2):
    """
    Robust constant-velocity Kalman smoothing on (T, D).

    Y: (T, D) numpy array of valid observations for a single obj_id.
       Missing frames are handled outside this function.
    """
    Y = np.asarray(Y, dtype=np.float32)
    T, D = Y.shape
    if T == 0:
        return Y.copy()

    # Remove NaN / inf from input
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    q_pos = float(max(q_pos, 0.0))
    q_vel = float(max(q_vel, 0.0))
    r_obs = float(max(r_obs, 1e-12))

    # Initial state: first observation as position, zero velocity
    x = Y[0].copy()                   # (D,)
    v = np.zeros(D, dtype=np.float32) # (D,)

    Pxx = np.ones(D, dtype=np.float32)
    Pxv = np.zeros(D, dtype=np.float32)
    Pvv = np.ones(D, dtype=np.float32)

    X = np.zeros_like(Y)
    X[0] = x

    eps = 1e-8
    max_val = 1e6

    for t in range(1, T):
        # ---- Prediction ----
        x_pred = x + v
        v_pred = v

        Pxx_pred = Pxx + 2 * Pxv + Pvv + q_pos
        Pxv_pred = Pxv + Pvv
        Pvv_pred = Pvv + q_vel

        Pxx_pred = np.clip(Pxx_pred, -max_val, max_val)
        Pxv_pred = np.clip(Pxv_pred, -max_val, max_val)
        Pvv_pred = np.clip(Pvv_pred, -max_val, max_val)

        # ---- Update ----
        y = Y[t]
        S = Pxx_pred + r_obs
        S = np.where(np.abs(S) < eps, eps, S)

        K_pos = Pxx_pred / S
        K_vel = Pxv_pred / S

        innovation = y - x_pred

        x = x_pred + K_pos * innovation
        v = v_pred + K_vel * innovation

        Pxx = (1.0 - K_pos) * Pxx_pred
        Pxv = (1.0 - K_pos) * Pxv_pred
        Pvv = Pvv_pred - K_vel * Pxv_pred

        # Clamp to avoid NaNs / inf
        x = np.nan_to_num(x, nan=0.0, posinf=max_val, neginf=-max_val)
        v = np.nan_to_num(v, nan=0.0, posinf=max_val, neginf=-max_val)
        Pxx = np.nan_to_num(Pxx, nan=1.0, posinf=max_val, neginf=0.0)
        Pxv = np.nan_to_num(Pxv, nan=0.0, posinf=max_val, neginf=-max_val)
        Pvv = np.nan_to_num(Pvv, nan=1.0, posinf=max_val, neginf=0.0)

        X[t] = x

    X = np.nan_to_num(X, nan=0.0, posinf=max_val, neginf=-max_val)
    return X

import torch

def kalman_smooth_mhr_params_multi_human_with_ids(
    mhr_dict,
    num_frames,
    frame_obj_ids,
    keys_to_smooth=None,
    kalman_cfg=None,
    empty_thresh=1e-6,
):
    """
    Apply Kalman smoothing per obj_id, using only frames where this obj_id exists.

    Data layout (unchanged):
        - mhr_dict[key]: (B, D), B = num_frames * num_humans
        - Flatten order in B:
              frame 0: slots 0..num_humans-1
              frame 1: slots 0..num_humans-1
              ...
        - slot index s in [0, num_humans) corresponds to obj_id = s + 1.

    frame_obj_ids:
        - list of length num_frames
        - frame_obj_ids[t] is a list of obj_id (1-based) present in frame t

    Behavior:
        - For each key in keys_to_smooth and each obj_id:
            * Build the list of (frame, B_index) where this obj_id exists.
            * Extract these rows as a track Y_valid of shape (T_valid, D).
            * Run Kalman on Y_valid (only valid frames are history).
            * Write smoothed values back only to those B_indices.
        - Missing frames for this obj_id are never used as Kalman history
          and keep their original values in mhr_dict.
    """
    if keys_to_smooth is None:
        keys_to_smooth = ["body_pose", "hand"]

    if kalman_cfg is None:
        kalman_cfg = {
            "body_pose": dict(q_pos=5e-4, q_vel=3e-4, r_obs=8e-2),
            "hand":      dict(q_pos=4e-4, q_vel=4e-4, r_obs=1.2e-1),
        }

    assert len(frame_obj_ids) == num_frames, "frame_obj_ids length must equal num_frames"

    new_mhr = {}

    # Infer B and num_humans
    any_key = next(iter(mhr_dict.keys()))
    B = mhr_dict[any_key].shape[0]
    assert B % num_frames == 0, "B must be divisible by num_frames"
    num_humans = B // num_frames

    for k, v in mhr_dict.items():
        if k in keys_to_smooth:
            cfg = kalman_cfg.get(k, kalman_cfg.get("body_pose"))
            device = v.device
            B, D = v.shape

            v_np = v.detach().cpu().numpy()  # (B, D)

            # Loop over obj_id = 1..num_humans
            for obj_id in range(1, num_humans + 1):
                slot = obj_id - 1  # slot index in [0, num_humans)

                # Collect all B indices where this obj_id is present
                valid_indices = []
                for t in range(num_frames):
                    if obj_id in frame_obj_ids[t]:
                        # B index = t * num_humans + slot
                        idx = t * num_humans + slot
                        valid_indices.append(idx)

                # If this obj_id never appears, skip
                if len(valid_indices) == 0:
                    continue

                track_valid = v_np[valid_indices, :]  # (T_valid, D)

                # Skip almost-empty tracks
                if np.linalg.norm(track_valid) < empty_thresh:
                    continue

                # Run Kalman only on valid frames for this obj_id
                smoothed_valid = kalman_smooth_constant_velocity_safe(
                    track_valid,
                    q_pos=cfg["q_pos"],
                    q_vel=cfg["q_vel"],
                    r_obs=cfg["r_obs"],
                )

                # If something went wrong, keep original
                if not np.isfinite(smoothed_valid).all():
                    continue

                # Write smoothed values back to those valid indices
                v_np[valid_indices, :] = smoothed_valid

            new_mhr[k] = torch.from_numpy(v_np).to(device)
        else:
            # Keys we do not smooth are copied unchanged
            new_mhr[k] = v

    return new_mhr

import numpy as np

def local_window_smooth(Y, window=9, weights=None):
    """
    Strong local smoothing over a temporal window.

    Args:
        Y:       np.ndarray, shape (T, D)
        window:  odd int, temporal window size (e.g., 7 or 9)
                 for frame t, we average over [t-half, t+half]
        weights: optional np.ndarray, shape (T,)
                 per-frame reliability/visibility in [0, 1].
                 If provided, use weighted average inside the window.

    Returns:
        Smoothed Y of shape (T, D)
    """
    Y = np.asarray(Y, dtype=np.float32)
    T, D = Y.shape
    out = np.zeros_like(Y)
    half = window // 2

    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        w = np.clip(w, 0.0, 1.0)
    else:
        w = None

    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)  # [s, e)

        if w is None:
            out[t] = Y[s:e].mean(axis=0)
        else:
            ww = w[s:e]
            ww_sum = ww.sum()
            if ww_sum < 1e-6:
                # if all weights ~0, fall back to simple mean
                out[t] = Y[s:e].mean(axis=0)
            else:
                ww_norm = ww / ww_sum
                out[t] = (Y[s:e] * ww_norm[:, None]).sum(axis=0)

    return out


import torch

def smooth_scale_shape_local(mhr, num_frames, window=9,
                             vis_scale=None, vis_shape=None):
    """
    Apply strong local window smoothing on 'scale' and 'shape' for multi-human case.

    Args:
        mhr:         dict with 'scale' and 'shape' tensors of shape (B, D)
        num_frames:  int, T
        window:      odd int, temporal window size
        vis_scale:   optional (B,) or (T,) visibility/confidence for scale
        vis_shape:   optional (B,) or (T,) visibility/confidence for shape

    Returns:
        new_scale, new_shape: tensors with the same shape as input
    """
    scale = mhr["scale"]
    shape = mhr["shape"]
    device = scale.device

    B, D_scale = scale.shape
    _, D_shape = shape.shape
    assert B % num_frames == 0, "B must be divisible by num_frames"
    num_humans = B // num_frames

    scale_np = scale.detach().cpu().numpy().reshape(num_frames, num_humans, D_scale)
    shape_np = shape.detach().cpu().numpy().reshape(num_frames, num_humans, D_shape)

    # Optional visibility weights per frame (shared across humans)
    if vis_scale is not None:
        vs = np.asarray(vis_scale, dtype=np.float32).reshape(num_frames)
    else:
        vs = None

    if vis_shape is not None:
        vh = np.asarray(vis_shape, dtype=np.float32).reshape(num_frames)
    else:
        vh = None

    for h in range(num_humans):
        scale_np[:, h, :] = local_window_smooth(scale_np[:, h, :], window=window, weights=vs)
        shape_np[:, h, :] = local_window_smooth(shape_np[:, h, :], window=window, weights=vh)

    scale_smooth = torch.from_numpy(scale_np.reshape(B, D_scale)).to(device)
    shape_smooth = torch.from_numpy(shape_np.reshape(B, D_shape)).to(device)
    return scale_smooth, shape_smooth
