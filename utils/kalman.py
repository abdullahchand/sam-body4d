import numpy as np

def kalman_smooth_constant_velocity(Y,
                                    q_pos=1e-4,
                                    q_vel=1e-6,
                                    r_obs=1e-2):
    """
    Apply constant-velocity Kalman filtering along the temporal dimension.

    Each parameter dimension is considered as an independent 1-D motion:
        state = [position, velocity]

    Motion model (Δt = 1):
        x_t = x_{t-1} + v_{t-1} + w_pos
        v_t = v_{t-1} + w_vel
    Observation:
        y_t = x_t + noise

    Args:
        Y:     np.ndarray of shape (B, D)
               B: number of frames, D: parameter dimension
        q_pos: process noise for position   (smaller → stronger smoothing)
        q_vel: process noise for velocity   (larger → allows fast motion)
        r_obs: observation noise            (larger → stronger smoothing)

    Returns:
        Smoothed position parameters of shape (B, D)
    """
    Y = np.asarray(Y, dtype=np.float32)
    T, D = Y.shape
    if T == 0:
        return Y.copy()

    # Initialize position from the first frame, and assume initial velocity = 0
    x = Y[0].copy()
    v = np.zeros(D, dtype=np.float32)

    # Initialize covariance terms: P is 2×2 per dimension:
    # [Pxx, Pxv;
    #  Pvx, Pvv]
    Pxx = np.ones(D, dtype=np.float32)
    Pxv = np.zeros(D, dtype=np.float32)
    Pvv = np.ones(D, dtype=np.float32)

    X_smooth = np.zeros_like(Y)
    X_smooth[0] = x

    for t in range(1, T):
        # ---------- Prediction ----------
        x_pred = x + v      # position
        v_pred = v          # velocity remains same

        # Covariance prediction: A = [[1,1],[0,1]]
        Pxx_pred = Pxx + 2 * Pxv + Pvv + q_pos
        Pxv_pred = Pxv + Pvv
        Pvv_pred = Pvv + q_vel

        # ---------- Update ----------
        y = Y[t]
        S = Pxx_pred + r_obs + 1e-8  # innovation covariance

        K_pos = Pxx_pred / S         # Kalman gain for position
        K_vel = Pxv_pred / S         # Kalman gain for velocity

        innovation = y - x_pred

        x = x_pred + K_pos * innovation
        v = v_pred + K_vel * innovation

        Pxx = (1.0 - K_pos) * Pxx_pred
        Pxv = (1.0 - K_pos) * Pxv_pred
        Pvv = Pvv_pred - K_vel * Pxv_pred

        X_smooth[t] = x

    return X_smooth

import torch

def kalman_smooth_mhr_params_multi_human(mhr_dict,
                                         num_frames,
                                         keys_to_smooth=None,
                                         kalman_cfg=None):
    """
    Apply temporal Kalman smoothing to MHR rig parameters when B = T * N
    (T frames, N humans per frame).

    Args:
        mhr_dict: dict containing tensors like:
                  "global_rot": (B, 3)
                  "body_pose":  (B, 133)
                  "hand":       (B, 108)
                  ...
                  where B = num_frames * num_humans
        num_frames: int, number of frames T in the video
        keys_to_smooth: list of keys to be smoothed.
                        If None → default: ["global_rot", "body_pose"]
        kalman_cfg: dict, key -> {q_pos, q_vel, r_obs}
                    Kalman settings per parameter group.

    Returns:
        new_mhr: dict with the same structure as mhr_dict,
                 but selected fields are Kalman-smoothed along time
                 for each human independently.
    """
    if keys_to_smooth is None:
        keys_to_smooth = ["global_rot", "body_pose"]

    if kalman_cfg is None:
        kalman_cfg = {
            "global_rot": dict(q_pos=1e-4, q_vel=1e-5, r_obs=1e-2),
            "body_pose":  dict(q_pos=5e-4, q_vel=1e-4, r_obs=5e-2),
            "hand":       dict(q_pos=5e-4, q_vel=1e-4, r_obs=5e-2),
        }

    new_mhr = {}
    # Use any field to infer B and N
    # Here we use "global_rot" safely (it always exists in your format)
    B = mhr_dict["global_rot"].shape[0]
    assert B % num_frames == 0, "B must be divisible by num_frames"
    num_humans = B // num_frames

    for k, v in mhr_dict.items():
        # v: (B, D)
        if k in keys_to_smooth:
            cfg = kalman_cfg.get(k, kalman_cfg["body_pose"])
            device = v.device
            B, D = v.shape

            # (B, D) → (T, N, D)
            v_np = v.detach().cpu().numpy().reshape(num_frames, num_humans, D)
            # (T, N, D) → (T, N*D) so we can filter all humans & dims at once
            v_flat = v_np.reshape(num_frames, num_humans * D)

            v_flat_smooth = kalman_smooth_constant_velocity(
                v_flat,
                q_pos=cfg["q_pos"],
                q_vel=cfg["q_vel"],
                r_obs=cfg["r_obs"],
            )

            # Back to (T, N, D) → (B, D)
            v_smooth_np = v_flat_smooth.reshape(num_frames * num_humans, D)
            new_mhr[k] = torch.from_numpy(v_smooth_np).to(device)
        else:
            # Fields not smoothed are copied as-is
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
