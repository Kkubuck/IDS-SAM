"""Mask-conditioned recognition image view."""

import cv2
import numpy as np
from PIL import Image


def mask_conditioned_view(
    img: Image.Image,
    mask_gray: Image.Image,
    blur_sigma: float,
    mask_invert: bool = False,
    reliability_gain=1.0,
    contour_gain=1.0,
) -> Image.Image:
    img_np = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    p = np.asarray(mask_gray, dtype=np.float32) / 255.0
    p = np.clip(p, 0.0, 1.0)
    if mask_invert:
        p = 1.0 - p
    p_s = cv2.GaussianBlur(p, (0, 0), sigmaX=1.2, sigmaY=1.2)
    p_s = np.clip(p_s, 0.0, 1.0)
    su = 0.12
    u_map = np.exp(-((p_s - 0.5) ** 2) / (2.0 * su**2 + 1e-12))
    u = float(np.mean(u_map))
    area = float(np.mean(p_s))
    pen_area = max(0.0, area - 0.35)
    ycc = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    e_img = np.sqrt(gx * gx + gy * gy)
    mx = cv2.Sobel(p_s, cv2.CV_32F, 1, 0, ksize=3)
    my = cv2.Sobel(p_s, cv2.CV_32F, 0, 1, ksize=3)
    e_mask = np.sqrt(mx * mx + my * my)
    e_mask_norm = e_mask / (float(e_mask.mean()) + 1e-06)
    align = float(np.mean(e_img * e_mask_norm) / (float(e_img.mean()) + 1e-06))
    r_logit = 2.5 * (align - 1.2) - 2.0 * (u - 0.35) - 3.0 * pen_area
    r = 1.0 / (1.0 + np.exp(-r_logit))
    paci_r_gain = reliability_gain
    paci_r_gain = max(0.1, min(5.0, paci_r_gain))
    r = float(np.clip(r * paci_r_gain, 0.0, 1.0))
    if r < 0.001:
        return img
    e = e_mask
    if float(e.max()) > 0:
        e = e / float(e.max())
    g1 = cv2.GaussianBlur(e, (0, 0), sigmaX=1.0, sigmaY=1.0)
    g2 = cv2.GaussianBlur(e, (0, 0), sigmaX=2.5, sigmaY=2.5)
    e_spread = 0.6 * g1 + 0.4 * g2
    (h, w) = p_s.shape
    patch = 16
    pw = max(1, w // patch)
    ph = max(1, h // patch)
    e_patch = cv2.resize(e_spread, (pw, ph), interpolation=cv2.INTER_AREA)
    e_patch_up = cv2.resize(e_patch, (w, h), interpolation=cv2.INTER_LINEAR)
    s_map = e_spread * e_patch_up
    s_map = s_map / (float(s_map.max()) + 1e-06)
    w_unc = np.exp(-((p_s - 0.5) ** 2) / (2.0 * su**2 + 1e-12))
    s_safe = s_map * (1.0 - w_unc)
    m = (p_s > 0.5).astype(np.uint8)
    m_area = float(m.sum())
    thin = 0.0
    if m_area > 0:
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        b = np.logical_xor(cv2.dilate(m, k3) > 0, cv2.erode(m, k3) > 0)
        ratio = float(b.sum()) / (m_area + 1e-06)
        thin = float(np.clip((ratio - 0.12) / (0.3 - 0.12), 0.0, 1.0))
    s_safe = np.clip(s_safe * (1.0 - 0.3 * thin), 0.0, 1.0)
    eta_scale = float(max(0.4, min(2.0, blur_sigma / 1.5))) if blur_sigma > 0 else 1.0
    paci_gain = contour_gain
    paci_gain = max(0.1, min(8.0, paci_gain))
    eta_y = 0.04 * eta_scale * paci_gain
    eta_c = 0.02 * eta_scale * paci_gain
    mu = cv2.blur(y, (15, 15))
    sign = np.sign(y - mu)
    sign[sign == 0] = 1.0
    y_new = np.clip(y + r * eta_y * sign * s_safe, 0.0, 1.0)
    cr = ycc[..., 1]
    cb = ycc[..., 2]
    bg = p_s < 0.2
    if int(bg.sum()) > 64:
        cb_mean = float(cb[bg].mean())
        cr_mean = float(cr[bg].mean())
    else:
        cb_mean = float(cb.mean())
        cr_mean = float(cr.mean())
    cb_dir = np.sign(cb_mean - cb)
    cr_dir = np.sign(cr_mean - cr)
    cb_new = np.clip(cb + r * eta_c * s_safe * cb_dir, 0.0, 1.0)
    cr_new = np.clip(cr + r * eta_c * s_safe * cr_dir, 0.0, 1.0)
    ycc_new = np.stack([y_new, cr_new, cb_new], axis=-1).astype(np.float32)
    out = cv2.cvtColor(ycc_new, cv2.COLOR_YCrCb2RGB)
    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")
