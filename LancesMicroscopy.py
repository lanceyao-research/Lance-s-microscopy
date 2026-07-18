# Re-defining everything after kernel reset
import os
import time
import numpy as np

import tkinter as tk
from tkinter import ttk

from skimage.draw import polygon
from skimage import io, filters

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

import socket
import threading
import json

CONFIG_FILE = "microscopy_config.json"

def load_config():
    """Load configuration from JSON file, return defaults if not found."""
    defaults = {
        "mode": "TEM",
        "fps": 30.0,
        "poisson_k": 100.0,
        "gaussian_std": 0.02,
        "percentile_low": 1.0,
        "percentile_high": 99.0,
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                for key in defaults:
                    if key not in saved:
                        saved[key] = defaults[key]
                return saved
    except Exception as e:
        print(f"Error loading config: {e}")
    return defaults

def save_config(config):
    """Save configuration to JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")

def generate_square_lattice(square_size=31000, gap=54000, extent=1.5e6, theta=0.0):
    step = square_size + gap
    coords = np.arange(-extent, extent + 1e-9, step)
    half = square_size / 2.0

    base = np.array([[-half, -half], [ half, -half], [ half,  half], [-half,  half]], dtype=float)

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s,  c]], dtype=float)

    squares = []
    for cx in coords:
        for cy in coords:
            sq = base + np.array([cx, cy], dtype=float)
            sq = (sq @ R.T)
            squares.append(sq)
    return squares

def world_grid(M, FOV, win_cx, win_cy):
    xs = np.linspace(win_cx - FOV/2.0, win_cx + FOV/2.0, M)
    ys = np.linspace(win_cy - FOV/2.0, win_cy + FOV/2.0, M)
    return np.meshgrid(xs, ys)

def world_to_pix(x, y, M, FOV, win_cx, win_cy):
    x0 = win_cx - FOV/2.0
    y0 = win_cy - FOV/2.0
    scale = (M - 1) / FOV
    col = (x - x0) * scale
    row = (y - y0) * scale
    return col, row


def square_mask(squares, outer_d, inner_d, M, win_cx, win_cy, FOV, trans=0.4):
    mask = np.zeros((M, M), dtype=bool)

    Xw, Yw = world_grid(M, FOV, win_cx, win_cy)
    R = np.sqrt(Xw**2 + Yw**2)
    r_inner, r_outer = inner_d/2.0, outer_d/2.0

    for sq in squares:
        cx, cy = sq[:, 0].mean(), sq[:, 1].mean()
        if (cx*cx + cy*cy) > (r_inner*r_inner):
            continue

        c, r = world_to_pix(sq[:, 0], sq[:, 1], M, FOV, win_cx, win_cy)
        rr, cc = polygon(r, c, shape=(M, M))
        mask[rr, cc] = True

    img = mask.astype(np.float32) * trans

    outer_mask = (R > r_outer)
    img[outer_mask]  = 1.0
    mask[outer_mask] = True

    return img, mask, outer_mask

def generate_lowpoly_for_squares(squares, prob, rng=None, k_range=(5, 20), sigma=1.0):
    if rng is None:
        rng = np.random.default_rng()

    lowpolys = []
    trans_poly = []
    for sq in squares:
        if rng.random() > prob:
            lowpolys.append(None)
            trans_poly.append(0.0)
            continue
        center = sq.mean(axis=0)
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        pts = rng.normal(loc=center, scale=sigma, size=(k, 2))
        hull = _convex_hull_xy(pts)
        if hull.shape[0] < 3:
            hull = sq.copy()
        lowpolys.append(hull)
        trans_poly.append(float(rng.uniform(0.4, 0.6)))

    return lowpolys, np.array(trans_poly, dtype=np.float32)


def _convex_hull_xy(points):
    pts = np.asarray(points, dtype=float)
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull

def sample_hole_polygons(outer_d, N, sz, n, rng=None, K=None):
    if rng is None:
        rng = np.random.default_rng()
    r_outer = outer_d * 0.5

    if K is None:
        K = int(rng.integers(0, N + 1))

    polys = []
    if K == 0:
        return polys

    thetas = rng.uniform(0, 2*np.pi, size=K)
    radii  = r_outer * np.sqrt(rng.uniform(0.0, 1.0, size=K))
    centers = np.c_[radii * np.cos(thetas), radii * np.sin(thetas)]

    for (xi, yi) in centers:
        pts = rng.normal(loc=(xi, yi), scale=sz, size=(n, 2))
        if pts.shape[0] < 3:
            continue
        hull = _convex_hull_xy(pts)
        if hull.shape[0] >= 3:
            polys.append(hull)

    return polys
    
def hole_mask_from_polys(polys, outer_d, M, FOV, win_cx, win_cy):
    mask = np.ones((M, M), dtype=bool)

    for hull in polys:
        c, r = world_to_pix(hull[:, 0], hull[:, 1], M, FOV, win_cx, win_cy)
        rr, cc = polygon(r, c, shape=(M, M))
        if rr.size:
            mask[rr, cc] = False

    Xw, Yw = world_grid(M, FOV, win_cx, win_cy)
    R = np.sqrt(Xw**2 + Yw**2)
    mask[R > (outer_d * 0.5)] = True
    return mask

def lowpoly_fill_within_squares_assign(polys, trans_poly, sq_mask, img,
                                       M, FOV, win_cx, win_cy, base_trans=1.0):
    mask = np.zeros_like(sq_mask, dtype=bool)
    trans_poly = np.asarray(trans_poly, dtype=np.float32)
    
    def f(x, k, a, b):
        t = (k - b) / (a - b)
        t = np.clip(t, 0.0, 1.0)
        s = t*t*(3 - 2*t)
        return np.asarray(x) ** s
    
    for hull, tpoly in zip(polys, trans_poly):
        if hull is None or hull.size < 3:
            continue

        c, r = world_to_pix(hull[:, 0], hull[:, 1], M, FOV, win_cx, win_cy)
        rr, cc = polygon(r, c, shape=img.shape)
        if rr.size == 0:
            continue

        inside = sq_mask[rr, cc]
        if not inside.any():
            continue

        val = float(base_trans) * f(float(tpoly),  FOV, 1e6, 1e4)
        img[rr[inside], cc[inside]]  = val
        mask[rr[inside], cc[inside]] = True

    return img, mask


class Repeater:
    def __init__(self, root, interval_ms, fn):
        self.root = root
        self.interval = int(interval_ms)
        self.fn = fn
        self._job = None
        self._running = False

    def _step(self):
        if not self._running:
            return
        self.fn()
        self._job = self.root.after(self.interval, self._step)

    def start(self):
        if self._running:
            return
        self._running = True
        self._step()

    def stop(self):
        self._running = False
        if self._job:
            self.root.after_cancel(self._job)
            self._job = None

class Microscopy:
    def __init__(self):
        self.M = 320
        self.OD = 3e6
        self.ID = 2.5e6
        self.sz = 1e5
        self.n = 20
        self.r_outer = self.OD / 2.0
        self.N = 10
        
        self.rng = np.random.default_rng()
        
        self.zoom_levels = np.array(
            [m * 10**e for e in range(3, 6) for m in range(1, 10)] + [1e6],
            dtype=float
        )
        
        self.trans = self.rng.uniform(0.3, 0.7)
        self.HW = self.rng.uniform(54000, 90000)
        self.BW = self.rng.uniform(31000, 37000)
        self.sigma = self.rng.uniform(1, 5.0)
        self.salt_prob = 10 ** self.rng.uniform(-8, -3)
        
        self.lowpoly_prob = self.rng.uniform(0.1, 0.5)
        self.lowpoly_sigma = 0.25 * self.HW
        self.lowpoly_k_range = (10, 40)
        
        self.theta = self.rng.uniform(0, 2*np.pi)
        self.FOV = 1e6
        self.x = 0
        self.y = 0
        
        # Noise parameters
        self.poisson_k = 100.0
        self.gaussian_std = 0.02
        
        self.squares = generate_square_lattice(self.HW, self.BW, extent=self.r_outer, theta=self.theta)
        self.holes1_polys = sample_hole_polygons(self.OD, self.N, self.sz, self.n, rng=self.rng)
        self.holes2_polys = sample_hole_polygons(self.OD, self.N, self.sz, self.n, rng=self.rng)

        self.target_x = self.x
        self.target_y = self.y
        self.move_active = False

        self.move_velocity = 2.0e5
        self.drift_active = False
        self.drift_vx = 0.0
        self.drift_vy = 0.0
        self._last_motion_t = time.perf_counter()

        self.zoom_target_idx = np.searchsorted(self.zoom_levels, self.FOV)

        self.square_lowpolys, self.square_lowpoly_trans = generate_lowpoly_for_squares(
            self.squares,
            self.lowpoly_prob,
            rng=self.rng,
            k_range=self.lowpoly_k_range,
            sigma=self.lowpoly_sigma,
        )
        self.square_highpolys = [None] * len(self.square_lowpolys)

    
    def render_highpoly(self, idx, img, sq_mask):
        highpolys = self.square_highpolys[idx]
        if highpolys is None:
            return img

        low_t = 0.2
        a, b = 1e4, 1e3

        t = (self.FOV - b) / (a - b)
        t = np.clip(t, 0.0, 1.0)
        s = t*t*(3 - 2*t)
        vis_factor = low_t*(1.0 - s) + 1.0*s

        for hull in highpolys:
            c, r = world_to_pix(hull[:, 0], hull[:, 1], self.M, self.FOV, self.x, self.y)
            rr, cc = polygon(r, c, shape=img.shape)
            if rr.size == 0:
                continue
            inside = sq_mask[rr, cc]
            if not inside.any():
                continue
            img[rr[inside], cc[inside]] = self.trans * vis_factor

        return img

    def init_highpoly_for_square(self, idx):
        if self.square_highpolys[idx] is not None:
            return
        print(f'Initializing high polys for window {idx}')

        lowpoly = self.square_lowpolys[idx]
        if lowpoly is None:
            return

        minx, miny = lowpoly.min(axis=0)
        maxx, maxy = lowpoly.max(axis=0)

        polys = []
        
        for _ in range(int(self.rng.uniform(1000, 10000))):
            cx = self.rng.uniform(minx, maxx)
            cy = self.rng.uniform(miny, maxy)

            k = int(self.rng.integers(3, 11))
            pts = self.rng.normal(loc=(cx, cy), scale=50.0, size=(k, 2))
            hull = _convex_hull_xy(pts)
            if hull.shape[0] >= 3:
                polys.append(hull)

        print(f'{len(polys)} polygons generated for window {idx}')
        self.square_highpolys[idx] = polys
        
    def render_clean(self):
        """Render the image without noise (for normalization reference)."""
        img0, sq_mask, _ = square_mask(self.squares, self.OD, self.ID, self.M, self.x, self.y, self.FOV, self.trans)
        holes_mask1 = hole_mask_from_polys(self.holes1_polys, self.OD, self.M, self.FOV, self.x, self.y)
        holes_mask2 = hole_mask_from_polys(self.holes2_polys, self.OD, self.M, self.FOV, self.x, self.y)
        img_lp, _ = lowpoly_fill_within_squares_assign(
            self.square_lowpolys, self.square_lowpoly_trans,
            sq_mask,
            img0,
            self.M, self.FOV, self.x, self.y,
            base_trans=self.trans
        )
        if self.FOV < 1e4:
            centers = np.array([sq.mean(axis=0) for sq in self.squares])
            dx = centers[:, 0] - self.x
            dy = centers[:, 1] - self.y
            idx = int(np.argmin(dx*dx + dy*dy))

            if self.square_lowpolys[idx] is not None:
                self.init_highpoly_for_square(idx)
                img_lp = self.render_highpoly(idx, img_lp, sq_mask)
                
        img = np.copy(img_lp)
        img[np.logical_and(~holes_mask1, sq_mask)] = 1.0
        img[np.logical_and(~holes_mask2, sq_mask)] = 0.0
        
        # Apply only Gaussian blur (part of the optical system, not noise)
        img = filters.gaussian(img, sigma=self.sigma, preserve_range=True)
        
        return np.clip(img, 0.0, 1.0).astype(np.float32)
    
    def apply_noise(self, img):
        """
        Apply noise to a clean image.
        
        Poisson noise: I' = Poisson(I * k) / k
          - Models shot noise from discrete electron counts
          - std = sqrt(I * k) / k = sqrt(I / k)
          - SNR = I / std = sqrt(I * k) → higher for bright pixels
          
        Gaussian noise: I'' = I' + N(0, σ)
          - Models readout noise (uniform across all pixels)
        """
        img = img.copy()
        
        # Poisson noise
        k = max(self.poisson_k, 1.0)
        img = np.random.poisson(np.maximum(img * k, 0).astype(np.float64)).astype(np.float64) / k
        
        # Gaussian noise
        if self.gaussian_std > 0:
            img = img + self.rng.normal(0, self.gaussian_std, img.shape)
        
        # Salt noise (hot pixels) - very small probability
        m = self.rng.random(img.shape) < self.salt_prob
        img[m] = np.maximum(img[m], 1.0)  # Hot pixels are bright
        
        return img.astype(np.float32)
    
    def apply_noise_to_line(self, line):
        """Apply noise to a single line."""
        line = line.copy()
        
        # Poisson noise
        k = max(self.poisson_k, 1.0)
        line = np.random.poisson(np.maximum(line * k, 0).astype(np.float64)).astype(np.float64) / k
        
        # Gaussian noise
        if self.gaussian_std > 0:
            line = line + self.rng.normal(0, self.gaussian_std, line.shape)
        
        # Salt noise
        m = self.rng.random(line.shape) < self.salt_prob
        line[m] = np.maximum(line[m], 1.0)
        
        return line.astype(np.float32)
        
    def x_plus(self):
        self.x += self.FOV * 0.005
        self.x = np.clip(self.x, -self.r_outer, self.r_outer)

    def x_minus(self):
        self.x -= self.FOV * 0.005
        self.x = np.clip(self.x, -self.r_outer, self.r_outer)

    def y_plus(self):
        self.y += self.FOV * 0.005
        self.y = np.clip(self.y, -self.r_outer, self.r_outer)

    def y_minus(self):
        self.y -= self.FOV * 0.005
        self.y = np.clip(self.y, -self.r_outer, self.r_outer)

    def z_out_dis(self):
        idx = np.searchsorted(self.zoom_levels, self.FOV, side='right')
        if idx < len(self.zoom_levels):
            self.FOV = self.zoom_levels[idx]

    def z_in_dis(self):
        idx = np.searchsorted(self.zoom_levels, self.FOV, side='left') - 1
        if idx >= 0:
            self.FOV = self.zoom_levels[idx]

    def capture(self):
        """Capture with noise applied."""
        clean = self.render_clean()
        return self.apply_noise(clean)
        
    def absolute_move(self, x_target, y_target, velocity=None):
        self.target_x = float(x_target)
        self.target_y = float(y_target)
        if velocity is not None:
            self.move_velocity = float(velocity)
        self.move_active = True

    def relative_move(self, dx, dy, velocity=None):
        new_x = np.clip(self.x + float(dx), -self.r_outer, self.r_outer)
        new_y = np.clip(self.y + float(dy), -self.r_outer, self.r_outer)
        self.target_x = new_x
        self.target_y = new_y
        if velocity is not None:
            self.move_velocity = float(velocity)
        self.move_active = True
    
    def increase(self, steps):
        idx = np.searchsorted(self.zoom_levels, self.FOV)
        idx_new = idx - int(steps)
        idx_new = max(0, min(len(self.zoom_levels) - 1, idx_new))
        self.FOV = self.zoom_levels[idx_new]

    def stageInfo(self):
        return {
            "FOV": float(self.FOV),
            "x": float(self.x),
            "y": float(self.y)
        }
    
    def update_motion(self):
        now = time.perf_counter()
        dt = now - self._last_motion_t
        self._last_motion_t = now
        dt = float(np.clip(dt, 0.0, 0.1))

        if self.drift_active:
            self.x = np.clip(self.x + self.drift_vx * dt, -self.r_outer, self.r_outer)
            self.y = np.clip(self.y + self.drift_vy * dt, -self.r_outer, self.r_outer)

        if self.move_active:
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = np.hypot(dx, dy)

            if dist < 1e-6:
                self.x = self.target_x
                self.y = self.target_y
                self.move_active = False
            else:
                step = self.move_velocity * dt
                if step >= dist:
                    self.x = self.target_x
                    self.y = self.target_y
                    self.move_active = False
                else:
                    self.x += step * dx / dist
                    self.y += step * dy / dist

                self.x = np.clip(self.x, -self.r_outer, self.r_outer)
                self.y = np.clip(self.y, -self.r_outer, self.r_outer)

    def drift(self, velocity, angle=None):
        velocity = float(velocity)
        if angle is None:
            angle = float(self.rng.uniform(0, 2*np.pi))
        else:
            angle = float(angle)
        self.drift_vx = velocity * np.cos(angle)
        self.drift_vy = velocity * np.sin(angle)
        self.drift_active = True

    def stop_drift(self):
        self.drift_active = False
        self.drift_vx = 0.0
        self.drift_vy = 0.0

    def is_moving(self):
        return bool(self.move_active)

# Tkinter GUI
microscope = Microscopy()

class MicroscopyUI:
    def __init__(self, root, microscope):
        self.root = root
        self.microscope = microscope
        self.root.title("Lance's Microscopy")
        self._running = True

        # Load configuration
        self.config = load_config()
        
        # Mode: "TEM" or "STEM"
        self.mode = tk.StringVar(value=self.config["mode"])
        
        # FPS control
        self.target_fps = tk.DoubleVar(value=self.config["fps"])
        
        # Noise controls
        self.poisson_k = tk.DoubleVar(value=self.config["poisson_k"])
        self.gaussian_std = tk.DoubleVar(value=self.config["gaussian_std"])
        self.microscope.poisson_k = self.config["poisson_k"]
        self.microscope.gaussian_std = self.config["gaussian_std"]
        
        # Percentile normalization controls
        self.percentile_low = tk.DoubleVar(value=self.config["percentile_low"])
        self.percentile_high = tk.DoubleVar(value=self.config["percentile_high"])
        
        # STEM scanning state
        M = self.microscope.M
        self.stem_current_line = M - 1
        self.stem_current_frame = np.zeros((M, M), dtype=np.float32)
        self.stem_last_complete_frame = None
        self.stem_last_complete_frame_clean = None  # Clean version for normalization
        self.stem_base_image = None
        self.stem_scan_start_time = None

        # Figure/Canvas
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, rowspan=4, sticky="nsew")

        # First frame
        first_img_clean = self.microscope.render_clean()
        first_img = self.microscope.apply_noise(first_img_clean)
        first_img_display = self._apply_normalization(first_img, first_img_clean)
        
        self.im = self.ax.imshow(first_img_display, cmap='gray', origin='lower', vmin=0, vmax=1, animated=False)
        self.text_x = self.ax.text(20, 325, "", fontsize=10, color='black')
        self.text_y = self.ax.text(20, 340, "", fontsize=10, color='black')
        self.text_fov = self.ax.text(20, 355, "", fontsize=10, color='black')
        
        self.canvas.draw()

        # Repeaters
        self.repeaters = {
            "left":  Repeater(root, 60, self.microscope.x_minus),
            "right": Repeater(root, 60, self.microscope.x_plus),
            "up":    Repeater(root, 60, self.microscope.y_plus),
            "down":  Repeater(root, 60, self.microscope.y_minus),
        }

        # Control panel frame (right side)
        control_frame = ttk.LabelFrame(root, text="Controls")
        control_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=5, pady=5)

        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(mode_frame, text="Mode:").pack(side="left")
        ttk.Radiobutton(mode_frame, text="TEM", variable=self.mode, value="TEM", 
                       command=self._on_mode_change).pack(side="left")
        ttk.Radiobutton(mode_frame, text="STEM", variable=self.mode, value="STEM",
                       command=self._on_mode_change).pack(side="left")

        # FPS slider
        fps_frame = ttk.Frame(control_frame)
        fps_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(fps_frame, text="FPS Limit:").pack(anchor="w")
        fps_inner = ttk.Frame(fps_frame)
        fps_inner.pack(fill="x")
        self.fps_slider = ttk.Scale(fps_inner, from_=0.5, to=30, orient="horizontal",
                                    variable=self.target_fps, command=self._on_fps_change)
        self.fps_slider.pack(side="left", fill="x", expand=True)
        self.fps_label = ttk.Label(fps_inner, text=f"{self.target_fps.get():.1f}", width=5)
        self.fps_label.pack(side="left", padx=5)

        # Separator
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", padx=5, pady=10)
        
        # Noise section
        ttk.Label(control_frame, text="Noise Settings", font=("", 9, "bold")).pack(anchor="w", padx=5)

        # Poisson k slider (log scale: 10^0 to 10^3)
        poisson_frame = ttk.Frame(control_frame)
        poisson_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(poisson_frame, text="Poisson k (signal):").pack(anchor="w")
        poisson_inner = ttk.Frame(poisson_frame)
        poisson_inner.pack(fill="x")
        self.poisson_slider = ttk.Scale(poisson_inner, from_=0, to=4, orient="horizontal",
                                        command=self._on_poisson_change)
        self.poisson_slider.set(np.log10(max(self.poisson_k.get(), 1)))
        self.poisson_slider.pack(side="left", fill="x", expand=True)
        self.poisson_label = ttk.Label(poisson_inner, text=f"{self.poisson_k.get():.0f}", width=6)
        self.poisson_label.pack(side="left", padx=5)

        # Gaussian std slider
        gaussian_frame = ttk.Frame(control_frame)
        gaussian_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(gaussian_frame, text="Gaussian σ:").pack(anchor="w")
        gaussian_inner = ttk.Frame(gaussian_frame)
        gaussian_inner.pack(fill="x")
        self.gaussian_slider = ttk.Scale(gaussian_inner, from_=0, to=0.2, orient="horizontal",
                                         variable=self.gaussian_std, command=self._on_gaussian_change)
        self.gaussian_slider.pack(side="left", fill="x", expand=True)
        self.gaussian_label = ttk.Label(gaussian_inner, text=f"{self.gaussian_std.get():.3f}", width=6)
        self.gaussian_label.pack(side="left", padx=5)

        # Separator
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", padx=5, pady=10)
        
        # Percentile normalization section
        ttk.Label(control_frame, text="Display Normalization", font=("", 9, "bold")).pack(anchor="w", padx=5)

        # Low percentile slider
        plow_frame = ttk.Frame(control_frame)
        plow_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(plow_frame, text="Low %:").pack(anchor="w")
        plow_inner = ttk.Frame(plow_frame)
        plow_inner.pack(fill="x")
        self.plow_slider = ttk.Scale(plow_inner, from_=0.01, to=10.0, orient="horizontal",
                                     variable=self.percentile_low, command=self._on_percentile_change)
        self.plow_slider.pack(side="left", fill="x", expand=True)
        self.plow_label = ttk.Label(plow_inner, text=f"{self.percentile_low.get():.2f}", width=6)
        self.plow_label.pack(side="left", padx=5)

        # High percentile slider
        phigh_frame = ttk.Frame(control_frame)
        phigh_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(phigh_frame, text="High %:").pack(anchor="w")
        phigh_inner = ttk.Frame(phigh_frame)
        phigh_inner.pack(fill="x")
        self.phigh_slider = ttk.Scale(phigh_inner, from_=90.0, to=99.99, orient="horizontal",
                                      variable=self.percentile_high, command=self._on_percentile_change)
        self.phigh_slider.pack(side="left", fill="x", expand=True)
        self.phigh_label = ttk.Label(phigh_inner, text=f"{self.percentile_high.get():.2f}", width=6)
        self.phigh_label.pack(side="left", padx=5)

        # Separator
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", padx=5, pady=10)

        # Navigation buttons
        ttk.Label(control_frame, text="Navigation", font=("", 9, "bold")).pack(anchor="w", padx=5)
        
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill="x", padx=5, pady=5)
        
        btn_grid = ttk.Frame(nav_frame)
        btn_grid.pack()
        
        self._make_hold_button(btn_grid, text="↑", row=0, col=1, key="up")
        self._make_hold_button(btn_grid, text="←", row=1, col=0, key="left")
        self._make_hold_button(btn_grid, text="→", row=1, col=2, key="right")
        self._make_hold_button(btn_grid, text="↓", row=2, col=1, key="down")

        # Zoom buttons
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(fill="x", padx=5, pady=10)
        ttk.Button(zoom_frame, text="Zoom In", command=self.microscope.z_in_dis).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(zoom_frame, text="Zoom Out", command=self.microscope.z_out_dis).pack(side="left", expand=True, fill="x", padx=2)

        # Keyboard bindings
        self._bind_hold('<KeyPress-Left>',  '<KeyRelease-Left>',  "left")
        self._bind_hold('<KeyPress-Right>', '<KeyRelease-Right>', "right")
        self._bind_hold('<KeyPress-Up>',    '<KeyRelease-Up>',    "up")
        self._bind_hold('<KeyPress-Down>',  '<KeyRelease-Down>',  "down")

        self.canvas_widget.focus_set()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize STEM with a complete first frame
        self._init_stem_first_frame()

        self._last_t = time.perf_counter()
        self._tick()

    def _init_stem_first_frame(self):
        """Initialize STEM with a complete first frame."""
        M = self.microscope.M
        clean_frame = self.microscope.render_clean()
        noisy_frame = self.microscope.apply_noise(clean_frame)
        self.stem_last_complete_frame = noisy_frame.copy()
        self.stem_last_complete_frame_clean = clean_frame.copy()
        self.stem_current_frame = np.zeros((M, M), dtype=np.float32)
        self.stem_current_line = M - 1
        self.stem_base_image = None

    def _apply_normalization(self, img_noisy, img_clean):
        """
        Apply percentile normalization based on CLEAN image.
        This prevents noise from affecting the normalization bounds.
        """
        plow = self.percentile_low.get()
        phigh = self.percentile_high.get()
        
        # Compute percentiles from CLEAN image
        vmin = np.percentile(img_clean, plow)
        vmax = np.percentile(img_clean, phigh)
        
        if vmax - vmin < 1e-10:
            vmax = vmin + 1e-10
        
        # Apply to NOISY image
        img_norm = (img_noisy - vmin) / (vmax - vmin)
        img_norm = np.clip(img_norm, 0.0, 1.0)
        
        return img_norm.astype(np.float32)

    # ---------- Config callbacks ----------
    def _on_mode_change(self):
        self.config["mode"] = self.mode.get()
        save_config(self.config)
        
        if self.mode.get() == "STEM":
            self._init_stem_first_frame()
            self.stem_scan_start_time = None
    
    def _on_fps_change(self, value=None):
        fps = self.target_fps.get()
        self.fps_label.config(text=f"{fps:.1f}")
        self.config["fps"] = fps
        save_config(self.config)
    
    def _on_poisson_change(self, value=None):
        log_val = float(self.poisson_slider.get())
        k = 10 ** log_val
        self.poisson_k.set(k)
        self.microscope.poisson_k = k
        self.poisson_label.config(text=f"{k:.0f}")
        self.config["poisson_k"] = k
        save_config(self.config)
    
    def _on_gaussian_change(self, value=None):
        std = self.gaussian_std.get()
        self.microscope.gaussian_std = std
        self.gaussian_label.config(text=f"{std:.3f}")
        self.config["gaussian_std"] = std
        save_config(self.config)
    
    def _on_percentile_change(self, value=None):
        plow = self.percentile_low.get()
        phigh = self.percentile_high.get()
        self.plow_label.config(text=f"{plow:.2f}")
        self.phigh_label.config(text=f"{phigh:.2f}")
        self.config["percentile_low"] = plow
        self.config["percentile_high"] = phigh
        save_config(self.config)

    # ---------- UI helpers ----------
    def _make_hold_button(self, parent, text, row, col, key):
        btn = ttk.Button(parent, text=text, width=3)
        btn.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
        btn.bind('<ButtonPress-1>', lambda e, k=key: self.repeaters[k].start())
        btn.bind('<ButtonRelease-1>', lambda e, k=key: self.repeaters[k].stop())
        btn.bind('<Leave>', lambda e, k=key: self.repeaters[k].stop())

    def _bind_hold(self, press_event, release_event, key):
        self.root.bind(press_event, lambda e, k=key: self.repeaters[k].start())
        self.root.bind(release_event, lambda e, k=key: self.repeaters[k].stop())

    # ---------- render loop ----------
    def _tick(self):
        if not self._running:
            return
        
        self.microscope.update_motion()
        
        if self.mode.get() == "TEM":
            img_clean = self.microscope.render_clean()
            img_noisy = self.microscope.apply_noise(img_clean)
            img = self._apply_normalization(img_noisy, img_clean)
        else:
            img = self._render_stem()
            img = 1.0 - img  # Invert for STEM
        
        self.im.set_data(img)

        t = time.perf_counter()
        dt = max(t - self._last_t, 1e-6)
        fps = 1.0 / dt
        self._last_t = t

        x_m = self.microscope.x / 1e9
        y_m = self.microscope.y / 1e9
        mag = 1000 * 1e6 / self.microscope.FOV

        mode_str = self.mode.get()
        self.text_x.set_text(f" x  = {x_m:.2e} m")
        self.text_y.set_text(f" y  = {y_m:.2e} m")
        self.text_fov.set_text(f"{mode_str} | mag = {format_mag(mag)} | {fps:.1f} FPS")
        
        self.canvas.draw()
        self.root.after(1, self._tick)

    def _render_stem(self):
        """
        Render STEM image with time-based scanning.
        Normalization is based on clean image to preserve proper noise characteristics.
        """
        M = self.microscope.M
        target_fps = max(self.target_fps.get(), 0.5)
        scan_duration = 1.0 / target_fps
        
        now = time.perf_counter()
        
        # Check if we need to start a new scan
        need_new_scan = (self.stem_scan_start_time is None or 
                         self.stem_current_line < 0)
        
        if need_new_scan:
            # Render the clean frame at CURRENT position
            self.stem_base_image = self.microscope.render_clean()
            self.stem_current_frame = np.zeros((M, M), dtype=np.float32)
            self.stem_current_frame_clean = np.zeros((M, M), dtype=np.float32)
            self.stem_current_line = M - 1
            self.stem_scan_start_time = now
        
        # Calculate progress based on elapsed time
        elapsed = now - self.stem_scan_start_time
        progress = elapsed / scan_duration
        
        # Target line based on progress
        target_line = M - 1 - int(progress * M)
        target_line = max(-1, target_line)
        
        # Scan lines from current position down to target
        while self.stem_current_line > target_line and self.stem_current_line >= 0:
            # Get clean line from the pre-rendered base image
            line_clean = self.stem_base_image[self.stem_current_line, :].copy()
            
            # Apply noise to this line
            line_noisy = self.microscope.apply_noise_to_line(line_clean)
            
            self.stem_current_frame[self.stem_current_line, :] = line_noisy
            self.stem_current_frame_clean[self.stem_current_line, :] = line_clean
            self.stem_current_line -= 1
        
        # Build display frame: scanned portion + last complete frame for unscanned
        if self.stem_last_complete_frame is not None:
            display_frame = self.stem_last_complete_frame.copy()
            display_frame_clean = self.stem_last_complete_frame_clean.copy()
        else:
            display_frame = np.zeros((M, M), dtype=np.float32)
            display_frame_clean = np.zeros((M, M), dtype=np.float32)
        
        # Overwrite with scanned portion
        scanned_bottom = self.stem_current_line + 1
        if scanned_bottom < M:
            display_frame[scanned_bottom:M, :] = self.stem_current_frame[scanned_bottom:M, :]
            display_frame_clean[scanned_bottom:M, :] = self.stem_current_frame_clean[scanned_bottom:M, :]
        
        # When scan completes, save as last complete frame
        if self.stem_current_line < 0:
            self.stem_last_complete_frame = self.stem_current_frame.copy()
            self.stem_last_complete_frame_clean = self.stem_current_frame_clean.copy()
        
        # Apply normalization based on CLEAN image
        normalized = self._apply_normalization(display_frame, display_frame_clean)
        
        return normalized

    def on_close(self):
        self._running = False
        save_config(self.config)
        for r in self.repeaters.values():
            r.stop()
        self.root.after(0, self.root.destroy)

def format_mag(mag):
    if mag >= 1e6:
        return f"{mag/1e6:.1f}Mx"
    elif mag >= 1e5:
        return f"{mag/1e3:.0f}Kx"
    else:
        return f"{mag:.0f}x"
    
def start_tcp_server(microscope, host="127.0.0.1", port=9999):

    def handle_client(conn):
        with conn:
            data = conn.recv(4096).decode()
    
            try:
                msg = json.loads(data)
    
                cmd = msg.get("cmd")
                args = msg.get("args", {})
    
                if cmd == "absolute_move":
                    microscope.absolute_move(
                        args["x"], args["y"],
                        velocity=args.get("velocity")
                    )
                    response = {"status": "OK", "values": None}
                elif cmd == "relative_move":
                    microscope.relative_move(
                        args["dx"], args["dy"],
                        velocity=args.get("velocity")
                    )
                    response = {"status": "OK", "values": None}
                elif cmd == "drift":
                    microscope.drift(
                        args["velocity"],
                        angle=args.get("angle")
                    )
                    response = {"status": "OK", "values": None}
                elif cmd == "stop_drift":
                    microscope.stop_drift()
                    response = {"status": "OK", "values": None}
                elif cmd == "increase":
                    microscope.increase(args["steps"])
                    response = {"status": "OK", "values": None}
                elif cmd == "stageInfo":
                    info = microscope.stageInfo()
                    response = {"status": "OK", "values": info}
                elif cmd == "is_moving":
                    response = {"status": "OK", "values": {"moving": microscope.is_moving()}}
                else:
                    response = {"status": "ERROR", "values": None}
    
            except Exception as e:
                response = {"status": "ERROR", "values": str(e)}
    
            conn.sendall(json.dumps(response).encode())

    def server_loop():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            print(f"TCP server listening on {host}:{port}")

            while True:
                conn, _ = s.accept()
                handle_client(conn)

    threading.Thread(target=server_loop, daemon=True).start()
    
# Run UI
root = tk.Tk()
root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=1)
for r in range(4):
    root.grid_rowconfigure(r, weight=1)
app = MicroscopyUI(root, microscope)
start_tcp_server(microscope)
root.mainloop()
