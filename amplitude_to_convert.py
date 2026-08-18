"""
Amplitude-Variable Trial — CoP Controlled (two force plates)
Fullscreen, CSV logging.

What this trial does:
  The path is a sine wave at a FIXED frequency whose AMPLITUDE varies from
  cycle to cycle following a fractional Gaussian noise (fGn) sequence with a
  specified Hurst exponent H (generated via the Davies-Harte method, ported
  directly from fgn_sim.m). The participant keeps a ball on the path by
  shifting weight between two force plates (combined, position-based CoP).

  - Fixed frequency (person-specific; set FIXED_FREQUENCY_HZ below)
  - Amplitude changes once per cycle, drawn from fgn_sim(H, sd)
  - 5 second calibration before the trial begins
  - Trial starts only after calibration; ball is a plain circle
"""

import pygame
import sys
import math
import asyncio
import time
import threading
import csv
from datetime import datetime

import numpy as np
import qtm
import matplotlib
matplotlib.use("Agg")   # save-only backend - doesn't need a display, safe to
                        # import alongside pygame's own display window
import matplotlib.pyplot as plt


# =====================================================================
#  TRIAL SETTINGS
# =====================================================================
TRIAL_DURATION_SEC = 300        # 5 minutes

# Fixed frequency for this trial. THIS IS PERSON-SPECIFIC - set based on
# that participant's preferred frequency (e.g. from the free-sway task).
FIXED_FREQUENCY_HZ = 0.3


HURST = 0.99           # <-- change per trial condition
RNG_SEED = None    # each trial gets a fully independent random sequence.
                    # A shared fixed seed across different H trials was
                    # found to make different H values produce visually
                    # near-identical amplitude patterns - the shared
                    # underlying random noise dominated the visible shape
                    # more than H's own effect on it. With seed=None,
                    # each run's H-driven fractal structure (smooth,
                    # persistent excursions for high H vs. rough, rapidly
                    # changing ones for low H) is clearly visible instead.


#  AMPLITUDE DISTRIBUTION - derived directly from this person's
#  free-sway task results (same pooled-cycles logic as the frequency
#  trial's FREQ_MEAN_HZ/FREQ_SD_HZ, but measuring CoP peak excursion
#  per cycle instead of cycle duration), fed straight into fgn_sim's
#  own mu/sd (NOT via normalize_01 into an arbitrary [min,max] range,
#  unlike an earlier version of this trial) - so the per-cycle
#  amplitude sequence is centered on this person's real mean sway with
#  a spread matching their real measured cycle-to-cycle variability.
AMP_MEAN_MM = 142.0    # from free-sway summary: mean lr_magnitude_mm
AMP_SD_MM = 6.51       # from free-sway summary: sd of lr_magnitude_mm

# Both get converted to px at runtime using the same mm-to-px ratio as
# COP_RANGE_MM, so the pathway and the red ball's CoP-driven range
# share one consistent physical scale. See AMP_MIN_PX/AMP_MAX_PX in
# main() for the actual runtime conversion.


# =====================================================================
#  FORCE PLATE SETTINGS (two plates, combined position-based CoP)
# =====================================================================
QTM_IP = "127.0.0.1"
QTM_PLATE_IDS = [3, 5]   # 3 = left foot plate, 5 = right foot plate

FZ_THRESHOLD_N = 20.0
COP_SMOOTH_ALPHA = 0.3
CALIBRATION_SECONDS = 5.0
EDGE_MARGIN_PX = 20

PLATE_WIDTH_MM = 600.0
PLATE_OFFSET_MM = {
    3: -PLATE_WIDTH_MM / 2,
    5: +PLATE_WIDTH_MM / 2,
}
# amplitude of ball's CoP-driven motion, in mm of real excursion mapped to
# the same AMP_MIN_PX..AMP_MAX_PX range as the path itself
COP_RANGE_MM = 200.0

# AXIS NOTE (confirmed July 2026 in QTM Project): raw y_a = true
# mediolateral axis, raw x_a = true anteroposterior axis.


# =====================================================================
#  DISPLAY (base values at reference 600x700, scaled at runtime)
# =====================================================================
FPS = 60
BG_COLOR = (15, 15, 30)
REF_W, REF_H = 600, 700
BASE_BALL_RADIUS = 40            # reduced from 34 - smaller ball
BASE_CENTER_DOT_RADIUS = 10       # small black center dot for precise alignment
BASE_AUTO_SPEED = 0.7           # vertical scroll speed in px per frame
BASE_GRID_SPACING = 60
BASE_FONT_SIZE = 14

BALL_Y_FRACTION = 0.72    # ball resting position, fraction of screen height from top

BASE_LINE_WIDTH = 9
LINE_COLOR = (180, 150, 40)     # dark yellow path
BALL_COLOR = (190, 190, 190)    # soft/light gray ball
CENTER_DOT_COLOR = (0, 0, 0)    # black center dot for precise alignment
GRID_COLOR = (22, 22, 40)


# =====================================================================
#  fGn GENERATION - Davies-Harte method, ported directly from fgn_sim.m
# =====================================================================
def fgn_sim(n, H, sd=1.0, mu=0.0, seed=None):
    """
    Python port of fgn_sim.m (Davies-Harte method). Generates a fractional
    Gaussian noise series of length n with Hurst exponent H.

    Reference: Beran, J. (1994). Statistics for long-memory processes.
    Original MATLAB: Nonlinear Analysis Core, Center for Human Movement
    Variability, University of Nebraska at Omaha.
    """
    rng = np.random.default_rng(seed)

    z = rng.standard_normal(2 * n)
    zr = z[0:n].copy()
    zi = z[n:2*n].copy()
    zic = -zi
    zi[0] = 0.0
    zr[0] = zr[0] * np.sqrt(2)
    zi[n-1] = 0.0
    zr[n-1] = zr[n-1] * np.sqrt(2)

    zr_full = np.concatenate([zr[0:n], zr[1:n-1][::-1]])
    zi_full = np.concatenate([zi[0:n], zic[1:n-1][::-1]])
    z_complex = zr_full + 1j * zi_full

    k = np.arange(0, n)
    gammak = ((np.abs(k - 1) ** (2*H)) - (2 * np.abs(k) ** (2*H)) + (np.abs(k + 1) ** (2*H))) / 2.0

    ind = np.concatenate([
        np.arange(0, n-1),
        np.array([n-1]),
        np.arange(1, n-1)[::-1]
    ])

    gkFGN0 = np.fft.ifft(gammak[ind]) * len(z_complex)
    gksqrt = np.real(gkFGN0)

    if np.all(gksqrt > 0):
        gksqrt = np.sqrt(gksqrt)
        z_complex = z_complex * gksqrt
        z_complex = np.fft.ifft(z_complex) * len(z_complex)
        z_complex = 0.5 * (n - 1) ** (-0.5) * z_complex
        z_complex = np.real(z_complex[0:n])
    else:
        raise ValueError(f"Re(gk)-vector not positive for n={n}, H={H} - try different parameters")

    return sd * z_complex + mu


def save_pathway_debug_plot(n_cycles, hurst, amp_mean_px, amp_sd_px, auto_speed, fps,
                             frequency, seed, out_path, points_per_cycle=60):
    """Generates and saves a diagnostic plot showing this trial's
    ACTUAL sine wave - amplitude driven by fgn_sim(H, sd=amp_sd_px,
    mu=amp_mean_px), phase advancing at the trial's FIXED frequency
    (unlike the frequency-variable trial, frequency itself doesn't
    change here, only amplitude does) - across ALL n_cycles, using
    THIS trial's real H, mean, sd, and seed. This directly shows
    whether H's fractal pattern (smooth, persistent stretches for
    high H vs. rough, rapidly-changing ones for low H) is visible in
    the amplitude-driven pathway itself, not just in the per-cycle
    amplitude numbers.

    This is a diagnostic aid, not part of the trial logic itself - it
    does not affect amp_values or the ball's actual motion."""
    amp_values_px = fgn_sim(n_cycles, hurst, sd=amp_sd_px, mu=amp_mean_px, seed=seed)
    amp_values_px = np.clip(amp_values_px, 5.0, None)

    cycle_length_px = (2 * math.pi) / frequency
    total_world_y = cycle_length_px * n_cycles
    px_per_sec = auto_speed * fps

    world_ys = np.linspace(0, total_world_y, n_cycles * points_per_cycle)
    times_sec = world_ys / px_per_sec

    def amp_at(world_y):
        pos = world_y / cycle_length_px
        idx = int(pos)
        frac = pos - idx
        i0 = max(0, min(idx, len(amp_values_px) - 1))
        i1 = max(0, min(idx + 1, len(amp_values_px) - 1))
        return amp_values_px[i0] * (1 - frac) + amp_values_px[i1] * frac

    y = np.array([amp_at(wy) * math.sin(frequency * wy) for wy in world_ys])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

    # top: the actual sine wave shape across the ENTIRE trial's cycles
    ax1.plot(times_sec, y, linewidth=0.6, color="tab:blue")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.set_title(f"H={hurst}: actual pathway sine wave, all {n_cycles} cycles "
                  f"(fixed frequency, H-driven amplitude)")
    ax1.set_ylabel("y (px)")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True, alpha=0.3)

    # bottom: the per-cycle amplitude envelope, for direct numeric reference
    ax2.plot(amp_values_px, color="tab:blue", linewidth=1.0, marker="o", markersize=1.5)
    ax2.axhline(amp_mean_px, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="mean")
    ax2.axhline(amp_mean_px + amp_sd_px, color="green", linestyle="--", linewidth=0.8, alpha=0.5, label="mean +/- sd")
    ax2.axhline(amp_mean_px - amp_sd_px, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_title(f"Full amplitude envelope for this trial ({n_cycles} cycles)")
    ax2.set_ylabel("Amplitude (px)")
    ax2.set_xlabel("Cycle number")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Pathway check - H={hurst}, mean={amp_mean_px:.1f}px, sd={amp_sd_px:.1f}px, seed={seed}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def normalize_01(values):
    mn, mx = np.min(values), np.max(values)
    r = mx - mn
    if r < 1e-15:
        return np.full_like(values, 0.5)
    return (values - mn) / r


def draw_smooth_line(surface, color, points, width):
    """Thick, anti-aliased polyline with round joins (no gaps at turns)."""
    if len(points) < 2:
        return
    half = max(1, width // 2)
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        nx, ny = -dy / length, dx / length
        ox, oy = nx * half, ny * half
        quad = [(x1 + ox, y1 + oy), (x2 + ox, y2 + oy),
                (x2 - ox, y2 - oy), (x1 - ox, y1 - oy)]
        pygame.draw.polygon(surface, color, quad)
    for (x, y) in points:
        pygame.draw.circle(surface, color, (int(x), int(y)), half)
    pygame.draw.aalines(surface, color, False, points)


class CycleAmplitudeMap:
    """Maps a vertical world position to an amplitude, interpolated between cycles."""
    def __init__(self, frequency, amp_values, amp_min, amp_max):
        self.frequency = frequency
        self.cycle_length = (2 * math.pi) / frequency
        self.amp_values = amp_values
        self.amp_min = amp_min
        self.amp_max = amp_max

    def get_amplitude(self, world_y):
        pos = world_y / self.cycle_length
        idx = int(pos)
        frac = pos - idx
        i0 = max(0, min(idx, len(self.amp_values) - 1))
        i1 = max(0, min(idx + 1, len(self.amp_values) - 1))
        v = self.amp_values[i0] * (1 - frac) + self.amp_values[i1] * frac
        return self.amp_min + v * (self.amp_max - self.amp_min)


# =====================================================================
#  TWO-PLATE COP INPUT (combined_cop_shared, position-based)
# =====================================================================
class QTMTwoPlateCopInput:
    """Streams CoP from two force plates and computes combined_cop_shared:
    a force-weighted average of each foot's ML position, converted to
    shared coordinates (local + plate physical offset)."""

    def __init__(self, qtm_ip, plate_ids, fz_threshold_n, smooth_alpha, plate_offset_mm):
        self.qtm_ip = qtm_ip
        self.plate_ids = list(plate_ids)
        self.fz_threshold_n = float(fz_threshold_n)
        self.smooth_alpha = float(smooth_alpha)
        self.plate_offset_mm = plate_offset_mm

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self._plate_raw = {pid: {"fz": 0.0, "ml": None, "ap": None} for pid in self.plate_ids}
        self._cop_combined_s = None

        self._whole_body_center_shared_mm = 0.0

        self.current_x_px = 0.0

    def start(self, width):
        self.current_x_px = width / 2
        self._running = True
        self._thread = threading.Thread(target=self._thread_entry, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_raw_snapshot(self):
        with self._lock:
            return {pid: dict(self._plate_raw[pid]) for pid in self.plate_ids}

    @property
    def loaded(self):
        """True if at least one plate currently exceeds the Fz threshold."""
        with self._lock:
            return any(abs(self._plate_raw[pid]["fz"]) >= self.fz_threshold_n for pid in self.plate_ids)

    def _get_combined_cop_shared_raw(self):
        if len(self.plate_ids) != 2:
            return None
        left_pid, right_pid = self.plate_ids[0], self.plate_ids[1]
        with self._lock:
            fz_left = self._plate_raw[left_pid]["fz"]
            ml_left = self._plate_raw[left_pid]["ml"]
            fz_right = self._plate_raw[right_pid]["fz"]
            ml_right = self._plate_raw[right_pid]["ml"]

        loaded_left = ml_left is not None and abs(fz_left) > self.fz_threshold_n
        loaded_right = ml_right is not None and abs(fz_right) > self.fz_threshold_n
        if not loaded_left and not loaded_right:
            return None

        fz_left_eff = fz_left if loaded_left else 0.0
        fz_right_eff = fz_right if loaded_right else 0.0
        total = fz_left_eff + fz_right_eff
        if total <= 0:
            return None

        ml_left_shared = (ml_left if ml_left is not None else 0.0) + self.plate_offset_mm[left_pid]
        ml_right_shared = (ml_right if ml_right is not None else 0.0) + self.plate_offset_mm[right_pid]

        return (fz_left_eff * ml_left_shared + fz_right_eff * ml_right_shared) / total

    def calibrate_center(self, seconds=5.0):
        """Stand naturally while this runs. Computes the whole-body resting
        CoP position, used as the zero-reference for the ball."""
        t_end = time.time() + float(seconds)
        combined_samples = []
        while time.time() < t_end:
            combined = self._get_combined_cop_shared_raw()
            if combined is not None:
                combined_samples.append(combined)
            time.sleep(0.01)

        if combined_samples:
            self._whole_body_center_shared_mm = float(np.mean(combined_samples))
        else:
            self._whole_body_center_shared_mm = 0.0
            print("  Warning: no combined CoP samples during calibration")
        print(f"  Whole-body resting CoP (shared coords): {self._whole_body_center_shared_mm:.1f}mm")

    def get_control_x_px(self, width, ball_radius, cop_range_mm, edge_margin_px):
        """Maps the smoothed, calibration-referenced CoP position to a
        screen x position, same +/- cop_range_mm -> full width mapping
        style used in the single-plate version, but from combined_cop_shared."""
        raw_cop = self._get_combined_cop_shared_raw()

        if raw_cop is not None:
            centered_raw = raw_cop - self._whole_body_center_shared_mm
            if self._cop_combined_s is None:
                self._cop_combined_s = centered_raw
            else:
                a = self.smooth_alpha
                self._cop_combined_s = a * centered_raw + (1.0 - a) * self._cop_combined_s

        if self._cop_combined_s is None:
            return self.current_x_px

        copv = max(-cop_range_mm, min(cop_range_mm, self._cop_combined_s))
        norm = (copv + cop_range_mm) / (2.0 * cop_range_mm)
        x_px = norm * width

        min_x = max(edge_margin_px, ball_radius)
        max_x = min(width - edge_margin_px, width - ball_radius)
        x_px = max(min_x, min(max_x, x_px))
        self.current_x_px = x_px
        return self.current_x_px

    def snapshot(self):
        raw = self.get_raw_snapshot()
        combined = self._get_combined_cop_shared_raw()
        centered = (combined - self._whole_body_center_shared_mm) if combined is not None else None
        return {"raw": raw, "combined_centered_mm": centered}

    def _thread_entry(self):
        asyncio.run(self._async_main())

    def _extract_ml_ap(self, s):
        if hasattr(s, "z") and hasattr(s, "x_a") and hasattr(s, "y_a"):
            fz = float(s.z)
            ml = float(s.y_a)
            ap = float(s.x_a)
            if not (np.isfinite(ml) and np.isfinite(ap)):
                return fz, None, None
            return fz, ml, ap
        try:
            fz = float(s[2])
            ml = float(s[7]) if len(s) >= 8 else np.nan
            ap = float(s[6]) if len(s) >= 7 else np.nan
            if not (np.isfinite(ml) and np.isfinite(ap)):
                return fz, None, None
            return fz, ml, ap
        except Exception:
            return 0.0, None, None

    async def _async_main(self):
        conn = await qtm.connect(self.qtm_ip)
        if conn is None:
            print("Could not connect to QTM.")
            return

        print(f"Connected to QTM at {self.qtm_ip} (Plates {self.plate_ids})")

        def on_packet(packet):
            try:
                header, plates = packet.get_force()
            except Exception:
                return

            for meta, samples_list in plates:
                pid = meta.id
                if pid not in self.plate_ids or not samples_list:
                    continue
                fz, ml, ap = self._extract_ml_ap(samples_list[-1])
                if ml is None:
                    continue
                with self._lock:
                    self._plate_raw[pid]["fz"] = fz
                    self._plate_raw[pid]["ml"] = ml
                    self._plate_raw[pid]["ap"] = ap

        stream_task = asyncio.create_task(
            conn.stream_frames(components=["force"], on_packet=on_packet))
        try:
            while self._running:
                await asyncio.sleep(0.01)
        finally:
            stream_task.cancel()
            try:
                res = conn.disconnect()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass
            print("QTM disconnected")


# =====================================================================
#  MAIN
# =====================================================================
def main():
    pygame.init()

    display_info = pygame.display.Info()
    WIDTH = display_info.current_w
    HEIGHT = display_info.current_h

    # FLICKERING FIX: DOUBLEBUF + vsync=1, with a fallback for systems/
    # drivers that don't support vsync=1.
    try:
        screen = pygame.display.set_mode(
            (WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.DOUBLEBUF, vsync=1
        )
    except pygame.error:
        print("vsync=1 not supported on this system, falling back without it.")
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.DOUBLEBUF)

    pygame.display.set_caption("Amplitude-Variable Trial - CoP logging")

    scale = HEIGHT / REF_H
    BALL_RADIUS    = max(10, int(BASE_BALL_RADIUS * scale))
    CENTER_DOT_RADIUS = max(2, int(BASE_CENTER_DOT_RADIUS * scale))

    # Convert the person-specific amplitude distribution (mm, from the
    # free-sway task) to px, using the SAME mm-to-px ratio as
    # COP_RANGE_MM (get_control_x_px uses this same ratio for the red
    # ball), so the white ball's pathway and the red ball's real CoP
    # range share one consistent physical scale.
    mm_to_px_ratio = WIDTH / (2.0 * COP_RANGE_MM)
    AMP_MEAN_PX    = AMP_MEAN_MM * mm_to_px_ratio
    AMP_SD_PX      = AMP_SD_MM * mm_to_px_ratio

    AUTO_SPEED     = BASE_AUTO_SPEED * scale
    FONT_SIZE      = max(14, int(BASE_FONT_SIZE * scale))
    LINE_WIDTH     = max(3, int(BASE_LINE_WIDTH * scale))
    BALL_Y         = int(HEIGHT * BALL_Y_FRACTION)

    FREQUENCY = FIXED_FREQUENCY_HZ * 2 * math.pi / (AUTO_SPEED * FPS)
    # NOTE: this keeps FREQUENCY consistent with the world_y -> sin(FREQUENCY*world_y)
    # convention below, derived directly from the desired real-world Hz.

    cycle_length = (2 * math.pi) / FREQUENCY
    total_distance = AUTO_SPEED * FPS * TRIAL_DURATION_SEC
    cycles_needed = int(math.ceil(total_distance / cycle_length)) + 5
    NUM_CYCLES = max(100, cycles_needed)

    print(f"Screen: {WIDTH}x{HEIGHT}, Scale: {scale:.2f}")
    print(f"Ball: {BALL_RADIUS}px at y={BALL_Y}px, Amplitude: mean={AMP_MEAN_PX:.1f}px, sd={AMP_SD_PX:.1f}px")
    print(f"Fixed frequency: {FIXED_FREQUENCY_HZ}Hz -> FREQUENCY={FREQUENCY:.5f} (world units)")
    print(f"5 min distance: {total_distance:.0f}px -> generating {NUM_CYCLES} cycles")
    print(f"Amplitude source: fgn_sim (Davies-Harte), H={HURST}, seed={RNG_SEED}")

    # Build the per-cycle amplitude sequence DIRECTLY from fgn_sim's
    # own mu/sd (same approach as the frequency-variable trial's
    # FREQ_MEAN_HZ/FREQ_SD_HZ) - centered on this person's real mean
    # sway (AMP_MEAN_PX) with a spread matching their real measured
    # cycle-to-cycle variability (AMP_SD_PX). Unlike an earlier version
    # of this trial, this is NOT passed through normalize_01, so the
    # realized min/max will vary somewhat per run rather than exactly
    # touching a fixed bound - but mu/sd match the measured values
    # exactly, consistent with how the frequency trial works.
    amp_values_px = fgn_sim(NUM_CYCLES, HURST, sd=AMP_SD_PX, mu=AMP_MEAN_PX, seed=RNG_SEED)

    # Safety floor: fgn_sim can occasionally produce a value at or
    # below zero (a physically impossible/undefined amplitude) if a
    # sample falls far enough below the mean - clip to a small positive
    # floor so the pathway never inverts or collapses to nothing.
    amp_values_px = np.clip(amp_values_px, 5.0, None)
    n_clipped = np.sum(amp_values_px <= 5.0)
    if n_clipped > 0:
        print(f"  NOTE: {n_clipped} cycle(s) had a non-positive raw amplitude and were "
              f"floored to 5px (safety bound, not expected to occur often given mean>>sd).")

    print(f"Amplitude pathway realized range: [{amp_values_px.min():.1f}, {amp_values_px.max():.1f}]px "
          f"(target mean={AMP_MEAN_PX:.1f}px, sd={AMP_SD_PX:.1f}px)")

    # Diagnostic plot: shows the ACTUAL sine wave this trial will draw
    # (amplitude driven by fgn_sim with THIS trial's real HURST value),
    # so you can directly verify H is shaping the pathway before every
    # run. Saved once per run, before the trial starts; does not
    # affect gameplay - uses the exact same RNG_SEED, so it reflects
    # this run's real amp_values.
    debug_plot_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_plot_path = f"pathway_debug_H{HURST}_{debug_plot_ts}.png"
    save_pathway_debug_plot(
        n_cycles=NUM_CYCLES,
        hurst=HURST,
        amp_mean_px=AMP_MEAN_PX,
        amp_sd_px=AMP_SD_PX,
        auto_speed=AUTO_SPEED,
        fps=FPS,
        frequency=FREQUENCY,
        seed=RNG_SEED,
        out_path=debug_plot_path,
    )
    print(f"  Pathway debug plot saved to {debug_plot_path}")

    amp_map = CycleAmplitudeMap(FREQUENCY, amp_values_px, 0.0, 1.0)

    def get_path_center_x(world_y):
        amp = amp_map.get_amplitude(world_y)
        return WIDTH // 2 + amp * math.sin(FREQUENCY * world_y)

    # Start QTM (two plates)
    cop = QTMTwoPlateCopInput(QTM_IP, QTM_PLATE_IDS, FZ_THRESHOLD_N, COP_SMOOTH_ALPHA, PLATE_OFFSET_MM)
    cop.start(WIDTH)

    # CSV setup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"cop_log_amplitude_H{HURST}_{ts}.csv"

    fields = [
        "frame", "t_sec", "trial_state",
        "loaded",
        "fz_left", "ml_left", "ap_left",
        "fz_right", "ml_right", "ap_right",
        "combined_cop_centered_mm",
        "ball_x_px", "path_cx_px", "offset_px",
        "world_y", "cycle_idx", "cycle_amp_px",
        "hurst", "fgn_sd", "fixed_frequency_hz",
    ]

    csvfile = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("timesnewroman", FONT_SIZE)
    font_big = pygame.font.SysFont("timesnewroman", max(20, int(28 * scale)), bold=True)
    font_instructions = pygame.font.SysFont("timesnewroman", max(22, int(30 * scale)))

    world_y = 0.0
    ball_x = WIDTH / 2.0
    frame_count = 0

    trial_state = "instructions"  # instructions -> waiting -> calibrating -> running -> done
    trial_start_time = None
    trial_elapsed = 0.0
    cycle_idx = 0

    instruction_lines = [
        "Stand on the two force plates, feet shoulder-width apart.",
        "Shift your weight left and right to keep the ball on the path.",
        "",
        "Press SPACE when you are ready to begin.",
    ]

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and trial_state == "instructions":
                        trial_state = "waiting"

            if trial_state == "instructions":
                pass   # just wait for SPACE, handled in the event loop above

            elif trial_state == "waiting":
                ball_x = WIDTH / 2.0
                if cop.loaded:
                    trial_state = "calibrating"
                    print(f"Plate loaded! Calibrating for {CALIBRATION_SECONDS}s - stand still...")
                    cop.calibrate_center(CALIBRATION_SECONDS)
                    print("Calibration done. Trial starting!")
                    trial_state = "running"
                    trial_start_time = time.perf_counter()
                    world_y = 0.0

            elif trial_state == "running":
                ball_x = cop.get_control_x_px(WIDTH, BALL_RADIUS, COP_RANGE_MM, EDGE_MARGIN_PX)
                trial_elapsed = time.perf_counter() - trial_start_time
                world_y = AUTO_SPEED * FPS * trial_elapsed
                if trial_elapsed >= TRIAL_DURATION_SEC:
                    trial_state = "done"
                    print(f"Trial complete! {TRIAL_DURATION_SEC}s elapsed.")

            path_cx = get_path_center_x(world_y)
            offset = ball_x - path_cx
            cycle_idx = int(world_y / cycle_length)
            cycle_amp = amp_map.get_amplitude(world_y)

            if trial_state == "running":
                snap = cop.snapshot()
                raw = snap["raw"]
                left_pid, right_pid = QTM_PLATE_IDS[0], QTM_PLATE_IDS[1]

                writer.writerow({
                    "frame": frame_count,
                    "t_sec": f"{trial_elapsed:.6f}",
                    "trial_state": trial_state,
                    "loaded": int(cop.loaded),
                    "fz_left": f"{raw[left_pid]['fz']:.4f}",
                    "ml_left": "" if raw[left_pid]['ml'] is None else f"{raw[left_pid]['ml']:.4f}",
                    "ap_left": "" if raw[left_pid]['ap'] is None else f"{raw[left_pid]['ap']:.4f}",
                    "fz_right": f"{raw[right_pid]['fz']:.4f}",
                    "ml_right": "" if raw[right_pid]['ml'] is None else f"{raw[right_pid]['ml']:.4f}",
                    "ap_right": "" if raw[right_pid]['ap'] is None else f"{raw[right_pid]['ap']:.4f}",
                    "combined_cop_centered_mm": "" if snap["combined_centered_mm"] is None else f"{snap['combined_centered_mm']:.4f}",
                    "ball_x_px": f"{ball_x:.4f}",
                    "path_cx_px": f"{path_cx:.4f}",
                    "offset_px": f"{offset:.4f}",
                    "world_y": f"{world_y:.4f}",
                    "cycle_idx": cycle_idx,
                    "cycle_amp_px": f"{cycle_amp:.4f}",
                    "hurst": HURST,
                    "fgn_sd": FGN_SD,
                    "fixed_frequency_hz": FIXED_FREQUENCY_HZ,
                })
                frame_count += 1

            # --- draw ---
            screen.fill(BG_COLOR)

            if trial_state == "instructions":
                # Instruction screen: no path, no ball - just the protocol text
                y_start = HEIGHT // 2 - int(120 * scale)
                line_spacing = int(44 * scale)
                for i, line in enumerate(instruction_lines):
                    if not line:
                        continue
                    msg = font_instructions.render(line, True, (220, 220, 220))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, y_start + i * line_spacing))

            else:
                # Path line: sample from ABOVE the ball down to below the bottom
                # of the screen, and make sure the path is defined for world_y
                # values corresponding to screen rows from HEIGHT down to 0,
                # so the line is never missing/cut off at the bottom of the
                # screen on the very first frames (world_y starts at 0, and
                # rows below BALL_Y correspond to world_y < 0, which
                # get_path_center_x must handle correctly via the amplitude
                # map's clamped indexing - see CycleAmplitudeMap.get_amplitude).
                step = 2
                pts = []
                for sy in range(-20, HEIGHT + 20, step):
                    wy = world_y + (BALL_Y - sy)
                    cxp = get_path_center_x(wy)
                    pts.append((cxp, sy))
                draw_smooth_line(screen, LINE_COLOR, pts, LINE_WIDTH)

                # Ball: soft gray circle with a small black center dot for
                # precise visual alignment with the path
                pygame.draw.circle(screen, BALL_COLOR, (int(ball_x), BALL_Y), BALL_RADIUS)
                pygame.draw.circle(screen, CENTER_DOT_COLOR, (int(ball_x), BALL_Y), CENTER_DOT_RADIUS)

                # No on-screen HUD showing cycle count / trial details during
                # the actual task, per request - only simple state messages
                if trial_state == "waiting":
                    msg = font_big.render("Step on the force plates to begin", True, (255, 220, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(80 * scale)))
                elif trial_state == "calibrating":
                    msg = font_big.render("Calibrating - stand still...", True, (255, 220, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(40 * scale)))
                elif trial_state == "done":
                    msg = font_big.render("Trial Complete!", True, (100, 255, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(40 * scale)))

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        cop.stop()
        time.sleep(0.2)
        csvfile.flush()
        csvfile.close()
        pygame.quit()

    print(f"\nSaved CSV: {csv_path}")
    print(f"  Total frames logged: {frame_count}")
    print(f"  Cycles traversed: {cycle_idx}")


if __name__ == "__main__":
    main()
