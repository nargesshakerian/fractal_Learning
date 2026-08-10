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


# =====================================================================
#  TRIAL SETTINGS
# =====================================================================
TRIAL_DURATION_SEC = 300        # 5 minutes

# Fixed frequency for this trial. THIS IS PERSON-SPECIFIC - set based on
# that participant's preferred frequency (e.g. from the free-sway task).
FIXED_FREQUENCY_HZ = 0.3

# fGn (Davies-Harte) parameters for the amplitude sequence.
# H changes between trials (this is the independent variable of interest).
# sd is person-specific too - fixed here for now, one value for everyone.
HURST = 0.5          # <-- change per trial condition
FGN_SD = 1.0          # <-- person-specific standard deviation, fixed for now
RNG_SEED = 19950711   # keep fixed for reproducibility across trials, or None for random

# Amplitude range that the (normalized) fGn sequence gets mapped into (px,
# scaled at runtime same as everything else below)
BASE_AMP_MIN = 50
BASE_AMP_MAX = 300


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
BASE_BALL_RADIUS = 34
BASE_AUTO_SPEED = 0.7           # vertical scroll speed in px per frame
BASE_GRID_SPACING = 60
BASE_FONT_SIZE = 14

BALL_Y_FRACTION = 0.72    # ball resting position, fraction of screen height from top

BASE_LINE_WIDTH = 9
LINE_COLOR = (180, 150, 40)     # dark yellow path
BALL_COLOR = (210, 60, 60)      # plain red ball (no image needed)
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
    AMP_MIN_PX     = int(BASE_AMP_MIN * scale)
    AMP_MAX_PX     = int(BASE_AMP_MAX * scale)
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
    print(f"Ball: {BALL_RADIUS}px at y={BALL_Y}px, Amp range: {AMP_MIN_PX}-{AMP_MAX_PX}px")
    print(f"Fixed frequency: {FIXED_FREQUENCY_HZ}Hz -> FREQUENCY={FREQUENCY:.5f} (world units)")
    print(f"5 min distance: {total_distance:.0f}px -> generating {NUM_CYCLES} cycles")
    print(f"Amplitude source: fgn_sim (Davies-Harte), H={HURST}, sd={FGN_SD}, seed={RNG_SEED}")

    # Build the amplitude pathway: one fGn value per cycle, normalized to [0,1]
    raw_fgn = fgn_sim(NUM_CYCLES, HURST, sd=FGN_SD, seed=RNG_SEED)
    amp_values = normalize_01(raw_fgn)
    amp_map = CycleAmplitudeMap(FREQUENCY, amp_values, AMP_MIN_PX, AMP_MAX_PX)

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
    font = pygame.font.SysFont("consolas", FONT_SIZE)
    font_big = pygame.font.SysFont("consolas", max(20, int(28 * scale)), bold=True)

    world_y = 0.0
    ball_x = WIDTH / 2.0
    frame_count = 0

    trial_state = "waiting"      # waiting -> calibrating -> running -> done
    trial_start_time = None
    trial_elapsed = 0.0
    cycle_idx = 0

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if trial_state == "waiting":
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

            # Ball: plain filled circle (no image needed)
            pygame.draw.circle(screen, BALL_COLOR, (int(ball_x), BALL_Y), BALL_RADIUS)

            # HUD
            if trial_state == "waiting":
                msg = font_big.render("Step on the force plates to begin", True, (255, 220, 100))
                screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(80 * scale)))
            elif trial_state == "calibrating":
                msg = font_big.render("Calibrating - stand still...", True, (255, 220, 100))
                screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(40 * scale)))
            elif trial_state == "running":
                remaining = max(0, TRIAL_DURATION_SEC - trial_elapsed)
                mins = int(remaining) // 60
                secs = int(remaining) % 60
                hud1 = f"H={HURST}  Cycle: {cycle_idx}  Time: {mins}:{secs:02d}"
                screen.blit(font.render(hud1, True, (110, 110, 140)), (15, 12))
                hud2 = f"ESC=quit"
                screen.blit(font.render(hud2, True, (90, 90, 110)), (15, 12 + FONT_SIZE + 6))
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