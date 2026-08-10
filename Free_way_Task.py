"""
Free-Sway Task — find preferred frequency and preferred amplitude.

Protocol:
  1. Read the instructions, press SPACE when ready.
  2. Step on the two force plates.
  3. Stand comfortably, feet shoulder-width apart (calibration phase).
  4. Freely shift weight left/right at whatever speed feels
     comfortable, for the recording duration.
  5. A single ball on screen shows real-time CoP position. The display
     range auto-scales to whatever excursion the person actually
     reaches, so the ball is never clipped at the screen edge.

No white ball, no imposed frequency - this task is purely about
observing the person's own natural preferred sway. Meant to be run
MULTIPLE times (e.g. 10 repeats) per person; each run saves to its own
timestamped CSV file so repeated runs never overwrite each other.
"""

import pygame
import sys
import math
import csv
import asyncio
import time
import threading
from datetime import datetime

import numpy as np
import qtm

#  TASK SETTINGS

CALIBRATION_SECONDS = 10.0
RECORD_SECONDS = 30.0
EDGE_MARGIN_PX = 20


#  FORCE PLATE SETTINGS

QTM_IP = "127.0.0.1"
QTM_PLATE_IDS = [3, 5]   # 3 = left foot plate, 5 = right foot plate

FZ_THRESHOLD_N = 20.0
COP_SMOOTH_ALPHA = 0.3

PLATE_WIDTH_MM = 600.0
PLATE_OFFSET_MM = {
    3: -PLATE_WIDTH_MM / 2,
    5: +PLATE_WIDTH_MM / 2,
}

# AXIS NOTE (confirmed July 2026 in QTM Project): raw y_a = true
# mediolateral axis, raw x_a = true anteroposterior axis.


#  DISPLAY

FPS = 60
BG_COLOR = (20, 20, 30)
BALL_RADIUS = 45
CENTER_DOT_RADIUS = 0
BALL_COLOR = (190, 190, 190)     # soft gray, matches the main amplitude task
CENTER_DOT_COLOR = (0, 0, 0)
CENTER_LINE_COLOR = (60, 60, 70)

# Auto-scaling: the display range starts at this many mm and grows to
# fit the largest excursion actually reached, with a bit of margin, so
# the ball is never clipped at the screen edge regardless of how far
# the person naturally sways.
INITIAL_DISPLAY_RANGE_MM = 150.0
DISPLAY_RANGE_MARGIN = 1.15   # keep 15% headroom above the current max


class QTMTwoPlateCopInput:
    """Streams CoP from two force plates and computes combined_cop_shared:
    a force-weighted average of each foot's ML position, converted to
    shared coordinates (local + plate physical offset). Reflects real
    spatial CoP excursion, not just weight ratio between feet."""

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

    def calibrate(self, seconds=5.0):
        """Stand naturally while this runs. Computes the whole-body
        resting CoP position, used as the zero-reference for the ball."""
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

    def get_centered_cop_mm(self):
        """Returns the smoothed, calibration-referenced CoP position in
        mm (positive = right, negative = left), or None if unavailable."""
        raw_cop = self._get_combined_cop_shared_raw()
        if raw_cop is not None:
            centered_raw = raw_cop - self._whole_body_center_shared_mm
            if self._cop_combined_s is None:
                self._cop_combined_s = centered_raw
            else:
                a = self.smooth_alpha
                self._cop_combined_s = a * centered_raw + (1.0 - a) * self._cop_combined_s
        return self._cop_combined_s

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


def main():
    pygame.init()

    display_info = pygame.display.Info()
    WIDTH = display_info.current_w
    HEIGHT = display_info.current_h

    try:
        screen = pygame.display.set_mode(
            (WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.DOUBLEBUF, vsync=1
        )
    except pygame.error:
        print("vsync=1 not supported on this system, falling back without it.")
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.DOUBLEBUF)

    pygame.display.set_caption("Free Sway Task - Preferred Frequency & Amplitude")

    clock = pygame.time.Clock()
    center_x = WIDTH // 2
    ball_y = HEIGHT // 2

    font = pygame.font.SysFont("timesnewroman", 28)
    font_big = pygame.font.SysFont("timesnewroman", 40, bold=True)
    font_instructions = pygame.font.SysFont("timesnewroman", 32)

    cop = QTMTwoPlateCopInput(QTM_IP, QTM_PLATE_IDS, FZ_THRESHOLD_N, COP_SMOOTH_ALPHA, PLATE_OFFSET_MM)
    cop.start(WIDTH)

    # timestamped CSV filename so repeated runs never overwrite each other
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"free_sway_log_{ts}.csv"
    log_file = open(csv_path, mode="w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "time", "frame", "ball_x", "combined_cop_shared_mm", "display_range_mm",
        "fz_left", "ml_left", "ap_left",
        "fz_right", "ml_right", "ap_right",
    ])

    instruction_lines = [
        "Stand on the two force plates, feet shoulder-width apart.",
        "Once the trial starts, freely shift your weight left and right",
        "at whatever speed feels comfortable to you.",
        "",
        "Press SPACE when you are ready to begin.",
    ]

    trial_state = "instructions"   # instructions -> waiting -> calibrating -> running -> done
    trial_start_time = None
    trial_elapsed = 0.0
    frame_count = 0
    ball_x = center_x

    # auto-scaling display range: starts small, grows to fit the
    # largest excursion actually reached (never shrinks mid-trial, so
    # the scale doesn't jump around distractingly)
    display_range_mm = INITIAL_DISPLAY_RANGE_MM

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
                pass

            elif trial_state == "waiting":
                ball_x = center_x
                if cop.loaded:
                    trial_state = "calibrating"
                    print(f"Plate loaded! Calibrating for {CALIBRATION_SECONDS:.1f}s - stand still...")
                    cop.calibrate(CALIBRATION_SECONDS)
                    print("Calibration done. Trial starting!")
                    trial_state = "running"
                    trial_start_time = time.perf_counter()

            elif trial_state == "running":
                trial_elapsed = time.perf_counter() - trial_start_time

                cop_mm = cop.get_centered_cop_mm()
                if cop_mm is not None:
                    # grow the display range if the person has exceeded
                    # it, with margin - never shrink mid-trial
                    needed_range = abs(cop_mm) * DISPLAY_RANGE_MARGIN
                    if needed_range > display_range_mm:
                        display_range_mm = needed_range

                    fraction = max(-1.0, min(1.0, cop_mm / display_range_mm))
                    max_extent_px = (WIDTH / 2) - EDGE_MARGIN_PX - BALL_RADIUS
                    ball_x = center_x + fraction * max_extent_px

                if trial_elapsed >= RECORD_SECONDS:
                    trial_state = "done"
                    print(f"Trial complete! {RECORD_SECONDS:.0f}s elapsed.")

            # --- logging (only during running) ---
            if trial_state == "running":
                raw_snapshot = cop.get_raw_snapshot()
                left_pid, right_pid = QTM_PLATE_IDS[0], QTM_PLATE_IDS[1]
                cop_mm = cop.get_centered_cop_mm()

                frame_count += 1
                log_writer.writerow([
                    round(trial_elapsed, 4),
                    frame_count,
                    round(ball_x, 2),
                    round(cop_mm, 2) if cop_mm is not None else "",
                    round(display_range_mm, 2),
                    round(raw_snapshot[left_pid]["fz"], 2),
                    round(raw_snapshot[left_pid]["ml"], 2) if raw_snapshot[left_pid]["ml"] is not None else "",
                    round(raw_snapshot[left_pid]["ap"], 2) if raw_snapshot[left_pid]["ap"] is not None else "",
                    round(raw_snapshot[right_pid]["fz"], 2),
                    round(raw_snapshot[right_pid]["ml"], 2) if raw_snapshot[right_pid]["ml"] is not None else "",
                    round(raw_snapshot[right_pid]["ap"], 2) if raw_snapshot[right_pid]["ap"] is not None else "",
                ])

            # --- draw ---
            screen.fill(BG_COLOR)

            if trial_state == "instructions":
                y_start = HEIGHT // 2 - 120
                line_spacing = 44
                for i, line in enumerate(instruction_lines):
                    if not line:
                        continue
                    msg = font_instructions.render(line, True, (220, 220, 220))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, y_start + i * line_spacing))
            else:
                pygame.draw.line(screen, CENTER_LINE_COLOR, (center_x, 0), (center_x, HEIGHT), 2)
                pygame.draw.circle(screen, BALL_COLOR, (int(ball_x), ball_y), BALL_RADIUS)
                pygame.draw.circle(screen, CENTER_DOT_COLOR, (int(ball_x), ball_y), CENTER_DOT_RADIUS)

                if trial_state == "waiting":
                    msg = font_big.render("Step on the force plates to begin", True, (255, 220, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 100))
                elif trial_state == "calibrating":
                    msg = font_big.render("Calibrating - stand still...", True, (255, 220, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 100))
                elif trial_state == "done":
                    msg = font_big.render("Trial Complete!", True, (100, 255, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 100))

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        cop.stop()
        time.sleep(0.2)
        pygame.quit()
        log_file.close()

    print(f"\nDone. {frame_count} frames recorded to {csv_path}")
    print(f"Final display range reached: {display_range_mm:.1f}mm")
    sys.exit()


if __name__ == "__main__":
    main()
