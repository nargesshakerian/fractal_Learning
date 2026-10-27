"""
CAL_FREQ_CoP.py
===============
Calibration task: preferred frequency, ball controlled by CoP.

Changes 2026-10-07 (nothing else in the task changed):
  1. Start screen: Subject ID (S01, S02, ...) + Group (FREQ / AMP).
     Both groups do all four calibrations, so either group is accepted.
  2. File naming, same system as the trial scripts:
       <Subject>_<Group>_<Controller>_CALFREQ_<Trial>_<type>
       e.g. S01_FREQ_CoP_CALFREQ_T01_log60hz.csv
            S01_FREQ_CoP_CALFREQ_T01_fp1000hz.csv
            S01_FREQ_CoP_CALFREQ_T01_summary.csv
     No timestamp; the trial number increases automatically.
     Files are saved in data/<Subject>/ next to this script.
  3. First instruction screen shortened (same content as before).
  4. QTM_DATA_FOLDER updated to Game\\ATTs (the old load carriage\\jumpVM
     path meant the 1000 Hz force file was never found).
"""

import pygame
import sys
import csv
import glob
import os
import asyncio
import time
import threading
import statistics
import re
from datetime import datetime

import numpy as np
import qtm_rt

# =====================================================================
#  FFT / STFT frequency estimation (computed alongside the original
#  cycle-counting method, added directly to the summary CSV output)
# =====================================================================
FFT_MIN_FREQ_OF_INTEREST = 0.05   # Hz - excludes slow drift/DC
FFT_MAX_FREQ_OF_INTEREST = 2.0    # Hz - excludes implausibly fast sway
STFT_WINDOW_DURATION_SEC = 5.0
STFT_OVERLAP_FRACTION = 0.5


def compute_fft_peak(t_arr, signal_arr):
    """Returns the single dominant FFT frequency (peak) for the given
    (uniformly-ish sampled) time/signal arrays. Returns None if the
    signal is too short to analyze."""
    if len(t_arr) < 4:
        return None
    fs = 1.0 / np.mean(np.diff(t_arr))
    centered = signal_arr - np.mean(signal_arr)
    windowed = centered * np.hanning(len(centered))
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=1 / fs)
    magnitude = np.abs(fft_vals)

    band_mask = (freqs >= FFT_MIN_FREQ_OF_INTEREST) & (freqs <= FFT_MAX_FREQ_OF_INTEREST)
    if not np.any(band_mask):
        return None
    f, m = freqs[band_mask], magnitude[band_mask]
    return float(f[np.argmax(m)])


def compute_stft_mean_sd(t_arr, signal_arr, window_duration_sec=STFT_WINDOW_DURATION_SEC,
                          overlap_fraction=STFT_OVERLAP_FRACTION):
    """Splits the signal into short overlapping windows, finds each
    window's own dominant FFT peak, then returns the mean/SD across
    these per-window peaks. Returns (None, None, []) if there isn't
    enough data for at least one window."""
    if len(t_arr) < 4:
        return None, None, []
    fs = 1.0 / np.mean(np.diff(t_arr))
    window_samples = int(window_duration_sec * fs)
    hop_samples = max(1, int(window_samples * (1 - overlap_fraction)))

    if window_samples < 4 or window_samples > len(signal_arr):
        return None, None, []

    peak_freqs = []
    i = 0
    while i + window_samples <= len(signal_arr):
        segment = signal_arr[i:i + window_samples]
        centered = segment - np.mean(segment)
        windowed = centered * np.hanning(len(centered))

        seg_fft = np.fft.rfft(windowed)
        seg_freqs = np.fft.rfftfreq(len(windowed), d=1 / fs)
        seg_mag = np.abs(seg_fft)

        band_mask = (seg_freqs >= FFT_MIN_FREQ_OF_INTEREST) & (seg_freqs <= FFT_MAX_FREQ_OF_INTEREST)
        if np.any(band_mask):
            peak_idx = np.argmax(seg_mag[band_mask])
            peak_freqs.append(seg_freqs[band_mask][peak_idx])

        i += hop_samples

    if len(peak_freqs) == 0:
        return None, None, []
    peak_freqs = np.array(peak_freqs)
    mean_freq = float(np.mean(peak_freqs))
    sd_freq = float(np.std(peak_freqs, ddof=1)) if len(peak_freqs) > 1 else 0.0
    return mean_freq, sd_freq, peak_freqs.tolist()

#  SCRIPT IDENTITY (used in file names - do not edit)
SCRIPT_NAME = "CAL_FREQ_CoP"
TASK_TAG = "CALFREQ"      # appears in the file names
CONTROLLER = "CoP"
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

#  TASK SETTINGS

CALIBRATION_SECONDS = 5.0
# Second calibration phase: measures the person's real, safe base-of-
# support range (see calibrate_amplitude()), used to clamp the ball's
# travel and set the warning-zone boundary - replaces the previous
# arbitrary, screen-based EDGE_MARGIN_PX-only limit.
AMPLITUDE_CALIBRATION_SECONDS = 10.0
RECORD_SECONDS = 60.0     # fixed duration - frequency is what varies here,
                          # not cycle count (unlike the amplitude/free-sway task)
EDGE_MARGIN_PX = 20

# Minimum excursion (mm) the signal must exceed before a zero-crossing
# is counted - same hysteresis dead-zone logic as the free-sway task,
# to avoid counting small noise wiggles near center as spurious cycles.
MIN_CROSSING_THRESHOLD_MM = 20.0


#  FORCE PLATE SETTINGS

QTM_IP = "127.0.0.1"
QTM_PLATE_IDS = [3, 5]   # 3 = left foot plate, 5 = right foot plate

# Where QTM auto-saves measurement files (see "Start Capture" dialog's
# "Automatic Capture Control" settings) - used to locate the
# full-rate (e.g. 1000Hz) force TSV export after each trial.
QTM_DATA_FOLDER = r"C:\Users\alikens\Game\ATTs"

FZ_THRESHOLD_N = 20.0
COP_SMOOTH_ALPHA = 0.3

PLATE_WIDTH_MM = 600.0
PLATE_OFFSET_MM = {
    3: -PLATE_WIDTH_MM / 2,
    5: +PLATE_WIDTH_MM / 2,
}

# AXIS NOTE (confirmed July 2026 in QTM Project): raw y_a = true
# mediolateral axis, raw x_a = true anteroposterior axis.

FPS = 60
BG_COLOR = (0, 0, 0)             # pure black background
BALL_RADIUS = 50
CENTER_DOT_RADIUS = 10
BALL_COLOR = (190, 190, 190)     # soft gray, matches the amplitude-related tasks
BALL_WARNING_COLOR = (212, 83, 74)   # ball turns this color once past the border line
CENTER_DOT_COLOR = (0, 0, 0)
CENTER_LINE_COLOR = (60, 60, 70)

# Two simple solid lines marking the calibrated safe base-of-support
# boundary (at +/- safe_amp_mm) - replaces the earlier soft
# gradient/shaded zone. The ball is clamped so it can reach a line but
# not go past it; it turns BALL_WARNING_COLOR once it reaches either line.
BORDER_LINE_COLOR = (160, 160, 175)
BORDER_LINE_WIDTH = 3

# Fixed, PERSON-INDEPENDENT mm-to-pixel reference scale (replaces the
# old auto-scaling display range). Deliberately NOT derived from each
# person's own safe_amp_mm - if it were, the scale would normalize
# itself and the border would always land at the same screen fraction
# for everyone, hiding real calibration differences. Instead, this is
# one fixed number for all trials/people, so someone with a smaller
# calibrated safe_amp_mm visibly sees the border sit closer to center,
# and someone with a larger one sees it sit further out.
# 200mm was chosen as a round number a bit above the largest base of
# support typically seen - adjust this single constant if your pilot
# data suggests otherwise.
REFERENCE_RANGE_MM = 200.0


class QTMTwoPlateCopInput:
    """Streams CoP from two force plates and computes combined_cop_shared:
    a force-weighted average of each foot's ML position, converted to
    shared coordinates (local + plate physical offset). Reflects real
    spatial CoP excursion, not just weight ratio between feet."""

    def __init__(self, qtm_ip, plate_ids, fz_threshold_n, smooth_alpha, plate_offset_mm, qtm_password="password"):
        self.qtm_ip = qtm_ip
        self.qtm_password = qtm_password
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
        self.status_text = "Not started"
        # Set once the streaming thread's measurement has stopped, so
        # export_1000hz_after_trial() (called from main(), after
        # cop.stop() has fully finished) knows which auto-saved TSV
        # files belong to this trial.
        self.trial_stop_time = None

        self.current_x_px = 0.0

    def start(self, width):
        self.current_x_px = width / 2
        self._running = True
        self._thread = threading.Thread(target=self._thread_entry, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        # Wait for the background thread to fully finish - this
        # includes the 8s wait for QTM's auto-save (see _async_main),
        # so by the time stop() returns, trial_stop_time is set and
        # QTM should have finished writing its TSV export.
        if self._thread is not None:
            self._thread.join(timeout=15.0)

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

    def calibrate_amplitude(self, seconds, screen, font, width, height):
        """Ask the person to slowly shift their weight left and right,
        as far as is comfortable and safe, for `seconds`. Records the
        centered CoP values (mm) reached, and returns a SYMMETRIC safe
        amplitude in mm, based on the SMALLER of the max positive and
        max negative excursions reached - so the resulting safe range
        never assumes the person can go further in one direction than
        they demonstrated in the other.

        This directly measures the person's real, safe base-of-support
        range - used to clamp the ball's on-screen travel and the
        warning-zone boundary, instead of an arbitrary screen-based
        limit.

        Shows the instruction message and waits (paused, no countdown
        running yet) for a SPACE press before starting the timed
        `seconds`-long data-collection countdown - so the person isn't
        rushed into starting before they're actually ready."""
        message = "Slowly shift your weight left and right, as far as is comfortable"
        ready_message = "Press SPACE when you are ready to begin"
        samples = []

        # --- Pause: show the instruction, wait for an explicit SPACE
        # press before the timed countdown starts ---
        waiting_for_ready = True
        while waiting_for_ready:
            screen.fill((20, 20, 30))
            msg_surface = font.render(message, True, (220, 220, 220))
            screen.blit(msg_surface, (width // 2 - msg_surface.get_width() // 2, height // 2 - 40))
            ready_surface = font.render(ready_message, True, (255, 220, 100))
            screen.blit(ready_surface, (width // 2 - ready_surface.get_width() // 2, height // 2 + 20))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass  # let the main loop handle actual quitting later
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting_for_ready = False

            time.sleep(0.01)

        t_end = time.time() + float(seconds)
        while time.time() < t_end:
            cop_mm = self.get_centered_cop_mm()
            if cop_mm is not None:
                samples.append(cop_mm)

            remaining = max(0, int(t_end - time.time()) + 1)
            screen.fill((20, 20, 30))
            msg_surface = font.render(message, True, (220, 220, 220))
            screen.blit(msg_surface, (width // 2 - msg_surface.get_width() // 2, height // 2 - 40))
            timer_surface = font.render(f"{remaining}s remaining", True, (150, 150, 150))
            screen.blit(timer_surface, (width // 2 - timer_surface.get_width() // 2, height // 2 + 20))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass  # let the main loop handle actual quitting later

            time.sleep(0.01)

        positive_samples = [s for s in samples if s > 0]
        negative_samples = [s for s in samples if s < 0]

        max_positive = max(positive_samples) if positive_samples else 0.0
        max_negative = min(negative_samples) if negative_samples else 0.0

        safe_amp_mm = min(abs(max_positive), abs(max_negative))
        if safe_amp_mm <= 0:
            print("  Warning: no clear left/right weight shift detected during amplitude calibration.")
            print(f"  Defaulting to a conservative amplitude of {0.5 * REFERENCE_RANGE_MM:.0f}mm")
            safe_amp_mm = 0.5 * REFERENCE_RANGE_MM

        print(f"  Max CoP reached during amplitude calibration: left={max_negative:.1f}mm, right={max_positive:.1f}mm")
        print(f"  Symmetric safe amplitude (base of support): {safe_amp_mm:.1f}mm")

        return safe_amp_mm

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
        self.status_text = "Connecting to QTM..."
        conn = await qtm_rt.connect(self.qtm_ip)
        if conn is None:
            self.status_text = "ERROR: could not connect to QTM"
            print("Could not connect to QTM.")
            return

        self.status_text = "Connected"
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

        # NOTE: connection.start() MUST happen INSIDE TakeControl, AFTER
        # control has been taken. This game/task only needs the LIVE
        # (rate-limited, ~60-100Hz) stream for controlling the ball -
        # the full 1000Hz raw data is retrieved SEPARATELY after the
        # trial ends, via export_1000hz_after_trial() below, which
        # relies on QTM's own automatic save (Batch capture /
        # "Save captured and processed measurement automatically")
        # rather than a manual save() call from this class.
        async with qtm_rt.TakeControl(conn, self.qtm_password):
            self.status_text = "Control taken - sending Start command..."
            try:
                await conn.start()
                self.status_text = "WAITING FOR SPACE IN THE QTM WINDOW"
                print("Started measurement/recording from this script.")
            except Exception as e:
                self.status_text = f"start() error: {e}"
                print(f"connection.start() response/error (may be harmless): {e}")

            stream_task = asyncio.create_task(
                conn.stream_frames(components=["3d", "force"], on_packet=on_packet))
            try:
                while self._running:
                    await asyncio.sleep(0.01)
            finally:
                stream_task.cancel()
                try:
                    await conn.stream_frames_stop()
                except Exception:
                    pass
                try:
                    await conn.stop()
                    print("Measurement stopped.")
                except Exception as e:
                    print(f"stop() error (may be harmless): {e}")

                # Record when the trial ended, so export_1000hz_after_trial()
                # (called separately, after this thread's stop()) can
                # identify which auto-saved TSV files were written for
                # THIS trial (by modification time), rather than any
                # older leftover files in the QTM data folder.
                self.trial_stop_time = time.time()
                print("Waiting for QTM's automatic save + TSV export to finish...")
                await asyncio.sleep(8.0)

        try:
            res = conn.disconnect()
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            pass
        print("QTM disconnected")


def export_1000hz_after_trial(qtm_data_folder, plate_ids, plate_offset_mm, trial_stop_time,
                               game_start_wall_time, out_path):
    """Finds the force TSV files QTM auto-saved for this trial (the
    ones with the newest modification time, at/after trial_stop_time),
    filters to just the plates used (via each file's FORCE_PLATE_NAME
    header), and combines them into a single full-rate (e.g. 1000Hz)
    CSV with all raw variables (Force/Moment/COP in all 3 axes) for
    each plate, plus combined_cop_shared_mm computed with the SAME
    formula (including PLATE_OFFSET_MM) as the live game uses.

    game_start_wall_time is the wall-clock time.time() at which the
    actual game trial began - used together with the TSV's own
    TIME_STAMP (the recording's absolute start time) to mark each row
    with whether it falls before (calibration/waiting) or at-or-after
    (the real running trial) the game's actual start, via a
    "game_started" column. Without this, the 1000Hz export can't be
    reliably trimmed to just the real trial - e.g. this script's
    export previously ran ~93s long for a ~60s trial, with no way to
    tell where the real trial actually started within it.

    Returns the output path on success, or None if no matching files
    were found."""
    ALL_VARS = ["Force_X", "Force_Y", "Force_Z", "Moment_X", "Moment_Y", "Moment_Z",
                "COP_X", "COP_Y", "COP_Z"]

    all_force_tsvs = glob.glob(os.path.join(qtm_data_folder, "*_f_*.tsv"))
    recent_tsvs = [p for p in all_force_tsvs if os.path.getmtime(p) >= trial_stop_time - 2.0]
    recent_tsvs.sort(key=os.path.getmtime)

    if not recent_tsvs:
        print(f"\nNo new force TSV files found in {qtm_data_folder!r} written "
              f"after this trial (trial_stop_time={trial_stop_time}).")
        print("Check that ExportTSV + Force data + 'Save captured and")
        print("processed measurement automatically' are enabled in QTM.")
        return None

    print(f"\nFound {len(recent_tsvs)} newly-written force TSV file(s): {recent_tsvs}")

    plate_data = {}
    recording_start_wall_time = None
    for tsv_path in recent_tsvs:
        with open(tsv_path, "r") as f:
            head = [next(f) for _ in range(15)]
        plate_name = None
        for line in head:
            if line.startswith("FORCE_PLATE_NAME"):
                plate_name = line.strip().split("\t")[1]
                break
        frequency_hz, rec_start, rows = parse_force_tsv(tsv_path)
        if rec_start is not None:
            recording_start_wall_time = rec_start   # same for every plate file of this trial
        plate_data[plate_name or tsv_path] = (frequency_hz, rows)
        print(f"  {tsv_path} -> plate '{plate_name}', {frequency_hz}Hz, {len(rows)} rows")

    target_plate_names = {f"FP{pid}" for pid in plate_ids}
    plate_data = {name: data for name, data in plate_data.items() if name in target_plate_names}
    print(f"Filtered to target plates {sorted(target_plate_names)}: {list(plate_data.keys())}")

    plate_names = list(plate_data.keys())
    if len(plate_names) != 2:
        print(f"\nExpected exactly 2 plate files, found {len(plate_names)} - aborting combine.")
        return None

    (freq_a, rows_a), (freq_b, rows_b) = plate_data.values()
    name_a, name_b = plate_names
    pid_a = int(name_a.replace("FP", ""))
    pid_b = int(name_b.replace("FP", ""))
    offset_a = plate_offset_mm[pid_a]
    offset_b = plate_offset_mm[pid_b]

    n = min(len(rows_a), len(rows_b))
    print(f"\nCombining {name_a} (offset={offset_a}mm) and {name_b} (offset={offset_b}mm), {n} matched samples...")

    # How many seconds into the recording the actual game started -
    # None if we couldn't determine either timestamp, in which case
    # game_started is left blank for every row rather than guessing.
    game_start_offset_sec = None
    if recording_start_wall_time is not None and game_start_wall_time is not None:
        game_start_offset_sec = game_start_wall_time - recording_start_wall_time
        print(f"Recording started at wall-clock {recording_start_wall_time:.3f}, "
              f"game started at {game_start_wall_time:.3f} "
              f"-> game_started at TIME={game_start_offset_sec:.3f}s into the recording")
    else:
        print("Warning: could not determine recording start time or game start time - "
              "the 'game_started' column will be left blank for all rows.")

    header = ["sample", "t_sec"]
    header += [f"{var}_{name_a}" for var in ALL_VARS]
    header += [f"{var}_{name_b}" for var in ALL_VARS]
    header += ["combined_cop_shared_mm", "game_started"]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n):
            ra, rb = rows_a[i], rows_b[i]
            try:
                values_a = [float(ra[var]) for var in ALL_VARS]
                values_b = [float(rb[var]) for var in ALL_VARS]
                row_t_sec = float(ra["TIME"])
            except (ValueError, KeyError):
                continue

            fz_a = values_a[ALL_VARS.index("Force_Z")]
            fz_b = values_b[ALL_VARS.index("Force_Z")]
            cop_y_a = values_a[ALL_VARS.index("COP_Y")]
            cop_y_b = values_b[ALL_VARS.index("COP_Y")]

            cop_y_a_shared = cop_y_a + offset_a
            cop_y_b_shared = cop_y_b + offset_b

            total_fz = abs(fz_a) + abs(fz_b)
            combined = ((abs(fz_a) * cop_y_a_shared + abs(fz_b) * cop_y_b_shared) / total_fz
                        if total_fz > 0 else "")

            if game_start_offset_sec is not None:
                game_started = int(row_t_sec >= game_start_offset_sec)
            else:
                game_started = ""

            writer.writerow(
                [ra["SAMPLE"], ra["TIME"]]
                + [round(v, 4) for v in values_a]
                + [round(v, 4) for v in values_b]
                + [round(combined, 4) if combined != "" else "", game_started]
            )
    print(f"Saved combined 1000Hz CSV (all variables) to {out_path}")
    return out_path


def parse_force_tsv(tsv_path):
    """Parses a QTM force TSV export file. Returns (frequency_hz,
    recording_start_wall_time, rows), where rows is a list of dicts
    with the file's actual columns (SAMPLE, TIME, Force_X/Y/Z,
    Moment_X/Y/Z, COP_X/Y/Z) - values are left as strings.
    recording_start_wall_time is the recording's absolute start time
    (as a time.time()-style Unix timestamp, parsed from the file's
    TIME_STAMP metadata line), used to translate each row's relative
    TIME (seconds since recording start) into an absolute wall-clock
    time comparable to game_start_wall_time. The real column header
    line is the one starting with "SAMPLE" (metadata lines above it
    are all ALL_CAPS or start with FORCE_PLATE_)."""
    frequency_hz = None
    recording_start_wall_time = None

    with open(tsv_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("FREQUENCY"):
            frequency_hz = float(line.strip().split("\t")[1])
        if line.startswith("TIME_STAMP"):
            # Format: "TIME_STAMP\t2026-09-07, 16:36:54.798\t262972.51436260"
            # - the second field is a human-readable local date/time;
            # parse that into a Unix timestamp.
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                date_str = parts[1].strip()
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d, %H:%M:%S.%f")
                    recording_start_wall_time = dt.timestamp()
                except ValueError:
                    recording_start_wall_time = None

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("SAMPLE"):
            header_idx = i
            break

    if header_idx is None:
        return frequency_hz, recording_start_wall_time, []

    column_names = lines[header_idx].strip().split("\t")
    rows = []
    for line in lines[header_idx + 1:]:
        values = line.strip().split("\t")
        if len(values) < len(column_names):
            continue
        rows.append(dict(zip(column_names, values)))

    return frequency_hz, recording_start_wall_time, rows


Z_90PCT = 1.6448536269514722   # two-sided 5%/95% z-score for a Gaussian


def compute_fft_peak_width_sd(t_arr, signal_arr):
    """Aaron's proposed method: from the SAME whole-trial FFT peak
    used by compute_fft_peak (not a sliding STFT), estimate the
    spread/SD of frequency directly from the WIDTH of that peak - a
    signal whose instantaneous frequency wanders around a mean value
    produces a correspondingly broadened FFT peak (same principle as
    FM spectral broadening), so the peak's width is itself a measure
    of that wander.

    Method: take the power spectrum (|FFT|^2) within the analysis
    band, normalize it to a cumulative distribution over frequency,
    find the frequencies where cumulative power reaches 5% and 95%
    (a 90% interval), and convert that width to an equivalent
    Gaussian SD (for a normal distribution the 5th/95th percentiles
    sit at +/-1.6449 SD from the mean, so SD = (f95-f05)/(2*1.6449)).

    IMPORTANT: this uses the FULL analysis band (FFT_MIN/MAX_FREQ_OF_
    INTEREST), not just a narrow lobe isolated around the peak by its
    nearest local minima. An earlier version tried isolating just the
    peak's own lobe this way and badly UNDERESTIMATED the true SD
    (~40x too small in testing) - real FFT power spectra have small
    sidelobe ripples right next to the main peak (a normal side-effect
    of windowing), and local-minimum detection kept stopping at the
    first tiny ripple instead of the genuine edge of the broadened
    peak. Validated 2026-09-23 against synthetic frequency-modulated
    signals with known true SD (0.05-0.15Hz range): this full-band
    version recovers the true SD reasonably closely (e.g. true 0.10Hz
    -> estimated 0.102Hz), while the lobe-isolated version does not.
    Not yet validated against a real recorded trial - do that once a
    raw signal CSV is available.

    Returns (peak_freq_hz, sd_hz, f05_hz, f95_hz), or all None if the
    signal is too short / has no usable peak."""
    if len(t_arr) < 4:
        return None, None, None, None
    fs = 1.0 / np.mean(np.diff(t_arr))
    centered = signal_arr - np.mean(signal_arr)
    windowed = centered * np.hanning(len(centered))
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=1 / fs)
    power = np.abs(fft_vals) ** 2

    band_mask = (freqs >= FFT_MIN_FREQ_OF_INTEREST) & (freqs <= FFT_MAX_FREQ_OF_INTEREST)
    if not np.any(band_mask):
        return None, None, None, None
    f_band = freqs[band_mask]
    p_band = power[band_mask]
    if np.sum(p_band) <= 0:
        return None, None, None, None

    peak_freq = float(f_band[np.argmax(p_band)])
    cum = np.cumsum(p_band)
    cum = cum / cum[-1]
    f05 = float(np.interp(0.05, cum, f_band))
    f95 = float(np.interp(0.95, cum, f_band))
    sd_hz = (f95 - f05) / (2 * Z_90PCT)

    return peak_freq, sd_hz, f05, f95



#  START SCREEN (Subject ID + Group) AND FILE NAMING
GROUP_OPTIONS = ["FREQ", "AMP"]
SUBJECT_PATTERN = re.compile(r"^S\d{2,3}$")   # S01, S02, ... S120



#  LAUNCHER SUPPORT
#  When this script is started by game_launcher.py it receives the subject,
#  group (and for trials the H level and quiet-standing times) on the
#  command line, e.g.
#      python FREQ_CoP_trial.py --subject S015 --group FREQ --hurst 0.5
#  and then skips the start screen. Started on its own (no arguments) it
#  shows the start screen as before.
LAUNCH_INFO = None


def parse_launcher_args():
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--subject")
    p.add_argument("--group", choices=GROUP_OPTIONS)
    p.add_argument("--hurst", type=float)
    p.add_argument("--quiet-pre", type=float, dest="quiet_pre")
    p.add_argument("--quiet-post", type=float, dest="quiet_post")
    args, _unknown = p.parse_known_args()
    if not args.subject:
        return None, None
    if not SUBJECT_PATTERN.match(args.subject) or args.group is None:
        print(f"Launcher arguments not valid: subject={args.subject!r} group={args.group!r}")
        sys.exit(2)
    info = {"subject": args.subject, "group": args.group}
    if args.quiet_pre is not None:
        info["quiet_pre_sec"] = max(0.0, args.quiet_pre)
    if args.quiet_post is not None:
        info["quiet_post_sec"] = max(0.0, args.quiet_post)
    return info, args.hurst


def get_session_info(screen, title, expected_group=None):
    """Start screen. Returns {"subject": "S01", "group": "FREQ"} or None
    if the window is closed / Esc is pressed. If expected_group is
    given, choosing the other group shows an error instead of starting
    (this script only runs the expected_group's pathway)."""
    if LAUNCH_INFO is not None:                      # started from game_launcher.py
        if expected_group is not None and LAUNCH_INFO["group"] != expected_group:
            print(f"This script runs the {expected_group} trial, but group {LAUNCH_INFO['group']} was given.")
            sys.exit(2)
        return dict(LAUNCH_INFO)
    clock = pygame.time.Clock()
    w, h = screen.get_size()
    font_title = pygame.font.SysFont("arial", max(24, int(h * 0.055)))
    font = pygame.font.SysFont("arial", max(18, int(h * 0.035)))
    font_small = pygame.font.SysFont("arial", max(14, int(h * 0.025)))

    bg, fg, muted, accent, err_col = (20, 24, 32), (235, 235, 235), (150, 155, 165), (70, 140, 230), (230, 90, 90)
    box, box_active = (40, 46, 58), (55, 62, 78)

    bw, bh = int(w * 0.28), int(h * 0.07)
    cx = w // 2 - bw // 2
    id_rect = pygame.Rect(cx, int(h * 0.32), bw, bh)
    grp_rect = pygame.Rect(cx, int(h * 0.50), bw, bh)
    opt_rects = [pygame.Rect(cx, grp_rect.bottom + i * bh, bw, bh) for i in range(len(GROUP_OPTIONS))]
    start_rect = pygame.Rect(cx, int(h * 0.74), bw, bh)

    subject, group, focus, dropdown_open, error = "S", None, "id", False, ""

    def try_submit():
        nonlocal error
        if not SUBJECT_PATTERN.match(subject):
            error = "Subject ID must look like S01, S02, ..."
            return None
        if group is None:
            error = "Please select a group."
            return None
        if expected_group is not None and group != expected_group:
            error = f"This script runs the {expected_group} trial - choose {expected_group} or open the {group} script."
            return None
        return {"subject": subject, "group": group}

    def draw_box(rect, active):
        pygame.draw.rect(screen, box_active if active else box, rect, border_radius=8)
        pygame.draw.rect(screen, accent if active else muted, rect, 2, border_radius=8)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    result = try_submit()
                    if result:
                        return result
                elif focus == "id":
                    if event.key == pygame.K_BACKSPACE:
                        subject = subject[:-1]
                    elif event.unicode.isalnum() and len(subject) < 4:
                        subject += event.unicode.upper()
                    error = ""
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if dropdown_open:
                    for opt, r in zip(GROUP_OPTIONS, opt_rects):
                        if r.collidepoint(pos):
                            group, error = opt, ""
                    dropdown_open = False
                    continue
                if id_rect.collidepoint(pos):
                    focus = "id"
                elif grp_rect.collidepoint(pos):
                    dropdown_open, focus = True, None
                elif start_rect.collidepoint(pos):
                    result = try_submit()
                    if result:
                        return result
                else:
                    focus = None

        screen.fill(bg)
        t = font_title.render(title, True, fg)
        screen.blit(t, t.get_rect(center=(w // 2, int(h * 0.15))))

        lab = font_small.render("Subject ID", True, muted)
        screen.blit(lab, (id_rect.x, id_rect.y - lab.get_height() - 6))
        draw_box(id_rect, focus == "id")
        txt = subject + ("|" if focus == "id" and pygame.time.get_ticks() // 500 % 2 else "")
        screen.blit(font.render(txt, True, fg), (id_rect.x + 14, id_rect.centery - font.get_height() // 2))

        lab = font_small.render("Group", True, muted)
        screen.blit(lab, (grp_rect.x, grp_rect.y - lab.get_height() - 6))
        draw_box(grp_rect, dropdown_open)
        gtxt = font.render(group or "Select...", True, fg if group else muted)
        screen.blit(gtxt, (grp_rect.x + 14, grp_rect.centery - font.get_height() // 2))
        screen.blit(font.render("v", True, muted), (grp_rect.right - 30, grp_rect.centery - font.get_height() // 2))

        pygame.draw.rect(screen, accent, start_rect, border_radius=8)
        st = font.render("Start", True, (255, 255, 255))
        screen.blit(st, st.get_rect(center=start_rect.center))

        if error:
            e = font_small.render(error, True, err_col)
            screen.blit(e, e.get_rect(center=(w // 2, start_rect.bottom + 30)))

        if dropdown_open:
            mouse = pygame.mouse.get_pos()
            for opt, r in zip(GROUP_OPTIONS, opt_rects):
                pygame.draw.rect(screen, box_active if r.collidepoint(mouse) else box, r)
                pygame.draw.rect(screen, muted, r, 1)
                screen.blit(font.render(opt, True, fg), (r.x + 14, r.centery - font.get_height() // 2))

        hint = font_small.render("Enter = start    Esc = quit", True, muted)
        screen.blit(hint, hint.get_rect(center=(w // 2, int(h * 0.93))))
        pygame.display.flip()
        clock.tick(30)


def make_cal_stem(save_dir, subject, group, controller, task_tag):
    """'S01_FREQ_CoP_CALFREQ_T01', with the trial number increased
    automatically so existing files are never overwritten."""
    os.makedirs(save_dir, exist_ok=True)
    base = f"{subject}_{group}_{controller}_{task_tag}"
    pattern = re.compile(re.escape(base) + r"_T(\d+)_")
    used = [int(m.group(1)) for f in os.listdir(save_dir) if (m := pattern.match(f))]
    return f"{base}_T{max(used, default=0) + 1:02d}"


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

    pygame.display.set_caption(f"{SCRIPT_NAME} - Calibration - preferred frequency (CoP)")

    # ---------------- start screen: Subject ID + Group ----------------
    info = get_session_info(screen, title="Calibration - preferred frequency (CoP)")
    if info is None:
        pygame.quit()
        sys.exit()
    save_dir = os.path.join(DATA_ROOT, info["subject"])
    stem = make_cal_stem(save_dir, info["subject"], info["group"], CONTROLLER, TASK_TAG)
    print(f"Files for this calibration: {save_dir}\\{stem}_*")

    clock = pygame.time.Clock()
    center_x = WIDTH // 2
    ball_y = HEIGHT // 2

    # Loading Arial by its exact file path (rather than SysFont, which
    # lets SDL pick/substitute a font and was found to still produce
    # slightly stretched glyphs even with bold=False) to guarantee the
    # real Arial Regular TTF is used with no synthetic transformation.
    ARIAL_PATH = r"C:\Windows\Fonts\arial.ttf"
    try:
        font = pygame.font.Font(ARIAL_PATH, 32)
        font_big = pygame.font.Font(ARIAL_PATH, 72)
        font_instructions = pygame.font.Font(ARIAL_PATH, 56)
    except FileNotFoundError:
        print(f"Warning: could not find {ARIAL_PATH} - falling back to SysFont.")
        font = pygame.font.SysFont("arial", 32)
        font_big = pygame.font.SysFont("arial", 72)
        font_instructions = pygame.font.SysFont("arial", 56)


    cop = QTMTwoPlateCopInput(QTM_IP, QTM_PLATE_IDS, FZ_THRESHOLD_N, COP_SMOOTH_ALPHA, PLATE_OFFSET_MM)
    cop.start(WIDTH)

    raw_1000hz_out_path = os.path.join(save_dir, f"{stem}_fp1000hz.csv")

    csv_path = os.path.join(save_dir, f"{stem}_log60hz.csv")
    log_file = open(csv_path, mode="w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "time", "frame", "ball_x", "combined_cop_shared_mm", "safe_amp_mm",
        "cycle_number",
        "fz_left", "ml_left", "ap_left",
        "fz_right", "ml_right", "ap_right",
    ])

    instruction_lines = [
        "Stand on the force plates, feet shoulder-width apart.",
        f"Shift your weight left and right, in your preffered speed, for {int(RECORD_SECONDS)} s.",
        "",
        "Press SPACE to begin.",
    ]

    trial_state = "instructions"   # instructions -> waiting -> ready_to_calibrate -> calibrating -> ready -> running -> done
    trial_start_time = None
    game_start_wall_time = None   # set once the real trial starts - wall-clock
                                    # time, comparable to cop.trial_stop_time,
                                    # used to mark the 1000Hz export's rows
    trial_elapsed = 0.0
    frame_count = 0
    ball_x = center_x

    # Set once calibrate_amplitude() runs (see the "ready_to_calibrate"
    # key handler below); None until then, meaning no clamp is applied
    # yet (there's no ball movement to clamp before calibration anyway).
    safe_amp_mm = None

    # --- cycle/frequency detection state (zero-crossing based) ---
    last_cop_sign = None
    zero_crossings = 0
    cycle_number = 0
    current_cycle_start_time = None
    cycle_frequencies = []

    # (t_sec, cop_mm) pairs collected during the real trial only, used
    # to compute FFT/STFT frequency estimates after the trial ends.
    signal_history = []

    # --- per-cycle amplitude peak tracking (same logic as the
    # free-sway/amplitude task) - tracked alongside frequency so the
    # summary can report both for each cycle. Only starts tracking
    # once we've had at least one crossing, so the initial transient
    # (before the first real excursion) doesn't contaminate cycle 1's
    # peaks - same reasoning as why cycle 1 is excluded from the
    # frequency mean.
    current_cycle_peak_pos = 0.0
    current_cycle_peak_neg = 0.0
    cycle_peaks = {}   # cycle_number -> (peak_right_mm, peak_left_mm)

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if trial_state == "instructions":
                            trial_state = "waiting"
                        elif trial_state == "ready_to_calibrate":
                            trial_state = "calibrating"
                            print(f"Calibrating for {CALIBRATION_SECONDS:.1f}s - stand still...")
                            cop.calibrate(CALIBRATION_SECONDS)
                            print(f"Amplitude calibration: shift weight left/right for "
                                  f"{AMPLITUDE_CALIBRATION_SECONDS:.1f}s ...")
                            safe_amp_mm = cop.calibrate_amplitude(
                                AMPLITUDE_CALIBRATION_SECONDS, screen, font_instructions, WIDTH, HEIGHT
                            )
                            print("Calibration done. Press SPACE when you are ready to start the trial.")
                            trial_state = "ready"
                        elif trial_state == "ready":
                            trial_state = "running"
                            trial_start_time = time.perf_counter()
                            # Also record the wall-clock time of the actual
                            # game start (not just the perf_counter used for
                            # in-game timing) - comparable to cop.trial_stop_time
                            # (also wall-clock), so export_1000hz_after_trial()
                            # can mark which rows of the 1000Hz export are
                            # calibration/waiting vs. the real running trial.
                            game_start_wall_time = time.time()
                            print("Trial started!")

            if trial_state == "instructions":
                pass

            elif trial_state == "waiting":
                ball_x = center_x
                if cop.loaded:
                    trial_state = "ready_to_calibrate"
                    print("Plate loaded! Press SPACE when the person is ready to hold still for calibration.")

            elif trial_state == "ready_to_calibrate":
                ball_x = center_x

            elif trial_state == "ready":
                ball_x = center_x

            elif trial_state == "running":
                trial_elapsed = time.perf_counter() - trial_start_time

                cop_mm = cop.get_centered_cop_mm()
                if cop_mm is not None:
                    signal_history.append((trial_elapsed, cop_mm))

                    # Fixed, person-independent mm-to-pixel scale (see
                    # REFERENCE_RANGE_MM above) - NOT derived from this
                    # person's own safe_amp_mm, so the resulting pixel
                    # position genuinely reflects how their calibration
                    # compares to others, rather than always landing at
                    # the same relative screen position for everyone.
                    max_extent_px = (WIDTH / 2) - EDGE_MARGIN_PX - BALL_RADIUS
                    px_per_mm = max_extent_px / REFERENCE_RANGE_MM
                    if safe_amp_mm is not None:
                        border_offset_px = max(0.0, min(max_extent_px, safe_amp_mm * px_per_mm))
                    else:
                        border_offset_px = max_extent_px  # shouldn't normally be reached in "running"

                    # Ball CAN reach the calibrated border line, but not
                    # go past it - clamped here, not just visually.
                    ball_offset_px = max(-border_offset_px, min(border_offset_px, cop_mm * px_per_mm))
                    ball_x = center_x + ball_offset_px

                    if cop_mm > MIN_CROSSING_THRESHOLD_MM:
                        current_sign = 1
                    elif cop_mm < -MIN_CROSSING_THRESHOLD_MM:
                        current_sign = -1
                    else:
                        current_sign = last_cop_sign

                    if current_sign is not None:
                        # track amplitude peaks within the current cycle,
                        # same as the free-sway task - only once we've
                        # had at least one crossing, so the initial
                        # transient doesn't contaminate cycle 1
                        if zero_crossings > 0:
                            if cop_mm > current_cycle_peak_pos:
                                current_cycle_peak_pos = cop_mm
                            if cop_mm < current_cycle_peak_neg:
                                current_cycle_peak_neg = cop_mm

                        if last_cop_sign is not None and current_sign != last_cop_sign:
                            zero_crossings += 1

                            if zero_crossings == 1:
                                current_cycle_start_time = trial_elapsed
                            elif zero_crossings % 2 == 0 and zero_crossings >= 2:
                                if current_cycle_start_time is not None:
                                    duration = trial_elapsed - current_cycle_start_time
                                    if duration > 0:
                                        cycle_number += 1
                                        freq_hz = 1.0 / duration
                                        cycle_frequencies.append((cycle_number, duration, freq_hz))
                                        cycle_peaks[cycle_number] = (current_cycle_peak_pos, current_cycle_peak_neg)
                                        print(f"  Cycle {cycle_number} complete: "
                                              f"duration={duration:.2f}s  freq={freq_hz:.3f}Hz  "
                                              f"peak_right={current_cycle_peak_pos:.1f}mm  peak_left={current_cycle_peak_neg:.1f}mm")
                                current_cycle_start_time = trial_elapsed
                                current_cycle_peak_pos = 0.0
                                current_cycle_peak_neg = 0.0
                        last_cop_sign = current_sign

                if trial_elapsed >= RECORD_SECONDS:
                    trial_state = "done"
                    print(f"Trial complete! {RECORD_SECONDS:.0f}s elapsed.")

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
                    round(safe_amp_mm, 2) if safe_amp_mm is not None else "",
                    cycle_number,
                    round(raw_snapshot[left_pid]["fz"], 2),
                    round(raw_snapshot[left_pid]["ml"], 2) if raw_snapshot[left_pid]["ml"] is not None else "",
                    round(raw_snapshot[left_pid]["ap"], 2) if raw_snapshot[left_pid]["ap"] is not None else "",
                    round(raw_snapshot[right_pid]["fz"], 2),
                    round(raw_snapshot[right_pid]["ml"], 2) if raw_snapshot[right_pid]["ml"] is not None else "",
                    round(raw_snapshot[right_pid]["ap"], 2) if raw_snapshot[right_pid]["ap"] is not None else "",
                ])

            screen.fill(BG_COLOR)

            if trial_state in ("instructions", "waiting", "ready_to_calibrate", "calibrating", "ready"):
                if trial_state == "instructions":
                    y_start = HEIGHT // 2 - 140
                    line_spacing = 52
                    for i, line in enumerate(instruction_lines):
                        if not line:
                            continue
                        msg = font_instructions.render(line, True, (220, 220, 220))
                        screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, y_start + i * line_spacing))

                elif trial_state == "waiting":
                    msg = font_big.render("Step on the force plates to begin", True, (255, 220, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 20))

                elif trial_state == "ready_to_calibrate":
                    msg = font_big.render("Press SPACE to begin calibration", True, (255, 220, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 20))

                elif trial_state == "calibrating":
                    msg = font_big.render("Calibrating - stand still...", True, (255, 220, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 20))

                elif trial_state == "ready":
                    msg = font_big.render("Calibration complete!", True, (100, 255, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 60))
                    msg2 = font_instructions.render("Press SPACE when you are ready to start", True, (220, 220, 220))
                    screen.blit(msg2, (WIDTH // 2 - msg2.get_width() // 2, HEIGHT // 2 + 10))

            else:
                pygame.draw.line(screen, CENTER_LINE_COLOR, (center_x, 0), (center_x, HEIGHT), 2)

                # Soft shaded warning zones near each edge. Unlike
                # before, these are now positioned from the person's
                # OWN calibrated safe_amp_mm (via the same fixed
                # px_per_mm scale used for the ball above) - this is
                # the actual fix: previously this block used the
                # screen-only max_extent_px directly, so the zone never
                # moved regardless of calibration. Now a smaller
                # safe_amp_mm visibly pulls the border in toward center.
                max_extent_px = (WIDTH / 2) - EDGE_MARGIN_PX - BALL_RADIUS
                px_per_mm = max_extent_px / REFERENCE_RANGE_MM
                if safe_amp_mm is not None:
                    border_offset_px = max(0.0, min(max_extent_px, safe_amp_mm * px_per_mm))
                else:
                    border_offset_px = max_extent_px  # shouldn't normally be reached in "running"

                left_edge_x = center_x - border_offset_px
                right_edge_x = center_x + border_offset_px

                # Two simple solid lines marking the calibrated safe
                # base-of-support boundary - no gradient/shaded zone.
                pygame.draw.line(screen, BORDER_LINE_COLOR, (left_edge_x, 0), (left_edge_x, HEIGHT), BORDER_LINE_WIDTH)
                pygame.draw.line(screen, BORDER_LINE_COLOR, (right_edge_x, 0), (right_edge_x, HEIGHT), BORDER_LINE_WIDTH)

                # Ball turns red once it has reached a line (it cannot
                # go past them - see the clamp above); white otherwise.
                reached_border = (ball_x <= left_edge_x) or (ball_x >= right_edge_x)
                current_ball_color = BALL_WARNING_COLOR if reached_border else BALL_COLOR
                pygame.draw.circle(screen, current_ball_color, (int(ball_x), ball_y), BALL_RADIUS)
                pygame.draw.circle(screen, CENTER_DOT_COLOR, (int(ball_x), ball_y), CENTER_DOT_RADIUS)

                if trial_state == "done":
                    msg = font_big.render("Trial Complete!", True, (100, 255, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 100))

                if trial_state == "running":
                    remaining = max(0, RECORD_SECONDS - trial_elapsed)
                    hud = font.render(f"{remaining:.0f}s remaining  |  Cycles: {cycle_number}", True, (120, 120, 140))
                    screen.blit(hud, (20, 20))

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        cop.stop()
        time.sleep(0.2)
        pygame.quit()
        log_file.close()

    print(f"\nDone. {frame_count} frames recorded to {csv_path}")
    print(f"Complete cycles detected: {cycle_number}")

    # Retrieve the full force-plate-rate (e.g. 1000Hz) data for this
    # trial, now that cop.stop() has finished (including QTM's own
    # auto-save + TSV export, which the stop() call waited for).
    if cop.trial_stop_time is not None:
        export_1000hz_after_trial(
            qtm_data_folder=QTM_DATA_FOLDER,
            plate_ids=QTM_PLATE_IDS,
            plate_offset_mm=PLATE_OFFSET_MM,
            trial_stop_time=cop.trial_stop_time,
            game_start_wall_time=game_start_wall_time,
            out_path=raw_1000hz_out_path,
        )
    else:
        print("\nWarning: cop.trial_stop_time was never set - skipping 1000Hz export.")

    if cycle_frequencies:
        print("\nPer-cycle frequencies and amplitude peaks:")
        print(f"  {'Cycle':>6} {'Duration(s)':>12} {'Freq(Hz)':>10} {'PeakRight(mm)':>14} {'PeakLeft(mm)':>13}")
        for cyc, dur, freq in cycle_frequencies:
            peak_pos, peak_neg = cycle_peaks.get(cyc, (float("nan"), float("nan")))
            note = "  <- excluded (first cycle may include the initial transient/timing offset)" if cyc == cycle_frequencies[0][0] else ""
            print(f"  {cyc:6d} {dur:12.2f} {freq:10.3f} {peak_pos:14.1f} {peak_neg:13.1f}{note}")

        # Drop the first cycle before computing mean/SD: its START time
        # is anchored to the FIRST zero-crossing, which (same as the
        # free-sway task's peak-tracking fix) can be reached after an
        # initial small-wiggle transient near center - making cycle 1's
        # measured duration/frequency unreliable compared to later,
        # fully-formed cycles.
        if len(cycle_frequencies) > 1:
            clean_frequencies = cycle_frequencies[1:]
        else:
            print("\n  Warning: only 1 cycle completed - cannot drop it, using it anyway.")
            clean_frequencies = cycle_frequencies

        freqs_only = [f for (_, _, f) in clean_frequencies]
        mean_freq_hz = statistics.mean(freqs_only)
        sd_freq_hz = statistics.stdev(freqs_only) if len(freqs_only) > 1 else 0.0

        # Amplitude peaks: use ALL cycles recorded in cycle_peaks
        # (except the first, same "drop the first cycle" convention
        # used for frequency, since cycle 1's start is anchored to the
        # first zero-crossing and can include the initial transient) -
        # previously this only included cycles that ALSO appeared in
        # clean_frequencies, which could drop some valid peak
        # measurements for no real reason if the two lists didn't line
        # up exactly.
        all_peak_cycle_numbers = sorted(cycle_peaks.keys())
        first_cycle_number = cycle_frequencies[0][0]
        clean_peak_cycle_numbers = [cyc for cyc in all_peak_cycle_numbers if cyc != first_cycle_number]
        if not clean_peak_cycle_numbers:
            clean_peak_cycle_numbers = all_peak_cycle_numbers   # fallback if only 1 cycle total

        clean_peak_magnitudes = [
            (abs(cycle_peaks[cyc][0]) + abs(cycle_peaks[cyc][1])) / 2.0
            for cyc in clean_peak_cycle_numbers
        ]
        mean_amp_mm = statistics.mean(clean_peak_magnitudes) if clean_peak_magnitudes else float("nan")

        print(f"\n  Mean frequency (excluding cycle 1, {len(freqs_only)} cycles used): {mean_freq_hz:.3f} Hz  (period {1/mean_freq_hz:.2f}s)")
        print(f"  SD of frequency (excluding cycle 1, {len(freqs_only)} cycles used): {sd_freq_hz:.3f} Hz")
        print(f"  Mean amplitude peak (excluding cycle 1, informational only): {mean_amp_mm:.1f}mm")

        # --- FFT / STFT frequency estimation, computed from the same
        # signal_history collected during the real trial ---
        fft_peak_hz = None
        fft_width_sd_hz = None
        fft_f05_hz = None
        fft_f95_hz = None
        stft_mean_hz = None
        stft_sd_hz = None
        stft_n_windows = 0
        if len(signal_history) >= 4:
            t_arr = np.array([s[0] for s in signal_history])
            sig_arr = np.array([s[1] for s in signal_history])
            fft_peak_hz = compute_fft_peak(t_arr, sig_arr)
            _, fft_width_sd_hz, fft_f05_hz, fft_f95_hz = compute_fft_peak_width_sd(t_arr, sig_arr)
            stft_mean_hz, stft_sd_hz, stft_window_peaks = compute_stft_mean_sd(t_arr, sig_arr)
            stft_n_windows = len(stft_window_peaks)

            if fft_peak_hz is not None:
                print(f"\n  FFT peak frequency: {fft_peak_hz:.4f} Hz")
            else:
                print("\n  FFT peak frequency: n/a (not enough data)")
            if fft_width_sd_hz is not None:
                print(f"  FFT peak-width SD (Aaron's 5%-95% method): {fft_width_sd_hz:.4f} Hz "
                      f"(f05={fft_f05_hz:.4f}, f95={fft_f95_hz:.4f})")
            else:
                print("  FFT peak-width SD: n/a (not enough data)")
            if stft_mean_hz is not None:
                print(f"  STFT windowed mean: {stft_mean_hz:.4f} Hz, SD: {stft_sd_hz:.4f} Hz "
                      f"({stft_n_windows} windows of {STFT_WINDOW_DURATION_SEC}s)")
            else:
                print("  STFT windowed: n/a (not enough data for even one window)")
        else:
            print("\n  FFT/STFT: not enough signal samples collected to analyze.")

        summary_path = os.path.join(save_dir, f"{stem}_summary.csv")
        with open(summary_path, "w", newline="") as sf:
            summary_writer = csv.writer(sf)
            summary_writer.writerow([
                "cycle_number", "duration_s", "frequency_hz",
                "peak_right_mm", "peak_left_mm", "lr_magnitude_mm",
                "excluded_from_mean",
            ])
            excluded_cycle_number = cycle_frequencies[0][0]
            for cyc, dur, freq in cycle_frequencies:
                peak_pos, peak_neg = cycle_peaks.get(cyc, (float("nan"), float("nan")))
                magnitude = (abs(peak_pos) + abs(peak_neg)) / 2.0
                summary_writer.writerow([
                    cyc, round(dur, 4), round(freq, 4),
                    round(peak_pos, 2), round(peak_neg, 2), round(magnitude, 2),
                    "yes" if cyc == excluded_cycle_number and len(cycle_frequencies) > 1 else "no",
                ])
            summary_writer.writerow([])
            summary_writer.writerow(["mean_frequency_hz (for FIXED_FREQUENCY_HZ)", round(mean_freq_hz, 4)])
            summary_writer.writerow(["sd_frequency_hz", round(sd_freq_hz, 4)])
            summary_writer.writerow(["mean_amplitude_peak_mm (informational)", round(mean_amp_mm, 2)])
            sd_amp_mm = statistics.stdev(clean_peak_magnitudes) if len(clean_peak_magnitudes) > 1 else 0.0
            summary_writer.writerow(["sd_amplitude_peak_mm", round(sd_amp_mm, 2)])
            summary_writer.writerow(["cycles_used", len(freqs_only)])
            summary_writer.writerow(["cycles_used_amplitude", len(clean_peak_magnitudes)])
            summary_writer.writerow([])
            summary_writer.writerow(["fft_peak_frequency_hz", round(fft_peak_hz, 4) if fft_peak_hz is not None else ""])
            summary_writer.writerow(["fft_peak_width_sd_hz (Aaron's 5%-95% method)", round(fft_width_sd_hz, 4) if fft_width_sd_hz is not None else ""])
            summary_writer.writerow(["fft_peak_width_f05_hz", round(fft_f05_hz, 4) if fft_f05_hz is not None else ""])
            summary_writer.writerow(["fft_peak_width_f95_hz", round(fft_f95_hz, 4) if fft_f95_hz is not None else ""])
            summary_writer.writerow(["stft_mean_frequency_hz", round(stft_mean_hz, 4) if stft_mean_hz is not None else ""])
            summary_writer.writerow(["stft_sd_frequency_hz", round(stft_sd_hz, 4) if stft_sd_hz is not None else ""])
            summary_writer.writerow(["stft_windows_used", stft_n_windows])
        print(f"\n  Summary saved to {summary_path}")
    else:
        print("\nNo complete cycles were detected during this trial.")

    sys.exit()


if __name__ == "__main__":
    LAUNCH_INFO, _launch_hurst = parse_launcher_args()
    main()
