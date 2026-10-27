"""
AMP_CoP_trial.py
================
Amplitude-variable tracking trial (fixed frequency, per-cycle amplitude
from fGn), ball CONTROLLED BY CoP (two force
plates). Motion capture is recorded in parallel, so the game log holds
BOTH CoP and whole-body CoM for every frame.

Based on AMP_VAR_922 (same changes as FREQ_CoP_trial.py). Changes (2026-10-07):
  1. Start screen: Subject ID (S01, S02, ...) + Group (FREQ / AMP).
  2. File naming: <Subject>_<Group>_<Controller>_<H>_<Trial>_<type>
       e.g. S01_AMP_CoP_H07_T01_game60hz.csv
            S01_AMP_CoP_H07_T01_fp1000hz.csv
            S01_AMP_CoP_H07_T01_mocap100hz.csv
            S01_AMP_CoP_H07_T01_pathway.png
            S01_AMP_CoP_H07_T01_meta.json
     No timestamp in the name; the trial number (T01, T02, ...)
     increases automatically so nothing is ever overwritten.
     Files are saved in  data/<Subject>/  next to this script.
  3. Live skeleton streaming added (CoM computed with de Leva 1996,
     female) - CoM is LOGGED only, the ball is still driven by CoP.
     The skeleton is streamed in GLOBAL coordinates ("skeleton:global").
     QTM's real-time skeleton component sends segment positions
     relative to the parent segment by default, which is why the old
     CoM-trial logs showed com_z ~389 mm instead of ~1000 mm.
  4. After the trial, the 100 Hz CoM is also rebuilt from QTM's
     auto-exported skeleton TSV (as in the CoM trial script), next to
     the 1000 Hz force-plate export.
  5. BUG FIX - WAVELENGTH_STRETCH: the stretch used to be applied to
     time (world_y / 2), so the pathway ran at HALF the intended
     frequency and only the first half of the generated cycles (whose
     H was checked) was ever shown. The stretch is now applied only to
     the on-screen drawing, so timing and cycle count are correct.
     Visible consequence: the path scrolls twice as fast on screen as
     before (same on-screen wavelength, correct frequency).
     In AMP_VAR_922 the halved timing also slowed the "fixed"
     frequency (0.33 Hz played as ~0.165 Hz).
 10. QTM_DATA_FOLDER updated to the Game\\ATTs folder (AMP_VAR_922
     still pointed to load carriage\\jumpVM, which is why the amplitude
     trial had no 1000 Hz force-plate file).
  6. cycle_idx in the log now matches the cycle actually on screen.
  7. Game log: clearer column names (cop_* / com_*), target and ball
     positions in mm, wall-clock time per frame, QC flags.
  8. Real-time QC (CoM frozen / CoM missing / participant stepped off)
     with counts shown at the end of the trial and stored in metadata.
  9. Metadata JSON per trial (centers used for centering, body weight,
     mm<->px scale, realized H, QTM source files, QC counts, ...).
"""

import pygame
import sys
import math
import asyncio
import time
import threading
import csv
import json
import re
import shutil
from datetime import datetime

import numpy as np
import qtm_rt
import glob
import os
import matplotlib
matplotlib.use("Agg")   # save-only backend - doesn't need a display, safe to
                        # import alongside pygame's own display window
import matplotlib.pyplot as plt


#  SCRIPT IDENTITY (used in file names and metadata - do not edit per trial)
SCRIPT_NAME = "AMP_CoP_trial"
SCRIPT_VERSION = "2026-10-26c"
MANIPULATION = "AMP"      # this script = amplitude-variable pathway
CONTROLLER = "CoP"        # ball is driven by CoP in this script

#  TRIAL SETTINGS
TRIAL_DURATION_SEC = 300        # 5 minutes

# Fixed frequency for this trial - same pace for everyone.
# NUM_CYCLES = round(FIXED_FREQUENCY_HZ * TRIAL_DURATION_SEC); the
# frequency actually used (EFFECTIVE_FREQUENCY_HZ = NUM_CYCLES /
# TRIAL_DURATION_SEC) is adjusted very slightly so NUM_CYCLES cycles
# exactly fill TRIAL_DURATION_SEC.
FIXED_FREQUENCY_HZ = 0.33

HURST = 0.5             # <-- change per trial condition
RNG_SEED = None          # each trial gets a fully independent random sequence

# Pathway H-matching (see generate_amplitude_pathway_within_tolerance).
H_TOLERANCE = 0.001              # HK grid step is 0.005 -> 0.001 = exact match on the grid
MAX_CANDIDATE_ATTEMPTS = 10000   # safety cap, not a target

#  Amplitude distribution - PERSON-SPECIFIC, from this person's
#  preferred-amplitude/free-sway task (mean/SD of per-cycle
#  lr_magnitude), fed directly into fgn_sim's mu/sd.
AMP_MEAN_MM = 129
AMP_SD_MM = 7


#  QTM / FORCE PLATE SETTINGS
QTM_IP = "127.0.0.1"
QTM_PLATE_IDS = [3, 5]   # 3 = left foot plate, 5 = right foot plate

# Where QTM auto-saves measurement files (force *_f_*.tsv and skeleton
# *_s_*.tsv exports). Both Force and Skeleton must be ticked in QTM's
# TSV export settings for the 1000 Hz and 100 Hz files to be produced.
QTM_DATA_FOLDER = r"C:\Users\alikens\Game\ATTs"

# Where THIS script saves its own files: data/<Subject>/ next to the script.
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FZ_THRESHOLD_N = 20.0
COP_SMOOTH_ALPHA = 0.3
CALIBRATION_SECONDS = 5.0     # stand still: sets the ball's zero point (resting CoP / CoM)
#  QUIET STANDING BEFORE / AFTER THE GAME
#  Order: calibration (CALIBRATION_SECONDS) -> quiet standing before the
#  game (quiet_pre) -> S starts the game -> quiet standing after the game
#  (quiet_post) -> "Trial Complete", ESC saves.
#  During both quiet periods the screen is blank; ESC is ignored during
#  quiet_post. The two durations below are only the DEFAULTS: they can be
#  changed on the start screen of every trial (0 = skip that period).
#  All periods are marked in the 1000 Hz and 100 Hz files ('phase'
#  column: calibration / quiet_pre / game / quiet_post) and in the meta file.
QUIET_PRE_SEC = 30.0
QUIET_POST_SEC = 30.0

#  PARAMETERS FROM CALIBRATION
#  True  = the person-specific values below are read from this person's
#          calibration summary files (latest CALFREQ / CALAMP of the same
#          subject, group and controller) and shown for confirmation.
#  False = the numbers typed in this file are used.
LOAD_PARAMS_FROM_CALIBRATION = True
#  setting in this script  <-  (calibration task, value in its summary file)
CAL_PARAM_MAP = {
    "FIXED_FREQUENCY_HZ": ("CALFREQ", "fft_peak_frequency_hz"),
    "AMP_MEAN_MM":        ("CALAMP",  "mean_amplitude_peak_mm"),
    "AMP_SD_MM":          ("CALAMP",  "sd_amplitude_peak_mm"),
}

EDGE_MARGIN_PX = 20

PLATE_WIDTH_MM = 600.0
PLATE_OFFSET_MM = {
    3: -PLATE_WIDTH_MM / 2,
    5: +PLATE_WIDTH_MM / 2,
}
# mm of CoP excursion mapped to half the screen width
COP_RANGE_MM = 200.0

# AXIS NOTE (confirmed July 2026 in QTM Project): raw y = true
# mediolateral axis, raw x = true anteroposterior axis (force plates
# and lab/skeleton coordinates).


#  REAL-TIME QUALITY CHECKS
COM_FREEZE_THRESHOLD_SEC = 0.10   # CoM unchanged longer than this -> "frozen"
STEP_OFF_FZ_N = 200.0             # total vertical force below this -> stepped off / unloaded
COM_HEIGHT_MIN_PLAUSIBLE_MM = 600.0   # sanity check on resting CoM height
SHOW_QC_WARNINGS_ON_SCREEN = True     # small text in the bottom-left corner


#  DISPLAY (base values at reference 600x700, scaled at runtime)
FPS = 60
BG_COLOR = (15, 15, 30)
REF_W, REF_H = 600, 700
BASE_BALL_RADIUS = 40
BASE_CENTER_DOT_RADIUS = 10
BASE_AUTO_SPEED = 0.7           # vertical scroll speed in px per frame (world units)
BASE_FONT_SIZE = 14

BALL_Y_FRACTION = 0.40    # ball resting position, fraction of screen height from top

BASE_LINE_WIDTH = 9
LINE_COLOR = (180, 150, 40)     # dark yellow path
BALL_COLOR = (190, 190, 190)    # soft/light gray ball
CENTER_DOT_COLOR = (0, 0, 0)    # black center dot for precise alignment

# Only a small window of the path is visible around the ball.
# The visible window is set in CYCLES of this participant's pathway, not in
# pixels, so every participant sees the same fraction of a cycle whatever
# their frequency, scroll speed or screen size.
#   VISIBLE_CYCLES  = how much of one AVERAGE cycle is visible (0.5 = half a cycle)
#   FRACTION_AHEAD  = share of that window ABOVE the ball (the part of the path
#                     still to come). 0.5 = centred on the ball (as before);
#                     1.0 = the whole window is ahead of the ball.
# Note: in FREQ trials single cycles are a little shorter / longer than the
# average, so the visible part of a given cycle varies around this value.
VISIBLE_CYCLES = 0.75
FRACTION_AHEAD = 0.5

# VISUAL ONLY: how many times taller one cycle is drawn on screen.
# Applied to the screen offset when drawing, NOT to time - so it does
# not change the pathway's frequency or cycle count (see change 5 above).
WAVELENGTH_STRETCH = 1


#  fGn GENERATION - Davies-Harte method, ported directly from fgn_sim.m
#  (identical to the amplitude-variable trial's implementation)
def fgn_sim(n, H, sd=1.0, mu=0.0, seed=None):
    """
    Python port of fgn_sim.m (Davies-Harte method). Generates a fractional
    Gaussian noise series of length n with Hurst exponent H.

    Reference: Beran, J. (1994). Statistics for long-memory processes.
    Original MATLAB: Nonlinear Analysis Core, Center for Human Movement
    Variability, University of Nebraska at Omaha.

    NOTE (2026-09-22): compared element-wise against lib_myl.fgn_sim
    (Vasileios's shared lab code) using identical base random noise.
    They differ by a constant factor (this version's raw output has
    std close to 1, lib_myl's is orders of magnitude smaller). This
    is NOT a bug to fix here - lib_myl's caller (fgn_noise() in
    Lua_epochs) always re-normalizes fgn_sim's raw output to unit
    variance and then rescales by the target sd itself, so lib_myl's
    raw output scale is irrelevant there. THIS script uses fgn_sim's
    raw output directly (sd * z_complex + mu, no renormalization), so
    it needs the raw output to already be close to unit variance for
    a given sd to mean what it says - which this version's scaling
    (the two "* len(z_complex)" terms below) provides. Do not "fix"
    this to match lib_myl bit-for-bit again without also adding the
    same renormalization step lib_myl's caller does.
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


# ---------------------------------------------------------------------
#  HK (Hurst-Kolmogorov) maximum-likelihood estimator of H for fGn.
#  Same estimator as the analysis pipeline (pipeline_tools.hurst_hk):
#  exact Gaussian likelihood of fGn, mean and variance profiled out,
#  H chosen on a grid 0.01-0.99 in steps of 0.005. Gives the same H as
#  the pipeline for the same series. Used here to check each candidate
#  pathway; replaces the earlier R/S (rescaled range) check.
#  Speed: the fGn correlation matrices depend only on the series length
#  and H, so they are inverted ONCE per trial (HKEstimator(n)); after
#  that each candidate takes about a millisecond.
# ---------------------------------------------------------------------
HK_H_GRID = np.linspace(0.01, 0.99, 197)      # step 0.005, same as the pipeline


class HKEstimator:
    def __init__(self, n):
        from scipy.linalg import toeplitz, cho_factor, cho_solve
        self.n = n
        k = np.arange(n, dtype=float)
        one = np.ones(n)
        eye = np.eye(n)
        inv, logdet = [], []
        for H in HK_H_GRID:
            acf = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
            c = cho_factor(toeplitz(acf))
            inv.append(cho_solve(c, eye))
            logdet.append(2 * np.sum(np.log(np.diag(c[0]))))
        self.inv = np.array(inv)                      # (grid, n, n) inverse correlation matrices
        self.logdet = np.array(logdet)
        self.inv_one = self.inv @ one                 # (grid, n)
        self.one_inv_one = self.inv_one @ one         # (grid,)

    def estimate(self, x):
        """H of series x (length n). Returns the grid value with the highest likelihood."""
        x = np.asarray(x, float)
        inv_x = self.inv @ x                          # (grid, n)
        x_inv_x = inv_x @ x
        one_inv_x = self.inv_one @ x
        mu = one_inv_x / self.one_inv_one             # best mean for each H
        e_inv_e = x_inv_x - 2 * mu * one_inv_x + mu ** 2 * self.one_inv_one
        s2 = e_inv_e / self.n                         # best variance for each H
        loglik = -0.5 * (self.n * np.log(s2) + self.logdet)
        return round(float(HK_H_GRID[int(np.argmax(loglik))]), 3)


def generate_amplitude_pathway_within_tolerance(n_cycles, target_H, amp_mean_mm, amp_sd_mm,
                                                  tolerance=0.01, max_attempts=5000):
    """Amplitude-variable trial's counterpart to the frequency-variable
    trial's generate_pathway_within_tolerance(). Generates independent
    fgn_sim draws (each clipped to a 5mm safety floor) one at a time,
    estimates each candidate's REALIZED Hurst exponent with the HK
    maximum-likelihood estimator (HKEstimator, same estimator as the
    analysis pipeline) on the cycle amplitudes, and stops as soon as
    one lands within +/-tolerance of target_H.

    UNLIKE the frequency version, there is no rescale-to-fit-duration
    step here - amplitude doesn't drive cycle timing (FIXED_FREQUENCY_HZ
    does, and it's literally fixed), so nothing needs to be rescaled to
    make cycles fit TRIAL_DURATION_SEC.

    HK works on a grid with steps of 0.005, so a tolerance of 0.001
    means "the HK estimate equals the target H exactly on that grid".
    Tested 2026-10-21 (n = 99 cycles, 30 searches per H): tolerance
    0.001 needed a median of 16-46 attempts (max 182), well under
    0.1 s per search. If max_attempts is reached, falls back to the
    closest candidate found so far and reports found=False.

    Replaces the earlier R/S (rescaled range, nolds.hurst_rs) check.

    Returns (amp_values_mm, estimated_H, n_attempts, found_within_tolerance)."""
    hk = HKEstimator(n_cycles)        # set up once per trial (~0.1 s)
    best_amp_values_mm = None
    best_estimated_H = None
    best_abs_error = None

    for attempt in range(1, max_attempts + 1):
        candidate = fgn_sim(n_cycles, target_H, sd=amp_sd_mm, mu=amp_mean_mm, seed=None)
        candidate = np.clip(candidate, 5.0, None)

        estimated_H = hk.estimate(candidate)   # H of the cycle amplitudes
        abs_error = abs(estimated_H - target_H)

        if best_abs_error is None or abs_error < best_abs_error:
            best_abs_error = abs_error
            best_estimated_H = estimated_H
            best_amp_values_mm = candidate

        if abs_error <= tolerance:
            return best_amp_values_mm, best_estimated_H, attempt, True

    return best_amp_values_mm, best_estimated_H, max_attempts, False


def normalize_01(values):
    mn, mx = np.min(values), np.max(values)
    r = mx - mn
    if r < 1e-15:
        return np.full_like(values, 0.5)
    return (values - mn) / r


def save_amplitude_debug_plot(amp_values_mm, hurst, amp_mean_mm, amp_sd_mm, fixed_freq_hz,
                               auto_speed, fps, seed, out_path, points_per_cycle=60):
    """Saves a diagnostic plot of THIS trial's actual sine wave (with
    its real per-cycle amplitude, not normalized) and per-cycle
    amplitude envelope, so you can check H is shaping the pathway
    before the trial starts. Diagnostic only - doesn't affect
    gameplay. Amplitude-variable trial's counterpart to
    save_frequency_debug_plot.

    IMPORTANT: amp_values_mm must be the SAME array actually used for
    the real trial (already fgn_sim'd and clipped by
    generate_amplitude_pathway_within_tolerance) - not a fresh fgn_sim
    draw. An earlier version of this function generated its own
    separate draw here, which (since RNG_SEED is usually None, and
    now that a candidate is selected via tolerance-search rather than
    a single draw) would show a DIFFERENT pathway than the one
    actually played - see the frequency-variable trial's file for the
    same fix and fuller explanation."""
    n_cycles = len(amp_values_mm)

    amp_map = CycleAmplitudeMap(fixed_freq_hz, n_cycles, auto_speed, fps)
    total_world_y = amp_map.cycle_length_px * n_cycles

    world_ys = np.linspace(0, total_world_y, n_cycles * points_per_cycle)
    y = np.array([amp_map.get_amplitude(wy, amp_values_mm) * math.sin(amp_map.get_phase(wy))
                  for wy in world_ys])
    times_sec = world_ys / amp_map.px_per_sec

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

    # top: the actual sine wave shape, WITH its real per-cycle
    # amplitude (mm) - shows the amplitude pattern directly, unlike
    # the frequency trial's normalized-to-1 plot
    ax1.plot(times_sec, y, linewidth=0.6, color="tab:blue")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.set_title(f"H={hurst}: actual pathway sine wave, all {n_cycles} cycles "
                  f"(real amplitude in mm - fixed frequency={fixed_freq_hz}Hz)")
    ax1.set_ylabel("Amplitude (mm)")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True, alpha=0.3)

    # bottom: the per-cycle amplitude envelope, for direct numeric reference
    ax2.plot(amp_values_mm, color="tab:blue", linewidth=1.0, marker="o", markersize=1.5)
    ax2.axhline(amp_mean_mm, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="mean")
    ax2.axhline(amp_mean_mm + amp_sd_mm, color="green", linestyle="--", linewidth=0.8, alpha=0.5, label="mean +/- sd")
    ax2.axhline(amp_mean_mm - amp_sd_mm, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_title(f"Full amplitude envelope for this trial ({n_cycles} cycles)")
    ax2.set_ylabel("Amplitude (mm)")
    ax2.set_xlabel("Cycle number")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Amplitude check - H={hurst}, mean={amp_mean_mm:.2f}mm, sd={amp_sd_mm:.2f}mm, seed={seed}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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


class CyclePhaseMap:
    """Maps a vertical world position to a PHASE (radians), by
    accumulating frequency*distance continuously across cycles - the
    frequency equivalent of CycleAmplitudeMap.

    Cycle boundaries are placed at PEAKS (phase = pi/2, 5*pi/2, ...),
    not at zero-crossings - each cycle runs peak-to-peak, and the
    frequency change happens at that peak.

    Frequency sits INSIDE sin(phase) rather than multiplying it, so
    swapping frequency abruptly would create a jump in the path.
    Instead, phase is accumulated continuously across cycles so the
    sine wave stays smooth even as frequency changes cycle to cycle."""

    def __init__(self, freq_values_hz, auto_speed, fps):
        """
        freq_values_hz: per-cycle frequency sequence (Hz)
        auto_speed, fps: convert real-world Hz to world_y (px) via the
                          AUTO_SPEED*FPS scroll rate
        """
        self.freq_values_hz = freq_values_hz
        self.px_per_sec = auto_speed * fps   # world_y px scrolled per real second

        # Precompute world_y (px) and cumulative phase (radians) at
        # each cycle boundary. Boundaries sit at peaks (phase starts
        # at pi/2, then +2*pi per cycle) rather than at zero-crossings,
        # so each freq_values_hz entry controls one full peak-to-peak
        # cycle and the frequency change happens exactly at the peak.
        self.boundary_world_y = [0.0]
        self.boundary_phase = [math.pi / 2]
        for freq_hz in freq_values_hz:
            cycle_duration_sec = 1.0 / freq_hz
            cycle_length_px = cycle_duration_sec * self.px_per_sec
            self.boundary_world_y.append(self.boundary_world_y[-1] + cycle_length_px)
            self.boundary_phase.append(self.boundary_phase[-1] + 2 * math.pi)

    def get_phase(self, world_y):
        """Returns the continuous phase (radians) at this world_y, via
        linear interpolation between the nearest precomputed cycle
        boundaries."""
        n = len(self.boundary_world_y)
        if world_y <= self.boundary_world_y[0]:
            # before the first boundary - extrapolate backward from
            # the first peak, using the first cycle's frequency
            freq_hz = self.freq_values_hz[0]
            return self.boundary_phase[0] + 2 * math.pi * freq_hz * (world_y / self.px_per_sec)

        if world_y >= self.boundary_world_y[-1]:
            # past the last precomputed boundary - extrapolate using
            # the last cycle's frequency, continuing from the last
            # known phase
            freq_hz = self.freq_values_hz[-1]
            extra_y = world_y - self.boundary_world_y[-1]
            extra_sec = extra_y / self.px_per_sec
            return self.boundary_phase[-1] + 2 * math.pi * freq_hz * extra_sec

        # binary search for the surrounding boundary pair
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.boundary_world_y[mid] <= world_y:
                lo = mid
            else:
                hi = mid
        y0, y1 = self.boundary_world_y[lo], self.boundary_world_y[hi]
        p0, p1 = self.boundary_phase[lo], self.boundary_phase[hi]
        frac = (world_y - y0) / (y1 - y0) if (y1 - y0) > 1e-9 else 0.0
        return p0 + frac * (p1 - p0)

    def get_cycle_index(self, world_y):
        """Returns which cycle (index into freq_values_hz) this
        world_y falls within, clamped to valid range."""
        n = len(self.boundary_world_y)
        if world_y <= self.boundary_world_y[0]:
            return 0
        if world_y >= self.boundary_world_y[-1]:
            return len(self.freq_values_hz) - 1
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.boundary_world_y[mid] <= world_y:
                lo = mid
            else:
                hi = mid
        return min(lo, len(self.freq_values_hz) - 1)


class CycleAmplitudeMap:
    """Maps a vertical world position to a PHASE (radians) using a
    FIXED frequency (every cycle is the same length), and separately
    tracks which cycle index a given world_y falls in, so the caller
    can look up that cycle's amplitude from a per-cycle amplitude
    sequence. This is the amplitude-variable trial's counterpart to
    CyclePhaseMap - frequency is constant here, amplitude varies.

    Cycle boundaries are placed at PEAKS (phase = pi/2, 5*pi/2, ...),
    matching CyclePhaseMap's convention, so the amplitude change at
    each boundary lines up with a peak rather than a zero-crossing."""

    def __init__(self, fixed_freq_hz, n_cycles, auto_speed, fps):
        """
        fixed_freq_hz: single frequency (Hz) used for every cycle
        n_cycles: how many cycles' worth of boundaries to precompute
        auto_speed, fps: convert real-world Hz to world_y (px) via the
                          AUTO_SPEED*FPS scroll rate
        """
        self.fixed_freq_hz = fixed_freq_hz
        self.px_per_sec = auto_speed * fps

        cycle_duration_sec = 1.0 / fixed_freq_hz
        self.cycle_length_px = cycle_duration_sec * self.px_per_sec
        self.n_cycles = n_cycles

    def get_phase(self, world_y):
        """Returns the continuous phase (radians) at this world_y -
        with a fixed frequency, this is just linear in world_y, no
        need for boundary interpolation."""
        return math.pi / 2 + 2 * math.pi * (world_y / self.cycle_length_px)

    def get_cycle_index(self, world_y):
        """Returns which cycle this world_y falls within, clamped to
        [0, n_cycles-1]."""
        idx = int(world_y / self.cycle_length_px)
        return max(0, min(idx, self.n_cycles - 1))

    def get_amplitude(self, world_y, amp_values_px):
        """Returns a smoothly-interpolated amplitude (px) at this
        world_y, rather than jumping abruptly to the next cycle's
        amplitude at each cycle boundary. Uses COSINE interpolation
        (not linear) between this cycle's own amplitude and the next
        cycle's amplitude - linear interpolation was tried first and
        made the VALUE continuous, but its SLOPE still changed
        abruptly right at each boundary (a sharp kink visible where
        one cycle's ramp met the next cycle's differently-sloped
        ramp). Cosine interpolation's derivative is exactly zero at
        both ends of each cycle, so consecutive cycles' ramps meet
        with matching (zero) slope - no visible kink at the peaks."""
        cycle_idx = self.get_cycle_index(world_y)
        cycle_start_y = cycle_idx * self.cycle_length_px
        frac_into_cycle = (world_y - cycle_start_y) / self.cycle_length_px  # 0..1

        current_amp = amp_values_px[cycle_idx]
        next_idx = min(cycle_idx + 1, self.n_cycles - 1)
        next_amp = amp_values_px[next_idx]

        smooth_frac = (1 - math.cos(frac_into_cycle * math.pi)) / 2.0
        return current_amp + (next_amp - current_amp) * smooth_frac


#  TWO-PLATE COP INPUT (combined_cop_shared, position-based)

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
    p.add_argument("--range", type=float, dest="range_mm")   # sensitivity (mm) for this participant
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
    if args.range_mm is not None:
        info["range_mm"] = args.range_mm
    return info, args.hurst


def get_session_info(screen, title, expected_group=None):
    """Start screen. Returns {"subject": "S01", "group": "FREQ",
    "quiet_pre_sec": 30.0, "quiet_post_sec": 30.0} or None if the window
    is closed / Esc is pressed. If expected_group is given, choosing the
    other group shows an error instead of starting (this script only runs
    the expected_group's pathway). The two quiet-standing durations start
    at QUIET_PRE_SEC / QUIET_POST_SEC and can be edited (whole seconds,
    0 = skip). Tab moves between the text boxes."""
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
    id_rect = pygame.Rect(cx, int(h * 0.27), bw, bh)
    grp_rect = pygame.Rect(cx, int(h * 0.42), bw, bh)
    opt_rects = [pygame.Rect(cx, grp_rect.bottom + i * bh, bw, bh) for i in range(len(GROUP_OPTIONS))]
    half = (bw - 16) // 2
    pre_rect = pygame.Rect(cx, int(h * 0.58), half, bh)
    post_rect = pygame.Rect(cx + bw - half, int(h * 0.58), half, bh)
    start_rect = pygame.Rect(cx, int(h * 0.74), bw, bh)

    subject, group, focus, dropdown_open, error = "S", None, "id", False, ""
    pre_txt, post_txt = f"{QUIET_PRE_SEC:g}", f"{QUIET_POST_SEC:g}"
    order = ["id", "pre", "post"]

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
        if not pre_txt.isdigit() or not post_txt.isdigit() or int(pre_txt) > 600 or int(post_txt) > 600:
            error = "Quiet standing times must be whole seconds between 0 and 600."
            return None
        return {"subject": subject, "group": group,
                "quiet_pre_sec": float(pre_txt), "quiet_post_sec": float(post_txt)}

    def draw_box(rect, active):
        pygame.draw.rect(screen, box_active if active else box, rect, border_radius=8)
        pygame.draw.rect(screen, accent if active else muted, rect, 2, border_radius=8)

    def draw_text_box(rect, label, text, name):
        lab = font_small.render(label, True, muted)
        screen.blit(lab, (rect.x, rect.y - lab.get_height() - 6))
        draw_box(rect, focus == name)
        shown = text + ("|" if focus == name and pygame.time.get_ticks() // 500 % 2 else "")
        screen.blit(font.render(shown, True, fg), (rect.x + 14, rect.centery - font.get_height() // 2))

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
                elif event.key == pygame.K_TAB:
                    focus = order[(order.index(focus) + 1) % len(order)] if focus in order else "id"
                    dropdown_open = False
                elif focus == "id":
                    if event.key == pygame.K_BACKSPACE:
                        subject = subject[:-1]
                    elif event.unicode.isalnum() and len(subject) < 4:
                        subject += event.unicode.upper()
                    error = ""
                elif focus in ("pre", "post"):
                    t = pre_txt if focus == "pre" else post_txt
                    if event.key == pygame.K_BACKSPACE:
                        t = t[:-1]
                    elif event.unicode.isdigit() and len(t) < 3:
                        t += event.unicode
                    if focus == "pre":
                        pre_txt = t
                    else:
                        post_txt = t
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
                elif pre_rect.collidepoint(pos):
                    focus = "pre"
                elif post_rect.collidepoint(pos):
                    focus = "post"
                elif start_rect.collidepoint(pos):
                    result = try_submit()
                    if result:
                        return result
                else:
                    focus = None

        screen.fill(bg)
        t = font_title.render(title, True, fg)
        screen.blit(t, t.get_rect(center=(w // 2, int(h * 0.13))))

        draw_text_box(id_rect, "Subject ID", subject, "id")

        lab = font_small.render("Group", True, muted)
        screen.blit(lab, (grp_rect.x, grp_rect.y - lab.get_height() - 6))
        draw_box(grp_rect, dropdown_open)
        gtxt = font.render(group or "Select...", True, fg if group else muted)
        screen.blit(gtxt, (grp_rect.x + 14, grp_rect.centery - font.get_height() // 2))
        screen.blit(font.render("v", True, muted), (grp_rect.right - 30, grp_rect.centery - font.get_height() // 2))

        draw_text_box(pre_rect, "Quiet before (s)", pre_txt, "pre")
        draw_text_box(post_rect, "Quiet after (s)", post_txt, "post")

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

        hint = font_small.render("Tab = next box    Enter = start    Esc = quit", True, muted)
        screen.blit(hint, hint.get_rect(center=(w // 2, int(h * 0.93))))
        pygame.display.flip()
        clock.tick(30)


def hurst_tag(hurst):
    """0.7 -> 'H07', 0.1 -> 'H01', 0.85 -> 'H085'."""
    s = f"{hurst:.2f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return "H" + s.replace(".", "")


def make_trial_stem(save_dir, subject, group, controller, hurst):
    """'S01_AMP_CoP_H07_T01', with the trial number increased
    automatically so existing files are never overwritten."""
    os.makedirs(save_dir, exist_ok=True)
    base = f"{subject}_{group}_{controller}_{hurst_tag(hurst)}"
    pattern = re.compile(re.escape(base) + r"_T(\d+)_")
    used = [int(m.group(1)) for f in os.listdir(save_dir) if (m := pattern.match(f))]
    return f"{base}_T{max(used, default=0) + 1:02d}"


#  WHOLE-BODY CoM (de Leva 1996, female) - same table as the CoM trial
DELEVA_SEGMENTS = {
    "Head":          (0.0668, 0.4841, "Neck", "Head"),
    "Trunk":         (0.4257, 0.4964, "Hips", "Spine2"),
    "LeftUpperArm":  (0.0255, 0.5754, "LeftShoulder", "LeftArm"),
    "LeftForearm":   (0.0138, 0.4559, "LeftArm", "LeftForeArm"),
    "LeftHand":      (0.0056, 0.7474, "LeftForeArm", "LeftHand"),
    "RightUpperArm": (0.0255, 0.5754, "RightShoulder", "RightArm"),
    "RightForearm":  (0.0138, 0.4559, "RightArm", "RightForeArm"),
    "RightHand":     (0.0056, 0.7474, "RightForeArm", "RightHand"),
    "LeftThigh":     (0.1478, 0.3612, "Hips", "LeftUpLeg"),
    "LeftShank":     (0.0481, 0.4416, "LeftUpLeg", "LeftLeg"),
    "LeftFoot":      (0.0129, 0.4014, "LeftLeg", "LeftFoot"),
    "RightThigh":    (0.1478, 0.3612, "Hips", "RightUpLeg"),
    "RightShank":    (0.0481, 0.4416, "RightUpLeg", "RightLeg"),
    "RightFoot":     (0.0129, 0.4014, "RightLeg", "RightFoot"),
}

# Segment ID -> name, QTM Sport Dynamic skeleton (also the column order
# in the auto-exported skeleton TSV).
SEGMENT_ID_TO_NAME = {
    1: "Hips", 2: "Spine", 3: "Spine1", 4: "Spine2", 5: "Neck", 6: "Head",
    7: "LeftShoulder", 8: "LeftArm", 9: "LeftForeArm", 10: "LeftForeArmRoll", 11: "LeftHand",
    12: "RightShoulder", 13: "RightArm", 14: "RightForeArm", 15: "RightForeArmRoll", 16: "RightHand",
    17: "LeftUpLeg", 18: "LeftLeg", 19: "LeftFoot", 20: "LeftToeBase",
    21: "RightUpLeg", 22: "RightLeg", 23: "RightFoot", 24: "RightToeBase",
}


def compute_whole_body_com(segment_positions):
    """Whole-body CoM (x, y, z) from {joint_name: (x, y, z)} in GLOBAL
    lab coordinates. Returns None if any required joint is missing or
    not finite (e.g. QTM sends NaN for an unsolved segment). Used by
    both the live stream and the post-trial 100 Hz export."""
    total_com = np.zeros(3)
    total_mass = 0.0
    for seg_name, (mass_frac, com_frac, prox_name, dist_name) in DELEVA_SEGMENTS.items():
        prox = segment_positions.get(prox_name)
        dist = segment_positions.get(dist_name)
        if prox is None or dist is None:
            return None
        prox = np.array(prox, dtype=float)[:3]
        dist = np.array(dist, dtype=float)[:3]
        if not (np.all(np.isfinite(prox)) and np.all(np.isfinite(dist))):
            return None
        total_com += mass_frac * (prox + com_frac * (dist - prox))
        total_mass += mass_frac
    if total_mass <= 0:
        return None
    return total_com / total_mass


#  QTM INPUT: two-plate CoP (controls the ball) + skeleton CoM (logged)
class QTMCopComInput:
    """Streams force plates AND the skeleton from QTM in one connection.

    CoP (control): force-weighted average of each foot's ML position in
    shared coordinates (local + plate offset) - unchanged from
    FREQ_VAR_922, so the ball behaves exactly as before.

    CoM (logged only): whole-body CoM from the GLOBAL skeleton stream.
    Only the current frame's segments are used; if any segment is
    missing/NaN, com_valid = 0 for that frame (stale positions from
    earlier frames are never mixed in)."""

    def __init__(self, qtm_ip, plate_ids, fz_threshold_n, smooth_alpha, plate_offset_mm,
                 qtm_password="password", qtm_data_folder=None):
        self.qtm_ip = qtm_ip
        self.qtm_password = qtm_password
        self.qtm_data_folder = qtm_data_folder
        self.plate_ids = list(plate_ids)
        self.fz_threshold_n = float(fz_threshold_n)
        self.smooth_alpha = float(smooth_alpha)
        self.plate_offset_mm = plate_offset_mm

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self._plate_raw = {pid: {"fz": 0.0, "ml": None, "ap": None} for pid in self.plate_ids}
        self._cop_combined_s = None          # smoothed, centered CoP (drives the ball)

        self._com = None                     # last valid CoM (x, y, z), global lab coords, mm
        self._com_valid = False              # was the CoM valid in the latest skeleton frame?
        self._com_last_change_wall = None    # wall time the CoM value last changed
        self.skeleton_seen = False           # True once a full skeleton has been solved

        # Calibration results (stored in metadata)
        self.cop_center_mm = 0.0             # resting combined CoP_ML (shared coords)
        self.com_center_mm = None            # resting CoM_ML (lab coords)
        self.com_height_mm = None            # resting CoM_Z (sanity check)
        self.body_weight_n = None            # mean total Fz during calibration

        self.status_text = "Not started"
        self.trial_stop_time = None
        self.current_x_px = 0.0

    def start(self, width):
        self.current_x_px = width / 2
        self._running = True
        self._thread = threading.Thread(target=self._thread_entry, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        # Waits for the thread, which itself waits for QTM's auto-save
        # + TSV export to finish (see _wait_for_export_stable).
        if self._thread is not None:
            self._thread.join(timeout=130.0)

    # ---------- force plates ----------
    def get_raw_snapshot(self):
        with self._lock:
            return {pid: dict(self._plate_raw[pid]) for pid in self.plate_ids}

    @property
    def loaded(self):
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

    def _total_fz(self):
        with self._lock:
            return sum(abs(self._plate_raw[pid]["fz"]) for pid in self.plate_ids)

    # ---------- skeleton / CoM ----------
    def get_com(self):
        """(com_xyz or None, valid_this_frame, frozen)."""
        now = time.time()
        with self._lock:
            com = self._com
            valid = self._com_valid
            last_change = self._com_last_change_wall
        frozen = bool(valid and last_change is not None and (now - last_change) > COM_FREEZE_THRESHOLD_SEC)
        return com, valid, frozen

    # ---------- calibration ----------
    def calibrate_center(self, seconds=5.0):
        """Stand still. Stores the resting CoP_ML (zero point for the
        ball), resting CoM_ML and CoM height, and body weight."""
        t_end = time.time() + float(seconds)
        cop_samples, com_ml, com_z, fz_total = [], [], [], []
        while time.time() < t_end:
            pygame.event.pump()   # keeps the window responsive (Windows) during the long calibration
            combined = self._get_combined_cop_shared_raw()
            if combined is not None:
                cop_samples.append(combined)
            fz_total.append(self._total_fz())
            com, valid, _ = self.get_com()
            if valid and com is not None:
                com_ml.append(com[1])   # raw Y = mediolateral
                com_z.append(com[2])
            time.sleep(0.01)

        if cop_samples:
            self.cop_center_mm = float(np.mean(cop_samples))
        else:
            self.cop_center_mm = 0.0
            print("  Warning: no combined CoP samples during calibration")
        self.body_weight_n = float(np.mean(fz_total)) if fz_total else None
        if com_ml:
            self.com_center_mm = float(np.mean(com_ml))
            self.com_height_mm = float(np.mean(com_z))
        else:
            print("  Warning: no valid CoM samples during calibration - is the skeleton tracked?")

        print(f"  Resting CoP_ML (shared coords): {self.cop_center_mm:.1f} mm")
        if self.body_weight_n:
            print(f"  Body weight: {self.body_weight_n:.1f} N (~{self.body_weight_n / 9.81:.1f} kg)")
        if self.com_center_mm is not None:
            print(f"  Resting CoM_ML: {self.com_center_mm:.1f} mm, CoM height: {self.com_height_mm:.1f} mm")
            if self.com_height_mm < COM_HEIGHT_MIN_PLAUSIBLE_MM:
                print(f"  WARNING: CoM height {self.com_height_mm:.0f} mm is implausibly low for "
                      f"standing - check the skeleton stream / coordinate system.")

    # ---------- ball control (CoP) ----------
    def get_control_x_px(self, width, ball_radius, cop_range_mm, edge_margin_px):
        raw_cop = self._get_combined_cop_shared_raw()
        if raw_cop is not None:
            centered_raw = raw_cop - self.cop_center_mm
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
        self.current_x_px = max(min_x, min(max_x, x_px))
        return self.current_x_px

    @property
    def control_signal_smoothed_mm(self):
        """The smoothed, centered CoP value that actually moved the ball."""
        return self._cop_combined_s

    def snapshot(self):
        raw = self.get_raw_snapshot()
        combined = self._get_combined_cop_shared_raw()
        com, com_valid, com_frozen = self.get_com()
        return {
            "raw": raw,
            "fz_total": sum(abs(raw[pid]["fz"]) for pid in self.plate_ids),
            "cop_ml_mm": combined,
            "cop_ml_centered_mm": (combined - self.cop_center_mm) if combined is not None else None,
            "com": com if com_valid else None,
            "com_valid": com_valid,
            "com_frozen": com_frozen,
            "com_ml_centered_mm": (com[1] - self.com_center_mm)
                                  if (com_valid and com is not None and self.com_center_mm is not None) else None,
        }

    # ---------- QTM streaming ----------
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

    async def _wait_for_export_stable(self, max_wait_sec=120.0, poll_interval_sec=2.0,
                                       required_stable_checks=3):
        """Waits until the set of force (*_f_*.tsv) AND skeleton
        (*_s_*.tsv) files written after trial_stop_time, and all their
        sizes, stop changing for a few consecutive checks (the skeleton
        file is large and keeps growing after it first appears)."""
        if self.qtm_data_folder is None or self.trial_stop_time is None:
            await asyncio.sleep(8.0)
            return

        last_snapshot = None
        stable_checks = 0
        waited = 0.0
        while waited < max_wait_sec:
            await asyncio.sleep(poll_interval_sec)
            waited += poll_interval_sec
            try:
                snapshot = {p: os.path.getsize(p)
                            for p in list_recent_qtm_files(self.qtm_data_folder, self.trial_stop_time)}
            except Exception:
                snapshot = None

            if snapshot and snapshot == last_snapshot:
                stable_checks += 1
                if stable_checks >= required_stable_checks:
                    print(f"  Export appears stable: {len(snapshot)} TSV file(s) found.")
                    return
            else:
                stable_checks = 0
            last_snapshot = snapshot

        n_last = len(last_snapshot) if last_snapshot else 0
        print(f"  Warning: gave up waiting for a stable export after {max_wait_sec}s "
              f"(last seen: {n_last} file(s)).")

    async def _async_main(self):
        self.status_text = "Connecting to QTM..."
        conn = await qtm_rt.connect(self.qtm_ip)
        if conn is None:
            self.status_text = "ERROR: could not connect to QTM"
            print("Could not connect to QTM.")
            return

        self.status_text = "Connected"
        print(f"Connected to QTM at {self.qtm_ip} (Plates {self.plate_ids} + skeleton)")

        def on_packet(packet):
            # --- force plates (control) ---
            try:
                _, plates = packet.get_force()
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
            except Exception:
                pass

            # --- skeleton (CoM, logged only) ---
            try:
                _, skeletons = packet.get_skeletons()
            except Exception:
                return
            positions = {}
            if skeletons and skeletons[0]:
                for seg_id, pos, rot in skeletons[0]:
                    seg_name = SEGMENT_ID_TO_NAME.get(seg_id)
                    if seg_name is not None:
                        positions[seg_name] = tuple(pos)[:3]
            com = compute_whole_body_com(positions)
            now = time.time()
            with self._lock:
                if com is None:
                    self._com_valid = False
                else:
                    com = tuple(float(v) for v in com)
                    if self._com is None or com != self._com:
                        self._com_last_change_wall = now
                    self._com = com
                    self._com_valid = True
                    self.skeleton_seen = True

        # connection.start() MUST happen inside TakeControl. The live
        # stream (~60-100 Hz) only drives the ball and the 60 Hz log;
        # full-rate data comes from QTM's own auto-saved TSV exports.
        #
        # "skeleton:global" = segment positions in lab coordinates.
        # Plain "skeleton" streams positions RELATIVE TO THE PARENT
        # segment, which gives a wrong whole-body CoM. "3d" is kept
        # because streaming the skeleton without it returned
        # segment_count = 0 in earlier testing.
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
                conn.stream_frames(components=["3d", "force", "skeleton:global"], on_packet=on_packet))
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

                self.trial_stop_time = time.time()
                print("Waiting for QTM's automatic save + TSV export to finish...")
                await self._wait_for_export_stable(max_wait_sec=120.0)

        try:
            res = conn.disconnect()
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            pass
        print("QTM disconnected")


def list_recent_qtm_files(qtm_data_folder, trial_stop_time):
    """All TSV exports QTM wrote for this trial (modified at/after
    trial_stop_time - 2 s): force plates (*_f_*.tsv), skeleton
    (*_s_*.tsv) and 3D markers / other data types (any other *.tsv)."""
    return sorted(p for p in glob.glob(os.path.join(qtm_data_folder, "*.tsv"))
                  if os.path.getmtime(p) >= trial_stop_time - 2.0)


# Which raw QTM TSV exports are copied into the participant folder.
# Only the skeleton (all segment positions + rotations, for angular
# momentum etc.) is kept. Force plates are NOT copied: plates 3 and 5 are
# already saved in <stem>_fp1000hz.csv, and QTM exports every plate in
# the lab (FP1-FP7), which only wastes space.
ARCHIVE_SKELETON_TSV = True
ARCHIVE_3D_TSV = False      # labeled marker trajectories - set True if ever needed


def archive_qtm_tsv(qtm_files, save_dir, stem):
    """Copies this trial's raw QTM skeleton export into the participant
    folder (see ARCHIVE_SKELETON_TSV / ARCHIVE_3D_TSV above):
        <stem>_qtm_skeleton.tsv   all segments: position + rotation (100 Hz)
        <stem>_qtm_3d.tsv         labeled 3D markers (only if ARCHIVE_3D_TSV)
    These files allow segment-level outcomes (trunk/pelvis motion, base of
    support / margin of stability, angular momentum) to be computed later.
    Returns {original path: archived file name}."""
    archived = {}
    for src in qtm_files:
        base = os.path.basename(src)
        if "_s_" in base:
            if not ARCHIVE_SKELETON_TSV:
                continue
        elif ARCHIVE_3D_TSV and "_f_" not in base:
            try:
                with open(src, "r") as f:
                    if "MARKER_NAMES" not in "".join(f.readline() for _ in range(15)):
                        continue
            except Exception:
                continue
        else:
            continue          # force plates, 2D, distance, ... are not copied
        name = None
        if "_s_" in base:
            name = "skeleton" if not any(v.endswith("_qtm_skeleton.tsv") for v in archived.values()) \
                else os.path.splitext(base)[0] + "_skeleton"
        elif "_f_" in base:
            plate = None
            try:
                with open(src, "r") as f:
                    for _ in range(20):
                        line = f.readline()
                        if line.startswith("FORCE_PLATE_NAME"):
                            plate = line.strip().split("\t")[1]
                            break
            except Exception:
                pass
            name = f"force_{plate}" if plate else os.path.splitext(base)[0]
        else:
            try:
                with open(src, "r") as f:
                    head = "".join(f.readline() for _ in range(15))
                if "MARKER_NAMES" in head:
                    name = "3d"
            except Exception:
                pass
        if name is None or any(v.endswith(f"_qtm_{name}.tsv") for v in archived.values()):
            name = os.path.splitext(base)[0]
        dst_name = f"{stem}_qtm_{name}.tsv"
        try:
            shutil.copy2(src, os.path.join(save_dir, dst_name))
            archived[src] = dst_name
        except Exception as e:
            print(f"  Warning: could not archive {src}: {e}")
    return archived


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


def export_1000hz_after_trial(qtm_data_folder, plate_ids, plate_offset_mm, trial_stop_time,
                               game_start_wall_time, out_path, phase_times=None):
    """Finds the force TSV files QTM auto-saved for this trial (the
    ones with the newest modification time, at/after trial_stop_time),
    filters to just the plates used (via each file's FORCE_PLATE_NAME
    header), and combines them into a single full-rate (e.g. 1000Hz)
    CSV with all raw variables (Force/Moment/COP in all 3 axes) for
    each plate, plus combined_cop_shared_mm computed with the SAME
    formula (including PLATE_OFFSET_MM) as the live game uses.

    game_start_wall_time is the wall-clock time.time() at which the
    actual game trial began (world_y=0, state="running") - used
    together with the TSV's own TIME_STAMP (the recording's absolute
    start time) to mark each row with whether it falls before
    (calibration/waiting) or at-or-after (the real running trial) the
    game's actual start, via a "game_started" column.

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
    header += ["combined_cop_shared_mm", "game_started", "phase"]

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
                + [round(combined, 4) if combined != "" else "", game_started,
                   phase_label(row_t_sec, recording_start_wall_time, phase_times)]
            )
    print(f"Saved combined 1000Hz CSV (all variables) to {out_path}")
    return out_path


def parse_skeleton_tsv(tsv_path):
    """Parses a QTM SKELETON TSV export file (the kind shown in your
    screenshots: FILE_VERSION/NO_OF_FRAMES/FREQUENCY/... header, then
    one header row "Frame  Time  Hips  X  Y  Z  QX  QY  QZ  QW  Spine
    X  Y  Z  ..." followed by one data row per frame).

    Returns (frequency_hz, recording_start_wall_time, segment_names,
    rows), where segment_names is the ordered list of segment names
    actually found in this file's header, and rows is a list of
    dicts: {"Frame": ..., "Time": ...,
    "<SegmentName>_X"/"_Y"/"_Z"/"_QX"/"_QY"/"_QZ"/"_QW": ...}, values
    as floats.

    The segment list (names, order, count) is read directly from THIS
    file's own header row rather than assumed - QTM's skeleton export
    can vary by model/config. Each segment's HEADER entry is expected
    to be (SegmentName, X, Y, Z, QX, QY, QZ, QW) - 8 tokens - if that
    pattern doesn't hold for a given file, this raises a ValueError
    naming exactly where the mismatch was found, instead of silently
    dropping every row.

    IMPORTANT (confirmed against a real _s_Q.tsv file, 2026-09-26):
    DATA rows have the SAME per-segment width as the header (8 tokens
    per segment) - NOT just the 7 numeric values. The first of those
    8 tokens is an EMPTY placeholder in data rows (where the header
    has the segment's name) - e.g. a real data row looks like
    "...\\t\\t327.250\\t441.748\\t979.240\\t0.004803\\t...", with the
    empty token right before Hips' X value. An earlier version of
    this function assumed data rows had only 7 tokens per segment (no
    placeholder), which meant the first placeholder (right after
    Frame/Time) was immediately fed to float(''), raising ValueError
    and silently skipping EVERY row via the try/except below - the
    100Hz export always came out with a header but zero data rows.
    Verified fix: parsing this exact file now yields exactly
    NO_OF_FRAMES rows, with real (non-zero) values once skeleton
    tracking locks on."""
    frequency_hz = None
    recording_start_wall_time = None

    with open(tsv_path, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("FREQUENCY"):
            frequency_hz = float(line.strip().split("\t")[1])
        if line.startswith("TIME_STAMP"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                date_str = parts[1].strip()
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d, %H:%M:%S.%f")
                    recording_start_wall_time = dt.timestamp()
                except ValueError:
                    recording_start_wall_time = None
        if line.startswith("Frame\tTime\t"):
            header_idx = i
            break

    if header_idx is None:
        return frequency_hz, recording_start_wall_time, [], []

    header_tokens = lines[header_idx].rstrip("\n").rstrip("\r").split("\t")
    # A trailing tab before the newline (seen in practice) produces
    # one or more empty trailing tokens after split() - strip those
    # so the segment-walk loop below doesn't mistake "" for a real
    # segment name.
    while header_tokens and header_tokens[-1] == "":
        header_tokens.pop()
    if header_tokens[:2] != ["Frame", "Time"]:
        raise ValueError(
            f"{tsv_path}: header row doesn't start with Frame/Time as expected: "
            f"{header_tokens[:2]!r}")

    # Walk the remaining HEADER tokens as repeating (SegmentName, X, Y,
    # Z, QX, QY, QZ, QW) groups of 8 - built from whatever this file
    # actually contains, not a hardcoded segment list.
    expected_suffixes = ["X", "Y", "Z", "QX", "QY", "QZ", "QW"]
    column_names = ["Frame", "Time"]
    segment_names_found = []
    i = 2
    while i < len(header_tokens):
        seg_name = header_tokens[i]
        group = header_tokens[i + 1:i + 8]
        if group != expected_suffixes:
            raise ValueError(
                f"{tsv_path}: expected segment '{seg_name}' at column {i} to be "
                f"followed by {expected_suffixes}, found {group}. This file's "
                f"skeleton layout doesn't match the assumed X/Y/Z/QX/QY/QZ/QW-per-"
                f"segment pattern - the header should be inspected directly.")
        column_names += [f"{seg_name}_{suf}" for suf in expected_suffixes]
        segment_names_found.append(seg_name)
        i += 8
    n_segments = len(segment_names_found)

    print(f"  Parsed {n_segments} segments from {os.path.basename(tsv_path)} "
          f"header: {segment_names_found}")

    rows = []
    for line in lines[header_idx + 1:]:
        raw_values = line.rstrip("\n").rstrip("\r").split("\t")
        while raw_values and raw_values[-1] == "":
            raw_values.pop()
        expected_len = 2 + n_segments * 8   # DATA rows mirror the header's
                                             # 8-tokens-per-segment width
        if len(raw_values) < expected_len:
            continue   # short/blank trailing line
        try:
            frame = float(raw_values[0])
            t_sec = float(raw_values[1])
            values = [frame, t_sec]
            idx = 2
            for _ in range(n_segments):
                idx += 1   # skip this segment's empty placeholder token
                values.extend(float(v) for v in raw_values[idx:idx + 7])
                idx += 7
        except ValueError:
            continue
        rows.append(dict(zip(column_names, values)))

    return frequency_hz, recording_start_wall_time, segment_names_found, rows




def export_100hz_com_after_trial(qtm_data_folder, trial_stop_time, game_start_wall_time, out_path,
                                 phase_times=None):
    """Finds the skeleton TSV QTM auto-saved for this trial (*_s_*.tsv,
    global coordinates, 100 Hz), recomputes whole-body CoM per frame
    with the SAME de Leva formula as the live stream, and writes
    frame/t_sec/com_x/y/z/com_valid/game_started/trial_start_marker.

    Unlike the CoM-trial version, frames where a segment is missing are
    KEPT with empty CoM and com_valid = 0, so the time base stays
    continuous and gaps are visible in the file.

    trial_start_marker = 1 on the first frame where game_started
    switches from 0 to 1."""
    all_tsvs = glob.glob(os.path.join(qtm_data_folder, "*_s_*.tsv"))
    recent_tsvs = [p for p in all_tsvs if os.path.getmtime(p) >= trial_stop_time - 2.0]
    recent_tsvs.sort(key=os.path.getmtime)

    if not recent_tsvs:
        print(f"\nNo new skeleton TSV file found in {qtm_data_folder!r} written "
              f"after this trial (trial_stop_time={trial_stop_time}).")
        print("Check that Export to TSV + Skeleton data type are enabled in QTM, and that")
        print("'Save captured and processed measurement automatically' is on.")
        return None

    if len(recent_tsvs) > 1:
        print(f"\nWarning: found {len(recent_tsvs)} candidate skeleton TSV files "
              f"{recent_tsvs} - using the most recently modified one.")
    tsv_path = recent_tsvs[-1]
    print(f"\nUsing skeleton TSV: {tsv_path}")

    frequency_hz, recording_start_wall_time, segment_names, rows = parse_skeleton_tsv(tsv_path)
    print(f"  {frequency_hz}Hz, {len(rows)} rows")

    game_start_offset_sec = None
    if recording_start_wall_time is not None and game_start_wall_time is not None:
        game_start_offset_sec = game_start_wall_time - recording_start_wall_time
        print(f"-> game_started at TIME={game_start_offset_sec:.3f}s into the recording")
    else:
        print("Warning: could not determine recording start time or game start time - "
              "'game_started'/'trial_start_marker' will be left blank for all rows.")

    header = ["frame", "t_sec", "com_x", "com_y", "com_z", "com_valid",
              "game_started", "trial_start_marker", "phase"]

    prev_game_started = 0
    n_valid = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            positions = {seg: (row[f"{seg}_X"], row[f"{seg}_Y"], row[f"{seg}_Z"]) for seg in segment_names}
            com = compute_whole_body_com(positions)

            row_t_sec = row["Time"]
            if game_start_offset_sec is not None:
                game_started = int(row_t_sec >= game_start_offset_sec)
            else:
                game_started = ""

            trial_start_marker = 0
            if game_started == 1 and prev_game_started == 0:
                trial_start_marker = 1
            if game_started != "":
                prev_game_started = game_started

            if com is None:
                com_cols = ["", "", "", 0]
            else:
                com_cols = [round(com[0], 4), round(com[1], 4), round(com[2], 4), 1]
                n_valid += 1

            writer.writerow([int(row["Frame"]), round(row_t_sec, 6)] + com_cols
                            + [game_started, trial_start_marker,
                               phase_label(row_t_sec, recording_start_wall_time, phase_times)])

    print(f"Saved 100Hz CoM CSV ({len(rows)} frames, {n_valid} with valid CoM) to {out_path}")
    return out_path


#  MAIN
def fmt(v, nd=4):
    """CSV-friendly number formatting; None/NaN -> empty cell."""
    if v is None:
        return ""
    try:
        if not np.isfinite(v):
            return ""
    except TypeError:
        pass
    return f"{v:.{nd}f}"



# ---------------------------------------------------------------------
#  Calibration parameters: find the latest calibration summary files of
#  this person and read the values this script needs (CAL_PARAM_MAP).
# ---------------------------------------------------------------------
def read_cal_summary(path):
    """Key/value lines at the bottom of a calibration summary CSV ->
    {key: float}. The key is the text before ' (' so
    'fft_peak_width_sd_hz (Aaron's 5%-95% method)' -> 'fft_peak_width_sd_hz'."""
    values = {}
    with open(path, "r", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip():
                continue
            key = row[0].split(" (")[0].strip()
            try:
                values[key] = float(row[1])
            except ValueError:
                continue
    return values


def find_latest_cal_summary(data_root, subject, group, controller, task_tag):
    """Newest calibration summary of this person and controller for one task,
    e.g. S015_AMP_CoP_CALAMP_T01_summary.csv. The group written in the
    calibration file name is NOT required to match (a calibration may have
    been saved under the other group); a file with the same group is
    preferred if one exists. Among the candidates the highest T number,
    then the newest file, is used. Looks in data_root/<subject> first,
    then anywhere under data_root."""
    pattern = re.compile(rf"^{re.escape(subject)}_([A-Za-z]+)_{re.escape(controller)}_"
                         rf"{task_tag}_T(\d+)_summary\.csv$")
    found = []
    folders = [os.path.join(data_root, subject)]
    folders += [d for d, _s, _f in os.walk(data_root) if d not in folders]
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            m = pattern.match(name)
            if m:
                path = os.path.join(folder, name)
                found.append((m.group(1) == group, int(m.group(2)), os.path.getmtime(path), path))
        if found:
            break
    return max(found)[3] if found else None


def load_calibration_params(data_root, subject, group, controller):
    """Returns (values, sources, error). values = {setting: number},
    sources = {task: file path}. error is a message if something is missing."""
    sources, tables = {}, {}
    for task in sorted({task for task, _key in CAL_PARAM_MAP.values()}):
        path = find_latest_cal_summary(data_root, subject, group, controller, task)
        if path is None:
            return None, sources, (f"No {task} calibration found for {subject} {controller}.|"
                                   f"Looking for {subject}_<group>_{controller}_{task}_Txx_summary.csv|"
                                   f"under {data_root}")
        sources[task] = path
        tables[task] = read_cal_summary(path)
    values = {}
    for setting, (task, key) in CAL_PARAM_MAP.items():
        if key not in tables[task]:
            return None, sources, f"'{key}' is missing in {os.path.basename(sources[task])}."
        values[setting] = tables[task][key]
    return values, sources, None


def params_screen(screen, title, lines, error=False):
    """Shows the loaded values (or an error). SPACE = continue, ESC = quit.
    Returns True to continue, False to quit."""
    w, h = screen.get_size()
    big = pygame.font.SysFont("arial", max(24, h // 22))
    small = pygame.font.SysFont("arial", max(16, h // 40))
    footer = "ESC = quit" if error else "SPACE = values are correct, start    |    ESC = quit"
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE and not error:
                    return True
        screen.fill((20, 24, 32))
        t = big.render(title, True, (230, 120, 100) if error else (255, 220, 100))
        screen.blit(t, (w // 2 - t.get_width() // 2, int(h * 0.18)))
        y = int(h * 0.32)
        for line in lines:
            s = small.render(line, True, (220, 220, 220))
            screen.blit(s, (w // 2 - s.get_width() // 2, y))
            y += int(small.get_height() * 1.5)
        f = small.render(footer, True, (150, 200, 255))
        screen.blit(f, (w // 2 - f.get_width() // 2, int(h * 0.85)))
        pygame.display.flip()
        time.sleep(0.02)


def phase_label(row_t_sec, recording_start_wall_time, phase_times):
    """'quiet_pre' / 'game' / 'quiet_post' for a QTM row, or '' outside them."""
    if recording_start_wall_time is None or not phase_times:
        return ""
    wall = recording_start_wall_time + row_t_sec
    for name, (start, end) in phase_times.items():
        if start is not None and end is not None and start <= wall < end:
            return name
    return ""


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

    pygame.display.set_caption(f"{SCRIPT_NAME} - {MANIPULATION} pathway, {CONTROLLER} control")

    # ---------------- start screen: Subject ID + Group ----------------
    info = get_session_info(screen, title=f"Fractal Balance Game - {MANIPULATION} trial ({CONTROLLER} control)",
                            expected_group=MANIPULATION)
    if info is None:
        pygame.quit()
        sys.exit()
    subject, group = info["subject"], info["group"]
    quiet_pre_sec = float(info.get("quiet_pre_sec", QUIET_PRE_SEC))
    quiet_post_sec = float(info.get("quiet_post_sec", QUIET_POST_SEC))
    print(f"Quiet standing: {quiet_pre_sec:.0f} s before, {quiet_post_sec:.0f} s after the game")

    save_dir = os.path.join(DATA_ROOT, subject)
    stem = make_trial_stem(save_dir, subject, group, CONTROLLER, HURST)
    game_path = os.path.join(save_dir, f"{stem}_game60hz.csv")
    fp_path = os.path.join(save_dir, f"{stem}_fp1000hz.csv")
    mocap_path = os.path.join(save_dir, f"{stem}_mocap100hz.csv")
    plot_path = os.path.join(save_dir, f"{stem}_pathway.png")
    meta_path = os.path.join(save_dir, f"{stem}_meta.json")
    print(f"Subject {subject}, group {group} -> files: {save_dir}\\{stem}_*")

    # ---------------- person-specific values from the calibration files ----------------
    params_source = None
    if LOAD_PARAMS_FROM_CALIBRATION:
        values, sources, error = load_calibration_params(DATA_ROOT, subject, group, CONTROLLER)
        if error:
            print("ERROR: " + error.replace("|", " "))
            params_screen(screen, "Calibration values not found", error.split("|") + ["",
                          "Run the calibration first, or set LOAD_PARAMS_FROM_CALIBRATION = False."], error=True)
            pygame.quit()
            sys.exit()
        globals().update(values)
        lines = [f"{k} = {v:g}" for k, v in values.items()]
        lines += [""] + [f"{task}: {os.path.basename(p)}" for task, p in sources.items()]
        print("Values from calibration:")
        for line in lines:
            print("  " + line)
        if not params_screen(screen, f"{subject} {group} {CONTROLLER} - values from calibration", lines):
            pygame.quit()
            sys.exit()
        params_source = {"values": values, "files": {k: os.path.basename(p) for k, p in sources.items()},
                         "map": {k: list(v) for k, v in CAL_PARAM_MAP.items()}}

    scale = HEIGHT / REF_H
    BALL_RADIUS    = max(10, int(BASE_BALL_RADIUS * scale))
    CENTER_DOT_RADIUS = max(2, int(BASE_CENTER_DOT_RADIUS * scale))
    AUTO_SPEED     = BASE_AUTO_SPEED * scale
    FONT_SIZE      = max(14, int(BASE_FONT_SIZE * scale))
    LINE_WIDTH     = max(3, int(BASE_LINE_WIDTH * scale))
    BALL_Y         = int(HEIGHT * BALL_Y_FRACTION)

    # Same mm<->px scale for the pathway and the CoP-driven ball.
    mm_to_px_ratio = WIDTH / (2.0 * COP_RANGE_MM)
    NUM_CYCLES = max(2, round(FIXED_FREQUENCY_HZ * TRIAL_DURATION_SEC))

    # ---------------- visible window: VISIBLE_CYCLES of one average cycle ----------------
    # On-screen height of one average cycle = scroll speed (px/s) x average period (s).
    # NUM_CYCLES cycles fill TRIAL_DURATION_SEC, so the average period is
    # TRIAL_DURATION_SEC / NUM_CYCLES.
    mean_cycle_px = AUTO_SPEED * FPS * WAVELENGTH_STRETCH * TRIAL_DURATION_SEC / NUM_CYCLES
    window_px = min(HEIGHT, VISIBLE_CYCLES * mean_cycle_px)
    WINDOW_AHEAD_PX = int(round(FRACTION_AHEAD * window_px))
    WINDOW_BEHIND_PX = int(round((1 - FRACTION_AHEAD) * window_px))
    print(f"One average cycle on screen = {mean_cycle_px:.0f} px; visible window = {window_px:.0f} px "
          f"({VISIBLE_CYCLES} cycle: {WINDOW_AHEAD_PX} px ahead of the ball, {WINDOW_BEHIND_PX} px behind)")
    EFFECTIVE_FREQUENCY_HZ = NUM_CYCLES / TRIAL_DURATION_SEC

    print(f"Screen: {WIDTH}x{HEIGHT}, Scale: {scale:.2f}, px per mm: {mm_to_px_ratio:.4f}")
    print(f"Fixed frequency: {FIXED_FREQUENCY_HZ}Hz (effective {EFFECTIVE_FREQUENCY_HZ:.4f}Hz) "
          f"-> {NUM_CYCLES} cycles in {TRIAL_DURATION_SEC}s")
    print(f"Amplitude: mean={AMP_MEAN_MM:.2f}mm, sd={AMP_SD_MM:.2f}mm")

    # ---------------- pathway (fGn amplitudes, H-matched) ----------------
    print(f"Searching for a pathway with realized H within +/-{H_TOLERANCE} of target {HURST}...")
    t0 = time.time()
    amp_values_mm, best_estimated_H, n_attempts, found_within_tolerance = generate_amplitude_pathway_within_tolerance(
        n_cycles=NUM_CYCLES,
        target_H=HURST,
        amp_mean_mm=AMP_MEAN_MM,
        amp_sd_mm=AMP_SD_MM,
        tolerance=H_TOLERANCE,
        max_attempts=MAX_CANDIDATE_ATTEMPTS,
    )
    search_elapsed = time.time() - t0
    if found_within_tolerance:
        print(f"  Found after {n_attempts} attempt(s) in {search_elapsed:.1f}s. "
              f"Realized H (HK) = {best_estimated_H:.4f} (target {HURST}).")
    else:
        print(f"  WARNING: gave up after {n_attempts} attempts ({search_elapsed:.1f}s) - using the closest "
              f"candidate: realized H = {best_estimated_H:.4f} (off by {abs(best_estimated_H - HURST):.4f}).")
    n_clipped = int(np.sum(amp_values_mm <= 5.0))
    if n_clipped > 0:
        print(f"  NOTE: {n_clipped} cycle(s) were floored to 5mm (safety bound).")
    print(f"Amplitude pathway realized range: [{amp_values_mm.min():.2f}, {amp_values_mm.max():.2f}]mm")
    if amp_values_mm.max() > COP_RANGE_MM:
        print(f"  WARNING: largest cycle amplitude ({amp_values_mm.max():.0f}mm) exceeds COP_RANGE_MM "
              f"({COP_RANGE_MM:.0f}mm) - the target will go past the ball's reachable range.")

    amp_values_px = amp_values_mm * mm_to_px_ratio

    save_amplitude_debug_plot(
        amp_values_mm=amp_values_mm, hurst=HURST, amp_mean_mm=AMP_MEAN_MM, amp_sd_mm=AMP_SD_MM,
        fixed_freq_hz=EFFECTIVE_FREQUENCY_HZ, auto_speed=AUTO_SPEED, fps=FPS, seed=RNG_SEED,
        out_path=plot_path,
    )
    print(f"  Pathway plot saved to {plot_path}")

    amp_map = CycleAmplitudeMap(EFFECTIVE_FREQUENCY_HZ, NUM_CYCLES, AUTO_SPEED, FPS)

    def get_path_center_x(world_y):
        # world_y is TIME-based (AUTO_SPEED * FPS * seconds). No stretch
        # here - see WAVELENGTH_STRETCH (applied only when drawing).
        return WIDTH // 2 + amp_map.get_amplitude(world_y, amp_values_px) * math.sin(amp_map.get_phase(world_y))

    # ---------------- QTM ----------------
    qtm = QTMCopComInput(QTM_IP, QTM_PLATE_IDS, FZ_THRESHOLD_N, COP_SMOOTH_ALPHA, PLATE_OFFSET_MM,
                         qtm_data_folder=QTM_DATA_FOLDER)
    qtm.start(WIDTH)

    # ---------------- game log (60 Hz) ----------------
    fields = [
        "frame", "t_sec", "wall_time", "trial_state", "controller", "loaded",
        # CoP (force plates)
        "fz_left", "ml_left", "ap_left", "fz_right", "ml_right", "ap_right", "fz_total",
        "cop_ml_mm", "cop_ml_centered_mm",
        # CoM (motion capture, global lab coordinates)
        "com_x_mm", "com_y_mm", "com_z_mm", "com_valid", "com_frozen", "com_ml_centered_mm",
        # what actually moved the ball (smoothed, centered CoP)
        "control_signal_smoothed_mm",
        # ball / target
        "ball_x_px", "path_cx_px", "offset_px", "ball_mm", "target_mm", "offset_mm",
        # pathway
        "world_y", "cycle_idx", "cycle_amplitude_mm", "current_amplitude_mm", "hurst", "fixed_frequency_hz",
    ]
    csvfile = open(game_path, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    clock = pygame.time.Clock()
    ARIAL_PATH = r"C:\Windows\Fonts\arial.ttf"
    try:
        font = pygame.font.Font(ARIAL_PATH, FONT_SIZE)
        font_big = pygame.font.Font(ARIAL_PATH, max(20, int(28 * scale)))
        font_instructions = pygame.font.Font(ARIAL_PATH, max(22, int(30 * scale)))
    except FileNotFoundError:
        print(f"Warning: could not find {ARIAL_PATH} - falling back to SysFont.")
        font = pygame.font.SysFont("arial", FONT_SIZE)
        font_big = pygame.font.SysFont("arial", max(20, int(28 * scale)))
        font_instructions = pygame.font.SysFont("arial", max(22, int(30 * scale)))

    world_y = 0.0
    ball_x = WIDTH / 2.0
    frame_count = 0

    # instructions -> waiting -> ready_for_qtm_trigger -> ready_for_calibration
    #   -> calibrating -> ready_for_game_start -> running -> done
    # SPACE (instructions) starts the flow; SPACE pressed in the QTM
    # window is QTM's own external trigger. ENTER = calibration, S = start.
    trial_state = "instructions"
    calib_start_wall = calib_end_wall = None
    quiet_pre_start_wall = quiet_pre_end_wall = None
    game_end_wall_time = quiet_post_start_wall = quiet_post_end_wall = None
    trial_start_time = None
    game_start_wall_time = None
    trial_elapsed = 0.0
    cycle_idx = 0
    completed = False

    # QC counters (running frames only)
    qc = {"frames_com_invalid": 0, "frames_com_frozen": 0, "frames_low_fz": 0}

    instruction_lines = [
        "Stand where the cameras can see your full body, feet shoulder-width apart.",
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
                        if trial_state != "quiet_post":   # no early exit while standing after the game
                            running = False
                    elif event.key == pygame.K_SPACE:
                        if trial_state == "instructions":
                            trial_state = "waiting"
                    elif event.key == pygame.K_RETURN:
                        if trial_state == "ready_for_calibration":
                            trial_state = "calibrating"
                            print(f"Calibrating for {CALIBRATION_SECONDS}s - stand still...")
                            screen.fill(BG_COLOR)
                            _m = font_big.render("Calibrating - stand still...", True, (255, 220, 100))
                            screen.blit(_m, (WIDTH // 2 - _m.get_width() // 2, HEIGHT // 2 - int(20 * scale)))
                            pygame.display.flip()
                            calib_start_wall = time.time()
                            qtm.calibrate_center(CALIBRATION_SECONDS)
                            calib_end_wall = time.time()
                            print("Calibration done.")
                            if quiet_pre_sec > 0:
                                trial_state = "quiet_pre"
                                quiet_pre_start_wall = time.time()
                                print(f"Quiet standing before the game: {quiet_pre_sec:.0f} s (screen blank)...")
                            else:
                                trial_state = "ready_for_game_start"
                    elif event.key == pygame.K_s:
                        if trial_state == "ready_for_game_start":
                            trial_state = "running"
                            trial_start_time = time.perf_counter()
                            game_start_wall_time = time.time()
                            world_y = 0.0
                            print("Trial started!")

            if trial_state == "waiting":
                ball_x = WIDTH / 2.0
                if qtm.loaded:
                    trial_state = "ready_for_qtm_trigger"
                    print("Plate loaded! Press SPACE now (QTM's external trigger) to start recording.")

            elif trial_state == "ready_for_qtm_trigger":
                ball_x = WIDTH / 2.0
                if qtm.loaded:
                    trial_state = "ready_for_calibration"
                    print("Recording started! Press ENTER to begin calibration.")

            elif trial_state in ("ready_for_calibration", "ready_for_game_start"):
                ball_x = WIDTH / 2.0

            elif trial_state == "quiet_pre":
                if time.time() - quiet_pre_start_wall >= quiet_pre_sec:
                    trial_state = "ready_for_game_start"
                    quiet_pre_end_wall = time.time()
                    print("Quiet standing before the game done. Press S to start the game.")

            elif trial_state == "running":
                ball_x = qtm.get_control_x_px(WIDTH, BALL_RADIUS, COP_RANGE_MM, EDGE_MARGIN_PX)
                trial_elapsed = time.perf_counter() - trial_start_time
                world_y = AUTO_SPEED * FPS * trial_elapsed
                if trial_elapsed >= TRIAL_DURATION_SEC:
                    trial_state = "quiet_post" if quiet_post_sec > 0 else "done"
                    completed = True
                    game_end_wall_time = quiet_post_start_wall = time.time()
                    if quiet_post_sec <= 0:
                        quiet_post_start_wall = None
                    if quiet_post_sec > 0:
                        print(f"Game complete! {TRIAL_DURATION_SEC}s elapsed. "
                              f"Quiet standing after the game: {quiet_post_sec:.0f} s (screen blank)...")
                    else:
                        print(f"Trial complete! {TRIAL_DURATION_SEC}s elapsed. Press ESC to save and exit.")

            elif trial_state == "quiet_post":
                if time.time() - quiet_post_start_wall >= quiet_post_sec:
                    trial_state = "done"
                    quiet_post_end_wall = time.time()
                    print("Quiet standing done. Press ESC to save and exit.")

            path_cx = get_path_center_x(world_y)
            offset = ball_x - path_cx
            cycle_idx = amp_map.get_cycle_index(world_y)
            cycle_amp_mm = amp_values_mm[cycle_idx]
            # smoothly interpolated amplitude actually drawn at this instant
            current_amp_mm = amp_map.get_amplitude(world_y, amp_values_mm)

            qc_now = []
            if trial_state == "running":
                snap = qtm.snapshot()
                raw = snap["raw"]
                com = snap["com"]
                left_pid, right_pid = QTM_PLATE_IDS[0], QTM_PLATE_IDS[1]

                if not snap["com_valid"]:
                    qc["frames_com_invalid"] += 1
                    qc_now.append("CoM missing")
                elif snap["com_frozen"]:
                    qc["frames_com_frozen"] += 1
                    qc_now.append("CoM frozen")
                if snap["fz_total"] < STEP_OFF_FZ_N:
                    qc["frames_low_fz"] += 1
                    qc_now.append("low force / off plate")

                writer.writerow({
                    "frame": frame_count,
                    "t_sec": f"{trial_elapsed:.6f}",
                    "wall_time": f"{time.time():.6f}",
                    "trial_state": trial_state,
                    "controller": CONTROLLER,
                    "loaded": int(qtm.loaded),
                    "fz_left": fmt(raw[left_pid]["fz"]),
                    "ml_left": fmt(raw[left_pid]["ml"]),
                    "ap_left": fmt(raw[left_pid]["ap"]),
                    "fz_right": fmt(raw[right_pid]["fz"]),
                    "ml_right": fmt(raw[right_pid]["ml"]),
                    "ap_right": fmt(raw[right_pid]["ap"]),
                    "fz_total": fmt(snap["fz_total"]),
                    "cop_ml_mm": fmt(snap["cop_ml_mm"]),
                    "cop_ml_centered_mm": fmt(snap["cop_ml_centered_mm"]),
                    "com_x_mm": fmt(com[0] if com else None),
                    "com_y_mm": fmt(com[1] if com else None),
                    "com_z_mm": fmt(com[2] if com else None),
                    "com_valid": int(snap["com_valid"]),
                    "com_frozen": int(snap["com_frozen"]),
                    "com_ml_centered_mm": fmt(snap["com_ml_centered_mm"]),
                    "control_signal_smoothed_mm": fmt(qtm.control_signal_smoothed_mm),
                    "ball_x_px": fmt(ball_x),
                    "path_cx_px": fmt(path_cx),
                    "offset_px": fmt(offset),
                    "ball_mm": fmt((ball_x - WIDTH / 2.0) / mm_to_px_ratio),
                    "target_mm": fmt((path_cx - WIDTH / 2.0) / mm_to_px_ratio),
                    "offset_mm": fmt(offset / mm_to_px_ratio),
                    "world_y": fmt(world_y),
                    "cycle_idx": cycle_idx,
                    "cycle_amplitude_mm": fmt(cycle_amp_mm),
                    "current_amplitude_mm": fmt(current_amp_mm),
                    "hurst": HURST,
                    "fixed_frequency_hz": fmt(EFFECTIVE_FREQUENCY_HZ, 6),
                })
                frame_count += 1

            # --- draw ---
            screen.fill(BG_COLOR)

            if trial_state in ("instructions", "waiting", "ready_for_qtm_trigger",
                                "ready_for_calibration", "calibrating", "ready_for_game_start"):
                if trial_state == "instructions":
                    y_start = HEIGHT // 2 - int(120 * scale)
                    line_spacing = int(44 * scale)
                    for i, line in enumerate(instruction_lines):
                        if not line:
                            continue
                        msg = font_instructions.render(line, True, (220, 220, 220))
                        screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, y_start + i * line_spacing))

                elif trial_state == "waiting":
                    msg = font_big.render("Step on the force plates to begin", True, (255, 220, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(20 * scale)))

                elif trial_state == "ready_for_qtm_trigger":
                    msg = font_big.render("Plate loaded!", True, (100, 255, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(60 * scale)))
                    msg2 = font_instructions.render("Press SPACE now to start recording in QTM", True, (220, 220, 220))
                    screen.blit(msg2, (WIDTH // 2 - msg2.get_width() // 2, HEIGHT // 2 + int(10 * scale)))

                elif trial_state == "ready_for_calibration":
                    msg = font_big.render("Recording started!", True, (100, 255, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(60 * scale)))
                    msg2 = font_instructions.render("Press ENTER to begin calibration", True, (220, 220, 220))
                    screen.blit(msg2, (WIDTH // 2 - msg2.get_width() // 2, HEIGHT // 2 + int(10 * scale)))

                elif trial_state == "calibrating":
                    msg = font_big.render("Calibrating - stand still...", True, (255, 220, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(20 * scale)))

                elif trial_state == "ready_for_game_start":
                    msg = font_big.render("Calibration complete!", True, (100, 255, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(60 * scale)))
                    msg2 = font_instructions.render("Press S to start the game", True, (220, 220, 220))
                    screen.blit(msg2, (WIDTH // 2 - msg2.get_width() // 2, HEIGHT // 2 + int(10 * scale)))
                    if qtm.com_height_mm is not None and qtm.com_height_mm < COM_HEIGHT_MIN_PLAUSIBLE_MM:
                        warn = font.render(f"Check mocap: CoM height {qtm.com_height_mm:.0f} mm looks wrong",
                                           True, (230, 120, 100))
                        screen.blit(warn, (WIDTH // 2 - warn.get_width() // 2, HEIGHT // 2 + int(70 * scale)))

                # Experimenter status line: QTM connection + whether the
                # skeleton (CoM) is being received.
                skel = "skeleton OK" if qtm.skeleton_seen else "NO skeleton yet"
                status_msg = font_instructions.render(f"QTM status: {qtm.status_text}  |  {skel}",
                                                      True, (150, 200, 255))
                screen.blit(status_msg, (WIDTH // 2 - status_msg.get_width() // 2, HEIGHT - int(60 * scale)))

            elif trial_state in ("quiet_pre", "quiet_post"):
                pass   # blank screen while the person stands still after the game

            else:
                # Path line. Each screen row sy shows the path at the time
                # it will reach the ball: the screen offset (BALL_Y - sy)
                # is divided by WAVELENGTH_STRETCH, so one cycle is drawn
                # WAVELENGTH_STRETCH times taller without changing timing.
                step = 2
                pts = []
                for sy in range(-20, HEIGHT + 20, step):
                    wy = world_y + (BALL_Y - sy) / WAVELENGTH_STRETCH
                    pts.append((get_path_center_x(wy), sy))
                draw_smooth_line(screen, LINE_COLOR, pts, LINE_WIDTH)

                window_top = BALL_Y - WINDOW_AHEAD_PX      # above the ball = path still to come
                window_bottom = BALL_Y + WINDOW_BEHIND_PX  # below the ball = path already passed
                if window_top > 0:
                    pygame.draw.rect(screen, BG_COLOR, (0, 0, WIDTH, window_top))
                if window_bottom < HEIGHT:
                    pygame.draw.rect(screen, BG_COLOR, (0, window_bottom, WIDTH, HEIGHT - window_bottom))

                pygame.draw.circle(screen, BALL_COLOR, (int(ball_x), BALL_Y), BALL_RADIUS)
                pygame.draw.circle(screen, CENTER_DOT_COLOR, (int(ball_x), BALL_Y), CENTER_DOT_RADIUS)

                if SHOW_QC_WARNINGS_ON_SCREEN and qc_now:
                    w_msg = font.render(" | ".join(qc_now), True, (200, 110, 90))
                    screen.blit(w_msg, (int(10 * scale), HEIGHT - w_msg.get_height() - int(10 * scale)))

                if trial_state == "done":
                    msg = font_big.render("Trial Complete!", True, (100, 255, 100))
                    screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - int(40 * scale)))
                    n = max(1, frame_count)
                    summary = (f"{frame_count} frames  |  CoM missing {100 * qc['frames_com_invalid'] / n:.1f}%  |  "
                               f"CoM frozen {100 * qc['frames_com_frozen'] / n:.1f}%  |  "
                               f"low force {100 * qc['frames_low_fz'] / n:.1f}%  |  Esc = save & exit")
                    s_msg = font.render(summary, True, (180, 180, 200))
                    screen.blit(s_msg, (WIDTH // 2 - s_msg.get_width() // 2, HEIGHT // 2 + int(10 * scale)))

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        qtm.stop()
        time.sleep(0.2)
        csvfile.flush()
        csvfile.close()
        pygame.quit()

    print(f"\nSaved game log: {game_path}")
    print(f"  Frames logged: {frame_count}, last cycle index: {cycle_idx} of {NUM_CYCLES - 1}")
    n = max(1, frame_count)
    for k, v in qc.items():
        print(f"  QC {k}: {v} ({100 * v / n:.1f}%)")

    # ---------------- full-rate exports from QTM ----------------
    fp_ok = mocap_ok = None
    phase_times = {"calibration": (calib_start_wall, calib_end_wall),
                   "quiet_pre": (quiet_pre_start_wall, quiet_pre_end_wall),
                   "game": (game_start_wall_time, game_end_wall_time),
                   "quiet_post": (quiet_post_start_wall, quiet_post_end_wall)}
    qtm_sources = []
    qtm_archived = {}
    if qtm.trial_stop_time is not None:
        qtm_sources = list_recent_qtm_files(QTM_DATA_FOLDER, qtm.trial_stop_time)
        fp_ok = export_1000hz_after_trial(
            qtm_data_folder=QTM_DATA_FOLDER, plate_ids=QTM_PLATE_IDS, plate_offset_mm=PLATE_OFFSET_MM,
            trial_stop_time=qtm.trial_stop_time, game_start_wall_time=game_start_wall_time, out_path=fp_path,
            phase_times=phase_times,
        )
        mocap_ok = export_100hz_com_after_trial(
            qtm_data_folder=QTM_DATA_FOLDER, trial_stop_time=qtm.trial_stop_time,
            game_start_wall_time=game_start_wall_time, out_path=mocap_path,
            phase_times=phase_times,
        )
        qtm_archived = archive_qtm_tsv(qtm_sources, save_dir, stem)
    else:
        print("\nWarning: trial_stop_time was never set - skipping 1000Hz/100Hz exports.")

    # ---------------- metadata ----------------
    meta = {
        "script": SCRIPT_NAME, "script_version": SCRIPT_VERSION,
        "subject": subject, "group": group, "manipulation": MANIPULATION, "controller": CONTROLLER,
        "trial_stem": stem, "completed_full_duration": completed,
        "game_start_wall_time": game_start_wall_time,
        "game_start_local": datetime.fromtimestamp(game_start_wall_time).isoformat() if game_start_wall_time else None,
        "trial_duration_sec": TRIAL_DURATION_SEC, "frames_logged": frame_count,
        "hurst_target": HURST, "hurst_realized_hk": float(best_estimated_H), "hurst_estimator": "HK maximum likelihood, grid step 0.005", "hurst_estimated_on": "cycle amplitudes",
        "hurst_found_within_tolerance": bool(found_within_tolerance), "hurst_tolerance": H_TOLERANCE,
        "pathway_attempts": int(n_attempts),
        "num_cycles": NUM_CYCLES, "last_cycle_reached": int(cycle_idx),
        "fixed_frequency_hz_target": FIXED_FREQUENCY_HZ, "fixed_frequency_hz_effective": EFFECTIVE_FREQUENCY_HZ,
        "amp_mean_mm": AMP_MEAN_MM, "amp_sd_mm": AMP_SD_MM,
        "amp_values_mm": [round(float(v), 4) for v in amp_values_mm],
        "cop_range_mm": COP_RANGE_MM, "px_per_mm": mm_to_px_ratio,
        "screen_px": [WIDTH, HEIGHT], "wavelength_stretch": WAVELENGTH_STRETCH, "visible_cycles": VISIBLE_CYCLES, "fraction_ahead": FRACTION_AHEAD, "window_ahead_px": WINDOW_AHEAD_PX, "window_behind_px": WINDOW_BEHIND_PX, "mean_cycle_px": mean_cycle_px,
        "cop_smooth_alpha": COP_SMOOTH_ALPHA,
        "cop_center_mm": qtm.cop_center_mm, "com_center_mm": qtm.com_center_mm,
        "com_height_mm": qtm.com_height_mm, "body_weight_n": qtm.body_weight_n,
        "body_mass_kg_est": (qtm.body_weight_n / 9.81) if qtm.body_weight_n else None,
        "plate_ids": QTM_PLATE_IDS, "plate_offset_mm": {str(k): v for k, v in PLATE_OFFSET_MM.items()},
        "qc": qc,
        "params_loaded_from_calibration": bool(LOAD_PARAMS_FROM_CALIBRATION), "params_source": params_source,
        "calibration_sec": CALIBRATION_SECONDS,
        "quiet_pre_sec_set": quiet_pre_sec, "quiet_post_sec_set": quiet_post_sec,
        "phases_wall_time": {k: list(v) for k, v in phase_times.items()},
        "phases_sec_from_game_start": {k: [None if (x is None or game_start_wall_time is None)
                                           else round(x - game_start_wall_time, 3) for x in v]
                                       for k, v in phase_times.items()},
        "files": {
            "game60hz": os.path.basename(game_path),
            "fp1000hz": os.path.basename(fp_path) if fp_ok else None,
            "mocap100hz": os.path.basename(mocap_path) if mocap_ok else None,
            "pathway_plot": os.path.basename(plot_path),
            "qtm_source_tsv": qtm_sources,
            "qtm_archived_tsv": sorted(qtm_archived.values()),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== FILES FOR THIS TRIAL ===")
    print(f"  game 60 Hz    : {'OK' if os.path.exists(game_path) else 'MISSING'}  {game_path}")
    print(f"  force 1000 Hz : {'OK' if fp_ok else 'MISSING'}  {fp_path}")
    print(f"  mocap 100 Hz  : {'OK' if mocap_ok else 'MISSING'}  {mocap_path}")
    print(f"  metadata      : {meta_path}")
    skel = [v for v in qtm_archived.values() if v.endswith("_qtm_skeleton.tsv")]
    print(f"  raw QTM TSV   : {len(qtm_archived)} file(s) archived"
          f"{' (skeleton OK)' if skel else ' - NO SKELETON TSV: check QTM export settings'}")
    for v in sorted(qtm_archived.values()):
        print(f"                  {v}")


if __name__ == "__main__":
    LAUNCH_INFO, _launch_hurst = parse_launcher_args()
    if _launch_hurst is not None and HURST is not None:
        if not 0.0 < _launch_hurst < 1.0:
            print(f"--hurst must be between 0 and 1, got {_launch_hurst}")
            sys.exit(2)
        HURST = _launch_hurst
    if LAUNCH_INFO is not None and LAUNCH_INFO.get("range_mm") is not None:
        if not 20.0 <= LAUNCH_INFO["range_mm"] <= 600.0:
            print(f"--range must be between 20 and 600 mm, got {LAUNCH_INFO['range_mm']}")
            sys.exit(2)
        COP_RANGE_MM = LAUNCH_INFO["range_mm"]   # participant's sensitivity from the launcher
        print(f"Sensitivity (COP_RANGE_MM) set by the launcher: {COP_RANGE_MM:g} mm")
    main()
