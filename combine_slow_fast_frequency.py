"""
Helper script: combine the SLOW and FAST preferred-frequency task
summary CSVs into one final mean/SD, for use in the
frequency-variable trial's FREQ_MEAN_HZ and FREQ_SD_HZ.

Why two trials: a single "whatever feels comfortable" 30s trial
tends to have a narrow SD (the person naturally settles into one
steady pace). Explicitly asking for a slower pace and a faster pace
in two separate trials captures this person's real usable range,
giving a more meaningful mean/SD for the frequency-variable trial.

Usage:
    python combine_slow_fast_frequency.py slow_summary.csv fast_summary.csv

What it does:
    1. Reads the per-cycle "frequency_hz" column from both files,
       skipping any row marked excluded_from_mean=yes (same cycle-1
       exclusion logic as the individual tasks).
    2. Pools ALL cycles from both files together (not two separate
       means averaged - a single pooled mean/SD across every valid
       cycle from both the slow and fast trials).
    3. Prints the combined mean and SD, ready to copy into
       FREQ_MEAN_HZ / FREQ_SD_HZ in the frequency-variable trial.
"""

import sys
import csv
import os
import statistics


def read_valid_frequencies(filepath):
    """Returns a list of frequency_hz values from rows NOT marked
    excluded_from_mean=yes."""
    freqs = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        if "frequency_hz" not in reader.fieldnames:
            print(f"  WARNING: {os.path.basename(filepath)} has no "
                  f"'frequency_hz' column - skipping.")
            return freqs

        for row in reader:
            raw_freq = row.get("frequency_hz", "")
            if raw_freq == "":
                continue
            excluded = row.get("excluded_from_mean", "no").strip().lower()
            if excluded == "yes":
                continue
            try:
                freqs.append(float(raw_freq))
            except ValueError:
                continue
    return freqs


def main():
    if len(sys.argv) < 3:
        print("Usage: python combine_slow_fast_frequency.py slow_summary.csv fast_summary.csv")
        sys.exit(1)

    slow_path, fast_path = sys.argv[1], sys.argv[2]

    for p in (slow_path, fast_path):
        if not os.path.isfile(p):
            print(f"File not found: {p}")
            sys.exit(1)

    slow_freqs = read_valid_frequencies(slow_path)
    fast_freqs = read_valid_frequencies(fast_path)

    print(f"Slow trial ({os.path.basename(slow_path)}): {len(slow_freqs)} valid cycles")
    print(f"  frequencies: {[round(f, 3) for f in slow_freqs]}")
    print(f"Fast trial ({os.path.basename(fast_path)}): {len(fast_freqs)} valid cycles")
    print(f"  frequencies: {[round(f, 3) for f in fast_freqs]}")
    print()

    all_freqs = slow_freqs + fast_freqs
    if len(all_freqs) < 2:
        print("Not enough valid cycles across both files to compute a meaningful mean/SD.")
        sys.exit(1)

    mean_freq = statistics.mean(all_freqs)
    sd_freq = statistics.stdev(all_freqs)

    print("=" * 60)
    print(f"Pooled across both trials: {len(all_freqs)} cycles total")
    print(f"Combined mean frequency: {mean_freq:.4f} Hz   -> use for FREQ_MEAN_HZ")
    print(f"Combined SD of frequency: {sd_freq:.4f} Hz    -> use for FREQ_SD_HZ")
    print("=" * 60)


if __name__ == "__main__":
    main()
