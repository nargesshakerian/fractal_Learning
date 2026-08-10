"""
Helper script: analyze repeated Free-Sway trials to compute the mean and
standard deviation of each trial's maximum CoP excursion. These values
feed directly into the Amplitude-Variable task (BASE_AMP_MAX and FGN_SD).

Usage:
    Put all free-sway CSV files (from repeated runs of
    free_sway_preferred.py) in one folder, then run:

        python analyze_free_sway_trials.py /path/to/folder

    Each file must have a column named "combined_cop_shared_mm"
    (this is what free_sway_preferred.py logs).

What it does, per file:
    1. Reads the combined_cop_shared_mm column.
    2. Finds the largest positive value and the largest (magnitude)
       negative value reached during that trial.
    3. Takes the SMALLER of these two magnitudes as that trial's "safe"
       max excursion - symmetric, same logic as calibrate_amplitude()
       in the main task, so the amplitude never assumes the person can
       go further in one direction than they demonstrated in the other.

Across all files, it then prints:
    - each trial's individual max excursion (so you can sanity-check
      for any obviously bad trials before trusting the average)
    - the mean and standard deviation across all trials

These printed values are meant to be copied by hand into the amplitude
task's settings (BASE_AMP_MAX from the mean, FGN_SD from the std) -
this script does not modify any other file automatically.
"""

import sys
import csv
import glob
import os
import statistics


def analyze_one_file(filepath):
    """Returns the trial's max excursion (mm), or None if the file has
    no usable data."""
    positive_vals = []
    negative_vals = []

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        if "combined_cop_shared_mm" not in reader.fieldnames:
            print(f"  WARNING: {os.path.basename(filepath)} has no "
                  f"'combined_cop_shared_mm' column - skipping.")
            return None

        for row in reader:
            raw = row.get("combined_cop_shared_mm", "")
            if raw == "":
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if val > 0:
                positive_vals.append(val)
            elif val < 0:
                negative_vals.append(val)

    if not positive_vals or not negative_vals:
        print(f"  WARNING: {os.path.basename(filepath)} has no clear "
              f"left/right excursion (missing positive or negative "
              f"values) - skipping.")
        return None

    max_positive = max(positive_vals)
    max_negative = min(negative_vals)   # most negative = largest leftward magnitude

    safe_excursion = min(abs(max_positive), abs(max_negative))
    print(f"  {os.path.basename(filepath):40s}  "
          f"max_right={max_positive:7.1f}mm  max_left={max_negative:7.1f}mm  "
          f"-> safe_excursion={safe_excursion:7.1f}mm")
    return safe_excursion


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_free_sway_trials.py /path/to/folder")
        print("(folder should contain the 10 free-sway CSV files)")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Not a folder: {folder}")
        sys.exit(1)

    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {folder}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) in {folder}\n")
    print("Per-trial results:")

    excursions = []
    for filepath in csv_files:
        result = analyze_one_file(filepath)
        if result is not None:
            excursions.append(result)

    print()
    if len(excursions) < 2:
        print(f"Only {len(excursions)} valid trial(s) found - need at least 2 "
              f"to compute a meaningful mean/std. Check the warnings above.")
        return

    mean_excursion = statistics.mean(excursions)
    std_excursion = statistics.stdev(excursions)   # sample std (n-1)

    print("=" * 60)
    print(f"Valid trials used: {len(excursions)} of {len(csv_files)}")
    print(f"Mean max excursion:  {mean_excursion:.2f} mm   -> use for BASE_AMP_MAX (after mm->px conversion)")
    print(f"Std of max excursion: {std_excursion:.2f} mm   -> use for FGN_SD")
    print("=" * 60)
    print()
    print("Next step: convert mean_excursion (mm) to px using the same")
    print("mm-to-px ratio as the amplitude task's COP_RANGE_MM, i.e.:")
    print("  mm_to_px_ratio = WIDTH / (2 * COP_RANGE_MM)")
    print("  BASE_AMP_MAX_px = mean_excursion_mm * mm_to_px_ratio")


if __name__ == "__main__":
    main()
