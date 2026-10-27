"""
game_launcher.py
================
One window to run every calibration and trial of the fractal balance game,
without the terminal.

HOW TO USE
  1. Put this file in the SAME folder as the game scripts:
       CAL_FREQ_CoP.py  CAL_AMP_CoP.py  CAL_FREQ_CoM.py  CAL_AMP_CoM.py
       FREQ_CoP_trial.py  FREQ_CoM_trial.py  AMP_CoP_trial.py  AMP_CoM_trial.py
       SINE_CoP_trial.py  SINE_CoM_trial.py
     (all data then goes to data/<Subject>/ in that folder, so the trials
     find the calibration files automatically).
  2. Start it once: double-click Start_Game_Launcher.bat, or open this file
     in PyCharm and press the green Run button.
  3. Type the Subject ID, pick the group, check the quiet-standing times.
  4. Click a button. The game opens full screen; everything inside the game
     (QTM, SPACE / ENTER / S / ESC) works exactly as before. When the game
     window closes, the launcher comes back and shows whether the files
     look OK.

WHAT IT DOES NOT DO
  It does not change anything inside the game. It only starts the same
  scripts with the subject, group, H and quiet times filled in, so the
  start screen is skipped and HURST does not have to be edited.

FILES IT WRITES
  data/<Subject>/<Subject>_session_log.csv   one row per run (+ your notes)
  data/<Subject>/console/<time>_<script>.txt the text each script printed
                                             (what you used to see in the
                                             terminal) - open it if a run
                                             is marked CHECK.
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

# ===================== SETTINGS =====================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")
# Python used to run the game scripts. sys.executable = the same Python
# that runs this launcher (the one with pygame, qtm_rt, numpy, scipy).
PYTHON_FOR_SCRIPTS = os.environ.get("GAME_PYTHON", sys.executable)
H_LEVELS = ["0.1", "0.3", "0.5", "0.7", "0.9"]
DEFAULT_QUIET_PRE = 30
DEFAULT_QUIET_POST = 30
CONTROLLERS = ["CoP", "CoM"]
SUBJECT_PATTERN = re.compile(r"^S\d{2,3}$")
# script file names (edit here if a file is renamed)
CAL_SCRIPTS = {("CALFREQ", c): f"CAL_FREQ_{c}.py" for c in CONTROLLERS}
CAL_SCRIPTS.update({("CALAMP", c): f"CAL_AMP_{c}.py" for c in CONTROLLERS})
TRIAL_SCRIPTS = {(g, c): f"{g}_{c}_trial.py" for g in ("FREQ", "AMP", "SINE") for c in CONTROLLERS}
CAL_LABELS = {"CALFREQ": "Frequency calibration", "CALAMP": "Amplitude calibration"}
# ====================================================


def script_path(name):
    return os.path.join(HERE, name)


def find_files(subject, pattern):
    folder = os.path.join(DATA_ROOT, subject)
    return sorted(glob.glob(os.path.join(folder, pattern)), key=os.path.getmtime)


def check_trial_result(subject, started_at):
    """Reads the meta file the trial just wrote. Returns (status, details)."""
    metas = [p for p in find_files(subject, "*_meta.json") if os.path.getmtime(p) >= started_at - 1]
    if not metas:
        return "CHECK", "no meta file written (trial stopped early?)"
    path = metas[-1]
    try:
        meta = json.load(open(path))
    except Exception as e:
        return "CHECK", f"meta file could not be read: {e}"
    problems = []
    if not meta.get("completed_full_duration"):
        problems.append("game did not run the full duration")
    if meta.get("hurst_found_within_tolerance") is False:
        problems.append(f"H search used closest pathway (H={meta.get('hurst_realized_hk')})")
    files = meta.get("files", {})
    for key in ("game60hz", "fp1000hz", "mocap100hz"):
        if not files.get(key):
            problems.append(f"{key} missing")
    n = max(1, meta.get("frames_logged") or 1)
    qc = meta.get("qc", {})
    for key, label in (("frames_com_invalid", "CoM missing"), ("frames_low_fz", "low force")):
        pct = 100.0 * qc.get(key, 0) / n
        if pct > 5:
            problems.append(f"{label} {pct:.0f}%")
    stem = meta.get("trial_stem", os.path.basename(path))
    if problems:
        return "CHECK", f"{stem}: " + "; ".join(problems)
    h = meta.get("hurst_realized_hk")
    return "OK", f"{stem}" + (f"  (pathway H = {h})" if h is not None else "")


def check_cal_result(subject, task, controller, started_at):
    summ = [p for p in find_files(subject, f"{subject}_*_{controller}_{task}_T*_summary.csv")
            if os.path.getmtime(p) >= started_at - 1]
    if not summ:
        return "CHECK", "no summary file written (calibration stopped early or no cycles found?)"
    return "OK", os.path.basename(summ[-1])


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fractal Balance Game - Launcher")
        self.geometry("1120x720")
        self.minsize(900, 640)
        self.proc = None
        self.run_info = None
        self.buttons = []

        style = ttk.Style(self)
        style.configure("Big.TButton", padding=(10, 6))
        style.configure("Head.TLabel", font=("Segoe UI", 11, "bold"))

        # ---------- participant ----------
        top = ttk.LabelFrame(self, text="Participant", padding=10)
        top.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(top, text="Subject ID").grid(row=0, column=0, sticky="w")
        self.subject = tk.StringVar(value="S")
        e = ttk.Entry(top, textvariable=self.subject, width=10, font=("Segoe UI", 12))
        e.grid(row=0, column=1, padx=(5, 20))
        ttk.Label(top, text="Group").grid(row=0, column=2, sticky="w")
        self.group = tk.StringVar(value="")
        for i, g in enumerate(("FREQ", "AMP")):
            ttk.Radiobutton(top, text=g, value=g, variable=self.group,
                            command=self.refresh).grid(row=0, column=3 + i, padx=4)
        ttk.Label(top, text="Quiet before (s)").grid(row=0, column=5, padx=(25, 4))
        self.q_pre = tk.StringVar(value=str(DEFAULT_QUIET_PRE))
        ttk.Entry(top, textvariable=self.q_pre, width=5).grid(row=0, column=6)
        ttk.Label(top, text="Quiet after (s)").grid(row=0, column=7, padx=(15, 4))
        self.q_post = tk.StringVar(value=str(DEFAULT_QUIET_POST))
        ttk.Entry(top, textvariable=self.q_post, width=5).grid(row=0, column=8)
        ttk.Button(top, text="Refresh", command=self.refresh).grid(row=0, column=9, padx=(20, 0))
        ttk.Button(top, text="Open data folder", command=self.open_folder).grid(row=0, column=10, padx=(8, 0))
        self.subject.trace_add("write", lambda *_: self.refresh())

        # ---------- calibrations + trials, one column per controller ----------
        mid = ttk.Frame(self)
        mid.pack(fill="x", padx=10, pady=5)
        self.cal_status = {}
        self.trial_status = {}
        self.h_choice = {}
        self.trial_btn = {}
        self.sens = {}
        self._sens_subject = None
        for col, ctrl in enumerate(CONTROLLERS):
            box = ttk.LabelFrame(mid, text=f"{ctrl}-controlled", padding=10)
            box.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 5, 0))
            mid.columnconfigure(col, weight=1)

            ttk.Label(box, text="1. Calibrations", style="Head.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
            r = 1
            for task in ("CALFREQ", "CALAMP"):
                b = ttk.Button(box, text=CAL_LABELS[task], style="Big.TButton", width=24,
                               command=lambda t=task, c=ctrl: self.run_cal(t, c))
                b.grid(row=r, column=0, sticky="w", pady=2)
                self.buttons.append(b)
                lab = ttk.Label(box, text="", foreground="#666666")
                lab.grid(row=r, column=1, columnspan=2, sticky="w", padx=8)
                self.cal_status[(task, ctrl)] = lab
                r += 1

            ttk.Label(box, text="2. Trials", style="Head.TLabel").grid(row=r, column=0, columnspan=3, sticky="w", pady=(12, 0))
            r += 1
            sf = ttk.Frame(box)
            sf.grid(row=r, column=0, columnspan=3, sticky="w", pady=(2, 6))
            ttk.Label(sf, text=f"Sensitivity for {ctrl} trials (mm)").pack(side="left")
            self.sens[ctrl] = tk.StringVar(value="")
            ttk.Entry(sf, textvariable=self.sens[ctrl], width=7, font=("Segoe UI", 11)).pack(side="left", padx=6)
            ttk.Label(sf, text="(saved for this participant)", foreground="#999999").pack(side="left")
            r += 1
            b = ttk.Button(box, text="SINE baseline", style="Big.TButton", width=24,
                           command=lambda c=ctrl: self.run_trial("SINE", c, None))
            b.grid(row=r, column=0, sticky="w", pady=2)
            self.buttons.append(b)
            lab = ttk.Label(box, text="", foreground="#666666")
            lab.grid(row=r, column=1, columnspan=2, sticky="w", padx=8)
            self.trial_status[("SINE", ctrl)] = lab
            r += 1

            b = ttk.Button(box, text="Game trial", style="Big.TButton", width=24,
                           command=lambda c=ctrl: self.run_trial(None, c, self.h_choice[c].get()))
            b.grid(row=r, column=0, sticky="w", pady=2)
            self.buttons.append(b)
            self.trial_btn[ctrl] = b
            ttk.Label(box, text="H").grid(row=r, column=1, sticky="e", padx=(8, 2))
            self.h_choice[ctrl] = tk.StringVar(value="")
            ttk.Combobox(box, textvariable=self.h_choice[ctrl], values=H_LEVELS, width=5,
                         state="readonly").grid(row=r, column=2, sticky="w")
            r += 1
            lab = ttk.Label(box, text="", foreground="#666666", wraplength=420, justify="left")
            lab.grid(row=r, column=0, columnspan=3, sticky="w")
            self.trial_status[("GAME", ctrl)] = lab

        # ---------- log ----------
        logf = ttk.LabelFrame(self, text="3. This session", padding=10)
        logf.pack(fill="both", expand=True, padx=10, pady=5)
        cols = ("time", "task", "h", "result", "details", "note")
        self.tree = ttk.Treeview(logf, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (70, 190, 45, 60, 420, 200)):
            self.tree.heading(c, text=c.capitalize() if c != "h" else "H")
            self.tree.column(c, width=w, anchor="w", stretch=(c in ("details", "note")))
        self.tree.tag_configure("OK", foreground="#1b7f3b")
        self.tree.tag_configure("CHECK", foreground="#b3261e")
        self.tree.pack(fill="both", expand=True)
        nf = ttk.Frame(logf)
        nf.pack(fill="x", pady=(6, 0))
        ttk.Label(nf, text="Note for the selected run:").pack(side="left")
        self.note = tk.StringVar()
        ttk.Entry(nf, textvariable=self.note).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(nf, text="Save note", command=self.save_note).pack(side="left")
        ttk.Button(nf, text="Show console text", command=self.show_console).pack(side="left", padx=(6, 0))

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, anchor="w", padding=(10, 4),
                  font=("Segoe UI", 10, "bold")).pack(fill="x", side="bottom")
        self.refresh()

    # ---------- helpers ----------
    def current(self, need_group=True):
        subj = self.subject.get().strip().upper()
        if not SUBJECT_PATTERN.match(subj):
            messagebox.showwarning("Subject ID", "Subject ID must look like S01, S015, ...")
            return None
        grp = self.group.get()
        if need_group and grp not in ("FREQ", "AMP"):
            messagebox.showwarning("Group", "Please choose the group (FREQ or AMP).")
            return None
        return subj, grp

    def quiet_times(self):
        try:
            pre, post = float(self.q_pre.get()), float(self.q_post.get())
            if not (0 <= pre <= 600 and 0 <= post <= 600):
                raise ValueError
            return pre, post
        except ValueError:
            messagebox.showwarning("Quiet standing", "Quiet standing times must be numbers between 0 and 600 s.")
            return None

    # ---------- participant sensitivity (saved per participant) ----------
    def sens_path(self, subj):
        return os.path.join(DATA_ROOT, subj, f"{subj}_sensitivity.json")

    def load_sensitivity(self, subj):
        """Fill the two sensitivity boxes from this participant's saved values
        (empty if none saved yet). Runs only when the Subject ID changes."""
        values = {}
        path = self.sens_path(subj) if SUBJECT_PATTERN.match(subj) else None
        if path and os.path.exists(path):
            try:
                values = json.load(open(path))
            except Exception:
                values = {}
        for ctrl in CONTROLLERS:
            v = values.get(f"{ctrl}_range_mm")
            self.sens[ctrl].set("" if v is None else f"{v:g}")

    def save_sensitivity(self, subj):
        values = {}
        for ctrl in CONTROLLERS:
            try:
                values[f"{ctrl}_range_mm"] = float(self.sens[ctrl].get())
            except ValueError:
                pass
        path = self.sens_path(subj)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        old = {}
        if os.path.exists(path):
            try:
                old = json.load(open(path))
            except Exception:
                old = {}
        old.update(values)
        old["updated"] = datetime.now().isoformat(timespec="seconds")
        json.dump(old, open(path, "w"), indent=2)

    def refresh(self):
        subj = self.subject.get().strip().upper()
        grp = self.group.get()
        ok_subj = bool(SUBJECT_PATTERN.match(subj))
        if subj != self._sens_subject:
            self._sens_subject = subj
            self.load_sensitivity(subj)
        for ctrl in CONTROLLERS:
            self.trial_btn[ctrl].configure(text=f"{grp} game trial" if grp else "Game trial (choose group)")
            for task in ("CALFREQ", "CALAMP"):
                lab = self.cal_status[(task, ctrl)]
                files = find_files(subj, f"{subj}_*_{ctrl}_{task}_T*_summary.csv") if ok_subj else []
                lab.configure(text=f"done ({len(files)}x) - latest {os.path.basename(files[-1]).split('_summary')[0][-3:]}"
                              if files else "not done yet",
                              foreground="#1b7f3b" if files else "#999999")
            n_sine = len(find_files(subj, f"{subj}_*_{ctrl}_SINE_T*_meta.json")) if ok_subj else 0
            self.trial_status[("SINE", ctrl)].configure(text=f"done {n_sine}x" if n_sine else "not done yet",
                                                        foreground="#1b7f3b" if n_sine else "#999999")
            counts = []
            if ok_subj and grp:
                for h in H_LEVELS:
                    tag = f"H{int(round(float(h) * 10)):02d}"
                    k = len(find_files(subj, f"{subj}_{grp}_{ctrl}_{tag}_T*_meta.json"))
                    if k:
                        counts.append(f"H{h}: {k}")
            self.trial_status[("GAME", ctrl)].configure(text="Done so far - " + (", ".join(counts) if counts else "none"))
        for ctrl in CONTROLLERS:
            path_ok = all(os.path.exists(script_path(n)) for n in
                          (CAL_SCRIPTS[("CALFREQ", ctrl)], CAL_SCRIPTS[("CALAMP", ctrl)], TRIAL_SCRIPTS[("SINE", ctrl)]))
            if not path_ok:
                self.status.set(f"Warning: some {ctrl} scripts are not in {HERE}")

    def open_folder(self):
        subj = self.subject.get().strip().upper()
        folder = os.path.join(DATA_ROOT, subj) if SUBJECT_PATTERN.match(subj) else DATA_ROOT
        os.makedirs(folder, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(folder)
        else:
            subprocess.Popen(["xdg-open", folder])

    # ---------- running ----------
    def run_cal(self, task, ctrl):
        cur = self.current()
        if not cur:
            return
        subj, grp = cur
        self.start(CAL_SCRIPTS[(task, ctrl)], subj, grp, f"{CAL_LABELS[task]} ({ctrl})", None,
                   ["--subject", subj, "--group", grp], kind=("cal", task, ctrl))

    def run_trial(self, kind, ctrl, h):
        cur = self.current()
        if not cur:
            return
        subj, grp = cur
        q = self.quiet_times()
        if q is None:
            return
        try:
            rng = float(self.sens[ctrl].get())
            if not 20 <= rng <= 600:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Sensitivity", f"Enter this participant's {ctrl} sensitivity in mm "
                                                  f"(a number between 20 and 600) before starting a {ctrl} trial.")
            return
        self.save_sensitivity(subj)
        args = ["--subject", subj, "--group", grp, "--quiet-pre", f"{q[0]:g}", "--quiet-post", f"{q[1]:g}",
                "--range", f"{rng:g}"]
        if kind == "SINE":
            script, label = TRIAL_SCRIPTS[("SINE", ctrl)], f"SINE baseline ({ctrl})"
        else:
            if h not in H_LEVELS:
                messagebox.showwarning("H level", f"Choose the H level for the {ctrl} trial first.")
                return
            script, label = TRIAL_SCRIPTS[(grp, ctrl)], f"{grp} trial ({ctrl})"
            args += ["--hurst", h]
        self.start(script, subj, grp, label, h, args, kind=("trial", rng))

    def start(self, script, subj, grp, label, h, args, kind):
        if self.proc is not None:
            return
        path = script_path(script)
        if not os.path.exists(path):
            messagebox.showerror("Script not found", f"{script} is not in\n{HERE}")
            return
        console_dir = os.path.join(DATA_ROOT, subj, "console")
        os.makedirs(console_dir, exist_ok=True)
        console = os.path.join(console_dir, f"{datetime.now():%Y%m%d_%H%M%S}_{script.replace('.py', '')}.txt")
        out = open(console, "w", encoding="utf-8", errors="replace")
        cmd = [PYTHON_FOR_SCRIPTS, "-u", path] + args
        out.write("Command: " + " ".join(cmd) + "\n\n")
        out.flush()
        started = time.time()
        try:
            self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
        except Exception as e:
            out.close()
            messagebox.showerror("Could not start", str(e))
            return
        self.run_info = dict(script=script, subj=subj, grp=grp, label=label, h=h or "", kind=kind,
                             started=started, console=console, out=out)
        for b in self.buttons:
            b.state(["disabled"])
        self.status.set(f"Running: {label}{'  H=' + h if h else ''}  - the game window is open "
                        f"(the launcher comes back when it closes)")
        self.after(500, self.poll)

    def poll(self):
        if self.proc is None:
            return
        code = self.proc.poll()
        if code is None:
            self.after(500, self.poll)
            return
        info = self.run_info
        info["out"].close()
        self.proc, self.run_info = None, None
        if info["kind"][0] == "cal":
            status, details = check_cal_result(info["subj"], info["kind"][1], info["kind"][2], info["started"])
        else:
            status, details = check_trial_result(info["subj"], info["started"])
            details += f"  [sensitivity {info['kind'][1]:g} mm]"
        if code not in (0, None):
            status = "CHECK"
            details = f"script ended with an error (code {code}) - see console text. " + details
        row = dict(time=datetime.now().strftime("%H:%M:%S"), date=datetime.now().strftime("%Y-%m-%d"),
                   subject=info["subj"], group=info["grp"], task=info["label"], h=info["h"],
                   result=status, details=details, note="", console=os.path.basename(info["console"]))
        self.append_log(row)
        iid = self.tree.insert("", "end", values=(row["time"], row["task"], row["h"], status, details, ""),
                               tags=(status,))
        self.tree.selection_set(iid)
        self.tree.see(iid)
        self.tree.item(iid, text=info["console"])
        for b in self.buttons:
            b.state(["!disabled"])
        self.status.set(f"Finished: {info['label']} - {status}")
        self.refresh()
        self.deiconify()
        self.lift()
        self.attributes("-topmost", True)
        self.after(300, lambda: self.attributes("-topmost", False))
        self.focus_force()

    # ---------- session log ----------
    def log_path(self, subj):
        return os.path.join(DATA_ROOT, subj, f"{subj}_session_log.csv")

    def append_log(self, row):
        path = self.log_path(row["subject"])
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def save_note(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Note", "Select a run in the list first.")
            return
        iid = sel[0]
        vals = list(self.tree.item(iid, "values"))
        vals[5] = self.note.get()
        self.tree.item(iid, values=vals)
        subj = self.subject.get().strip().upper()
        path = self.log_path(subj)
        console = os.path.basename(self.tree.item(iid, "text"))
        if os.path.exists(path):
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            for r in rows:
                if r.get("console") == console:
                    r["note"] = self.note.get()
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        self.note.set("")
        self.status.set("Note saved.")

    def show_console(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Console text", "Select a run in the list first.")
            return
        path = self.tree.item(sel[0], "text")
        if not path or not os.path.exists(path):
            return
        win = tk.Toplevel(self)
        win.title(os.path.basename(path))
        win.geometry("900x500")
        txt = tk.Text(win, wrap="none", font=("Consolas", 9))
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", open(path, encoding="utf-8", errors="replace").read())
        txt.see("end")

    def destroy(self):
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("Game still running", "A game script is still running. Close the launcher anyway?"):
                return
        super().destroy()


if __name__ == "__main__":
    Launcher().mainloop()
