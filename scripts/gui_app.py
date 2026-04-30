from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from src_mfg.gui_workflows import (
    WORKFLOWS,
    WorkflowSpec,
    build_article_pipeline_inputs,
    build_command,
    normalize_values,
    stringify_value,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

BG_APP = "#F4F7FB"
BG_PANEL = "#FFFFFF"
BG_HERO = "#123554"
FG_HERO = "#F7FBFF"
FG_MUTED = "#5D6B7D"
ACCENT = "#C4662A"
ACCENT_SOFT = "#F5E4D7"
BORDER = "#D8E2ED"
LOG_BG = "#0F1722"
LOG_FG = "#D7E6F5"

RECOMMENDED_STEPS = [
    (
        "build_basis",
        "Build PCA Basis",
        "Start here for the article. This creates the global basis used by the Kaggle PCA connectivity analysis.",
    ),
    (
        "batch_phase",
        "Run Phase Analysis",
        "Main article path. This gives the within-phase connectivity results from the Grasp-and-Lift Kaggle data.",
    ),
    (
        "pre_event",
        "Run Pre-Event EMA",
        "Complementary article analysis. Use it to show what builds before key movement events.",
    ),
    (
        "meta_analysis",
        "Export Meta Analysis",
        "Generate article-ready text, CSV, and JSON summaries across subjects.",
    ),
    (
        "meta_sensitivity",
        "Run Sensitivity Grid",
        "Check whether key edges survive changes in top-k, min-subjects, and p0 inflation.",
    ),
    (
        "article_package",
        "Build Article Package",
        "Collect tables, meta exports, figure indexes, and provenance into one final article folder.",
    ),
]


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        canvas = tk.Canvas(self, highlightthickness=0, background=BG_APP)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas, style="App.TFrame")

        self.inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas
        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 if int(event.delta) > 0 else 1
        self.canvas.yview_scroll(delta, "units")


class StartHereTab(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        open_workflow_callback,
        run_pipeline_callback,
        open_results_callback,
        check_status_callback,
    ) -> None:
        super().__init__(master, style="App.TFrame")
        self._open_workflow_callback = open_workflow_callback
        self._run_pipeline_callback = run_pipeline_callback
        self._open_results_callback = open_results_callback
        self._check_status_callback = check_status_callback
        self.dataset_root_var = tk.StringVar(
            value="data/grasp-and-lift-eeg-detection/train"
        )
        self.out_root_var = tk.StringVar(value="out")
        self.readiness_var = tk.StringVar(
            value="Click 'Check Setup & Results' to verify data, outputs, and article package readiness."
        )

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True, padx=14, pady=14)
        body = scroll.inner

        self._build_hero(body)
        self._build_quick_run(body, start_row=1)
        self._build_readiness(body, start_row=3)
        self._build_step_section(
            body,
            title="Recommended Kaggle Article Flow",
            subtitle="This project is now centered on the Grasp-and-Lift Kaggle dataset. These are the steps to use first for the article.",
            steps=RECOMMENDED_STEPS,
            start_row=5,
            open_button_label="Open Recommended Preset",
        )
        self._build_tips(body, start_row=11)

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

    def _build_hero(self, parent: ttk.Frame) -> None:
        hero = tk.Frame(parent, bg=BG_HERO, padx=24, pady=20, highlightthickness=0)
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))

        tk.Label(
            hero,
            text="Kaggle Article Workspace",
            bg=BG_HERO,
            fg=FG_HERO,
            font=("Segoe UI Semibold", 22),
        ).pack(anchor="w")
        tk.Label(
            hero,
            text=(
                "Main focus: Grasp-and-Lift EEG Detection (Kaggle).\n"
                "The recommended route is PCA basis -> phase analysis -> pre-event EMA -> meta-analysis -> sensitivity -> article package."
            ),
            bg=BG_HERO,
            fg="#D9E9F6",
            font=("Segoe UI", 11),
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

    def _build_quick_run(self, parent: ttk.Frame, *, start_row: int) -> None:
        card = tk.Frame(
            parent,
            bg=BG_PANEL,
            padx=18,
            pady=16,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        card.grid(row=start_row, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        tk.Label(
            card,
            text="One-Click Article Run",
            bg=BG_PANEL,
            fg="#18324B",
            font=("Segoe UI Semibold", 14),
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        tk.Label(
            card,
            text=(
                "This runs the full Kaggle article pipeline automatically:\n"
                "PCA basis -> phase analysis -> pre-event EMA -> meta-analysis -> sensitivity -> article package."
            ),
            bg=BG_PANEL,
            fg=FG_MUTED,
            font=("Segoe UI", 10),
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 12))

        tk.Label(
            card,
            text="Dataset root",
            bg=BG_PANEL,
            fg="#18324B",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=2, column=0, sticky="w", pady=(0, 4))
        ttk.Entry(card, textvariable=self.dataset_root_var, width=60).grid(
            row=2, column=1, sticky="ew", padx=(0, 8)
        )
        ttk.Button(
            card,
            text="Browse",
            command=lambda: self._browse_into(self.dataset_root_var),
        ).grid(row=2, column=2, sticky="w")

        tk.Label(
            card,
            text="Article output root",
            bg=BG_PANEL,
            fg="#18324B",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=3, column=0, sticky="w", pady=(10, 4))
        ttk.Entry(card, textvariable=self.out_root_var, width=60).grid(
            row=3, column=1, sticky="ew", padx=(0, 8)
        )
        ttk.Button(
            card,
            text="Browse",
            command=lambda: self._browse_into(self.out_root_var),
        ).grid(row=3, column=2, sticky="w")

        ttk.Button(
            card,
            text="Run Full Article Pipeline",
            style="Accent.TButton",
            command=self._run_full_pipeline,
        ).grid(row=4, column=0, sticky="w", pady=(14, 0))
        ttk.Button(
            card,
            text="Open Results Browser",
            command=lambda: self._open_results_callback(self.out_root_var.get()),
        ).grid(row=4, column=1, sticky="w", pady=(14, 0))
        ttk.Button(
            card,
            text="Open Final Package",
            command=self._open_final_package,
        ).grid(row=4, column=2, sticky="w", pady=(14, 0))
        ttk.Button(
            card,
            text="Check Setup & Results",
            command=self._check_readiness,
        ).grid(row=5, column=0, sticky="w", pady=(10, 0))

        card.columnconfigure(1, weight=1)

    def _browse_into(self, var: tk.StringVar) -> None:
        current = str(var.get()).strip()
        initial_dir = str((REPO_ROOT / current).resolve()) if current else str(REPO_ROOT)
        selected = filedialog.askdirectory(initialdir=initial_dir)
        if not selected:
            return
        try:
            var.set(os.path.relpath(Path(selected).resolve(), REPO_ROOT))
        except OSError:
            var.set(selected)

    def _run_full_pipeline(self) -> None:
        self._run_pipeline_callback(
            self.dataset_root_var.get().strip(),
            self.out_root_var.get().strip(),
        )

    def _open_final_package(self) -> None:
        package_path = Path(self.out_root_var.get().strip() or "out") / "article_package"
        self._open_results_callback(str(package_path))

    def _check_readiness(self) -> None:
        report = self._check_status_callback(
            self.dataset_root_var.get().strip(),
            self.out_root_var.get().strip(),
        )
        self.readiness_var.set(report)

    def _build_readiness(self, parent: ttk.Frame, *, start_row: int) -> None:
        card = tk.Frame(
            parent,
            bg=BG_PANEL,
            padx=18,
            pady=16,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        card.grid(row=start_row, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        tk.Label(
            card,
            text="Project Readiness",
            bg=BG_PANEL,
            fg="#18324B",
            font=("Segoe UI Semibold", 14),
        ).pack(anchor="w")
        tk.Label(
            card,
            textvariable=self.readiness_var,
            bg=BG_PANEL,
            fg=FG_MUTED,
            font=("Consolas", 10),
            justify="left",
            wraplength=1080,
        ).pack(anchor="w", pady=(8, 0), fill="x")

    def _build_step_section(
        self,
        parent: ttk.Frame,
        *,
        title: str,
        subtitle: str,
        steps: list[tuple[str, str, str]],
        start_row: int,
        open_button_label: str,
    ) -> None:
        ttk.Label(parent, text=title, style="SectionTitle.TLabel").grid(
            row=start_row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 2),
        )
        ttk.Label(
            parent,
            text=subtitle,
            style="Muted.TLabel",
            wraplength=1080,
            justify="left",
        ).grid(row=start_row + 1, column=0, columnspan=2, sticky="w", pady=(0, 10))

        for idx, (workflow_key, label, desc) in enumerate(steps):
            card = tk.Frame(
                parent,
                bg=BG_PANEL,
                padx=18,
                pady=16,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            row = start_row + 2 + idx // 2
            col = idx % 2
            padx = (0, 12) if col == 0 else (0, 0)
            card.grid(row=row, column=col, sticky="nsew", padx=padx, pady=(0, 12))

            tk.Label(
                card,
                text=label,
                bg=BG_PANEL,
                fg="#18324B",
                font=("Segoe UI Semibold", 13),
            ).pack(anchor="w")
            tk.Label(
                card,
                text=desc,
                bg=BG_PANEL,
                fg=FG_MUTED,
                font=("Segoe UI", 10),
                justify="left",
                wraplength=430,
            ).pack(anchor="w", pady=(8, 14))
            ttk.Button(
                card,
                text=open_button_label,
                style="Accent.TButton",
                command=lambda key=workflow_key: self._open_workflow_callback(
                    key, True
                ),
            ).pack(anchor="w")

    def _build_tips(self, parent: ttk.Frame, *, start_row: int) -> None:
        box = tk.Frame(
            parent,
            bg=ACCENT_SOFT,
            padx=18,
            pady=16,
            highlightbackground="#E6C6B0",
            highlightthickness=1,
        )
        box.grid(row=start_row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Label(
            box,
            text="Article Tips",
            bg=ACCENT_SOFT,
            fg="#7A3D16",
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w")
        tk.Label(
            box,
            text=(
                "Use the recommended presets first.\n"
                "Leave advanced settings hidden unless you already know why you want to change them.\n"
                "For the paper, treat phase analysis as the main result and pre-event EMA as a complementary analysis.\n"
                "Use Sensitivity Grid to support claims that are stable, not parameter-picked.\n"
                "Finish with Article Package to gather the tables and provenance you can actually cite.\n"
                "Secondary dataset paths are intentionally omitted from the interface to keep the project focused."
            ),
            bg=ACCENT_SOFT,
            fg="#7A3D16",
            font=("Segoe UI", 10),
            justify="left",
            wraplength=1080,
        ).pack(anchor="w", pady=(8, 0))


class WorkflowTab(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        spec: WorkflowSpec,
        run_callback,
        open_output_callback,
    ) -> None:
        super().__init__(master, style="App.TFrame")
        self.spec = spec
        self._run_callback = run_callback
        self._open_output_callback = open_output_callback
        self.vars: dict[str, tk.Variable] = {}
        self.show_advanced_var = tk.BooleanVar(value=False)
        self.advanced_frame: ttk.LabelFrame | None = None

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True, padx=14, pady=14)
        body = scroll.inner

        self._build_header(body)
        self._build_toolbar(body)
        self._build_form(body)
        self._build_preview(body)
        self._build_actions(body)

        body.columnconfigure(0, weight=1)
        self.apply_initial_state()

    def _build_header(self, parent: ttk.Frame) -> None:
        header = tk.Frame(
            parent,
            bg=BG_PANEL,
            padx=18,
            pady=16,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        tk.Label(
            header,
            text=self.spec.title,
            bg=BG_PANEL,
            fg="#18324B",
            font=("Segoe UI Semibold", 18),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=self.spec.description,
            bg=BG_PANEL,
            fg=FG_MUTED,
            font=("Segoe UI", 10),
            justify="left",
            wraplength=1040,
        ).pack(anchor="w", pady=(6, 10))

        if self.spec.recommended_preset:
            tk.Label(
                header,
                text=f"Recommended for Kaggle article: {self.spec.recommended_preset}",
                bg=ACCENT_SOFT,
                fg="#7A3D16",
                font=("Segoe UI", 9, "bold"),
                padx=10,
                pady=4,
            ).pack(anchor="w")

    def _build_toolbar(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent, style="Card.TFrame")
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        preset_names = ["Custom"] + list(self.spec.presets.keys())
        self.preset_var = tk.StringVar(value="Custom")
        ttk.Label(toolbar, text="Preset", style="FieldLabel.TLabel").pack(side="left")
        ttk.Combobox(
            toolbar,
            textvariable=self.preset_var,
            values=preset_names,
            state="readonly",
            width=28,
        ).pack(side="left", padx=(8, 8))

        ttk.Button(
            toolbar,
            text="Use Recommended",
            style="Accent.TButton",
            command=self.apply_recommended_preset,
        ).pack(side="left")
        ttk.Button(
            toolbar,
            text="Apply Preset",
            command=self.apply_selected_preset,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            toolbar,
            text="Reset To Raw Defaults",
            command=self.apply_defaults,
        ).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(
            toolbar,
            text="Show advanced settings",
            variable=self.show_advanced_var,
            command=self.toggle_advanced,
        ).pack(side="right")

    def _build_form(self, parent: ttk.Frame) -> None:
        essential = ttk.LabelFrame(
            parent,
            text="Essential Settings",
            padding=14,
            style="Card.TLabelframe",
        )
        essential.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        self._render_fields(essential, advanced=False)

        advanced = ttk.LabelFrame(
            parent,
            text="Advanced Settings",
            padding=14,
            style="Card.TLabelframe",
        )
        self._render_fields(advanced, advanced=True)
        self.advanced_frame = advanced

    def _render_fields(self, parent: ttk.Frame, *, advanced: bool) -> None:
        row_idx = 0
        for field in self.spec.fields:
            if field.advanced != advanced:
                continue

            ttk.Label(parent, text=field.label, style="FieldLabel.TLabel").grid(
                row=row_idx * 2,
                column=0,
                sticky="w",
                padx=(0, 12),
                pady=(0, 2),
            )

            if field.kind == "bool":
                var = tk.BooleanVar(value=bool(field.default))
                widget = ttk.Checkbutton(parent, variable=var)
                widget.grid(row=row_idx * 2, column=1, sticky="w")
            elif field.kind == "choice":
                var = tk.StringVar(value=str(field.default))
                widget = ttk.Combobox(
                    parent,
                    textvariable=var,
                    values=list(field.choices),
                    state="readonly",
                    width=42,
                )
                widget.grid(row=row_idx * 2, column=1, sticky="ew")
            else:
                var = tk.StringVar(value=stringify_value(field.kind, field.default))
                widget = ttk.Entry(parent, textvariable=var, width=52)
                widget.grid(row=row_idx * 2, column=1, sticky="ew")
                if field.kind in {"dir", "file_open", "file_save"}:
                    ttk.Button(
                        parent,
                        text="Browse",
                        command=lambda current=field: self._browse(current),
                    ).grid(row=row_idx * 2, column=2, sticky="w", padx=(8, 0))

            self.vars[field.key] = var
            var.trace_add("write", lambda *_args: self.refresh_preview())

            help_text = field.help_text or self._default_help(field.kind, field.advanced)
            ttk.Label(
                parent,
                text=help_text,
                style="Muted.TLabel",
                wraplength=860,
                justify="left",
            ).grid(
                row=row_idx * 2 + 1,
                column=1,
                columnspan=2,
                sticky="w",
                pady=(0, 8),
            )
            row_idx += 1

        parent.columnconfigure(1, weight=1)

    def _build_preview(self, parent: ttk.Frame) -> None:
        preview_frame = ttk.LabelFrame(
            parent,
            text="Command Preview",
            padding=10,
            style="Card.TLabelframe",
        )
        preview_frame.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        self.preview = tk.Text(
            preview_frame,
            height=4,
            wrap="word",
            bg="#FBFCFE",
            fg="#24405A",
            relief="flat",
            font=("Consolas", 10),
        )
        self.preview.pack(fill="x")
        self.preview.configure(state="disabled")

    def _build_actions(self, parent: ttk.Frame) -> None:
        button_row = ttk.Frame(parent, style="Card.TFrame")
        button_row.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(
            button_row,
            text="Run Workflow",
            style="Accent.TButton",
            command=self.run,
        ).pack(side="left")
        ttk.Button(
            button_row,
            text="Open Output Folder",
            command=self.open_output,
        ).pack(side="left", padx=(8, 0))

    @staticmethod
    def _default_help(kind: str, advanced: bool) -> str:
        if kind in {"int_list", "str_list"}:
            return "Enter values separated by spaces or commas."
        if kind in {"dir", "file_open", "file_save"}:
            return "You can type a path or use Browse."
        if advanced:
            return "Optional advanced setting."
        return ""

    def _browse(self, field) -> None:
        current = str(self.vars[field.key].get()).strip()
        initial_dir = (
            str((REPO_ROOT / current).resolve().parent)
            if current
            else str(REPO_ROOT)
        )

        if field.kind == "dir":
            selected = filedialog.askdirectory(title=field.label, initialdir=initial_dir)
        elif field.kind == "file_open":
            selected = filedialog.askopenfilename(
                title=field.label,
                initialdir=initial_dir,
            )
        else:
            selected = filedialog.asksaveasfilename(
                title=field.label,
                initialdir=initial_dir,
                initialfile=Path(current).name if current else "",
            )

        if selected:
            self.vars[field.key].set(self._make_relative(selected))

    @staticmethod
    def _make_relative(path_str: str) -> str:
        try:
            path = Path(path_str).resolve()
            return os.path.relpath(path, REPO_ROOT)
        except OSError:
            return path_str

    def apply_initial_state(self) -> None:
        if self.spec.recommended_preset:
            self.apply_recommended_preset()
        else:
            self.apply_defaults()
        self.toggle_advanced()

    def apply_defaults(self) -> None:
        for field in self.spec.fields:
            self.vars[field.key].set(stringify_value(field.kind, field.default))
        self.preset_var.set("Custom")
        self.refresh_preview()

    def apply_recommended_preset(self) -> None:
        if not self.spec.recommended_preset:
            self.apply_defaults()
            return
        self.preset_var.set(self.spec.recommended_preset)
        self.apply_selected_preset()

    def apply_selected_preset(self) -> None:
        preset_name = self.preset_var.get()
        if preset_name == "Custom":
            return
        preset = self.spec.presets.get(preset_name, {})
        for field in self.spec.fields:
            value = preset.get(field.key, field.default)
            self.vars[field.key].set(stringify_value(field.kind, value))
        self.refresh_preview()

    def toggle_advanced(self) -> None:
        if self.advanced_frame is None:
            return
        if self.show_advanced_var.get():
            self.advanced_frame.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        else:
            self.advanced_frame.grid_forget()

    def get_raw_values(self) -> dict[str, object]:
        data: dict[str, object] = {}
        for field in self.spec.fields:
            data[field.key] = self.vars[field.key].get()
        return data

    def refresh_preview(self) -> None:
        try:
            cmd, _values = build_command(
                self.spec,
                self.get_raw_values(),
                python_executable=sys.executable,
            )
            text = subprocess.list2cmdline(cmd)
        except Exception as exc:
            text = f"Validation error: {exc}"

        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def run(self) -> None:
        self._run_callback(self.spec, self.get_raw_values())

    def open_output(self) -> None:
        self._open_output_callback(self.spec, self.get_raw_values())


class MethodologyTab(ttk.Frame):
    def __init__(self, master: tk.Misc, open_path_callback) -> None:
        super().__init__(master, style="App.TFrame")
        self._open_path_callback = open_path_callback

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True, padx=14, pady=14)
        body = scroll.inner
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        self._build_intro(body)
        self._build_card(
            body,
            row=1,
            column=0,
            title="What We Estimate",
            text=(
                "The pipeline estimates event-locked, directed lagged dependence "
                "between EEG channels. Mathematically, it uses normalized signals, "
                "Legendre product-basis mixed moments, PCA dependence modes, and "
                "cross-subject reproducibility tests."
            ),
        )
        self._build_card(
            body,
            row=1,
            column=1,
            title="Theoretical Basis",
            text=(
                "The method follows time-delay multi-feature dependence analysis: "
                "joint-density features are estimated with an orthonormal product "
                "basis and reduced with PCA."
            ),
        )
        self._build_card(
            body,
            row=2,
            column=0,
            title="Interpretation",
            text=(
                "Use 'directional lagged innovation-coupling' or 'directed functional "
                "connectivity'. Do not claim anatomical causality. Zero-lag visual "
                "effects especially can reflect common causes or volume conduction."
            ),
        )
        self._build_card(
            body,
            row=2,
            column=1,
            title="Current Preset",
            text=(
                "Current article presets use m=4, event-locked nonnegative lags, "
                "EDF source normalization in directional mode, and an AR/EMA Student "
                "target innovation transform."
            ),
        )
        self._build_doc_buttons(body, row=3)

    def _build_intro(self, parent: ttk.Frame) -> None:
        hero = tk.Frame(parent, bg=BG_HERO, padx=24, pady=20)
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        tk.Label(
            hero,
            text="Methodology",
            bg=BG_HERO,
            fg=FG_HERO,
            font=("Segoe UI Semibold", 22),
        ).pack(anchor="w")
        tk.Label(
            hero,
            text=(
                "A compact guide to the active method, interpretation, outputs, and "
                "recommended project workflow."
            ),
            bg=BG_HERO,
            fg="#D9E9F6",
            font=("Segoe UI", 11),
            justify="left",
            wraplength=1080,
        ).pack(anchor="w", pady=(8, 0))

    def _build_card(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        column: int,
        title: str,
        text: str,
    ) -> None:
        card = tk.Frame(
            parent,
            bg=BG_PANEL,
            padx=18,
            pady=16,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        card.grid(
            row=row,
            column=column,
            sticky="nsew",
            padx=(0, 12) if column == 0 else (0, 0),
            pady=(0, 12),
        )
        tk.Label(
            card,
            text=title,
            bg=BG_PANEL,
            fg="#18324B",
            font=("Segoe UI Semibold", 13),
        ).pack(anchor="w")
        tk.Label(
            card,
            text=text,
            bg=BG_PANEL,
            fg=FG_MUTED,
            font=("Segoe UI", 10),
            justify="left",
            wraplength=480,
        ).pack(anchor="w", pady=(8, 0))

    def _build_doc_buttons(self, parent: ttk.Frame, *, row: int) -> None:
        card = tk.Frame(
            parent,
            bg=ACCENT_SOFT,
            padx=18,
            pady=16,
            highlightbackground="#E6C6B0",
            highlightthickness=1,
        )
        card.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        tk.Label(
            card,
            text="Open Documentation",
            bg=ACCENT_SOFT,
            fg="#7A3D16",
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w")
        buttons = tk.Frame(card, bg=ACCENT_SOFT)
        buttons.pack(anchor="w", pady=(10, 0))
        for label, rel_path in [
            ("Methodology", "docs/METHODOLOGY.md"),
            ("Project Overview", "docs/PROJECT_OVERVIEW.md"),
            ("Final Package", "out/article_package/article_summary.md"),
        ]:
            ttk.Button(
                buttons,
                text=label,
                command=lambda path=rel_path: self._open_path_callback(
                    (REPO_ROOT / path).resolve()
                ),
            ).pack(side="left", padx=(0, 8))


class PipelineGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MFG EEG Kaggle Article GUI")
        self.geometry("1420x940")
        self.minsize(1180, 780)
        self.configure(bg=BG_APP)

        self.proc: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[tuple[str, str] | tuple[str, int]] = queue.Queue()
        self.active_output: Path | None = None
        self.pending_runs: list[tuple[WorkflowSpec, dict[str, object]]] = []
        self.run_queue_name: str | None = None
        self.total_queue_count = 0
        self.current_queue_index = 0
        self.preview_image: tk.PhotoImage | None = None
        self.result_item_paths: dict[str, Path] = {}

        self._setup_styles()
        self._build_layout()

        self.after(100, self._poll_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("App.TFrame", background=BG_APP)
        style.configure("Card.TFrame", background=BG_APP)
        style.configure("TNotebook", background=BG_APP, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", BG_PANEL)])
        style.configure("Card.TLabelframe", background=BG_PANEL, bordercolor=BORDER)
        style.configure(
            "Card.TLabelframe.Label",
            background=BG_PANEL,
            foreground="#18324B",
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "SectionTitle.TLabel",
            background=BG_APP,
            foreground="#18324B",
            font=("Segoe UI Semibold", 14),
        )
        style.configure(
            "Muted.TLabel",
            background=BG_APP,
            foreground=FG_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "FieldLabel.TLabel",
            background=BG_PANEL,
            foreground="#18324B",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10))
        style.map(
            "Accent.TButton",
            background=[("!disabled", ACCENT), ("active", "#A94F17")],
            foreground=[("!disabled", "#FFFFFF")],
        )

    def _build_layout(self) -> None:
        self._build_header()

        paned = ttk.Panedwindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        top_holder = ttk.Frame(paned, style="App.TFrame")
        log_holder = ttk.Frame(paned, style="App.TFrame")
        paned.add(top_holder, weight=4)
        paned.add(log_holder, weight=2)

        self.notebook = ttk.Notebook(top_holder)
        self.notebook.pack(fill="both", expand=True)

        self.notebook.add(
            StartHereTab(
                self.notebook,
                self.open_workflow,
                self.run_article_pipeline,
                self.open_results_root,
                self.build_readiness_report,
            ),
            text="Start Here",
        )
        self.notebook.add(
            MethodologyTab(self.notebook, self.open_project_path),
            text="Methodology",
        )

        self.tabs: dict[str, WorkflowTab] = {}
        for workflow in WORKFLOWS:
            tab = WorkflowTab(self.notebook, workflow, self.run_workflow, self.open_output)
            self.tabs[workflow.key] = tab
            self.notebook.add(tab, text=workflow.title)

        self._build_bottom_panel(log_holder)

    def _build_header(self) -> None:
        hero = tk.Frame(self, bg=BG_HERO, padx=22, pady=18)
        hero.pack(fill="x", padx=14, pady=(14, 12))

        left = tk.Frame(hero, bg=BG_HERO)
        left.pack(side="left", fill="both", expand=True)

        tk.Label(
            left,
            text="MFG EEG Article Workspace",
            bg=BG_HERO,
            fg=FG_HERO,
            font=("Segoe UI Semibold", 23),
        ).pack(anchor="w")
        tk.Label(
            left,
            text=(
                "Primary dataset: Grasp-and-Lift EEG Detection from Kaggle.\n"
                "The interface below is tuned to help you produce article-ready figures, tables, and reproducible outputs."
            ),
            bg=BG_HERO,
            fg="#D9E9F6",
            font=("Segoe UI", 11),
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        right = tk.Frame(hero, bg="#0D2A43", padx=16, pady=14)
        right.pack(side="right", padx=(20, 0))
        tk.Label(
            right,
            text="Recommended order",
            bg="#0D2A43",
            fg="#F1F7FB",
            font=("Segoe UI Semibold", 11),
        ).pack(anchor="w")
        tk.Label(
            right,
            text="1. PCA basis\n2. Phase analysis\n3. Pre-event EMA\n4. Meta-analysis\n5. Sensitivity\n6. Article package",
            bg="#0D2A43",
            fg="#CFE0EE",
            font=("Segoe UI", 10),
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

    def _build_bottom_panel(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent, style="App.TFrame")
        controls.pack(fill="x", pady=(0, 8))
        self.status_var = tk.StringVar(
            value=f"Ready for the Kaggle article workflow. Python: {sys.executable}"
        )
        ttk.Label(controls, textvariable=self.status_var, style="Muted.TLabel").pack(
            side="left"
        )
        ttk.Button(controls, text="Stop Current Run", command=self.stop_process).pack(
            side="right"
        )
        ttk.Button(controls, text="Clear Log", command=self.clear_log).pack(
            side="right", padx=(0, 8)
        )
        ttk.Button(
            controls,
            text="Open Last Output",
            command=self.open_last_output,
        ).pack(side="right", padx=(0, 8))

        bottom_notebook = ttk.Notebook(parent)
        bottom_notebook.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(
            bottom_notebook,
            text="Live Log",
            padding=8,
            style="Card.TLabelframe",
        )
        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            height=18,
            bg=LOG_BG,
            fg=LOG_FG,
            insertbackground="#FFFFFF",
            relief="flat",
            font=("Consolas", 10),
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview,
        )
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        bottom_notebook.add(log_frame, text="Live Log")

        results_frame = ttk.LabelFrame(
            bottom_notebook,
            text="Results Browser",
            padding=8,
            style="Card.TLabelframe",
        )
        self._build_results_panel(results_frame)
        bottom_notebook.add(results_frame, text="Results Browser")

    def _build_results_panel(self, parent: ttk.LabelFrame) -> None:
        top = ttk.Frame(parent, style="App.TFrame")
        top.pack(fill="x", pady=(0, 8))

        self.results_root_var = tk.StringVar(value="out")
        ttk.Label(top, text="Root", style="FieldLabel.TLabel").pack(side="left")
        ttk.Entry(top, textvariable=self.results_root_var, width=52).pack(
            side="left", padx=(8, 8), fill="x", expand=True
        )
        ttk.Button(top, text="Browse", command=self._browse_results_root).pack(
            side="left"
        )
        ttk.Button(top, text="Refresh", command=self.refresh_results_tree).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(top, text="Open Root", command=self._open_results_root_current).pack(
            side="left", padx=(8, 0)
        )

        shortcuts = ttk.Frame(parent, style="App.TFrame")
        shortcuts.pack(fill="x", pady=(0, 8))
        ttk.Label(shortcuts, text="Quick views", style="Muted.TLabel").pack(side="left")
        for label, path in [
            ("Article Package", "out/article_package"),
            ("Meta Tables", "out/article_meta_kaggle"),
            ("Sensitivity", "out/article_meta_sensitivity"),
            ("Phase Results", "out/phase_mats_pca_by_subject"),
            ("Pre-Event", "out/experimental_kaggle_by_subject"),
        ]:
            ttk.Button(
                shortcuts,
                text=label,
                command=lambda p=path: self._set_results_root(self._coerce_path(p)),
            ).pack(side="left", padx=(8, 0))

        split = ttk.Panedwindow(parent, orient="horizontal")
        split.pack(fill="both", expand=True)

        tree_frame = ttk.Frame(split, style="App.TFrame")
        preview_frame = ttk.Frame(split, style="App.TFrame")
        split.add(tree_frame, weight=2)
        split.add(preview_frame, weight=3)

        self.results_tree = ttk.Treeview(tree_frame, show="tree")
        self.results_tree.pack(side="left", fill="both", expand=True)
        tree_scroll = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self.results_tree.yview
        )
        tree_scroll.pack(side="right", fill="y")
        self.results_tree.configure(yscrollcommand=tree_scroll.set)
        self.results_tree.bind("<<TreeviewSelect>>", self._on_result_selected)

        preview_top = ttk.Frame(preview_frame, style="App.TFrame")
        preview_top.pack(fill="x", pady=(0, 8))
        self.preview_title_var = tk.StringVar(value="Select a result file to preview.")
        ttk.Label(
            preview_top,
            textvariable=self.preview_title_var,
            style="FieldLabel.TLabel",
        ).pack(side="left")
        ttk.Button(
            preview_top,
            text="Open Selected",
            command=self.open_selected_result,
        ).pack(side="right")

        self.preview_text = tk.Text(
            preview_frame,
            wrap="word",
            bg="#FBFCFE",
            fg="#24405A",
            relief="flat",
            font=("Consolas", 10),
            height=16,
        )
        self.preview_text.pack(fill="both", expand=True)

        self.preview_image_label = ttk.Label(preview_frame)
        self.preview_image_label.pack(fill="both", expand=True)
        self.preview_image_label.pack_forget()

        self.refresh_results_tree()

    def _browse_results_root(self) -> None:
        current = self.results_root_var.get().strip()
        initial = self._coerce_path(current) if current else REPO_ROOT
        selected = filedialog.askdirectory(initialdir=str(initial))
        if not selected:
            return
        self._set_results_root(Path(selected))

    def _set_results_root(self, path: Path) -> None:
        try:
            shown = os.path.relpath(path.resolve(), REPO_ROOT)
        except OSError:
            shown = str(path)
        self.results_root_var.set(shown)
        self.refresh_results_tree()

    def open_results_root(self, path_str: str) -> None:
        path_str = path_str.strip()
        if not path_str:
            return
        path = self._coerce_path(path_str)
        self._set_results_root(path)
        self._open_path(path)

    def _open_results_root_current(self) -> None:
        path = self._coerce_path(self.results_root_var.get().strip() or "out")
        if not path.exists():
            messagebox.showinfo("Results root not found", f"Path does not exist:\n{path}")
            return
        self._open_path(path)

    def refresh_results_tree(self) -> None:
        self.results_tree.delete(*self.results_tree.get_children())
        self.result_item_paths.clear()
        root_path = self._coerce_path(self.results_root_var.get().strip() or "out")
        self.preview_title_var.set(f"Results root: {root_path}")
        if not root_path.exists():
            self._show_preview_text(
                "Results root missing",
                f"The selected results root does not exist yet:\n{root_path}",
            )
            return
        root_item = self.results_tree.insert("", "end", text=root_path.name or str(root_path))
        self.result_item_paths[root_item] = root_path
        self._insert_result_children(root_item, root_path)
        self.results_tree.item(root_item, open=True)

    def _insert_result_children(self, parent_id: str, path: Path) -> None:
        try:
            children = sorted(
                path.iterdir(),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except OSError:
            return

        for child in children:
            item_id = self.results_tree.insert(parent_id, "end", text=child.name)
            self.result_item_paths[item_id] = child
            if child.is_dir():
                self._insert_result_children(item_id, child)

    def _on_result_selected(self, _event: tk.Event) -> None:
        item = self._selected_result_item()
        if item is None:
            return
        path = self.result_item_paths.get(item)
        if path is None:
            return
        self._preview_path(path)

    def _selected_result_item(self) -> str | None:
        selection = self.results_tree.selection()
        return selection[0] if selection else None

    def open_selected_result(self) -> None:
        item = self._selected_result_item()
        if item is None:
            messagebox.showinfo("Nothing selected", "Select a file or directory first.")
            return
        path = self.result_item_paths.get(item)
        if path is None:
            return
        self._open_path(path)

    def _preview_path(self, path: Path) -> None:
        if path.is_dir():
            try:
                entries = list(path.iterdir())
            except OSError as exc:
                self._show_preview_text(path.name, f"Could not read directory:\n{exc}")
                return
            dirs = sum(1 for entry in entries if entry.is_dir())
            files = sum(1 for entry in entries if entry.is_file())
            self._show_preview_text(
                path.name,
                f"Directory: {path}\n\nSubdirectories: {dirs}\nFiles: {files}",
            )
            return

        suffix = path.suffix.lower()
        if suffix in {".png", ".gif", ".ppm", ".pgm"}:
            self._show_preview_image(path)
            return
        if suffix in {".txt", ".csv", ".json", ".md", ".py"}:
            self._show_preview_text_file(path)
            return
        if suffix == ".npz":
            self._show_preview_npz(path)
            return

        self._show_preview_text(
            path.name,
            f"Path: {path}\nSize: {path.stat().st_size} bytes",
        )

    def _show_preview_text_file(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read(20000)
        except OSError as exc:
            self._show_preview_text(path.name, f"Could not read file:\n{exc}")
            return
        self._show_preview_text(path.name, content)

    def _show_preview_npz(self, path: Path) -> None:
        try:
            with np.load(path, allow_pickle=False) as data:
                lines = [f"NPZ file: {path}", ""]
                for key in sorted(data.files):
                    arr = data[key]
                    lines.append(
                        f"{key}: shape={arr.shape} dtype={arr.dtype}"
                    )
        except Exception as exc:
            self._show_preview_text(path.name, f"Could not inspect NPZ:\n{exc}")
            return
        self._show_preview_text(path.name, "\n".join(lines))

    def _show_preview_text(self, title: str, content: str) -> None:
        self.preview_title_var.set(title)
        self.preview_image_label.pack_forget()
        self.preview_text.pack(fill="both", expand=True)
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", content)
        self.preview_text.configure(state="disabled")
        self.preview_image = None

    def _show_preview_image(self, path: Path) -> None:
        try:
            image = tk.PhotoImage(file=str(path))
        except Exception as exc:
            self._show_preview_text(path.name, f"Could not load image:\n{exc}")
            return

        factor = max(1, image.width() // 900, image.height() // 500)
        if factor > 1:
            image = image.subsample(factor, factor)

        self.preview_title_var.set(path.name)
        self.preview_text.pack_forget()
        self.preview_image_label.pack(fill="both", expand=True)
        self.preview_image_label.configure(image=image)
        self.preview_image = image

    def open_workflow(self, workflow_key: str, apply_recommended: bool = True) -> None:
        tab = self.tabs.get(workflow_key)
        if tab is None:
            return
        self.notebook.select(tab)
        if apply_recommended:
            tab.apply_recommended_preset()

    def open_project_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showinfo("Path not found", f"Path does not exist yet:\n{path}")
            return
        self._open_path(path)

    def run_article_pipeline(self, dataset_root: str, out_root: str) -> None:
        dataset_root = dataset_root.strip()
        out_root = out_root.strip()
        if not dataset_root or not out_root:
            messagebox.showerror(
                "Missing settings",
                "Dataset root and article output root are required.",
            )
            return

        items = build_article_pipeline_inputs(
            dataset_root=dataset_root,
            out_root=out_root,
        )
        self._start_run_queue(
            items,
            queue_name="Full Kaggle Article Pipeline",
            results_root=self._coerce_path(out_root),
        )

    def build_readiness_report(self, dataset_root: str, out_root: str) -> str:
        dataset_path = self._coerce_path(dataset_root or "data/grasp-and-lift-eeg-detection/train")
        output_path = self._coerce_path(out_root or "out")

        def mark(ok: bool) -> str:
            return "[OK]" if ok else "[MISSING]"

        data_files = list(dataset_path.glob("subj*_series*_data.csv")) if dataset_path.exists() else []
        event_files = list(dataset_path.glob("subj*_series*_events.csv")) if dataset_path.exists() else []
        subjects = {
            name.split("_series")[0]
            for name in [path.stem for path in data_files]
            if "_series" in name
        }

        expected = [
            ("PCA basis", output_path / "basis_allpairs.npz"),
            ("Phase results", output_path / "phase_mats_pca_by_subject"),
            ("Pre-event EMA", output_path / "experimental_kaggle_by_subject"),
            ("Meta-analysis", output_path / "article_meta_kaggle"),
            ("Sensitivity", output_path / "article_meta_sensitivity"),
            ("Article package", output_path / "article_package" / "article_summary.md"),
        ]

        lines = [
            f"{mark(dataset_path.exists())} Dataset root: {dataset_path}",
            f"{mark(bool(data_files))} Data files: {len(data_files)}",
            f"{mark(bool(event_files))} Event files: {len(event_files)}",
            f"{mark(bool(subjects))} Subjects detected: {len(subjects)}",
            "",
            "Method: HCR/PCA-style time-delay multi-feature dependence analysis.",
            "Variant: event-locked, cross-subject, meta-analysis pipeline.",
            "Safe wording: directional lagged innovation-coupling, not anatomical causality.",
            "",
            f"Article output root: {output_path}",
        ]
        for label, path in expected:
            lines.append(f"{mark(path.exists())} {label}: {path}")

        if data_files and event_files and len(data_files) != len(event_files):
            lines.extend(
                [
                    "",
                    "[WARN] Data/event file counts differ. Check that every *_data.csv has matching *_events.csv.",
                ]
            )
        if not dataset_path.exists():
            lines.extend(
                [
                    "",
                    "[TIP] Point Dataset root to data/grasp-and-lift-eeg-detection/train.",
                ]
            )
        if not (output_path / "article_package" / "article_summary.md").exists():
            lines.extend(
                [
                    "",
                    "[TIP] Run the full article pipeline or run Article Package after meta/sensitivity outputs exist.",
                ]
            )
        return "\n".join(lines)

    def append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    def _resolve_output_path(
        self,
        spec: WorkflowSpec,
        values: dict[str, object],
    ) -> Path | None:
        if spec.output_key is None:
            return None
        raw = values.get(spec.output_key)
        if raw in (None, ""):
            return None
        path = Path(str(raw))
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path

    def _coerce_path(self, raw: str | Path) -> Path:
        path = raw if isinstance(raw, Path) else Path(str(raw))
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path

    def _start_run_queue(
        self,
        items: list[tuple[WorkflowSpec, dict[str, object]]],
        *,
        queue_name: str,
        results_root: Path | None,
    ) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showinfo(
                "Process already running",
                "Stop the current run before starting another workflow.",
            )
            return

        self.pending_runs = list(items)
        self.run_queue_name = queue_name
        self.total_queue_count = len(self.pending_runs)
        self.current_queue_index = 0
        if results_root is not None:
            self._set_results_root(results_root)
        self.append_log(f"\n=== {queue_name} ===\n")
        self._launch_next_queued_run()

    def _launch_next_queued_run(self) -> None:
        if not self.pending_runs:
            self.run_queue_name = None
            self.total_queue_count = 0
            self.current_queue_index = 0
            self.status_var.set("Ready for the next Kaggle article step.")
            self.refresh_results_tree()
            return

        self.current_queue_index += 1
        spec, raw_values = self.pending_runs.pop(0)
        try:
            cmd, values = build_command(
                spec,
                raw_values,
                python_executable=sys.executable,
            )
        except Exception as exc:
            self.pending_runs.clear()
            self.run_queue_name = None
            self.total_queue_count = 0
            self.current_queue_index = 0
            self.status_var.set("Validation error in queued run.")
            messagebox.showerror("Validation error", str(exc))
            return

        self.active_output = self._resolve_output_path(spec, values)
        if self.active_output is not None:
            target = self.active_output.parent if self.active_output.suffix else self.active_output
            self._set_results_root(target)
        step_label = (
            f"step {self.current_queue_index}/{self.total_queue_count}"
            if self.total_queue_count
            else "single step"
        )
        self.append_log(f"\n--- {spec.title} ({step_label}) ---\n")
        self.append_log(subprocess.list2cmdline(cmd) + "\n\n")
        self.status_var.set(f"Running {step_label}: {spec.title}")

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            self.proc = None
            self.pending_runs.clear()
            self.run_queue_name = None
            self.total_queue_count = 0
            self.current_queue_index = 0
            self.status_var.set("Failed to start process.")
            messagebox.showerror("Launch failed", str(exc))
            return

        threading.Thread(target=self._read_process_output, daemon=True).start()

    def run_workflow(self, spec: WorkflowSpec, raw_values: dict[str, object]) -> None:
        try:
            normalized = normalize_values(spec, raw_values)
        except Exception as exc:
            messagebox.showerror("Validation error", str(exc))
            return
        self._start_run_queue(
            [(spec, raw_values)],
            queue_name=spec.title,
            results_root=self._resolve_output_path(spec, normalized),
        )

    def _read_process_output(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self.log_queue.put(("line", line))
        exit_code = proc.wait()
        self.log_queue.put(("done", exit_code))

    def _poll_log_queue(self) -> None:
        while True:
            try:
                kind, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "line":
                self.append_log(str(payload))
                continue

            exit_code = int(payload)
            if exit_code == 0:
                self.append_log("\n[OK] Step finished successfully.\n")
                self.proc = None
                if self.pending_runs:
                    self._launch_next_queued_run()
                    continue
                self.append_log("\n[OK] Run queue completed successfully.\n")
                self.status_var.set("Ready for the next Kaggle article step.")
                self.total_queue_count = 0
                self.current_queue_index = 0
            else:
                self.append_log(f"\n[ERROR] Workflow exited with code {exit_code}.\n")
                self.pending_runs.clear()
                self.run_queue_name = None
                self.total_queue_count = 0
                self.current_queue_index = 0
                self.status_var.set(f"Workflow failed (exit code {exit_code}).")
            self.proc = None

        self.after(100, self._poll_log_queue)

    def stop_process(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            messagebox.showinfo("No active run", "There is no running workflow to stop.")
            return
        self.pending_runs.clear()
        self.run_queue_name = None
        self.total_queue_count = 0
        self.current_queue_index = 0
        self.proc.terminate()
        self.append_log("\n[INFO] Stop requested.\n")
        self.status_var.set("Stopping current workflow...")

    def open_output(self, spec: WorkflowSpec, raw_values: dict[str, object]) -> None:
        try:
            values = normalize_values(spec, raw_values)
        except Exception as exc:
            messagebox.showerror("Validation error", str(exc))
            return

        path = self._resolve_output_path(spec, values)
        if path is None:
            messagebox.showinfo(
                "No output path",
                "This workflow does not define an output path.",
            )
            return
        if not path.exists():
            if path.suffix:
                parent = path.parent
                if parent.exists():
                    path = parent
                else:
                    messagebox.showinfo(
                        "Output not found",
                        f"Path does not exist yet:\n{path}",
                    )
                    return
            else:
                messagebox.showinfo(
                    "Output not found",
                    f"Path does not exist yet:\n{path}",
                )
                return

        self.active_output = path
        self._set_results_root(path.parent if path.is_file() else path)
        self._open_path(path)

    def open_last_output(self) -> None:
        if self.active_output is None:
            messagebox.showinfo(
                "No output yet",
                "Run a workflow first or use a tab's Open Output button.",
            )
            return
        path = self.active_output
        if path.is_file():
            path = path.parent
        if not path.exists():
            messagebox.showinfo("Output not found", f"Path does not exist:\n{path}")
            return
        self._set_results_root(path)
        self._open_path(path)

    @staticmethod
    def _open_path(path: Path) -> None:
        try:
            os.startfile(str(path))
        except AttributeError:
            messagebox.showinfo("Open path", str(path))
        except OSError as exc:
            messagebox.showerror("Open failed", str(exc))

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno(
                "Quit",
                "A workflow is still running. Quit and stop it?",
            ):
                return
            self.proc.terminate()
        self.destroy()


def main() -> None:
    app = PipelineGui()
    app.mainloop()


if __name__ == "__main__":
    main()
