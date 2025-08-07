import torch
import numpy as np
import io
import sys
from typing import Optional, TextIO, Iterable
import time
import tempfile
import os
import subprocess
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import pandas as pd

import colorsys

from networks import level_delta_utils as delta
from networks import flexvit
import utils

import tqdm
import tqdm.contrib.itertools as titer
import mixbox


def make_latex_table(file: TextIO, data: np.ndarray, colors: np.ndarray = None, column_names: Optional[Iterable[str]] = None, row_names: Optional[Iterable[str]] = None):
    rows, cols = data.shape
    print("\\begin{tabular}{", end='', file=file)
    if column_names is not None:
        print("|l|", end='', file=file)
    cols_str = "l".join(['|'] * (cols+1))
    print(f"{cols_str}}}\n\\hline", file=file)
    if column_names is not None:
        print(" & ".join(column_names), end='', file=file)
        if row_names is None:
            print(" \\\\ \\hline")
        else:
            print(" \\\\ \\hhline{|=|", end='', file=file)
            cols_str = "=".join(['|'] * (cols+1))
            print(f"{cols_str}}}", file=file)

    if row_names is None:
        for i in range(rows):
            row = data[i, :]
            if colors is not None:
                color_row = colors[i, :, :]

                def make_cell(c, d):
                    color = ','.join(map(str, c))
                    cmd = f"\\cellcolor[RGB]{{{color}}}"
                    cell = f"{cmd} {d:.2f}"
                    return cell
                print(' & '.join(map(make_cell, color_row, row)),
                      '\\\\ \\hline', file=file)
            else:
                print(' & '.join(map(str, row)), '\\\\ \\hline', file=file)
    else:
        for rname, i in zip(row_names, range(rows)):
            row = data[i, :]
            if colors is not None:
                color_row = colors[i, :, :]

                def make_cell(c, d):
                    color = ','.join(map(str, c))
                    cmd = f"\\cellcolor[RGB]{{{color}}}"
                    cell = f"{cmd} {d:.2f}"
                    return cell
                print(f"{rname} &", ' & '.join(
                    map(make_cell, color_row, row)), '\\\\ \\hline', file=file)
            else:
                print(f"{rname} &", ' & '.join(map(str, row)),
                      '\\\\ \\hline', file=file)
    print('\\end{tabular}', file=file)


def make_color_map(data):
    maxv = data.max()
    minv = data.min()

    def make_color(v):
        f = (v - minv) / maxv
        hue = 10 + 100 * (1 - f)
        rgb = colorsys.hsv_to_rgb(hue / 360, 0.6, 1.0)
        rgb *= np.array([255, 255, 255])
        return np.array(rgb)

    colors = np.zeros((*data.shape, 3), dtype=np.int_)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            colors[i, j] = make_color(data[i, j])
    return colors

def make_latex_table_string(data: np.ndarray, colors: np.ndarray = None,
                            column_names: Optional[Iterable[str]] = None,
                            row_names: Optional[Iterable[str]] = None) -> str:
    from io import StringIO
    buf = StringIO()
    make_latex_table(buf, data, colors, column_names, row_names)
    return buf.getvalue()

def save_table_as_pdf(data: np.ndarray, output_pdf: str,
                      colors_array: np.ndarray = None,
                      column_names: Optional[Iterable[str]] = None,
                      row_names: Optional[Iterable[str]] = None):
    """
    Save table as a PDF file using ReportLab (no LaTeX required).

    Args:
        data: 2D NumPy array with table data
        output_pdf: Output PDF file path
        colors_array: Optional 3D NumPy array of shape (rows, cols, 3) for RGB colors (0-255)
        column_names: Optional list of column names
        row_names: Optional list of row names
    """

    # Prepare table data
    table_data = []

    # Add column headers
    if column_names is not None:
        if row_names is not None:
            table_data.append([""] + list(column_names))
        else:
            table_data.append(list(column_names))

    # Add rows with optional row names
    for i in range(data.shape[0]):
        row_data = list(data[i, :])
        if row_names is not None:
            row_data = [row_names[i]] + row_data
        table_data.append(row_data)

    # Create table
    table = Table(table_data)

    # Style settings
    style = TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ])

    # Bold header row
    if column_names is not None:
        style.add('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
        style.add('TEXTCOLOR', (0, 0), (-1, 0), colors.black)
        style.add('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')

    # Add cell background colors if provided
    if colors_array is not None:
        start_row = 1 if column_names is not None else 0
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                rgb = tuple(colors_array[r, c] / 255.0)  # normalize to 0-1
                table_cell = (c + (1 if row_names is not None else 0), r + start_row)
                style.add('BACKGROUND', table_cell, table_cell, colors.Color(*rgb))

    table.setStyle(style)

    # Save PDF
    pdf = SimpleDocTemplate(output_pdf, pagesize=letter)
    pdf.build([table])


# if __name__ == "__main__":
#     points = 20
#     data = np.linspace(0, 1, points).reshape(points, 1)
#     colors = make_color_map(data)
#     make_latex_table(sys.stdout, data, colors)


def time_manager(manager: delta.FileDeltaManager, level_from: int, level_to: int, iters: int):
    total_time = 0

    for i in tqdm.tqdm(range(iters), leave=False):
        manager.move_to(level_from)
        start = time.monotonic_ns()
        manager.move_to(level_to)
        end = time.monotonic_ns()
        total_time += end - start

    milliseconds = total_time / 10e6
    milliseconds /= iters
    return milliseconds


def measure_model_switch_time(model_paths, trials=100, device="cuda") -> pd.DataFrame:
    """
    Measure the average switching time between models.
    Args:
        model_paths (list of str): Paths to saved .pt Torch models.
        trials (int): Number of trials for averaging.
        device (str): 'cuda' or 'cpu'.
    Returns:
        pandas.DataFrame: Table of average switching times (ms).
    """
    
    num_models = len(model_paths)
    switch_times = np.zeros((num_models, num_models), dtype=np.float32)

    # Load all models into GPU/CPU memory once
    models = [torch.load(p, map_location=device).eval() for p in model_paths]

    # Benchmark switching time
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue  # no switching cost for same model
            torch.cuda.synchronize()
            start = time.monotonic_ns()
            for _ in range(trials):
                _ = torch.load(model_paths[j], map_location=device).eval()  # simulate switching
            torch.cuda.synchronize()
            elapsed_ms = (time.monotonic_ns() - start) / (trials) / 1e6
            switch_times[i, j] = elapsed_ms

    # Print table
    print("\n=== Model Switching Time Matrix (ms) ===")
    header = "     " + " ".join([f"M{j}" for j in range(num_models)])
    print(header)
    for i in range(num_models):
        row_str = f"M{i} " + " ".join([f"{switch_times[i, j]:6.3f}" for j in range(num_models)])
        print(row_str)

    # Create table
    df = pd.DataFrame(
        switch_times,
        columns=[f"To_L{j}" for j in range(num_models)],
        index=[f"From_L{i}" for i in range(num_models)]
    )

    return df

# NOT IN USE
def benchmark_switching_linear(model_paths, trials=100, device="cuda"):
    results = {}

    # 1️⃣ Load all models in GPU memory
    print("Loading all models into GPU memory...")
    models_gpu = [torch.load(p, map_location=device).eval() for p in model_paths]
    torch.cuda.synchronize()
    start = time.monotonic_ns()
    for _ in range(trials):
        for m in models_gpu:
            _ = m  # simulate switching
    torch.cuda.synchronize()
    gpu_time = (time.monotonic_ns() - start) / (trials * len(models_gpu)) / 1e6
    results["GPU list"] = gpu_time

    # 2️⃣ Load all models in CPU memory, move to GPU on switch
    print("Loading all models into CPU memory...")
    models_cpu = [torch.load(p, map_location="cpu").eval() for p in model_paths]
    torch.cuda.synchronize()
    start = time.monotonic_ns()
    for _ in range(trials):
        for m in models_cpu:
            m_gpu = m.to(device, non_blocking=True)
    torch.cuda.synchronize()
    cpu_to_gpu_time = (time.monotonic_ns() - start) / (trials * len(models_cpu)) / 1e6
    results["CPU to GPU"] = cpu_to_gpu_time

    # 3️⃣ Load from disk every time
    print("Benchmarking load from disk...")
    torch.cuda.synchronize()
    start = time.monotonic_ns()
    for _ in range(trials):
        for p in model_paths:
            m = torch.load(p, map_location=device).eval()
    torch.cuda.synchronize()
    disk_time = (time.monotonic_ns() - start) / (trials * len(model_paths)) / 1e6
    results["Disk load"] = disk_time

    # Print results table
    print("\n=== Model Switching Benchmark ===")
    print(f"{'Method':<15} | {'Avg Switch Time (ms)':>20}")
    print("-" * 40)
    for method, t in results.items():
        print(f"{method:<15} | {t:>20.4f}")

    return results

# This function benchmarks the switching time between multiple models loaded from disk, CPU, and GPU.
def benchmark_switching(model_paths, trials=100, device="cuda"):
    num_models = len(model_paths)

    # Preload models in GPU
    print("Loading all models into GPU memory...")
    models_gpu = [torch.load(p, map_location=device).eval() for p in model_paths]

    # Preload models in CPU
    print("Loading all models into CPU memory...")
    models_cpu = [torch.load(p, map_location="cpu").eval() for p in model_paths]

    # Initialize result matrices
    gpu_gpu_times = np.zeros((num_models, num_models), dtype=np.float32)
    cpu_gpu_times = np.zeros((num_models, num_models), dtype=np.float32)
    disk_gpu_times = np.zeros((num_models, num_models), dtype=np.float32)

    # === 1. GPU→GPU Switching ===
    for i in range(num_models):
        for j in range(num_models):
            if i == j: 
                continue
            torch.cuda.synchronize()
            start = time.monotonic_ns()
            for _ in range(trials):
                _ = models_gpu[j]  # Just switch reference
            torch.cuda.synchronize()
            gpu_gpu_times[i, j] = (time.monotonic_ns() - start) / trials / 1e6  # ms

    # === 2. CPU→GPU Switching ===
    for i in range(num_models):
        for j in range(num_models):
            if i == j: 
                continue
            torch.cuda.synchronize()
            start = time.monotonic_ns()
            for _ in range(trials):
                _ = models_cpu[j].to(device, non_blocking=True)
            torch.cuda.synchronize()
            cpu_gpu_times[i, j] = (time.monotonic_ns() - start) / trials / 1e6

    # === 3. Disk→GPU Switching ===
    for i in range(num_models):
        for j in range(num_models):
            if i == j: 
                continue
            torch.cuda.synchronize()
            start = time.monotonic_ns()
            for _ in range(trials):
                _ = torch.load(model_paths[j], map_location=device).eval()
            torch.cuda.synchronize()
            disk_gpu_times[i, j] = (time.monotonic_ns() - start) / trials / 1e6

    # Function to print table nicely
    def print_table(name, matrix):
        print(f"\n=== {name} (ms) ===")
        header = "     " + " ".join([f"M{j}" for j in range(num_models)])
        print(header)
        for i in range(num_models):
            row_str = f"M{i} " + " ".join([f"{matrix[i,j]:6.3f}" for j in range(num_models)])
            print(row_str)

    # Print all tables
    print_table("GPU→GPU Switching", gpu_gpu_times)
    print_table("CPU→GPU Switching", cpu_gpu_times)
    print_table("Disk→GPU Switching", disk_gpu_times)

    return gpu_gpu_times, cpu_gpu_times, disk_gpu_times


def save_switching_table(times, labels, output_pdf="switching_times.pdf"):
    """
    Saves switching time table as PDF using matplotlib.

    Args:
        times (np.ndarray): 2D array of switching times in ms.
        labels (list): Model labels (rows/cols).
        output_pdf (str): Output PDF file path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    # Create table
    table_data = [[""] + labels]  # header
    for i, row_label in enumerate(labels):
        row = [row_label] + [f"{times[i, j]:.2f}" if i != j else "-" for j in range(len(labels))]
        table_data.append(row)

    table = ax.table(cellText=table_data, loc="center", cellLoc="center", colLoc="center")

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save as PDF
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Switching time table saved to {output_pdf}")

DELTA_FILENAME = "vit.delta"

FLEXVIT_CONFIG = flexvit.ViTConfig(
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

# This script generates the table with the delta file switching timings
if __name__ == "__main__":
    device = utils.get_device()

    model = FLEXVIT_CONFIG.make_model()
    model.load_state_dict(torch.load("./pretrained/FlexViT.pt", map_location=device))
    model.eval()
    num_iters = 1000

    # for i in range(model.max_level() + 1):
    # model.set_level_use(i)
    # reg_model = model.make_base_copy().eval()
    # torch.save(reg_model, "./pretrained/FlexViT_level_{}.pt".format(i))

    model_paths = [
        "./pretrained/FlexViT_level_0.pt",
        "./pretrained/FlexViT_level_1.pt",
        "./pretrained/FlexViT_level_2.pt",
        "./pretrained/FlexViT_level_3.pt",
        "./pretrained/FlexViT_level_4.pt"
    ]

    # #------------------------------------------------------SWITCHING WITH BAG OF MODELS-------------------------------------------
    # df_switch_times = measure_model_switch_time(model_paths, trials=num_iters, device=device)
    # df_switch_times.to_csv("switch_times.csv")


    # gpu_gpu, cpu_gpu, disk_gpu = benchmark_switching(model_paths, trials=num_iters, device=device)

    # labels = [f"Level {i}" for i in range(len(model_paths))]
    # save_switching_table(gpu_gpu, labels, output_pdf="./figures/gpu_gpu_switching_times.pdf")
    # save_switching_table(cpu_gpu, labels, output_pdf="./figures/cpu_gpu_switching_times.pdf")
    # save_switching_table(disk_gpu, labels, output_pdf="./figures/disk_gpu_switching_times.pdf")

    #------------------------------------------------------SWITCHING WITH DELTA FILE----------------------------------------------

    # first create this delta file
    with open(DELTA_FILENAME, "wb") as file:
        delta.FileDeltaManager.make_delta_file(file, model, starting_level=0)

    reg_config = FLEXVIT_CONFIG.create_base_config(0).no_prebuilt()
    with delta.file_delta_manager(DELTA_FILENAME, reg_config) as manager:
        reg_model = manager.managed_model()

        data = np.zeros((manager.max_level() + 1, manager.max_level() + 1))

        for i, j in titer.product(range(manager.max_level() + 1), range(manager.max_level() + 1)):
            if i == j:
                data[i, j] = 0.0
            else:
                data[i, j] = time_manager(manager, i, j, num_iters)

        column_names = [*map(str, range(manager.max_level() + 1))]
        row_names = [*map(str, range(manager.max_level() + 1))]

        color_map = make_color_map(data)

        make_latex_table(sys.stdout, data, color_map, column_names, row_names)
        save_table_as_pdf(data, "./figures/delta_file_switch_time.pdf", color_map, column_names, row_names)
        
