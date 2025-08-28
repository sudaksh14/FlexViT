import numpy as np
import sys
from typing import Optional, TextIO, Iterable
import time
import os
import colorsys
import matplotlib.pyplot as plt
import torch

from networks import level_delta_utils as delta
from networks import flexvit
import utils
import tqdm
import tqdm.contrib.itertools as titer


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

def normalize_colors(colors):
    """Normalize color values to be within the 0-1 range."""
    colors = np.array(colors)
    if colors.max() > 1.0 or colors.min() < 0.0:
        colors = colors / 255.0
    return colors


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

def save_table_to_pdf(data, colors, column_names, row_names, pdf_filename="./figures/delta_level_switching_time_scala.pdf"):
    # Create a figure and axis for the table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Normalize color values
    colors = normalize_colors(colors)

    # Create the table
    table = ax.table(cellText=data, cellColours=colors, colLabels=column_names, rowLabels=row_names, loc='center')

    # Adjust table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save the table
    plt.savefig(pdf_filename, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

DELTA_FILENAME = "vit.delta"

# FlexViT Config is irrelevant for Scala, only relevant variable is hidden_dims
FLEXVIT_CONFIG = flexvit.ViTConfig(
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

# This script generates the table with the delta file switching timings
if __name__ == "__main__":
    model = FLEXVIT_CONFIG.make_model()
    model.eval()
    _ = model(torch.randn(1, 3, 224, 224))
    print("Model Working!")

    num_iter = 1000

    # first create this delta file
    device = utils.get_device()

    # Open the delta file and create the delta
    with open(DELTA_FILENAME, "wb") as file:
        delta.FileDeltaManager.make_delta_file(file, model, starting_level=0)

    # Create the regular configuration and open the delta file manager
    reg_config = FLEXVIT_CONFIG.create_base_config(0).no_prebuilt()
    with delta.file_delta_manager(DELTA_FILENAME, reg_config) as manager:
        # Put the managed model on GPU once
        manager.set_managed_model(manager.managed_model().to(device))

        # Initialize the data array
        data = np.zeros((manager.max_level() + 1, manager.max_level() + 1))

        # Iterate over levels and measure time
        for i, j in titer.product(range(manager.max_level() + 1), range(manager.max_level() + 1)):
            if i == j:
                data[i, j] = 0.0
            else:
                # print(f"Timing move from level {i} to level {j}")
                data[i, j] = time_manager(manager, i, j, num_iter)
                # print(f"Time taken: {data[i, j]:.2f} ms")

        # Prepare column and row names for the table
        column_names = ['', *map(str, range(manager.max_level() + 1))]
        row_names = [*map(str, range(manager.max_level() + 1))]

        # Generate colors for the table
        colors = make_color_map(data)

        # Create the LaTeX table
        make_latex_table(sys.stdout, data, colors, column_names, row_names)
        save_table_to_pdf(data, colors, column_names[1:], row_names)
