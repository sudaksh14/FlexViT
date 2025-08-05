import torch
import numpy as np
import io
import sys
from typing import Optional, TextIO, Iterable
import time
import tempfile
import os
import subprocess

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

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

def save_table_from_latex(data: np.ndarray, output_pdf: str,
                      colors: np.ndarray = None,
                      column_names: Optional[Iterable[str]] = None,
                      row_names: Optional[Iterable[str]] = None):
    # Create LaTeX document
    latex_content = r"""\documentclass[preview]{standalone}
                        \usepackage[table]{xcolor}
                        \usepackage{hhline}
                        \begin{document}
                        """ + make_latex_table_string(data, colors, column_names, row_names) + "\n\\end{document}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, "table.tex")
        with open(tex_file, "w") as f:
            f.write(latex_content)

        # Compile LaTeX to PDF
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], cwd=tmpdir, stdout=subprocess.DEVNULL)

        # Move output PDF to target path
        pdf_path = os.path.join(tmpdir, "table.pdf")
        os.rename(pdf_path, output_pdf)

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
        save_table_as_pdf(data, "./figures/switch_time.pdf", color_map, column_names, row_names)
        
