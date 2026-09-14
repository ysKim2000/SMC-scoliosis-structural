"""
Publication-ready image preprocessing pipeline.

Outputs
-------
fig_preprocessing_pipeline.png : 300 dpi raster figure
fig_preprocessing_pipeline.pdf : vector figure with embedded X-ray panels

The operations and parameters mirror multimodal/dataset.py.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FIGURE_DIR.parents[2]
EXAMPLE_DICOM = (
    PROJECT_DIR
    / "data"
    / "matched_data"
    / "PA"
    / "23573753_090902_1_1_.Seq1.Ser1.Img1.dcm"
)

BLUE = "#2166AC"
TEAL = "#1B9E77"
GOLD = "#E6A817"
PURPLE = "#8E44AD"
DARK = "#2B2B2B"
MID_GRAY = "#666666"
LIGHT_BLUE = "#EAF2FA"
LIGHT_TEAL = "#E8F5F1"
LIGHT_GOLD = "#FFF5D6"
LIGHT_PURPLE = "#F3EAF8"
LIGHT_GRAY = "#F3F3F3"


def normalize_minmax(image):
    image = image.astype(np.float32)
    vmin, vmax = float(image.min()), float(image.max())
    if vmax <= vmin:
        return np.zeros_like(image, dtype=np.float32)
    return (image - vmin) / (vmax - vmin)


def read_dicom(path):
    dataset = pydicom.dcmread(path, force=True)
    image = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return dataset, image * slope + intercept


def window_and_invert(dataset, image):
    output = image.copy()
    try:
        output = apply_voi_lut(output, dataset).astype(np.float32)
    except Exception:
        pass
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        output = output.max() - output
    return output


def percentile_clip(image, lower=1, upper=99):
    low, high = np.percentile(image, [lower, upper])
    if high <= low:
        return image.copy()
    return np.clip(image, low, high)


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    image_u8 = (normalize_minmax(image) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=grid_size,
    )
    return clahe.apply(image_u8).astype(np.float32) / 255.0


def pad_to_square(image, pad_value=0):
    height, width = image.shape
    if height == width:
        return image.copy()
    if height > width:
        left = (height - width) // 2
        right = height - width - left
        return np.pad(
            image,
            ((0, 0), (left, right)),
            mode="constant",
            constant_values=pad_value,
        )
    top = (width - height) // 2
    bottom = width - height - top
    return np.pad(
        image,
        ((top, bottom), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def preprocessing_stages(path):
    dataset, rescaled = read_dicom(path)
    windowed = window_and_invert(dataset, rescaled)
    clipped = percentile_clip(windowed, lower=1, upper=99)
    clahe = apply_clahe(clipped, clip_limit=2.0, grid_size=(8, 8))
    blurred = cv2.GaussianBlur(clahe, (3, 3), sigmaX=0.5)
    normalized_padded = pad_to_square(normalize_minmax(blurred), pad_value=0)
    resized = cv2.resize(
        normalized_padded,
        (512, 512),
        interpolation=cv2.INTER_AREA,
    )
    final_u8 = (resized * 255).astype(np.uint8)
    return [
        rescaled,
        windowed,
        clipped,
        clahe,
        blurred,
        normalized_padded,
        final_u8,
    ]


def add_arrow_between_axes(fig, left_ax, right_ax):
    left = left_ax.get_position()
    right = right_ax.get_position()
    y = (left.y0 + left.y1) / 2
    plt.annotate(
        "",
        xy=(right.x0 - 0.003, y),
        xytext=(left.x1 + 0.003, y),
        xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops=dict(
            arrowstyle="-|>",
            color="#777777",
            lw=1.25,
            mutation_scale=12,
        ),
        annotation_clip=False,
    )


def add_flow_box(ax, x, y, width, height, text, face, edge, fontsize=9):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.25,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=DARK,
        fontsize=fontsize,
        linespacing=1.25,
    )


def add_flow_arrow(ax, x1, y1, x2, y2, color="#777777"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.25,
            mutation_scale=12,
        ),
    )


def main():
    if not EXAMPLE_DICOM.exists():
        raise FileNotFoundError(f"Example DICOM not found: {EXAMPLE_DICOM}")

    stages = preprocessing_stages(EXAMPLE_DICOM)
    titles = [
        "1  DICOM read",
        "2  VOI LUT",
        "3  Intensity clipping",
        "4  CLAHE",
        "5  Gaussian blur",
        "6  Normalize & pad",
        "7  Resize & cast",
    ]
    subtitles = [
        "Rescale slope/intercept",
        "MONOCHROME1 inversion",
        "1st–99th percentile",
        "clip 2.0 · grid 8×8",
        "3×3 kernel · σ=0.5",
        "[0,1] · zero-pad square",
        "512×512 · uint8",
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(14, 7.2), facecolor="white")
    image_grid = fig.add_gridspec(
        1,
        7,
        left=0.035,
        right=0.985,
        bottom=0.48,
        top=0.86,
        wspace=0.16,
    )

    fig.text(
        0.025,
        0.945,
        "A",
        fontsize=15,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    fig.text(
        0.055,
        0.945,
        "Deterministic offline DICOM preprocessing",
        fontsize=14,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    fig.text(
        0.055,
        0.905,
        "The same sequence is applied independently to PA and lateral radiographs.",
        fontsize=9.5,
        color=MID_GRAY,
        va="top",
    )

    image_axes = []
    for index, (image, title, subtitle) in enumerate(
        zip(stages, titles, subtitles)
    ):
        axis = fig.add_subplot(image_grid[0, index])
        if index == 0:
            low, high = np.percentile(image, [1, 99])
            axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        else:
            axis.imshow(image, cmap="gray")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(BLUE if index in (0, 6) else "#A8A8A8")
            spine.set_linewidth(1.2 if index in (0, 6) else 0.7)
        axis.set_title(
            title,
            fontsize=9.5,
            fontweight="bold",
            color=DARK,
            pad=8,
        )
        axis.text(
            0.5,
            -0.075,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.6,
            color=MID_GRAY,
        )
        image_axes.append(axis)

    for left_axis, right_axis in zip(image_axes[:-1], image_axes[1:]):
        add_arrow_between_axes(fig, left_axis, right_axis)

    flow_ax = fig.add_axes([0.025, 0.065, 0.96, 0.31])
    flow_ax.set_xlim(0, 1)
    flow_ax.set_ylim(0, 1)
    flow_ax.axis("off")

    flow_ax.text(
        0.0,
        1.02,
        "B",
        fontsize=15,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    flow_ax.text(
        0.032,
        1.02,
        "Model input transforms",
        fontsize=14,
        fontweight="bold",
        color=DARK,
        va="top",
    )

    start_x, start_w = 0.035, 0.16
    box_w, box_h = 0.17, 0.22
    row_y_train, row_y_test = 0.53, 0.14
    x_aug, x_rgb, x_norm, x_model = 0.285, 0.49, 0.695, 0.88

    add_flow_box(
        flow_ax,
        start_x,
        0.335,
        start_w,
        0.26,
        "Cached image\n512×512 grayscale",
        LIGHT_BLUE,
        BLUE,
        fontsize=9.2,
    )

    flow_ax.text(
        0.225,
        row_y_train + box_h / 2,
        "Training",
        ha="right",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=TEAL,
    )
    flow_ax.text(
        0.225,
        row_y_test + box_h / 2,
        "Inference",
        ha="right",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=PURPLE,
    )

    add_flow_box(
        flow_ax,
        x_aug,
        row_y_train,
        box_w,
        box_h,
        "Random augmentation\nγ 0.98–1.02 · contrast 0.95–1.05\nnoise σ 0–0.01",
        LIGHT_TEAL,
        TEAL,
        fontsize=8.1,
    )
    add_flow_box(
        flow_ax,
        x_aug,
        row_y_test,
        box_w,
        box_h,
        "No augmentation",
        LIGHT_PURPLE,
        PURPLE,
    )
    for row_y in (row_y_train, row_y_test):
        add_flow_box(
            flow_ax,
            x_rgb,
            row_y,
            box_w,
            box_h,
            "Grayscale → RGB\nchannel replication",
            LIGHT_GRAY,
            "#888888",
        )
        add_flow_box(
            flow_ax,
            x_norm,
            row_y,
            box_w,
            box_h,
            "ImageNet normalization\nμ=(0.485, 0.456, 0.406)\nσ=(0.229, 0.224, 0.225)",
            LIGHT_GOLD,
            GOLD,
            fontsize=7.9,
        )

    add_flow_box(
        flow_ax,
        x_model,
        0.335,
        0.11,
        0.26,
        "Model\ninput",
        LIGHT_BLUE,
        BLUE,
        fontsize=9.5,
    )

    start_right = start_x + start_w
    add_flow_arrow(
        flow_ax,
        start_right,
        0.465,
        x_aug,
        row_y_train + box_h / 2,
        color=TEAL,
    )
    add_flow_arrow(
        flow_ax,
        start_right,
        0.465,
        x_aug,
        row_y_test + box_h / 2,
        color=PURPLE,
    )
    for row_y, color in (
        (row_y_train, TEAL),
        (row_y_test, PURPLE),
    ):
        center_y = row_y + box_h / 2
        add_flow_arrow(
            flow_ax,
            x_aug + box_w,
            center_y,
            x_rgb,
            center_y,
            color=color,
        )
        add_flow_arrow(
            flow_ax,
            x_rgb + box_w,
            center_y,
            x_norm,
            center_y,
            color=color,
        )
        add_flow_arrow(
            flow_ax,
            x_norm + box_w,
            center_y,
            x_model,
            0.465,
            color=color,
        )

    flow_ax.text(
        start_x,
        0.02,
        "Offline outputs are cached once; stochastic augmentation is applied on-the-fly only during training.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MID_GRAY,
        style="italic",
    )

    png_path = FIGURE_DIR / "fig_preprocessing_pipeline.png"
    pdf_path = FIGURE_DIR / "fig_preprocessing_pipeline.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
