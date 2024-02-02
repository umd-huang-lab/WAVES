import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from PIL import Image
from IPython import display
import time
import tempfile
from .image_utils import to_pil

plt.rcParams.update({"figure.max_open_warning": 0})


# Save figure to buffer
def save_figure_to_buffer(fig, dpi=140):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf


# Save figure to PIL
def save_figure_to_pil(fig, dpi=140):
    return Image.open(save_figure_to_buffer(fig, dpi))


# Save figure
def save_figure_to_file(fig, save_path, dpi=140):
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)


# Visualize image grid
def visualize_image_grid(
    images,
    col_headers,
    row_headers,
    fontsize=10,
    column_first=False,
    title=None,
    title_fontsize=10,
):
    # Subplot
    if column_first:
        images = [list(row) for row in zip(*images)]
    num_rows, num_cols = len(images), len(images[0])
    assert num_rows == len(row_headers) and num_cols == len(col_headers)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    # Show images
    for i in range(num_rows):
        for j in range(num_cols):
            image = images[i][j]
            ax = (
                axs[i][j]
                if (num_rows > 1 and num_cols > 1)
                else (axs[i] if num_cols == 1 else axs[j])
            )
            ax.imshow(image)
            ax.axis("off")
    plt.tight_layout()
    # Column headers
    for j, col in enumerate(col_headers):
        ax = (
            axs[0, j]
            if (num_rows > 1 and num_cols > 1)
            else (axs[j] if num_rows == 1 else axs[0])
        )
        bbox = ax.get_window_extent().transformed(fig.transFigure.inverted())
        x_center = (bbox.x0 + bbox.x1) / 2
        fig.text(x_center, 1.0, col, ha="center", va="center", fontsize=fontsize)
    # Row headers
    for i, row in enumerate(row_headers):
        ax = axs[i][0] if num_cols > 1 else axs[i]
        bbox = ax.get_window_extent().transformed(fig.transFigure.inverted())
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0, y_center, row, ha="right", va="center", fontsize=fontsize)
    # Determine the leftmost x-coordinate from the row headers
    leftmost_x = min(
        [t.get_window_extent().x0 for t in fig.texts if t.get_text() in row_headers]
    )
    title_x = leftmost_x / fig.dpi / fig.get_figwidth()
    # Determine the uppermost y-coordinate from the column headers
    uppermost_y = max(
        [t.get_window_extent().y1 for t in fig.texts if t.get_text() in col_headers]
    )
    title_y = (
        uppermost_y / fig.dpi / fig.get_figheight() + 0.01
    )  # adding a small offset
    # Add title
    if title:
        fig.text(
            title_x, title_y, title, ha="left", va="bottom", fontsize=title_fontsize
        )
    # Plot and return
    img = save_figure_to_pil(fig)
    plt.close(fig)
    return img


# Visualize image list
def visualize_image_list(images, titles, max_per_row=4, fontsize=10):
    assert len(images) == len(titles)
    # Calculate rows and columns based on max_per_row
    num_rows = (len(images) - 1) // max_per_row + 1
    num_cols = min(len(images), max_per_row)
    # Subplot
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    # Flatten axes for easier iteration
    if num_rows > 1 and num_cols > 1:
        axs = axs.ravel()
    elif num_rows == 1 or num_cols == 1:
        axs = [axs]
    # Show images with titles
    for ax, image, title in zip(axs, images, titles):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, fontsize=fontsize)
    # If there are more axes than images (due to setting max_per_row), hide the extra axes
    for i in range(len(images), len(axs)):
        axs[i].axis("off")
    # Plot and return
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, left=0.1)
    # Plot and return
    img = save_figure_to_pil(fig)
    plt.close(fig)
    return img


# Display the image in Jupyter notebook
def display_media(file_path):
    # Determine the file extension
    file_ext = file_path.split(".")[-1].lower()

    if file_ext in ["jpg", "jpeg", "png", "bmp"]:
        display.display(display.Image(filename=file_path))
    elif file_ext == "gif":
        # For GIFs, embed them in an HTML image tag with a unique timestamp to avoid caching issues
        display.display(display.HTML(f'<img src="{file_path}?{time.time()}">'))
    else:
        print(f"Unsupported file type: {file_ext}")


# Concatenate nested images
def concatenate_images(images, column_first=False, save_path=None, display=False):
    if column_first:
        images = [list(row) for row in zip(*images)]
    # Concatenate inner lists vertically
    vertical_concatenated_images = [np.vstack(img_list) for img_list in images]
    # Concatenate the vertically concatenated images horizontally
    concatenated_images = np.hstack(vertical_concatenated_images)
    # Create a new figure with the concatenated image
    fig, ax = plt.subplots(
        figsize=(concatenated_images.shape[1] / 140, concatenated_images.shape[0] / 140)
    )
    ax.axis("off")
    ax.imshow(concatenated_images, aspect="auto")

    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight", pad_inches=0)

    if display:
        if save_path:
            display_media(save_path)
        else:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmpfile:
                plt.savefig(tmpfile.name, dpi=140, bbox_inches="tight", pad_inches=0)
                display_media(tmpfile.name)


# Make a GIF with list of imaegs
def make_gif(images, save_path, loop=1, duration=0.5, display=False):
    imageio.mimsave(save_path, images, loop=loop, duration=duration)
    if display:
        display_media(save_path)


# Visualize a supervised dataset and check class names
def visualize_supervised_dataset(
    dataset, class_names, n_classes=5, n_samples_per_class=5, norm_type="naive"
):
    images = [
        [
            to_pil(
                dataset[
                    next(
                        idx
                        for idx, (_, label) in enumerate(dataset.imgs)
                        if label == cid
                    )
                    + sid
                ][0],
                norm_type=norm_type,
            )[0]
            for sid in range(n_samples_per_class)
        ]
        for cid in range(n_classes)
    ]
    col_headers = ["Sample " + str(sid + 1) for sid in range(n_samples_per_class)]
    row_headers = [class_names[cid] for cid in range(n_classes)]
    return visualize_image_grid(images, col_headers, row_headers, fontsize=10)
