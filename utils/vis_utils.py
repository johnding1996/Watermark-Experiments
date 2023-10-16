import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from .data_utils import unnormalize_and_to_pil


# Visualize image grid
def visualize_image_grid(
    images, col_headers, row_headers, fontsize=10, column_first=False
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
            if num_rows > 1 and num_cols > 1:
                axs[i, j].imshow(image)
                axs[i, j].axis("off")
            else:
                ax = axs[i] if num_cols == 1 else axs[j]
                ax.imshow(image)
                ax.axis("off")
    # Column headers
    for j, col in enumerate(col_headers):
        fig.text(
            (2 * j + 2 * 0.85) / (2 * num_cols + 0.85),
            1.0,
            col,
            ha="center",
            va="center",
            fontsize=fontsize,
        )
    # Row headers
    for i, row in enumerate(row_headers):
        fig.text(
            0,
            (2 * num_rows - 2 * i - 0.75) / (2 * num_rows + 0.75),
            row,
            ha="right",
            va="center",
            fontsize=fontsize,
        )
    # Plot and return
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, left=0.1)
    return fig


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
    return fig


# Visualize a imagenet subset and check class names
def visualize_imagenet_subset(dataset, class_names, n_classes=5, n_samples_per_class=5):
    images = [
        [
            unnormalize_and_to_pil(
                dataset[
                    next(
                        idx
                        for idx, (_, label) in enumerate(dataset.imgs)
                        if label == cid
                    )
                    + sid
                ][0],
                norm_type="ImageNet",
            )[0]
            for sid in range(n_samples_per_class)
        ]
        for cid in range(n_classes)
    ]
    col_headers = ["Sample " + str(sid + 1) for sid in range(n_samples_per_class)]
    row_headers = [class_names[cid] for cid in range(n_classes)]
    return visualize_image_grid(images, col_headers, row_headers, fontsize=10)


# Make a GIF with list of matplotlib figures
def make_gif(figs, filepath, loop=1, duration=0.5):
    imageio_images = []
    for fig in figs:
        # Save figure to the stream in PNG format
        buf = BytesIO()
        # Save figure to the stream in PNG format
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        buf.seek(0)
        # Read image from the stream
        imageio_images.append(imageio.imread(buf))
    imageio.mimsave(filepath, imageio_images, loop=loop, duration=duration)
