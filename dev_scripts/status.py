import click
import os
from PIL import Image, ImageDraw
from dev import (
    LIMIT,
    check_file_existence,
    parse_image_dir_path,
    encode_image_to_string,
    save_json,
)


def create_placeholder_image(size=512):
    # Create a 512x512 image with 50% gray background
    image = Image.new("RGB", (size, size), (128, 128, 128))
    draw = ImageDraw.Draw(image)
    # Define the dark red color
    dark_red = (139, 0, 0)
    # Draw two lines to form the cross
    # Line from top-left to bottom-right
    draw.line((0, 0, 511, 511), fill=dark_red, width=10)
    # Line from top-right to bottom-left
    draw.line((511, 0, 0, 511), fill=dark_red, width=10)
    return image


def get_image_dir_thumbnails(path, sampled, limit=5000):
    thumbnails = []
    for i in range(limit):
        if i in sampled:
            image_path = os.path.join(path, f"{i}.png")
            if os.path.exists(image_path):
                thumbnails.append(encode_image_to_string(Image.open(image_path)))
            else:
                thumbnails.append(encode_image_to_string(create_placeholder_image()))
        else:
            thumbnails.append(None)
    return thumbnails


@click.command()
@click.option(
    "--path", "-p", type=str, default=os.getcwd(), help="Path to image directory"
)
@click.option("--dry", "-d", is_flag=True, default=False, help="Dry run")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode")
def main(path, dry, quiet, limit=LIMIT):
    dataset_name, _, _, _ = parse_image_dir_path(path, quiet=quiet)
    existences = check_file_existence(path, name_pattern="{}.png", limit=limit)
    if not quiet:
        print(f"Found {sum(existences)} images out of {limit}")
    thumbnails = get_image_dir_thumbnails(path, sampled=[0, 1, 10, 100], limit=limit)
    data = {}
    for i in range(limit):
        data[str(i)] = {"exist": existences[i], "thumbnail": thumbnails[i]}
    json_path = os.path.join(
        os.environ.get("RESULT_DIR"),
        dataset_name,
        f"{str(path).split('/')[-1]}-status.json",
    )
    save_json(data, json_path)
    if not quiet:
        print(f"Image directory status saved to {json_path}")


if __name__ == "__main__":
    main()
