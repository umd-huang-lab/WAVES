import os
import warnings
import stat
import orjson
import gzip
import base64
from io import BytesIO
import numpy as np
from PIL import Image


def chmod_group_write(path):
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")
    if os.stat(path).st_uid == os.getuid():
        current_permissions = stat.S_IMODE(os.lstat(path).st_mode)
        os.chmod(path, current_permissions | stat.S_IWGRP)


def compare_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not compare_dicts(dict1[key], dict2[key]):
                return False
        else:
            if dict1[key] != dict2[key]:
                return False
    return True


def load_json(filepath):
    try:
        with open(filepath, "rb") as json_file:
            return orjson.loads(json_file.read())
    except orjson.JSONDecodeError:
        warnings.warn(f"Found invalid JSON file {filepath}, deleting")
        os.remove(filepath)
        return None


def save_json(data, filepath):
    if os.path.exists(filepath) and (existing_data := load_json(filepath)) is not None:
        if compare_dicts(data, existing_data):
            return
    with open(filepath, "wb") as json_file:
        json_file.write(orjson.dumps(data))
    chmod_group_write(filepath)


def encode_array_to_string(array):
    # Convert shape and dtype to byte string using orjson
    meta = orjson.dumps({"shape": array.shape, "dtype": str(array.dtype)})
    # Combine metadata and array bytes
    combined = meta + b"\x00" + array.tobytes()
    # Compress and encode to Base64
    compressed = gzip.compress(combined)
    return base64.b64encode(compressed).decode("utf-8")


def decode_array_from_string(encoded_string):
    # Decode from Base64 and decompress
    decoded_bytes = base64.b64decode(encoded_string)
    decompressed = gzip.decompress(decoded_bytes)
    # Split metadata and array data
    meta_encoded, array_bytes = decompressed.split(b"\x00", 1)
    # Deserialize metadata
    meta = orjson.loads(meta_encoded)
    shape, dtype = meta["shape"], meta["dtype"]
    # Convert bytes back to NumPy array
    return np.frombuffer(array_bytes, dtype=dtype).reshape(shape)


def encode_image_to_string(image, quality=90):
    # Save the image to a byte buffer in JPEG format
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    # Encode the buffer to a base64 string
    return base64.b64encode(gzip.compress(buffered.getvalue())).decode("utf-8")


def decode_image_from_string(encoded_string):
    # Decode the base64 string to bytes
    img_data = gzip.decompress(base64.b64decode(encoded_string))
    # Read the image from bytes
    image = Image.open(BytesIO(img_data))
    return image
