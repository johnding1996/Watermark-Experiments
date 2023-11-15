import orjson
import gzip
import base64
from io import BytesIO
import numpy as np
from PIL import Image


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


DECODE_MODES = ["tree_ring", "stable_sig", "stegastamp"]

GROUND_TRUTH_MESSAGES = {
    "tree_ring": decode_array_from_string(
        "H4sIALRwUmUC/42SvYrCQBSFLW18iam3EZcUFgErkYBFSLcYENZgISgoiMjCVj6FzyFCCoUlTQgrE8TnkcNhcEZIbjhw7x3Ox/zdu1fr+XQ1U/0vr/c5+VDfmx1WKlksp5uup35anX+jLHrVWFHX0FS2cw2h8x+z69KBYs1MxvVjDXkPZpvh/iC8B1SUzKTMWTZTlEV5iBBtzqVAHKJEI4KrphL9OwInU1AzUqbku9W/s+mf1f+93D+5/1Vz8z5j+djIv79qrKj2wFS20x5Al5LZdelApxszGdc/3aBUM9sM9weRasgPmEmZs2zGD/xgmyPanEuB2ObHDBFcNXXMwiE4mYKakTIl363+nU3/rP7v5f7J/a+am/cZewKA1ipNFgUAAA=="
    ),
    "stable_sig": decode_array_from_string(
        "H4sIADtrUmUC/6tWKs5ILEhVsoo2sYjVUUopqQRxlJLy83OUahkYGRkZgBBMMIAAhAvmwUUZIRIgcQBxGJ0kTgAAAA=="
    ),
    "stegastamp": decode_array_from_string(
        "H4sIAGRrUmUC/6tWKs5ILEhVsoo2NDCI1VFKKakE8ZSS8vNzlGoZGBkZGRhAmAFCgxGIC+dBCAaYEEyCEVkOzISrYIToZIAoY2AEAG5jy4ODAAAA"
    ),
}

METRIC_MODES = []
