"""OpenEXR constants and enum lookup tables."""

import OpenEXR

# Compression enum -> human-readable string
COMPRESSION_NAMES: dict[int, str] = {
    OpenEXR.Compression.NO_COMPRESSION: "NO_COMPRESSION",
    OpenEXR.Compression.RLE_COMPRESSION: "RLE",
    OpenEXR.Compression.ZIPS_COMPRESSION: "ZIPS",
    OpenEXR.Compression.ZIP_COMPRESSION: "ZIP",
    OpenEXR.Compression.PIZ_COMPRESSION: "PIZ",
    OpenEXR.Compression.PXR24_COMPRESSION: "PXR24",
    OpenEXR.Compression.B44_COMPRESSION: "B44",
    OpenEXR.Compression.B44A_COMPRESSION: "B44A",
    OpenEXR.Compression.DWAA_COMPRESSION: "DWAA",
    OpenEXR.Compression.DWAB_COMPRESSION: "DWAB",
}

# Pixel type enum -> human-readable string
PIXEL_TYPE_NAMES: dict = {
    OpenEXR.PixelType.HALF: "HALF",
    OpenEXR.PixelType.FLOAT: "FLOAT",
    OpenEXR.PixelType.UINT: "UINT",
}

# Storage type enum -> human-readable string
STORAGE_TYPE_NAMES: dict = {
    OpenEXR.Storage.scanlineimage: "scanlineimage",
    OpenEXR.Storage.tiledimage: "tiledimage",
    OpenEXR.Storage.deepscanline: "deepscanline",
    OpenEXR.Storage.deeptile: "deeptile",
}


def compression_to_str(compression) -> str:
    """Convert a compression enum value to a string."""
    return COMPRESSION_NAMES.get(compression, str(compression).split(".")[-1])


def pixel_type_to_str(pixel_type) -> str:
    """Convert a pixel type enum value to a string."""
    return PIXEL_TYPE_NAMES.get(pixel_type, str(pixel_type).split(".")[-1])


def storage_type_to_str(storage_type) -> str:
    """Convert a storage type enum value to a string."""
    return STORAGE_TYPE_NAMES.get(storage_type, str(storage_type).split(".")[-1])
