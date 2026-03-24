"""OpenEXR 常數與枚舉對應表"""

import OpenEXR

# 壓縮格式枚舉 → 可讀字串
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

# Pixel type 枚舉 → 可讀字串
PIXEL_TYPE_NAMES: dict = {
    OpenEXR.PixelType.HALF: "HALF",
    OpenEXR.PixelType.FLOAT: "FLOAT",
    OpenEXR.PixelType.UINT: "UINT",
}

# Storage type 枚舉 → 可讀字串
STORAGE_TYPE_NAMES: dict = {
    OpenEXR.Storage.scanlineimage: "scanlineimage",
    OpenEXR.Storage.tiledimage: "tiledimage",
    OpenEXR.Storage.deepscanline: "deepscanline",
    OpenEXR.Storage.deeptile: "deeptile",
}


def compression_to_str(compression) -> str:
    """將壓縮格式枚舉轉為字串"""
    return COMPRESSION_NAMES.get(compression, str(compression).split(".")[-1])


def pixel_type_to_str(pixel_type) -> str:
    """將 pixel type 枚舉轉為字串"""
    return PIXEL_TYPE_NAMES.get(pixel_type, str(pixel_type).split(".")[-1])


def storage_type_to_str(storage_type) -> str:
    """將 storage type 枚舉轉為字串"""
    return STORAGE_TYPE_NAMES.get(storage_type, str(storage_type).split(".")[-1])
