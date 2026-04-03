"""DPX format constants and type helpers for OIIO-based DPX reading."""

from tx_types import serialize_metadata_value  # noqa: F401  re-exported for dpx_reader

# Known logarithmic transfer characteristic names (as returned by OIIO)
DPX_LOG_TRANSFERS: frozenset[str] = frozenset({
    "Printing density",
    "Logarithmic",
})

# OIIO Orientation integer -> human-readable string (TIFF/EXIF orientation)
ORIENTATION_NAMES: dict[int, str] = {
    1: "left to right, top to bottom",
    2: "right to left, top to bottom",
    3: "left to right, bottom to top",
    4: "right to left, bottom to top",
    5: "top to bottom, left to right",
    6: "top to bottom, right to left",
    7: "bottom to top, left to right",
    8: "bottom to top, right to left",
}


def orientation_to_str(orientation: int) -> str:
    """Convert an OIIO orientation integer to a human-readable string."""
    return ORIENTATION_NAMES.get(orientation, f"unknown ({orientation})")


def is_log_transfer(transfer: str | None) -> bool:
    """Return True if the transfer characteristic is logarithmic (e.g. Cineon log)."""
    if transfer is None:
        return False
    return transfer in DPX_LOG_TRANSFERS
