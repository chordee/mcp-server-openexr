"""OpenImageIO type helpers for TX file parsing."""


def typedesc_to_str(typedesc) -> str:
    """Convert an oiio.TypeDesc to an uppercase string (e.g. 'HALF', 'FLOAT', 'UINT8')."""
    return str(typedesc).upper()


def serialize_metadata_value(value) -> object:
    """Convert a metadata value to a JSON-serializable Python object."""
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [serialize_metadata_value(x) for x in value]
    # numpy scalar types
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    try:
        return str(value)
    except Exception:
        return repr(value)
