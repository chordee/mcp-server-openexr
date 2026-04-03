"""Business logic layer for reading DPX files via OpenImageIO."""

import os
import numpy as np
import OpenImageIO as oiio

from tx_types import typedesc_to_str
from dpx_types import serialize_metadata_value, orientation_to_str, is_log_transfer


class DpxReader:
    def validate_file(self, file_path: str) -> tuple[bool, str | None]:
        """Check that the file exists, can be opened by OIIO, and has DPX metadata."""
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        if not os.path.isfile(file_path):
            return False, f"Path is not a file: {file_path}"
        inp = oiio.ImageInput.open(file_path)
        if inp is None:
            return False, f"Failed to open file: {oiio.geterror()}"
        spec = inp.spec()
        inp.close()
        if spec.getattribute("dpx:Version") is None:
            return False, f"File does not appear to be a DPX file: {file_path}"
        return True, None

    def get_file_info(self, file_path: str) -> dict:
        """Return basic DPX info: resolution, channel list, bit depth, transfer, and timecode."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        file_size = os.path.getsize(file_path)
        inp = oiio.ImageInput.open(file_path)
        try:
            spec = inp.spec()
        finally:
            inp.close()

        attribs = {pv.name: pv.value for pv in spec.extra_attribs}
        orientation_raw = attribs.get("Orientation")

        return {
            "file_path": file_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / 1024 / 1024, 3),
            "dpx_version": attribs.get("dpx:Version"),
            "width": spec.width,
            "height": spec.height,
            "nchannels": spec.nchannels,
            "channel_names": list(spec.channelnames),
            "pixel_format": typedesc_to_str(spec.format),
            "bits_per_sample": attribs.get("oiio:BitsPerSample"),
            "image_descriptor": attribs.get("dpx:ImageDescriptor"),
            "transfer": attribs.get("dpx:Transfer"),
            "colorimetric": attribs.get("dpx:Colorimetric"),
            "packing": attribs.get("dpx:Packing"),
            "orientation": orientation_to_str(orientation_raw) if orientation_raw is not None else None,
            "pixel_aspect_ratio": attribs.get("PixelAspectRatio"),
            "frame_rate": attribs.get("dpx:FrameRate"),
            "timecode": attribs.get("dpx:TimeCode"),
            "input_device": attribs.get("dpx:InputDevice"),
        }

    def get_header(self, file_path: str) -> dict:
        """Return all header metadata attributes for the DPX file."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        inp = oiio.ImageInput.open(file_path)
        try:
            spec = inp.spec()
        finally:
            inp.close()

        metadata = {}
        for pv in spec.extra_attribs:
            try:
                metadata[pv.name] = serialize_metadata_value(pv.value)
            except Exception as e:
                metadata[pv.name] = f"<serialization failed: {e}>"

        return {
            "file_path": file_path,
            "width": spec.width,
            "height": spec.height,
            "nchannels": spec.nchannels,
            "channel_names": list(spec.channelnames),
            "pixel_format": typedesc_to_str(spec.format),
            "metadata": metadata,
        }

    def get_pixel_stats(
        self,
        file_path: str,
        channels: list[str] | None = None,
        ignore_nan: bool = True,
    ) -> dict:
        """Compute pixel statistics per channel: min, max, mean, percentiles, NaN/Inf counts."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        inp = oiio.ImageInput.open(file_path)
        try:
            spec = inp.spec()
            available_channels = list(spec.channelnames)
            attribs = {pv.name: pv.value for pv in spec.extra_attribs}
            bits_per_sample = attribs.get("oiio:BitsPerSample")
            transfer = attribs.get("dpx:Transfer")

            if channels:
                missing = [c for c in channels if c not in available_channels]
                if missing:
                    return {"error": f"Channels not found: {missing}. Available: {available_channels}"}
                target_channels = channels
            else:
                target_channels = available_channels

            stats = {}
            for ch_name in target_channels:
                ch_idx = available_channels.index(ch_name)
                raw = inp.read_image(0, 0, ch_idx, ch_idx + 1, oiio.FLOAT)
                if raw is None:
                    stats[ch_name] = {"error": f"Failed to read pixel data: {oiio.geterror()}"}
                    continue

                flat = raw.flatten().astype(np.float64)
                nan_count = int(np.count_nonzero(np.isnan(flat)))
                inf_count = int(np.count_nonzero(np.isinf(flat)))
                total_pixels = flat.size

                valid = flat[np.isfinite(flat)] if ignore_nan else flat

                if valid.size == 0:
                    stats[ch_name] = {
                        "pixel_count": total_pixels,
                        "nan_count": nan_count,
                        "inf_count": inf_count,
                        "min": None,
                        "max": None,
                        "mean": None,
                        "percentiles": {"p25": None, "p50": None, "p75": None, "p95": None},
                        "warning": "All pixels are NaN or Inf",
                    }
                else:
                    p25, p50, p75, p95 = np.percentile(valid, [25, 50, 75, 95])
                    stats[ch_name] = {
                        "pixel_count": total_pixels,
                        "nan_count": nan_count,
                        "inf_count": inf_count,
                        "min": float(np.min(valid)),
                        "max": float(np.max(valid)),
                        "mean": float(np.mean(valid)),
                        "percentiles": {
                            "p25": float(p25),
                            "p50": float(p50),
                            "p75": float(p75),
                            "p95": float(p95),
                        },
                    }
        finally:
            inp.close()

        # Build stats_note to help users interpret the float values
        stats_note = None
        if bits_per_sample:
            note_parts = [
                f"Pixel values are normalized to 0-1 from {bits_per_sample}-bit integer data "
                f"(code range 0-{(2 ** bits_per_sample) - 1})."
            ]
            if is_log_transfer(transfer):
                note_parts.append(
                    f"Transfer is '{transfer}' (logarithmic); values are in log code space, not linear light."
                )
            stats_note = " ".join(note_parts)

        return {
            "file_path": file_path,
            "width": spec.width,
            "height": spec.height,
            "ignore_nan": ignore_nan,
            "bits_per_sample": bits_per_sample,
            "transfer": transfer,
            "stats_note": stats_note,
            "stats": stats,
        }
