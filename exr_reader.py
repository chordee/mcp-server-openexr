"""Business logic layer for reading OpenEXR files."""

import os
import re
import fnmatch
import numpy as np
import OpenEXR

from exr_types import compression_to_str, pixel_type_to_str, storage_type_to_str


def _serialize_header_value(v):
    """Recursively serialize a header value to a JSON-compatible type."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_serialize_header_value(item) for item in v]
    type_name = type(v).__name__
    if "Compression" in type_name:
        return compression_to_str(v)
    if "PixelType" in type_name:
        return pixel_type_to_str(v)
    if "Storage" in type_name:
        return storage_type_to_str(v)
    if "LineOrder" in type_name:
        return str(v).split(".")[-1].rstrip(": 0123456789>")
    # Channel object
    if "Channel" in type_name and hasattr(v, "name"):
        info: dict = {"name": v.name}
        try:
            info["pixel_type"] = pixel_type_to_str(v.type())
        except Exception:
            pass
        try:
            info["x_sampling"] = v.xSampling
            info["y_sampling"] = v.ySampling
        except Exception:
            pass
        return info
    # Primitive types pass through unchanged
    if isinstance(v, (int, float, str, bool)):
        return v
    try:
        return str(v)
    except Exception:
        return repr(v)


class ExrReader:
    def validate_file(self, file_path: str) -> tuple[bool, str | None]:
        """Check that the file exists and can be opened as an EXR."""
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        if not os.path.isfile(file_path):
            return False, f"Path is not a file: {file_path}"
        try:
            f = OpenEXR.File(file_path)
            _ = f.parts
            return True, None
        except Exception as e:
            return False, f"Failed to open EXR file: {e}"

    def get_file_info(self, file_path: str) -> dict:
        """Return basic file info: resolution, part count, channel list, compression."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        f = OpenEXR.File(file_path)
        parts = f.parts
        file_size = os.path.getsize(file_path)

        parts_info = []
        for i, p in enumerate(parts):
            channels = list(p.channels.keys())
            parts_info.append({
                "index": i,
                "name": p.name(),
                "width": p.width(),
                "height": p.height(),
                "type": storage_type_to_str(p.type()),
                "compression": compression_to_str(p.compression()),
                "channels": channels,
                "channel_count": len(channels),
            })

        return {
            "file_path": file_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / 1024 / 1024, 3),
            "part_count": len(parts),
            "parts": parts_info,
        }

    def get_header(self, file_path: str, part_index: int = 0) -> dict:
        """Return all header attributes for the given part."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        f = OpenEXR.File(file_path)
        parts = f.parts
        if part_index >= len(parts):
            return {"error": f"part_index {part_index} out of range (file has {len(parts)} parts)"}

        p = parts[part_index]
        raw_header = p.header

        serialized = {}
        for k, v in raw_header.items():
            try:
                serialized[k] = _serialize_header_value(v)
            except Exception as e:
                serialized[k] = f"<serialization failed: {e}>"

        return {
            "file_path": file_path,
            "part_index": part_index,
            "part_name": p.name(),
            "header": serialized,
        }

    def get_channels(self, file_path: str, part_index: int = 0) -> dict:
        """Return channel details (pixel type, sampling) for the given part."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        f = OpenEXR.File(file_path)
        parts = f.parts
        if part_index >= len(parts):
            return {"error": f"part_index {part_index} out of range (file has {len(parts)} parts)"}

        p = parts[part_index]
        channels_info = {}
        for ch_name, ch in p.channels.items():
            channels_info[ch_name] = {
                "pixel_type": pixel_type_to_str(ch.type()),
                "x_sampling": ch.xSampling,
                "y_sampling": ch.ySampling,
                "p_linear": ch.pLinear,
            }

        return {
            "file_path": file_path,
            "part_index": part_index,
            "part_name": p.name(),
            "width": p.width(),
            "height": p.height(),
            "channels": channels_info,
        }

    def get_pixel_stats(
        self,
        file_path: str,
        channels: list[str] | None = None,
        part_index: int = 0,
        ignore_nan: bool = True,
    ) -> dict:
        """Compute pixel statistics (min/max/mean/percentiles, NaN/Inf counts) per channel."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        f = OpenEXR.File(file_path)
        parts = f.parts
        if part_index >= len(parts):
            return {"error": f"part_index {part_index} out of range (file has {len(parts)} parts)"}

        p = parts[part_index]
        available_channels = list(p.channels.keys())

        if channels:
            missing = [c for c in channels if c not in available_channels]
            if missing:
                return {"error": f"Channels not found: {missing}. Available: {available_channels}"}
            target_channels = channels
        else:
            target_channels = available_channels

        stats = {}
        for ch_name in target_channels:
            ch = p.channels[ch_name]
            arr = ch.pixels

            if arr is None:
                stats[ch_name] = {"error": "Failed to read pixel data"}
                continue

            # Flatten to 1D for statistics
            flat = arr.flatten().astype(np.float64)

            nan_count = int(np.count_nonzero(np.isnan(flat)))
            inf_count = int(np.count_nonzero(np.isinf(flat)))
            total_pixels = flat.size

            if ignore_nan:
                valid = flat[np.isfinite(flat)]
            else:
                valid = flat

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

        return {
            "file_path": file_path,
            "part_index": part_index,
            "part_name": p.name(),
            "ignore_nan": ignore_nan,
            "stats": stats,
        }

    def get_sequence_info(
        self,
        directory: str,
        pattern: str = "*.exr",
        max_files: int = 50,
    ) -> dict:
        """Scan a directory for an EXR sequence and check consistency across frames."""
        if not os.path.isdir(directory):
            return {"error": f"Directory not found: {directory}"}

        all_files = sorted([
            f for f in os.listdir(directory)
            if fnmatch.fnmatch(f.lower(), pattern.lower())
        ])

        if not all_files:
            return {
                "directory": directory,
                "pattern": pattern,
                "file_count": 0,
                "files": [],
                "message": "No matching EXR files found",
            }

        scanned_files = all_files[:max_files]
        truncated = len(all_files) > max_files

        # Extract frame numbers from filenames
        frame_numbers = []
        frame_pattern = re.compile(r"(\d+)(?=\.\w+$)")
        for fname in scanned_files:
            m = frame_pattern.search(fname)
            if m:
                frame_numbers.append(int(m.group(1)))

        # Use the first file as reference for consistency checks
        reference_info = None
        inconsistencies = []
        files_info = []

        for fname in scanned_files:
            fpath = os.path.join(directory, fname)
            try:
                f = OpenEXR.File(fpath)
                parts = f.parts
                if parts:
                    p = parts[0]
                    info = {
                        "filename": fname,
                        "width": p.width(),
                        "height": p.height(),
                        "channels": sorted(p.channels.keys()),
                        "compression": compression_to_str(p.compression()),
                        "file_size_bytes": os.path.getsize(fpath),
                    }
                    if reference_info is None:
                        reference_info = info.copy()
                    else:
                        diffs = []
                        if info["width"] != reference_info["width"] or info["height"] != reference_info["height"]:
                            diffs.append(
                                f"Resolution mismatch: {info['width']}x{info['height']}"
                                f" vs {reference_info['width']}x{reference_info['height']}"
                            )
                        if info["channels"] != reference_info["channels"]:
                            diffs.append("Channel list mismatch")
                        if diffs:
                            inconsistencies.append({"filename": fname, "issues": diffs})
                    files_info.append(info)
                else:
                    files_info.append({"filename": fname, "error": "No parts found"})
            except Exception as e:
                files_info.append({"filename": fname, "error": str(e)})

        # Detect missing frames
        missing_frames = []
        if len(frame_numbers) >= 2:
            frame_set = set(frame_numbers)
            full_range = range(min(frame_numbers), max(frame_numbers) + 1)
            missing_frames = [f for f in full_range if f not in frame_set]

        return {
            "directory": directory,
            "pattern": pattern,
            "total_files_found": len(all_files),
            "scanned_files": len(scanned_files),
            "truncated": truncated,
            "frame_range": {
                "start": min(frame_numbers) if frame_numbers else None,
                "end": max(frame_numbers) if frame_numbers else None,
                "count": len(frame_numbers),
                "missing": missing_frames[:20],
            },
            "reference_info": reference_info,
            "is_consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies,
            "files": files_info,
        }

    def compare_channels(
        self,
        file_path_a: str,
        file_path_b: str,
        channels: list[str] | None = None,
        part_index: int = 0,
    ) -> dict:
        """Compare pixel differences between two EXR files, channel by channel."""
        for path in [file_path_a, file_path_b]:
            ok, err = self.validate_file(path)
            if not ok:
                return {"error": err}

        fa = OpenEXR.File(file_path_a)
        fb = OpenEXR.File(file_path_b)

        parts_a = fa.parts
        parts_b = fb.parts

        if part_index >= len(parts_a):
            return {"error": f"file_a: part_index {part_index} out of range"}
        if part_index >= len(parts_b):
            return {"error": f"file_b: part_index {part_index} out of range"}

        pa = parts_a[part_index]
        pb = parts_b[part_index]

        channels_a = set(pa.channels.keys())
        channels_b = set(pb.channels.keys())

        only_in_a = sorted(channels_a - channels_b)
        only_in_b = sorted(channels_b - channels_a)
        common = sorted(channels_a & channels_b)

        if channels:
            missing = [c for c in channels if c not in channels_a and c not in channels_b]
            if missing:
                return {"error": f"Channels not found in either file: {missing}"}
            compare_list = [c for c in channels if c in common]
        else:
            compare_list = common

        size_match = (pa.width() == pb.width() and pa.height() == pb.height())

        channel_diffs = {}
        if size_match and compare_list:
            for ch_name in compare_list:
                arr_a = pa.channels[ch_name].pixels.flatten().astype(np.float64)
                arr_b = pb.channels[ch_name].pixels.flatten().astype(np.float64)

                abs_diff = np.abs(arr_a - arr_b)
                valid_mask = np.isfinite(arr_a) & np.isfinite(arr_b)
                valid_diff = abs_diff[valid_mask]

                channel_diffs[ch_name] = {
                    "max_abs_diff": float(np.max(valid_diff)) if valid_diff.size > 0 else None,
                    "mean_abs_diff": float(np.mean(valid_diff)) if valid_diff.size > 0 else None,
                    "rms_diff": float(np.sqrt(np.mean(valid_diff ** 2))) if valid_diff.size > 0 else None,
                    "identical": bool(np.all(valid_diff == 0)),
                    "pixels_compared": int(valid_mask.sum()),
                }
        elif not size_match:
            channel_diffs = {
                "error": (
                    f"Resolution mismatch, cannot compare pixels: "
                    f"{pa.width()}x{pa.height()} vs {pb.width()}x{pb.height()}"
                )
            }

        return {
            "file_path_a": file_path_a,
            "file_path_b": file_path_b,
            "part_index": part_index,
            "size_a": {"width": pa.width(), "height": pa.height()},
            "size_b": {"width": pb.width(), "height": pb.height()},
            "size_match": size_match,
            "channels_only_in_a": only_in_a,
            "channels_only_in_b": only_in_b,
            "common_channels": common,
            "compared_channels": compare_list,
            "channel_diffs": channel_diffs,
        }

    def check_validity(
        self,
        file_path: str,
        check_pixels: bool = True,
        channels: list[str] | None = None,
    ) -> dict:
        """Validate that an EXR can be opened and optionally scan for NaN/Inf pixels."""
        if not os.path.exists(file_path):
            return {
                "file_path": file_path,
                "valid": False,
                "error": "File not found",
            }

        try:
            f = OpenEXR.File(file_path)
            parts = f.parts
        except Exception as e:
            return {
                "file_path": file_path,
                "valid": False,
                "error": f"Failed to open file: {e}",
            }

        if not parts:
            return {
                "file_path": file_path,
                "valid": False,
                "error": "EXR file contains no parts",
            }

        result = {
            "file_path": file_path,
            "valid": True,
            "part_count": len(parts),
            "file_size_bytes": os.path.getsize(file_path),
        }

        if check_pixels:
            pixel_issues = []
            for i, p in enumerate(parts):
                available = p.channels.keys()
                target = [c for c in channels if c in available] if channels else list(available)
                for ch_name in target:
                    ch = p.channels[ch_name]
                    try:
                        arr = ch.pixels.flatten().astype(np.float64)
                        nan_count = int(np.count_nonzero(np.isnan(arr)))
                        inf_count = int(np.count_nonzero(np.isinf(arr)))
                        if nan_count > 0 or inf_count > 0:
                            pixel_issues.append({
                                "part_index": i,
                                "channel": ch_name,
                                "nan_count": nan_count,
                                "inf_count": inf_count,
                                "total_pixels": arr.size,
                            })
                    except Exception as e:
                        pixel_issues.append({
                            "part_index": i,
                            "channel": ch_name,
                            "error": str(e),
                        })

            result["has_pixel_issues"] = len(pixel_issues) > 0
            result["pixel_issues"] = pixel_issues
        else:
            result["has_pixel_issues"] = None
            result["pixel_issues"] = []

        return result
