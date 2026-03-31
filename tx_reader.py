"""Business logic layer for reading TX (OpenImageIO MIP-mapped texture) files."""

import os
import re
import fnmatch
import numpy as np
import OpenImageIO as oiio

from tx_types import typedesc_to_str, serialize_metadata_value


class TxReader:
    def validate_file(self, file_path: str) -> tuple[bool, str | None]:
        """Check that the file exists and can be opened by OIIO."""
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        if not os.path.isfile(file_path):
            return False, f"Path is not a file: {file_path}"
        inp = oiio.ImageInput.open(file_path)
        if inp is None:
            return False, f"Failed to open TX file: {oiio.geterror()}"
        inp.close()
        return True, None

    def _collect_mip_levels(self, inp) -> list:
        """Seek through all MIP levels of subimage 0 and return a list of ImageSpec."""
        specs = []
        level = 0
        while inp.seek_subimage(0, level):
            specs.append(inp.spec())
            level += 1
        return specs

    def _channel_format(self, spec, ch_idx: int) -> str:
        """Return the pixel format string for a specific channel index."""
        fmts = spec.channelformats
        if fmts and ch_idx < len(fmts):
            return typedesc_to_str(fmts[ch_idx])
        return typedesc_to_str(spec.format)

    def get_file_info(self, file_path: str) -> dict:
        """Return basic TX info: resolution, MIP pyramid, channel list, colorspace."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        file_size = os.path.getsize(file_path)
        inp = oiio.ImageInput.open(file_path)
        try:
            mip_specs = self._collect_mip_levels(inp)
        finally:
            inp.close()

        if not mip_specs:
            return {"error": "File contains no readable subimage/miplevel data"}

        base_spec = mip_specs[0]

        mip_levels = []
        for level, spec in enumerate(mip_specs):
            mip_levels.append({
                "level": level,
                "width": spec.width,
                "height": spec.height,
                "nchannels": spec.nchannels,
                "channel_names": list(spec.channelnames),
                "pixel_format": typedesc_to_str(spec.format),
                "tile_width": spec.tile_width,
                "tile_height": spec.tile_height,
            })

        # Extract common metadata from base spec
        attribs = {pv.name: pv.value for pv in base_spec.extra_attribs}

        return {
            "file_path": file_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / 1024 / 1024, 3),
            "format_name": attribs.get("format", "tiff"),
            "mip_level_count": len(mip_levels),
            "mip_levels": mip_levels,
            "colorspace": attribs.get("oiio:ColorSpace"),
            "compression": attribs.get("Compression") or attribs.get("compression"),
            "texture_format": attribs.get("textureformat"),
            "software": attribs.get("Software") or attribs.get("software"),
        }

    def get_header(self, file_path: str, mip_level: int = 0) -> dict:
        """Return all metadata attributes for the given MIP level."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        inp = oiio.ImageInput.open(file_path)
        try:
            mip_specs = self._collect_mip_levels(inp)
        finally:
            inp.close()

        if not mip_specs:
            return {"error": "File contains no readable subimage/miplevel data"}
        if mip_level >= len(mip_specs):
            return {"error": f"mip_level {mip_level} out of range (file has {len(mip_specs)} levels)"}

        spec = mip_specs[mip_level]
        metadata = {}
        for pv in spec.extra_attribs:
            try:
                metadata[pv.name] = serialize_metadata_value(pv.value)
            except Exception as e:
                metadata[pv.name] = f"<serialization failed: {e}>"

        return {
            "file_path": file_path,
            "mip_level": mip_level,
            "width": spec.width,
            "height": spec.height,
            "nchannels": spec.nchannels,
            "channel_names": list(spec.channelnames),
            "tile_width": spec.tile_width,
            "tile_height": spec.tile_height,
            "pixel_format": typedesc_to_str(spec.format),
            "metadata": metadata,
        }

    def get_channels(self, file_path: str, mip_level: int = 0) -> dict:
        """Return channel details: names, pixel types, and tiling info."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        inp = oiio.ImageInput.open(file_path)
        try:
            mip_specs = self._collect_mip_levels(inp)
        finally:
            inp.close()

        if not mip_specs:
            return {"error": "File contains no readable subimage/miplevel data"}
        if mip_level >= len(mip_specs):
            return {"error": f"mip_level {mip_level} out of range (file has {len(mip_specs)} levels)"}

        spec = mip_specs[mip_level]
        channels_info = {}
        for i, ch_name in enumerate(spec.channelnames):
            channels_info[ch_name] = {
                "index": i,
                "pixel_type": self._channel_format(spec, i),
            }

        return {
            "file_path": file_path,
            "mip_level": mip_level,
            "width": spec.width,
            "height": spec.height,
            "nchannels": spec.nchannels,
            "channels": channels_info,
            "tile_width": spec.tile_width,
            "tile_height": spec.tile_height,
            "is_tiled": spec.tile_width > 0,
        }

    def get_pixel_stats(
        self,
        file_path: str,
        channels: list[str] | None = None,
        mip_level: int = 0,
        ignore_nan: bool = True,
    ) -> dict:
        """Compute pixel statistics per channel at the specified MIP level."""
        ok, err = self.validate_file(file_path)
        if not ok:
            return {"error": err}

        inp = oiio.ImageInput.open(file_path)
        try:
            mip_specs = self._collect_mip_levels(inp)
            if not mip_specs:
                return {"error": "File contains no readable subimage/miplevel data"}
            if mip_level >= len(mip_specs):
                return {"error": f"mip_level {mip_level} out of range (file has {len(mip_specs)} levels)"}

            spec = mip_specs[mip_level]
            available_channels = list(spec.channelnames)

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
                raw = inp.read_image(0, mip_level, ch_idx, ch_idx + 1, oiio.FLOAT)
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

        return {
            "file_path": file_path,
            "mip_level": mip_level,
            "mip_level_width": spec.width,
            "mip_level_height": spec.height,
            "ignore_nan": ignore_nan,
            "stats": stats,
        }

    def get_sequence_info(
        self,
        directory: str,
        pattern: str = "*.tx",
        max_files: int = 50,
    ) -> dict:
        """Scan a directory for a TX sequence and check consistency across frames."""
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
                "message": "No matching TX files found",
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

        reference_info = None
        inconsistencies = []
        files_info = []

        for fname in scanned_files:
            fpath = os.path.join(directory, fname)
            inp = oiio.ImageInput.open(fpath)
            if inp is None:
                files_info.append({"filename": fname, "error": oiio.geterror()})
                continue
            try:
                if not inp.seek_subimage(0, 0):
                    files_info.append({"filename": fname, "error": "Cannot seek to subimage 0"})
                    continue
                spec = inp.spec()

                # Count MIP levels
                mip_count = 0
                level = 0
                while inp.seek_subimage(0, level):
                    mip_count += 1
                    level += 1

                info = {
                    "filename": fname,
                    "width": spec.width,
                    "height": spec.height,
                    "nchannels": spec.nchannels,
                    "channel_names": list(spec.channelnames),
                    "mip_level_count": mip_count,
                    "tile_width": spec.tile_width,
                    "tile_height": spec.tile_height,
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
                    if info["channel_names"] != reference_info["channel_names"]:
                        diffs.append("Channel list mismatch")
                    if info["mip_level_count"] != reference_info["mip_level_count"]:
                        diffs.append(
                            f"MIP level count mismatch: {info['mip_level_count']}"
                            f" vs {reference_info['mip_level_count']}"
                        )
                    if info["tile_width"] != reference_info["tile_width"] or info["tile_height"] != reference_info["tile_height"]:
                        diffs.append(
                            f"Tile size mismatch: {info['tile_width']}x{info['tile_height']}"
                            f" vs {reference_info['tile_width']}x{reference_info['tile_height']}"
                        )
                    if diffs:
                        inconsistencies.append({"filename": fname, "issues": diffs})

                files_info.append({
                    "filename": fname,
                    "width": info["width"],
                    "height": info["height"],
                    "nchannels": info["nchannels"],
                    "mip_level_count": info["mip_level_count"],
                    "file_size_bytes": info["file_size_bytes"],
                })
            except Exception as e:
                files_info.append({"filename": fname, "error": str(e)})
            finally:
                inp.close()

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
