"""Convert EXR sequences to video using the bundled ffmpeg binary from imageio-ffmpeg."""

import fnmatch
import os
import re
import subprocess

import imageio_ffmpeg
import numpy as np
import OpenEXR

_CODEC_BASE = {
    "h264":        ["-c:v", "libx264",   "-pix_fmt", "yuv420p"],
    "h265":        ["-c:v", "libx265",   "-pix_fmt", "yuv420p"],
    "prores_hq":   ["-c:v", "prores_ks", "-profile:v", "3"],
    "prores_4444": ["-c:v", "prores_ks", "-profile:v", "4"],
}

_DEFAULT_CRF = {"h264": 18, "h265": 20}
_CRF_CODECS = {"h264", "h265"}

_EVEN_DIM_CODECS = {"h264", "h265"}

# ACEScg (AP1) ↔ linear Rec.709 matrices (ACES CTL specification)
_AP1_TO_REC709 = np.array([
    [ 1.7050509508950576, -0.1375690625174089, -0.09139290469178879],
    [-0.7022008975075449,  1.6407397736069257,  0.01694323600833710],
    [ 0.0317562669406936, -0.0407505738058769,  0.90449476825212970],
], dtype=np.float32)

_REC709_TO_AP1 = np.linalg.inv(_AP1_TO_REC709).astype(np.float32)

_VALID_COLORSPACES = ("linear", "acescg", "srgb")


def _sort_by_frame_number(filenames: list[str]) -> list[str]:
    _re = re.compile(r"(\d+)(?=\.\w+$)")

    def _key(f: str):
        m = _re.search(f)
        return (int(m.group(1)), f) if m else (float("inf"), f)

    return sorted(filenames, key=_key)


def _auto_detect_rgb_channels(available: set[str]) -> tuple[list[str] | None, str | None]:
    if all(c in available for c in ("R", "G", "B")):
        return ["R", "G", "B"], None
    for packed in ("RGBA", "RGB", "rgba", "rgb"):
        if packed in available:
            return [packed], None
    return None, f"Could not auto-detect RGB channels. Available: {sorted(available)}"


def _build_rgb_frame(
    part: OpenEXR.Part,
    ch_names: list[str],
) -> tuple[np.ndarray | None, str | None]:
    """Return (H, W, 3) float32 array, or (None, error_str).

    ch_names: 3 separate names ["R","G","B"] or 1 packed name with ≥3 components ["RGBA"].
    """
    available = part.channels.keys()

    if len(ch_names) == 1:
        name = ch_names[0]
        if name not in available:
            return None, f"Channel '{name}' not found. Available: {sorted(available)}"
        px = part.channels[name].pixels.astype(np.float32)
        if px.ndim != 3 or px.shape[2] < 3:
            return None, f"Channel '{name}' has shape {px.shape}; expected (H, W, ≥3)"
        return px[..., :3], None

    arrays = []
    for name in ch_names:
        if name not in available:
            return None, f"Channel '{name}' not found. Available: {sorted(available)}"
        px = part.channels[name].pixels.astype(np.float32)
        if px.ndim == 3:
            px = px[..., 0]
        arrays.append(px)
    return np.stack(arrays, axis=-1), None


def _apply_matrix(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    return (rgb.reshape(-1, 3) @ matrix.T).reshape(h, w, 3).astype(np.float32)


def _srgb_decode(rgb: np.ndarray) -> np.ndarray:
    """sRGB gamma → linear light."""
    rgb = rgb.astype(np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _srgb_encode(rgb: np.ndarray) -> np.ndarray:
    """Linear light → sRGB gamma."""
    rgb = rgb.astype(np.float32)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * (rgb ** (1.0 / 2.4)) - 0.055)


def _color_pipeline(
    rgb: np.ndarray,
    input_colorspace: str,
    output_colorspace: str,
    tonemap: str,
    exposure: float,
) -> tuple[np.ndarray | None, str | None]:
    """Convert (H, W, 3) float32 from input_colorspace to output_colorspace → uint8.

    Pipeline:
    1. Decode input gamma / primaries → linear Rec.709 working space
    2. Apply exposure
    3. Apply tone mapping
    4. Convert primaries to output space
    5. Encode output gamma
    6. Clamp [0, 1] → uint8
    """
    # 1. Bring everything into linear Rec.709
    if input_colorspace == "srgb":
        rgb = _srgb_decode(rgb)
    elif input_colorspace == "acescg":
        rgb = _apply_matrix(rgb, _AP1_TO_REC709)
    # linear: already in Rec.709 linear, no-op

    # 2. Exposure
    rgb = rgb * float(exposure)

    # 3. Tone mapping
    if tonemap == "reinhard":
        rgb = rgb / (1.0 + rgb)
    elif tonemap != "none":
        return None, f"tonemap must be 'none' or 'reinhard', got '{tonemap}'"

    # 4. Convert primaries to output space
    if output_colorspace == "acescg":
        rgb = _apply_matrix(rgb, _REC709_TO_AP1)
    # srgb / linear: stays in Rec.709

    # 5. Clamp and encode output gamma
    rgb = np.clip(rgb, 0.0, 1.0)
    if output_colorspace == "srgb":
        rgb = _srgb_encode(rgb)

    return (rgb * 255.0).round().astype(np.uint8), None


class ExrVideoWriter:
    def exr_sequence_to_video(
        self,
        directory: str,
        output_path: str,
        pattern: str = "*.exr",
        fps: float = 24.0,
        codec: str = "h264",
        crf: int | None = None,
        input_colorspace: str = "linear",
        output_colorspace: str = "srgb",
        tonemap: str = "none",
        exposure: float = 1.0,
        scale: float = 1.0,
        channels: list[str] | None = None,
        part_index: int = 0,
    ) -> dict:
        if not os.path.isdir(directory):
            return {"error": f"Directory not found: {directory}"}

        if codec not in _CODEC_BASE:
            return {"error": f"codec must be one of {list(_CODEC_BASE.keys())}, got '{codec}'"}

        if crf is not None:
            if codec not in _CRF_CODECS:
                return {"error": f"crf is only supported for h264/h265, not '{codec}'"}
            if not (0 <= crf <= 51):
                return {"error": f"crf must be between 0 and 51, got {crf}"}
        effective_crf = crf if crf is not None else _DEFAULT_CRF.get(codec)

        if input_colorspace not in _VALID_COLORSPACES:
            return {"error": f"input_colorspace must be one of {_VALID_COLORSPACES}, got '{input_colorspace}'"}
        if output_colorspace not in _VALID_COLORSPACES:
            return {"error": f"output_colorspace must be one of {_VALID_COLORSPACES}, got '{output_colorspace}'"}

        if tonemap not in ("none", "reinhard"):
            return {"error": f"tonemap must be 'none' or 'reinhard', got '{tonemap}'"}

        if scale <= 0:
            return {"error": "scale must be greater than 0"}

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.isdir(out_dir):
            return {"error": f"Output directory does not exist: {out_dir}"}

        matched = [
            f for f in os.listdir(directory)
            if fnmatch.fnmatch(f.lower(), pattern.lower())
        ]
        if not matched:
            return {"error": f"No files found matching '{pattern}' in {directory}"}

        sorted_files = _sort_by_frame_number(matched)

        first_path = os.path.join(directory, sorted_files[0])
        try:
            f0 = OpenEXR.File(first_path)
            if part_index >= len(f0.parts):
                return {"error": f"part_index {part_index} out of range (file has {len(f0.parts)} parts)"}
            p0 = f0.parts[part_index]
            W, H = p0.width(), p0.height()

            if channels:
                ch_names = channels
            else:
                ch_names, err = _auto_detect_rgb_channels(set(p0.channels.keys()))
                if err:
                    return {"error": err}

            _, err = _build_rgb_frame(p0, ch_names)
            if err:
                return {"error": err}
        except Exception as e:
            return {"error": f"Failed to read first frame '{sorted_files[0]}': {e}"}

        enc_W, enc_H = W - (W % 2), H - (H % 2)

        out_W = max(2, round(enc_W * scale))
        out_H = max(2, round(enc_H * scale))
        if codec in _EVEN_DIM_CODECS:
            out_W -= out_W % 2
            out_H -= out_H % 2

        vf_args = []
        if abs(scale - 1.0) > 1e-6:
            vf_args = ["-vf", f"scale={out_W}:{out_H}"]

        dim_notes = []
        if enc_W != W or enc_H != H:
            dim_notes.append(f"input cropped to {enc_W}x{enc_H} (even dimensions required)")
        if out_W != round(enc_W * scale) or out_H != round(enc_H * scale):
            dim_notes.append(f"output adjusted to {out_W}x{out_H} (even dimensions required)")

        codec_args = list(_CODEC_BASE[codec])
        if effective_crf is not None:
            codec_args += ["-crf", str(effective_crf)]

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{enc_W}x{enc_H}",
            "-r", str(fps),
            "-pix_fmt", "rgb24",
            "-i", "pipe:0",
            *vf_args,
            *codec_args,
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        frames_written = 0
        frame_errors = []

        for fname in sorted_files:
            fpath = os.path.join(directory, fname)
            try:
                f = OpenEXR.File(fpath)
                part = f.parts[part_index]
                rgb, err = _build_rgb_frame(part, ch_names)
                if err:
                    frame_errors.append(f"{fname}: {err}")
                    continue

                uint8_frame, err = _color_pipeline(
                    rgb, input_colorspace, output_colorspace, tonemap, exposure
                )
                if err:
                    proc.stdin.close()
                    proc.wait()
                    return {"error": err}

                proc.stdin.write(uint8_frame[:enc_H, :enc_W].tobytes())
                frames_written += 1
            except Exception as e:
                frame_errors.append(f"{fname}: {e}")

        proc.stdin.close()
        _, stderr_bytes = proc.communicate()

        if proc.returncode != 0:
            return {
                "error": "ffmpeg encoding failed",
                "stderr": stderr_bytes.decode(errors="replace")[-2000:],
            }

        result = {
            "output_path": output_path,
            "frames": frames_written,
            "input_resolution": f"{enc_W}x{enc_H}",
            "output_resolution": f"{out_W}x{out_H}",
            "fps": fps,
            "codec": codec,
            "crf": effective_crf,
            "input_colorspace": input_colorspace,
            "output_colorspace": output_colorspace,
            "tonemap": tonemap,
            "exposure": exposure,
            "scale": scale,
            "channels": ch_names,
        }
        if dim_notes:
            result["notes"] = dim_notes
        if frame_errors:
            result["frame_errors"] = frame_errors[:10]
        return result
