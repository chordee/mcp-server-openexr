"""MCP tool definitions for the OpenEXR server."""

import asyncio
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from exr_reader import ExrReader
from tx_reader import TxReader

mcp = FastMCP("mcp-server-openexr")
EXR = ExrReader()
TX = TxReader()


async def _handle_errors(coro):
    """Wrap a coroutine and convert exceptions into error dicts."""
    try:
        return await coro
    except FileNotFoundError as e:
        return {"error": "FileNotFoundError", "message": str(e)}
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


@mcp.tool()
async def get_exr_info(
    file_path: Annotated[str, Field(description="Absolute path to the EXR file")],
) -> dict:
    """
    Return basic EXR file info: resolution, part count, channel list, and compression.
    Good starting point for exploring an unfamiliar EXR file.
    """
    return await _handle_errors(
        asyncio.to_thread(EXR.get_file_info, file_path)
    )


@mcp.tool()
async def get_exr_header(
    file_path: Annotated[str, Field(description="Absolute path to the EXR file")],
    part_index: Annotated[int, Field(description="Part index (0-based)", ge=0)] = 0,
) -> dict:
    """
    Return all header attributes for the given part, including custom metadata
    such as Houdini render settings and color space information.
    """
    return await _handle_errors(
        asyncio.to_thread(EXR.get_header, file_path, part_index)
    )


@mcp.tool()
async def get_exr_channels(
    file_path: Annotated[str, Field(description="Absolute path to the EXR file")],
    part_index: Annotated[int, Field(description="Part index (0-based)", ge=0)] = 0,
) -> dict:
    """
    Return channel details for the given part: pixel type (HALF/FLOAT/UINT) and sampling.
    """
    return await _handle_errors(
        asyncio.to_thread(EXR.get_channels, file_path, part_index)
    )


@mcp.tool()
async def get_exr_pixel_stats(
    file_path: Annotated[str, Field(description="Absolute path to the EXR file")],
    channels: Annotated[
        list[str] | None,
        Field(description="Channel names to compute stats for; null means all channels")
    ] = None,
    part_index: Annotated[int, Field(description="Part index (0-based)", ge=0)] = 0,
    ignore_nan: Annotated[bool, Field(description="Exclude NaN/Inf values from statistics")] = True,
) -> dict:
    """
    Compute pixel statistics per channel: min, max, mean, percentiles (p25/p50/p75/p95),
    and NaN/Inf pixel counts. Useful for QC and brightness analysis.
    """
    return await _handle_errors(
        asyncio.to_thread(EXR.get_pixel_stats, file_path, channels, part_index, ignore_nan)
    )


@mcp.tool()
async def get_exr_sequence_info(
    directory: Annotated[str, Field(description="Directory containing the EXR sequence")],
    pattern: Annotated[str, Field(description="Filename glob pattern, e.g. '*.exr' or 'beauty.*.exr'")] = "*.exr",
    max_files: Annotated[int, Field(description="Maximum number of files to scan", ge=1, le=500)] = 50,
) -> dict:
    """
    Scan a directory for an EXR sequence. Reports frame range, missing frames,
    and any inconsistencies in resolution or channels across frames.
    """
    return await _handle_errors(
        asyncio.to_thread(EXR.get_sequence_info, directory, pattern, max_files)
    )


@mcp.tool()
async def compare_exr_channels(
    file_path_a: Annotated[str, Field(description="Absolute path to the first EXR file")],
    file_path_b: Annotated[str, Field(description="Absolute path to the second EXR file")],
    channels: Annotated[
        list[str] | None,
        Field(description="Channel names to compare; null means all common channels")
    ] = None,
    part_index: Annotated[int, Field(description="Part index (0-based)", ge=0)] = 0,
) -> dict:
    """
    Compare channels between two EXR files. Reports max, mean, and RMS differences.
    Useful for QC between different render versions.
    """
    return await _handle_errors(
        asyncio.to_thread(EXR.compare_channels, file_path_a, file_path_b, channels, part_index)
    )


@mcp.tool()
async def check_exr_validity(
    file_path: Annotated[str, Field(description="Absolute path to the EXR file")],
    check_pixels: Annotated[bool, Field(description="Scan pixel data to detect NaN/Inf")] = True,
    channels: Annotated[
        list[str] | None,
        Field(description="Channels to scan; null means all channels. Use this to limit I/O.")
    ] = None,
) -> dict:
    """
    Validate that an EXR file can be opened and optionally scan for NaN/Inf pixels.
    By default scans all channels; specify channels to reduce I/O on large files.
    """
    return await _handle_errors(
        asyncio.to_thread(EXR.check_validity, file_path, check_pixels, channels)
    )


@mcp.tool()
async def extract_exr_part(
    file_path: Annotated[str, Field(description="Absolute path to the source EXR file")],
    part_name: Annotated[
        str | None,
        Field(description="Name of the part to extract (e.g. 'C', 'AO'). Takes precedence over part_index.")
    ] = None,
    part_index: Annotated[
        int | None,
        Field(description="Index of the part to extract (0-based). Used when part_name is not specified.", ge=0)
    ] = None,
) -> dict:
    """
    Extract a single part from a multi-part EXR and save it as a new single-part EXR file.

    Specify the target part by name (e.g. 'C', 'AO') or by index. Name takes precedence.
    Output is written to <source_dir>/<part_name>/<source_filename>.
    The output directory must not already exist (non-destructive).
    Only essential header attributes are carried over; renderer-specific metadata
    (render times, camera matrices, etc.) is omitted since the source file retains
    the full record.
    """
    return await _handle_errors(
        asyncio.to_thread(EXR.extract_part, file_path, part_index, part_name)
    )


@mcp.tool()
async def get_tx_info(
    file_path: Annotated[str, Field(description="Absolute path to the TX file")],
) -> dict:
    """
    Return basic TX file info: resolution, MIP pyramid, channel list, tile size,
    color space, and compression. Good starting point for inspecting a maketx texture.
    """
    return await _handle_errors(
        asyncio.to_thread(TX.get_file_info, file_path)
    )


@mcp.tool()
async def get_tx_header(
    file_path: Annotated[str, Field(description="Absolute path to the TX file")],
    mip_level: Annotated[int, Field(description="MIP level to read (0 = full resolution)", ge=0)] = 0,
) -> dict:
    """
    Return all metadata attributes for the given MIP level, including color space,
    texture format, software tag, and any custom maketx/Arnold metadata.
    """
    return await _handle_errors(
        asyncio.to_thread(TX.get_header, file_path, mip_level)
    )


@mcp.tool()
async def get_tx_channels(
    file_path: Annotated[str, Field(description="Absolute path to the TX file")],
    mip_level: Annotated[int, Field(description="MIP level to read (0 = full resolution)", ge=0)] = 0,
) -> dict:
    """
    Return channel details for the TX file: channel names, pixel types (HALF/FLOAT/UINT8),
    and tiling information. Channels are identical across all MIP levels.
    """
    return await _handle_errors(
        asyncio.to_thread(TX.get_channels, file_path, mip_level)
    )


@mcp.tool()
async def get_tx_pixel_stats(
    file_path: Annotated[str, Field(description="Absolute path to the TX file")],
    channels: Annotated[
        list[str] | None,
        Field(description="Channel names to compute stats for; null means all channels")
    ] = None,
    mip_level: Annotated[
        int,
        Field(description="MIP level to sample (0 = full resolution, higher = smaller/faster)", ge=0)
    ] = 0,
    ignore_nan: Annotated[bool, Field(description="Exclude NaN/Inf values from statistics")] = True,
) -> dict:
    """
    Compute pixel statistics per channel at the specified MIP level: min, max, mean,
    percentiles (p25/p50/p75/p95), and NaN/Inf counts. Use a higher MIP level
    (e.g. mip_level=4) for fast approximate statistics on large textures.
    """
    return await _handle_errors(
        asyncio.to_thread(TX.get_pixel_stats, file_path, channels, mip_level, ignore_nan)
    )


@mcp.tool()
async def get_tx_sequence_info(
    directory: Annotated[str, Field(description="Directory containing the TX sequence")],
    pattern: Annotated[str, Field(description="Filename glob pattern, e.g. '*.tx' or 'diffuse.*.tx'")] = "*.tx",
    max_files: Annotated[int, Field(description="Maximum number of files to scan", ge=1, le=500)] = 50,
) -> dict:
    """
    Scan a directory for a TX texture sequence. Reports frame range, missing frames,
    and inconsistencies in resolution, channels, MIP level count, or tile size across frames.
    """
    return await _handle_errors(
        asyncio.to_thread(TX.get_sequence_info, directory, pattern, max_files)
    )


if __name__ == "__main__":
    mcp.run()
