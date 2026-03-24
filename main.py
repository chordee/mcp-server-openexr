"""MCP tool definitions for the OpenEXR server."""

import asyncio
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from exr_reader import ExrReader

mcp = FastMCP("mcp-server-openexr")
EXR = ExrReader()


async def _handle_exr_errors(coro):
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
    return await _handle_exr_errors(
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
    return await _handle_exr_errors(
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
    return await _handle_exr_errors(
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
    return await _handle_exr_errors(
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
    return await _handle_exr_errors(
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
    return await _handle_exr_errors(
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
    return await _handle_exr_errors(
        asyncio.to_thread(EXR.check_validity, file_path, check_pixels, channels)
    )


if __name__ == "__main__":
    mcp.run()
