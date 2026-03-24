"""MCP Server for OpenEXR — MCP 工具定義層"""

import asyncio
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from exr_reader import ExrReader

mcp = FastMCP("mcp-server-openexr")
EXR = ExrReader()


async def _handle_exr_errors(coro):
    """統一錯誤處理包裝"""
    try:
        return await coro
    except FileNotFoundError as e:
        return {"error": "FileNotFoundError", "message": str(e)}
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


@mcp.tool()
async def get_exr_info(
    file_path: Annotated[str, Field(description="EXR 檔案的完整路徑")],
) -> dict:
    """
    取得 EXR 檔案的基本資訊，包括解析度、part 數量、channel 清單與壓縮格式。
    適合作為探索 EXR 檔案的第一步。
    """
    return await _handle_exr_errors(
        asyncio.to_thread(EXR.get_file_info, file_path)
    )


@mcp.tool()
async def get_exr_header(
    file_path: Annotated[str, Field(description="EXR 檔案的完整路徑")],
    part_index: Annotated[int, Field(description="Part 索引，從 0 開始", ge=0)] = 0,
) -> dict:
    """
    取得 EXR 指定 part 的完整 header attributes，包含自訂屬性（如 Houdini 渲染設定、色彩空間資訊）。
    """
    return await _handle_exr_errors(
        asyncio.to_thread(EXR.get_header, file_path, part_index)
    )


@mcp.tool()
async def get_exr_channels(
    file_path: Annotated[str, Field(description="EXR 檔案的完整路徑")],
    part_index: Annotated[int, Field(description="Part 索引，從 0 開始", ge=0)] = 0,
) -> dict:
    """
    取得 EXR 指定 part 的 channel 詳細資訊，包括 pixel type（HALF/FLOAT/UINT）與 sampling 設定。
    """
    return await _handle_exr_errors(
        asyncio.to_thread(EXR.get_channels, file_path, part_index)
    )


@mcp.tool()
async def get_exr_pixel_stats(
    file_path: Annotated[str, Field(description="EXR 檔案的完整路徑")],
    channels: Annotated[
        list[str] | None,
        Field(description="要統計的 channel 名稱清單；若為 null 則統計所有 channel")
    ] = None,
    part_index: Annotated[int, Field(description="Part 索引，從 0 開始", ge=0)] = 0,
    ignore_nan: Annotated[bool, Field(description="計算統計時是否忽略 NaN/Inf 值")] = True,
) -> dict:
    """
    計算 EXR 指定 channel 的像素統計資料，包括 min/max/mean 以及 NaN/Inf 像素計數。
    適合用於 QC 確認渲染結果是否正常。
    """
    return await _handle_exr_errors(
        asyncio.to_thread(EXR.get_pixel_stats, file_path, channels, part_index, ignore_nan)
    )


@mcp.tool()
async def get_exr_sequence_info(
    directory: Annotated[str, Field(description="包含 EXR 序列的目錄路徑")],
    pattern: Annotated[str, Field(description="檔名 glob 模式，例如 '*.exr' 或 'beauty.*.exr'")] = "*.exr",
    max_files: Annotated[int, Field(description="最多掃描的檔案數量", ge=1, le=500)] = 50,
) -> dict:
    """
    掃描目錄中的 EXR 序列，回報幀範圍、缺幀情況與各幀之間的一致性（解析度、channel）。
    適合用於渲染完成後的序列完整性確認。
    """
    return await _handle_exr_errors(
        asyncio.to_thread(EXR.get_sequence_info, directory, pattern, max_files)
    )


@mcp.tool()
async def compare_exr_channels(
    file_path_a: Annotated[str, Field(description="第一個 EXR 檔案的完整路徑")],
    file_path_b: Annotated[str, Field(description="第二個 EXR 檔案的完整路徑")],
    channels: Annotated[
        list[str] | None,
        Field(description="要比較的 channel 名稱清單；若為 null 則比較所有共同 channel")
    ] = None,
    part_index: Annotated[int, Field(description="Part 索引，從 0 開始", ge=0)] = 0,
) -> dict:
    """
    比較兩個 EXR 檔案的 channel 差異，回報最大差異、平均差異與 RMS 差異。
    適合用於比較不同版本渲染結果的 QC。
    """
    return await _handle_exr_errors(
        asyncio.to_thread(EXR.compare_channels, file_path_a, file_path_b, channels, part_index)
    )


@mcp.tool()
async def check_exr_validity(
    file_path: Annotated[str, Field(description="EXR 檔案的完整路徑")],
    check_pixels: Annotated[bool, Field(description="是否掃描像素資料以偵測 NaN/Inf")] = True,
) -> dict:
    """
    驗證 EXR 檔案是否可正常開啟，並可選擇性地掃描所有 channel 的像素資料以偵測 NaN/Inf 問題。
    適合用於渲染後的自動化 QC 檢查。
    """
    return await _handle_exr_errors(
        asyncio.to_thread(EXR.check_validity, file_path, check_pixels)
    )


if __name__ == "__main__":
    mcp.run()
