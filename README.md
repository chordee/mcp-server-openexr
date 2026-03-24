# mcp-server-openexr

MCP Server，提供 Claude 直接查詢本地 OpenEXR 渲染檔的能力，包含 metadata、channel 資訊與像素統計。

## 環境需求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## 安裝

```bash
cd D:\dev\mcp-server-openexr
uv sync
```

## 啟動

```bash
uv run python main.py
```

## MCP 工具清單

| 工具 | 用途 |
| ---- | ---- |
| `get_exr_info` | EXR 基本資訊：解析度、part 數量、channel 清單、壓縮格式 |
| `get_exr_header` | 完整 header attributes（含自訂屬性，如 Houdini 渲染設定） |
| `get_exr_channels` | Channel 詳細資訊（pixel_type, sampling） |
| `get_exr_pixel_stats` | 像素統計：min/max/mean、NaN/Inf 計數 |
| `get_exr_sequence_info` | 掃描目錄中的 EXR 序列，檢查缺幀與一致性 |
| `compare_exr_channels` | 比較兩個 EXR 的 channel 差異（QC 用途） |
| `check_exr_validity` | 驗證 EXR 可否開啟、偵測 NaN/Inf 問題 |

## Claude Desktop 設定

在 `claude_desktop_config.json` 中加入：

```json
{
  "mcpServers": {
    "openexr": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "D:\\dev\\mcp-server-openexr",
        "python",
        "main.py"
      ]
    }
  }
}
```
