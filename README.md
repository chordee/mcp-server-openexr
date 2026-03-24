# mcp-server-openexr

MCP Server that gives Claude direct access to local OpenEXR files,
including metadata, channel info, and pixel statistics.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
git clone https://github.com/chordee/mcp-server-openexr.git
cd mcp-server-openexr
uv sync
```

## Running

```bash
uv run main.py
```

## Tools

| Tool | Description |
| ---- | ----------- |
| `get_exr_info` | Resolution, part count, channel list, compression |
| `get_exr_header` | Full header attributes including custom metadata |
| `get_exr_channels` | Channel pixel type (HALF/FLOAT/UINT) and sampling |
| `get_exr_pixel_stats` | Pixel statistics: min/max/mean and NaN/Inf counts |
| `get_exr_sequence_info` | Scan EXR sequences, detect missing frames |
| `compare_exr_channels` | Compare channel differences between two EXR files |
| `check_exr_validity` | Validate EXR integrity and detect NaN/Inf pixels |

## Claude Desktop Configuration

Add the following to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "openexr": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/mcp-server-openexr",
        "main.py"
      ]
    }
  }
}
```
