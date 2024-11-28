# obsidian-brain MCP server

Give claude access to your obsidian brain

### Install

#### Claude Desktop

On MacOS: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
On Windows: `%APPDATA%/Claude/claude_desktop_config.json`

<details>
  <summary>Development/Unpublished Servers Configuration</summary>
  ```
  "mcpServers": {
    "obsidian-brain": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/shubhaankar/github.com/Shubhaankar-Sharma/obsidian_brain",
        "run",
        "obsidian-brain"
      ],
      "env": {
        "OBSIDIAN_MIND_PATH": "/path/to/obsidian/vault"
      }
    }
  }
  ```
</details>

<details>
  <summary>Published Servers Configuration</summary>
  ```
  "mcpServers": {
    "obsidian-brain": {
      "command": "uvx",
      "args": [
        "obsidian-brain"
      ],
      "env": {
        "OBSIDIAN_MIND_PATH": "/path/to/obsidian/vault"
      }
    }
  }
  ```
</details>

## Development

### Building and Publishing

To prepare the package for distribution:

1. Sync dependencies and update lockfile:
```bash
uv sync
```

2. Build package distributions:
```bash
uv build
```

This will create source and wheel distributions in the `dist/` directory.

3. Publish to PyPI:
```bash
uv publish
```

Note: You'll need to set PyPI credentials via environment variables or command flags:
- Token: `--token` or `UV_PUBLISH_TOKEN`
- Or username/password: `--username`/`UV_PUBLISH_USERNAME` and `--password`/`UV_PUBLISH_PASSWORD`

### Debugging

Since MCP servers run over stdio, debugging can be challenging. For the best debugging
experience, we strongly recommend using the [MCP Inspector](https://github.com/modelcontextprotocol/inspector).


You can launch the MCP Inspector via [`npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) with this command:

```bash
npx @modelcontextprotocol/inspector uv --directory /Users/shubhaankar/github.com/Shubhaankar-Sharma/obsidian_brain run obsidian-brain
```


Upon launching, the Inspector will display a URL that you can access in your browser to begin debugging.