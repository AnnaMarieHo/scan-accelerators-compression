# Scan Accelerators & Compression

This project implements various database scan acceleration techniques, including Bitmap Indexing and Zone Map Skipping.

## Prerequisites

You will need **uv** installed on your machine. It is a fast Python package and project manager.

- **macOS/Linux:** `curl -LsSf https://astral.sh | sh`
- **Windows:** `powershell -c "irm https://astral.sh | iex"`

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com
   cd scan-accelerators-compression
   uv sync
   uv run generate_db.py
   uv run main.py
   ```

## Project Structure

`baseline.py:` Standard scan implementation.
`bitmap_index.py:` Implementation of bitmap-based acceleration.
`zone_map_skipping.py:` Implementation of zone maps for block skipping.
`generate_db.py:` Script to generate synthetic_data.csv.
`pyproject.toml:` Project metadata and dependencies.
