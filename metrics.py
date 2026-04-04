"""Light-weight byte and build estimates for the mini column store."""

import numpy as np


def bytes_raw_segments(segments):
    total = 0
    for seg in segments:
        d = seg["data"]
        try:
            total += int(d.memory_usage(deep=True))
        except Exception:
            total += int(d.to_numpy(dtype=object).nbytes)
    return total


def bytes_zone_map_metadata(num_segments):
    """Per segment: min + max as int64 style footprint (approx)."""
    return int(num_segments * 16)


def summarize_column_storage(segments):
    raw = bytes_raw_segments(segments)
    zm = bytes_zone_map_metadata(len(segments))
    return {"raw_data_bytes": raw, "zone_map_metadata_bytes": zm, "total_with_zonemap_bytes": raw + zm}
