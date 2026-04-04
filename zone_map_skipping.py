import time
import numpy as np
from evaluate_predicates import predicate_evaluator

class ZoneMapSkipping:
    def __init__(self, storage):
        self.storage = storage

    def should_skip(self, segment, value, operator):
        
        s_min = segment["min"]
        s_max = segment["max"]

        if operator == ">":
            return s_max <= value
        elif operator == ">=":
            return s_max < value
        elif operator == "<":
            return s_min >= value
        elif operator == "<=":
            return s_min > value
        elif operator == "=":
            # Skip if the value is outside the segment's entire range
            return value < s_min or value > s_max
        elif operator == "!=":
            # Only skip if the entire segment is ONLY this value
            return s_min == value and s_max == value
        
        return False

    def zone_map_skipping(self, value, operator):
        # Use zone map skipping to scan the data
        start_time = time.time()
        segments_skipped = 0
        total_segments = len(self.storage)
        final_bitset = []
        actual_values = []

        row_ids = []
        wasted_segments = 0
        for segment in self.storage:
            if self.should_skip(segment, value, operator):
                segments_skipped += 1
                final_bitset.extend([False] * segment["count"])
            else:
                data = segment["data"]
                match = np.asarray(
                    predicate_evaluator.evaluate_predicate(data, value, operator),
                    dtype=bool,
                )
                final_bitset.extend(match.tolist())
                actual_values.extend(data[match].tolist())
                base = segment.get("base_row", 0)
                row_ids.extend((base + np.flatnonzero(match)).tolist())
                if not match.any():
                    wasted_segments += 1

        end_time = time.time()

        return {
            "values": actual_values,
            "row_ids": row_ids,
            "bitset": np.array(final_bitset),
            "metrics": {
                "query_time": end_time - start_time,
                "skip_ratio": segments_skipped / total_segments if total_segments else 0.0,
                "segments_skipped": segments_skipped,
                "wasted_segments": wasted_segments,
            },
        }

