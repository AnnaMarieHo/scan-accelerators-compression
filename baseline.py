import time
import numpy as np
from evaluate_predicates import predicate_evaluator
class Baseline:
    def __init__(self, storage):
        self.storage = storage

    def baseline_scan(self, value, operator):
        start_time = time.time()
        final_bitset = []
        actual_values = []

        row_ids = []
        bytes_scanned = 0
        for segment in self.storage:
            data = segment["data"]
            match = predicate_evaluator.evaluate_predicate(data, value, operator)
            match = np.asarray(match, dtype=bool)
            final_bitset.extend(match.tolist())
            actual_values.extend(data[match].tolist())
            base = segment.get("base_row", 0)
            row_ids.extend((base + np.flatnonzero(match)).tolist())
            try:
                bytes_scanned += int(data.memory_usage(deep=True))
            except Exception:
                bytes_scanned += data.to_numpy(dtype=object).nbytes

        end_time = time.time()

        return {
            "values": actual_values,
            "row_ids": row_ids,
            "bitset": np.array(final_bitset),
            "metrics": {
                "query_time": end_time - start_time,
                "segments_scanned": len(self.storage),
                "skip_ratio": 0.0,
                "segments_skipped": 0,
                "bytes_scanned": bytes_scanned,
            },
        }