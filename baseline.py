import time
import numpy as np
from evaluate_predicates import predicate_evaluator
class Baseline:
    def __init__(self, storage):
        self.storage = storage

    def baseline_scan(self, value, operator):
        start_time = time.time()
        segments_skipped = 0
        total_segments = len(self.storage)
        final_bitset = []
        actual_values = []

        for segment in self.storage:
            match = predicate_evaluator.evaluate_predicate(segment["data"], value, operator)
            match = np.asarray(match, dtype=bool)
            final_bitset.extend(match.tolist())
            actual_values.extend(segment["data"][match].tolist())

        end_time = time.time()

        return {
            "values": actual_values,
            "bitset": np.array(final_bitset),
            "metrics": {
                "query_time": end_time - start_time,
                "segments_scanned": len(self.storage),
                "skip_ratio": 0.0,
                "segments_skipped": 0,
            }
        }