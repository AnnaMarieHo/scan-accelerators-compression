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
            final_bitset.extend(match)
            actual_values.extend(match)
        end_time = time.time()

        return {
            "values": actual_values,
            "bitset": np.array(final_bitset),
            "metrics": {
                "query_time": end_time - start_time,
                "skip_ratio": segments_skipped / total_segments,
                "segments_skipped": segments_skipped,
            }
        }