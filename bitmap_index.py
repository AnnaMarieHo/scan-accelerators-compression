import time
import numpy as np


class BitmapIndex:
    def __init__(self, precomputed_bitmap, column_values):
        self.precomputed_bitmap = precomputed_bitmap
        self.column_values = np.asarray(column_values)

    def bitmap_index_scan(self, value):
        start_time = time.time()

        if value not in self.precomputed_bitmap:
            return None

        bitset = np.asarray(self.precomputed_bitmap[value], dtype=bool)
        actual_values = self.column_values[bitset].tolist()
        end_time = time.time()
        return {
            "values": actual_values,
            "bitset": bitset,
            "metrics": {
                "query_time": end_time - start_time,
                "skip_ratio": 1.0,
                "segments_skipped": "N/A",
            }
        }


