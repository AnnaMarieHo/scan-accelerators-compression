import time

import numpy as np


class BitmapIndex:
    def __init__(self, precomputed_bitmap, column_values):
        self.precomputed_bitmap = precomputed_bitmap
        self.column_values = np.asarray(column_values)
        self.n = len(self.column_values)

    def bitmap_index_scan(self, value):
        start_time = time.perf_counter()

        if value not in self.precomputed_bitmap:
            return {
                "values": [],
                "row_ids": [],
                "bitset": np.zeros(self.n, dtype=bool),
                "metrics": {
                    "query_time": time.perf_counter() - start_time,
                    "bitmap_bytes_for_column": self.bitmap_column_bytes(),
                },
            }

        bitset = np.asarray(self.precomputed_bitmap[value], dtype=bool)
        row_ids = np.flatnonzero(bitset).tolist()
        actual_values = self.column_values[bitset].tolist()
        return {
            "values": actual_values,
            "row_ids": row_ids,
            "bitset": bitset,
            "metrics": {
                "query_time": time.perf_counter() - start_time,
                "bitmap_bytes_for_column": self.bitmap_column_bytes(),
            },
        }

    def bitmap_column_bytes(self):
        """Sum of numpy array storage for all equality bitmaps in this column."""
        return int(sum(np.asarray(a).nbytes for a in self.precomputed_bitmap.values()))

    @staticmethod
    def combine_and(*bitsets):
        """Element-wise AND across same-length boolean arrays (``numpy.logical_and.reduce``)."""
        if not bitsets:
            raise ValueError("combine_and requires at least one bitset")
        return np.logical_and.reduce(bitsets)

    @staticmethod
    def combine_or(*bitsets):
        """Element-wise OR across same-length boolean arrays (``numpy.logical_or.reduce``)."""
        if not bitsets:
            raise ValueError("combine_or requires at least one bitset")
        return np.logical_or.reduce(bitsets)

    @staticmethod
    def combine_not(bitset):
        """Element-wise boolean NOT (unary ``~`` on a NumPy bool array)."""
        return ~np.asarray(bitset, dtype=bool)
