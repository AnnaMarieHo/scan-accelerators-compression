import time

class BitmapIndex:
    def __init__(self, precomputed_bitmap):
        self.precomputed_bitmap = precomputed_bitmap

    def bitmap_index_scan(self, value):
        start_time = time.time()

        if value not in self.precomputed_bitmap:
            return None
        
        bitset = self.precomputed_bitmap[value]
        end_time = time.time()
        return {
            "bitset": bitset,
            "metrics": {
                "query_time": end_time - start_time,
            }
        }


