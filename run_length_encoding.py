import time
import numpy as np
class RunLengthEncoding:
    def __init__(self, storage):
        self.storage = storage
        self.compressed_data = self.run_len_encoding()

    def run_len_encoding(self):
        compressed_data = []
        
        for segment in self.storage:
            data = segment["data"]
            if len(data) == 0:
                continue

            first_occurrence = data.iloc[0]
            start_index = 0
            run_length = 1
            for i in range(1, len(data)):
                if data.iloc[i] == first_occurrence:
                    run_length += 1
                else:
                    compressed_data.append((first_occurrence, start_index, run_length))
                    first_occurrence = data.iloc[i]
                    start_index = i
                    run_length = 1
            
            compressed_data.append((first_occurrence, start_index, run_length))
                
        return compressed_data

    def direct_count(self, value):
        start_time = time.time()
        total_count = sum(run_len for val, start, run_len in self.compressed_data if val == value)
        end_time = time.time()
        return {
            "total_count": total_count,
            "metrics": {
                "query_time": end_time - start_time,
            }
        }
