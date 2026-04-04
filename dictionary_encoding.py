import time

import numpy as np
import pandas as pd


def _scalar_key(v):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


class DictionaryEncoding:
    def __init__(self, segments, column_name):
        self.column_name = column_name
        results = self.dictionary_encode(segments) 
        self.encoded_segments = results["encoded_segments"]
        self.dictionary = results["dictionary"]
        self.reverse_dict = results["reverse_dict"]
    
    def dictionary_encode(self, segments):
        all_data = pd.concat([seg["data"] for seg in segments])
        unique_vals = all_data.unique()

        dictionary = {_scalar_key(val): i for i, val in enumerate(unique_vals)}
        reverse_dict = {i: val for val, i in dictionary.items()}
        
        encoded_segments = []
        for segment in segments:
            encoded_data = (
                segment["data"].map(lambda x: dictionary[_scalar_key(x)]).reset_index(drop=True)
            )
            encoded_segments.append(
                {
                    "data": encoded_data,
                    "base_row": int(segment.get("base_row", 0)),
                    "min": encoded_data.min(),
                    "max": encoded_data.max(),
                    "count": len(encoded_data),
                }
            )
            
        return {
            "encoded_segments": encoded_segments,
            "dictionary": dictionary,
            "reverse_dict": reverse_dict,
        }

    def query_dictionary(self, target_value):
        start_time = time.perf_counter()
        segments_skipped = 0
        
        key = _scalar_key(target_value)
        if key not in self.dictionary:
            return {
                "matching_row_ids": [],
                "metrics": {
                    "query_time": time.perf_counter() - start_time,
                    "segments_skipped": 0,
                    "approx_encoded_bytes": self._approx_encoded_bytes(),
                },
            }

        target_code = self.dictionary[key]
        matching_row_ids = []

        for segment in self.encoded_segments:
            if target_code < segment["min"] or target_code > segment["max"]:
                segments_skipped += 1
                continue

            arr = segment["data"].to_numpy()
            base = segment["base_row"]
            local = np.flatnonzero(arr == target_code)
            matching_row_ids.extend((base + local).tolist())

        end_time = time.perf_counter()

        return {
            "matching_row_ids": matching_row_ids,
            "metrics": {
                "query_time": end_time - start_time,
                "segments_skipped": segments_skipped,
                "approx_encoded_bytes": self._approx_encoded_bytes(),
            },
        }

    def _approx_encoded_bytes(self):
        codes = sum(seg["data"].to_numpy().nbytes for seg in self.encoded_segments)
        # dictionary: code -> string key overhead approximate
        dict_ovh = sum(len(str(k)) + 16 for k in self.dictionary)
        return int(codes + dict_ovh)