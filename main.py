import pandas as pd
import numpy as np
import json
import time
import sys
# from evaluate_predicates import predicate_evaluator
from zone_map_skipping import ZoneMapSkipping
from baseline import Baseline
from bitmap_index import BitmapIndex

class MiniColumnStore:
    def __init__(self, csv_path, segment_size=1024):
        # Use pandas to ingest data from csv
        self.df = pd.read_csv(csv_path)
        self.segment_size = segment_size
        self.columns = {} # dict of numpy arrays
        self.storage = {} # dict of metadata and numpy arrays (data segments)
        self.bitmap_indicies = {} # dict of bitmap indices
        self.populate_columns()
        self.load_segments()
        self.create_bitmap_indices("status")


    def populate_columns(self):
        # initialize columns by acessing column field on the dataframe
        for column in self.df.columns:
            self.columns[column] = self.df[column]

    def load_segments(self):
        # Convert cols to numpy array  
        for column in self.columns:
            self.storage[column] = []
            col_data = self.columns[column].to_numpy()
            # split contiguous columns into segments
            for i in range(0, len(self.columns[column]), self.segment_size):                
                segment = self.columns[column][i:i+self.segment_size]
                segment = {
                    "data": segment,
                    "min": segment.min() if segment.size > 0 else None,
                    "max": segment.max() if segment.size > 0 else None,
                    "count": len(segment),
                }
                self.storage[column].append(segment)


    def create_bitmap_indices(self, column_name):
        unique_values = self.columns[column_name].unique()
        self.bitmap_indicies[column_name] = {}  
        for value in unique_values:
            # Store as a numpy boolean array for 'AND/OR' support
            mask = (self.columns[column_name] == value).to_numpy()
            self.bitmap_indicies[column_name][value] = mask
        print(f"Bitmap Index created for '{column_name}' with {len(unique_values)} unique values: {unique_values}")


    def query(self, column_name, value, operator):
        # 2. Run a Baseline Scan (The 'Oracle')
        baseline_result = Baseline(self.storage[column_name]).baseline_scan(value, operator)
        print("--------------------------------")
        print("Baseline Scan Results:")
        print("--------------------------------")
        # print(f"Query Values Count: {sum(baseline_result['values'])}")
        print(f"Query Bitset Count: {sum(baseline_result['bitset'])}")
        print(f"Query Time: {baseline_result['metrics']['query_time']}")
        print(f"Skip Ratio: {baseline_result['metrics']['skip_ratio']}")
        print(f"Segments Skipped: {baseline_result['metrics']['segments_skipped']}")
        
        # 3. Run the Accelerated Scans and compare
        # ZONE MAP SKIPPING
        zone_map_result = ZoneMapSkipping(self.storage[column_name]).zone_map_skipping(value, operator)
        print("--------------------------------")
        print("Zone Map Skipping Results:")
        print("--------------------------------")
        # print(f"Query Values Count: {sum(zone_map_result['values'])}")
        print(f"Query Bitset Count: {sum(zone_map_result['bitset'])}")
        print(f"Query Time: {zone_map_result['metrics']['query_time']}")
        print(f"Skip Ratio: {zone_map_result['metrics']['skip_ratio']}")
        print(f"Segments Skipped: {zone_map_result['metrics']['segments_skipped']}")
        
        # BITMAP INDEX
        if column_name in self.bitmap_indicies:
            bitmap_index_result = BitmapIndex(self.bitmap_indicies[column_name]).bitmap_index_scan(value)
            print("--------------------------------")
            print("Bitmap Index Results:")
            print("--------------------------------")
            print(f"Query Bitset Count: {sum(bitmap_index_result['bitset'])}")
            print(f"Query Time: {bitmap_index_result['metrics']['query_time']}")
            return baseline_result, zone_map_result, bitmap_index_result
        else:
            return baseline_result, zone_map_result, None


def main():
    # 1. Initialize the 'Storage Layer'
    store = MiniColumnStore("synthetic_data.csv")


    queries = [
        ("status", "Active", "="),
        ("cluster_id", 10, ">"),
        # ("random_val", 500, "<"),
        # ("timestamp", 16000000, ">="),
        # ("sensor_reading", 14, "<="),
    ]

    for col, val, op in queries:
        baseline_result, zone_map_result, bitmap_index_result = store.query(col, val, op)
        

if __name__ == "__main__":
    main()