#!/usr/bin/env python3
"""
Reproducibility Script for Scan Accelerators & Compression Experiments
=======================================================================
This script:
1. Generates 4 synthetic datasets with different characteristics
2. Runs 6+ query types including AND predicates and aggregations
3. Reports build time, query time, bytes used, and structure-specific metrics
4. Verifies correctness (no false negatives, exact matches where required)

Datasets:
- clustered: sorted runs (benefits RLE, zone maps)
- low_cardinality: few unique values (benefits bitmap, dictionary)
- high_cardinality: many unique values (minimal compression benefit)
- mixed: realistic mix of column types

Usage:
    python reproduce.py                    # Run with defaults
    python reproduce.py --rows 500000      # Custom row count
    python reproduce.py --seed 42          # Reproducible random seed
    python reproduce.py --output results.csv  # Save results to CSV
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline import Baseline
from bitmap_index import BitmapIndex
from bit_slicing import BitSlicing
from bitweaving import BitWeaving
from column_imprints import ColumnImprints
from delta_bitpacking import DeltaBitPacking
from dictionary_encoding import DictionaryEncoding
from run_length_encoding import RunLengthEncoding
from zone_map_skipping import ZoneMapSkipping


# =============================================================================
# DATASET GENERATORS
# =============================================================================

def generate_clustered_dataset(n_rows: int, seed: int, output_path: str) -> pd.DataFrame:
    """
    Dataset 1: CLUSTERED
    - Sorted runs benefit RLE and zone map skipping
    - Long runs of identical values
    """
    print(f"\n  Generating CLUSTERED dataset...")
    np.random.seed(seed)
    
    # cluster_id: long sorted runs (great for RLE)
    cluster_ids = []
    current_cluster = 0
    while len(cluster_ids) < n_rows:
        run_length = np.random.randint(5000, 20000)  # Long runs
        cluster_ids.extend([current_cluster] * min(run_length, n_rows - len(cluster_ids)))
        current_cluster = (current_cluster + 1) % 50
    cluster_ids = np.array(cluster_ids[:n_rows], dtype=np.int64)
    
    # sorted_value: monotonically increasing (perfect for delta encoding)
    sorted_values = np.sort(np.random.randint(0, n_rows * 10, size=n_rows))
    
    # region: sorted by region, then random within (good zone map locality)
    regions = np.repeat(np.arange(100), n_rows // 100 + 1)[:n_rows]
    
    df = pd.DataFrame({
        "id": np.arange(n_rows, dtype=np.int64),
        "cluster_id": cluster_ids,
        "sorted_value": sorted_values,
        "region": regions,
        "random_noise": np.random.randint(0, 1000, size=n_rows, dtype=np.int64),
    })
    df.to_csv(output_path, index=False)
    return df


def generate_low_cardinality_dataset(n_rows: int, seed: int, output_path: str) -> pd.DataFrame:
    """
    Dataset 2: LOW CARDINALITY
    - Few unique values benefit bitmap index and dictionary encoding
    - High repetition rates
    """
    print(f"  Generating LOW_CARDINALITY dataset...")
    np.random.seed(seed)
    
    # status: 3 unique values (perfect for bitmap)
    status = np.random.choice(["Active", "Inactive", "Pending"], size=n_rows, p=[0.5, 0.3, 0.2])
    
    # category: 10 unique values
    category = np.random.randint(0, 10, size=n_rows, dtype=np.int64)
    
    # flag: binary (2 unique values)
    flag = np.random.randint(0, 2, size=n_rows, dtype=np.int64)
    
    # priority: 5 unique values
    priority = np.random.choice([1, 2, 3, 4, 5], size=n_rows)
    
    df = pd.DataFrame({
        "id": np.arange(n_rows, dtype=np.int64),
        "status": status,
        "category": category,
        "flag": flag,
        "priority": priority,
    })
    df.to_csv(output_path, index=False)
    return df


def generate_high_cardinality_dataset(n_rows: int, seed: int, output_path: str) -> pd.DataFrame:
    """
    Dataset 3: HIGH CARDINALITY
    - Many unique values, random distribution
    - Compression techniques offer minimal benefit
    - Tests worst-case scenarios
    """
    print(f"  Generating HIGH_CARDINALITY dataset...")
    np.random.seed(seed)
    
    # unique_id: all unique values (no compression benefit)
    unique_ids = np.random.permutation(n_rows).astype(np.int64)
    
    # random_val: uniform random, high cardinality
    random_vals = np.random.randint(0, n_rows, size=n_rows, dtype=np.int64)
    
    # hash_val: simulated hash values (very high cardinality)
    hash_vals = np.random.randint(0, 2**31, size=n_rows, dtype=np.int64)
    
    # timestamp: random (not sorted, no delta benefit)
    timestamps = np.random.randint(1_600_000_000, 1_700_000_000, size=n_rows, dtype=np.int64)
    
    df = pd.DataFrame({
        "id": np.arange(n_rows, dtype=np.int64),
        "unique_id": unique_ids,
        "random_val": random_vals,
        "hash_val": hash_vals,
        "timestamp": timestamps,
    })
    df.to_csv(output_path, index=False)
    return df


def generate_mixed_dataset(n_rows: int, seed: int, output_path: str) -> pd.DataFrame:
    """
    Dataset 4: MIXED (realistic)
    - Combination of column types
    - Some benefit from each technique
    """
    print(f"  Generating MIXED dataset...")
    np.random.seed(seed)
    random.seed(seed)
    
    # id: unique
    ids = np.arange(n_rows, dtype=np.int64)
    
    # cluster_id: low cardinality with runs
    cluster_ids = []
    current_cluster = 0
    while len(cluster_ids) < n_rows:
        run_length = np.random.randint(100, 5000)
        cluster_ids.extend([current_cluster] * min(run_length, n_rows - len(cluster_ids)))
        current_cluster = (current_cluster + 1) % 50
    cluster_ids = np.array(cluster_ids[:n_rows], dtype=np.int64)
    
    # status: categorical
    statuses = np.random.choice(["Active", "Inactive", "Pending"], size=n_rows, p=[0.5, 0.3, 0.2])
    
    # timestamp: monotonic increasing (good for delta)
    timestamps = 1_600_000_000 + np.cumsum(np.random.randint(1, 10, size=n_rows))
    
    # sensor_reading: medium cardinality
    sensor_readings = np.random.randint(10, 30, size=n_rows, dtype=np.int64)
    
    # amount: high cardinality float-like integers
    amounts = np.random.randint(100, 100000, size=n_rows, dtype=np.int64)
    
    df = pd.DataFrame({
        "id": ids,
        "cluster_id": cluster_ids,
        "status": statuses,
        "timestamp": timestamps,
        "sensor_reading": sensor_readings,
        "amount": amounts,
    })
    df.to_csv(output_path, index=False)
    return df


def generate_all_datasets(n_rows: int, seed: int, output_dir: str = ".") -> dict:
    """Generate all 4 datasets and return paths."""
    print(f"\n{'='*60}")
    print(f"GENERATING 4 DATASETS")
    print(f"{'='*60}")
    print(f"Rows per dataset: {n_rows:,}")
    print(f"Seed: {seed}")
    
    datasets = {}
    
    t0 = time.perf_counter()
    
    datasets["clustered"] = {
        "path": os.path.join(output_dir, "data_clustered.csv"),
        "df": generate_clustered_dataset(n_rows, seed, os.path.join(output_dir, "data_clustered.csv")),
        "description": "Sorted runs, benefits RLE & zone maps",
    }
    
    datasets["low_cardinality"] = {
        "path": os.path.join(output_dir, "data_low_card.csv"),
        "df": generate_low_cardinality_dataset(n_rows, seed + 1, os.path.join(output_dir, "data_low_card.csv")),
        "description": "Few unique values, benefits bitmap & dictionary",
    }
    
    datasets["high_cardinality"] = {
        "path": os.path.join(output_dir, "data_high_card.csv"),
        "df": generate_high_cardinality_dataset(n_rows, seed + 2, os.path.join(output_dir, "data_high_card.csv")),
        "description": "Many unique values, minimal compression benefit",
    }
    
    datasets["mixed"] = {
        "path": os.path.join(output_dir, "data_mixed.csv"),
        "df": generate_mixed_dataset(n_rows, seed + 3, os.path.join(output_dir, "data_mixed.csv")),
        "description": "Realistic mix of column types",
    }
    
    elapsed = time.perf_counter() - t0
    print(f"\nAll datasets generated in {elapsed:.2f}s")
    
    for name, info in datasets.items():
        size_mb = os.path.getsize(info["path"]) / (1024 * 1024)
        print(f"  {name}: {size_mb:.2f} MB - {info['description']}")
    
    return datasets


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """Runs all experiments and collects results."""
    
    def __init__(self, df: pd.DataFrame, dataset_name: str, segment_size: int = 1024):
        self.df = df
        self.dataset_name = dataset_name
        self.segment_size = segment_size
        self.n_rows = len(self.df)
        self.results = []
        self.build_times = {}
        self.build_bytes = {}
        
        # Build segmented storage
        self.storage = {}
        for column in self.df.columns:
            self.storage[column] = []
            series = self.df[column]
            for i in range(0, len(series), segment_size):
                chunk = series.iloc[i : i + segment_size]
                segment = {
                    "data": chunk.reset_index(drop=True),
                    "base_row": i,
                    "min": chunk.min() if len(chunk) > 0 else None,
                    "max": chunk.max() if len(chunk) > 0 else None,
                    "count": len(chunk),
                }
                self.storage[column].append(segment)
        
        # Build bitmap index for categorical columns
        self.bitmap_indices = {}
        for col in self.df.columns:
            if self.df[col].dtype == object or self.df[col].nunique() <= 20:
                t0 = time.perf_counter()
                unique_values = self.df[col].unique()
                self.bitmap_indices[col] = {}
                for value in unique_values:
                    mask = (self.df[col] == value).to_numpy()
                    self.bitmap_indices[col][value] = mask
                self.build_times[f"bitmap_{col}"] = time.perf_counter() - t0
                self.build_bytes[f"bitmap_{col}"] = sum(m.nbytes for m in self.bitmap_indices[col].values())
    
    def _build_structure(self, name: str, constructor, *args, **kwargs):
        """Build a structure and record build time."""
        t0 = time.perf_counter()
        obj = constructor(*args, **kwargs)
        build_time = time.perf_counter() - t0
        return obj, build_time
    
    def run_query(self, column: str, value, operator: str, query_type: str = "range") -> dict:
        """Run a single query through all applicable techniques."""
        segs = self.storage[column]
        result = {
            "dataset": self.dataset_name,
            "column": column,
            "operator": operator,
            "value": str(value),
            "query_type": query_type,
            "n_rows": self.n_rows,
        }
        
        print(f"\n  [{self.dataset_name}] {column} {operator} {value} ({query_type})")
        
        # 1. Baseline scan (oracle)
        baseline_obj, build_time = self._build_structure("baseline", Baseline, segs)
        baseline = baseline_obj.baseline_scan(value, operator)
        baseline_count = int(baseline["bitset"].sum())
        baseline_ids = set(baseline["row_ids"])
        result["baseline_count"] = baseline_count
        result["baseline_time"] = baseline["metrics"]["query_time"]
        result["baseline_build_time"] = build_time
        result["baseline_bytes"] = baseline["metrics"].get("bytes_scanned", self.n_rows * 8)
        result["selectivity"] = baseline_count / self.n_rows if self.n_rows > 0 else 0
        
        # 2. Zone map skipping
        try:
            zm_obj, build_time = self._build_structure("zonemap", ZoneMapSkipping, segs)
            zm = zm_obj.zone_map_skipping(value, operator)
            zm_count = int(zm["bitset"].sum())
            zm_mismatch = int(np.sum(baseline["bitset"] != zm["bitset"]))
            result["zonemap_count"] = zm_count
            result["zonemap_time"] = zm["metrics"]["query_time"]
            result["zonemap_build_time"] = build_time
            result["zonemap_skip_ratio"] = zm["metrics"]["skip_ratio"]
            result["zonemap_segments_skipped"] = zm["metrics"]["segments_skipped"]
            result["zonemap_mismatch"] = zm_mismatch
            result["zonemap_bytes"] = len(segs) * 24  # min/max/count per segment
        except Exception as e:
            result["zonemap_error"] = str(e)
        
        # 3. Bitmap index (only for = on indexed columns)
        if column in self.bitmap_indices and operator == "=":
            try:
                bi_obj, build_time = self._build_structure(
                    "bitmap", BitmapIndex, 
                    self.bitmap_indices[column], self.df[column].to_numpy()
                )
                bi = bi_obj.bitmap_index_scan(value)
                bi_count = int(bi["bitset"].sum())
                bi_mismatch = int(np.sum(baseline["bitset"] != bi["bitset"]))
                result["bitmap_count"] = bi_count
                result["bitmap_time"] = bi["metrics"]["query_time"]
                result["bitmap_build_time"] = self.build_times.get(f"bitmap_{column}", 0)
                result["bitmap_bytes"] = self.build_bytes.get(f"bitmap_{column}", 0)
                result["bitmap_mismatch"] = bi_mismatch
                result["bitmap_cardinality"] = len(self.bitmap_indices[column])
            except Exception as e:
                result["bitmap_error"] = str(e)
        
        # 4. RLE (count only)
        try:
            rle_obj, build_time = self._build_structure("rle", RunLengthEncoding, segs)
            rle_result = rle_obj.direct_count(value)
            result["rle_count"] = rle_result["total_count"]
            result["rle_time"] = rle_result["metrics"]["query_time"]
            result["rle_build_time"] = build_time
            result["rle_bytes"] = rle_result["metrics"].get("approx_compressed_bytes", 0)
            result["rle_compression_ratio"] = rle_result["metrics"].get("compression_ratio", 1.0)
            result["rle_run_count"] = rle_result["metrics"].get("run_count", 0)
        except Exception as e:
            result["rle_error"] = str(e)
        
        # 5. Dictionary encoding
        try:
            de_obj, build_time = self._build_structure("dictionary", DictionaryEncoding, segs, column)
            de_result = de_obj.query_dictionary(value)
            de_count = len(de_result["matching_row_ids"])
            result["dict_count"] = de_count
            result["dict_time"] = de_result["metrics"]["query_time"]
            result["dict_build_time"] = build_time
            result["dict_segments_skipped"] = de_result["metrics"]["segments_skipped"]
            result["dict_bytes"] = de_result["metrics"].get("approx_encoded_bytes", 0)
            result["dict_cardinality"] = de_result["metrics"].get("dictionary_size", 0)
        except Exception as e:
            result["dict_error"] = str(e)
        
        # Check if column is numeric for remaining techniques
        first_dtype = segs[0]["data"].dtype if segs else None
        is_integer = first_dtype is not None and pd.api.types.is_integer_dtype(first_dtype)
        
        # 6. Delta + bit packing
        if is_integer:
            try:
                dbp_obj, build_time = self._build_structure("delta", DeltaBitPacking, segs)
                dbp_result = dbp_obj.query(value)
                dbp_count = len(dbp_result["matching_row_ids"])
                result["delta_count"] = dbp_count
                result["delta_time"] = dbp_result["time"]
                result["delta_build_time"] = build_time
                result["delta_bytes"] = dbp_obj.compressed_storage_bytes()
                result["delta_compression_ratio"] = (self.n_rows * 8) / max(result["delta_bytes"], 1)
            except Exception as e:
                result["delta_error"] = str(e)
        
        # 7. Bit-slicing (only <, <=, BETWEEN on integers)
        if is_integer and operator in ("<", "<=", "BETWEEN"):
            try:
                bs_obj, build_time = self._build_structure("bitslice", BitSlicing, segs)
                bs_result = bs_obj.query(value, operator)
                bs_count = int(bs_result["bitset"].sum())
                bs_mismatch = int(np.sum(baseline["bitset"] != bs_result["bitset"]))
                result["bitslice_count"] = bs_count
                result["bitslice_time"] = bs_result["metrics"]["query_time"]
                result["bitslice_build_time"] = build_time
                result["bitslice_bytes"] = bs_result["metrics"]["bytes_for_slices"]
                result["bitslice_mismatch"] = bs_mismatch
                result["bitslice_bit_width"] = bs_obj.bit_width
            except Exception as e:
                result["bitslice_error"] = str(e)
        
        # 8. BitWeaving (all operators on integers)
        if is_integer:
            try:
                bw_obj, build_time = self._build_structure("bitweave", BitWeaving, segs)
                bw_result = bw_obj.query(value, operator)
                bw_count = int(bw_result["bitset"].sum())
                bw_mismatch = int(np.sum(baseline["bitset"] != bw_result["bitset"]))
                result["bitweave_count"] = bw_count
                result["bitweave_time"] = bw_result["metrics"]["query_time"]
                result["bitweave_build_time"] = build_time
                result["bitweave_bytes"] = bw_result["metrics"]["bytes_for_words"]
                result["bitweave_mismatch"] = bw_mismatch
                result["bitweave_bit_width"] = bw_obj.bit_width
            except Exception as e:
                result["bitweave_error"] = str(e)
        
        # 9. Column Imprints
        if is_integer:
            try:
                ci_obj, build_time = self._build_structure("imprints", ColumnImprints, segs)
                ci_result = ci_obj.query(value, operator)
                ci_ids = set(ci_result["matching_row_ids"])
                ci_count = len(ci_ids)
                false_negs = len(baseline_ids - ci_ids)
                false_pos = len(ci_ids - baseline_ids)
                result["imprints_count"] = ci_count
                result["imprints_time"] = ci_result["metrics"]["query_time"]
                result["imprints_build_time"] = build_time
                result["imprints_segments_pruned"] = ci_result["metrics"]["segments_pruned"]
                result["imprints_bytes"] = ci_result["metrics"].get("bytes_for_imprints", 0)
                result["imprints_false_neg"] = false_negs
                result["imprints_false_pos"] = false_pos
            except Exception as e:
                result["imprints_error"] = str(e)
        
        self.results.append(result)
        return result
    
    def run_and_query(self, column1: str, value1, op1: str, 
                       column2: str, value2, op2: str) -> dict:
        """Run AND of two predicates."""
        segs1 = self.storage[column1]
        segs2 = self.storage[column2]
        
        result = {
            "dataset": self.dataset_name,
            "column": f"{column1} AND {column2}",
            "operator": f"{op1} AND {op2}",
            "value": f"{value1} AND {value2}",
            "query_type": "and_predicate",
            "n_rows": self.n_rows,
        }
        
        print(f"\n  [{self.dataset_name}] {column1} {op1} {value1} AND {column2} {op2} {value2}")
        
        # Baseline: compute both conditions and AND them
        t0 = time.perf_counter()
        baseline1 = Baseline(segs1).baseline_scan(value1, op1)
        baseline2 = Baseline(segs2).baseline_scan(value2, op2)
        combined_mask = baseline1["bitset"] & baseline2["bitset"]
        baseline_time = time.perf_counter() - t0
        
        baseline_count = int(combined_mask.sum())
        segs1_bytes = sum(seg["data"].nbytes for seg in segs1)
        segs2_bytes = sum(seg["data"].nbytes for seg in segs2)
        
        result["baseline_count"] = baseline_count
        result["baseline_time"] = baseline_time
        result["baseline_build_time"] = 0.0
        result["baseline_bytes"] = (segs1_bytes + segs2_bytes) / 2
        result["selectivity"] = baseline_count / self.n_rows if self.n_rows > 0 else 0
        
        # Zone map: AND of both zone map results
        try:
            t0 = time.perf_counter()
            zm1 = ZoneMapSkipping(segs1).zone_map_skipping(value1, op1)
            zm2 = ZoneMapSkipping(segs2).zone_map_skipping(value2, op2)
            zm_mask = zm1["bitset"] & zm2["bitset"]
            zm_time = time.perf_counter() - t0
            
            zm_count = int(zm_mask.sum())
            zm_mismatch = int(np.sum(combined_mask != zm_mask))
            result["zonemap_count"] = zm_count
            result["zonemap_time"] = zm_time
            result["zonemap_mismatch"] = zm_mismatch
            result["zonemap_skip_ratio"] = (zm1["metrics"]["skip_ratio"] + zm2["metrics"]["skip_ratio"]) / 2
        except Exception as e:
            result["zonemap_error"] = str(e)
        
        # BitWeaving: AND of both results
        first_dtype1 = segs1[0]["data"].dtype if segs1 else None
        first_dtype2 = segs2[0]["data"].dtype if segs2 else None
        is_int1 = first_dtype1 is not None and pd.api.types.is_integer_dtype(first_dtype1)
        is_int2 = first_dtype2 is not None and pd.api.types.is_integer_dtype(first_dtype2)
        
        if is_int1 and is_int2:
            try:
                t0 = time.perf_counter()
                bw1 = BitWeaving(segs1).query(value1, op1)
                bw2 = BitWeaving(segs2).query(value2, op2)
                bw_mask = bw1["bitset"] & bw2["bitset"]
                bw_time = time.perf_counter() - t0
                
                bw_count = int(bw_mask.sum())
                bw_mismatch = int(np.sum(combined_mask != bw_mask))
                result["bitweave_count"] = bw_count
                result["bitweave_time"] = bw_time
                result["bitweave_mismatch"] = bw_mismatch
            except Exception as e:
                result["bitweave_error"] = str(e)
        
        self.results.append(result)
        return result

    def run_aggregation_query(self, column: str, value, operator: str, 
                               agg_column: str, agg_func: str = "sum") -> dict:
        """Run aggregation-friendly query (filter then aggregate)."""
        segs = self.storage[column]
        agg_segs = self.storage[agg_column]
        
        result = {
            "dataset": self.dataset_name,
            "column": column,
            "operator": operator,
            "value": str(value),
            "query_type": f"aggregation_{agg_func}",
            "agg_column": agg_column,
            "n_rows": self.n_rows,
        }
        
        print(f"\n  [{self.dataset_name}] {agg_func.upper()}({agg_column}) WHERE {column} {operator} {value}")
        
        # Baseline aggregation
        t0 = time.perf_counter()
        baseline = Baseline(segs).baseline_scan(value, operator)
        mask = baseline["bitset"]
        
        # Get full column data for aggregation
        full_agg_data = np.concatenate([seg["data"].to_numpy() for seg in agg_segs])
        
        if agg_func == "sum":
            agg_result = np.sum(full_agg_data[mask])
        elif agg_func == "count":
            agg_result = int(mask.sum())
        elif agg_func == "avg":
            agg_result = np.mean(full_agg_data[mask]) if mask.sum() > 0 else 0
        elif agg_func == "min":
            agg_result = np.min(full_agg_data[mask]) if mask.sum() > 0 else None
        elif agg_func == "max":
            agg_result = np.max(full_agg_data[mask]) if mask.sum() > 0 else None
        else:
            agg_result = None
        
        baseline_time = time.perf_counter() - t0
        segs_bytes = sum(seg["data"].nbytes for seg in segs)
        
        result["baseline_count"] = int(mask.sum())
        result["baseline_time"] = baseline_time
        result["baseline_build_time"] = 0.0
        result["baseline_bytes"] = segs_bytes
        result["baseline_agg_result"] = float(agg_result) if agg_result is not None else None
        result["selectivity"] = int(mask.sum()) / self.n_rows if self.n_rows > 0 else 0
        
        # Zone map (for filtering)
        try:
            t0 = time.perf_counter()
            zm_obj = ZoneMapSkipping(segs)
            zm_result = zm_obj.zone_map_skipping(value, operator)
            zm_mask = zm_result["bitset"]
            
            # Compute aggregation on filtered result
            if agg_func == "sum":
                zm_agg = np.sum(full_agg_data[zm_mask])
            elif agg_func == "count":
                zm_agg = int(zm_mask.sum())
            elif agg_func == "avg":
                zm_agg = np.mean(full_agg_data[zm_mask]) if zm_mask.sum() > 0 else 0
            elif agg_func == "min":
                zm_agg = np.min(full_agg_data[zm_mask]) if zm_mask.sum() > 0 else None
            elif agg_func == "max":
                zm_agg = np.max(full_agg_data[zm_mask]) if zm_mask.sum() > 0 else None
            else:
                zm_agg = None
            
            zm_time = time.perf_counter() - t0
            result["zonemap_time"] = zm_time
            result["zonemap_count"] = int(zm_mask.sum())
            result["zonemap_agg_result"] = float(zm_agg) if zm_agg is not None else None
        except Exception as e:
            result["zonemap_error"] = str(e)
        
        # RLE (for count aggregation)
        if agg_func == "count":
            try:
                t0 = time.perf_counter()
                rle_obj = RunLengthEncoding(segs)
                rle_result = rle_obj.direct_count(value)
                rle_time = time.perf_counter() - t0
                
                result["rle_agg_result"] = rle_result["total_count"]
                result["rle_time"] = rle_time
                result["rle_build_time"] = 0.0
                result["rle_bytes"] = rle_result["metrics"].get("approx_compressed_bytes", 0)
                result["rle_speedup"] = baseline_time / rle_time if rle_time > 0 else 0
            except Exception as e:
                result["rle_error"] = str(e)
        
        # Dictionary (for all aggregations)
        try:
            t0 = time.perf_counter()
            de_obj = DictionaryEncoding(segs, column)
            de_result = de_obj.query_dictionary(value)
            de_mask = np.zeros(self.n_rows, dtype=bool)
            de_mask[np.array(de_result["matching_row_ids"])] = True
            
            if agg_func == "sum":
                de_agg = np.sum(full_agg_data[de_mask])
            elif agg_func == "count":
                de_agg = int(de_mask.sum())
            elif agg_func == "avg":
                de_agg = np.mean(full_agg_data[de_mask]) if de_mask.sum() > 0 else 0
            else:
                de_agg = None
            
            de_time = time.perf_counter() - t0
            result["dict_time"] = de_time
            result["dict_build_time"] = 0.0
            result["dict_agg_result"] = float(de_agg) if de_agg is not None else None
        except Exception as e:
            result["dict_error"] = str(e)
        
        self.results.append(result)
        return result


def run_experiment_matrix(datasets: dict, segment_size: int = 1024) -> list:
    """
    Run the full experiment matrix:
    - 4 datasets × 6+ query types
    - Queries: equality, selective range, non-selective range, AND, aggregation
    """
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT MATRIX")
    print(f"{'='*60}")
    print(f"Datasets: {len(datasets)}")
    print(f"Segment size: {segment_size}")
    
    all_results = []
    
    # Define query templates per dataset type
    query_matrix = {
        "clustered": [
            # Equality (benefits from sorted runs)
            ("cluster_id", 25, "=", "equality"),
            # Selective range (<10% selectivity)
            ("cluster_id", 5, "<", "selective_range"),
            # Non-selective range (>50% selectivity)
            ("cluster_id", 10, ">", "nonselective_range"),
            # Range on sorted column (zone map benefit)
            ("sorted_value", 1000000, "<", "sorted_range"),
        ],
        "low_cardinality": [
            # Equality on categorical
            ("category", 5, "=", "equality"),
            # Selective range
            ("category", 2, "<", "selective_range"),
            # Non-selective range
            ("category", 3, ">", "nonselective_range"),
            # Binary column
            ("flag", 1, "=", "binary_equality"),
        ],
        "high_cardinality": [
            # Equality (rare value)
            ("random_val", 500000, "=", "equality_rare"),
            # Selective range
            ("random_val", 100000, "<", "selective_range"),
            # Non-selective range
            ("random_val", 500000, ">", "nonselective_range"),
            # Hash column (worst case)
            ("hash_val", 1000000000, "<", "hash_range"),
        ],
        "mixed": [
            # Equality
            ("cluster_id", 25, "=", "equality"),
            # Selective range (sensor < 15 is ~25%)
            ("sensor_reading", 15, "<", "selective_range"),
            # Non-selective range (amount > 10000 is ~90%)
            ("amount", 10000, ">", "nonselective_range"),
            # Timestamp range
            ("timestamp", 1_600_500_000, "<=", "timestamp_range"),
        ],
    }
    
    # AND predicate templates
    and_queries = {
        "clustered": ("cluster_id", 25, ">", "region", 50, "<"),
        "low_cardinality": ("category", 5, "=", "flag", 1, "="),
        "high_cardinality": ("random_val", 500000, "<", "hash_val", 1000000000, "<"),
        "mixed": ("cluster_id", 25, ">", "sensor_reading", 20, "<"),
    }
    
    # Aggregation templates
    agg_queries = {
        "clustered": ("cluster_id", 25, "=", "random_noise", "sum"),
        "low_cardinality": ("category", 5, "=", "priority", "sum"),
        "high_cardinality": ("random_val", 500000, "<", "hash_val", "count"),
        "mixed": ("cluster_id", 25, "=", "amount", "sum"),
    }
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'─'*40}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*40}")
        
        runner = ExperimentRunner(dataset_info["df"], dataset_name, segment_size)
        
        # Run standard queries
        for col, val, op, qtype in query_matrix.get(dataset_name, []):
            runner.run_query(col, val, op, qtype)
        
        # Run AND query
        if dataset_name in and_queries:
            col1, val1, op1, col2, val2, op2 = and_queries[dataset_name]
            runner.run_and_query(col1, val1, op1, col2, val2, op2)
        
        # Run aggregation query
        if dataset_name in agg_queries:
            col, val, op, agg_col, agg_func = agg_queries[dataset_name]
            runner.run_aggregation_query(col, val, op, agg_col, agg_func)
        
        all_results.extend(runner.results)
    
    return all_results


def print_summary(results: list):
    """Print comprehensive summary."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    # Group by dataset
    by_dataset = {}
    for r in results:
        ds = r.get("dataset", "unknown")
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(r)
    
    print(f"\nTotal experiments: {len(results)}")
    print(f"Datasets: {list(by_dataset.keys())}")
    
    # Correctness summary
    print(f"\n--- CORRECTNESS ---")
    techniques = ["zonemap", "bitmap", "bitslice", "bitweave"]
    for tech in techniques:
        mismatch_key = f"{tech}_mismatch"
        mismatches = [r.get(mismatch_key, -1) for r in results if mismatch_key in r]
        if mismatches:
            exact = sum(1 for m in mismatches if m == 0)
            print(f"  {tech:12s}: {exact}/{len(mismatches)} exact matches")
    
    # Column imprints (false negatives)
    fn_results = [r.get("imprints_false_neg", -1) for r in results if "imprints_false_neg" in r]
    if fn_results:
        no_fn = sum(1 for fn in fn_results if fn == 0)
        print(f"  {'imprints':12s}: {no_fn}/{len(fn_results)} with 0 false negatives")
    
    # Timing summary by technique
    print(f"\n--- AVERAGE QUERY TIMES (seconds) ---")
    for tech in ["baseline", "zonemap", "bitmap", "rle", "dict", "delta", "bitslice", "bitweave", "imprints"]:
        key = f"{tech}_time"
        times = [r[key] for r in results if key in r]
        if times:
            print(f"  {tech:12s}: {np.mean(times):.6f} (n={len(times)})")
    
    # Build time summary
    print(f"\n--- AVERAGE BUILD TIMES (seconds) ---")
    for tech in ["zonemap", "bitmap", "rle", "dict", "delta", "bitslice", "bitweave", "imprints"]:
        key = f"{tech}_build_time"
        times = [r[key] for r in results if key in r]
        if times:
            print(f"  {tech:12s}: {np.mean(times):.6f} (n={len(times)})")
    
    # Space usage summary
    print(f"\n--- AVERAGE BYTES USED ---")
    for tech in ["zonemap", "bitmap", "rle", "dict", "delta", "bitslice", "bitweave", "imprints"]:
        key = f"{tech}_bytes"
        sizes = [r[key] for r in results if key in r]
        if sizes:
            avg_kb = np.mean(sizes) / 1024
            print(f"  {tech:12s}: {avg_kb:.1f} KB (n={len(sizes)})")
    
    # Best technique per query type
    print(f"\n--- BEST TECHNIQUE BY QUERY TYPE ---")
    by_qtype = {}
    for r in results:
        qtype = r.get("query_type", "unknown")
        if qtype not in by_qtype:
            by_qtype[qtype] = []
        by_qtype[qtype].append(r)
    
    for qtype, qresults in by_qtype.items():
        best_tech = None
        best_time = float('inf')
        for tech in ["zonemap", "bitmap", "bitslice", "bitweave", "imprints"]:
            key = f"{tech}_time"
            times = [r[key] for r in qresults if key in r and r.get(f"{tech}_mismatch", -1) == 0]
            if times and np.mean(times) < best_time:
                best_time = np.mean(times)
                best_tech = tech
        if best_tech:
            print(f"  {qtype:20s}: {best_tech} ({best_time:.6f}s)")


def save_results(results: list, output_path: str):
    """Save results to CSV."""
    if not results:
        print("No results to save.")
        return
    
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    all_keys = sorted(all_keys)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproducibility script for scan accelerators & compression experiments"
    )
    parser.add_argument(
        "--rows", type=int, default=1_000_000,
        help="Number of rows per dataset (default: 1,000,000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--segment-size", type=int, default=1024,
        help="Segment size for columnar storage (default: 1024)"
    )
    parser.add_argument(
        "--output", type=str, default="experiment_results.csv",
        help="Path to save results CSV (default: experiment_results.csv)"
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip data generation (use existing data files)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"SCAN ACCELERATORS & COMPRESSION REPRODUCIBILITY")
    print(f"{'#'*60}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python:    {sys.version.split()[0]}")
    print(f"NumPy:     {np.__version__}")
    print(f"Pandas:    {pd.__version__}")
    
    # Step 1: Generate datasets
    if not args.skip_generate:
        datasets = generate_all_datasets(args.rows, args.seed)
    else:
        # Load existing datasets
        print(f"\nLoading existing datasets...")
        datasets = {}
        for name in ["clustered", "low_cardinality", "high_cardinality", "mixed"]:
            path = f"data_{name.replace('_cardinality', '_card')}.csv"
            if os.path.exists(path):
                datasets[name] = {"path": path, "df": pd.read_csv(path)}
                print(f"  Loaded: {path}")
            else:
                print(f"  Missing: {path}")
        
        if not datasets:
            print("ERROR: No datasets found. Run without --skip-generate.")
            sys.exit(1)
    
    # Step 2: Run experiment matrix
    results = run_experiment_matrix(datasets, args.segment_size)
    
    # Step 3: Print summary
    print_summary(results)
    
    # Step 4: Save results
    save_results(results, args.output)
    
    print(f"\n{'#'*60}")
    print(f"DONE - Results saved to {args.output}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()