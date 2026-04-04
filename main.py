import time

import numpy as np
import pandas as pd

from baseline import Baseline
from bitmap_index import BitmapIndex
from delta_bitpacking import DeltaBitPacking
from dictionary_encoding import DictionaryEncoding
from metrics import summarize_column_storage
from run_length_encoding import RunLengthEncoding
from zone_map_skipping import ZoneMapSkipping


class MiniColumnStore:
    def __init__(self, csv_path, segment_size=1024):
        t_build = time.perf_counter()
        self.df = pd.read_csv(csv_path)
        self.segment_size = segment_size
        self.columns = {}
        self.storage = {}
        self.bitmap_indicies = {}
        self.n_rows = len(self.df)

        self.populate_columns()
        self.load_segments()
        self.create_bitmap_indices("status")

        self.build_metrics = {
            "build_time_s": time.perf_counter() - t_build,
            "rows": self.n_rows,
            "segment_size": segment_size,
            "segments_per_column": {c: len(self.storage[c]) for c in self.storage},
            "bitmap_bytes_status": self._bitmap_total_bytes("status"),
        }

    def populate_columns(self):
        for column in self.df.columns:
            self.columns[column] = self.df[column]

    def load_segments(self):
        for column in self.columns:
            self.storage[column] = []
            series = self.columns[column]
            for i in range(0, len(series), self.segment_size):
                chunk = series.iloc[i : i + self.segment_size]
                segment = {
                    "data": chunk.reset_index(drop=True),
                    "base_row": i,
                    "min": chunk.min() if len(chunk) > 0 else None,
                    "max": chunk.max() if len(chunk) > 0 else None,
                    "count": len(chunk),
                }
                self.storage[column].append(segment)

    def _bitmap_total_bytes(self, column_name):
        if column_name not in self.bitmap_indicies:
            return 0
        return int(
            sum(np.asarray(a).nbytes for a in self.bitmap_indicies[column_name].values())
        )

    def create_bitmap_indices(self, column_name):
        unique_values = self.columns[column_name].unique()
        self.bitmap_indicies[column_name] = {}
        for value in unique_values:
            mask = (self.columns[column_name] == value).to_numpy()
            self.bitmap_indicies[column_name][value] = mask
        print(
            f"Bitmap index for '{column_name}': {len(unique_values)} values, "
            f"~{self._bitmap_total_bytes(column_name)} bytes"
        )

    def print_build_summary(self):
        print("================ BUILD SUMMARY ================")
        print(f"Build time:     {self.build_metrics['build_time_s']:.4f}s")
        print(f"Rows:           {self.build_metrics['rows']}")
        print(f"Segment size:   {self.build_metrics['segment_size']}")
        print(f"Bitmap (status): {self.build_metrics['bitmap_bytes_status']} bytes")
        for col, segs in self.storage.items():
            s = summarize_column_storage(segs)
            print(
                f"Column '{col}': raw={s['raw_data_bytes']} B, "
                f"zone meta +~{s['zone_map_metadata_bytes']} B"
            )
        print("==============================================\n")

    def query(self, column_name, value, operator):
        segs = self.storage[column_name]
        col_summary = summarize_column_storage(segs)

        print(f"\n{'='*60}\nQUERY: {column_name} {operator} {value}\n{'='*60}")

        baseline_result = Baseline(segs).baseline_scan(value, operator)
        print("--- Baseline scan (oracle) ---")
        print(f"  Matches:        {int(baseline_result['bitset'].sum())}")
        print(f"  Row IDs (head): {baseline_result['row_ids'][:10]}{'...' if len(baseline_result['row_ids']) > 10 else ''}")
        print(f"  Query time:     {baseline_result['metrics']['query_time']:.6f}s")
        print(f"  Bytes scanned:  {baseline_result['metrics'].get('bytes_scanned', 'n/a')}")

        zone_map_result = ZoneMapSkipping(segs).zone_map_skipping(value, operator)
        print("--- Zone map skipping ---")
        print(f"  Matches:           {int(zone_map_result['bitset'].sum())}")
        print(f"  Query time:        {zone_map_result['metrics']['query_time']:.6f}s")
        print(f"  Skip ratio:        {zone_map_result['metrics']['skip_ratio']:.4f}")
        print(f"  Segments skipped:  {zone_map_result['metrics']['segments_skipped']}")
        print(
            f"  Wasted segments:   {zone_map_result['metrics'].get('wasted_segments', 0)} "
            f"(inspected but 0 hits — 'false alarm' segments)"
        )

        oracle_bits = baseline_result["bitset"]
        zone_bits = zone_map_result["bitset"]
        if oracle_bits.shape == zone_bits.shape:
            mismatches = int(np.sum(oracle_bits != zone_bits))
            print(f"  Tuple-level mismatches vs baseline: {mismatches} (should be 0)")
        else:
            print("  WARN: bitset length mismatch")

        if column_name in self.bitmap_indicies:
            bitmap_index_result = BitmapIndex(
                self.bitmap_indicies[column_name],
                self.columns[column_name].to_numpy(),
            ).bitmap_index_scan(value)
            print("--- Bitmap index ---")
            b = bitmap_index_result["bitset"]
            print(f"  Matches:     {int(b.sum())}")
            print(f"  Query time:  {bitmap_index_result['metrics']['query_time']:.6f}s")
            print(
                f"  Column bitmap storage: {bitmap_index_result['metrics']['bitmap_bytes_for_column']} bytes"
            )
            if operator == "=" and b.shape == oracle_bits.shape:
                print(
                    f"  vs baseline mismatches: {int(np.sum(oracle_bits != b))} "
                    "(equality on indexed column should match for =)"
                )

        rle = RunLengthEncoding(segs)
        rle_result = rle.direct_count(value)
        print("--- RLE (direct count) ---")
        print(f"  Total count:           {rle_result['total_count']}")
        print(f"  Query time:            {rle_result['metrics']['query_time']:.6f}s")
        print(
            f"  Approx compressed B:   {rle_result['metrics'].get('approx_compressed_bytes', 'n/a')}"
        )

        dict_enc = DictionaryEncoding(segs, column_name)
        dictionary_result = dict_enc.query_dictionary(value)
        print("--- Dictionary encoding ---")
        print(f"  Matches:               {len(dictionary_result['matching_row_ids'])}")
        print(f"  Query time:            {dictionary_result['metrics']['query_time']:.6f}s")
        print(f"  Segments skipped:      {dictionary_result['metrics']['segments_skipped']}")
        print(
            f"  Approx encoded bytes:  {dictionary_result['metrics'].get('approx_encoded_bytes', 'n/a')}"
        )

        first_dtype = segs[0]["data"].dtype if segs else None
        if first_dtype is not None and pd.api.types.is_numeric_dtype(first_dtype):
            delta_engine = DeltaBitPacking(segs)
            result = delta_engine.query(value)
            print("--- Delta + bit packing (equality) ---")
            print(f"  Matches:            {len(result['matching_row_ids'])}")
            print(f"  Query time:         {result['time']:.6f}s")
            print(f"  Segments skipped:   {result['skipped']}")
            original_size = sum(len(s["data"]) * 8 for s in segs)
            compressed_size = delta_engine.compressed_storage_bytes()
            print(f"  Raw int64-ish size: {original_size} bytes (8 * values)")
            print(f"  Packed + meta est.: {compressed_size} bytes")
            if original_size:
                print(
                    f"  Storage savings:    {100 * (1 - compressed_size / original_size):.2f}%"
                )
            if operator == "=":
                base_ids = set(baseline_result["row_ids"])
                delta_ids = set(result["matching_row_ids"])
                print(
                    f"  vs baseline set diff: {len(base_ids.symmetric_difference(delta_ids))} "
                    "(should be 0 for =)"
                )
        else:
            print("--- Delta + bit packing: skipped (non-numeric) ---")

        print(f"\n--- Column footprint (this query context) ---")
        print(f"  Raw + zone map metadata: {col_summary['total_with_zonemap_bytes']} bytes")

    def query_conjunction(self, predicates, mode="AND", label=""):
        """predicates: list of (column_name, value, operator)."""
        mode = mode.upper()
        print(f"\n{'#'*60}\nCONJUNCTION {mode}: {label or predicates}\n{'#'*60}")

        baseline_masks = []
        t0 = time.perf_counter()
        for col, val, op in predicates:
            baseline_masks.append(
                Baseline(self.storage[col]).baseline_scan(val, op)["bitset"]
            )
        baseline_time = time.perf_counter() - t0
        if mode == "AND":
            oracle = BitmapIndex.combine_and(*baseline_masks)
        elif mode == "OR":
            oracle = BitmapIndex.combine_or(*baseline_masks)
        else:
            raise ValueError("mode must be AND or OR")
        oracle_count = int(np.sum(oracle))
        oracle_ids = np.flatnonzero(oracle).tolist()

        acc_masks = []
        t1 = time.perf_counter()
        for col, val, op in predicates:
            if col in self.bitmap_indicies and op == "=":
                res = BitmapIndex(
                    self.bitmap_indicies[col],
                    self.columns[col].to_numpy(),
                ).bitmap_index_scan(val)
                acc_masks.append(res["bitset"])
            else:
                acc_masks.append(
                    ZoneMapSkipping(self.storage[col]).zone_map_skipping(val, op)["bitset"]
                )
        accel_time = time.perf_counter() - t1
        if mode == "AND":
            accelerated = BitmapIndex.combine_and(*acc_masks)
        else:
            accelerated = BitmapIndex.combine_or(*acc_masks)
        accel_count = int(np.sum(accelerated))
        mismatches = int(np.sum(oracle != accelerated))

        print(f"Oracle ({mode}) match count:      {oracle_count}")
        print(f"Accelerated ({mode}) match count:   {accel_count}")
        print(f"Mismatched positions vs oracle:     {mismatches} (should be 0)")
        print(f"Baseline predicate eval time:       {baseline_time:.6f}s (sum of scans)")
        print(f"Accelerated eval time:              {accel_time:.6f}s (sum of scans)")
        print(f"Sample oracle row IDs:              {oracle_ids[:15]}{'...' if len(oracle_ids) > 15 else ''}")

    def demo_bitmap_logic(self):
        """AND / OR / NOT demo on low-cardinality status bitmaps."""
        if "status" not in self.bitmap_indicies:
            return
        b = self.bitmap_indicies["status"]
        vals = self.columns["status"].to_numpy()
        engine = BitmapIndex(b, vals)
        active = engine.bitmap_index_scan("Active")["bitset"]
        pending = engine.bitmap_index_scan("Pending")["bitset"]
        not_active = BitmapIndex.combine_not(active)
        and_or = BitmapIndex.combine_or(active, pending)
        and_mask = BitmapIndex.combine_and(active, pending)
        print(
            f"\n--- Bitmap logic demo (status) ---\n"
            f"  count(Active OR Pending): {int(and_or.sum())}\n"
            f"  count(Active AND Pending): {int(and_mask.sum())}\n"
            f"  count(NOT Active):        {int(not_active.sum())}"
        )


def main():
    store = MiniColumnStore("synthetic_data.csv")
    store.print_build_summary()

    queries = [
        ("status", "Active", "="),
        ("cluster_id", 10, ">"),
        ("random_val", 500, "<"),
        ("timestamp", 1_600_000_000 + 100_000, ">="),
        ("sensor_reading", 14, "<="),
        ("cluster_id", 25, "="),
    ]

    for col, val, op in queries:
        store.query(col, val, op)

    store.query_conjunction(
        [("status", "Active", "="), ("cluster_id", 15, ">")],
        mode="AND",
        label="Active AND cluster_id > 15",
    )
    store.query_conjunction(
        [("status", "Inactive", "="), ("random_val", 200, ">=")],
        mode="AND",
        label="Inactive AND random_val >= 200",
    )
    store.demo_bitmap_logic()


if __name__ == "__main__":
    main()
