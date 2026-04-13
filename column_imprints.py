import time
import numpy as np

class ColumnImprints:
    """
    Column Imprints: store range-bit imprints per segment group.
    - For each segment, store min/max and a bit-imprint (set bits for value ranges).
    - Allows false positives (segment may not have matches even if imprint says "maybe").
    - No false negatives (if segment has a match, imprint will include it).
    - Supports <, <=, >, >=, =, !=, BETWEEN with segment-level pruning.
    """

    def __init__(self, segments, bit_width=None):
        """
        segments: list of {"data": array, "base_row": int, ...}
        Builds imprints per segment (no concatenation; each segment is independent).
        """
        self.segments = segments
        self.n_segs = len(segments)
        
        if self.n_segs == 0:
            self.bit_width = 1
            self.seg_imprints = []
            self.seg_stats = []
            return

        # determine bit width from all data
        all_data = []
        for seg in segments:
            data = np.asarray(seg["data"], dtype=np.int64)
            if len(data) > 0:
                all_data.append(data)
        
        if all_data:
            combined = np.concatenate(all_data)
            if bit_width is None:
                if len(combined) == 0:
                    self.bit_width = 1
                else:
                    mn = int(combined.min())
                    mx = int(combined.max())
                    mx_abs = max(abs(mn), abs(mx))
                    needed = 1 if mx_abs == 0 else int(mx_abs).bit_length()
                    if np.any(combined < 0):
                        needed += 1
                    self.bit_width = min(max(needed, 1), 64)
            else:
                self.bit_width = int(bit_width)
        else:
            self.bit_width = 1

        # build per-segment imprints and stats
        self.seg_imprints = []
        self.seg_stats = []
        
        self.mask = (1 << self.bit_width) - 1
        self.bias = 1 << (self.bit_width - 1)

        for seg in segments:
            data = np.asarray(seg["data"], dtype=np.int64)
            if len(data) == 0:
                # empty segment
                self.seg_stats.append({
                    "min": None, "max": None, "count": 0,
                    "base_row": int(seg.get("base_row", 0))
                })
                self.seg_imprints.append(np.uint64(0))
                continue

            mn = int(data.min())
            mx = int(data.max())
            count = len(data)
            
            # compute imprint: set bit for each value in segment
            imprint = np.uint64(0)
            if self.bit_width <= 64:
                # convert to unsigned-lex for consistency
                for val in data:
                    u_val = np.uint64(int(val) ^ int(self.bias)) & np.uint64(self.mask)
                    # set bit corresponding to this value's position
                    # (treat the value mod 64 as a bit position for simplicity)
                    bit_pos = int(u_val % 64)
                    imprint |= (np.uint64(1) << np.uint64(bit_pos))
            
            self.seg_stats.append({
                "min": mn, "max": mx, "count": count,
                "base_row": int(seg.get("base_row", 0))
            })
            self.seg_imprints.append(imprint)
            
    def _can_prune_segment(self, seg_idx, value, operator):
        """
        Return True if segment can be safely pruned (no possible matches).
        Return False if segment might have matches (may be false positive).
        Conservative: only prune when we are 100% certain no match exists.
        """
        stat = self.seg_stats[seg_idx]
        if stat["min"] is None:
            # empty segment
            return True

        mn = stat["min"]
        mx = stat["max"]

        if operator == "<":
            # x < value: prune only if ALL values in segment are >= value
            # i.e., if mn >= value, then all values >= value, so no x < value
            return mn >= value
        elif operator == "<=":
            # x <= value: prune only if ALL values in segment are > value
            # i.e., if mn > value, then all values > value, so no x <= value
            return mn > value
        elif operator == ">":
            # x > value: prune only if ALL values in segment are <= value
            # i.e., if mx <= value, then all values <= value, so no x > value
            return mx <= value
        elif operator == ">=":
            # x >= value: prune only if ALL values in segment are < value
            # i.e., if mx < value, then all values < value, so no x >= value
            return mx < value
        elif operator == "=":
            # exact match: prune only if value strictly outside [min, max]
            return value < mn or value > mx
        elif operator == "!=":
            # not equal: conservative — never prune (almost all values will match)
            return False
        elif operator == "BETWEEN":
            low, high = value
            # prune only if segment entirely outside [low, high]
            return mx < low or mn > high
        else:
            return False
        
    def query(self, value, operator):
        """
        Run query using segment-level pruning via imprints.
        Return matching rows and prune statistics.
        """
        start = time.perf_counter()
        
        matching_row_ids = []
        segments_examined = 0
        segments_pruned = 0

        for seg_idx, seg in enumerate(self.segments):
            # check if we can prune this segment
            if self._can_prune_segment(seg_idx, value, operator):
                segments_pruned += 1
                continue

            # segment might have matches; scan it fully
            segments_examined += 1
            data = np.asarray(seg["data"], dtype=np.int64)
            if len(data) == 0:
                continue

            base_row = int(seg.get("base_row", 0))
            
            # apply filter on this segment's data
            if operator == "<":
                mask = data < value
            elif operator == "<=":
                mask = data <= value
            elif operator == ">":
                mask = data > value
            elif operator == ">=":
                mask = data >= value
            elif operator == "=":
                mask = data == value
            elif operator == "!=":
                mask = data != value
            elif operator == "BETWEEN":
                low, high = value
                mask = (data >= low) & (data <= high)
            else:
                raise ValueError("Unsupported operator for ColumnImprints: " + str(operator))

            # append matching global row ids
            local_ids = np.where(mask)[0]
            global_ids = (local_ids + base_row).tolist()
            matching_row_ids.extend(global_ids)

        end = time.perf_counter()
        metrics = {
            "query_time": end - start,
            "segments_examined": segments_examined,
            "segments_pruned": segments_pruned,
            "total_segments": self.n_segs,
            "bytes_for_imprints": int(self.n_segs * 8),  # one uint64 per segment
        }
        return {
            "matching_row_ids": matching_row_ids,
            "metrics": metrics,
        }

    def verify_against_baseline(self, baseline_row_ids, value, operator):
        """
        Verify that imprints produce same or superset of baseline matches (no false negatives).
        """
        ci_ids = set(self.query(value, operator)["matching_row_ids"])
        base_ids = set(baseline_row_ids)
        # should have no false negatives (ci_ids should include all base_ids)
        false_negatives = base_ids - ci_ids
        # may have false positives (ci_ids may include rows not in base_ids)
        false_positives = ci_ids - base_ids
        return {
            "false_negatives": len(false_negatives),
            "false_positives": len(false_positives),
        }