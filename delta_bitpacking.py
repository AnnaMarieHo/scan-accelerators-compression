import math
import time


class DeltaBitPacking:
    """Delta-encode per segment, then pack unsigned (biased) deltas with minimal bit width."""

    def __init__(self, segments):
        self.compressed_segments = self.encode(segments)

    def encode(self, segments):
        encoded_output = []
        for segment in segments:
            data = segment["data"]
            if len(data) == 0:
                continue
            arr = data.to_numpy()
            base_value = int(arr[0])
            if len(arr) == 1:
                deltas = []
            else:
                deltas = [int(arr[i] - arr[i - 1]) for i in range(1, len(arr))]

            if not deltas:
                min_delta = 0
                encoded_deltas = []
                bit_width = 1
            else:
                min_delta = min(deltas)
                max_delta = max(deltas)
                span = max_delta - min_delta
                if span == 0:
                    bit_width = 1
                else:
                    bit_width = max(1, math.ceil(math.log2(span + 1)))
                encoded_deltas = [d - min_delta for d in deltas]

            packed_bytes = bytearray()
            current_byte = 0
            bits_filled = 0
            for ed in encoded_deltas:
                for bit_idx in range(bit_width):
                    bit = (ed >> (bit_width - 1 - bit_idx)) & 1
                    current_byte = (current_byte << 1) | bit
                    bits_filled += 1
                    if bits_filled == 8:
                        packed_bytes.append(current_byte)
                        current_byte = 0
                        bits_filled = 0
            if bits_filled > 0:
                current_byte <<= 8 - bits_filled
                packed_bytes.append(current_byte)

            encoded_output.append(
                {
                    "base_row": int(segment.get("base_row", 0)),
                    "base_value": base_value,
                    "min_delta": int(min_delta),
                    "packed_deltas": bytes(packed_bytes),
                    "bit_width": int(bit_width),
                    "num_deltas": len(encoded_deltas),
                    "count": int(len(arr)),
                    "min": int(arr.min()),
                    "max": int(arr.max()),
                }
            )
        return encoded_output

    def compressed_storage_bytes(self):
        total = 0
        for seg in self.compressed_segments:
            total += (
                8  # base_value
                + 8  # min_delta
                + 4  # bit_width
                + 4  # num_deltas / count meta (packed as int sizes below)
                + 8  # min
                + 8  # max
                + 4  # base_row
                + len(seg["packed_deltas"])
            )
        return total

    @staticmethod
    def _unpack_unsigned(packed_data, bit_width, num_codes):
        if num_codes == 0 or bit_width == 0:
            return []
        bits = []
        for byte in packed_data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        needed = num_codes * bit_width
        bits = bits[:needed]
        out = []
        for j in range(num_codes):
            chunk = bits[j * bit_width : (j + 1) * bit_width]
            v = 0
            for b in chunk:
                v = (v << 1) | b
            out.append(v)
        return out

    def _reconstruct_values(self, seg):
        base = seg["base_value"]
        min_d = seg["min_delta"]
        unpacked = self._unpack_unsigned(
            seg["packed_deltas"], seg["bit_width"], seg["num_deltas"]
        )
        vals = [base]
        for u in unpacked:
            delta = u + min_d
            vals.append(vals[-1] + delta)
        return vals

    def query_equality(self, target_value):
        t0 = time.perf_counter()
        matching_row_ids = []
        segments_skipped = 0
        for seg in self.compressed_segments:
            if target_value < seg["min"] or target_value > seg["max"]:
                segments_skipped += 1
                continue
            vals = self._reconstruct_values(seg)
            base_row = seg["base_row"]
            for i, v in enumerate(vals):
                if v == target_value:
                    matching_row_ids.append(base_row + i)
        return {
            "matching_row_ids": matching_row_ids,
            "time": time.perf_counter() - t0,
            "skipped": segments_skipped,
        }

    def query(self, target_value):
        return self.query_equality(target_value)
