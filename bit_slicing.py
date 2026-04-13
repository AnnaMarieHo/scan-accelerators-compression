import time
import numpy as np

class BitSlicing:
    """
    Build bit-slices (one bitmap per bit position) for a concatenated integer column.
    Supports operators: '<', '<=', 'BETWEEN' (inclusive).
    Provides sum_via_slices for non-negative integers.

    This implementation always sets mask/bias/uvals/row_ids/bit_slices so attributes
    are available on all code paths and uses uint64 numpy ops for bit-shifts.
    """

    def __init__(self, segments, bit_width=None):
        parts = []
        rowid_parts = []
        for seg in segments:
            data = seg["data"]
            raw = np.asarray(data)

            # Coerce/validate to integer values
            if raw.size == 0:
                arr = np.array([], dtype=np.int64)
            elif raw.dtype.kind == "f":
                # floats allowed only if integer-valued and finite
                if not np.all(np.isfinite(raw)):
                    raise ValueError("BitSlicing: float column contains non-finite values (NaN/inf)")
                frac = np.modf(raw)[0]
                if np.any(frac != 0):
                    raise ValueError("BitSlicing: float column has fractional values; bit-slicing requires integers")
                arr = raw.astype(np.int64)
            elif raw.dtype.kind in ("i", "u"):
                arr = raw.astype(np.int64)
            else:
                # try best-effort conversion for object dtype
                try:
                    arr = np.asarray(raw, dtype=np.int64)
                except Exception:
                    raise ValueError(
                        "BitSlicing: column contains non-integer values. Convert to integer dtype before using bit-slicing."
                    )

            parts.append(arr)
            base = int(seg.get("base_row", 0))
            rowid_parts.append(np.arange(len(arr), dtype=np.int64) + base)

        # values and row_ids always set (possibly empty)
        if parts:
            self.values = np.concatenate(parts).astype(np.int64)
            self.row_ids = np.concatenate(rowid_parts).astype(np.int64)
        else:
            self.values = np.array([], dtype=np.int64)
            self.row_ids = np.array([], dtype=np.int64)

        self.n = int(len(self.values))

        # determine bit_width (at least 1, capped at 64)
        if bit_width is None:
            if self.n == 0:
                self.bit_width = 1
            else:
                mn = int(self.values.min())
                mx = int(self.values.max())
                mx_abs = max(abs(mn), abs(mx))
                needed = 1 if mx_abs == 0 else int(mx_abs).bit_length()
                if np.any(self.values < 0):
                    needed += 1
                self.bit_width = min(max(needed, 1), 64)
        else:
            self.bit_width = int(bit_width)

        # set mask / bias unconditionally so attributes exist
        self.mask = (1 << self.bit_width) - 1
        # Only use bias if there are negative values; otherwise direct unsigned comparison
        self.has_negatives = self.n > 0 and np.any(self.values < 0)
        if self.has_negatives:
            self.bias = 1 << (self.bit_width - 1)
        else:
            self.bias = 0  # no bias needed for all-positive data

        # build unsigned-lex mapped values (uint64) safely
        if self.n > 0:
            vals_i64 = self.values.astype(np.int64)
            vals_u = vals_i64.astype(np.uint64)
            mask_u = np.uint64(self.mask)
            bias_u = np.uint64(self.bias)
            u = np.bitwise_xor(vals_u, bias_u) & mask_u
            self.uvals = u.astype(np.uint64)
        else:
            self.uvals = np.array([], dtype=np.uint64)

        # build bit_slices (MSB -> LSB) and ensure each is boolean numpy array
        self.bit_slices = []
        for bit in range(self.bit_width - 1, -1, -1):
            if self.n > 0:
                self.bit_slices.append(((self.uvals >> np.uint64(bit)) & np.uint64(1)).astype(bool))
            else:
                self.bit_slices.append(np.zeros(0, dtype=bool))

    def _to_unsigned_lex(self, v):
        """Convert scalar integer v to unsigned-lex uint64 matching self.uvals."""
        v64 = int(v)
        # XOR with bias (0 if all-positive, else sign-flip bias)
        v_u = np.uint64(v64) ^ np.uint64(self.bias)
        return v_u & np.uint64(self.mask)

    def _compute_lt_mask(self, v):
        """Return boolean mask where value < v using bit-slices (MSB->LSB)."""
        if self.n == 0:
            return np.zeros(0, dtype=bool)
        v_u = self._to_unsigned_lex(v)
        eq = np.ones(self.n, dtype=bool)
        lt = np.zeros(self.n, dtype=bool)
        for bit_idx, slice_b in enumerate(self.bit_slices):
            bit = self.bit_width - 1 - bit_idx
            vi = bool((v_u >> np.uint64(bit)) & np.uint64(1))
            if vi:
                lt = np.logical_or(lt, np.logical_and(eq, np.logical_not(slice_b)))
                eq = np.logical_and(eq, slice_b)
            else:
                eq = np.logical_and(eq, np.logical_not(slice_b))
            if not np.any(eq):
                # no remaining equals; keep lt as-is
                continue
        return lt

    def query(self, value, operator):
        """
        value: int or (low, high) for BETWEEN
        operator: one of '<', '<=', 'BETWEEN'
        returns dict with 'bitset' (np.bool_), 'matching_row_ids', 'metrics'
        """
        start = time.perf_counter()
        if operator == "<":
            mask = self._compute_lt_mask(value)
        elif operator == "<=":
            lt = self._compute_lt_mask(value)
            # equality mask
            v_u = self._to_unsigned_lex(value)
            eq = np.ones(self.n, dtype=bool)
            for bit_idx, slice_b in enumerate(self.bit_slices):
                bit = self.bit_width - 1 - bit_idx
                vi = bool((v_u >> np.uint64(bit)) & np.uint64(1))
                if vi:
                    eq = np.logical_and(eq, slice_b)
                else:
                    eq = np.logical_and(eq, np.logical_not(slice_b))
                if not np.any(eq):
                    break
            mask = np.logical_or(lt, eq)
        elif operator == "BETWEEN":
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                raise ValueError("BETWEEN requires a (low, high) tuple/list as value")
            low, high = value
            lt_low = self._compute_lt_mask(low)
            lt_high = self._compute_lt_mask(high)
            # <= high = lt_high OR eq_high
            v_u = self._to_unsigned_lex(high)
            eq_high = np.ones(self.n, dtype=bool)
            for bit_idx, slice_b in enumerate(self.bit_slices):
                bit = self.bit_width - 1 - bit_idx
                vi = bool((v_u >> np.uint64(bit)) & np.uint64(1))
                if vi:
                    eq_high = np.logical_and(eq_high, slice_b)
                else:
                    eq_high = np.logical_and(eq_high, np.logical_not(slice_b))
                if not np.any(eq_high):
                    break
            le_high = np.logical_or(lt_high, eq_high)
            mask = np.logical_and(np.logical_not(lt_low), le_high)
        else:
            raise ValueError("Unsupported operator for BitSlicing: " + str(operator))
        # DEBUG: find mismatches
        # if operator == "<=":
        #     print(f"DEBUG: bit_width={self.bit_width}, mask={hex(self.mask)}, bias={hex(self.bias)}")
        #     print(f"DEBUG: min(values)={self.values.min()}, max(values)={self.values.max()}")
        #     v_u = self._to_unsigned_lex(value)
        #     print(f"DEBUG: value={value}, v_u={hex(v_u)}")
        #     baseline_mask = self.values <= value
        #     mismatches = np.where(mask != baseline_mask)[0]
        #     if len(mismatches) > 0:
        #         print(f"DEBUG BitSlicing: {len(mismatches)} mismatches for <= {value}")
        #         for idx in mismatches[:5]:  # show first 5
        #             u_val = self.uvals[idx]
        #             print(f"  Row {idx}: value={self.values[idx]}, uval={hex(u_val)}, mask={mask[idx]}, baseline={baseline_mask[idx]}")

        end = time.perf_counter()
        metrics = {
            "query_time": end - start,
            "bytes_for_slices": int(sum(s.nbytes for s in self.bit_slices)),
            "total_values": int(self.n),
        }
        return {
            "bitset": np.asarray(mask, dtype=bool),
            "matching_row_ids": self.row_ids[mask].tolist(),
            "metrics": metrics,
        }

    def sum_via_slices(self, mask=None):
        """
        Reconstruct sum from slices for non-negative integers.
        If mask is None, sum all values.
        """
        if self.n == 0:
            return 0
        if np.any(self.values < 0):
            raise ValueError("sum_via_slices only supported for non-negative integers")
        slices_lsb_first = list(reversed(self.bit_slices))
        if mask is None:
            counts = [int(s.sum()) for s in slices_lsb_first]
        else:
            counts = [int(np.sum(s[mask])) for s in slices_lsb_first]
        total = 0
        for i, c in enumerate(counts):
            total += c * (1 << i)
        return total