import time
import numpy as np

class BitWeaving:
    """
    BitWeaving-lite: vertical packed bit-planes (word-level).
    Supports operators: '<', '<=', 'BETWEEN', '=', '!=', '>', '>='.
    """

    def __init__(self, segments, bit_width=None, block_size=64):
        parts = []
        rowid_parts = []
        for seg in segments:
            data = seg["data"]
            raw = np.asarray(data)
            if raw.size == 0:
                arr = np.array([], dtype=np.int64)
            elif raw.dtype.kind == "f":
                if not np.all(np.isfinite(raw)):
                    raise ValueError("BitWeaving: float column contains non-finite values")
                frac = np.modf(raw)[0]
                if np.any(frac != 0):
                    raise ValueError("BitWeaving: float column has fractional values")
                arr = raw.astype(np.int64)
            elif raw.dtype.kind in ("i", "u"):
                arr = raw.astype(np.int64)
            else:
                try:
                    arr = np.asarray(raw, dtype=np.int64)
                except Exception:
                    raise ValueError("BitWeaving: column contains non-integer values")
            parts.append(arr)
            base = int(seg.get("base_row", 0))
            rowid_parts.append(np.arange(len(arr), dtype=np.int64) + base)

        if parts:
            self.values = np.concatenate(parts).astype(np.int64)
            self.row_ids = np.concatenate(rowid_parts).astype(np.int64)
        else:
            self.values = np.array([], dtype=np.int64)
            self.row_ids = np.array([], dtype=np.int64)

        self.n = int(len(self.values))
        self.block_size = int(block_size)

        # determine bit_width
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

        # prepare unsigned-lex mapping
        self.mask = (1 << self.bit_width) - 1
        
        # Only use bias if there are negative values
        self.has_negatives = self.n > 0 and np.any(self.values < 0)
        if self.has_negatives:
            self.bias = 1 << (self.bit_width - 1)
        else:
            self.bias = 0  # no bias for all-positive data
            
        if self.n > 0:
            vals_i64 = self.values.astype(np.int64)
            vals_u = vals_i64.astype(np.uint64)
            mask_u = np.uint64(self.mask)
            bias_u = np.uint64(self.bias)
            u = np.bitwise_xor(vals_u, bias_u) & mask_u
            self.uvals = u.astype(np.uint64)
        else:
            self.uvals = np.array([], dtype=np.uint64)

        # number of words (blocks)
        self.n_words = (self.n + self.block_size - 1) // self.block_size
        # valid bits mask per word (last word may be partial)
        self.valid_mask_words = np.zeros(self.n_words, dtype=np.uint64)
        # avoid shifting by 64 (undefined) — handle full 64-bit mask explicitly
        if self.block_size >= 64:
            full_word_mask = np.uint64(0xFFFFFFFFFFFFFFFF)
        else:
            full_word_mask = (np.uint64(1) << np.uint64(self.block_size)) - np.uint64(1)
        for wi in range(self.n_words):
            start = wi * self.block_size
            end = min(start + self.block_size, self.n)
            bits = end - start
            if bits == self.block_size:
                self.valid_mask_words[wi] = full_word_mask
            else:
                if bits >= 64:
                    self.valid_mask_words[wi] = np.uint64(0xFFFFFFFFFFFFFFFF)
                else:
                    self.valid_mask_words[wi] = (np.uint64(1) << np.uint64(bits)) - np.uint64(1)

        # Build per-bit packed words (MSB->LSB). each element is array shape (n_words,) dtype uint64
        self.bit_words = []
        for bit in range(self.bit_width - 1, -1, -1):
            words = np.zeros(self.n_words, dtype=np.uint64)
            for wi in range(self.n_words):
                start = wi * self.block_size
                end = min(start + self.block_size, self.n)
                chunk = self.uvals[start:end]
                word = np.uint64(0)
                for i, val in enumerate(chunk):
                    if (val >> np.uint64(bit)) & np.uint64(1):
                        word |= np.uint64(1) << np.uint64(i)
                words[wi] = word
            self.bit_words.append(words)

    def _to_unsigned_lex(self, v):
        """Convert scalar integer v to unsigned-lex uint64 matching self.uvals."""
        v64 = int(v)
        v_u = np.uint64(v64) ^ np.uint64(self.bias)
        return v_u & np.uint64(self.mask)

    def _not_words(self, words):
        """Safe bitwise NOT for uint64 arrays - XOR with all-ones instead of ~"""
        return np.bitwise_xor(words, np.uint64(0xFFFFFFFFFFFFFFFF))

    def _unpack_words_to_mask(self, words):
        """Unpack array of uint64 words (block_size bits each) to boolean mask of length n."""
        mask = np.zeros(self.n, dtype=bool)
        for wi, word in enumerate(words):
            start = wi * self.block_size
            end = min(start + self.block_size, self.n)
            for i in range(end - start):
                if (word >> np.uint64(i)) & np.uint64(1):
                    mask[start + i] = True
        return mask

    def _compute_lt_words(self, v_u):
        """Compute lt words using MSB->LSB lexicographic comparison."""
        eq_words = self.valid_mask_words.copy()
        lt_words = np.zeros(self.n_words, dtype=np.uint64)
        for bit_idx, words in enumerate(self.bit_words):
            bit = self.bit_width - 1 - bit_idx
            vi = bool((v_u >> np.uint64(bit)) & np.uint64(1))
            if vi:
                # x_bit=0, v_bit=1 => x < v for those positions still equal
                lt_words = lt_words | (eq_words & self._not_words(words))
                eq_words = eq_words & words
            else:
                # x_bit=1, v_bit=0 => x > v; just update eq
                eq_words = eq_words & self._not_words(words)
        return lt_words

    def _compute_eq_words(self, v_u):
        """Compute equality words."""
        eq_words = self.valid_mask_words.copy()
        for bit_idx, words in enumerate(self.bit_words):
            bit = self.bit_width - 1 - bit_idx
            vi = bool((v_u >> np.uint64(bit)) & np.uint64(1))
            if vi:
                eq_words = eq_words & words
            else:
                eq_words = eq_words & self._not_words(words)
        return eq_words

    def query(self, value, operator):
        start = time.perf_counter()
        
        if operator == "<":
            v_u = self._to_unsigned_lex(value)
            lt_words = self._compute_lt_words(v_u)
            mask = self._unpack_words_to_mask(lt_words)
        elif operator == "<=":
            v_u = self._to_unsigned_lex(value)
            lt_words = self._compute_lt_words(v_u)
            eq_words = self._compute_eq_words(v_u)
            le_words = lt_words | eq_words
            mask = self._unpack_words_to_mask(le_words)
        elif operator == ">":
            v_u = self._to_unsigned_lex(value)
            lt_words = self._compute_lt_words(v_u)
            eq_words = self._compute_eq_words(v_u)
            le_words = lt_words | eq_words
            # x > v == not (x <= v), but we must also mask to valid bits
            gt_words = self.valid_mask_words & self._not_words(le_words)
            mask = self._unpack_words_to_mask(gt_words)
        elif operator == ">=":
            v_u = self._to_unsigned_lex(value)
            lt_words = self._compute_lt_words(v_u)
            # x >= v == not (x < v)
            ge_words = self.valid_mask_words & self._not_words(lt_words)
            mask = self._unpack_words_to_mask(ge_words)
        elif operator == "=":
            v_u = self._to_unsigned_lex(value)
            eq_words = self._compute_eq_words(v_u)
            mask = self._unpack_words_to_mask(eq_words)
        elif operator == "!=":
            v_u = self._to_unsigned_lex(value)
            eq_words = self._compute_eq_words(v_u)
            ne_words = self.valid_mask_words & self._not_words(eq_words)
            mask = self._unpack_words_to_mask(ne_words)
        elif operator == "BETWEEN":
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                raise ValueError("BETWEEN requires a (low, high) tuple/list as value")
            low, high = value
            v_low = self._to_unsigned_lex(low)
            v_high = self._to_unsigned_lex(high)
            lt_low = self._compute_lt_words(v_low)
            lt_high = self._compute_lt_words(v_high)
            eq_high = self._compute_eq_words(v_high)
            le_high = lt_high | eq_high
            # x >= low AND x <= high
            ge_low = self.valid_mask_words & self._not_words(lt_low)
            between_words = ge_low & le_high
            mask = self._unpack_words_to_mask(between_words)
        else:
            raise ValueError("Unsupported operator for BitWeaving: " + str(operator))

        end = time.perf_counter()
        metrics = {
            "query_time": end - start,
            "bytes_for_words": int(sum(w.nbytes for w in self.bit_words)),
            "total_values": int(self.n),
        }
        return {
            "bitset": mask,
            "matching_row_ids": self.row_ids[mask].tolist(),
            "metrics": metrics,
        }