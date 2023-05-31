import operator
import numpy as np


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Revised with np.array for better efficiency in setting value.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        # self._value = [neutral_element for _ in range(2 * capacity)]
        self._value = np.zeros(2 * capacity) + neutral_element
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def to_array(self, idxs):
        if type(idxs) == list:
            return np.array(idxs)
        elif type(idxs) == np.ndarray:
            return idxs
        else:
            raise TypeError

    def set_items(self, idxs, vals):
        '''input np.arrys, faster than original code'''
        idxs, vals = self.to_array(idxs).copy(), self.to_array(vals).copy()
        idxs += self._capacity
        self._value[idxs] = vals
        idxs //= 2
        for idx in idxs:
            while idx >= 1:
                self._value[idx] = self._operation(
                    self._value[2 * idx],
                    self._value[2 * idx + 1]
                )
                idx //= 2

    def get_items(self, idxs):
        assert np.all((idxs >= 0) & (idxs < self._capacity))
        idxs = self.to_array(idxs).copy()
        idxs += self._capacity
        return self._value[idxs]

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

    def print(self):
        idx = 1
        num = 1
        while num <= self._capacity:
            print('   ' * ((self._capacity - num) // 2) + str(self._value[idx:idx + num]))
            idx += num
            num *= 2


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        if start == 0 and end is None:
            return self._value[1]
        return super(SumSegmentTree, self).reduce(start, end)

    # def set_items(self, idxs, vals):
    #     '''input np.arrys, faster especially for sumtree'''
    #     idxs, vals = self.to_array(idxs).copy(), self.to_array(vals).copy()
    #     idxs += self._capacity
    #     deltas = vals - self._value[idxs]
    #     self._value[idxs] = vals
    #     idxs //= 2
    #     for idx, delta in zip(idxs, deltas):
    #         while idx >= 1:
    #             self._value[idx] += delta
    #             idx //= 2

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""
        if start == 0 and end is None:
            return self._value[1]
        return super(MinSegmentTree, self).reduce(start, end)
