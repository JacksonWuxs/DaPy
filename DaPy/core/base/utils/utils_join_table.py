from itertools import chain, repeat
from operator import itemgetter
from collections import deque
from DaPy.core.base.constant import SHEET_DIM
from DaPy.core.base.utils import count_nan

def inner_join(left, other, left_on, right_on, joined):
    # creating the union indexes
    # cost O(4n) in the worst situation
    union_l = left._group_index_by_column_value([left_on])
    union_r = other._group_index_by_column_value([right_on], deque)
    union_inner = set(union_l.keys()) & set(union_r.keys())
    left_ind, right_ind = [], []

    # create index list
    # cost O(n) in the worst situation
    for uni_key in union_inner: 
        lind = union_l.get(uni_key, [])
        rind = union_r.get(uni_key, [])
        for left_index in lind:
            left_ind.extend([left_index] * len(rind))
            right_ind.extend(rind)
    return create_join_by_index(left, other, left_ind, right_ind, joined, False)

def outer_join(left, other, left_on, right_on, joined):
    # creating the union indexes
    # cost O(n) in the constant situation
    union_l = left._group_index_by_column_value([left_on])
    union_r_set = set()
    left_ind, right_ind, unchange, right_tail = [], [], [], []

    # create index list
    # cost O(n) in the worst situation
    for i, value in enumerate(other._data[right_on]):
        lind = union_l.get((value,))
        if lind:
            left_ind.extend(lind)
            right_ind.extend(repeat(i, len(lind)))
            union_r_set.add(value)
        else:
            right_tail.append(i)

    # merge the unmatched index from the left table: O(k)
    for key, val in union_l.items():
        if key[0] not in union_r_set:
            left_ind.extend(val)
            right_ind.extend([-1] * len(val))

    # merge the unmatched index from the right table: O(k)
    right_ind.extend(right_tail)
    return create_join_by_index(left, other, left_ind, right_ind, joined, True)

def left_join(left, right, left_on, right_on, joined):
    union_r = right._group_index_by_column_value([right_on], engine=deque)
    left_ind, right_ind = [], []
    for i, val in enumerate(left[left_on]):
        rind = union_r.get((val,))
        if rind:
            left_ind.extend([i] * len(rind))
            right_ind.extend(rind)
        else:
            left_ind.append(i)
            right_ind.append(-1)
    return create_join_by_index(left, right, left_ind, right_ind, joined, True)

def create_join_by_index(left, other, left_index, right_index, joined, add_last):
    if add_last:
        left.append_row([])
        other.append_row([])
        
    for getter, data in zip([left_index, right_index], [left, other]):
        for miss, (col, seq) in zip(data._missing, data.iter_items()):
            col = joined._check_col_new_name(col)
            subseq = seq[getter]
            if miss != 0:
                miss = count_nan(data._isnan, subseq)
                subseq = [left.nan if data._isnan(v) else v for v in subseq]
            joined._data[col] = subseq
            joined._missing.append(miss)
            joined._columns.append(col)
            
    ln, col = len(max(joined._data.values(), key=len)), len(joined._data)
    joined._dim = SHEET_DIM(ln, col)
    for i, seq in enumerate(joined.iter_values()):
        bias = ln - len(seq)
        seq.extend([joined.nan] * bias)
        joined._missing[i] += bias
    if add_last:
        left.drop_row(-1)
        other.drop_row(-1)
    return joined
