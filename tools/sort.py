from more_itertools import time_limited
import taichi as ti


@ti.func
def getindex_withkeys(value, keys, value_copy, index, row, left, right):
    i = left - 1
    povit = value_copy[row, index[row, right]]

    for j in range(left, right):
        if value_copy[row, index[row, j]] <= povit:
            i += 1
            index[row, i], index[row, j] = index[row, j], index[row, i]
            keys[row, i], keys[row, j] = keys[row, j], keys[row, i]
            value[row, i], value[row, j] = (
                value[row, j],
                value[row, i],
            )
    index[row, i + 1], index[row, right] = index[row, right], index[row, i + 1]
    keys[row, i + 1], keys[row, right] = keys[row, right], keys[row, i + 1]
    value[row, i + 1], value[row, right] = (
        value[row, right],
        value[row, i + 1],
    )
    return i + 1


@ti.func
def quicksort_row_withkeys(value, stack, keys, value_copy, index, row, left, right):

    top = -1
    top += 1
    stack[row, top] = left
    top += 1
    stack[row, top] = right
    while top >= 0:
        right = stack[row, top]
        top -= 1
        left = stack[row, top]
        top -= 1
        p = getindex_withkeys(value, keys, value_copy, index, row, left, right)
        if p - 1 > left:
            top += 1
            stack[row, top] = left
            top += 1
            stack[row, top] = p - 1
        if p + 1 < right:
            top += 1
            stack[row, top] = p + 1
            top += 1
            stack[row, top] = right


@ti.kernel
def parallel_row_withkeys(
    value: ti.template(),
    stack: ti.template(),
    keys: ti.template(),
    value_copy: ti.template(),
    index: ti.template(),
):

    left, right = 0, value.shape[1] - 1
    for row in range(value.shape[0]):
        quicksort_row_withkeys(value, stack, keys, value_copy, index, row, left, right)


@ti.func
def getindex_withoutkeys(value, row, left, right):
    i = left - 1
    povit = value[row, right]

    for j in range(left, right):
        if value[row, j] <= povit:
            i += 1
            value[row, i], value[row, j] = (
                value[row, j],
                value[row, i],
            )
    value[row, i + 1], value[row, right] = (value[row, right], value[row, i + 1])
    return i + 1


@ti.func
def quicksort_row_withoutkeys(value, stack, row, left, right):

    top = -1
    top += 1
    stack[row, top] = left
    top += 1
    stack[row, top] = right
    while top >= 0:
        right = stack[row, top]
        top -= 1
        left = stack[row, top]
        top -= 1
        p = getindex_withoutkeys(value, row, left, right)
        if p - 1 > left:
            top += 1
            stack[row, top] = left
            top += 1
            stack[row, top] = p - 1
        if p + 1 < right:
            top += 1
            stack[row, top] = p + 1
            top += 1
            stack[row, top] = right


@ti.kernel
def parallel_row_withoutkeys(value: ti.template(), stack: ti.template()):

    left, right = 0, value.shape[1] - 1
    for row in range(value.shape[0]):
        quicksort_row_withoutkeys(value, stack, row, left, right)


def quicksort(value, keys=None):
    """
    使用快速排序来对2Dfield每一行进行从小到大排序,每一行是串行排序,不同行之间并行排序.
    value : 2D field. 待排序的field.
    keys : 基于排序value来排序keys.
           如果没有的话,则返回排序后的values.
    """

    fb1 = ti.FieldsBuilder()
    stack = ti.field(dtype=ti.i32)  # 模拟堆栈
    fb1.dense(ti.ij, value.shape).place(stack)
    fb1_snode_tree = fb1.finalize()

    if keys is None:
        parallel_row_withoutkeys(value, stack)
        fb1_snode_tree.destroy()
    else:
        assert keys.shape == value.shape
        fb2 = ti.FieldsBuilder()
        value_copy = ti.field(dtype=value.dtype)  # value的一份复制
        fb2.dense(ti.ij, value.shape).place(value_copy)
        fb2_snode_tree = fb2.finalize()
        value_copy.copy_from(value)

        fb3 = ti.FieldsBuilder()
        index = ti.field(dtype=ti.i32)  # value的index
        fb3.dense(ti.ij, value.shape).place(index)
        fb3_snode_tree = fb3.finalize()

        @ti.kernel
        def fill_index(index: ti.template()):
            for i in range(index.shape[0]):
                for j in range(index.shape[1]):
                    index[i, j] = j

        fill_index(index)

        parallel_row_withkeys(value, stack, keys, value_copy, index)
        fb1_snode_tree.destroy()
        fb2_snode_tree.destroy()
        fb3_snode_tree.destroy()


if __name__ == "__main__":
    import numpy as np
    from time import time

    ti.init(arch=ti.cpu)

    def get_time_sort(N):
        col = 50
        arr_np = np.random.random((N, col))
        # arr_ti = ti.field(ti.f64, shape=(N, col))

        # @ti.kernel
        # def fill():
        #     for i in range(N):
        #         for j in range(col):
        #             arr_ti[i, j] = ti.random(ti.f64)

        # fill()
        # arr_ti.from_numpy(arr_np)
        # print(f"Start sorting a field with shape of {arr_ti.shape}...")
        # s = time()
        # quicksort(arr_ti)
        # e = time()
        # print("taichi sort without keys", e - s, "s.")
        # t1 = e - s
        s = time()
        np.sort(arr_np)
        e = time()
        t2 = e - s
        print("numpy sort without keys", e - s, "s.")
        # print(N, col, t1, t2)

    def get_time_sort_withkeys(N):
        col = 30
        arr_np = np.random.random((N, col))
        arr_ti = ti.field(ti.f64, shape=(N, col))
        arr_ti.from_numpy(arr_np)
        print(f"Start sorting a field with shape of {arr_ti.shape}...")
        s = time()
        quicksort(arr_ti, arr_ti)
        e = time()
        print("taichi sort without keys", e - s, "s.")
        s = time()
        np.sort(arr_np)
        e = time()
        print("numpy sort without keys", e - s, "s.")

    # for N in [100, 1000, 10000, 100000, 1000000]:
    #     t1, t2 = get_time_sort(N)
    #     time_list.append([N, 80, t1, t2])
    # print(time_list)
    N = 20000000
    get_time_sort(N)
