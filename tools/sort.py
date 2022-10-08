import taichi as ti


@ti.func
def getindex_with_keys(value, keys, row, left, right):
    i = left - 1
    povit = value[row, keys[row, right]]

    for j in range(left, right):
        if value[row, keys[row, j]] <= povit:
            i += 1
            keys[row, i], keys[row, j] = keys[row, j], keys[row, i]
    keys[row, i + 1], keys[row, right] = keys[row, right], keys[row, i + 1]
    return i + 1


@ti.func
def quicksort_row_with_keys(value, stack, keys, row, left, right):

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
        p = getindex_with_keys(value, keys, row, left, right)
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
def parallel_row_with_keys(
    value: ti.template(), stack: ti.template(), keys: ti.template()
):

    left, right = 0, value.shape[1] - 1
    for row in range(value.shape[0]):
        quicksort_row_with_keys(value, stack, keys, row, left, right)


@ti.func
def getindex_without_keys(value, row, left, right):
    i = left - 1
    povit = value[row, right]

    for j in range(left, right):
        if value[row, j] <= povit:
            i += 1
            value[row, i], value[row, j] = value[row, j], value[row, i]
    value[row, i + 1], value[row, right] = value[row, right], value[row, i + 1]
    return i + 1


@ti.func
def quicksort_row_without_keys(value, stack, row, left, right):

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
        p = getindex_without_keys(value, row, left, right)
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
def parallel_row_without_keys(value: ti.template(), stack: ti.template()):

    left, right = 0, value.shape[1] - 1
    for row in range(value.shape[0]):
        quicksort_row_without_keys(value, stack, row, left, right)


def quicksort(value, keys=False):
    """
    使用快速排序来对2Dfield每一行进行从小到大排序,每一行是串行排序,不同行之间并行排序.
    value : 2D field.
    keys : 如果True,则返回排序value的index的2Dfield, value没有变化.
           如果False,则返回排序后的values.
    """

    fb = ti.FieldsBuilder()
    stack = ti.field(dtype=ti.i32)
    fb.dense(ti.ij, value.shape).place(stack)
    fb_snode_tree = fb.finalize()

    if keys:
        keys = ti.field(ti.i32, shape=value.shape)

        @ti.kernel
        def fill(keys: ti.template()):
            for i in range(keys.shape[0]):
                for j in range(keys.shape[1]):
                    keys[i, j] = j

        fill(keys)
        parallel_row_with_keys(value, stack, keys)
        fb_snode_tree.destroy()
        return keys
    else:
        parallel_row_without_keys(value, stack)
        fb_snode_tree.destroy()


if __name__ == "__main__":
    from time import time
    import numpy as np

    ti.init(arch=ti.cpu)

    arr_ti = ti.field(ti.f32, shape=(3000000, 80))

    @ti.kernel
    def fill(arr: ti.template()):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr[i, j] = ti.random(ti.f32)

    fill(arr_ti)
    arr_np = arr_ti.to_numpy()

    print(f"Start sorting a field with shape of {arr_ti.shape}...")
    s = time()
    quicksort(arr_ti)
    e = time()
    print("taichi:", e - s)

    s = time()
    np.sort(arr_np)
    e = time()
    print("numpy:", e - s)

    print("checking array is sorted...")

    @ti.kernel
    def check(arr: ti.template()) -> int:
        tmp = 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1] - 1):
                if arr[i, j] > arr[i, j + 1]:
                    tmp += 1
        return tmp

    if check(arr_ti) == 0:
        print("array is sorted.")
