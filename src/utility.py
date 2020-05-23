def replace(array, start, length, value):
    for i in range(start, start + length):
        array[i] = value

    return array
