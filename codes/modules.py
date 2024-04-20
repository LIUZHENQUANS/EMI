from layers import *
from keras.layers import concatenate

def Inception(size, rank, input):
    x_pool = pool(size, input)
    x_0 = Conv_1(size, input)

    x_1 = Conv_1(size, input)
    x_1 = Conv(size, x_1)

    x_2 = Conv_1(size, input)
    x_2 = Conv(size, x_2)
    x_2 = Conv(size, x_2)

    x_3 = Conv_1(size, input)
    x_3 = Conv(size, x_3)
    x_3 = Conv(size, x_3)
    x_3 = Conv(size, x_3)

    x_4 = Conv_1(size, input)
    x_4 = Conv(size, x_4)
    x_4 = Conv(size, x_4)
    x_4 = Conv(size, x_4)
    x_4 = Conv(size, x_4)

    x_5 = Conv_1(size, input)
    x_5 = Conv(size, x_5)
    x_5 = Conv(size, x_5)
    x_5 = Conv(size, x_5)
    x_5 = Conv(size, x_5)
    x_5 = Conv(size, x_5)

    x_6 = Conv_1(size, input)
    x_6 = Conv(size, x_6)
    x_6 = Conv(size, x_6)
    x_6 = Conv(size, x_6)
    x_6 = Conv(size, x_6)
    x_6 = Conv(size, x_6)
    x_6 = Conv(size, x_6)

    if rank == 1:
        return concatenate([x_pool, x_0, x_1])
    elif rank == 2:
        return concatenate([x_pool, x_0, x_1, x_2])
    elif rank == 3:
        return concatenate([x_pool, x_0, x_1, x_2, x_3])
    elif rank == 4:
        return concatenate([x_pool, x_0, x_1, x_2, x_3, x_4])
    elif rank == 5:
        return concatenate([x_pool, x_0, x_1, x_2, x_3, x_4, x_5])
    elif rank == 6:
        return concatenate([x_pool, x_0, x_1, x_2, x_3, x_4, x_5, x_6])


def Eception(size, rank, input):
    x_pool = pool(size, input)
    x0 = Conv_1(size, input)

    x = Conv_1(size, input)

    x10 = Conv(size, x)
    x11 = Conv(size, x)

    x20 = Conv(size, x10)
    x21 = Conv(size, x10)

    x30 = Conv(size, x20)
    x31 = Conv(size, x20)

    x40 = Conv(size, x30)
    x41 = Conv(size, x30)

    x50 = Conv(size, x40)
    x51 = Conv(size, x40)

    x60 = Conv(size, x50)
    x61 = Conv(size, x50)

    x70 = Conv(size, x60)
    x71 = Conv(size, x60)

    x80 = Conv(size, x70)
    x81 = Conv(size, x70)


    if rank == 1:
        return concatenate([x_pool, x0, x11])
    elif rank == 2:
        return concatenate([x_pool, x0, x11, x21])
    elif rank == 3:
        return concatenate([x_pool, x0, x11, x21, x31])
    elif rank == 4:
        return concatenate([x_pool, x0, x11, x21, x31, x41])
    elif rank == 5:
        return concatenate([x_pool, x0, x11, x21, x31, x41, x51])
    elif rank == 6:
        return concatenate([x_pool, x0, x11, x21, x31, x41, x51, x61])
    elif rank == 8:
        return concatenate([x_pool, x0, x11, x21, x31, x41, x51, x61, x71, x81])



def Lception(size, rank, input):
    x_pool = pool(size, input)
    x0 = Conv_1(size, input)

    x = Conv_1(size, input)

    x10 = DWConv(size, x)
    x11 = DWConv(size, x)

    x20 = Conv(size, x10)
    x21 = Conv(size, x10)

    x30 = DWConv(size, x20)
    x31 = DWConv(size, x20)

    x40 = Conv(size, x30)
    x41 = Conv(size, x30)

    x50 = DWConv(size, x40)
    x51 = DWConv(size, x40)

    x60 = Conv(size, x50)
    x61 = Conv(size, x50)

    x70 = DWConv(size, x60)
    x71 = DWConv(size, x60)

    x80 = Conv(size, x70)
    x81 = Conv(size, x70)

    if rank == 1:
        return concatenate([x_pool, x0, x11])
    elif rank == 2:
        return concatenate([x_pool, x0, x11, x21])
    elif rank == 3:
        return concatenate([x_pool, x0, x11, x21, x31])
    elif rank == 4:
        return concatenate([x_pool, x0, x11, x21, x31, x41])
    elif rank == 5:
        return concatenate([x_pool, x0, x11, x21, x31, x41, x51])
    elif rank == 6:
        return concatenate([x_pool, x0, x11, x21, x31, x41, x51, x61])
    elif rank == 8:
        return concatenate([x_pool, x0, x11, x21, x31, x41, x51, x61, x71, x81])