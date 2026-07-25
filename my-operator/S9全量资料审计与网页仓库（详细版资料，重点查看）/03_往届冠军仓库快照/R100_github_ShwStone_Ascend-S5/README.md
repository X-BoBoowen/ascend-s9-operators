# XJTU-宇宙不怎么闪烁

在[昇腾算子挑战赛 S5 赛季](https://www.hiascend.com/developer/contests/details/690885c8c3264eb4947b353e4c2ecb04)中，西安交通大学“宇宙不怎么闪烁”编写了 Copysign 和 BitwiseLeftshift 两题，最终分别是第九名和第十六名。

其实我们曾两度成为Copysign的榜首，但是随着离截止时间越来越近，更多的高手发力了，我们也没有想到新的优化手段，确实是心服口服。

我们的Copysign距离榜首差距不大，在这里分享编写中的一些经验，也虚心向各路高手请教。

代码在Github上开源：[ShwStone/Ascend-S5](https://github.com/ShwStone/Ascend-S5)。

## 基础向量化算法

Copysign是一个向量算子，输入两个向量，输出一个向量。公式如下：

![](images/copysign-formular.png)

这里的等于零其实是浮点数特色：浮点数同时存在 `+0.0` 和 `-0.0` 两种零。可以在 python 中验证：

```python
>>> a = torch.tensor([-0.0,0.0])
>>> b = torch.tensor([1,2])
>>> torch.copysign(b,a)
tensor([-1.,  2.])
```

不过，用于测试的随机数据生成器并没有生成超过千分之一的这种数据，即使没考虑也可以通过。后文我会说明如何正确实现这一 feature。

一般来说，向量算子都是 memory-bound 的。这个点的前提是，计算过程必须完全向量化（即 SIMD）。如何实现有分支结构的 SIMD 呢？

翻看 API 文档，可以找到 [`Select` 接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_0070.html)。它可以根据二进制掩码，在两个向量中逐元素选择，生成一个新向量。这个二进制掩码的格式，和 [`Compare` 接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_0066.html)的输出格式是一样的。所以算法就很明了了：

```cpp
LocalTensor<T> inputLocal = inputQue.DeQue<T>();
LocalTensor<T> otherLocal = otherQue.DeQue<T>();
LocalTensor<T> outLocal = outQue.AllocTensor<T>();
LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

CompareScalar(cmp, otherLocal, T(0), CMPMODE::GE, currentLength);
Abs(inputLocal, inputLocal, currentLength);
Muls(otherLocal, inputLocal, T(-1), currentLength);
Select(outLocal, cmp, inputLocal, otherLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, currentLength);
```

简单来说，根据 `other` 的正负生成一段掩码，根据此掩码在 `Abs(input)` 和 `-Abs(input)` 中选择。

在这个算法基础上，套用一些入门时的 tiling 模板即可。

## bfloat16

这个算子还需要处理 `bfloat16` 类型。这种类型几乎不被任何原生 API 支持。这并不意味着我们没有办法。

第一种方法是，由于 `float` 是 `bfloat16` 精度的超集，我们可以将 `bfloat16` 向上转换为 `float` 类型进行计算，而不损失任何精度。大致如下：

```cpp
LocalTensor<float> inputFloat = inputBuf.Get<float>();
LocalTensor<float> otherFloat = otherBuf.Get<float>();
LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

Cast(otherFloat, otherLocal, RoundMode::CAST_NONE, currentLength);
CompareScalar(cmp, otherFloat, float(0), CMPMODE::GE, currentLength);

Cast(inputFloat, inputLocal, RoundMode::CAST_NONE, currentLength);
Abs(inputFloat, inputFloat, currentLength);
Muls(otherFloat, inputFloat, float(-1), currentLength);

Select(otherFloat, cmp, inputFloat, otherFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE, currentLength);
Cast(outLocal, otherFloat, RoundMode::CAST_TRUNC, currentLength);
```

第二种方法实际上是基于内存实现的 trick。使用 [`LocalTensor::ReinterpretCast` 接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_00110.html)。这个函数类似于 C 语言中的指针强转，在内存不变的情况下把 tensor 视为另一种类型。

根据 [`Cast` 接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_0073.html)中的介绍，`bfloat16` 和 `half` 的内存排布如下：

![bfloat16](images/bfloat16.png)
![half](images/half.png)

S 是符号位，E 是指数位，M 是数值位。可以总结出一下两点：

- 将 `bfloat16` 解释为 `half` 后，符号不变。
- 将转换后的 `half` 做 `Abs` 操作后，再转回 `bfloat16`，效果相当于直接做 `Abs`。

基于这两点，我们就不必使用精度转换的额外空间。代码如下：

```cpp
using T = bfloat16_t;
using F = float16_t;

LocalTensor<F> inputF = inputLocal.ReinterpretCast<F>();
LocalTensor<F> otherF = otherLocal.ReinterpretCast<F>();
LocalTensor<F> outF = outLocal.ReinterpretCast<F>();
LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

CompareScalar(cmp, otherF, F(0), CMPMODE::GE, currentLength);
Abs(inputF, inputF, currentLength);
Muls(otherF, inputF, F(-1), currentLength);
Select(outF, cmp, inputF, otherF, SELMODE::VSEL_TENSOR_TENSOR_MODE, currentLength);
```

## 符号位

如之前所说，为了正确处理正零与负零的情况，可以用如下算法替换 `CompareScalar`：

1. 转换为无符号整形，调用 `RightShift`，右移 15/31 位。
2. 比较结果是否等于 1。

## 广播

所谓广播，是指 `other` 的一些维度为 1，但是对应 `input` 的维度不为 1，则将对应 other 维度进行复制。可以参考：[Boardcasting | Pytorch](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)。

实现广播需要在 host 和 kernel 侧同时下手。分析数据范围可以发现，最后一维保证不会广播。我们把最后一维记作一个 vector,在 tiling 时对 vector 进行分割。

具体来说，要考虑这些点：

1. 复制到 LocalTensor 时要求 32B 对齐，因此要计算一个 vector 向上取整之后的长度。在 Global 上相邻的 vector，在 Local 之间就隔了若干 dummy Bytes。用 DataCopyPad 就可以完美处理这种 dummy Bytes。
2. 核内的迭代也以 vector 为最小单位。原先一个 Loop 可以处理多个元素，现在是一个 Loop 处理多个 vector。处理数量由 `UB大小 / vector对齐后大小 / 同时存在的tensor数量` 决定。

读入用循环完成，因为不同位置的 `input` 对应 `other` 位置没有规律。有如下简单的算法：根据每一维是否广播，可以确定对应的循环步长。如果需要广播，则步长为零，也就是一直待在第零维；否则就是正常情况。

```cpp
for (int32_t i = 0; i < currentVector; i++) {
    int32_t inputID = i + beginVector;

    int32_t dim1 = inputID / (outDim2 * outDim3);
    int32_t dim2 = inputID / outDim3 % outDim2;
    int32_t dim3 = inputID % outDim3;

    int32_t otherID = dim1 * otherStride1 * otherDim2 * otherDim3 +
                        dim2 * otherStride2 * otherDim3 +
                        dim3 * otherStride3;

    DataCopyPad(inputLocal[i * alignVectorLength],
                inputGM[inputID * vectorLength], copyParams, padParams);
    DataCopyPad(otherLocal[i * alignVectorLength],
                otherGM[otherID * vectorLength], copyParams, padParams);
}
```

上面的代码对前三维进行枚举，也就是以 vector 为单位读入。`inputID` 是当前读入 vector 在 `input` 中的一维位置。通过除法转化为三维，然后乘以步长就可以得到 `otherID`。

输出则比较简单，因为不存在广播的问题，可以用 `DataCopyPad` 一次性解决。

```cpp
LocalTensor<T> outLocal = outQue.DeQue<T>();

int32_t contentSize = vectorLength * sizeof(T);
int32_t alignSize = alignVectorLength * sizeof(T);
int32_t srcStride = (alignSize - (contentSize + 31) / 32 * 32) / 32;

DataCopyExtParams copyParams{
    static_cast<uint16_t>(currentVector),
    static_cast<uint32_t>(vectorLength * sizeof(T)),
    static_cast<uint32_t>(srcStride), 0, 0};
DataCopyPad(outGM[beginVector * vectorLength], outLocal, copyParams);

outQue.FreeTensor(outLocal);
```

那么，怎么确定哪些维度需要广播呢？这里需要 Host 侧操作。注意广播是需要两向量维度向右对齐的，所以处理起来有很多减号：

```cpp
uint8_t board_cast = 0;
int32_t out_dims[3];

for (int32_t i = 0; i < 3; i++) {

    int32_t other_dim = 1, out_dim = 1;
    if (i + other_dim_num - 4 >= 0)
        other_dim = other_shape.GetDim(i + other_dim_num - 4);
    if (i + out_dim_num - 4 >= 0)
        out_dim = out_shape.GetDim(i + out_dim_num - 4);

    if (other_dim < out_dim) board_cast |= (uint8_t(1) << i);
    out_dims[i] = out_dim;
}

tiling.set_board_cast(board_cast);
```

在这里，我们把广播信息用二进制位的形式存到一个 `uint8` 里，传入 tiling 结构体即可。

`Compute` 函数不需要任何修改。

## 优化：Double Buffer

通过对不同计算单元运行时间的分析，可以清楚的发现，MTE2/MTE3 占据的时间超过了 95%，说明算子是标准的 memory bound。那么，对于算法的优化就意义不大，应该重点优化读写。

注意到运行时间约等于 MTE2+MTE3，说明读入和输出互相阻塞了。那按照 [使能 double buffer](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0033.html) 进行优化，通过同时存在两个 tensor 的方式使得各单元可以并行。

这个比较简单，在 `pipe.InitBuffer` 处将 1 改成 2 即可。但是要注意，在计算单核能存下的最大元素数时，要考虑到现在的内存占用是翻倍的。

```cpp
int32_t tile_block = ub_size / ub_per_block / BUFFER_NUM; // BUFFER_NUM = 2
pipe.InitBuffer(inputQue, BUFFER_NUM, tileLength * sizeof(T));
pipe.InitBuffer(otherQue, BUFFER_NUM, tileLength * sizeof(T));
pipe.InitBuffer(outQue, BUFFER_NUM, tileLength * sizeof(T));
```

优化后运行时间约等于 max(MTE2,MTE3)。

## 优化：多核读取 L2 Cache

假设一共 20 个核，每个核迭代 10 次。由于各核是并行的，所以，每个核第一次被迭代的 20 个数据块其实是同时被读取的。此时，如果这些数据块是相邻的，就可以充分发挥 L2 Cache 的优势。反之，按照传统 tiling 方案，让每个核负责的 10 个 数据块相邻，就会降低缓存命中率。

换而言之，一共两百份的数据，之前每个核分走连续的 10 块，现在每个核分走跳跃的 10 块，间距是 19。

这个实现比较复杂。因为核间 tiling 存在大小核的情况，使得两个数据块的间距是不定的；同时，因为核内 tiling 还包括尾块，所以需要对尾块特殊处理。

给出参考代码如下：

```cpp
// in CopysignKernel::Init

coreLength = (coreID < former_num) ? former_length : tail_length;

tileLength = tile_length;
tileNum = (coreLength / tileLength);
lastTileLength = coreLength - tileLength * tileNum;

if (lastTileLength > 0) tileNum++;
else lastTileLength = tileLength;

loopLength = tileLength * GetBlockNum();

// former/tail 处理大小核，lastTile 处理尾块
int32_t formerTileNum = former_length / tile_length;
int32_t formerLastTile = former_length - formerTileNum * tile_length;
int32_t tailTileNum = tail_length / tile_length;
int32_t tailLastTile = tail_length - tailTileNum * tile_length;

if (formerTileNum == tailTileNum) {
    if (coreID < former_num) lastLoopLength = coreID * (formerLastTile - tileLength);
    else lastLoopLength = former_num * (formerLastTile - tileLength) + (coreID - former_num) * (tailLastTile - tileLength);
} else {
    if (coreID < former_num) lastLoopLength = coreID * (formerLastTile - tileLength);
    else lastLoopLength = 0;
}
if (tileNum == 1) lastLoopLength = 0;

int32_t totalLength = former_length * former_num + tail_length * (GetBlockNum() - former_num);
int32_t beginLength = (tileNum > 1 ? coreID * tileLength : (
    (coreID < former_num)
        ? (former_length * coreID)
        : (tail_length * coreID +
            (former_length - tail_length) * former_num)));
```

```cpp
// in CopysignKernel::CopyIn

int32_t currentLength =
    (progress == tileNum - 1) ? lastTileLength : tileLength;

// 本次循环数据块的起点

int32_t currentPosition = progress * loopLength;
if (progress == tileNum - 1) currentPosition += lastLoopLength;
```
