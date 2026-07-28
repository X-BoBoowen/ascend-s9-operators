#include "kernel_operator.h"
constexpr uint32_t UB_BYTES = 32 * 1024;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t MAX_COPY_ROWS = 4095;

class KernelConcat {
public:
    __aicore__ inline KernelConcat() {}

    __aicore__ inline void Init(
        GM_ADDR y,
        uint64_t outer,
        uint64_t outRowBytes,
        uint64_t baseRowsPerBlock,
        uint32_t extraBlocks,
        uint32_t copyMode,
        uint64_t baseWorkBlocks,
        uint32_t extraWorkBlocks,
        uint32_t blockIdx)
    {
        outer_ = outer;
        outRowBytes_ = outRowBytes;
        rows_ = baseRowsPerBlock + (blockIdx < extraBlocks ? 1U : 0U);
        firstRow_ =
            blockIdx * baseRowsPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        copyMode_ = copyMode;
        workBlocks_ =
            baseWorkBlocks +
            (blockIdx < extraWorkBlocks ? 1U : 0U);
        firstWorkBlock_ =
            blockIdx * baseWorkBlocks +
            (blockIdx < extraWorkBlocks
                ? blockIdx
                : extraWorkBlocks);

        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(y),
            outer_ * outRowBytes_);
        pipe_.InitBuffer(
            copyBuffer_,
            UB_BYTES);
        mte2ToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(
                AscendC::HardEvent::MTE2_MTE3));
        mte3ToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(
                AscendC::HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void CopyOne(
        GM_ADDR inputAddress,
        uint64_t widthBytes,
        uint64_t outputOffsetBytes)
    {
        if (rows_ == 0 || widthBytes == 0) {
            return;
        }

        AscendC::GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(inputAddress),
            outer_ * widthBytes);
        CopyInput(inputGm, widthBytes, outputOffsetBytes);
    }

    __aicore__ inline void CopyFlat(
        GM_ADDR x,
        uint32_t inputCount,
        const uint64_t* inputRowBytes)
    {
        if (copyMode_ != 1 || workBlocks_ == 0 ||
            outRowBytes_ == 0) {
            return;
        }
        const uint64_t totalBytes = outer_ * outRowBytes_;
        uint64_t cursor = firstWorkBlock_ * ALIGN_BYTES;
        uint64_t end =
            (firstWorkBlock_ + workBlocks_) * ALIGN_BYTES;
        if (end > totalBytes) {
            end = totalBytes;
        }
        if (cursor >= end) {
            return;
        }

        AscendC::GlobalTensor<uint64_t> descriptor;
        descriptor.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint64_t*>(x),
            inputCount + 256);
        const uint64_t pointerOffsetWords =
            descriptor.GetValue(0) / sizeof(uint64_t);

        while (cursor < end) {
            const uint64_t row = cursor / outRowBytes_;
            const uint64_t rowBase = row * outRowBytes_;
            const uint64_t rowStart = cursor - rowBase;
            uint64_t rowEnd = end - rowBase;
            if (rowEnd > outRowBytes_) {
                rowEnd = outRowBytes_;
            }

            uint64_t inputPrefix = 0;
            for (uint32_t i = 0; i < inputCount; ++i) {
                const uint64_t widthBytes = inputRowBytes[i];
                const uint64_t inputEnd = inputPrefix + widthBytes;
                if (widthBytes != 0 &&
                    rowStart < inputEnd &&
                    rowEnd > inputPrefix) {
                    const uint64_t pieceStart =
                        rowStart > inputPrefix
                            ? rowStart
                            : inputPrefix;
                    const uint64_t pieceEnd =
                        rowEnd < inputEnd
                            ? rowEnd
                            : inputEnd;
                    const uint64_t inputAddress =
                        descriptor.GetValue(pointerOffsetWords + i);
                    CopyFlatPiece(
                        reinterpret_cast<GM_ADDR>(inputAddress),
                        row * widthBytes +
                            pieceStart - inputPrefix,
                        rowBase + pieceStart,
                        pieceEnd - pieceStart);
                }
                inputPrefix = inputEnd;
                if (inputPrefix >= rowEnd) {
                    break;
                }
            }
            cursor = rowBase + rowEnd;
        }
    }

private:
    __aicore__ inline void CopyFlatPiece(
        GM_ADDR inputAddress,
        uint64_t inputOffsetBytes,
        uint64_t outputOffsetBytes,
        uint64_t lengthBytes)
    {
        AscendC::GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(inputAddress),
            inputOffsetBytes + lengthBytes);
        for (uint64_t offset = 0;
             offset < lengthBytes;
             offset += UB_BYTES) {
            const uint32_t chunkBytes = static_cast<uint32_t>(
                lengthBytes - offset < UB_BYTES
                    ? lengthBytes - offset
                    : UB_BYTES);
            const uint32_t alignedChunkBytes =
                (chunkBytes + ALIGN_BYTES - 1U) /
                ALIGN_BYTES * ALIGN_BYTES;
            AscendC::LocalTensor<uint8_t> local =
                copyBuffer_.Get<uint8_t>();

            AscendC::DataCopyExtParams inputParams;
            inputParams.blockCount = 1;
            inputParams.blockLen = chunkBytes;
            inputParams.srcStride = 0;
            inputParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<uint8_t> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = static_cast<uint8_t>(
                alignedChunkBytes - chunkBytes);
            padParams.paddingValue = 0;
            AscendC::DataCopyPad(
                local,
                inputGm[inputOffsetBytes + offset],
                inputParams,
                padParams);
            AscendC::SetFlag<
                AscendC::HardEvent::MTE2_MTE3>(
                    mte2ToMte3Event_);
            AscendC::WaitFlag<
                AscendC::HardEvent::MTE2_MTE3>(
                    mte2ToMte3Event_);

            AscendC::DataCopyExtParams outputParams;
            outputParams.blockCount = 1;
            outputParams.blockLen = chunkBytes;
            outputParams.srcStride = 0;
            outputParams.dstStride = 0;
            AscendC::DataCopyPad(
                yGm_[outputOffsetBytes + offset],
                local,
                outputParams);
            AscendC::SetFlag<
                AscendC::HardEvent::MTE3_MTE2>(
                    mte3ToMte2Event_);
            AscendC::WaitFlag<
                AscendC::HardEvent::MTE3_MTE2>(
                    mte3ToMte2Event_);
        }
    }

    __aicore__ inline void CopyInput(
        AscendC::GlobalTensor<uint8_t>& inputGm,
        uint64_t widthBytes,
        uint64_t outputOffsetBytes)
    {
        for (uint64_t chunkOffset = 0;
             chunkOffset < widthBytes;
             chunkOffset += UB_BYTES) {
            const uint32_t chunkBytes = static_cast<uint32_t>(
                widthBytes - chunkOffset < UB_BYTES
                    ? widthBytes - chunkOffset
                    : UB_BYTES);
            const uint32_t alignedChunkBytes =
                (chunkBytes + ALIGN_BYTES - 1U) /
                ALIGN_BYTES * ALIGN_BYTES;
            uint32_t rowsPerCopy =
                UB_BYTES / alignedChunkBytes;
            rowsPerCopy = rowsPerCopy < MAX_COPY_ROWS
                ? rowsPerCopy
                : MAX_COPY_ROWS;

            const uint64_t inputStride =
                widthBytes - chunkBytes;
            const uint64_t outputStride =
                outRowBytes_ - chunkBytes;
            if (inputStride > UINT32_MAX ||
                outputStride > UINT32_MAX) {
                rowsPerCopy = 1;
            }

            for (uint64_t rowOffset = 0;
                 rowOffset < rows_;
                 rowOffset += rowsPerCopy) {
                const uint32_t currentRows =
                    static_cast<uint32_t>(
                        rows_ - rowOffset < rowsPerCopy
                            ? rows_ - rowOffset
                            : rowsPerCopy);
                AscendC::LocalTensor<uint8_t> local =
                    copyBuffer_.Get<uint8_t>();

                AscendC::DataCopyPadExtParams<uint8_t> padParams;
                padParams.isPad = true;
                padParams.leftPadding = 0;
                padParams.rightPadding = static_cast<uint8_t>(
                    alignedChunkBytes - chunkBytes);
                padParams.paddingValue = 0;

                AscendC::DataCopyExtParams inputCopyParams;
                inputCopyParams.blockLen = chunkBytes;
                inputCopyParams.dstStride = 0;
                if (alignedChunkBytes == chunkBytes ||
                    currentRows >= 16) {
                    inputCopyParams.blockCount =
                        static_cast<uint16_t>(currentRows);
                    inputCopyParams.srcStride =
                        currentRows == 1
                            ? 0
                            : static_cast<uint32_t>(inputStride);
                    AscendC::DataCopyPad(
                        local,
                        inputGm[
                            (firstRow_ + rowOffset) * widthBytes +
                            chunkOffset],
                        inputCopyParams,
                        padParams);
                } else {
                    inputCopyParams.blockCount = 1;
                    inputCopyParams.srcStride = 0;
                    for (uint32_t row = 0;
                         row < currentRows;
                         ++row) {
                        AscendC::DataCopyPad(
                            local[row * alignedChunkBytes],
                            inputGm[
                                (firstRow_ + rowOffset + row) *
                                    widthBytes +
                                chunkOffset],
                            inputCopyParams,
                            padParams);
                    }
                }
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE2_MTE3>(
                        mte2ToMte3Event_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE2_MTE3>(
                        mte2ToMte3Event_);

                AscendC::DataCopyExtParams outputCopyParams;
                outputCopyParams.blockCount =
                    static_cast<uint16_t>(currentRows);
                outputCopyParams.blockLen = chunkBytes;
                outputCopyParams.srcStride = 0;
                outputCopyParams.dstStride =
                    currentRows == 1
                        ? 0
                        : static_cast<uint32_t>(outputStride);
                AscendC::DataCopyPad(
                    yGm_[
                        (firstRow_ + rowOffset) * outRowBytes_ +
                        outputOffsetBytes +
                        chunkOffset],
                    local,
                    outputCopyParams);
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE3_MTE2>(
                        mte3ToMte2Event_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE3_MTE2>(
                        mte3ToMte2Event_);
            }
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> copyBuffer_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint64_t outer_;
    uint64_t outRowBytes_;
    uint64_t rows_;
    uint64_t firstRow_;
    uint32_t copyMode_;
    uint64_t workBlocks_;
    uint64_t firstWorkBlock_;
    event_t mte2ToMte3Event_;
    event_t mte3ToMte2Event_;
};

extern "C" __global__ __aicore__ void concat(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.outer == 0 ||
        tilingData.outRowBytes == 0) {
        return;
    }
    KernelConcat op;
    op.Init(
        y,
        tilingData.outer,
        tilingData.outRowBytes,
        tilingData.baseRowsPerBlock,
        tilingData.extraBlocks,
        tilingData.copyMode,
        tilingData.baseWorkBlocks,
        tilingData.extraWorkBlocks,
        AscendC::GetBlockIdx());

    if (tilingData.copyMode == 1) {
        op.CopyFlat(
            x,
            tilingData.inputCount,
            tilingData.inputRowBytes);
        return;
    }

    AscendC::GlobalTensor<uint64_t> descriptor;
    descriptor.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint64_t*>(x),
        tilingData.inputCount + 256);
    const uint64_t pointerOffsetWords =
        descriptor.GetValue(0) / sizeof(uint64_t);
    uint64_t outputOffsetBytes = 0;
    for (uint32_t i = 0; i < tilingData.inputCount; ++i) {
        const uint64_t inputAddress =
            descriptor.GetValue(pointerOffsetWords + i);
        const uint64_t widthBytes =
            tilingData.inputRowBytes[i];
        op.CopyOne(
            reinterpret_cast<GM_ADDR>(inputAddress),
            widthBytes,
            outputOffsetBytes);
        outputOffsetBytes += widthBytes;
    }
}
