#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;

class KernelConcatFast {
public:
    __aicore__ inline KernelConcatFast() {}

    __aicore__ inline void Init(
        GM_ADDR x0,
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR x3,
        GM_ADDR x4,
        GM_ADDR x5,
        GM_ADDR x6,
        GM_ADDR x7,
        GM_ADDR x8,
        GM_ADDR x9,
        GM_ADDR x10,
        GM_ADDR x11,
        GM_ADDR x12,
        GM_ADDR x13,
        GM_ADDR x14,
        GM_ADDR x15,
        GM_ADDR y,
        uint32_t outer,
        uint32_t outInner,
        uint32_t inputCount,
        uint32_t baseRowsPerBlock,
        uint32_t extraBlocks,
        uint32_t tileRows,
        uint32_t maxAlignedWidth,
        const uint32_t* widths,
        const uint32_t* offsets,
        uint32_t blockIdx)
    {
        inputAddresses_[0] = x0;
        inputAddresses_[1] = x1;
        inputAddresses_[2] = x2;
        inputAddresses_[3] = x3;
        inputAddresses_[4] = x4;
        inputAddresses_[5] = x5;
        inputAddresses_[6] = x6;
        inputAddresses_[7] = x7;
        inputAddresses_[8] = x8;
        inputAddresses_[9] = x9;
        inputAddresses_[10] = x10;
        inputAddresses_[11] = x11;
        inputAddresses_[12] = x12;
        inputAddresses_[13] = x13;
        inputAddresses_[14] = x14;
        inputAddresses_[15] = x15;
        outer_ = outer;
        outInner_ = outInner;
        inputCount_ = inputCount;
        tileRows_ = tileRows;
        maxAlignedWidth_ = maxAlignedWidth;
        rows_ = baseRowsPerBlock + (blockIdx < extraBlocks ? 1U : 0U);
        firstRow_ =
            blockIdx * baseRowsPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        for (uint32_t i = 0; i < inputCount_; ++i) {
            widths_[i] = widths[i];
            offsets_[i] = offsets[i];
        }

        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(y),
            outer_ * outInner_);
        pipe_.InitBuffer(
            inputQueue_,
            BUFFER_NUM,
            tileRows_ * maxAlignedWidth_ * sizeof(half));
        pipe_.InitBuffer(
            outputQueue_,
            BUFFER_NUM,
            tileRows_ * maxAlignedWidth_ * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t inputIdx = 0;
             inputIdx < inputCount_;
             ++inputIdx) {
            const uint32_t width = widths_[inputIdx];
            if (width == 0) {
                continue;
            }
            AscendC::GlobalTensor<half> inputGm;
            inputGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(
                    inputAddresses_[inputIdx]),
                outer_ * width);

            for (uint32_t rowOffset = 0;
                 rowOffset < rows_;
                 rowOffset += tileRows_) {
                const uint32_t currentRows =
                    (rows_ - rowOffset < tileRows_)
                        ? rows_ - rowOffset
                        : tileRows_;
                CopySegment(
                    inputGm,
                    width,
                    offsets_[inputIdx],
                    rowOffset,
                    currentRows);
            }
        }
    }

private:
    __aicore__ inline void CopySegment(
        AscendC::GlobalTensor<half>& inputGm,
        uint32_t width,
        uint32_t outputOffset,
        uint32_t rowOffset,
        uint32_t currentRows)
    {
        AscendC::LocalTensor<half> inputLocal =
            inputQueue_.AllocTensor<half>();

        AscendC::DataCopyExtParams inputCopyParams;
        inputCopyParams.blockCount =
            static_cast<uint16_t>(currentRows);
        inputCopyParams.blockLen = width * sizeof(half);
        inputCopyParams.srcStride = 0;
        inputCopyParams.dstStride = 0;

        AscendC::DataCopyPadExtParams<half> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint8_t>(
            (width + 15U) / 16U * 16U - width);
        padParams.paddingValue = static_cast<half>(0);
        AscendC::DataCopyPad(
            inputLocal,
            inputGm[(firstRow_ + rowOffset) * width],
            inputCopyParams,
            padParams);
        inputQueue_.EnQue(inputLocal);

        inputLocal = inputQueue_.DeQue<half>();
        AscendC::LocalTensor<half> outputLocal =
            outputQueue_.AllocTensor<half>();
        const uint32_t alignedWidth =
            (width + 15U) / 16U * 16U;
        VectorCopy(
            outputLocal,
            inputLocal,
            currentRows * alignedWidth);
        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);

        AscendC::DataCopyExtParams outputCopyParams;
        outputCopyParams.blockCount =
            static_cast<uint16_t>(currentRows);
        outputCopyParams.blockLen = width * sizeof(half);
        outputCopyParams.srcStride = 0;
        outputCopyParams.dstStride =
            (outInner_ - width) * sizeof(half);
        outputLocal = outputQueue_.DeQue<half>();
        AscendC::DataCopyPad(
            yGm_[
                (firstRow_ + rowOffset) * outInner_ +
                outputOffset],
            outputLocal,
            outputCopyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline void VectorCopy(
        const AscendC::LocalTensor<half>& dst,
        const AscendC::LocalTensor<half>& src,
        uint32_t count)
    {
        constexpr uint32_t ELEMENTS_PER_REPEAT = 128;
        constexpr uint32_t MAX_REPEATS = 255;
        AscendC::CopyRepeatParams repeatParams(1, 1, 8, 8);
        uint32_t offset = 0;
        while (count - offset >= ELEMENTS_PER_REPEAT) {
            const uint32_t remainingRepeats =
                (count - offset) / ELEMENTS_PER_REPEAT;
            const uint8_t repeats = static_cast<uint8_t>(
                remainingRepeats > MAX_REPEATS
                    ? MAX_REPEATS
                    : remainingRepeats);
            AscendC::Copy(
                dst[offset],
                src[offset],
                static_cast<uint64_t>(ELEMENTS_PER_REPEAT),
                repeats,
                repeatParams);
            offset +=
                static_cast<uint32_t>(repeats) *
                ELEMENTS_PER_REPEAT;
        }
        if (offset < count) {
            AscendC::Copy(
                dst[offset],
                src[offset],
                static_cast<uint64_t>(count - offset),
                static_cast<uint8_t>(1),
                repeatParams);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    AscendC::GlobalTensor<half> yGm_;
    GM_ADDR inputAddresses_[16];
    uint32_t outer_;
    uint32_t outInner_;
    uint32_t inputCount_;
    uint32_t tileRows_;
    uint32_t maxAlignedWidth_;
    uint32_t rows_;
    uint32_t firstRow_;
    uint32_t widths_[32];
    uint32_t offsets_[32];
};

extern "C" __global__ __aicore__ void concat_fast(
    GM_ADDR x0,
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR x3,
    GM_ADDR x4,
    GM_ADDR x5,
    GM_ADDR x6,
    GM_ADDR x7,
    GM_ADDR x8,
    GM_ADDR x9,
    GM_ADDR x10,
    GM_ADDR x11,
    GM_ADDR x12,
    GM_ADDR x13,
    GM_ADDR x14,
    GM_ADDR x15,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelConcatFast op;
    op.Init(
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        x11,
        x12,
        x13,
        x14,
        x15,
        y,
        tilingData.outer,
        tilingData.outInner,
        tilingData.inputCount,
        tilingData.baseRowsPerBlock,
        tilingData.extraBlocks,
        tilingData.tileRows,
        tilingData.maxAlignedWidth,
        tilingData.widths,
        tilingData.offsets,
        AscendC::GetBlockIdx());
    op.Process();
}
