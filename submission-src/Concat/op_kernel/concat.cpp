#include "kernel_operator.h"

namespace {
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t MAX_INPUT_COUNT = 2048;
constexpr uint32_t MAX_COPY_ROWS = 4095;
constexpr uint32_t MAX_DMA_STRIDE = 0xFFFFFFFFU;

// 910B gives each vector core 192 KB of UB. Two 64 KB staging halves plus the
// 16 KB pointer table stay clear of that ceiling while letting one burst carry
// many whole rows. Depth beyond two would shrink each transfer instead of
// adding parallelism: the concurrency here comes from having 40 cores in
// flight, not from queueing more requests inside one core.
constexpr uint32_t STAGE_BYTES = 64 * 1024;
constexpr uint32_t POINTER_TABLE_BYTES = MAX_INPUT_COUNT * sizeof(uint64_t);

constexpr uint32_t MODE_ROW_BLOCK = 0;
constexpr uint32_t MODE_ROW_GENERIC = 1;
constexpr uint32_t MODE_FLAT = 2;

__aicore__ inline uint32_t AlignUp32(const uint32_t value)
{
    return (value + ALIGN_BYTES - 1U) / ALIGN_BYTES * ALIGN_BYTES;
}
}  // namespace

// Holds the two staging halves plus the input pointer table. The pointer
// table is fetched with one bulk DataCopy instead of per-input scalar GM
// reads, which are several hundred cycles each and do not pipeline.
class ConcatContext {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalBytes)
    {
        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(y), totalBytes);
        pipe_.InitBuffer(stageQueue_, 2, STAGE_BYTES);
        pipe_.InitBuffer(pointerBuffer_, POINTER_TABLE_BYTES);
        descriptorGm_ = x;
    }

    __aicore__ inline void LoadPointers(uint32_t inputCount)
    {
        AscendC::GlobalTensor<uint64_t> descriptor;
        descriptor.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint64_t*>(descriptorGm_),
            inputCount + 256);
        // The offset word itself is unavoidably a scalar read; everything
        // after it moves as one block.
        const uint64_t pointerOffsetWords =
            descriptor.GetValue(0) / sizeof(uint64_t);

        pointers_ = pointerBuffer_.Get<uint64_t>();
        const uint32_t wantedBytes =
            AlignUp32(inputCount * static_cast<uint32_t>(sizeof(uint64_t)));
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = wantedBytes;
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopyPadExtParams<uint64_t> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(
            pointers_,
            descriptor[pointerOffsetWords],
            params,
            padParams);
        // Scalar reads of the table must wait for the bulk transfer.
        const event_t pointerEvent = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(pointerEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(pointerEvent);
    }

    __aicore__ inline GM_ADDR InputAddress(uint32_t index)
    {
        return reinterpret_cast<GM_ADDR>(pointers_.GetValue(index));
    }

    __aicore__ inline AscendC::TQue<AscendC::TPosition::VECCALC, 2>&
    Stage()
    {
        return stageQueue_;
    }

    __aicore__ inline AscendC::GlobalTensor<uint8_t>& Output()
    {
        return yGm_;
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 2> stageQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pointerBuffer_;
    AscendC::LocalTensor<uint64_t> pointers_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    GM_ADDR descriptorGm_;
};

// Row-oriented copier. A core owns [firstRow, firstRow + rows), which is one
// fully contiguous span of the output, so the aligned path assembles whole
// rows in UB and writes them back with a single large aligned burst.
class RowCopier {
public:
    __aicore__ inline void Init(
        ConcatContext* context,
        uint64_t outRowBytes,
        uint64_t innerBytes,
        uint64_t baseRowsPerBlock,
        uint32_t extraBlocks,
        uint32_t blockIdx)
    {
        context_ = context;
        outRowBytes_ = outRowBytes;
        innerBytes_ = innerBytes;
        rows_ = baseRowsPerBlock + (blockIdx < extraBlocks ? 1U : 0U);
        firstRow_ =
            blockIdx * baseRowsPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
    }

    __aicore__ inline void Assemble(
        uint32_t inputCount,
        const uint32_t* dimExtents)
    {
        if (rows_ == 0 || outRowBytes_ == 0) {
            return;
        }
        // A staged batch must hold at least one whole row for the single
        // write-back to stay contiguous; wider rows go to the generic path.
        // That bound also keeps the UB-side stride inside its field, since
        // any gap is smaller than one staged row.
        if (outRowBytes_ > STAGE_BYTES) {
            Generic(inputCount, dimExtents);
            return;
        }
        const uint32_t rowsPerBatch = static_cast<uint32_t>(
            STAGE_BYTES / outRowBytes_);
        const uint32_t cappedRows =
            rowsPerBatch < MAX_COPY_ROWS ? rowsPerBatch : MAX_COPY_ROWS;

        for (uint64_t rowOffset = 0;
             rowOffset < rows_;
             rowOffset += cappedRows) {
            const uint32_t batchRows = static_cast<uint32_t>(
                rows_ - rowOffset < cappedRows
                    ? rows_ - rowOffset
                    : cappedRows);
            AscendC::LocalTensor<uint8_t> stage =
                context_->Stage().AllocTensor<uint8_t>();

            uint64_t columnBytes = 0;
            for (uint32_t i = 0; i < inputCount; ++i) {
                const uint64_t rowBytes =
                    static_cast<uint64_t>(dimExtents[i]) * innerBytes_;
                if (rowBytes == 0) {
                    continue;
                }
                AscendC::GlobalTensor<uint8_t> inputGm;
                inputGm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ uint8_t*>(
                        context_->InputAddress(i)),
                    (firstRow_ + rowOffset + batchRows) * rowBytes);

                AscendC::DataCopyExtParams params;
                params.blockCount = static_cast<uint16_t>(batchRows);
                params.blockLen = static_cast<uint32_t>(rowBytes);
                // Rows of one input sit back to back in GM, so the gap
                // between consecutive blocks is zero bytes. The UB side
                // skips the columns owned by the other inputs; aligned mode
                // guarantees that remainder is a whole number of 32 B units.
                params.srcStride = 0;
                params.dstStride = static_cast<uint32_t>(
                    (outRowBytes_ - rowBytes) / ALIGN_BYTES);
                AscendC::DataCopyPadExtParams<uint8_t> padParams;
                padParams.isPad = false;
                padParams.leftPadding = 0;
                padParams.rightPadding = 0;
                padParams.paddingValue = 0;
                AscendC::DataCopyPad(
                    stage[columnBytes],
                    inputGm[(firstRow_ + rowOffset) * rowBytes],
                    params,
                    padParams);
                columnBytes += rowBytes;
            }

            context_->Stage().EnQue(stage);
            AscendC::LocalTensor<uint8_t> ready =
                context_->Stage().DeQue<uint8_t>();

            AscendC::DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = static_cast<uint32_t>(
                static_cast<uint64_t>(batchRows) * outRowBytes_);
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            AscendC::DataCopyPad(
                context_->Output()[(firstRow_ + rowOffset) * outRowBytes_],
                ready,
                outParams);
            context_->Stage().FreeTensor(ready);
        }
    }

    // Unaligned rows cannot be packed at outRowBytes_ spacing inside UB,
    // so each input keeps its own staged transfer and write-back.
    __aicore__ inline void Generic(
        uint32_t inputCount,
        const uint32_t* dimExtents)
    {
        if (rows_ == 0 || outRowBytes_ == 0) {
            return;
        }
        uint64_t outputOffset = 0;
        for (uint32_t i = 0; i < inputCount; ++i) {
            const uint64_t rowBytes =
                static_cast<uint64_t>(dimExtents[i]) * innerBytes_;
            if (rowBytes == 0) {
                continue;
            }
            CopyColumn(
                context_->InputAddress(i), rowBytes, outputOffset);
            outputOffset += rowBytes;
        }
    }

private:
    __aicore__ inline void CopyColumn(
        GM_ADDR inputAddress,
        uint64_t rowBytes,
        uint64_t outputOffset)
    {
        AscendC::GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(inputAddress),
            (firstRow_ + rows_) * rowBytes);

        for (uint64_t chunkOffset = 0;
             chunkOffset < rowBytes;
             chunkOffset += STAGE_BYTES) {
            const uint32_t chunkBytes = static_cast<uint32_t>(
                rowBytes - chunkOffset < STAGE_BYTES
                    ? rowBytes - chunkOffset
                    : STAGE_BYTES);
            const uint32_t alignedChunk = AlignUp32(chunkBytes);
            uint32_t rowsPerCopy = STAGE_BYTES / alignedChunk;
            if (rowsPerCopy > MAX_COPY_ROWS) {
                rowsPerCopy = MAX_COPY_ROWS;
            }
            const uint64_t srcGap = rowBytes - chunkBytes;
            const uint64_t dstGap = outRowBytes_ - chunkBytes;
            if (srcGap > MAX_DMA_STRIDE || dstGap > MAX_DMA_STRIDE) {
                rowsPerCopy = 1;
            }
            if (rowsPerCopy == 0) {
                rowsPerCopy = 1;
            }

            for (uint64_t rowOffset = 0;
                 rowOffset < rows_;
                 rowOffset += rowsPerCopy) {
                const uint32_t batchRows = static_cast<uint32_t>(
                    rows_ - rowOffset < rowsPerCopy
                        ? rows_ - rowOffset
                        : rowsPerCopy);
                AscendC::LocalTensor<uint8_t> stage =
                    context_->Stage().AllocTensor<uint8_t>();

                AscendC::DataCopyPadExtParams<uint8_t> padParams;
                padParams.isPad = true;
                padParams.leftPadding = 0;
                padParams.rightPadding = static_cast<uint8_t>(
                    alignedChunk - chunkBytes);
                padParams.paddingValue = 0;

                AscendC::DataCopyExtParams inParams;
                inParams.blockLen = chunkBytes;
                inParams.dstStride = 0;
                if (alignedChunk == chunkBytes || batchRows >= 16) {
                    inParams.blockCount =
                        static_cast<uint16_t>(batchRows);
                    inParams.srcStride = batchRows == 1
                        ? 0
                        : static_cast<uint32_t>(srcGap);
                    AscendC::DataCopyPad(
                        stage,
                        inputGm[
                            (firstRow_ + rowOffset) * rowBytes +
                            chunkOffset],
                        inParams,
                        padParams);
                } else {
                    inParams.blockCount = 1;
                    inParams.srcStride = 0;
                    for (uint32_t row = 0; row < batchRows; ++row) {
                        AscendC::DataCopyPad(
                            stage[row * alignedChunk],
                            inputGm[
                                (firstRow_ + rowOffset + row) * rowBytes +
                                chunkOffset],
                            inParams,
                            padParams);
                    }
                }

                context_->Stage().EnQue(stage);
                AscendC::LocalTensor<uint8_t> ready =
                    context_->Stage().DeQue<uint8_t>();

                AscendC::DataCopyExtParams outParams;
                outParams.blockCount = static_cast<uint16_t>(batchRows);
                outParams.blockLen = chunkBytes;
                outParams.srcStride = 0;
                outParams.dstStride = batchRows == 1
                    ? 0
                    : static_cast<uint32_t>(dstGap);
                AscendC::DataCopyPad(
                    context_->Output()[
                        (firstRow_ + rowOffset) * outRowBytes_ +
                        outputOffset + chunkOffset],
                    ready,
                    outParams);
                context_->Stage().FreeTensor(ready);
            }
        }
    }

    ConcatContext* context_;
    uint64_t outRowBytes_;
    uint64_t innerBytes_;
    uint64_t rows_;
    uint64_t firstRow_;
};

// Flat mode splits the output byte range so a single very wide row still
// spreads across all cores. Work units are 512 B by default so each core
// issues few large transfers rather than many 32 B ones.
class FlatCopier {
public:
    __aicore__ inline void Init(
        ConcatContext* context,
        uint64_t outRowBytes,
        uint64_t innerBytes,
        uint64_t totalBytes,
        uint64_t baseWorkBlocks,
        uint32_t extraWorkBlocks,
        uint32_t workUnitBytes,
        uint32_t blockIdx)
    {
        context_ = context;
        outRowBytes_ = outRowBytes;
        innerBytes_ = innerBytes;
        totalBytes_ = totalBytes;
        const uint64_t blocks =
            baseWorkBlocks + (blockIdx < extraWorkBlocks ? 1U : 0U);
        const uint64_t firstBlock =
            blockIdx * baseWorkBlocks +
            (blockIdx < extraWorkBlocks ? blockIdx : extraWorkBlocks);
        begin_ = firstBlock * workUnitBytes;
        end_ = (firstBlock + blocks) * workUnitBytes;
        if (end_ > totalBytes_) {
            end_ = totalBytes_;
        }
    }

    __aicore__ inline void Run(
        uint32_t inputCount,
        const uint32_t* dimExtents)
    {
        if (begin_ >= end_ || outRowBytes_ == 0) {
            return;
        }
        uint64_t cursor = begin_;
        while (cursor < end_) {
            const uint64_t row = cursor / outRowBytes_;
            const uint64_t rowBase = row * outRowBytes_;
            const uint64_t rowStart = cursor - rowBase;
            uint64_t rowEnd = end_ - rowBase;
            if (rowEnd > outRowBytes_) {
                rowEnd = outRowBytes_;
            }

            uint64_t prefix = 0;
            for (uint32_t i = 0; i < inputCount; ++i) {
                const uint64_t rowBytes =
                    static_cast<uint64_t>(dimExtents[i]) * innerBytes_;
                const uint64_t inputEnd = prefix + rowBytes;
                if (rowBytes != 0 &&
                    rowStart < inputEnd &&
                    rowEnd > prefix) {
                    const uint64_t pieceStart =
                        rowStart > prefix ? rowStart : prefix;
                    const uint64_t pieceEnd =
                        rowEnd < inputEnd ? rowEnd : inputEnd;
                    CopyPiece(
                        context_->InputAddress(i),
                        row * rowBytes + pieceStart - prefix,
                        rowBase + pieceStart,
                        pieceEnd - pieceStart);
                }
                prefix = inputEnd;
                if (prefix >= rowEnd) {
                    break;
                }
            }
            cursor = rowBase + rowEnd;
        }
    }

private:
    __aicore__ inline void CopyPiece(
        GM_ADDR inputAddress,
        uint64_t inputOffset,
        uint64_t outputOffset,
        uint64_t lengthBytes)
    {
        AscendC::GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(inputAddress),
            inputOffset + lengthBytes);
        for (uint64_t offset = 0;
             offset < lengthBytes;
             offset += STAGE_BYTES) {
            const uint32_t chunkBytes = static_cast<uint32_t>(
                lengthBytes - offset < STAGE_BYTES
                    ? lengthBytes - offset
                    : STAGE_BYTES);
            AscendC::LocalTensor<uint8_t> stage =
                context_->Stage().AllocTensor<uint8_t>();

            AscendC::DataCopyExtParams inParams;
            inParams.blockCount = 1;
            inParams.blockLen = chunkBytes;
            inParams.srcStride = 0;
            inParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<uint8_t> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = static_cast<uint8_t>(
                AlignUp32(chunkBytes) - chunkBytes);
            padParams.paddingValue = 0;
            AscendC::DataCopyPad(
                stage,
                inputGm[inputOffset + offset],
                inParams,
                padParams);

            context_->Stage().EnQue(stage);
            AscendC::LocalTensor<uint8_t> ready =
                context_->Stage().DeQue<uint8_t>();

            AscendC::DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = chunkBytes;
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            AscendC::DataCopyPad(
                context_->Output()[outputOffset + offset],
                ready,
                outParams);
            context_->Stage().FreeTensor(ready);
        }
    }

    ConcatContext* context_;
    uint64_t outRowBytes_;
    uint64_t innerBytes_;
    uint64_t totalBytes_;
    uint64_t begin_;
    uint64_t end_;
};

extern "C" __global__ __aicore__ void concat(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.outer == 0 || tilingData.outRowBytes == 0) {
        return;
    }

    ConcatContext context;
    context.Init(x, y, tilingData.totalBytes);
    context.LoadPointers(tilingData.inputCount);
    const uint32_t blockIdx =
        static_cast<uint32_t>(AscendC::GetBlockIdx());

    if (tilingData.mode == MODE_FLAT) {
        FlatCopier copier;
        copier.Init(
            &context,
            tilingData.outRowBytes,
            tilingData.innerBytes,
            tilingData.totalBytes,
            tilingData.baseWorkBlocks,
            tilingData.extraWorkBlocks,
            tilingData.workUnitBytes,
            blockIdx);
        copier.Run(tilingData.inputCount, tilingData.dimExtents);
        return;
    }

    RowCopier copier;
    copier.Init(
        &context,
        tilingData.outRowBytes,
        tilingData.innerBytes,
        tilingData.baseRowsPerBlock,
        tilingData.extraBlocks,
        blockIdx);
    if (tilingData.mode == MODE_ROW_BLOCK) {
        copier.Assemble(tilingData.inputCount, tilingData.dimExtents);
    } else {
        copier.Generic(tilingData.inputCount, tilingData.dimExtents);
    }
}
