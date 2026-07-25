#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALGIN = 32;
// constexpr int32_t SIZE = 3968;

class KernelGatherV3 {
public:
    __aicore__ inline KernelGatherV3() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR out, uint32_t dimension, uint32_t dataType, uint32_t dataSize, uint32_t batchDims)
    {
        this->dimension = dimension;
        this->dataType = dataType;
        this->dataSize = dataSize;
        this->batchDims = batchDims;
        this->SIZE = 1984;
        axisGm.SetGlobalBuffer((__gm__ DTYPE_AXIS *)axis, 1);
        this->axis = axisGm(0);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, dataSize);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES *)indices, dataSize);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT *)out, dataSize);

        pipe.InitBuffer(temp1Float, 8 * sizeof(float)); 
        pipe.InitBuffer(temp2Float, 8 * sizeof(float));

        pipe.InitBuffer(tempBits, this->SIZE * sizeof(uint8_t));

        pipe.InitBuffer(accumulateQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));
        pipe.InitBuffer(tempQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));
        pipe.InitBuffer(maxQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));

        pipe.InitBuffer(tempQueueHalf, BUFFER_NUM, this->SIZE * sizeof(half));

        pipe.InitBuffer(tempQueueInt32, BUFFER_NUM, this->SIZE * sizeof(int32_t));
        pipe.InitBuffer(tempQueueInt8, BUFFER_NUM, this->SIZE * sizeof(int8_t));
        pipe.InitBuffer(tempQueueInt32T, BUFFER_NUM, this->SIZE * sizeof(int32_t));

        int32_t typeSize = 2;
        if (dataType == 0){
            typeSize = 4;
        }
        int32_t elementsPerBlock = 32 / typeSize;
        int32_t elementsPerRepeat = 256 / typeSize;
        int32_t firstMaxRepeat = this->SIZE / elementsPerRepeat;
        int32_t finalWorkLocalNeedSize =  (firstMaxRepeat + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        pipe.InitBuffer(workQueue, BUFFER_NUM, finalWorkLocalNeedSize * sizeof(float));

    }
    __aicore__ inline void Process(uint32_t inputXShape[5]){
        if (this->dataType == 1){
            uint32_t all_size = this->dataSize;
            uint32_t loop = all_size / this->SIZE;
            uint32_t remain = all_size % this->SIZE;
            // if (remain) ++loop;
            // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
            int32_t index_x = 0;
            for (int i = 0; i < loop; ++i){
                LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                AscendC::DataCopy(tempLocal, indicesGm[i * this->SIZE], this->SIZE);
                tempQueueInt32.EnQue(tempLocal);
                tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                for (int j = 0; j < this->SIZE; ++j){
                    int32_t index = tempLocal.GetValue(j);
                    outGm(index_x) = static_cast<DTYPE_OUT>(xGm.GetValue(index));
                    index_x++;
                }
                tempQueueInt32.FreeTensor(tempLocal);
            }
            // AscendC::printf("remain = %d\n", remain);
            if (remain){
                LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                AscendC::DataCopy(tempLocal, indicesGm[loop * this->SIZE], this->SIZE);
                tempQueueInt32.EnQue(tempLocal);
                tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                for (int i = 0; i < remain; ++i){
                    int32_t index = tempLocal.GetValue(i);
                    outGm(index_x) = static_cast<DTYPE_OUT>(xGm(index));
                    index_x++;
                }
                tempQueueInt32.FreeTensor(tempLocal);
            }
            // tempQueueInt32.FreeTensor(tempLocal);
        }
        else if (this->dataType == 0 && this->dimension == 2){
            if (this->axis == 0){
                uint32_t all_size = this->dataSize;
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                // if (remain) ++loop;
                // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                // LocalTensor<DTYPE_X> tempLocalFloat = tempQueueFloat.AllocTensor<DTYPE_X>();
                int32_t index_x = 0;
                for (int i = 0; i < loop; ++i){
                    LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                    AscendC::DataCopy(tempLocal, indicesGm[i * this->SIZE], this->SIZE);
                    tempQueueInt32.EnQue(tempLocal);
                    tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                    for (int j = 0; j < this->SIZE; ++j){
                        int32_t index = tempLocal.GetValue(j);
                        // 第index行的拷贝,每次拷贝SIZE大小
                        uint32_t all_size_cp = inputXShape[1];
                        uint32_t loop_cp = all_size_cp / this->SIZE;
                        if (all_size_cp % this->SIZE) ++loop_cp;
                        uint32_t start = index * all_size_cp;
                        uint32_t start_x = index_x * all_size_cp;
                        LocalTensor<DTYPE_X> tempLocalFloat = tempQueueFloat.AllocTensor<DTYPE_X>();
                        for (int k = 0; k < loop_cp; ++k){
                            AscendC::DataCopy(tempLocalFloat, xGm[start + k * this->SIZE], this->SIZE);
                            tempQueueFloat.EnQue(tempLocalFloat);
                            tempLocalFloat = tempQueueFloat.DeQue<DTYPE_X>();
                            tempLocalFloat.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                            AscendC::DataCopy(outGm[start_x + k * this->SIZE], tempLocalFloat, this->SIZE);
                        }
                        tempQueueFloat.FreeTensor(tempLocalFloat);
                        index_x++;
                    }
                    tempQueueInt32.FreeTensor(tempLocal);
                }
                // AscendC::printf("remain = %d\n", remain);
                if (remain){
                    LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                    AscendC::DataCopy(tempLocal, indicesGm[loop * this->SIZE], this->SIZE);
                    tempQueueInt32.EnQue(tempLocal);
                    tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                    for (int j = 0; j < remain; ++j){
                        int32_t index = tempLocal.GetValue(j);
                        // AscendC::printf("index = %d, index_x = %d\n", index, index_x);
                        // 第index行的拷贝,每次拷贝SIZE大小
                        uint32_t all_size_cp = inputXShape[1];
                        uint32_t loop_cp = all_size_cp / this->SIZE;
                        if (all_size_cp % this->SIZE) ++loop_cp;
                        uint32_t start = index * all_size_cp;
                        uint32_t start_x = index_x * all_size_cp;
                        LocalTensor<DTYPE_X> tempLocalFloat = tempQueueFloat.AllocTensor<DTYPE_X>();
                        for (int k = 0; k < loop_cp; ++k){
                            AscendC::DataCopy(tempLocalFloat, xGm[start + k * this->SIZE], this->SIZE);
                            tempQueueFloat.EnQue(tempLocalFloat);
                            tempLocalFloat = tempQueueFloat.DeQue<DTYPE_X>();
                            tempLocalFloat.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                            AscendC::DataCopy(outGm[start_x + k * this->SIZE], tempLocalFloat, this->SIZE);
                        }
                        tempQueueFloat.FreeTensor(tempLocalFloat);
                        index_x++;
                    }
                    tempQueueInt32.FreeTensor(tempLocal);
                }
                // tempQueueInt32.FreeTensor(tempLocal);
                // tempQueueFloat.FreeTensor(tempLocalFloat);
            }else{
                uint32_t all_size = this->dataSize;
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                // if (remain) ++loop;
                LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                int32_t index_x = 0;
                for (int i = 0; i < loop; ++i){
                    // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                    AscendC::DataCopy(tempLocal, indicesGm[i * this->SIZE], this->SIZE);
                    tempQueueInt32.EnQue(tempLocal);
                    tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                    for (int j = 0; j < this->SIZE; ++j){
                        int32_t index = tempLocal.GetValue(j);
                        // 一个一个写写入第index列
                        for (int k = 0; k < inputXShape[0]; ++k){
                            outGm(index_x + k * this->dataSize) = static_cast<DTYPE_OUT>(xGm.GetValue(index + k * inputXShape[1]));
                        }
                        index_x++;
                    }
                    // tempQueueInt32.FreeTensor(tempLocal);
                }
                // AscendC::printf("remain = %d\n", remain);
                if (remain){
                    // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                    AscendC::DataCopy(tempLocal, indicesGm[loop * this->SIZE], this->SIZE);
                    tempQueueInt32.EnQue(tempLocal);
                    tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                    for (int i = 0; i < remain; ++i){
                        int32_t index = tempLocal.GetValue(i);
                        for (int k = 0; k < inputXShape[0]; ++k){
                            outGm(index_x + k * this->dataSize) = static_cast<DTYPE_OUT>(xGm(index + k * inputXShape[1]));
                        }
                        index_x++;
                    }
                    // tempQueueInt32.FreeTensor(tempLocal);
                }
                tempQueueInt32.FreeTensor(tempLocal);
            }
        }
        else if (this->dataType == 2){
            if (this->axis == 0){
                uint32_t all_size = this->dataSize;
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                // if (remain) ++loop;
                // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                // LocalTensor<DTYPE_X> tempLocalFloat = tempQueueFloat.AllocTensor<DTYPE_X>();
                int32_t index_x = 0;
                for (int i = 0; i < loop; ++i){
                    LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                    AscendC::DataCopy(tempLocal, indicesGm[i * this->SIZE], this->SIZE);
                    tempQueueInt32.EnQue(tempLocal);
                    tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                    for (int j = 0; j < this->SIZE; ++j){
                        int32_t index = tempLocal.GetValue(j);
                        // 第index行的拷贝,每次拷贝SIZE大小
                        uint32_t all_size_cp = inputXShape[1] * inputXShape[2];
                        uint32_t loop_cp = all_size_cp / this->SIZE;
                        if (all_size_cp % this->SIZE) ++loop_cp;
                        uint32_t start = index * all_size_cp;
                        uint32_t start_x = index_x * all_size_cp;
                        LocalTensor<DTYPE_X> tempLocalInt8 = tempQueueInt8.AllocTensor<DTYPE_X>();
                        for (int k = 0; k < loop_cp; ++k){
                            AscendC::DataCopy(tempLocalInt8, xGm[start + k * this->SIZE], this->SIZE);
                            tempQueueInt8.EnQue(tempLocalInt8);
                            tempLocalInt8 = tempQueueInt8.DeQue<DTYPE_X>();
                            tempLocalInt8.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                            AscendC::DataCopy(outGm[start_x + k * this->SIZE], tempLocalInt8, this->SIZE);
                        }
                        tempQueueInt8.FreeTensor(tempLocalInt8);
                        index_x++;
                    }
                    tempQueueInt32.FreeTensor(tempLocal);
                }
                // AscendC::printf("remain = %d\n", remain);
                if (remain){
                    LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                    AscendC::DataCopy(tempLocal, indicesGm[loop * this->SIZE], this->SIZE);
                    tempQueueInt32.EnQue(tempLocal);
                    tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                    for (int j = 0; j < remain; ++j){
                        int32_t index = tempLocal.GetValue(j);
                        // AscendC::printf("index = %d, index_x = %d\n", index, index_x);
                        // 第index行的拷贝,每次拷贝SIZE大小
                        uint32_t all_size_cp = inputXShape[1] * inputXShape[2];
                        uint32_t loop_cp = all_size_cp / this->SIZE;
                        if (all_size_cp % this->SIZE) ++loop_cp;
                        uint32_t start = index * all_size_cp;
                        uint32_t start_x = index_x * all_size_cp;
                        LocalTensor<DTYPE_X> tempLocalInt8 = tempQueueInt8.AllocTensor<DTYPE_X>();
                        for (int k = 0; k < loop_cp; ++k){
                            AscendC::DataCopy(tempLocalInt8, xGm[start + k * this->SIZE], this->SIZE);
                            tempQueueInt8.EnQue(tempLocalInt8);
                            tempLocalInt8 = tempQueueInt8.DeQue<DTYPE_X>();
                            tempLocalInt8.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                            AscendC::DataCopy(outGm[start_x + k * this->SIZE], tempLocalInt8, this->SIZE);
                        }
                        tempQueueInt8.FreeTensor(tempLocalInt8);
                        index_x++;
                    }
                    tempQueueInt32.FreeTensor(tempLocal);
                }
            }else if (this->axis == 1){
                uint32_t all_size = this->dataSize;
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                // if (remain) ++loop;
                // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                // LocalTensor<DTYPE_X> tempLocalFloat = tempQueueFloat.AllocTensor<DTYPE_X>();
                int32_t index_x = 0;
                for (int t = 0; t < inputXShape[0]; ++t){
                    for (int i = 0; i < loop; ++i){
                        LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                        AscendC::DataCopy(tempLocal, indicesGm[i * this->SIZE], this->SIZE);
                        tempQueueInt32.EnQue(tempLocal);
                        tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                        for (int j = 0; j < this->SIZE; ++j){
                            int32_t index = tempLocal.GetValue(j);
                            // 第index行的拷贝,每次拷贝SIZE大小
                            uint32_t all_size_cp = inputXShape[2];
                            uint32_t loop_cp = all_size_cp / this->SIZE;
                            if (all_size_cp % this->SIZE) ++loop_cp;
                            uint32_t start = index * all_size_cp;
                            uint32_t start_x = index_x * all_size_cp;
                            LocalTensor<DTYPE_X> tempLocalInt8 = tempQueueInt8.AllocTensor<DTYPE_X>();
                            for (int k = 0; k < loop_cp; ++k){
                                AscendC::DataCopy(tempLocalInt8, xGm[t * inputXShape[1] * inputXShape[2] + start + k * this->SIZE], this->SIZE);
                                tempQueueInt8.EnQue(tempLocalInt8);
                                tempLocalInt8 = tempQueueInt8.DeQue<DTYPE_X>();
                                tempLocalInt8.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                AscendC::DataCopy(outGm[start_x + k * this->SIZE], tempLocalInt8, this->SIZE);
                            }
                            tempQueueInt8.FreeTensor(tempLocalInt8);
                            index_x++;
                        }
                        tempQueueInt32.FreeTensor(tempLocal);
                    }
                    // AscendC::printf("remain = %d\n", remain);
                    if (remain){
                        LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                        AscendC::DataCopy(tempLocal, indicesGm[loop * this->SIZE], this->SIZE);
                        tempQueueInt32.EnQue(tempLocal);
                        tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                        for (int j = 0; j < remain; ++j){
                            int32_t index = tempLocal.GetValue(j);
                            // AscendC::printf("index = %d, index_x = %d\n", index, index_x);
                            // 第index行的拷贝,每次拷贝SIZE大小
                            uint32_t all_size_cp = inputXShape[2];
                            uint32_t loop_cp = all_size_cp / this->SIZE;
                            if (all_size_cp % this->SIZE) ++loop_cp;
                            uint32_t start = index * all_size_cp;
                            uint32_t start_x = index_x * all_size_cp;
                            LocalTensor<DTYPE_X> tempLocalInt8 = tempQueueInt8.AllocTensor<DTYPE_X>();
                            for (int k = 0; k < loop_cp; ++k){
                                AscendC::DataCopy(tempLocalInt8, xGm[t * inputXShape[1] * inputXShape[2] + start + k * this->SIZE], this->SIZE);
                                tempQueueInt8.EnQue(tempLocalInt8);
                                tempLocalInt8 = tempQueueInt8.DeQue<DTYPE_X>();
                                tempLocalInt8.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                AscendC::DataCopy(outGm[start_x + k * this->SIZE], tempLocalInt8, this->SIZE);
                            }
                            tempQueueInt8.FreeTensor(tempLocalInt8);
                            index_x++;
                        }
                        tempQueueInt32.FreeTensor(tempLocal);
                    }
                }
            }else{
                uint32_t all_size = this->dataSize;
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                int32_t index_x = 0;
                for (int k = 0; k < inputXShape[0] * inputXShape[1]; ++k){
                    for (int i = 0; i < loop; ++i){
                        LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                        AscendC::DataCopy(tempLocal, indicesGm[i * this->SIZE], this->SIZE);
                        tempQueueInt32.EnQue(tempLocal);
                        tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                        for (int j = 0; j < this->SIZE; ++j){
                            int32_t index = tempLocal.GetValue(j);
                            outGm(index_x) = static_cast<DTYPE_OUT>(xGm(index + k * inputXShape[2]));
                            index_x++;
                        }
                        tempQueueInt32.FreeTensor(tempLocal);
                    }
                    // AscendC::printf("remain = %d\n", remain);
                    if (remain){
                        LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                        AscendC::DataCopy(tempLocal, indicesGm[loop * this->SIZE], this->SIZE);
                        tempQueueInt32.EnQue(tempLocal);
                        tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                        for (int j = 0; j < remain; ++j){
                            int32_t index = tempLocal.GetValue(j);
                            // AscendC::printf("index = %d, index_x = %d\n", index, index_x);
                            // 第index行的拷贝,每次拷贝SIZE大小
                            outGm(index_x) = static_cast<DTYPE_OUT>(xGm(index + k * inputXShape[2]));
                            index_x++;
                        }
                        tempQueueInt32.FreeTensor(tempLocal);
                    }
                }
            }
        }
        else if (this->dataType == 3){
            if (this->axis == 0){
                uint32_t all_size = this->dataSize;
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                // if (remain) ++loop;
                LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                int32_t index_x = 0;

                uint32_t all_size_cp = inputXShape[1] * inputXShape[2] * inputXShape[3];
                uint32_t loop_cp = all_size_cp / this->SIZE;
                if (all_size_cp % this->SIZE) ++loop_cp;
                for (int i = 0; i < loop; ++i){
                    // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                    AscendC::DataCopy(tempLocal, indicesGm[i * this->SIZE], this->SIZE);
                    tempQueueInt32.EnQue(tempLocal);
                    tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                    for (int j = 0; j < this->SIZE; ++j){
                        int32_t index = tempLocal.GetValue(j);
                        // 第index行的拷贝,每次拷贝SIZE大小
                        // uint32_t all_size_cp = inputXShape[1] * inputXShape[2] * inputXShape[3];
                        // uint32_t loop_cp = all_size_cp / this->SIZE;
                        // if (all_size_cp % this->SIZE) ++loop_cp;
                        uint32_t start = index * all_size_cp;
                        uint32_t start_x = index_x * all_size_cp;
                        // LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                        for (int k = 0; k < loop_cp; ++k){
                            AscendC::DataCopy(tempLocalInt32, xGm[start + k * this->SIZE], this->SIZE);
                            tempQueueInt32T.EnQue(tempLocalInt32);
                            tempLocalInt32 = tempQueueInt32T.DeQue<DTYPE_X>();
                            tempLocalInt32.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                            AscendC::DataCopy(outGm[start_x + k * this->SIZE], tempLocalInt32, this->SIZE);
                        }
                        // tempQueueInt32T.FreeTensor(tempLocalInt32);
                        index_x++;
                    }
                    // tempQueueInt32.FreeTensor(tempLocal);
                }
                // AscendC::printf("remain = %d\n", remain);
                if (remain){
                    // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                    AscendC::DataCopy(tempLocal, indicesGm[loop * this->SIZE], this->SIZE);
                    tempQueueInt32.EnQue(tempLocal);
                    tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                    for (int j = 0; j < remain; ++j){
                        int32_t index = tempLocal.GetValue(j);
                        // AscendC::printf("index = %d, index_x = %d\n", index, index_x);
                        // 第index行的拷贝,每次拷贝SIZE大小
                        // uint32_t all_size_cp = inputXShape[1] * inputXShape[2] * inputXShape[3];
                        // uint32_t loop_cp = all_size_cp / this->SIZE;
                        // if (all_size_cp % this->SIZE) ++loop_cp;
                        uint32_t start = index * all_size_cp;
                        uint32_t start_x = index_x * all_size_cp;
                        // LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                        for (int k = 0; k < loop_cp; ++k){
                            AscendC::DataCopy(tempLocalInt32, xGm[start + k * this->SIZE], this->SIZE);
                            tempQueueInt32T.EnQue(tempLocalInt32);
                            tempLocalInt32 = tempQueueInt32T.DeQue<DTYPE_X>();
                            tempLocalInt32.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                            AscendC::DataCopy(outGm[start_x + k * this->SIZE], tempLocalInt32, this->SIZE);
                        }
                        // tempQueueInt32T.FreeTensor(tempLocalInt32);
                        index_x++;
                    }
                    // tempQueueInt32.FreeTensor(tempLocal);
                }
                tempQueueInt32.FreeTensor(tempLocal);
                tempQueueInt32T.FreeTensor(tempLocalInt32);
            }else if (this->axis == 1){
                uint32_t all_size = this->dataSize;
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                // if (remain) ++loop;
                LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                int32_t index_x = 0;

                uint32_t all_size_cp = inputXShape[2] * inputXShape[3];
                uint32_t loop_cp = all_size_cp / this->SIZE;
                if (all_size_cp % this->SIZE) ++loop_cp;
                for (int t = 0; t < inputXShape[0]; ++t){
                    for (int i = 0; i < loop; ++i){
                        // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                        AscendC::DataCopy(tempLocal, indicesGm[i * this->SIZE], this->SIZE);
                        tempQueueInt32.EnQue(tempLocal);
                        tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                        for (int j = 0; j < this->SIZE; ++j){
                            int32_t index = tempLocal.GetValue(j);
                            // 第index行的拷贝,每次拷贝SIZE大小
                            uint32_t start = index * all_size_cp;
                            uint32_t start_x = index_x * all_size_cp;
                            // LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                            for (int k = 0; k < loop_cp; ++k){
                                AscendC::DataCopy(tempLocalInt32, xGm[t * inputXShape[1] * inputXShape[2] * inputXShape[3] + start + k * this->SIZE], this->SIZE);
                                tempQueueInt32T.EnQue(tempLocalInt32);
                                tempLocalInt32 = tempQueueInt32T.DeQue<DTYPE_X>();
                                tempLocalInt32.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                AscendC::DataCopy(outGm[start_x + k * this->SIZE], tempLocalInt32, this->SIZE);
                            }
                            // tempQueueInt32T.FreeTensor(tempLocalInt32);
                            index_x++;
                        }
                        // tempQueueInt32.FreeTensor(tempLocal);
                    }
                    // AscendC::printf("remain = %d\n", remain);
                    if (remain){
                        // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                        AscendC::DataCopy(tempLocal, indicesGm[loop * this->SIZE], this->SIZE);
                        tempQueueInt32.EnQue(tempLocal);
                        tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                        for (int j = 0; j < remain; ++j){
                            int32_t index = tempLocal.GetValue(j);
                            // AscendC::printf("index = %d, index_x = %d\n", index, index_x);
                            // 第index行的拷贝,每次拷贝SIZE大小
                            // uint32_t all_size_cp = inputXShape[2] * inputXShape[3];
                            // uint32_t loop_cp = all_size_cp / this->SIZE;
                            // if (all_size_cp % this->SIZE) ++loop_cp;
                            uint32_t start = index * all_size_cp;
                            uint32_t start_x = index_x * all_size_cp;
                            // LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                            for (int k = 0; k < loop_cp; ++k){
                                AscendC::DataCopy(tempLocalInt32, xGm[t * inputXShape[1] * inputXShape[2] * inputXShape[3] + start + k * this->SIZE], this->SIZE);
                                tempQueueInt32T.EnQue(tempLocalInt32);
                                tempLocalInt32 = tempQueueInt32T.DeQue<DTYPE_X>();
                                tempLocalInt32.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                AscendC::DataCopy(outGm[start_x + k * this->SIZE], tempLocalInt32, this->SIZE);
                            }
                            // tempQueueInt32T.FreeTensor(tempLocalInt32);
                            index_x++;
                        }
                        // tempQueueInt32.FreeTensor(tempLocal);
                    }
                }
                tempQueueInt32T.FreeTensor(tempLocalInt32);
                tempQueueInt32.FreeTensor(tempLocal);
            }
            else if (this->axis == 2){
                // if (this->batchDims == 1){
                //     while(true) AscendC::printf("test\n");
                // }
                uint32_t all_size = this->dataSize / inputXShape[0];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;

                int32_t index_x = 0;

                uint32_t all_size_cp = inputXShape[3];
                uint32_t loop_cp = all_size_cp / this->SIZE;
                uint32_t remain_cp = all_size_cp % this->SIZE;
                // if (remain_cp % 32 != 0) remain_cp = remain_cp / 32 * 32;
                // uint32_t LOOP = remain_cp / ALGIN;
                // uint32_t REMAIN = remain_cp % ALGIN;
                for (int batch = 0; batch < inputXShape[0]; ++batch){
                    // if (remain) ++loop;
                    // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                    // LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                    
                    for (int t = 0; t < inputXShape[1]; ++t){
                        for (int i = 0; i < loop; ++i){
                            LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                            AscendC::DataCopy(tempLocal, indicesGm[batch * all_size + i * this->SIZE], this->SIZE);
                            tempQueueInt32.EnQue(tempLocal);
                            tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                            for (int j = 0; j < this->SIZE; ++j){
                                int32_t index = tempLocal.GetValue(j);
                                // 第index行的拷贝,每次拷贝SIZE大小
                                // uint32_t all_size_cp = inputXShape[3];
                                // uint32_t loop_cp = all_size_cp / this->SIZE;
                                // if (all_size_cp % this->SIZE) ++loop_cp;
                                // uint32_t start = index * all_size_cp;
                                // uint32_t start_x = index_x * all_size_cp;
                                // LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                                // for (int t = 0; t < inputXShape[0] * inputXShape[1]; ++t){
                                for (int k = 0; k < loop_cp; ++k){
                                    LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                                    AscendC::DataCopy(tempLocalInt32, xGm[batch * inputXShape[1] * inputXShape[2] * inputXShape[3] + t * inputXShape[2] * inputXShape[3] + index * all_size_cp + k * this->SIZE], this->SIZE);
                                    tempQueueInt32T.EnQue(tempLocalInt32);
                                    tempLocalInt32 = tempQueueInt32T.DeQue<DTYPE_X>();
                                    tempLocalInt32.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                    AscendC::DataCopy(outGm[index_x * all_size_cp + k * this->SIZE], tempLocalInt32, this->SIZE);
                                    tempQueueInt32T.FreeTensor(tempLocalInt32);
                                }
                                if (remain_cp){
                                    // if (LOOP){
                                        LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                                        AscendC::DataCopy(tempLocalInt32, xGm[batch * inputXShape[1] * inputXShape[2] * inputXShape[3] + t * inputXShape[2] * inputXShape[3] + index * all_size_cp + loop_cp * this->SIZE], this->SIZE);
                                        tempQueueInt32T.EnQue(tempLocalInt32);
                                        tempLocalInt32 = tempQueueInt32T.DeQue<DTYPE_X>();
                                        tempLocalInt32.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                        AscendC::DataCopy(outGm[index_x * all_size_cp + loop_cp * this->SIZE], tempLocalInt32, this->SIZE);
                                        tempQueueInt32T.FreeTensor(tempLocalInt32);
                                    // }
                                    // if (REMAIN){
                                    //     for (int k = 0; k < REMAIN; ++k){
                                    //         outGm(index_x * all_size_cp + t * this->dataSize * inputXShape[3] + loop_cp * this->SIZE +ALGIN* LOOP + k) = 
                                    //         static_cast<DTYPE_OUT>(xGm(t * inputXShape[2] * inputXShape[3] + index * all_size_cp + loop_cp * this->SIZE +ALGIN* LOOP + k));
                                    //     }
                                    // }
                                }
                                // }
                                // tempQueueInt32T.FreeTensor(tempLocalInt32);
                                index_x++;
                            }
                            tempQueueInt32.FreeTensor(tempLocal);
                        }
                        // AscendC::printf("remain = %d\n", remain);
                        if (remain){
                            LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                            AscendC::DataCopy(tempLocal, indicesGm[batch * all_size + loop * this->SIZE], this->SIZE);
                            tempQueueInt32.EnQue(tempLocal);
                            tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                            for (int j = 0; j < remain; ++j){
                                int32_t index = tempLocal.GetValue(j);
                                // AscendC::printf("index = %d, index_x = %d\n", index, index_x);
                                // 第index行的拷贝,每次拷贝SIZE大小
                                // uint32_t all_size_cp = inputXShape[3];
                                // uint32_t loop_cp = all_size_cp / this->SIZE;
                                // if (all_size_cp % this->SIZE) ++loop_cp;
                                // uint32_t start = index * all_size_cp;
                                // uint32_t start_x = index_x * all_size_cp;
                                // LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                                // for (int t = 0; t < inputXShape[0] * inputXShape[1]; ++t){
                                for (int k = 0; k < loop_cp; ++k){
                                    LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                                    AscendC::DataCopy(tempLocalInt32, xGm[batch * inputXShape[1] * inputXShape[2] * inputXShape[3] + t * inputXShape[2] * inputXShape[3] + index * all_size_cp + k * this->SIZE], this->SIZE);
                                    tempQueueInt32T.EnQue(tempLocalInt32);
                                    tempLocalInt32 = tempQueueInt32T.DeQue<DTYPE_X>();
                                    tempLocalInt32.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                    AscendC::DataCopy(outGm[index_x * all_size_cp + k * this->SIZE], tempLocalInt32, this->SIZE);
                                    tempQueueInt32T.FreeTensor(tempLocalInt32);
                                }
                                if (remain_cp){
                                    // 拷贝remain_cp个元素，如果是8的倍数就可以使用DataCopy，如果不是就得一个一个的赋值
                                    // if (LOOP){
                                        LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                                        AscendC::DataCopy(tempLocalInt32, xGm[batch * inputXShape[1] * inputXShape[2] * inputXShape[3] + t * inputXShape[2] * inputXShape[3] + index * all_size_cp + loop_cp * this->SIZE], this->SIZE);
                                        tempQueueInt32T.EnQue(tempLocalInt32);
                                        tempLocalInt32 = tempQueueInt32T.DeQue<DTYPE_X>();
                                        tempLocalInt32.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                        // AscendC::DataCopy(outGm[index_x * all_size_cp + t * this->dataSize * all_size_cp], tempLocalInt32, ALGIN* LOOP);
                                        // for (int s = 0; s < ALGIN* LOOP; ++s){
                                        //     int32_t tp = tempLocalInt32.GetValue(s);
                                        //     outGm(index_x * all_size_cp + t * this->dataSize * all_size_cp + s) = static_cast<DTYPE_OUT>(tp);
                                        // }
                                        AscendC::DataCopy(outGm[index_x * all_size_cp + loop_cp * this->SIZE], tempLocalInt32, this->SIZE);
                                        tempQueueInt32T.FreeTensor(tempLocalInt32);
                                    // }
                                    // if (REMAIN){
                                    //     for (int k = 0; k < REMAIN; ++k){
                                            // outGm(index_x * all_size_cp + t * this->dataSize * inputXShape[3] + loop_cp * this->SIZE +ALGIN* LOOP + k) = 
                                            // static_cast<DTYPE_OUT>(xGm(t * inputXShape[2] * inputXShape[3] + index * all_size_cp + loop_cp * this->SIZE +ALGIN* LOOP + k));
                                        // }
                                    // }
                                }
                                // }
                                // tempQueueInt32T.FreeTensor(tempLocalInt32);
                                index_x++;
                            }
                            tempQueueInt32.FreeTensor(tempLocal);
                        }
                    }
                }
                // }
                // tempQueueInt32.FreeTensor(tempLocal);
                // tempQueueInt32T.FreeTensor(tempLocalInt32);
            }
            else{
                uint32_t all_size = this->dataSize;
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                int32_t index_x = 0;
                for (int k = 0; k < inputXShape[0] * inputXShape[1] * inputXShape[2]; ++k){
                    for (int i = 0; i < loop; ++i){
                        LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                        AscendC::DataCopy(tempLocal, indicesGm[i * this->SIZE], this->SIZE);
                        tempQueueInt32.EnQue(tempLocal);
                        tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                        for (int j = 0; j < this->SIZE; ++j){
                            int32_t index = tempLocal.GetValue(j);
                            outGm(index_x) = static_cast<DTYPE_OUT>(xGm(index + k * inputXShape[3]));
                            index_x++;
                        }
                        tempQueueInt32.FreeTensor(tempLocal);
                    }
                    // AscendC::printf("remain = %d\n", remain);
                    if (remain){
                        LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                        AscendC::DataCopy(tempLocal, indicesGm[loop * this->SIZE], this->SIZE);
                        tempQueueInt32.EnQue(tempLocal);
                        tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                        for (int j = 0; j < remain; ++j){
                            int32_t index = tempLocal.GetValue(j);
                            // AscendC::printf("index = %d, index_x = %d\n", index, index_x);
                            // 第index行的拷贝,每次拷贝SIZE大小
                            outGm(index_x) = static_cast<DTYPE_OUT>(xGm(index + k * inputXShape[3]));
                            index_x++;
                        }
                        tempQueueInt32.FreeTensor(tempLocal);
                    }
                }
            }
        }
        else if (this->dataType == 0 && this->dimension == 5){
            // if (this->axis == 3 && this->batchDims == 2){
            //     while(true) AscendC::printf("test\n");
            // }
            return;
            uint32_t all_size = this->dataSize / (inputXShape[0] * inputXShape[1]);
            uint32_t loop = all_size / this->SIZE;
            uint32_t remain = all_size % this->SIZE;

            int32_t index_x = 0;

            uint32_t all_size_cp = inputXShape[4];
            uint32_t loop_cp = all_size_cp / this->SIZE;
            uint32_t remain_cp = all_size_cp % this->SIZE;
            // if (remain_cp % 32 != 0) remain_cp = remain_cp / 32 * 32;
            // uint32_t LOOP = remain_cp / ALGIN;
            // uint32_t REMAIN = remain_cp % ALGIN;
            for (int batch = 0; batch < inputXShape[0] * inputXShape[1]; ++batch){
                // if (remain) ++loop;
                // LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                // LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                
                for (int t = 0; t < inputXShape[2]; ++t){
                    for (int i = 0; i < loop; ++i){
                        LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                        AscendC::DataCopy(tempLocal, indicesGm[batch * all_size + i * this->SIZE], this->SIZE);
                        tempQueueInt32.EnQue(tempLocal);
                        tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                        for (int j = 0; j < this->SIZE; ++j){
                            int32_t index = tempLocal.GetValue(j);
                            // 第index行的拷贝,每次拷贝SIZE大小
                            // uint32_t all_size_cp = inputXShape[3];
                            // uint32_t loop_cp = all_size_cp / this->SIZE;
                            // if (all_size_cp % this->SIZE) ++loop_cp;
                            // uint32_t start = index * all_size_cp;
                            // uint32_t start_x = index_x * all_size_cp;
                            // LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                            // for (int t = 0; t < inputXShape[0] * inputXShape[1]; ++t){
                            for (int k = 0; k < loop_cp; ++k){
                                LocalTensor<DTYPE_X> tempLocalFloat = tempQueueFloat.AllocTensor<DTYPE_X>();
                                AscendC::DataCopy(tempLocalFloat, xGm[batch * inputXShape[2] * inputXShape[3] * inputXShape[4]
                                     + t * inputXShape[3] * inputXShape[4] + index * all_size_cp + k * this->SIZE], this->SIZE);
                                tempQueueFloat.EnQue(tempLocalFloat);
                                tempLocalFloat = tempQueueFloat.DeQue<DTYPE_X>();
                                tempLocalFloat.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                AscendC::DataCopy(outGm[index_x * all_size_cp + k * this->SIZE], tempLocalFloat, this->SIZE);
                                tempQueueFloat.FreeTensor(tempLocalFloat);
                            }
                            if (remain_cp){
                                // if (LOOP){
                                    LocalTensor<DTYPE_X> tempLocalFloat = tempQueueFloat.AllocTensor<DTYPE_X>();
                                    AscendC::DataCopy(tempLocalFloat, xGm[batch * inputXShape[2] * inputXShape[3] * inputXShape[4]
                                        + t * inputXShape[3] * inputXShape[4] + index * all_size_cp + loop_cp * this->SIZE], this->SIZE);
                                    tempQueueFloat.EnQue(tempLocalFloat);
                                    tempLocalFloat = tempQueueFloat.DeQue<DTYPE_X>();
                                    tempLocalFloat.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                    AscendC::DataCopy(outGm[index_x * all_size_cp + loop_cp * this->SIZE], tempLocalFloat, this->SIZE);
                                    tempQueueFloat.FreeTensor(tempLocalFloat);
                                // }
                                // if (REMAIN){
                                //     for (int k = 0; k < REMAIN; ++k){
                                //         outGm(index_x * all_size_cp + t * this->dataSize * inputXShape[3] + loop_cp * this->SIZE +ALGIN* LOOP + k) = 
                                //         static_cast<DTYPE_OUT>(xGm(t * inputXShape[2] * inputXShape[3] + index * all_size_cp + loop_cp * this->SIZE +ALGIN* LOOP + k));
                                //     }
                                // }
                            }
                            // }
                            // tempQueueInt32T.FreeTensor(tempLocalInt32);
                            index_x++;
                        }
                        tempQueueInt32.FreeTensor(tempLocal);
                    }
                    // AscendC::printf("remain = %d\n", remain);
                    if (remain){
                        LocalTensor<DTYPE_INDICES> tempLocal = tempQueueInt32.AllocTensor<DTYPE_INDICES>();
                        AscendC::DataCopy(tempLocal, indicesGm[batch * all_size + loop * this->SIZE], this->SIZE);
                        tempQueueInt32.EnQue(tempLocal);
                        tempLocal = tempQueueInt32.DeQue<DTYPE_INDICES>();

                        for (int j = 0; j < remain; ++j){
                            int32_t index = tempLocal.GetValue(j);
                            // AscendC::printf("index = %d, index_x = %d\n", index, index_x);
                            // 第index行的拷贝,每次拷贝SIZE大小
                            // uint32_t all_size_cp = inputXShape[3];
                            // uint32_t loop_cp = all_size_cp / this->SIZE;
                            // if (all_size_cp % this->SIZE) ++loop_cp;
                            // uint32_t start = index * all_size_cp;
                            // uint32_t start_x = index_x * all_size_cp;
                            // LocalTensor<DTYPE_X> tempLocalInt32 = tempQueueInt32T.AllocTensor<DTYPE_X>();
                            // for (int t = 0; t < inputXShape[0] * inputXShape[1]; ++t){
                            for (int k = 0; k < loop_cp; ++k){
                                LocalTensor<DTYPE_X> tempLocalFloat = tempQueueFloat.AllocTensor<DTYPE_X>();
                                AscendC::DataCopy(tempLocalFloat, xGm[batch * inputXShape[2] * inputXShape[3] * inputXShape[4]
                                     + t * inputXShape[3] * inputXShape[4] + index * all_size_cp + k * this->SIZE], this->SIZE);
                                tempQueueFloat.EnQue(tempLocalFloat);
                                tempLocalFloat = tempQueueFloat.DeQue<DTYPE_X>();
                                tempLocalFloat.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                AscendC::DataCopy(outGm[index_x * all_size_cp + k * this->SIZE], tempLocalFloat, this->SIZE);
                                tempQueueFloat.FreeTensor(tempLocalFloat);
                            }
                            if (remain_cp){
                                // 拷贝remain_cp个元素，如果是8的倍数就可以使用DataCopy，如果不是就得一个一个的赋值
                                // if (LOOP){
                                    LocalTensor<DTYPE_X> tempLocalFloat = tempQueueFloat.AllocTensor<DTYPE_X>();
                                    AscendC::DataCopy(tempLocalFloat, xGm[batch * inputXShape[2] * inputXShape[3] * inputXShape[4]
                                        + t * inputXShape[3] * inputXShape[4] + index * all_size_cp + loop_cp * this->SIZE], this->SIZE);
                                    tempQueueFloat.EnQue(tempLocalFloat);
                                    tempLocalFloat = tempQueueFloat.DeQue<DTYPE_X>();
                                    tempLocalFloat.GetValue(0); // bug有可能是这个cann版本的，可能是流同步的问题
                                    // AscendC::DataCopy(outGm[index_x * all_size_cp + t * this->dataSize * all_size_cp], tempLocalInt32, ALGIN* LOOP);
                                    // for (int s = 0; s < ALGIN* LOOP; ++s){
                                    //     int32_t tp = tempLocalInt32.GetValue(s);
                                    //     outGm(index_x * all_size_cp + t * this->dataSize * all_size_cp + s) = static_cast<DTYPE_OUT>(tp);
                                    // }
                                    AscendC::DataCopy(outGm[index_x * all_size_cp + loop_cp * this->SIZE], tempLocalFloat, this->SIZE);
                                    tempQueueFloat.FreeTensor(tempLocalFloat);
                                // }
                                // if (REMAIN){
                                //     for (int k = 0; k < REMAIN; ++k){
                                        // outGm(index_x * all_size_cp + t * this->dataSize * inputXShape[3] + loop_cp * this->SIZE +ALGIN* LOOP + k) = 
                                        // static_cast<DTYPE_OUT>(xGm(t * inputXShape[2] * inputXShape[3] + index * all_size_cp + loop_cp * this->SIZE +ALGIN* LOOP + k));
                                    // }
                                // }
                            }
                            // }
                            // tempQueueInt32T.FreeTensor(tempLocalInt32);
                            index_x++;
                        }
                        tempQueueInt32.FreeTensor(tempLocal);
                    }
                }
            }
        }
    }



private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> accumulateQueueFloat, tempQueueFloat, tempQueueInt32T, tempQueueHalf, tempQueueInt32, tempQueueInt8;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue, maxQueueFloat;
    TBuf<QuePosition::VECCALC> temp1Float, temp2Float, tempBits; 
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_AXIS> axisGm;
    GlobalTensor<DTYPE_OUT> outGm;

    uint32_t dataType; // 运行时数据类型
    int32_t axis;
    uint32_t dimension;
    int32_t SIZE;
    uint32_t dataSize;
    uint32_t batchDims;
};

extern "C" __global__ __aicore__ void gather_v3(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelGatherV3 op;
    op.Init(x, indices, axis, out, tiling_data.dimension, tiling_data.dataType, tiling_data.dataSize, tiling_data.batchDims);
    op.Process(tiling_data.inputXShape);
}