#include "kernel_operator.h"

#include <type_traits>



using namespace AscendC;

constexpr int bufferNum=1;

class KernelScatterReduce {
public:
    __aicore__ inline KernelScatterReduce() {}
   __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR src,GM_ADDR y,GM_ADDR count,uint32_t inputSize,uint8_t type,uint16_t* mmInputDims,uint8_t nInputDims,int8_t includeSelf,uint8_t dim )
    {


        inputGm.SetGlobalBuffer((__gm__ DTYPE_SELF *)input, inputSize);
        indexGm.SetGlobalBuffer((__gm__ DTYPE_INDEX *)index, inputSize);
        srcGm.SetGlobalBuffer((__gm__ DTYPE_SRC *)src, inputSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, inputSize);
        countGm.SetGlobalBuffer((__gm__ int *)count, inputSize);


        this->inputSize=inputSize;
        this->type=type;

        this->mmInputDims=mmInputDims;
        this->nInputDims=nInputDims;

        this->includeSelf=includeSelf;
        this->dim=dim;
    }
    __aicore__ inline void Process()
    {

        int initValue=1;
        for(int i=0;i<inputSize;++i){
            yGm.SetValue(i,inputGm.GetValue(i));
        }
        if(includeSelf==0){
            initValue=0;
        }
        for(int i=0;i<inputSize;++i){
            countGm.SetValue(i,initValue);
        }
        int singleDiff=1;
        for(int i=nInputDims-1;i>dim;i--){
            singleDiff*=mmInputDims[i];
        }

        for(int i=0;i<inputSize;++i){
            int inputIdx=i;
            int srcJ=i/singleDiff%mmInputDims[dim];
            int destJ=indexGm.GetValue(inputIdx);
            //    int outputIndex=(i-i%(singleDiff*mmIndexDims[dim]))/mmIndexDims[dim]*mmInputDims[dim]+destK*singleDiff+i%singleDiff;
            int outputIndex=(destJ-srcJ)*singleDiff+inputIdx;
            // //////printf("$$$$$$$$$$$$$$$$$$$$$$$outputIndex:%d srcJ:%d\n",i,srcJ);

            // //////printf("$$$$$$$$$$$$$$$$$$$$$$$inputIdx:%d destJ:%d j:%d\n",inputIdx,destJ,j);
            int count=countGm.GetValue(outputIndex);
            float res;
            float y=static_cast<float>(yGm.GetValue(outputIndex));
            float src=static_cast<float>(srcGm.GetValue(inputIdx));
            if(type==0){//sum
                if(count!=0){
                    res=y+src;
                }else{
                    res=src;
                }
            }else if(type==1){//prod
                if(count!=0){
                    res=y*src;
                }else{
                    res=src;
                }
            }else if(type==2){//mean
                if(count!=0){
                    res=y+(src-y)/(count+1);
                }else{
                    res=src;
                }
            }else if(type==3){//amin
                if(count!=0){
                    res=min(y,src);
                }else{
                    res=src;
                }
            }else if(type==4){//amax
                if(count!=0){
                    res=max(y,src);
                }else{
                    res=src;
                }
            }

            countGm.SetValue(outputIndex,count+1);
            yGm.SetValue(outputIndex,static_cast<DTYPE_Y>(res));
        }
            
    }  




private:
    GlobalTensor<DTYPE_SELF> inputGm;
    GlobalTensor<DTYPE_INDEX> indexGm;
    GlobalTensor<DTYPE_SRC> srcGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<int>countGm;
    TPipe *pipe;


    uint32_t inputSize;
    uint16_t *mmInputDims;
    uint8_t nInputDims;
    uint8_t dim;
    uint8_t type;
    int8_t includeSelf;
};

template<uint8_t type>
class KernelScatterReduce_multiple_core {
    public:
        __aicore__ inline KernelScatterReduce_multiple_core() {}
       
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR src,GM_ADDR y
            ,int inputSize,int J 
            ,int totalTilingNum,int KS
            ,int lineSize,int8_t includeSelf
            ,TPipe* pipeIn
        )
        {

    
            //直接算最开始的偏移，所以这里不需要计算偏移量
            inputGm.SetGlobalBuffer((__gm__ DTYPE_SELF *)input, inputSize);
            indexGm.SetGlobalBuffer((__gm__ DTYPE_INDEX *)index, inputSize);
            srcGm.SetGlobalBuffer((__gm__ DTYPE_SRC *)src, inputSize);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, inputSize);
    
    

            //计算当前核心需要算多少块，以下单位都是块
            int formerNum = totalTilingNum  % GetBlockNum();
            if (GetBlockIdx() < formerNum) {
                this->tilingNum=totalTilingNum/GetBlockNum()+1;
                this->beginIndex=this->tilingNum*GetBlockIdx();
            }else{
                this->tilingNum=totalTilingNum/GetBlockNum();
                this->beginIndex=this->tilingNum*GetBlockIdx()+formerNum;
            }


            this->J=J;
            this->KS=KS;
            this->lineSize=lineSize;
            this->tilingSize=J*lineSize;
            this->includeSelf=includeSelf;
            //一行搬运的32B块数，切片搬运需要用到，所以这里应该算错了，应该用lineSize来算
            this->blockNum=lineSize*sizeof(DTYPE_Y)/32;

            this->pipe = pipeIn;
             
            int spaceSize=tilingSize*sizeof(DTYPE_Y);
            pipe->InitBuffer(QueueInput, bufferNum, spaceSize);
            pipe->InitBuffer(QueueSrc, bufferNum, spaceSize);
            pipe->InitBuffer(QueueY, bufferNum, spaceSize);
    
            pipe->InitBuffer(BufTmpY, lineSize*sizeof(DTYPE_Y));
            pipe->InitBuffer(BufTemp, tilingSize*4);//后续需要copy count
            pipe->InitBuffer(BufIndexK, tilingSize*4);//升序index（以字节为偏移单位）

            //申请临时计算空间
            tmpYLocal=BufTmpY.Get<DTYPE_Y>();

            if constexpr(type==0 || type==1 ||type==2){//sum 0 prod 1 mean 2 
                pipe->InitBuffer(QueueCount, bufferNum, spaceSize);
                pipe->InitBuffer(BufCountY, spaceSize);
                tmpCountLocal=BufCountY.Get<DTYPE_Y>();
            }

            this->tailBlockSize=KS%lineSize;

            //index解析成int类型，以下是以B为单位
            if constexpr(std::is_same<DTYPE_Y, float>::value){
                pipe->InitBuffer(QueueIndex, bufferNum, spaceSize);
            }else{
                pipe->InitBuffer(QueueIndex, bufferNum, 2*spaceSize);
            }

        }
    
    
        
    
        __aicore__ inline void Process()
        {   



            //tiling块下标
            int endIndex=beginIndex+tilingNum;
            int tilingNum_per_line=KS/lineSize;
            int JKS=J*KS;

            indexKLocal=BufIndexK.Get<int32_t>();
            //生成下标，以B为单位
            if constexpr(std::is_same<DTYPE_Y, float>::value){
                int currentIndex=0;
                for(int j=0;j<J;j++){
                    ArithProgression(indexKLocal[currentIndex],0,4,lineSize);
                    currentIndex+=lineSize;
                }
            }else{
                int currentIndex=0;
                for(int j=0;j<J;j++){
                    ArithProgression(indexKLocal[currentIndex],0,2,lineSize);
                    currentIndex+=lineSize;
                }
            }


            for(int i=beginIndex;i<endIndex;i++){    
                // i表示块编号
                // (KS/tilingNum_per_line)表示一行可以分成多少32B块
                // i/每行块数==行数（共HI行），每行J*KS个元素
                // i%每行块数=行内偏移的块号，每块(32/sizeof(DTYPE_Y)个元素
                // TODO:让块长变大
                int gmIndex=i/tilingNum_per_line*JKS+i%tilingNum_per_line*lineSize;
                //////printf("$$$$$$$$$$$$$$$$$$$$4 gmIndex:%d\n",gmIndex);
                //i后面是尾块
                //行内偏移的块号是最后一块
                if(i%tilingNum_per_line==tilingNum_per_line-1 && tailBlockSize!=0){
                    copyIn(gmIndex);
                    compute(true);//非完整计算
                    copyOut(gmIndex);
                    //不能重复计算
                    gmIndex+=tailBlockSize;
                    copyIn(gmIndex);
                    compute(false);
                    copyOut(gmIndex);
                }else{
                    copyIn(gmIndex);
                    compute(false);
                    copyOut(gmIndex);
                }
            }  
        }
        __aicore__ inline void copyIn(int gmIndex){
            AscendC::SliceInfo dstSliceInfo[] = {{0, tilingSize-1, 0, blockNum, tilingSize}};
            AscendC::SliceInfo srcSliceInfo[] = {{0,(uint32_t)((J-1)*KS+lineSize-1),(uint32_t)(KS-lineSize),blockNum,(uint32_t)((J-1)*KS+lineSize)}};
        
            
            uint32_t blockNum_index;
            if constexpr(std::is_same<DTYPE_Y, float>::value){
                blockNum_index=blockNum;
            }else{
                blockNum_index=blockNum*2;//half类型下，index比half大一倍，所以乘2
            }
           
            AscendC::SliceInfo dstSliceInfo_index[] = {{0, tilingSize-1, 0, blockNum_index, tilingSize}};
            //当数据类型是fp16，应该考虑使用两块32B的index？ 其他是以元素为单位，只有blockNum是32B为单位
            AscendC::SliceInfo srcSliceInfo_index[] = {{0,(uint32_t)((J-1)*KS+lineSize-1),(uint32_t)(KS-lineSize),blockNum_index,(uint32_t)((J-1)*KS+lineSize)}};
            
            LocalTensor<DTYPE_Y> inputLocal = QueueInput.AllocTensor<DTYPE_Y>();
            LocalTensor<DTYPE_Y> srcLocal = QueueSrc.AllocTensor<DTYPE_Y>();
            LocalTensor<int32_t> indexLocal = QueueIndex.AllocTensor<int32_t>();
    
            DataCopy(inputLocal, inputGm[gmIndex], dstSliceInfo, srcSliceInfo);
            DataCopy(srcLocal, srcGm[gmIndex], dstSliceInfo, srcSliceInfo);
            DataCopy(indexLocal, indexGm[gmIndex], dstSliceInfo_index, srcSliceInfo_index);




            DTYPE_Y initValue;
            if (includeSelf==0){
                initValue=0;
            }else{
                initValue=1;
            }
            if constexpr(type==0 || type==1 ||type==2){//sum 0 prod 1 mean 2 
                LocalTensor<DTYPE_Y> countLocal = QueueCount.AllocTensor<DTYPE_Y>();  
                //填充count
                Duplicate(countLocal, initValue,tilingSize);
                QueueCount.EnQue(countLocal);    
            }

            
            QueueInput.EnQue(inputLocal);
            QueueIndex.EnQue(indexLocal);
            QueueSrc.EnQue(srcLocal);    
        }
        __aicore__ inline void compute(bool tag){
    
    
            LocalTensor inputLocal=QueueInput.DeQue<DTYPE_Y>();
            LocalTensor indexLocal=QueueIndex.DeQue<int32_t>();
            LocalTensor srcLocal=QueueSrc.DeQue<DTYPE_Y>();
            LocalTensor<DTYPE_Y> yLocal = QueueY.AllocTensor<DTYPE_Y>();
            LocalTensor<DTYPE_Y>  countLocal;
            if constexpr(type==0 || type==1 ||type==2){//sum 0 prod 1 mean 2 
                countLocal=QueueCount.DeQue<DTYPE_Y>();  
            }
            //用于尾块处理
            uint32_t curlineSize;

            if(tag){
                curlineSize=tailBlockSize;
            }else{
                curlineSize=lineSize;
            }


            if (includeSelf==1){
                //为什么需要copy？是为了编程范式中的同步 ylocal初始值：includeself肯定就是inputLocal；反之
                DataCopy(yLocal, inputLocal,tilingSize);
            }else {//需要后处理
                if constexpr(type==0 || type==2){//sum 0 mean 2
                    Duplicate(yLocal, static_cast<DTYPE_Y>(0.0),tilingSize);
                }else if constexpr(type==1){//prod 1
                    Duplicate(yLocal, static_cast<DTYPE_Y>(1.0),tilingSize);
                }else if constexpr(type==3){//amin 最大值
                    if constexpr(std::is_same<DTYPE_Y, float>::value){
                        Duplicate(yLocal, static_cast<DTYPE_Y>(3.4028234663852886e38f),tilingSize);
                    }else{
                        Duplicate(yLocal, static_cast<DTYPE_Y>(65504.0f),tilingSize);
                    }
                }else if constexpr(type==4){//amax 最小值
                    if constexpr(std::is_same<DTYPE_Y, float>::value){
                        Duplicate(yLocal, static_cast<DTYPE_Y>(-3.4028234663852886e38f),tilingSize);
                    }else{
                        Duplicate(yLocal, static_cast<DTYPE_Y>(-65504.0f),tilingSize);
                    }                
                }
            }
                //printf("$$$$$$$$$$$$$$$$$$$$4 compute0\n");

            
            int32_t currentIndex=0;
            int32_t DataTypeSize;
            if constexpr(std::is_same<DTYPE_Y, float>::value){
                DataTypeSize=4;
            }else{
                DataTypeSize=2;
            }
            // outputIndex=destJ*lineSize+k;计算下标以B为单位
            LocalTensor indexForScatter=BufTemp.Get<int32_t>();

            Muls(indexForScatter, indexLocal, (int32_t)lineSize, tilingSize);
            Muls(indexLocal, indexForScatter, (int32_t)DataTypeSize, tilingSize);
            Add(indexLocal, indexLocal, indexKLocal, tilingSize);

            //int转uint
            QueueIndex.EnQue(indexLocal);
            LocalTensor<uint32_t> newIndex=QueueIndex.DeQue<uint32_t>();
            

            for(uint16_t j=0;j<J;j++){//一行一行处理元素

                //服了，gather又需要无符号
                Gather(tmpYLocal, yLocal,newIndex[currentIndex] , (uint32_t)0, curlineSize);

                if constexpr(type==0 || type==1 ||type==2){//sum 0 prod 1 mean 2 
                    Gather(tmpCountLocal, countLocal, newIndex[currentIndex], (uint32_t)0, curlineSize);
                }
                

                //根据类型进行运算
                if constexpr(type==0 || type==2){//sum 0 mean 2 ；mean和sum样处理有mean有溢出风险
                    Add(tmpYLocal, tmpYLocal,srcLocal[currentIndex], curlineSize);
                }else if constexpr(type==1){//prod 
                    Mul(tmpYLocal, tmpYLocal, srcLocal[currentIndex], curlineSize);
                }else if constexpr(type==3){//amin 
                    Min(tmpYLocal, tmpYLocal, srcLocal[currentIndex], curlineSize);
                }else if constexpr(type==4){//amax
                    Max(tmpYLocal, tmpYLocal, srcLocal[currentIndex], curlineSize);
                }

                if constexpr(type==0 || type==1 ||type==2){//sum 0 prod 1 mean 2 
                    //count++
                    Adds(tmpCountLocal,tmpCountLocal,(DTYPE_Y)1.0, curlineSize);
                }
                uint64_t mask[2]={1,0};
                for(int k=0;k<curlineSize;k++){//scatter元素 默认当前lineSize*sizeof（T）<= 128 瓶颈
                    int outputIndex=indexForScatter[currentIndex](k);//必须有GetValue？获得dest在第几行
                    Copy(yLocal[outputIndex], tmpYLocal, mask, 1, { 1, 1, 8, 8 });
                    if constexpr(type==0 || type==1 ||type==2){//sum 0 prod 1 mean 2 
                        Copy(countLocal[outputIndex], tmpCountLocal, mask, 1, { 1, 1, 8, 8 });
                    }                    
                    mask[0]<<=1;
                }

                    
                currentIndex+=lineSize;
            }



            //后处理，大小是tilingSize
            if constexpr(type==2){//mean：需要除以count
                Div(yLocal, yLocal, countLocal, tilingSize);
            }
            if (includeSelf==0){//尾块需要还原：count==0就还原
                LocalTensor<uint8_t> maskLocal = BufTemp.Get<uint8_t>();
                if constexpr(type==0 || type==1 || type==2){//sum
                    //int32_t（只支持CMPMODE::EQ）,那就不用这个了，直接用min（1,x）,但是select只支持bit操作
                    CompareScalar(maskLocal, countLocal, static_cast<DTYPE_Y>(0.0), AscendC::CMPMODE::EQ, tilingSize);
                    Select(yLocal, maskLocal, inputLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tilingSize);
                 }
                else if constexpr(type==3){//amin
                    if constexpr(std::is_same<DTYPE_Y, float>::value){
                        CompareScalar(maskLocal, yLocal, static_cast<DTYPE_Y>(3.4028234663852886e38f), AscendC::CMPMODE::EQ, tilingSize);
                        Select(yLocal, maskLocal, inputLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tilingSize); 
                    }else{
                        CompareScalar(maskLocal, yLocal, static_cast<DTYPE_Y>(65504.0f), AscendC::CMPMODE::EQ, tilingSize);
                        Select(yLocal, maskLocal, inputLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tilingSize); 
                    }
                }
                else if constexpr(type==4){//amax
                    if constexpr(std::is_same<DTYPE_Y, float>::value){
                        CompareScalar(maskLocal, yLocal, static_cast<DTYPE_Y>(-3.4028234663852886e38f), AscendC::CMPMODE::EQ, tilingSize);
                        Select(yLocal, maskLocal, inputLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tilingSize); 
                    }else{
                        CompareScalar(maskLocal, yLocal, static_cast<DTYPE_Y>(-65504.0f), AscendC::CMPMODE::EQ, tilingSize);
                        Select(yLocal, maskLocal, inputLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tilingSize); 
                    }  
                }

            }

            QueueInput.FreeTensor(inputLocal);
            QueueIndex.FreeTensor(indexLocal);
            QueueSrc.FreeTensor(srcLocal);
            if constexpr(type==0 || type==1 ||type==2){//sum 0 prod 1 mean 2 
                QueueCount.FreeTensor(countLocal);
            }

            QueueY.EnQue(yLocal);
        }
        __aicore__ inline void copyOut(int gmIndex){
            AscendC::SliceInfo dstSliceInfo[] = {{0, tilingSize-1, 0, blockNum, tilingSize}};
            AscendC::SliceInfo srcSliceInfo[] = {{0,(uint32_t)((J-1)*KS+lineSize-1),(uint32_t)(KS-lineSize),blockNum,(uint32_t)((J-1)*KS+lineSize)}};

            LocalTensor yLocal=QueueY.DeQue<DTYPE_Y>();
    
            DataCopy(yGm[gmIndex],yLocal, srcSliceInfo,dstSliceInfo);
                //printf("$$$$$$$$$$$$$$$$$$$$4 copyOut:%d\n",gmIndex);

            QueueY.FreeTensor(yLocal);
        }
    
    private:
        GlobalTensor<DTYPE_SELF> inputGm;
        GlobalTensor<DTYPE_INDEX> indexGm;
        GlobalTensor<DTYPE_SRC> srcGm;
        GlobalTensor<DTYPE_Y> yGm;
        GlobalTensor<int>countGm;
        TPipe *pipe;
    
    
        TQue<QuePosition::VECIN, bufferNum> QueueInput;
        TQue<QuePosition::VECIN, bufferNum> QueueIndex;
        TQue<QuePosition::VECIN, bufferNum> QueueSrc;
    
        TQue<QuePosition::VECIN, bufferNum> QueueCount;
        TQue<QuePosition::VECOUT, bufferNum> QueueY;
    
        TBuf<QuePosition::VECCALC> BufTmpY;
        TBuf<QuePosition::VECCALC> BufCountY;
        TBuf<QuePosition::VECCALC> BufTemp;


        TBuf<QuePosition::VECCALC> BufIndexK;

        LocalTensor<int32_t> indexKLocal;
        LocalTensor<DTYPE_Y> tmpYLocal;
        LocalTensor<DTYPE_Y> tmpCountLocal;

        uint32_t KS;
        uint32_t beginIndex;
        uint32_t tilingNum;

        uint32_t tailBlockSize;

        //一次处理的元素数tilingSize=J*lineSize
        uint32_t tilingSize;
        //一次处理一行元素个数
        uint32_t lineSize;
        //搬运的32B块数
        uint32_t blockNum;   
        uint32_t J;    
 
        int8_t includeSelf;
};

template<uint8_t type>
class KernelScatterReduce_multiple_core_a_without_self {
    public:
        __aicore__ inline KernelScatterReduce_multiple_core_a_without_self() {}
       
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR src,GM_ADDR y
            ,int inputSize,int J 
            ,int totalTilingNum,int KS
            ,int lineSize,int8_t includeSelf
            ,TPipe* pipeIn
        )
        {

    
            //直接算最开始的偏移，所以这里不需要计算偏移量
            inputGm.SetGlobalBuffer((__gm__ DTYPE_SELF *)input, inputSize);
            indexGm.SetGlobalBuffer((__gm__ DTYPE_INDEX *)index, inputSize);
            srcGm.SetGlobalBuffer((__gm__ DTYPE_SRC *)src, inputSize);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, inputSize);
    
    

            //计算当前核心需要算多少块，以下单位都是块
            int formerNum = totalTilingNum  % GetBlockNum();
            if (GetBlockIdx() < formerNum) {
                this->tilingNum=totalTilingNum/GetBlockNum()+1;
                this->beginIndex=this->tilingNum*GetBlockIdx();
            }else{
                this->tilingNum=totalTilingNum/GetBlockNum();
                this->beginIndex=this->tilingNum*GetBlockIdx()+formerNum;
            }


            this->J=J;
            this->KS=KS;
            this->lineSize=lineSize;
            this->tilingSize=J*lineSize;
            this->includeSelf=includeSelf;
            //一行搬运的32B块数，切片搬运需要用到，所以这里应该算错了，应该用lineSize来算
            this->blockNum=lineSize*sizeof(DTYPE_Y)/32;

            this->pipe = pipeIn;
             
            int spaceSize=tilingSize*sizeof(DTYPE_Y);
            pipe->InitBuffer(QueueInput, bufferNum, spaceSize);
            pipe->InitBuffer(QueueSrc, bufferNum, spaceSize);
            pipe->InitBuffer(QueueY, bufferNum, spaceSize+256);
    
            pipe->InitBuffer(BufTmpY, spaceSize);
            pipe->InitBuffer(BufTemp, tilingSize*4*2);//后续需要copy count
            pipe->InitBuffer(BufIndexK, tilingSize*4);//升序index（以字节为偏移单位）
            pipe->InitBuffer(BufMask, tilingSize>>3);

            //申请临时计算空间
            tmpYLocal=BufTmpY.Get<DTYPE_Y>();

            this->tailBlockSize=KS%lineSize;

            //index解析成int类型，以下是以B为单位
            pipe->InitBuffer(QueueIndex, bufferNum, tilingSize*4);

        }
    
    
        
    
        __aicore__ inline void Process()
        {   



            //tiling块下标
            int endIndex=beginIndex+tilingNum;
            int tilingNum_per_line=KS/lineSize;
            int JKS=J*KS;

            LocalTensor indexKLocal_tmp=BufIndexK.Get<int>();
            //生成下标，以B为单位
            if constexpr(std::is_same<DTYPE_Y, float>::value){//用于实现转置
                int currentIndex=0;
                const int diffV=4*lineSize;
                ArithProgression(indexKLocal_tmp,0,diffV,J);
                for(int k=0;k<lineSize;k++){
                    Adds(indexKLocal_tmp[currentIndex+J],indexKLocal_tmp[currentIndex],(int)4*1,J);
                    currentIndex+=J;
                }
            }else{
                int currentIndex=0;
                const int diffV=2*lineSize;

                ArithProgression(indexKLocal_tmp,0,diffV,J);
                for(int k=0;k<lineSize;k++){
                    Adds(indexKLocal_tmp[currentIndex+J],indexKLocal_tmp[currentIndex],(int)2*1,J);
                    currentIndex+=J;
                }
            }

            indexKLocal=BufIndexK.Get<uint32_t>();

            for(int i=beginIndex;i<endIndex;i++){    
                // i表示块编号
                // (KS/tilingNum_per_line)表示一行可以分成多少32B块
                // i/每行块数==行数（共HI行），每行J*KS个元素
                // i%每行块数=行内偏移的块号，每块(32/sizeof(DTYPE_Y)个元素
                // TODO:让块长变大
                int gmIndex=i/tilingNum_per_line*JKS+i%tilingNum_per_line*lineSize;
                //////printf("$$$$$$$$$$$$$$$$$$$$4 gmIndex:%d\n",gmIndex);
                //i后面是尾块
                //行内偏移的块号是最后一块
                if(i%tilingNum_per_line==tilingNum_per_line-1 && tailBlockSize!=0){
                    copyIn(gmIndex);
                    compute(true);//非完整计算
                    copyOut(gmIndex);
                    //不能重复计算
                    gmIndex+=tailBlockSize;
                    copyIn(gmIndex);
                    compute(false);
                    copyOut(gmIndex);
                }else{
                    copyIn(gmIndex);
                    compute(false);
                    copyOut(gmIndex);
                }
            }  
        }
        __aicore__ inline void copyIn(int gmIndex){
            AscendC::SliceInfo dstSliceInfo[] = {{0, tilingSize-1, 0, blockNum, tilingSize}};
            AscendC::SliceInfo srcSliceInfo[] = {{0,(uint32_t)((J-1)*KS+lineSize-1),(uint32_t)(KS-lineSize),blockNum,(uint32_t)((J-1)*KS+lineSize)}};
        
            
            uint32_t blockNum_index;
            if constexpr(std::is_same<DTYPE_Y, float>::value){
                blockNum_index=blockNum;
            }else{
                blockNum_index=blockNum*2;//half类型下，index比half大一倍，所以乘2
            }
           
            AscendC::SliceInfo dstSliceInfo_index[] = {{0, tilingSize-1, 0, blockNum_index, tilingSize}};
            //当数据类型是fp16，应该考虑使用两块32B的index？ 其他是以元素为单位，只有blockNum是32B为单位
            AscendC::SliceInfo srcSliceInfo_index[] = {{0,(uint32_t)((J-1)*KS+lineSize-1),(uint32_t)(KS-lineSize),blockNum_index,(uint32_t)((J-1)*KS+lineSize)}};
            
            LocalTensor<DTYPE_Y> inputLocal = QueueInput.AllocTensor<DTYPE_Y>();
            LocalTensor<DTYPE_Y> srcLocal = QueueSrc.AllocTensor<DTYPE_Y>();
            LocalTensor<int32_t> indexLocal = QueueIndex.AllocTensor<int32_t>();
    
            DataCopy(inputLocal, inputGm[gmIndex], dstSliceInfo, srcSliceInfo);
            DataCopy(srcLocal, srcGm[gmIndex], dstSliceInfo, srcSliceInfo);
            DataCopy(indexLocal, indexGm[gmIndex], dstSliceInfo_index, srcSliceInfo_index);


            
            QueueInput.EnQue(inputLocal);
            QueueIndex.EnQue(indexLocal);
            QueueSrc.EnQue(srcLocal);    
        }
        __aicore__ inline void compute(bool tag){
    
    
            LocalTensor inputLocal=QueueInput.DeQue<DTYPE_Y>();
            LocalTensor indexLocal=QueueIndex.DeQue<int32_t>();
            LocalTensor srcLocal=QueueSrc.DeQue<DTYPE_Y>();
            LocalTensor<DTYPE_Y> yLocal = QueueY.AllocTensor<DTYPE_Y>();

            LocalTensor Y=BufTemp.Get<DTYPE_Y>();

            //用于尾块处理
            uint32_t curlineSize;

            if(!tag){
                curlineSize=lineSize;
            }else{//尾块处理
                curlineSize=tailBlockSize;
                if constexpr(type==3){//amin 最大值
                    if constexpr(std::is_same<DTYPE_Y, float>::value){
                        Duplicate(Y, static_cast<DTYPE_Y>(3.4028234663852886e38f),tilingSize*2);
                    }else{
                        Duplicate(Y, static_cast<DTYPE_Y>(65504.0f),tilingSize*2);
                    }
                }else if constexpr(type==4){//amax 最小值
                    if constexpr(std::is_same<DTYPE_Y, float>::value){
                        Duplicate(Y, static_cast<DTYPE_Y>(-3.4028234663852886e38f),tilingSize*2);
                    }else{
                        Duplicate(Y, static_cast<DTYPE_Y>(-65504.0f),tilingSize*2);
                    }                
                }
            }

            


            for(int j=0;j<J;j++){//一行一行处理元素

                if constexpr(type==3){//amin 最大值
                    if constexpr(std::is_same<DTYPE_Y, float>::value){
                        Duplicate(tmpYLocal, static_cast<DTYPE_Y>(3.4028234663852886e38f),tilingSize);
                    }else{
                        Duplicate(tmpYLocal, static_cast<DTYPE_Y>(65504.0f),tilingSize);
                    }
                }else if constexpr(type==4){//amax 最小值
                    if constexpr(std::is_same<DTYPE_Y, float>::value){
                        Duplicate(tmpYLocal, static_cast<DTYPE_Y>(-3.4028234663852886e38f),tilingSize);
                    }else{
                        Duplicate(tmpYLocal, static_cast<DTYPE_Y>(-65504.0f),tilingSize);
                    }                
                }
                LocalTensor<uint8_t> maskLocal = BufMask.Get<uint8_t>();
                CompareScalar(maskLocal, indexLocal, j, AscendC::CMPMODE::EQ, tilingSize);
                Select(tmpYLocal, maskLocal, srcLocal, tmpYLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tilingSize); 

                //转置tmpYLocal-> lineSize*J
                Gather(tmpYLocal, tmpYLocal , indexKLocal , (uint32_t)0, tilingSize);

                int curIndex=0;
                int YBeginIndex=j*lineSize;
                LocalTensor<DTYPE_Y> workLocal = BufMask.Get<DTYPE_Y>();//应该够了
                for(int k=0;k<curlineSize;k++){
                    if constexpr(type==3){//amin 
                        ReduceMin<DTYPE_Y>(Y[(YBeginIndex+k)*2], tmpYLocal[curIndex], workLocal, J,false);
                    }else if constexpr(type==4){//amax
                        ReduceMax<DTYPE_Y>(Y[(YBeginIndex+k)*2], tmpYLocal[curIndex], workLocal, J,false);
                    }
                    curIndex+=J;
                }
            }
            uint64_t tilingSizeVar = tilingSize;

            //收集回来 lineSize*2<256
            GatherMask(yLocal, Y, 1, true,J , { static_cast<uint8_t>( lineSize*2), 1, 8, 8 }, tilingSizeVar);

            //后处理，大小是tilingSize
            LocalTensor<uint8_t> maskLocal = BufTemp.Get<uint8_t>();
            if constexpr(type==3){//amin
                if constexpr(std::is_same<DTYPE_Y, float>::value){
                    CompareScalar(maskLocal, yLocal, static_cast<DTYPE_Y>(3.4028234663852886e38f), AscendC::CMPMODE::EQ, tilingSize);
                    Select(yLocal, maskLocal, inputLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tilingSize); 
                }else{
                    CompareScalar(maskLocal, yLocal, static_cast<DTYPE_Y>(65504.0f), AscendC::CMPMODE::EQ, tilingSize);
                    Select(yLocal, maskLocal, inputLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tilingSize); 
                }
            }
            else if constexpr(type==4){//amax
                if constexpr(std::is_same<DTYPE_Y, float>::value){
                    CompareScalar(maskLocal, yLocal, static_cast<DTYPE_Y>(-3.4028234663852886e38f), AscendC::CMPMODE::EQ, tilingSize);
                    Select(yLocal, maskLocal, inputLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tilingSize); 
                }else{
                    CompareScalar(maskLocal, yLocal, static_cast<DTYPE_Y>(-65504.0f), AscendC::CMPMODE::EQ, tilingSize);
                    Select(yLocal, maskLocal, inputLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tilingSize); 
                }  
            }

            

            QueueInput.FreeTensor(inputLocal);
            QueueIndex.FreeTensor(indexLocal);
            QueueSrc.FreeTensor(srcLocal);

            QueueY.EnQue(yLocal);
        }
        __aicore__ inline void copyOut(int gmIndex){
            AscendC::SliceInfo dstSliceInfo[] = {{0, tilingSize-1, 0, blockNum, tilingSize}};
            AscendC::SliceInfo srcSliceInfo[] = {{0,(uint32_t)((J-1)*KS+lineSize-1),(uint32_t)(KS-lineSize),blockNum,(uint32_t)((J-1)*KS+lineSize)}};

            LocalTensor yLocal=QueueY.DeQue<DTYPE_Y>();
    
            DataCopy(yGm[gmIndex],yLocal, srcSliceInfo,dstSliceInfo);
                //printf("$$$$$$$$$$$$$$$$$$$$4 copyOut:%d\n",gmIndex);

            QueueY.FreeTensor(yLocal);
        }
    
    private:
        GlobalTensor<DTYPE_SELF> inputGm;
        GlobalTensor<DTYPE_INDEX> indexGm;
        GlobalTensor<DTYPE_SRC> srcGm;
        GlobalTensor<DTYPE_Y> yGm;
        GlobalTensor<int>countGm;
        TPipe *pipe;
    
    
        TQue<QuePosition::VECIN, bufferNum> QueueInput;
        TQue<QuePosition::VECIN, bufferNum> QueueIndex;
        TQue<QuePosition::VECIN, bufferNum> QueueSrc;
    
        TQue<QuePosition::VECIN, bufferNum> QueueCount;
        TQue<QuePosition::VECOUT, bufferNum> QueueY;
    
        TBuf<QuePosition::VECCALC> BufTmpY;
        TBuf<QuePosition::VECCALC> BufCountY;
        TBuf<QuePosition::VECCALC> BufTemp;
        TBuf<QuePosition::VECCALC> BufMask;



        TBuf<QuePosition::VECCALC> BufIndexK;

        LocalTensor<uint32_t> indexKLocal;
        LocalTensor<DTYPE_Y> tmpYLocal;
        LocalTensor<DTYPE_Y> tmpCountLocal;

        uint32_t KS;
        uint32_t beginIndex;
        uint32_t tilingNum;

        uint32_t tailBlockSize;

        //一次处理的元素数tilingSize=J*lineSize
        uint32_t tilingSize;
        //一次处理一行元素个数
        uint32_t lineSize;
        //搬运的32B块数
        uint32_t blockNum;   
        uint32_t J;    
 
        int8_t includeSelf;
};

extern "C" __global__ __aicore__ void scatter_reduce(GM_ADDR self, GM_ADDR index, GM_ADDR src, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if(TILING_KEY_IS(1)){
        KernelScatterReduce op;
        GM_ADDR count=GetUserWorkspace(workspace);
        op.Init(self,index,src,y,count,tiling_data.inputSize,tiling_data.type
            ,tiling_data.mmInputDims
           ,tiling_data.nInputDims
            ,tiling_data.includeSelf
            ,tiling_data.dim
           );
        op.Process();
    }else if(TILING_KEY_IS(2)){
        KernelScatterReduce_multiple_core<0> op;
        TPipe pipe;
        op.Init(self,index,src,y
            ,tiling_data.inputSize
            ,tiling_data.J
           ,tiling_data.totalTilingNum
            ,tiling_data.KS
            ,tiling_data.lineSize
            ,tiling_data.includeSelf, &pipe
           );
        op.Process();
    }else if(TILING_KEY_IS(3)){
        KernelScatterReduce_multiple_core<1> op;
        TPipe pipe;
        op.Init(self,index,src,y
            ,tiling_data.inputSize
            ,tiling_data.J
           ,tiling_data.totalTilingNum
            ,tiling_data.KS
            ,tiling_data.lineSize
            ,tiling_data.includeSelf, &pipe
           );
        op.Process();
    }else if(TILING_KEY_IS(4)){
        KernelScatterReduce_multiple_core<2> op;
        TPipe pipe;
        op.Init(self,index,src,y
            ,tiling_data.inputSize
            ,tiling_data.J
           ,tiling_data.totalTilingNum
            ,tiling_data.KS
            ,tiling_data.lineSize
            ,tiling_data.includeSelf, &pipe
           );
        op.Process();
    }else if(TILING_KEY_IS(5)){
        if(!tiling_data.includeSelf){
            KernelScatterReduce_multiple_core_a_without_self<3> op;
            TPipe pipe;
            op.Init(self,index,src,y
                ,tiling_data.inputSize
                ,tiling_data.J
               ,tiling_data.totalTilingNum
                ,tiling_data.KS
                ,tiling_data.lineSize
                ,tiling_data.includeSelf, &pipe
               );
            op.Process();
        }else{
            KernelScatterReduce_multiple_core<3> op;
            TPipe pipe;
            op.Init(self,index,src,y
                ,tiling_data.inputSize
                ,tiling_data.J
               ,tiling_data.totalTilingNum
                ,tiling_data.KS
                ,tiling_data.lineSize
                ,tiling_data.includeSelf, &pipe
               );
            op.Process();
        }
    }else if(TILING_KEY_IS(6)){
        if(!tiling_data.includeSelf){
            KernelScatterReduce_multiple_core_a_without_self<4> op;
            TPipe pipe;
            op.Init(self,index,src,y
                ,tiling_data.inputSize
                ,tiling_data.J
               ,tiling_data.totalTilingNum
                ,tiling_data.KS
                ,tiling_data.lineSize
                ,tiling_data.includeSelf, &pipe
               );
            op.Process();
        }else{
            KernelScatterReduce_multiple_core<4> op;
            TPipe pipe;
            op.Init(self,index,src,y
                ,tiling_data.inputSize
                ,tiling_data.J
               ,tiling_data.totalTilingNum
                ,tiling_data.KS
                ,tiling_data.lineSize
                ,tiling_data.includeSelf, &pipe
               );
            op.Process();
        }
    }

}