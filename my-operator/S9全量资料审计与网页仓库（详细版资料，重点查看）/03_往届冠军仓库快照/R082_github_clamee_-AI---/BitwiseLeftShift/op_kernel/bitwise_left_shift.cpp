#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"
//#include <cstdint>
//#include <bitset>
using namespace AscendC;

#define ny n1
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define TILE_SIZE 4096

//const uint32_t bits = 0xBE000000;
    
//const float custom_float = *reinterpret_cast<float*>(&bits);

template<typename T> class BruteForce {
public:
	__aicore__ inline BruteForce() {}
	__aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t n1[3], uint32_t n2[3]) {
		ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
		uint32_t total1 = 1, total2 = 1, totaly = 1;
		for (int i = 0; i < 3; ++i) {
			this->n1[i] = n1[i];
			this->n2[i] = n2[i];
	//		this->ny[i] = ny[i];
			total1 *= n1[i];
			total2 *= n2[i];
	//		totaly *= ny[i];
		}
		
		x1Gm.SetGlobalBuffer((__gm__ T*)x1, total1);
		x2Gm.SetGlobalBuffer((__gm__ T*)x2, total2);
		yGm.SetGlobalBuffer((__gm__ T*)y, totaly);
	}
	__aicore__ inline void Process() {
		for (uint32_t i0 = 0; i0 < ny[0]; ++i0) {
			for (uint32_t i1 = 0; i1 < ny[1]; ++i1) {
				for (uint32_t i2 = 0; i2 < ny[2]; ++i2) {
					uint32_t indices[3] = {i0, i1, i2};
					uint32_t idx1 = 0, idx2 = 0, idxy = 0;
					for (int j = 0; j < 3; ++j) {
						idx1 = idx1 * n1[j] + indices[j] % n1[j];
						idx2 = idx2 * n2[j] + indices[j] % n2[j];
						idxy = idxy * ny[j] + indices[j];
					}
					auto a = x1Gm.GetValue(idx1);
					auto b = x2Gm.GetValue(idx2);
					yGm.SetValue(idxy, a<<b);
				}
			}
		}
	}
	
private:
	GlobalTensor<T> x1Gm, x2Gm, yGm;
	uint32_t n1[3], n2[3];
};
template<typename T> class LeftShiftKernalFastBroadCast {
public:
	__aicore__ inline LeftShiftKernalFastBroadCast() {}
	__aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t size, uint32_t length, uint32_t n1[3], uint32_t n2[3]) {
		ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
		const unsigned num_cores = GetBlockNum();
		unsigned L = GetBlockIdx() * length;
		unsigned R = (GetBlockIdx() + 1) * length;
		this->addid = L;
		if (R > size) {
			R = size;
		}
		this->L = 0;
		this->R = R - L;
		uint32_t total2 = 1;
		for (int i = 0; i < 3; ++i) {
			this->n1[i] = n1[i];
			this->n2[i] = n2[i];
	//		this->ny[i] = ny[i];
			total2 *= n2[i];
		}
		x1Gm.SetGlobalBuffer((__gm__ T*)x1 + L, length);
		x2Gm.SetGlobalBuffer((__gm__ T*)x2 , total2);
		yGm.SetGlobalBuffer((__gm__ T*)y + L, length);
	}
	__aicore__ inline void Process() {
		uint32_t idx1 = L + addid;
                uint32_t i2 = idx1 % n1[2];
                uint32_t temp = idx1 / n1[2];
                uint32_t i1 = temp % n1[1];
                uint32_t i0 = (temp / n1[1]) % n1[0];
                        //uint32_t idx2 = (i0 % n2[0]) * (n2[1] * n2[2]) + (i1 % n2[1]) * n2[2] + (i2 % n2[2]);
		uint32_t j0=i0%n2[0];
		uint32_t j1=i1%n2[1];
		uint32_t j2=i2%n2[2];
		uint32_t s0=n2[1]*n2[2];
		#define s1 n2[2]
		#define s2 1
		uint32_t idx2=j0*s0+j1*s1+j2;
		for (uint32_t i = L; i < R; ++i) {
			T a = x1Gm.GetValue(i);
			T b = x2Gm.GetValue(idx2);
			yGm.SetValue(i, a<<b);
			i2++;idx2++;
			if(i2==n1[2])
			{
				i2=0;j2=0;
				i1++;//idx2+=s1;
				if(i1==n1[1])
				{
					i1=0;j1=0;
					i0++;j0++;
					//idx2+=s0;
					if(j0==n2[0])
					{
						j0=0;
						idx2=0;
					}
				}
				else
				{
					j1++;
					if(j1==n2[1])
					{
						j1=0;
						idx2-=s0;
					}
				}
			}
			else
			{
				j2++;
				if(j2==n2[2])
				{
					j2=0;
					idx2-=s1;
				}
			}


		}
		//AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
		//DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(yGm);
	}
	
private:
	GlobalTensor<T> x1Gm, x2Gm, yGm;
	uint32_t n1[3], n2[3];
	uint32_t L, R, addid;
};
template<typename T> class LeftShiftKernalFast {
public:
	__aicore__ inline LeftShiftKernalFast() {}
	__aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t size, uint32_t length) {
		ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
		const unsigned num_cores = GetBlockNum();
		unsigned L = GetBlockIdx() * length;
		unsigned R = (GetBlockIdx() + 1) * length;
		if (R > size) {
			R = size;
		}
		this->L = 0;
		this->R = R - L;
		x1Gm.SetGlobalBuffer((__gm__ T*)x1 + L, length);
		x2Gm.SetGlobalBuffer((__gm__ T*)x2 + L, length);
		yGm.SetGlobalBuffer((__gm__ T*)y + L, length);
	}
	__aicore__ inline void Process() {
		for (int i = L; i < R; ++i) {
			T a = x1Gm.GetValue(i);
			T b = x2Gm.GetValue(i);
			yGm.SetValue(i, a<<b);
		}
		//AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
		//DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(yGm);
	}
	
private:
	GlobalTensor<T> x1Gm, x2Gm, yGm;
	uint32_t L, R;
};

template <typename T>
class KernelGcd
{
public:
	__aicore__ inline KernelGcd()
	{
	}
	__aicore__ inline void Init(int N0, int N1, int N2, TPipe *pipe, GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
	{
		N[0] = N0;
		N[1] = N1;
		N[2] = N2;
		sizeX1 = 1;
		sizeX2 = 1;
		for (int i = 0;i < 3;i++) {
			sizeX1 *= N[i];
			sizeX2 *= N[i];
		}
		x1Gm.SetGlobalBuffer((__gm__ T *)x1, sizeX1);  // 正确：sizeX1是元素总数
		x2Gm.SetGlobalBuffer((__gm__ T *)x2, sizeX2);
		yGm.SetGlobalBuffer((__gm__ T *)y, sizeX1);

		if constexpr(std::is_same<T, int8_t>::value) {
                        pipe->InitBuffer(tBufNext, 2 * TILE_SIZE * sizeof(half));
//                        pipe->InitBuffer(tBufMask, TILE_SIZE * sizeof(uint8_t));
                pipe->InitBuffer(inX1, 1, TILE_SIZE * sizeof(T));
                pipe->InitBuffer(inX2, 1, TILE_SIZE * sizeof(T));
                pipe->InitBuffer(outY, 1, TILE_SIZE * sizeof(T));
		}

		if constexpr(std::is_same<T, int16_t>::value) {
			pipe->InitBuffer(tBufNext, 4 * TILE_SIZE * sizeof(half));
			pipe->InitBuffer(tBufMask, 2 * TILE_SIZE * sizeof(uint8_t));
                pipe->InitBuffer(inX1, 1, 2*TILE_SIZE * sizeof(T));
                pipe->InitBuffer(inX2, 1, 2*TILE_SIZE * sizeof(T));
                pipe->InitBuffer(outY, 1, 2*TILE_SIZE * sizeof(T));
		}
		
		if constexpr(std::is_same<T, int32_t>::value) {
//			pipe->InitBuffer(tBufNext, 2 * TILE_SIZE * sizeof(float));
//			pipe->InitBuffer(tBufMask, TILE_SIZE * sizeof(uint8_t));
                pipe->InitBuffer(inX1, 2, TILE_SIZE * sizeof(T));
                pipe->InitBuffer(inX2, 2, TILE_SIZE * sizeof(T));
                pipe->InitBuffer(outY, 2, TILE_SIZE * sizeof(T));
		}
		
//		pipe->InitBuffer(inX1, 1, TILE_SIZE * sizeof(T));
//		pipe->InitBuffer(inX2, 1, TILE_SIZE * sizeof(T));
//		pipe->InitBuffer(outY, 1, TILE_SIZE * sizeof(T));
	}
	
	__aicore__ inline void CopyInX1(int64_t offset, int len) {
		LocalTensor<T> x1 = inX1.AllocTensor<T>();
		DataCopyExtParams copyParamsX;
		copyParamsX.blockCount = 1;
		copyParamsX.blockLen = len * sizeof(T);
		copyParamsX.srcStride = 0;
		copyParamsX.dstStride = 0;
		DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
		DataCopyPad(x1, x1Gm[offset], copyParamsX, padParams);
		inX1.EnQue(x1);
	}
	
	__aicore__ inline void CopyInX2(int64_t offset, int len) {
		LocalTensor<T> x2 = inX2.AllocTensor<T>();
		DataCopyExtParams copyParamsX;
		copyParamsX.blockCount = 1;
		copyParamsX.blockLen = len * sizeof(T);
		copyParamsX.srcStride = 0;
		copyParamsX.dstStride = 0;
		DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
		DataCopyPad(x2, x2Gm[offset], copyParamsX, padParams);
		inX2.EnQue(x2);
	}
	
	__aicore__ inline void CopyOut(int64_t offset, int len) {
		LocalTensor<T> y = outY.DeQue<T>();
		DataCopyExtParams copyParamsX;
		copyParamsX.blockCount = 1;
		copyParamsX.blockLen = len * sizeof(T);
		copyParamsX.srcStride = 0;
		copyParamsX.dstStride = 0;
		DataCopyPad(yGm[offset], y, copyParamsX);
		outY.FreeTensor(y);
	}
        __aicore__ inline void GcdLikely8(const LocalTensor<T>& c, const LocalTensor<T>& a, const LocalTensor<T>& b) {
                auto tmp = tBufNext.Get<half>();
                auto tmp2 = tBufNext.Get<half>()[TILE_SIZE];
                auto tmp_h = tmp.template ReinterpretCast<int16_t>();
                auto tmp2_h = tmp2.template ReinterpretCast<int16_t>();
		Cast(tmp,a,AscendC::RoundMode::CAST_NONE, TILE_SIZE);
		Cast(tmp2,b,AscendC::RoundMode::CAST_NONE, TILE_SIZE);
		Cast(tmp_h,tmp,AscendC::RoundMode::CAST_RINT, TILE_SIZE);
		Cast(tmp2_h,tmp2,AscendC::RoundMode::CAST_RINT, TILE_SIZE);
		int16_t u15=15;
		int16_t	u10=10;
		Adds(tmp2_h,tmp2_h,u15,TILE_SIZE);
		ShiftLeft(tmp2_h,tmp2_h,u10,TILE_SIZE);
		Cast(tmp2_h,tmp2,AscendC::RoundMode::CAST_RINT, TILE_SIZE);
		Mul(tmp_h,tmp_h,tmp2_h,TILE_SIZE);
		Cast(tmp,tmp_h,AscendC::RoundMode::CAST_RINT, TILE_SIZE);
        //        Cast(tmp2,tmp2_h,AscendC::RoundMode::CAST_RINT, TILE_SIZE);
		Cast(c,tmp,AscendC::RoundMode::CAST_NONE, TILE_SIZE);
        }
// 原GCD代码（完全不动）
	__aicore__ inline void GcdLikely16(const LocalTensor<T>& c, const LocalTensor<T>& a, const LocalTensor<T>& b) {
		auto tmp = tBufNext.Get<T>();
		auto tmp2 = tBufNext.Get<T>()[2*TILE_SIZE];
		auto a_h = a.template ReinterpretCast<half>();
		auto tmp_h = tmp.template ReinterpretCast<half>();
		auto tmp2_h = tmp2.template ReinterpretCast<half>();
		//auto tmp4_h = tmp4.template ReinterpretCast<int16_t>();
		auto mask = tBufMask.Get<uint8_t>();
		uint32_t t=sizeof(T);
		T mxb=(8*t);
		Mins(b,b,mxb,2*TILE_SIZE);
		const T ONE=1;
		for(uint32_t i=0;(1<<i)<=mxb;i++)
		{
			ShiftLeft(tmp2,a,(T)(1<<i),2*TILE_SIZE);
			Duplicate<T>(tmp, ONE ,2*TILE_SIZE);
			And(tmp,tmp,b,2*TILE_SIZE);
			//Cast(tmp2, tmp, AscendC::RoundMode::CAST_NONE, TILE_SIZE);
			CompareScalar(mask, tmp_h, (half)0.0f, CMPMODE::EQ, 2*TILE_SIZE);
			Select(a_h, mask, a_h, tmp2_h, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 2*TILE_SIZE);
			ShiftRight(b,b,ONE,2*TILE_SIZE);
		}
		DataCopy(c,a,2*TILE_SIZE);
	}
	
	 __aicore__ inline void GcdLikely32(const LocalTensor<T>& c, const LocalTensor<T>& a, const LocalTensor<T>& b) {
		auto a_h = a.template ReinterpretCast<float>();
                auto b_h = b.template ReinterpretCast<float>();
                auto c_h = c.template ReinterpretCast<float>();
		const int32_t u23=23;
//                const T u5=5;
		//Adds(b,b,127,TILE_SIZE);
                ShiftLeft(b,b,u23,TILE_SIZE);
		Cast(a_h,a,AscendC::RoundMode::CAST_NONE,TILE_SIZE);
//                ShiftLeft(tmp_h,tmp_h,u23,TILE_SIZE);
                Add(a,a,b,TILE_SIZE);
                Cast(c,a_h,AscendC::RoundMode::CAST_RINT,TILE_SIZE);
//                Mul(c,a,b,TILE_SIZE);
        }

	__aicore__ inline void Compute() {
		auto x1 = inX1.DeQue<T>();
		auto x2 = inX2.DeQue<T>();
		auto y = outY.AllocTensor<T>();
		
		if constexpr(std::is_same<T, int16_t>::value) {
			GcdLikely16(y, x1, x2);  // 匹配int16_t
		} else if constexpr(std::is_same<T, int32_t>::value) {
			GcdLikely32(y, x1, x2);  // 匹配int32_t
		}
		else if constexpr(std::is_same<T, int8_t>::value)
		{
			GcdLikely8(y,x1,x2);
		}
		outY.EnQue(y);
		inX1.FreeTensor(x1);
		inX2.FreeTensor(x2);
	}
	
	__aicore__ inline void ProcessFast()
	{
		for (int64_t i = GetBlockIdx() * TILE_SIZE;i < sizeX1;i+=TILE_SIZE * GetBlockNum()) {
	/*		if constexpr(std::is_same<T, int16_t>::value) {
                       // GcdLikely16(y, x1, x2);  // 匹配int16_t
		        CopyInX1(i, MIN(sizeX1 - i, 2*TILE_SIZE));
                        CopyInX2(i, MIN(sizeX1 - i, 2*TILE_SIZE));
                        Compute();
                        CopyOut(i, MIN(sizeX1 - i, 2*TILE_SIZE));
                }else {*/
			CopyInX1(i, MIN(sizeX1 - i, TILE_SIZE));
			CopyInX2(i, MIN(sizeX1 - i, TILE_SIZE));
			Compute();
			CopyOut(i, MIN(sizeX1 - i, TILE_SIZE));
		//}
		}
	}
	
	__aicore__ inline void Process() {
		ProcessFast();
	}
	
	TQue<QuePosition::VECIN, 1> inX1, inX2;
	TQue<QuePosition::VECOUT, 1> outY;
	TBuf<TPosition::VECCALC> tBufNext, tBufMask;
	
	int64_t N[3], sizeX1, sizeX2;
	GlobalTensor<T> x1Gm, x2Gm, yGm;
};
#define TILE_SIZE2 1200
template<typename T> class LcmKernalFastBroadCast64 {
public:
	__aicore__ inline LcmKernalFastBroadCast64() {}
	__aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t size, uint32_t length, uint32_t n1[3], uint32_t n2[3],TPipe *pipe) {
		//ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
		const unsigned num_cores = GetBlockNum();
		unsigned L = GetBlockIdx() * length;
		unsigned R = (GetBlockIdx() + 1) * length;
		this->addid = L;
		if (R > size) {
			R = size;
		}
		this->L = 0;
		this->R = R - L;
		uint32_t total2 = 1;
		//n2[1]=1;
		for (int i = 0; i < 3; ++i) {
			this->n1[i] = n1[i];
			this->n2[i] = n2[i];
			//		this->ny[i] = ny[i];
			if(i!=1)
				total2 *= n2[i];
			else
				this->n2[i]=1;
		}
		x1Gm.SetGlobalBuffer((__gm__ T*)x1 + L, length);
		x2Gm.SetGlobalBuffer((__gm__ T*)x2 , total2);
		yGm.SetGlobalBuffer((__gm__ T*)y + L, length);
		pipe->InitBuffer(inX1, 2, TILE_SIZE2 * sizeof(T));
		pipe->InitBuffer(inX2, 2, TILE_SIZE2 * sizeof(T));
		pipe->InitBuffer(outY, 2, TILE_SIZE2 * sizeof(T));
		pipe->InitBuffer(tBufNext, 2 * TILE_SIZE2 * sizeof(int32_t));
	//	pipe->InitBuffer(tBufFL, 3 * TILE_SIZE2 * sizeof(float));
	//	pipe->InitBuffer(tBufMask, TILE_SIZE2 * sizeof(uint8_t));
	}

	__aicore__ inline void CopyInX1(int64_t offset, int len) {
		LocalTensor<T> x1 = inX1.AllocTensor<T>();
		DataCopyExtParams copyParamsX;
		copyParamsX.blockCount = 1;
		copyParamsX.blockLen = len * sizeof(T);
		copyParamsX.srcStride = 0;
		copyParamsX.dstStride = 0;
		DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
		DataCopyPad(x1, x1Gm[offset], copyParamsX, padParams);
		inX1.EnQue(x1);
	}
	
	__aicore__ inline void CopyInX2(int64_t offset, int len) {
		LocalTensor<T> x2 = inX2.AllocTensor<T>();
		DataCopyExtParams copyParamsX;
		copyParamsX.blockCount = 1;
		copyParamsX.blockLen = len * sizeof(T);
		copyParamsX.srcStride = 0;
		copyParamsX.dstStride = 0;
		DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
		DataCopyPad(x2, x2Gm[offset], copyParamsX, padParams);
		inX2.EnQue(x2);
	}

	__aicore__ inline void CopyOut(int64_t offset, int len) {
		LocalTensor<T> y = outY.DeQue<T>();
		DataCopyExtParams copyParamsX;
		copyParamsX.blockCount = 1;
		copyParamsX.blockLen = len * sizeof(T);
		copyParamsX.srcStride = 0;
		copyParamsX.dstStride = 0;
		DataCopyPad(yGm[offset], y, copyParamsX);
		outY.FreeTensor(y);
	}
	__aicore__ inline void Compute(){
		auto x1 = inX1.DeQue<T>();
		auto x2 = inX2.DeQue<T>();
		auto y = outY.AllocTensor<T>();
		auto a = tBufNext.Get<int32_t>();
		auto b =  tBufNext.Get<int32_t>()[TILE_SIZE2];
		auto a_h = a.template ReinterpretCast<float>();
                auto b_h = b.template ReinterpretCast<float>();
                //auto c_h = c.template ReinterpretCast<float>();
                const int32_t u23=23;
//                const T u5=5; 
                //Adds(b,b,127,TILE_SIZE);
		Cast(b,x2,AscendC::RoundMode::CAST_NONE,TILE_SIZE2);
                ShiftLeft(b,b,u23,TILE_SIZE2);
                Cast(a_h,x1,AscendC::RoundMode::CAST_RINT,TILE_SIZE2);
//                ShiftLeft(tmp_h,tmp_h,u23,TILE_SIZE);
                Add(a,a,b,TILE_SIZE2);
                Cast(y,a_h,AscendC::RoundMode::CAST_RINT,TILE_SIZE2);

		outY.EnQue(y);
		inX1.FreeTensor(x1);
		inX2.FreeTensor(x2);
		
	}
	__aicore__ inline void Process() {
		uint32_t idx1 = L + addid;
		uint32_t i2 = idx1 % n1[2];
		uint32_t temp = idx1 / n1[2];
		uint32_t i1 = temp % n1[1];
		uint32_t i0 = (temp / n1[1]) % n1[0];
		//uint32_t idx2 = (i0 % n2[0]) * (n2[1] * n2[2]) + (i1 % n2[1]) * n2[2] + (i2 % n2[2]);
		uint32_t j0=i0%n2[0];
		uint32_t j1=i1%n2[1];
		uint32_t j2=i2%n2[2];
		uint32_t s0=n2[1]*n2[2];
#define s1 n2[2]
#define s2 1
		uint32_t idx2=j0*s0+j1*s1+j2;
		for (uint32_t i = L; i < R; ) {
			uint32_t sz=MIN(R-i, n1[2]-i2);
			CopyInX1(i, sz);
			CopyInX2(idx2, sz);
			
			if constexpr(std::is_same<T, int64_t>::value)
				Compute();
			CopyOut(i, sz);
			i1++;
			i+=sz;
			if(i1==n1[1])
			{
				i1=0;
				idx2+=sz;
			}
			if(i2==0)continue;
			i2=0;
			idx2+=sz;
			if(i1!=n1[1])
				idx2-=s0;
		}
		//AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
		//DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(yGm);
	}

private:
	GlobalTensor<T> x1Gm, x2Gm, yGm;
	uint32_t n1[3], n2[3];
	uint32_t L, R, addid;
	TQue<QuePosition::VECIN, 1> inX1, inX2;
	TQue<QuePosition::VECOUT, 1> outY;
	TBuf<TPosition::VECCALC> tBufNext;
};
extern "C" __global__ __aicore__ void bitwise_left_shift(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
	GET_TILING_DATA(tiling_data, tiling);
	if (tiling_data.status == 0) {
		BruteForce<DTYPE_X1> op;
		op.Init(x1, x2, y, tiling_data.n1, tiling_data.n1);
		op.Process();
	}
	else if(tiling_data.status == 1)
	{
		LeftShiftKernalFastBroadCast<DTYPE_X1> op;
		op.Init(x1, x2, y, tiling_data.size, tiling_data.length, tiling_data.n1, tiling_data.n1);
		op.Process();
	}
	/*else if(tiling_data.status==2){
		LeftShiftKernalFast<DTYPE_X1> op;
		op.Init(x1, x2, y, tiling_data.size, tiling_data.length);
		op.Process();
	}*/
	else if(tiling_data.status==5){
		LcmKernalFastBroadCast64<DTYPE_X1> op;
		TPipe pipe;
		op.Init(x1, x2, y, tiling_data.size, tiling_data.length, tiling_data.n1, tiling_data.n1,&pipe);
		op.Process();
	}
	/*else if(tiling_data.status==3)
	{
//		TILE_SIZE*=2;
		KernelGcd<int16_t> op;
		TPipe pipe;
		op.Init(
			tiling_data.n1[0], tiling_data.n1[1], tiling_data.n1[2], &pipe,
			x1, x2, y
			);
		op.Process();
	}*/
	else
	{
		KernelGcd<DTYPE_X1> op;
		TPipe pipe;
		op.Init(
			tiling_data.n1[0], tiling_data.n1[1], tiling_data.n1[2], &pipe,
			x1, x2, y
			);
		op.Process();
	}
}
