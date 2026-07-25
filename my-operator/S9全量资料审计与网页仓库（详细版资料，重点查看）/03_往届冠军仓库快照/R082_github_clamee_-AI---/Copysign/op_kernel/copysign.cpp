#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"
using namespace AscendC;

#define ny n1
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define TILE_SIZE 4096

template <typename T>
class KernelGcd
{
public:
	__aicore__ inline KernelGcd()
	{
	}
	__aicore__ inline void Init(int N0, int N1, int N2, int N3, TPipe *pipe, GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
	{
		N[0] = N0;
		N[1] = N1;
		N[2] = N2;
		N[3]=  N3;
		sizeX1 = 1;
		sizeX2 = 1;
		for (int i = 0;i < 4;i++) {
			sizeX1 *= N[i];
			sizeX2 *= N[i];
		}
		x1Gm.SetGlobalBuffer((__gm__ T *)x1, sizeX1);  // 正确：sizeX1是元素总数
		x2Gm.SetGlobalBuffer((__gm__ T *)x2, sizeX2);
		
		yGm.SetGlobalBuffer((__gm__ T *)y, sizeX1);
		/*
		if constexpr(std::is_same<T, float>::value) {
			pipe->InitBuffer(tBufNext, 3 * TILE_SIZE * sizeof(int16_t));
			//pipe->InitBuffer(tBufFL, 2 * TILE_SIZE * sizeof(float));
			pipe->InitBuffer(tBufMask, TILE_SIZE * sizeof(uint8_t));
			pipe->InitBuffer(tBufFL, 2 * TILE_SIZE * sizeof(float));
		}
		
		if constexpr(std::is_same<T, int32_t>::value) {
			pipe->InitBuffer(tBufNext, 2 * TILE_SIZE * sizeof(int32_t));
			pipe->InitBuffer(tBufFL, 3 * TILE_SIZE * sizeof(float));
			pipe->InitBuffer(tBufMask, TILE_SIZE * sizeof(uint8_t));
		}*/
		
		pipe->InitBuffer(inX1, 2, TILE_SIZE * sizeof(T));
		pipe->InitBuffer(inX2, 2, TILE_SIZE * sizeof(T));
		pipe->InitBuffer(outY, 2, TILE_SIZE * sizeof(T));
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
	
	__aicore__ inline void GcdLikely16(const LocalTensor<T>& c, const LocalTensor<T>& a, const LocalTensor<T>& b) {
    		auto a_h = a.template ReinterpretCast<uint16_t>();
   		auto b_h = b.template ReinterpretCast<uint16_t>();
   		auto c_h = c.template ReinterpretCast<uint16_t>();
   		const uint16_t t1=1,t15=15;
		ShiftLeft(a_h,a_h,t1,TILE_SIZE);
		ShiftRight(a_h,a_h,t1,TILE_SIZE);
		ShiftRight(b_h,b_h,t15,TILE_SIZE);
		ShiftLeft(b_h,b_h,t15,TILE_SIZE);
		Or(c_h,a_h,b_h,TILE_SIZE);
	}
	
	__aicore__ inline void GcdLikely32(const LocalTensor<T>& c, const LocalTensor<T>& a, const LocalTensor<T>& b) {
    		auto a_h = a.template ReinterpretCast<uint32_t>();
   		auto b_h = b.template ReinterpretCast<uint32_t>();
   		auto c_h = c.template ReinterpretCast<uint32_t>();
   		auto a_h2 = a.template ReinterpretCast<uint16_t>();
   		auto b_h2 = b.template ReinterpretCast<uint16_t>();
   		auto c_h2 = c.template ReinterpretCast<uint16_t>();
   		const uint32_t t1=1,t31=31;
		ShiftLeft(a_h,a_h,t1,TILE_SIZE);
		ShiftRight(a_h,a_h,t1,TILE_SIZE);
		ShiftRight(b_h,b_h,t31,TILE_SIZE);
		ShiftLeft(b_h,b_h,t31,TILE_SIZE);
		Or(c_h2,a_h2,b_h2,2*TILE_SIZE);
	}
	
	__aicore__ inline void Compute() {
		auto x1 = inX1.DeQue<T>();
		auto x2 = inX2.DeQue<T>();
		auto y = outY.AllocTensor<T>();
		if constexpr(std::is_same<T, float>::value) {
			GcdLikely32(y, x1, x2);  
		} else {
			GcdLikely16(y, x1, x2);  
		}
		outY.EnQue(y);
		inX1.FreeTensor(x1);
		inX2.FreeTensor(x2);
	}
	
	__aicore__ inline void ProcessFast()
	{
		for (int64_t i = GetBlockIdx() * TILE_SIZE;i < sizeX1;i+=TILE_SIZE * GetBlockNum()) {
			CopyInX1(i, MIN(sizeX1 - i, TILE_SIZE));
			CopyInX2(i, MIN(sizeX1 - i, TILE_SIZE));
			Compute();
			CopyOut(i, MIN(sizeX1 - i, TILE_SIZE));
		}
	}
	
	__aicore__ inline void Process() {
		ProcessFast();
	}
	
	TQue<QuePosition::VECIN, 1> inX1, inX2;
	TQue<QuePosition::VECOUT, 1> outY;
	TBuf<TPosition::VECCALC> tBufNext, tBufMask, tBufFL;
	
	int64_t N[4], sizeX1, sizeX2;
	GlobalTensor<T> x1Gm, x2Gm, yGm;
};

#define TILE_SIZE2 1200
template<typename T> class LcmKernalFastBroadCast {
public:
	__aicore__ inline LcmKernalFastBroadCast() {}
	__aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint64_t size, uint64_t length, uint32_t n1[4], uint32_t n2[4],TPipe *pipe) {
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
		uint64_t total2 = 1;
		for (int i = 0; i < 4; ++i) {
			this->n1[i] = n1[i];
			this->n2[i] = n2[i];
			//		this->ny[i] = ny[i];
			total2 *= n2[i];
		}
		x1Gm.SetGlobalBuffer((__gm__ T*)x1 + L, length);
		x2Gm.SetGlobalBuffer((__gm__ T*)x2 , total2);
		yGm.SetGlobalBuffer((__gm__ T*)y + L, length);
		pipe->InitBuffer(inX1, 2, TILE_SIZE2 * sizeof(T));
		pipe->InitBuffer(inX2, 2, TILE_SIZE2 * sizeof(T));
		pipe->InitBuffer(outY, 2, TILE_SIZE2 * sizeof(T));
		//pipe->InitBuffer(tBufNext, 2 * TILE_SIZE2 * sizeof(int64_t));
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
	__aicore__ inline void GcdLikely16(const LocalTensor<T>& c, const LocalTensor<T>& a, const LocalTensor<T>& b) {
		auto a_h = a.template ReinterpretCast<uint16_t>();
		auto b_h = b.template ReinterpretCast<uint16_t>();
		auto c_h = c.template ReinterpretCast<uint16_t>();
		const uint16_t t1=1,t15=15;
		ShiftLeft(a_h,a_h,t1,TILE_SIZE2);
		ShiftRight(a_h,a_h,t1,TILE_SIZE2);
		ShiftRight(b_h,b_h,t15,TILE_SIZE2);
		ShiftLeft(b_h,b_h,t15,TILE_SIZE2);
		Or(c_h,a_h,b_h,TILE_SIZE2);
	}

	__aicore__ inline void GcdLikely32(const LocalTensor<T>& c, const LocalTensor<T>& a, const LocalTensor<T>& b) {
		auto a_h = a.template ReinterpretCast<uint32_t>();
		auto b_h = b.template ReinterpretCast<uint32_t>();
		auto c_h = c.template ReinterpretCast<uint32_t>();
		auto a_h2 = a.template ReinterpretCast<uint16_t>();
		auto b_h2 = b.template ReinterpretCast<uint16_t>();
		auto c_h2 = c.template ReinterpretCast<uint16_t>();
		const uint32_t t1=1,t31=31;
		ShiftLeft(a_h,a_h,t1,TILE_SIZE2);
		ShiftRight(a_h,a_h,t1,TILE_SIZE2);
		ShiftRight(b_h,b_h,t31,TILE_SIZE2);
		ShiftLeft(b_h,b_h,t31,TILE_SIZE2);
		Or(c_h2,a_h2,b_h2,2*TILE_SIZE2);
	}

	__aicore__ inline void Compute() {
		auto x1 = inX1.DeQue<T>();
		auto x2 = inX2.DeQue<T>();
		auto y = outY.AllocTensor<T>();
		if constexpr(std::is_same<T, float>::value) {
			GcdLikely32(y, x1, x2);
		} else {
			GcdLikely16(y, x1, x2);
		}
		outY.EnQue(y);
		inX1.FreeTensor(x1);
		inX2.FreeTensor(x2);
	}
	__aicore__ inline void Process() {
		/*
		uint64_t idx1 = L + addid;
		uint64_t i3 = idx1 % n1[3];
		uint64_t temp = idx1 / n1[3];
		uint64_t i2 = temp % n1[2];
		temp /= n1[2];
		uint64_t i1 = temp % n1[1];
		uint64_t i0 = (temp / n1[1]) % n1[0];
		uint64_t j0 = i0 % n2[0];
		uint64_t j1 = i1 % n2[1];
		uint64_t j2 = i2 % n2[2];
		uint64_t j3 = i3 % n2[3];

		uint64_t s0 = n2[1] * n2[2] * n2[3];
		uint64_t s1 = n2[2] * n2[3];
		uint64_t s2 = n2[3];
		#define s3 1
		uint64_t idx2 = j0 * s0 + j1 * s1 + j2 * s2 + j3 * s3;
*/

		uint64_t idx1 = L + addid;
		uint64_t i2 = idx1 % n1[2];
		uint64_t temp = idx1 / n1[2];
		uint64_t i1 = temp % n1[1];
		uint64_t i0 = (temp / n1[1]) % n1[0];
		uint64_t j0=i0%n2[0];
		uint64_t j1=i1%n2[1];
		uint64_t j2=i2%n2[2];
		uint64_t s0=n2[1]*n2[2];
		#define s1 n2[2]
		#define s2 1
		uint32_t idx2=j0*s0+j1*s1+j2;
		for (uint64_t i = L; i < R; ) {
			uint64_t sz = MIN(R - i, n1[2] - i2);
			CopyInX1(i, sz);
			CopyInX2(idx2, sz);
			Compute();
			CopyOut(i, sz);
		/*	i2++;
			i+=sz;
			idx2+=sz;
			i3=0;
			if(i2==n1[2])
			{
				i2=0;j2=0;
				i1++;
				if(i1==n1[1])
				{
					i1=0;j1=0;
					i0++;j0++;
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
			*/

			i1++;
			i+=sz;
			idx2+=sz;
			i2=0;
			if(i1==n1[1])
			{
				i1=0;j1=0;
				i0++;j0++;
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
		//AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
		//DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(yGm);
	}

private:
	GlobalTensor<T> x1Gm, x2Gm, yGm;
	uint64_t n1[4], n2[4];
	uint64_t L, R, addid;
	TQue<QuePosition::VECIN, 1> inX1, inX2;
	TQue<QuePosition::VECOUT, 1> outY;
	//TBuf<TPosition::VECCALC> tBufNext;
};
extern "C" __global__ __aicore__ void copysign(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
	GET_TILING_DATA(tiling_data, tiling);
//	if (tiling_data.status == 0) {
//		BruteForce<DTYPE_X1> op;
//		op.Init(x1, x2, y, tiling_data.n1, tiling_data.n2);
//		op.Process();
//	}
	 if(tiling_data.status == 1)
	{
		LcmKernalFastBroadCast<DTYPE_X1> op;
		TPipe pipe;
		op.Init(x1, x2, y, tiling_data.size, tiling_data.length, tiling_data.n1, tiling_data.n2,&pipe);
		op.Process();
	}
	/*
	else {
		LeftShiftKernalFast<DTYPE_X1> op;
		op.Init(x1, x2, y, tiling_data.size, tiling_data.length);
		op.Process();
	}*/
	else
	{
		KernelGcd<DTYPE_X1> op;
		TPipe pipe;
		op.Init(
			tiling_data.n1[0], tiling_data.n1[1], tiling_data.n1[2], tiling_data.n1[3], &pipe,
			x1, x2, y
			);
		op.Process();
	}/*
	else
	{
		KernelGcd<int32_t> op;
		TPipe pipe;
		op.Init(
			tiling_data.n1[0], tiling_data.n1[1], tiling_data.n1[2], &pipe,
			x1, x2, y
			);
		op.Process();
	}*/
}
