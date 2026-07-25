#include "kernel_operator.h"
using namespace AscendC;

constexpr uint64_t pre[65] = {0, 1ull,3ull,5ull,11ull,17ull,39ull,65ull,139ull,261ull,531ull,1025ull,2095ull,4097ull,8259ull,16405ull,32907ull,65537ull,131367ull,262145ull,524827ull,1048645ull,2098179ull,4194305ull,8390831ull,16777233ull,33558531ull,67109125ull,134225995ull,268435457ull,536887863ull,1073741825ull,2147516555ull,4294968325ull,8590000131ull,17179869265ull,34359871791ull,68719476737ull,137439215619ull,274877911045ull,549756338843ull,1099511627777ull,2199024312423ull,4398046511105ull,8796095120395ull,17592186061077ull,35184376283139ull,70368744177665ull,140737496778927ull,281474976710721ull,562949970199059ull,1125899906908165ull,2251799847243787ull,4503599627370497ull,9007199321981223ull,18014398509483025ull,36028797153190091ull,72057594038190085ull,144115188344291331ull,288230376151711745ull,576460752840837695ull,1152921504606846977ull,2305843010287435779ull,4611686018428436805ull,9223372039002292363ull};


#define ny n1
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define REP 16
#define TILE_SIZE 4096
template<typename T>
__aicore__ inline T gcd_2(T a, T b) {
	while (b != 0) {
		T temp = b;
		b = a % b;
		a = temp;
	}
	return a;
}

template<typename T> class BruteForce {
public:
	__aicore__ inline BruteForce() {}
	__aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t n1[3], uint32_t n2[3]) {
		//ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
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
					int64_t a = x1Gm.GetValue(idx1);
					int64_t b = x2Gm.GetValue(idx2);
					if (a==0 || b==0)
					{
						yGm.SetValue(idxy, 0);
						continue;
					}
					if (a < 0) {
						a = -a;
					}
					if (b < 0) {
						b = -b;
					}
					int64_t aa = a;
					int64_t bb = b;
					while (b) {
						int64_t A = b;
						int64_t B = a % b;
						a = A;
						b = B;
					}
					aa = aa/a*bb;
					if(aa<0)aa=-aa;
					yGm.SetValue(idxy, aa);
				}
			}
		}
	}
	
private:
	GlobalTensor<T> x1Gm, x2Gm, yGm;
	uint32_t n1[3], n2[3];
};
template<typename T> class LcmKernalFastBroadCast {
public:
	__aicore__ inline LcmKernalFastBroadCast() {}
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
			T a = x1Gm.GetValue(i);// 步骤1：从 idx1 反推 (i0, i1, i2)
			//uint32_t idx1 = i + addid;
			//uint32_t i2 = idx1 % n1[2];
			//uint32_t temp = idx1 / n1[2];
			//uint32_t i1 = temp % n1[1];
			//uint32_t i0 = (temp / n1[1]) % n1[0];
			//uint32_t idx2 = (i0 % n2[0]) * (n2[1] * n2[2]) + (i1 % n2[1]) * n2[2] + (i2 % n2[2]);
			
			T b = x2Gm.GetValue(idx2);
			a = (a > 0 ? a : -a);
			b = (b > 0 ? b : -b);
			if (a == 0||b == 0){
				yGm.SetValue(i, 0);
			}
			else {
				
				T aa = a;
				T bb = b;
				T shift = ScalarGetSFFValue<1>(a | b);
				a >>= ScalarGetSFFValue<1>(a);
				do {
					b >>= ScalarGetSFFValue<1>(b);
					if(a <= 64 && b <= 64){
						a = 64 - ScalarCountLeadingZero(pre[a] & pre[b]);
						break;
					}
					if (a > b) {
						a ^= b ^= a ^= b;
					}
					b -= a;
				} while (b);
				a <<= shift;
				aa /= a;
				aa *= bb;
				if(aa<0)aa=-aa;
				yGm.SetValue(i, aa);
			}
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
template<typename T> class LcmKernalFast {
public:
	__aicore__ inline LcmKernalFast() {}
	__aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t size, uint32_t length) {
		//ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
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
			a = (a > 0 ? a : -a);
			b = (b > 0 ? b : -b);
			if (a == 0){
				yGm.SetValue(i, 0);
			}
			else if (b == 0) {
				yGm.SetValue(i, 0);
			}
			else {
				T aa = a;
				T bb = b;
				T shift = ScalarGetSFFValue<1>(a | b);
				a >>= ScalarGetSFFValue<1>(a);
				do {
					b >>= ScalarGetSFFValue<1>(b);
					if(a <= 64 && b <= 64){
						a = 64 - ScalarCountLeadingZero(pre[a] & pre[b]);
						break;
					}
					if (a > b) {
						a ^= b ^= a ^= b;
					}
					b -= a;
				} while (b);
				a <<= shift;
				aa /= a;
				aa *= bb;
				if(aa<0)aa=-aa;
				
				yGm.SetValue(i, aa);
			}
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
		
		if constexpr(std::is_same<T, int16_t>::value) {
			pipe->InitBuffer(tBufNext, 3 * TILE_SIZE * sizeof(int16_t));
			//pipe->InitBuffer(tBufFL, 2 * TILE_SIZE * sizeof(float));
			pipe->InitBuffer(tBufMask, TILE_SIZE * sizeof(uint8_t));
			pipe->InitBuffer(tBufFL, 2 * TILE_SIZE * sizeof(float));
		}
		
		if constexpr(std::is_same<T, int32_t>::value) {
			pipe->InitBuffer(tBufNext, 2 * TILE_SIZE * sizeof(int32_t));
			pipe->InitBuffer(tBufFL, 3 * TILE_SIZE * sizeof(float));
			pipe->InitBuffer(tBufMask, TILE_SIZE * sizeof(uint8_t));
		}
		
		pipe->InitBuffer(inX1, 1, TILE_SIZE * sizeof(T));
		pipe->InitBuffer(inX2, 1, TILE_SIZE * sizeof(T));
		pipe->InitBuffer(outY, 1, TILE_SIZE * sizeof(T));
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
	
// 原GCD代码（完全不动）
	__aicore__ inline void GcdLikely16(const LocalTensor<int16_t>& c, const LocalTensor<int16_t>& a, const LocalTensor<int16_t>& b) {
		auto n_a = tBufNext.Get<int16_t>();
		auto n_b = tBufNext.Get<int16_t>()[TILE_SIZE];
		auto mask = tBufMask.Get<uint8_t>();
		auto a_h = a.ReinterpretCast<half>();
		auto b_h = b.ReinterpretCast<half>();
		auto n_a_h = n_a.ReinterpretCast<half>();
		auto n_b_h = n_b.ReinterpretCast<half>();
		
		Not(n_a, a, TILE_SIZE);
		Not(n_b, b, TILE_SIZE);
		Adds(n_a, n_a, (int16_t) 1, TILE_SIZE);
		Adds(n_b, n_b, (int16_t) 1, TILE_SIZE);
		Max(a, a, n_a, TILE_SIZE);
		Max(b, b, n_b, TILE_SIZE);
		
		auto a_abs=tBufFL.Get<float>();
		auto b_abs=tBufNext.Get<int16_t>()[2 * TILE_SIZE];
		auto c_h=tBufFL.Get<float>()[TILE_SIZE];
		
		Cast(a_abs, a, AscendC::RoundMode::CAST_NONE, TILE_SIZE);
		DataCopy(b_abs, b, TILE_SIZE);
		
		for (int i = 0;i < 50;i++) {
			Min(n_a, a, b, TILE_SIZE);
			Max(n_b, a, b, TILE_SIZE);
			Sub(n_b, n_b, n_a, TILE_SIZE);
			CompareScalar(mask, b_h, (half)0.0f, CMPMODE::EQ, TILE_SIZE);
			Select(a_h, mask, a_h, n_a_h, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
			Select(b_h, mask, b_h, n_b_h, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
		}
		
		Max(n_a, a, b, TILE_SIZE);
		Min(n_b, a, b, TILE_SIZE);
		DataCopy(a, n_a, TILE_SIZE);
		DataCopy(b, n_b, TILE_SIZE);
		
		int16_t tmp = 1;
		half* tmp_h = (half*) &tmp;
		CompareScalar(mask, b_h, *tmp_h, CMPMODE::NE, TILE_SIZE);
		Select(a_h, mask, a_h, *tmp_h, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_SIZE);
		Select(b_h, mask, b_h, (half)0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_SIZE);
		
		DataCopy(c, a, TILE_SIZE);
		
		CompareScalar(mask, b_h, (half)0.0f, CMPMODE::EQ, TILE_SIZE);
		
		for (int i = 0;i < TILE_SIZE;i+=8) {
			uint8_t bit = mask.GetValue(i / 8);
			if (bit == 0) continue;
			for (int j = 0;j < 8;j++) {
				if ((bit & (1<<j)) == 0) {
					auto A=a.GetValue(i + j);
					auto B=b.GetValue(i + j);
					A=gcd_2(A, B);
					c.SetValue(i + j, A);
				}
			}
		}
		const int16_t ONE=1;
		//Cast(c,a_abs, AscendC::RoundMode::CAST_RINT, TILE_SIZE);
		//Cast(c_h,c,static_cast<AscendC::RoundMode>(0), TILE_SIZE);
		Maxs(c,c,ONE,TILE_SIZE);
		Cast(c_h,c,AscendC::RoundMode::CAST_NONE, TILE_SIZE);
		Div(c_h,a_abs,c_h,TILE_SIZE);
		Cast(c,c_h,AscendC::RoundMode::CAST_RINT, TILE_SIZE);
		Mul(c,c,b_abs,TILE_SIZE);
		/*Not(n_a, c, TILE_SIZE);
		Adds(n_a, n_a, (int16_t) 1, TILE_SIZE);
		Max(c, c, n_a, TILE_SIZE);
		auto n_a = tBufNext.Get<int16_t>();
		auto n_b = tBufNext.Get<int16_t>()[TILE_SIZE];
		auto mask = tBufMask.Get<uint8_t>();
		auto mask1 = tBufMask.Get<uint8_t>()[TILE_SIZE];
		auto a_h = a.ReinterpretCast<half>();
		auto b_h = b.ReinterpretCast<half>();
		auto n_a_h = n_a.ReinterpretCast<half>();
		auto n_b_h = n_b.ReinterpretCast<half>();
		
		Not(n_a, a, TILE_SIZE);
		Not(n_b, b, TILE_SIZE);
		Adds(n_a, n_a, (int16_t) 1, TILE_SIZE);
		Adds(n_b, n_b, (int16_t) 1, TILE_SIZE);
		Max(a, a, n_a, TILE_SIZE);
		Max(b, b, n_b, TILE_SIZE);
		
		for (int i = 0;i < REP;i++) {
		Min(n_a, a, b, TILE_SIZE);
		Max(n_b, a, b, TILE_SIZE);
		Sub(n_b, n_b, n_a, TILE_SIZE);
		CompareScalar(mask, b_h, (half)0.0f, CMPMODE::EQ, TILE_SIZE);
		Select(a_h, mask, a_h, n_a_h, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
		Select(b_h, mask, b_h, n_b_h, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
		}
		
		Max(n_a, a, b, TILE_SIZE);
		Min(n_b, a, b, TILE_SIZE);
		DataCopy(a, n_a, TILE_SIZE);
		DataCopy(b, n_b, TILE_SIZE);
		
		int16_t tmp = 1;
		half* tmp_h = (half*) &tmp;
		CompareScalar(mask, b_h, *tmp_h, CMPMODE::NE, TILE_SIZE);
		Select(a_h, mask, a_h, *tmp_h, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_SIZE);
		Select(b_h, mask, b_h, (half)0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_SIZE);
		
		DataCopy(c, a, TILE_SIZE);
		
		CompareScalar(mask, b_h, (half)0.0f, CMPMODE::EQ, TILE_SIZE);
		
		for (int i = 0;i < TILE_SIZE;i+=8) {
		uint8_t bit = mask.GetValue(i / 8);
		if (bit == 0) continue;
		for (int j = 0;j < 8;j++) {
		if ((bit & (1<<j)) == 0) {
		c.SetValue(i + j, gcd_2(a.GetValue(i + j), b.GetValue(i + j)));
		}
		}
		}*/
	}
	
	__aicore__ inline void GcdLikely32(const LocalTensor<int32_t>& c, const LocalTensor<int32_t>& a, const LocalTensor<int32_t>& b) {
		auto n_a = tBufNext.Get<int32_t>();
		auto n_b = tBufNext.Get<int32_t>()[TILE_SIZE];
		auto mask = tBufMask.Get<uint8_t>();
		
		Not(n_a.ReinterpretCast<uint16_t>(), a.ReinterpretCast<uint16_t>(), 2*TILE_SIZE);
		Not(n_b.ReinterpretCast<uint16_t>(), b.ReinterpretCast<uint16_t>(), 2*TILE_SIZE);
		Adds(n_a, n_a, (int32_t) 1, TILE_SIZE);
		Adds(n_b, n_b, (int32_t) 1, TILE_SIZE);
		Max(a, a, n_a, TILE_SIZE);
		Max(b, b, n_b, TILE_SIZE);
		
		auto a_abs=tBufFL.Get<float>();
		auto b_abs=tBufFL.Get<float>()[TILE_SIZE];
		auto ans=tBufFL.Get<float>()[2 * TILE_SIZE];
		
		//DataCopy(a_abs, a, TILE_SIZE);
		//DataCopy(b_abs, b, TILE_SIZE);
		Cast(a_abs, a, AscendC::RoundMode::CAST_NONE, TILE_SIZE);
		Cast(b_abs, b, AscendC::RoundMode::CAST_NONE, TILE_SIZE);
		
		//	Cast(c, a_abs, AscendC::RoundMode::CAST_RINT, TILE_SIZE);
		
		for (int i = 0;i < REP;i++) {
			
			if(i&1)//b%a
			{
				CompareScalar(mask, a_abs, (float)0.0f, CMPMODE::EQ, TILE_SIZE);
				Select(a_abs,mask,b_abs,a_abs, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,TILE_SIZE);
				//Select(ans, mask, b_abs, ans, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
				Fmod(ans,b_abs,a_abs,TILE_SIZE);
				//Select(b_abs,mask,a_abs,b_abs,TILE_SIZE);
				DataCopy(b_abs,ans,TILE_SIZE);
			}
			else
			{
				CompareScalar(mask, b_abs, (float)0.0f, CMPMODE::EQ, TILE_SIZE);
				Select(b_abs,mask,a_abs,b_abs, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,TILE_SIZE);
				//Select(ans, mask, a_abs, ans, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
				Fmod(ans,a_abs,b_abs,TILE_SIZE);
				DataCopy(a_abs,ans,TILE_SIZE);
			}
		}
		//Fmod(ans,a_abs,b_abs,TILE_SIZE);
		Cast(ans, a, AscendC::RoundMode::CAST_NONE, TILE_SIZE);
		//const int32_t ONE=1;
		//Maxs(c,c,ONE,TILE_SIZE);
		Maxs(a_abs,a_abs,1.0f,TILE_SIZE);
		//Cast(b_abs, c, AscendC::RoundMode::CAST_NONE, TILE_SIZE);
		Div(ans, ans, a_abs,TILE_SIZE);
		Cast(c, ans, AscendC::RoundMode::CAST_RINT, TILE_SIZE);
		Mul(c,c,b,TILE_SIZE);
//		Not(n_a.ReinterpretCast<uint16_t>(), c.ReinterpretCast<uint16_t>(), 2 * TILE_SIZE);
//		Adds(n_a, n_a, (int32_t) 1, TILE_SIZE);
//		Max(c, c, n_a, TILE_SIZE);
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
	
	int64_t N[3], sizeX1, sizeX2;
	GlobalTensor<T> x1Gm, x2Gm, yGm;
};
#define TILE_SIZE2 1024
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
		for (int i = 0; i < 3; ++i) {
			this->n1[i] = n1[i];
			this->n2[i] = n2[i];
			//		this->ny[i] = ny[i];
			total2 *= n2[i];
		}
		x1Gm.SetGlobalBuffer((__gm__ T*)x1 + L, length);
		x2Gm.SetGlobalBuffer((__gm__ T*)x2 , total2);
		yGm.SetGlobalBuffer((__gm__ T*)y + L, length);
		pipe->InitBuffer(inX1, 1, TILE_SIZE2 * sizeof(T));
		pipe->InitBuffer(inX2, 1, TILE_SIZE2 * sizeof(T));
		pipe->InitBuffer(outY, 1, TILE_SIZE2 * sizeof(T));
		pipe->InitBuffer(tBufNext, 4 * TILE_SIZE2 * sizeof(int32_t));
		pipe->InitBuffer(tBufFL, 3 * TILE_SIZE2 * sizeof(float));
		pipe->InitBuffer(tBufMask, TILE_SIZE2 * sizeof(uint8_t));
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
	__aicore__ inline void Compute(uint32_t len){
		auto x1 = inX1.DeQue<T>();
		auto x2 = inX2.DeQue<T>();
		auto y = outY.AllocTensor<T>();
		if(len<=50)
		{
		for(uint32_t i=0;i<len;i++)
		{
			T a = x1.GetValue(i);
			T b = x2.GetValue(i);

			a = (a > 0 ? a : -a);
			b = (b > 0 ? b : -b);
			if (a == 0||b == 0){
				y.SetValue(i, 0);
			}
			else {
				T aa = a;
				T bb = b;
				T shift = ScalarGetSFFValue<1>(a | b);
				a >>= ScalarGetSFFValue<1>(a);
				do {
					b >>= ScalarGetSFFValue<1>(b);
					if(a <= 64 && b <= 64){
						a = 64 - ScalarCountLeadingZero(pre[a] & pre[b]);
						break;
					}
					if (a > b) {
						a ^= b ^= a ^= b;
					}
					b -= a;
				} while (b);
				a <<= shift;
				aa /= a;
				aa *= bb;
				if(aa<0)aa=-aa;
				y.SetValue(i, aa);
			}
		}
		}
		else
		{
			const int64_t lim=(1ll<<24);
			auto A = tBufNext.Get<int32_t>()[2*TILE_SIZE2];
			auto B = tBufNext.Get<int32_t>()[3*TILE_SIZE2];
			for(uint32_t i=0;i<len;i++)
			{
				T a = x1.GetValue(i);
				T b = x2.GetValue(i);
//				ja[i]=a;
//				jb[i]=b;
				a = (a > 0 ? a : -a);
				b = (b > 0 ? b : -b);
				ja[i]=a;
				jb[i]=b;
				if (a == 0||b == 0){
					A.SetValue(i, 0);
					B.SetValue(i, 0);
				}
				else {
					T aa = a;
					T bb = b;
					T shift = ScalarGetSFFValue<1>(a | b);
					a >>= ScalarGetSFFValue<1>(a);
					do {
						b >>= ScalarGetSFFValue<1>(b);
						if(a < lim || b < lim){
							if(a>=lim)a%=b;
							if(b>=lim)b%=a;
						//	A.SetValue(i,(int32_t)a);
						//	B.SetValue(i,(int32_t)b);
							break;
						}
						if (a > b) {
							a ^= b ^= a ^= b;
						}
						b -= a;
					} while (b);
					if(b!=0)
					{
						A.SetValue(i,(int32_t)a);
						B.SetValue(i,(int32_t)b);
						ja[i]=(aa>>shift);
					}
					else
					{
						A.SetValue(i,1);
                                                B.SetValue(i,1);
                                                ja[i]=(aa>>shift)/a;
					}
					//a <<= shift;
					//aa /= a;
					//aa *= bb;
					//if(aa<0)aa=-aa;
					//y.SetValue(i, aa);
				}
			}
			
			{
				{
					auto n_a = tBufNext.Get<int32_t>();
					auto n_b = tBufNext.Get<int32_t>()[TILE_SIZE2];
					auto mask = tBufMask.Get<uint8_t>();
					
					//Not(n_a.ReinterpretCast<uint16_t>(), A.ReinterpretCast<uint16_t>(), 2*TILE_SIZE2);
					//Not(n_b.ReinterpretCast<uint16_t>(), B.ReinterpretCast<uint16_t>(), 2*TILE_SIZE2);
					//Adds(n_a, n_a, (int32_t) 1, TILE_SIZE2);
					//Adds(n_b, n_b, (int32_t) 1, TILE_SIZE2);
					//Max(A, A, n_a, TILE_SIZE2);
					//Max(B, B, n_b, TILE_SIZE2);
					
					auto a_abs=tBufFL.Get<float>();
					auto b_abs=tBufFL.Get<float>()[TILE_SIZE2];
					auto ans=tBufFL.Get<float>()[2 * TILE_SIZE2];
					
					//DataCopy(a_abs, a, TILE_SIZE);
					//DataCopy(b_abs, b, TILE_SIZE);
					Cast(a_abs, A, AscendC::RoundMode::CAST_NONE, TILE_SIZE2);
					Cast(b_abs, B, AscendC::RoundMode::CAST_NONE, TILE_SIZE2);
					
					//	Cast(c, a_abs, AscendC::RoundMode::CAST_RINT, TILE_SIZE);
					
					for (int i = 0;i < 24;i++) {
						
						if(i&1)//b%a
						{
							CompareScalar(mask, a_abs, (float)0.0f, CMPMODE::EQ, TILE_SIZE2);
							Select(a_abs,mask,b_abs,a_abs, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,TILE_SIZE2);
							//Select(ans, mask, b_abs, ans, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
							Fmod(ans,b_abs,a_abs,TILE_SIZE2);
							//Select(b_abs,mask,a_abs,b_abs,TILE_SIZE);
							DataCopy(b_abs,ans,TILE_SIZE2);
						}
						else
						{
							CompareScalar(mask, b_abs, (float)0.0f, CMPMODE::EQ, TILE_SIZE2);
							Select(b_abs,mask,a_abs,b_abs, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,TILE_SIZE2);
							//Select(ans, mask, a_abs, ans, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
							Fmod(ans,a_abs,b_abs,TILE_SIZE2);
							DataCopy(a_abs,ans,TILE_SIZE2);
						}
					}
					//Fmod(ans,a_abs,b_abs,TILE_SIZE);
					//Cast(ans, A, AscendC::RoundMode::CAST_NONE, TILE_SIZE);
					//const int32_t ONE=1;
					//Maxs(c,c,ONE,TILE_SIZE);
					Maxs(a_abs,a_abs,1.0f,TILE_SIZE2);
					//Cast(b_abs, c, AscendC::RoundMode::CAST_NONE, TILE_SIZE);
					//Div(ans, ans, a_abs,TILE_SIZE);
					Cast(A, a_abs, AscendC::RoundMode::CAST_RINT, TILE_SIZE2);
				}
				for(uint32_t i=0;i<len;i++)
				{
				//	T b = x2.GetValue(i);
					T c = A.GetValue(i);
					T a = ja[i]/c*jb[i];
					//if(b<0)b=-b;
					//a=a/c*b;
					if(a<0)a=-a;
					y.SetValue(i,a);
				}
			}
		}
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
			Compute(sz);
			CopyOut(i, sz);
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
	uint32_t n1[3], n2[3];
	uint32_t L, R, addid;
	TQue<QuePosition::VECIN, 1> inX1, inX2;
	TQue<QuePosition::VECOUT, 1> outY;
	TBuf<TPosition::VECCALC> tBufNext,tBufMask,tBufFL;
	T ja[TILE_SIZE2],jb[TILE_SIZE2];
};
extern "C" __global__ __aicore__ void lcm(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
	GET_TILING_DATA(tiling_data, tiling);
	if (tiling_data.status == 0) {
		BruteForce<DTYPE_X1> op;
		op.Init(x1, x2, y, tiling_data.n1, tiling_data.n2);
		op.Process();
	}
	else if(tiling_data.status == 1)
	{
		LcmKernalFastBroadCast<DTYPE_X1> op;
		op.Init(x1, x2, y, tiling_data.size, tiling_data.length, tiling_data.n1, tiling_data.n2);
		op.Process();
	}
	else if(tiling_data.status == 2){
		LcmKernalFast<DTYPE_X1> op;
		op.Init(x1, x2, y, tiling_data.size, tiling_data.length);
		op.Process();
	}
	else if(tiling_data.status==3)
	{
		KernelGcd<int16_t> op;
		TPipe pipe;
		op.Init(
			tiling_data.n1[0], tiling_data.n1[1], tiling_data.n1[2], &pipe,
			x1, x2, y
			);
		op.Process();
	}
	else if(tiling_data.status==4)
	{
		KernelGcd<int32_t> op;
		TPipe pipe;
		op.Init(
			tiling_data.n1[0], tiling_data.n1[1], tiling_data.n1[2], &pipe,
			x1, x2, y
			);
		op.Process();
	}
	else
	{
		LcmKernalFastBroadCast64<DTYPE_X1> op;
		TPipe pipe;
		op.Init(x1, x2, y, tiling_data.size, tiling_data.length, tiling_data.n1, tiling_data.n2,&pipe);
		op.Process();
	}
}
