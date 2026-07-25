

#include "kernel_operator.h"  
#include <type_traits>  

// 定义常量  
static constexpr float PI = 3.14159265358979323846f;  
static constexpr float TWO_PI = 2.0f * PI;  
static constexpr float PI_OVER_2 = PI / 2.0f;  
// 定义切比雪夫区间处理宏
// Calculate函数辅助信息
static constexpr int CHEB_ORDER = 7;
static constexpr float BOUNDARY_19 = 10.2f;  // 渐近计算阈值
static constexpr float C_CHEB = 1.9f;  
// t_min 和 t_max 数组
static constexpr float t_min_array[] = { 0.0000000000f, 0.0759878419f, 0.1412429379f, 0.1978891821f, 0.2475247525f, 0.2922550728f, 0.3319660906f, 0.3674573450f, 0.3993676922f, 0.4282132235f, 0.4544149318f, 0.4783205105f, 0.5002192459f, 0.5203534238f, 0.5389281730f, 0.5561180137f, 0.5720720721f, 0.5869190751f, 0.6007704710f, 0.6137229987f, 0.6258614786f, 0.6372603842f, 0.6479851783f, 0.6580940036f, 0.6676385033f, 0.6766645395f, 0.6852132920f, 0.6933216925f, 0.7010228167f, 0.7083466432f, 0.7153202789f, 0.7219681727f, 0.7283126658f, 0.7343741022f, 0.7401709402f, 0.7457201693f, 0.7510373553f, 0.7561366918f, 0.7610313284f, 0.7657333772f, 0.7702539299f, 0.7746033226f, 0.7787911214f, 0.7828261180f, 0.7867165495f, 0.7904700716f, 0.7940937415f, 0.7975942047f, 0.8009776606f, 0.8042498390f, 0.8074161604f, 0.8104816987f, 0.8134511537f, 0.8163289907f, 0.8191194028f, 0.8218262806f, 0.8244533361f, 0.8270040646f, 0.8294817142f, 0.8318893960f, 0.8342300470f, 0.8365063999f, 0.8387210819f, 0.8408765791f };
static constexpr float t_max_array[] = { 0.0759878419f, 0.1412429379f, 0.1978891821f, 0.2475247525f, 0.2922550728f, 0.3319660906f, 0.3674573450f, 0.3993676922f, 0.4282132235f, 0.4544149318f, 0.4783205105f, 0.5002192459f, 0.5203534238f, 0.5389281730f, 0.5561180137f, 0.5720720721f, 0.5869190751f, 0.6007704710f, 0.6137229987f, 0.6258614786f, 0.6372603842f, 0.6479851783f, 0.6580940036f, 0.6676385033f, 0.6766645395f, 0.6852132920f, 0.6933216925f, 0.7010228167f, 0.7083466432f, 0.7153202789f, 0.7219681727f, 0.7283126658f, 0.7343741022f, 0.7401709402f, 0.7457201693f, 0.7510373553f, 0.7561366918f, 0.7610313284f, 0.7657333772f, 0.7702539299f, 0.7746033226f, 0.7787911214f, 0.7828261180f, 0.7867165495f, 0.7904700716f, 0.7940937415f, 0.7975942047f, 0.8009776606f, 0.8042498390f, 0.8074161604f, 0.8104816987f, 0.8134511537f, 0.8163289907f, 0.8191194028f, 0.8218262806f, 0.8244533361f, 0.8270040646f, 0.8294817142f, 0.8318893960f, 0.8342300470f, 0.8365063999f, 0.8387210819f, 0.8408765791f, 0.8429752066f };


static constexpr float cheb_coef_0[] = { 0.0765761745f, 0.2326756206f, 0.3866436121f, 0.5326258241f, 0.6597280474f, 0.7482526890f, 0.7739860427f, 0.7213886892f, 0.5977733041f, 0.4465729994f, 0.3419555644f, 0.3503938362f, 0.4700562955f, 0.6018078063f, 0.6165317440f, 0.4976648069f, 0.3957812859f, 0.4531055653f, 0.5764264734f, 0.5545526964f, 0.4346643606f, 0.4604366605f, 0.5673196460f, 0.5062752114f, 0.4387062343f, 0.5371166585f, 0.5213443752f, 0.4485197188f, 0.5371515734f, 0.4987574038f, 0.4713243010f, 0.5401040006f, 0.4648555376f, 0.5218966185f, 0.4923836049f, 0.4962583522f, 0.5110965275f, 0.4850809487f, 0.5161328171f, 0.4844786180f, 0.5135321843f, 0.4896795830f, 0.5059234313f, 0.4994852107f, 0.4946936901f, 0.5103161649f, 0.4872126578f, 0.5111783042f, 0.4946418271f, 0.4976654144f, 0.5075369946f, 0.4933065393f, 0.5007468152f, 0.5044901610f, 0.4961450459f, 0.4992072288f, 0.5028147673f, 0.4994928516f, 0.4988453980f, 0.5004113222f, 0.4997727453f, 0.4995997024f, 0.5013106510f, 0.5015145028f };

static constexpr float cheb_coef_1[] = { 0.0780853004f, 0.0777632370f, 0.0757530484f, 0.0694403826f, 0.0564123294f, 0.0306110860f, -0.0060779169f, -0.0464905089f, -0.0749651266f, -0.0720118710f, -0.0283631855f, 0.0373574355f, 0.0768272565f, 0.0467886222f, -0.0339451418f, -0.0757951173f, -0.0161433783f, 0.0681238599f, 0.0402142431f, -0.0598294223f, -0.0417078969f, 0.0636999253f, 0.0226584867f, -0.0712335568f, 0.0198740643f, 0.0531209798f, -0.0647140492f, 0.0150675347f, 0.0428067966f, -0.0666457827f, 0.0507646021f, -0.0136256334f, -0.0236227399f, 0.0489571155f, -0.0602672154f, 0.0611235720f, -0.0565649901f, 0.0507621394f, -0.0463406165f, 0.0445376775f, -0.0455308985f, 0.0486031660f, -0.0520867463f, 0.0532557658f, -0.0485353046f, 0.0345613346f, -0.0104710177f, -0.0190829686f, 0.0424300771f, -0.0446654913f, 0.0187776968f, 0.0215582537f, -0.0432132417f, 0.0213327248f, 0.0245082908f, -0.0377920795f, -0.0032121549f, 0.0374344348f, -0.0056087196f, -0.0348285062f, 0.0024728472f, 0.0328226601f, 0.0102928083f, -0.0244064166f };

static constexpr float cheb_coef_2[] = { 0.0015373079f, 0.0013224712f, 0.0008418230f, -0.0000828337f, -0.0016142705f, -0.0035297176f, -0.0051544307f, -0.0053359965f, -0.0029187683f, 0.0021018160f, 0.0073184568f, 0.0082366504f, 0.0018890423f, -0.0078483411f, -0.0103333039f, 0.0003151975f, 0.0122046497f, 0.0061439166f, -0.0115061073f, -0.0090939238f, 0.0123678252f, 0.0081777644f, -0.0157164891f, -0.0014521065f, 0.0173738463f, -0.0117680507f, -0.0071364025f, 0.0193602300f, -0.0155011038f, 0.0008392238f, 0.0139046706f, -0.0215042704f, 0.0207419092f, -0.0142873611f, 0.0056949492f, 0.0024064048f, -0.0087361348f, 0.0130141297f, -0.0154262167f, 0.0161997819f, -0.0153603451f, 0.0126542531f, -0.0076460104f, 0.0000318600f, 0.0097900542f, -0.0199855631f, 0.0268464922f, -0.0255344366f, 0.0129032381f, 0.0080573176f, -0.0256769775f, 0.0248068623f, -0.0015216434f, -0.0246041113f, 0.0229532641f, 0.0090656333f, -0.0286057450f, 0.0021632812f, 0.0283679186f, -0.0043575519f, -0.0285589508f, -0.0025766353f, 0.0269244868f, 0.0172421063f };

static constexpr float cheb_coef_3[] = { 0.0000282836f, 0.0000061475f, -0.0000371750f, -0.0001006381f, -0.0001825420f, -0.0002288552f, -0.0001879894f, -0.0000115474f, 0.0002762905f, 0.0005137468f, 0.0004340556f, -0.0001011978f, -0.0007637559f, -0.0007800902f, 0.0002323562f, 0.0012469662f, 0.0005726576f, -0.0012962625f, -0.0011819127f, 0.0013881040f, 0.0014487315f, -0.0018963406f, -0.0010797006f, 0.0027355503f, -0.0005535999f, -0.0026382640f, 0.0031305110f, -0.0005263145f, -0.0027363323f, 0.0042264511f, -0.0032638965f, 0.0007093574f, 0.0021205124f, -0.0042886130f, 0.0054767399f, -0.0058257524f, 0.0056751796f, -0.0053737872f, 0.0051919304f, -0.0053025483f, 0.0057831904f, -0.0066011239f, 0.0075676271f, -0.0082755266f, 0.0080733033f, -0.0061919261f, 0.0021592874f, 0.0034936505f, -0.0086046779f, 0.0097446002f, -0.0044924785f, -0.0050199197f, 0.0109928921f, -0.0058993819f, -0.0067893713f, 0.0113460054f, 0.0008366955f, -0.0124212669f, 0.0021230264f, 0.0128460390f, -0.0011071936f, -0.0135143247f, -0.0043508367f, 0.0112633034f };

static constexpr float cheb_coef_4[] = { 0.0000000348f, -0.0000018427f, -0.0000043452f, -0.0000068145f, -0.0000085465f, -0.0000057901f, 0.0000026386f, 0.0000151656f, 0.0000240371f, 0.0000168592f, -0.0000122766f, -0.0000457138f, -0.0000418086f, 0.0000216471f, 0.0000857302f, 0.0000408352f, -0.0000968760f, -0.0001067681f, 0.0000999798f, 0.0001603503f, -0.0001380416f, -0.0001823093f, 0.0002457381f, 0.0000977540f, -0.0003740624f, 0.0002014095f, 0.0002468020f, -0.0005207855f, 0.0003784755f, 0.0000649698f, -0.0005212996f, 0.0007645140f, -0.0007308291f, 0.0004847987f, -0.0001376766f, -0.0002153212f, 0.0005187908f, -0.0007525632f, 0.0009151239f, -0.0010062493f, 0.0010136302f, -0.0009047164f, 0.0006278808f, -0.0001302037f, -0.0005989152f, 0.0014558119f, -0.0021524672f, 0.0022302080f, -0.0012763271f, -0.0006219140f, 0.0024635693f, -0.0026197050f, 0.0003074309f, 0.0026943214f, -0.0027863649f, -0.0009735700f, 0.0036813866f, -0.0004386478f, -0.0039391156f, 0.0007839842f, 0.0043264041f, 0.0002563960f, -0.0044899473f, -0.0028405464f };

static constexpr float cheb_coef_5[] = { -0.0000000727f, -0.0000001663f, -0.0000002511f, -0.0000002770f, -0.0000001903f, 0.0000001389f, 0.0000006170f, 0.0000009431f, 0.0000006216f, -0.0000006130f, -0.0000020637f, -0.0000019084f, 0.0000010707f, 0.0000044353f, 0.0000025524f, -0.0000051290f, -0.0000072933f, 0.0000046310f, 0.0000123664f, -0.0000056487f, -0.0000175605f, 0.0000125135f, 0.0000191992f, -0.0000285803f, -0.0000041174f, 0.0000414585f, -0.0000388404f, -0.0000057826f, 0.0000561054f, -0.0000747875f, 0.0000501918f, 0.0000038656f, -0.0000639550f, 0.0001116518f, -0.0001394146f, 0.0001490830f, -0.0001475697f, 0.0001431636f, -0.0001434577f, 0.0001546242f, -0.0001810493f, 0.0002241263f, -0.0002794324f, 0.0003323318f, -0.0003537528f, 0.0003018420f, -0.0001386465f, -0.0001321519f, 0.0004181823f, -0.0005311524f, 0.0002912702f, 0.0002502733f, -0.0006635734f, 0.0004161741f, 0.0004056483f, -0.0007954072f, -0.0000049482f, 0.0009189765f, -0.0002215753f, -0.0010231130f, 0.0001535539f, 0.0011773218f, 0.0003307183f, -0.0010918857f };

static constexpr float cheb_coef_6[] = { -0.0000000061f, -0.0000000090f, -0.0000000099f, -0.0000000063f, 0.0000000043f, 0.0000000213f, 0.0000000339f, 0.0000000240f, -0.0000000202f, -0.0000000775f, -0.0000000803f, 0.0000000316f, 0.0000001848f, 0.0000001420f, -0.0000001969f, -0.0000003934f, 0.0000001109f, 0.0000007018f, -0.0000000290f, -0.0000011273f, 0.0000002028f, 0.0000016705f, -0.0000011235f, -0.0000017252f, 0.0000030441f, -0.0000005406f, -0.0000036141f, 0.0000053308f, -0.0000026674f, -0.0000028824f, 0.0000081043f, -0.0000104221f, 0.0000090462f, -0.0000047187f, -0.0000011704f, 0.0000073592f, -0.0000130697f, 0.0000179971f, -0.0000220910f, 0.0000252618f, -0.0000270803f, 0.0000264886f, -0.0000216495f, 0.0000102131f, 0.0000095680f, -0.0000364277f, 0.0000628475f, -0.0000734744f, 0.0000501983f, 0.0000119046f, -0.0000839980f, 0.0001033138f, -0.0000235935f, -0.0001032338f, 0.0001255241f, 0.0000299931f, -0.0001695856f, 0.0000356998f, 0.0001920705f, -0.0000562960f, -0.0002287824f, 0.0000035870f, 0.0002625615f, 0.0001527631f };



template<typename TYPE_X, typename TYPE_Y> 
class FresnelCos {  
    static constexpr int32_t BUFFER_NUM = 2;  

public:  
    __aicore__ inline FresnelCos() {}  
    __aicore__ inline void PreloadLUT() {
        auto tmin = B_lut_tmin.Get<float>();
        auto tmax = B_lut_tmax.Get<float>();
        auto c0 = B_lut_c0.Get<float>();
        auto c1 = B_lut_c1.Get<float>();
        auto c2 = B_lut_c2.Get<float>();
        auto c3 = B_lut_c3.Get<float>();
        auto c4 = B_lut_c4.Get<float>();
        auto c5 = B_lut_c5.Get<float>();
        auto c6 = B_lut_c6.Get<float>();

        #pragma unroll
        for (int i = 0; i < 64; ++i) {
        tmin.SetValue(i, t_min_array[i]);
        tmax.SetValue(i, t_max_array[i]);
        c0.SetValue(i, cheb_coef_0[i]);
        c1.SetValue(i, cheb_coef_1[i]);
        c2.SetValue(i, cheb_coef_2[i]);
        c3.SetValue(i, cheb_coef_3[i]);
        c4.SetValue(i, cheb_coef_4[i]);
        c5.SetValue(i, cheb_coef_5[i]);
        c6.SetValue(i, cheb_coef_6[i]);
    }
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM,   
                                uint32_t block_size, uint32_t core_size, uint32_t core_remain) {  
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");  
        this->blockLength = core_size + (AscendC::GetBlockNum() == AscendC::GetBlockIdx() + 1 ? core_remain : 0);  
        this->tileLength = block_size;  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);  
        
        auto startPointer = core_size * AscendC::GetBlockIdx();  
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, this->blockLength);  
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, this->blockLength);  
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0 ? 1 : 0);  

        // 初始化缓冲区  
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));  
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));  
        
        // 优化后的缓冲区分配 - 减少到最小必要数量
        // 基础计算缓冲区 (4个)
        pipe.InitBuffer(B_tmp1, this->tileLength * sizeof(float)); // 多功能临时变量1  
        pipe.InitBuffer(B_inner1, this->tileLength * sizeof(float)); // 多功能临时变量1  
        pipe.InitBuffer(B_inner2, this->tileLength * sizeof(float)); // 多功能临时变量2  
        pipe.InitBuffer(B_inner3, this->tileLength * sizeof(float)); // 多功能临时变量3  
        pipe.InitBuffer(B_inner4, this->tileLength * sizeof(float)); // 多功能临时变量4  
        pipe.InitBuffer(B_inner5, this->tileLength * sizeof(float)); // 多功能临时变量5
	    pipe.InitBuffer(B_inner6, this->tileLength * sizeof(float)); //


        pipe.InitBuffer(B_tri1, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_tri2, this->tileLength * sizeof(float));
        // pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
        // 掩码缓冲区 (3个足够，通过时序复用)
        
        pipe.InitBuffer(B_mask1, this->tileLength * sizeof(uint16_t));  // 主掩码1
        pipe.InitBuffer(B_mask2, this->tileLength * sizeof(uint16_t));  // 主掩码2
        pipe.InitBuffer(B_mask3, this->tileLength * sizeof(int32_t));  // 主掩码2
        pipe.InitBuffer(B_mask4, this->tileLength * sizeof(uint32_t));
        pipe.InitBuffer(B_mask_temp, this->tileLength * sizeof(uint16_t)); // 临时掩码
        pipe.InitBuffer(B_c1, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_c2, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_c3, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_c4, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_c5, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_c6, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_c7, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_lut_tmin, 64 * sizeof(float));
        pipe.InitBuffer(B_lut_tmax, 64 * sizeof(float));
        pipe.InitBuffer(B_lut_c0, 64 * sizeof(float));
        pipe.InitBuffer(B_lut_c1, 64 * sizeof(float));
        pipe.InitBuffer(B_lut_c2, 64 * sizeof(float));
        pipe.InitBuffer(B_lut_c3, 64 * sizeof(float));
        pipe.InitBuffer(B_lut_c4, 64 * sizeof(float));
        pipe.InitBuffer(B_lut_c5, 64 * sizeof(float));
        pipe.InitBuffer(B_lut_c6, 64 * sizeof(float));
        // pipe.InitBuffer(B_tmin, this->tileLength * sizeof(float));
        // pipe.InitBuffer(B_tmax, this->tileLength * sizeof(float));
        // // 结果缓冲区 (2个，循环复用)
        pipe.InitBuffer(B_result1, this->tileLength * sizeof(float)); // 结果缓冲区1
        // pipe.InitBuffer(B_result2, this->tileLength * sizeof(float)); // 结果缓冲区2  
        
        // 类型转换缓冲区  
        if constexpr (!std::is_same_v<TYPE_X, float>) {  
            pipe.InitBuffer(B_fx, this->tileLength * sizeof(float));  
        }  
        if constexpr (!std::is_same_v<TYPE_Y, float>) {  
            pipe.InitBuffer(B_fy, this->tileLength * sizeof(float));  
        }  
    }  

    __aicore__ inline void Process() {  
        PreloadLUT();
        for (int32_t i = 0; i < static_cast<int32_t>(this->tileNum); i++) {  
            uint32_t length = (i == this->tileNum - 1) ? (this->blockLength - i * this->tileLength) : this->tileLength;  
            CopyIn(i, length);  
            Compute(i, length);  
            CopyOut(i, length);  
        }  
    }  

private:  
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {  
        AscendC::LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>();  
        AscendC::DataCopy(x, Gm_x[progress * this->tileLength], length);  
        Q_x.EnQue(x);  
    }  

    __aicore__ inline void Compute(int32_t progress, uint32_t length) {  
        AscendC::LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();  
        AscendC::LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();  

        AscendC::LocalTensor<float> fx = (std::is_same_v<TYPE_X, float>) ? x.template ReinterpretCast<float>() : B_fx.Get<float>();
        AscendC::LocalTensor<float> fy = (std::is_same_v<TYPE_Y, float>) ? y.template ReinterpretCast<float>() : B_fy.Get<float>();

        if constexpr (!std::is_same_v<TYPE_X, float>) {  
            AscendC::Cast(fx, x, AscendC::RoundMode::CAST_NONE, length);  
        }  

        Calculate(fx, fy, length);  

        if constexpr (!std::is_same_v<TYPE_Y, float>) {  
            AscendC::Cast(y, fy, AscendC::RoundMode::CAST_NONE, length);  
        }  

        Q_x.FreeTensor(x);  
        Q_y.EnQue<TYPE_Y>(y);  
    }  

    __aicore__ inline void Calculate(AscendC::LocalTensor<float> &x, AscendC::LocalTensor<float> &y, uint32_t length) {  
        // === 🎯 最优缓冲区策略：按顺序计算并立即合并，避免缓冲区冲突 ===
        
        // 🔒 基础数据（全程保护，只读）
        auto neg_mask = B_mask_temp.Get<uint16_t>();   // 负数掩码 - 全程保护
        AscendC::Duplicate(B_tmp1.Get<float>(), 0.0f, length);    // 常数0用于比较
        AscendC::Compare(neg_mask, x, B_tmp1.Get<float>(), AscendC::CMPMODE::LT, length); // 负数掩码
        auto x_abs = B_tmp1.Get<float>();              // |x| - 全程保护

        
        
        // 计算基础数据（一次性计算，后续只读）
        AscendC::Abs(x_abs, x, length);                           // |x|    
                auto near_zero_mask = B_mask4.Get<uint16_t>();
        // auto x_abs_modified = B_inner1.Get<float>();
        
        // 检测接近0的值（阈值设为1e-10）
        AscendC::Duplicate(B_inner2.Get<float>(), 1e-8f, length);
        AscendC::Compare(near_zero_mask, x_abs, B_inner2.Get<float>(), AscendC::CMPMODE::LE, length);
        
        // 对于接近0的值，将x_abs修改为1.0避免计算问题
        AscendC::Duplicate(B_inner2.Get<float>(), 1.0f, length);
        AscendC::Select(x_abs, near_zero_mask, B_inner2.Get<float>(), x_abs, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
        
        // 使用修改后的x_abs_modified进行后续计算（代替原来的x_abs）
        // AscendC::Adds(x_abs, x_abs_modified, 0.0f, length);  // 复制回x_abs
        
        // � 工作缓冲区（用于当前段计算和掩码操作）
        auto current_result = B_result1.Get<float>();    // 当前段结果
        auto temp_calc = current_result;   // 临时计算
        auto mask_work = B_mask1.Get<uint16_t>();        // 工作掩码
        auto mask_temp = B_mask2.Get<uint16_t>();        // 临时掩码
        
        // 🚀 初始化最终结果为0
        // AscendC::Duplicate(y, 0.0f, length);
        
        // ===== 🏆 按优先级顺序计算并立即合并（从低到高优先级）=====
        
        // 1️⃣ 渐近计算 (|x| > 4.55) - 最低优先级
        // AscendC::Duplicate(temp_calc, 10.2f, length);
        CalculateAsymptotic(x_abs, y, length);
        // AscendC::Compare(mask_work, x_abs, temp_calc, AscendC::CMPMODE::GT, length);
        
        // AscendC::Select(y, mask_work, current_result, y, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
        AscendC::LocalTensor<float> B_coefs[CHEB_ORDER] = {
            B_c1.Get<float>(), B_c2.Get<float>(), B_c3.Get<float>(), 
            B_c4.Get<float>(), B_c5.Get<float>(), B_c6.Get<float>(),
            B_c7.Get<float>()
        };
        auto selected_t_min = B_tri1.Get<float>(); 
        auto selected_t_max = B_tri2.Get<float>(); 
        
    // 初始化系数和边界

        
            // 使用宏处理各个切比雪夫区间
    // auto index = B_mask2.Get<uint32_t>();
    AscendC::Muls(temp_calc, x_abs, 4.0f/0.159375f, length);  // 除以 0.159375
                 // 取 floor
    AscendC::Mins(temp_calc, temp_calc, 252.0f, length);

// 3. 限制 byte_offset >= 0
    AscendC::Maxs(temp_calc, temp_calc, 0.0f, length);
    auto nuindex = B_mask3.Get<int32_t>();
    
    AscendC::Cast(nuindex, temp_calc, AscendC::RoundMode::CAST_FLOOR, length);  
    auto index = nuindex.ReinterpretCast<uint32_t>();


    auto lookup_tmin = B_lut_tmin.Get<float>();
    auto lookup_tmax = B_lut_tmax.Get<float>();
    auto lookup_coef_0 = B_lut_c0.Get<float>(); 
    auto lookup_coef_1 = B_lut_c1.Get<float>(); 
    auto lookup_coef_2 = B_lut_c2.Get<float>(); 
    auto lookup_coef_3 = B_lut_c3.Get<float>(); 
    auto lookup_coef_4 = B_lut_c4.Get<float>(); 
    auto lookup_coef_5 = B_lut_c5.Get<float>(); 
    auto lookup_coef_6 = B_lut_c6.Get<float>();
   

    
    AscendC::Gather(B_coefs[0], lookup_coef_0, index, 0, length);
    

    AscendC::Gather(B_coefs[1], lookup_coef_1, index, 0, length);


    AscendC::Gather(B_coefs[2], lookup_coef_2, index, 0, length);
    

    AscendC::Gather(B_coefs[3], lookup_coef_3, index, 0, length);
    

    AscendC::Gather(B_coefs[4], lookup_coef_4, index, 0, length);
    

    AscendC::Gather(B_coefs[5], lookup_coef_5, index, 0, length);
    

    AscendC::Gather(B_coefs[6], lookup_coef_6, index, 0, length);
    // AscendC::Gather(B_coefs, cheb_coef_array, index,0, length);
    AscendC::Gather(selected_t_min, lookup_tmin, index,0, length);
    AscendC::Gather(selected_t_max, lookup_tmax, index, 0,length);
    // AscendC::Muls(y, selected_t_max, 1.0f, length);
    // // 创建主掩码（所有切比雪夫区间的并集）
    auto main_mask = B_mask1.Get<uint16_t>();
    AscendC::Duplicate(temp_calc, 0.0f, length);
    AscendC::Compare(main_mask, x_abs, temp_calc, AscendC::CMPMODE::GT, length);
    AscendC::Duplicate(temp_calc, 10.2f, length);
    AscendC::Compare(mask_temp, x_abs, temp_calc, AscendC::CMPMODE::LE, length);
    AscendC::And(main_mask, main_mask, mask_temp, length);

    // AscendC::Duplicate(y, B_coefs[1][63], length);
    // 计算切比雪夫多项式
    CalculateChebyshev(x_abs, current_result, B_coefs, selected_t_min, selected_t_max, length);
    AscendC::Select(y, main_mask, current_result, y, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, length);

        
    // y = current_result;
    // 🔄 处理负数：C(-x) = -C(x)
    AscendC::Muls(temp_calc, y, -1.0f, length);
    AscendC::Select(y, neg_mask, temp_calc, y, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
    AscendC::Duplicate(temp_calc, 0.0f, length);
    AscendC::Select(y, near_zero_mask, temp_calc, y, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
    }    // 简化的切比雪夫多项式计算 - 避免复杂的缓冲区复用  
    // 🏆 完整的11阶切比雪夫多项式计算 - 高精度实现

__aicore__ inline void CalculateChebyshev(
    AscendC::LocalTensor<float> &x_abs,  
    AscendC::LocalTensor<float> &result,  
    AscendC::LocalTensor<float> coefs[CHEB_ORDER],
    AscendC::LocalTensor<float> &t_min,
    AscendC::LocalTensor<float> &t_max,  
    uint32_t length) 
{  
    // 工作缓冲区
    auto t  = B_inner1.Get<float>();      // 映射变量 t
    auto y  = B_inner2.Get<float>();      // 映射后变量 y ∈ [-1, 1]
    auto T0 = B_inner3.Get<float>();      // T_{n-2}
    auto T1 = B_inner4.Get<float>();      // T_{n-1}
    auto Tn = B_inner5.Get<float>();      // T_n（工作区）
    auto tmp = B_inner6.Get<float>();     // 临时计算缓冲区

    // 1) 计算 t = x / (x + c)
    AscendC::Adds(tmp, x_abs, C_CHEB, length);  // tmp = x + c
    AscendC::Div(t,  x_abs, tmp, length);       // t = x / (x + c)

    // 2) y = 2 * (t - t_min) / (t_max - t_min) - 1
    AscendC::Sub(y, t, t_min, length);          // t - t_min
    AscendC::Sub(tmp, t_max, t_min, length);    // t_max - t_min
    AscendC::Div(y, y, tmp, length);            // (t - t_min) / (t_max - t_min)
    AscendC::Muls(y, y, 2.0f, length);
    AscendC::Adds(y, y, -1.0f, length);

    // 3) 初始化 T0=1, T1=y
    AscendC::Duplicate(T0, 1.0f, length);
    AscendC::DataCopy(T1, y, length);

    // result = c0*T0 + c1*T1
    AscendC::Mul(result, T0, coefs[0], length);
    AscendC::Mul(tmp,    T1, coefs[1], length);
    AscendC::Add(result, result, tmp,   length);

    // 4) 递推：对 n=2..6
    #define CHEB_STEP(CI)                                      \
        /* Tn = 2*y*T1 - T0 */                                 \
        AscendC::Mul(Tn,  y,  T1, length);                     \
        AscendC::Muls(Tn, Tn, 2.0f, length);                   \
        AscendC::Sub(Tn,  Tn, T0, length);                     \
        /* result += c_n * Tn */                               \
        AscendC::Mul(tmp, Tn, coefs[CI], length);              \
        AscendC::Add(result, result, tmp, length);             \
        /* 旋转句柄：T0 <- T1, T1 <- Tn, Tn <- T0(old) */      \
        { auto _swap = T0; T0 = T1; T1 = Tn; Tn = _swap; }

    CHEB_STEP(2)
    CHEB_STEP(3)
    CHEB_STEP(4)
    CHEB_STEP(5)
    CHEB_STEP(6)

    #undef CHEB_STEP
}

    // 渐近计算 - 🚨 修复缓冲区冲突：不能复用基础数据缓冲区！
__aicore__ inline void CalculateAsymptotic(AscendC::LocalTensor<float> &x_abs,
                                          AscendC::LocalTensor<float> &result,
                                          uint32_t length) {
    // C(x) = (1/2) + (1-0.121exp[-2(x-1)])/(πx) * sin((π/2)x²)
    
    // 初始化常数 1/2
    
    
    // 计算角度 θ = (π/2)x²
    auto inner1 = B_inner1.Get<float>();
    AscendC::Mul(inner1, x_abs, x_abs, length);     // x²
    AscendC::Muls(inner1, inner1, PI/2.0f, length);   // (π/2)x²
    SinWithReduction(inner1, result, length);   // sin((π/2)x²)                                        
    // 计算 sin((π/2)x²) - 使用角度规约确保精度

    auto inner2 = B_inner2.Get<float>();
    auto inner4 = B_inner4.Get<float>();
    // 计算 exp[-2(x-1)]
    AscendC::Duplicate(inner4, 1.0f, length);
    // AscendC::Sub(inner2, x_abs, inner4, length);      // (x-1)
    // AscendC::Muls(inner2, inner2, -2.0f, length);     // -2(x-1)
    // AscendC::Exp(inner2, inner2, length);             // exp[-2(x-1)]
    
    // 计算 1-0.121exp[-2(x-1)]
    // auto inner3 = B_inner3.Get<float>();
    // AscendC::Muls(inner2, inner2, 0.121f, length);    // 0.121exp[-2(x-1)]
    // // AscendC::Duplicate(inner3, 1.0f, length);
    // AscendC::Sub(inner3, inner4, inner2, length);       // 1-0.121exp[-2(x-1)]

    // 计算 1/(πx)
    AscendC::Muls(inner2, x_abs, PI, length);       // πx
    // AscendC::Duplicate(inner1, 1.0f, length);
    AscendC::Div(inner2, inner4, inner2, length);       // 1/(πx)

    // 计算 (1-0.121exp[-2(x-1)])/(πx)
    // AscendC::Mul(inner3, inner3, inner2, length);       // (1-0.121exp[-2(x-1)])/(πx)
    
    // 计算 (1-0.121exp[-2(x-1)])/(πx) * sin((π/2)x²)
    AscendC::Mul(inner2, inner2, result, length); // (1-0.121exp[-2(x-1)])/(πx) * sin((π/2)x²)

    // 最终结果: (1/2) + (1-0.121exp[-2(x-1)])/(πx) * sin((π/2)x²)
    // AscendC::Muls(result, x_abs, 1.0f, length);
    AscendC::Adds(result, inner2, 0.5f, length);  // 1/2
}



__aicore__ inline void TaylorSinImproved(AscendC::LocalTensor<float> &x, AscendC::LocalTensor<float> &result1, uint32_t length) {
    // sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9! - x¹¹/11! + x¹³/13! - x¹⁵/15!
    //        = x - x³/6 + x⁵/120 - x⁷/5040 + x⁹/362880 - x¹¹/39916800 + x¹³/6227020800 - x¹⁵/1307674368000
    AscendC::LocalTensor<float> x2 = B_tri1.Get<float>();  // 复用
    AscendC::LocalTensor<float> term = B_tri2.Get<float>(); // 复用
    // AscendC::Mul(result1, x, x, length);  // x²

    AscendC::Mul(x2, x, x, length);  // x²
    // AscendC::Muls(result1, x1,1.0f, length);  // x²
    AscendC::Duplicate(result1, 0.0f, length);

    // 第1项: x
    AscendC::Add(result1, result1, x, length);

    // // 第2项: -x³/6
    AscendC::Mul(term, x, x2, length);         // x³
    AscendC::Muls(term, term, -1.0f/6.0f, length); // -x³/6
    AscendC::Add(result1, result1, term, length);

    // // 第3项: +x⁵/120
    AscendC::Mul(term, term, x2, length);      // x⁵
    AscendC::Muls(term, term, -1.0f/20.0f, length); // x⁵/120 = (-x⁵/6) * (-1/20) ✅ 修复了 -20.0f 的错误
    AscendC::Add(result1, result1, term, length);
    // 第4项: -x⁷/5040
    AscendC::Mul(term, term, x2, length);      // x⁷
    AscendC::Muls(term, term, -1.0f/42.0f, length); // -x⁷/5040 = (x⁷/120) * (-1/42)
    AscendC::Add(result1, result1, term, length);

}

    // 🚨 修复正弦函数缓冲区冲突：使用专用工作区


    // 🚨 修复泰勒Sin计算缓冲区冲突
    __aicore__ inline void SinWithReduction(AscendC::LocalTensor<float> &angle, 
                                        AscendC::LocalTensor<float> &result,
                                        uint32_t length) {
    // 只需要3个float缓冲区 + 1个mask缓冲区
    auto thetaReduced = B_c1.Get<float>();  // 直接复用result作为工作空间
    auto tmp1 = B_c2.Get<float>();
    auto tmp2 = B_c3.Get<float>();
    auto pi05 = B_c4.Get<float>();
    auto pi10 = B_c5.Get<float>();
    auto pi15 = B_c6.Get<float>();
    auto pi20 = B_c7.Get<float>();
    auto neg = B_inner4.Get<float>();  // 结果缓冲区
    auto mask = B_mask2.Get<uint16_t>();  // 只需要一个mask，复用
    AscendC::Duplicate(neg, 1.0f, length);
    AscendC::Duplicate(pi05, 0.5f * PI, length);
    AscendC::Duplicate(pi10, PI, length);
    AscendC::Duplicate(pi15, 1.5f * PI, length);
    AscendC::Duplicate(pi20, 2.0f * PI, length);
    // ===== 步骤1: 角度约化到[0, 2π] =====
    AscendC::Muls(tmp1, angle, 1.0f / TWO_PI, length);
    AscendC::Floor(tmp1, tmp1, length);
    AscendC::Muls(tmp1, tmp1, TWO_PI, length);
    AscendC::Sub(thetaReduced, angle, tmp1, length);  // 约化后的角度
        
    // 处理第二象限 [π/2, π): sin(θ) = sin(π-θ)
    AscendC::Compare(mask, thetaReduced, pi10, AscendC::CMPMODE::LT, length);
    AscendC::Compare(tmp2.ReinterpretCast<uint16_t>(), thetaReduced, pi05, AscendC::CMPMODE::GE, length);
    AscendC::And(mask, mask, tmp2.ReinterpretCast<uint16_t>(), length);  // [π/2, π)
    
    // 计算 sin(π - θ) = sin(π/2 + (π/2 - θ)) = cos(π/2 - θ) = sin(π - θ)

    AscendC::Sub(tmp1, pi10, thetaReduced, length);  // π - θ
    AscendC::Select(thetaReduced, mask, tmp1, thetaReduced, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
    
    // 处理第三象限 [π, 3π/2): sin(θ) = -sin(θ-π)
    AscendC::Compare(mask, thetaReduced, pi15, AscendC::CMPMODE::LT, length);
    AscendC::Compare(tmp2.ReinterpretCast<uint16_t>(), thetaReduced, pi10, AscendC::CMPMODE::GE, length);
    AscendC::And(mask, mask, tmp2.ReinterpretCast<uint16_t>(), length);  // [π, 3π/2)
    
    AscendC::Sub(tmp1, thetaReduced, pi10, length);  // θ - π
    AscendC::Select(thetaReduced, mask, tmp1, thetaReduced, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, length);

    AscendC::Duplicate(tmp2, -1.0f, length);  // -sin(θ - π)
    AscendC::Select(neg, mask, tmp2, neg, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
    
    // 处理第四象限 [3π/2, 2π): sin(θ) = -sin(2π-θ)
    AscendC::Compare(mask, thetaReduced, pi15, AscendC::CMPMODE::GE, length);  // >= 3π/2
    
    AscendC::Sub(tmp1, pi20, thetaReduced, length);  // 2π - θ
    AscendC::Select(thetaReduced, mask, tmp1, thetaReduced, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, length);


    AscendC::Select(neg, mask, tmp2, neg, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
    TaylorSinImproved(thetaReduced, result, length);
    Mul(result, result, neg, length);  // 最终结果 = sign * sin(x)
}
    // 🚨 修复泰勒Cos计算缓冲区冲突  


    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {  
        AscendC::LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();  
        AscendC::DataCopy(Gm_y[progress * this->tileLength], y, length);  
        Q_y.FreeTensor(y);  
    }  

private:  
    AscendC::TPipe pipe;  
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> Q_x;  
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> Q_y;  
    
    // 优化后的缓冲区定义 - 大幅减少内存使用
    // 多功能临时缓冲区 (4个)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_tmp1;  
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_inner1, B_inner2, B_inner3, B_inner4,B_inner5,B_inner6; 
    // 掩码缓冲区 (3个)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_mask1, B_mask2,B_mask3,B_mask_temp,B_mask4;  
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_tri1, B_tri2; 
    // 结果缓冲区 (2个，用于交替存储不同段的结果)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_result1;  
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_c1, B_c2, B_c3, B_c4, B_c5, B_c6,B_c7;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_lut_tmin, B_lut_tmax;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_lut_c0, B_lut_c1, B_lut_c2, B_lut_c3, B_lut_c4, B_lut_c5, B_lut_c6;
    // AscendC::TBuf<AscendC::QuePosition::VECCALC> B_tmin, B_tmax，B_x2;
    // 类型转换缓冲区（条件性）  
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_fx, B_fy;  
    
    AscendC::GlobalTensor<TYPE_X> Gm_x;  
    AscendC::GlobalTensor<TYPE_Y> Gm_y;  
    uint32_t blockLength;  
    uint32_t tileNum;  
    uint32_t tileLength;  
};  

namespace {  
extern "C" __global__ __aicore__ void fresnel_cos(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {  
    GET_TILING_DATA(tiling_data, tiling);  
    FresnelCos<DTYPE_X, DTYPE_Y> op;  
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);  
    
    op.Process();  
}  
}
