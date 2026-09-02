#include "pch.h"

//============================================================
//                     TUNING PARAMETERS
//============================================================
static constexpr float  hj_init_delta_frac = 0.07f;
static constexpr float  hj_min_delta = 1e-4f;
static constexpr float  hj_shrink_factor = 0.55f;
static constexpr float  hj_pattern_gain = 0.7f;

static constexpr float  hj_angle_scale_proximal = 1.4f;
static constexpr float  hj_angle_scale_distal = 0.6f;
static constexpr float  hj_length_scale_proximal = 1.4f;
static constexpr float  hj_length_scale_distal = 0.6f;

static constexpr float  tau = 0.2f;

static constexpr float  SWEEP_MIN_BOUND = 0.05f;
static constexpr float  INV_SWEEP_MIN_BOUND = 20.0f;

static constexpr float  OBSTACLE_CLEARANCE = 0.05f;

//============================================================
//                     ABI / SIZE LIMITS
//============================================================
static constexpr int            AGP_MAX_FULL_DIM = 16;
static constexpr int            AGP_MAX_LINK_POINTS = AGP_MAX_FULL_DIM + 1;
static constexpr unsigned       MAX_OBSTACLES = 4u;
static constexpr unsigned       AGP_MULTI_MAX_COUNT = 7u;
static constexpr size_t         AGP_INTERVAL_SLAB_BYTES = 67108864u;
static constexpr int            AGP_MAX_GENERATED_SEEDS = 64;
static constexpr size_t         AGP_MAX_LBFGS_HISTORY = 16u;
static constexpr int            LBFGS_M = 8;
static constexpr int            AGP_SEED_STRIDE = 16;

//============================================================
//                     SIMD / MATH CONSTANTS
//============================================================
static __declspec(align(32)) const __m256   VEC_TWOPI = _mm256_set1_ps(6.28318530717958647692f);
static __declspec(align(32)) const __m256   VEC_PI = _mm256_set1_ps(3.14159265358979323846f);
static __declspec(align(32)) const __m256   VEC_PI_2 = _mm256_set1_ps(1.57079632679489661923f);
static __declspec(align(32)) const __m256   INV_TWOPI = _mm256_set1_ps(0.15915494309189533577f);
static __declspec(align(32)) const __m256   VEC_BIAS = _mm256_set1_ps(12582912.0f);
static __declspec(align(32)) const __m256   VEC_COS_P5 = _mm256_set1_ps(-0.0013888889f);
static __declspec(align(32)) const __m256   VEC_COS_P3 = _mm256_set1_ps(0.0416666667f);
static __declspec(align(32)) const __m256   VEC_COS_P1 = _mm256_set1_ps(-0.5f);
static __declspec(align(32)) const __m256   VEC_COS_P0 = _mm256_set1_ps(1.0f);
static __declspec(align(32)) const __m256   VEC_SIN_P5 = _mm256_set1_ps(-0.0001984127f);
static __declspec(align(32)) const __m256   VEC_SIN_P3 = _mm256_set1_ps(0.0083333333f);
static __declspec(align(32)) const __m256   VEC_SIN_P1 = _mm256_set1_ps(-0.16666666f);
static __declspec(align(32)) const __m256   VEC_SIN_P0 = _mm256_set1_ps(1.0f);
static __declspec(align(32)) const __m256   VEC_ZERO = _mm256_setzero_ps();

static __declspec(align(32)) const __m256   veps = _mm256_set1_ps(1e-6f);
static __declspec(align(32)) const __m256   vflt_max = _mm256_set1_ps(FLT_MAX);
static __declspec(align(32)) const __m256   v2 = _mm256_set1_ps(2.0f);
static __declspec(align(32)) const __m256   vNEG_PI = _mm256_set1_ps(-3.14159265358979323846f);
static __declspec(align(32)) const __m256   vabs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

static __declspec(align(16)) const __m128   vpi4 = _mm_set1_ps(3.14159265358979323846f);
static __declspec(align(16)) const __m128   vnpi4 = _mm_set1_ps(-3.14159265358979323846f);
static __declspec(align(16)) const __m128   vtwo_pi4 = _mm_set1_ps(6.28318530717958647692f);

static constexpr float  TWO_PI = 6.28318530717958647692f;
static constexpr float  PI = 3.14159265358979323846f;
static constexpr float  PI_2 = 1.57079632679489661923f;
static constexpr float  INV_TWO_PI = 0.15915494309189533577f;
static constexpr float  BIAS = 12582912.0f;

//============================================================
//                     MATH MACROS & UTILITIES
//============================================================

#define XOR_STEP(s) (s)^=(s)<<13;(s)^=(s)>>17;(s)^=(s)<<5;

#define FABE13_SIN(x, result_var)\
const float _x_ = x;\
const float _ax_ = fabsf(_x_);\
float _r_;\
if (_ax_ < TWO_PI) {\
    _r_ = _ax_;\
} else {\
    float _q_ = fmaf(_ax_, INV_TWO_PI, BIAS);\
    _q_ -= BIAS;\
    _r_ = fmaf(_q_, -TWO_PI, _ax_);\
}\
bool _sfl_ = _r_ > PI;\
if (_sfl_)\
_r_ = TWO_PI - _r_;\
bool _cfl_ = _r_ > PI_2;\
if (_cfl_)\
_r_ = PI - _r_;\
const float _t2_ = _r_ * _r_;\
float _s = fmaf(_t2_,\
    fmaf(_t2_,\
    fmaf(_t2_, -0.0001984127f, 0.0083333333f),\
    -0.16666666f),\
    1.0f) *\
    _r_;\
result_var = _x_ < 0.0f ^ _sfl_ ? -_s : _s;

#define FABE13_SINCOS(_in_, _sin_out_, _cos_out_, _n_)\
int _i_ = 0;\
const int _limit_ = _n_ & ~7;\
if (_n_ >= 8)\
{\
    while (_i_ < _limit_) {\
        const __m256 _vx_ = _mm256_load_ps(&(_in_)[_i_]);\
        const __m256 _vax_ = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _vx_);\
        __m256 _max_ = _mm256_max_ps(_vax_, _mm256_permute2f128_ps(_vax_, _vax_, 0x01));\
        _max_ = _mm256_max_ps(_max_, _mm256_permute_ps(_max_, 0x4E));\
        _max_ = _mm256_max_ps(_max_, _mm256_permute_ps(_max_, 0xB1));\
        const float max_val = _mm256_cvtss_f32(_max_);\
        __m256 _r_;\
        if (max_val < TWO_PI) {\
            _r_ = _vax_;\
        } else {\
            __m256 _q_ = _mm256_fmadd_ps(_vax_, INV_TWOPI, VEC_BIAS);\
            _q_ = _mm256_sub_ps(_q_, VEC_BIAS);\
            _r_ = _mm256_fnmadd_ps(VEC_TWOPI, _q_, _vax_);\
        }\
        const __m256 _r1_ =\
        _mm256_min_ps(_r_, _mm256_sub_ps(VEC_TWOPI, _r_));\
        const __m256 _r2_ =\
        _mm256_min_ps(_r1_, _mm256_sub_ps(VEC_PI, _r1_));\
        const __m256 _t2_ = _mm256_mul_ps(_r2_, _r2_);\
        const __m256 _cosv_ =\
        _mm256_fmadd_ps(_t2_,\
        _mm256_fmadd_ps(_t2_,\
        _mm256_fmadd_ps(_t2_,\
        VEC_COS_P5,\
        VEC_COS_P3),\
        VEC_COS_P1),\
        VEC_COS_P0);\
        const __m256 _sinv_ =\
        _mm256_mul_ps(_mm256_fmadd_ps(_t2_,\
        _mm256_fmadd_ps(_t2_,\
        _mm256_fmadd_ps(_t2_,\
        VEC_SIN_P5,\
        VEC_SIN_P3),\
        VEC_SIN_P1),\
        VEC_SIN_P0),\
        _r2_);\
        const __m256 _cflip_ =\
        _mm256_cmp_ps(_r1_, VEC_PI_2, _CMP_GT_OQ);\
        const __m256 _sflip_ =\
        _mm256_xor_ps(_mm256_cmp_ps(_vx_, VEC_ZERO, _CMP_LT_OQ),\
        _mm256_cmp_ps(_r_, VEC_PI, _CMP_GT_OQ));\
        _mm256_store_ps(&(_cos_out_)[_i_],\
        _mm256_blendv_ps(_cosv_,\
        _mm256_sub_ps(VEC_ZERO,\
        _cosv_),\
        _cflip_));\
        _mm256_store_ps(&(_sin_out_)[_i_],\
        _mm256_blendv_ps(_sinv_,\
        _mm256_sub_ps(VEC_ZERO,\
        _sinv_),\
        _sflip_));\
        _i_ += 8;\
    }\
}\
while (_i_ < _n_) {\
    const float _x_ = (_in_)[_i_];\
    const float _ax_ = fabsf(_x_);\
    float _r_;\
    if (_ax_ < TWO_PI) {\
        _r_ = _ax_;\
    } else {\
        float _q_ = fmaf(_ax_, INV_TWO_PI, BIAS);\
        _q_ -= BIAS;\
        _r_ = fmaf(_q_, -TWO_PI, _ax_);\
    }\
    const bool _sflip_ = _r_ > PI;\
    if (_sflip_)\
    _r_ = TWO_PI - _r_;\
    const bool _cflip_ = _r_ > PI_2;\
    if (_cflip_)\
    _r_ = PI - _r_;\
    const float _t2_ = _r_ * _r_;\
    const float _c_ = fmaf(_t2_,\
    fmaf(_t2_,\
    fmaf(_t2_, -0.0013888889f, 0.0416666667f),\
    -0.5f),\
    1.0f);\
    const float _s_ = fmaf(_t2_,\
    fmaf(_t2_,\
    fmaf(_t2_, -0.0001984127f, 0.0083333333f),\
    -0.16666666f),\
    1.0f) *\
    _r_;\
    (_cos_out_)[_i_] = _cflip_ ? -_c_ : _c_;\
    (_sin_out_)[_i_] = _x_ < 0.0f ^ _sflip_ ? -_s_ : _s_;\
    ++_i_;\
}

template<int D>
static __declspec(noalias) __forceinline
float step(const float _m,
	const float invm,
	const float x1,
	const float x2,
	const float y1,
	const float y2,
	const float _r,
	const unsigned idx1,
	const unsigned idx2) noexcept
{
	if (idx1 != idx2)
		return fmaf(x1, 0.5f, x2 * 0.5f);

	const float diff = y2 - y1;
	const unsigned sign_bits = *reinterpret_cast<const unsigned*>(&diff) & 0x80000000u ^ 0x3f800000u;
	const float sign_mult = *reinterpret_cast<const float*>(&sign_bits);
	const float base = invm * fabsf(diff);
	float pow_val;
	if constexpr (D == 1)      pow_val = base;
	else if constexpr (D == 2) pow_val = base * base;
	else if constexpr (D == 3) { const float b2 = base * base; pow_val = b2 * base; }
	else if constexpr (D == 4) { const float b2 = base * base; pow_val = b2 * b2; }
	else if constexpr (D == 5) { const float b2 = base * base; const float b4 = b2 * b2; pow_val = b4 * base; }
	else if constexpr (D == 6) { const float b3 = base * base * base; pow_val = b3 * b3; }
	else if constexpr (D == 7) { const float b3 = base * base * base; const float b6 = b3 * b3; pow_val = b6 * base; }
	else if constexpr (D == 8) { const float b2 = base * base; const float b4 = b2 * b2; pow_val = b4 * b4; }
	else if constexpr (D == 9) { const float b3 = base * base * base; const float b6 = b3 * b3; pow_val = b6 * b3; }
	else if constexpr (D == 10) { const float b2 = base * base; const float b4 = b2 * b2; const float b8 = b4 * b4; pow_val = b8 * b2; }
	else if constexpr (D == 11) { const float b2 = base * base; const float b4 = b2 * b2; const float b8 = b4 * b4; pow_val = b8 * b2 * base; }
	else if constexpr (D == 12) { const float b3 = base * base * base; const float b6 = b3 * b3; pow_val = b6 * b6; }
	else if constexpr (D == 13) { const float b3 = base * base * base; const float b6 = b3 * b3; const float b12 = b6 * b6; pow_val = b12 * base; }
	else if constexpr (D == 14) { const float b7 = base * base * base * base * base * base * base; pow_val = b7 * b7; }
	else if constexpr (D == 15) { const float b7 = base * base * base * base * base * base * base; const float b14 = b7 * b7; pow_val = b14 * base; }
	else { const float b2 = base * base; const float b4 = b2 * b2; const float b8 = b4 * b4; pow_val = b8 * b8; }
	return fmaf(sign_mult * pow_val, _r, x1 + x2) * 0.5f;
}

static __declspec(noalias) __forceinline
unsigned long long gray_decode(unsigned long long g) noexcept
{
	g ^= g >> 32; g ^= g >> 16; g ^= g >> 8; g ^= g >> 4; g ^= g >> 2; g ^= g >> 1;
	return g;
}

static __declspec(noalias) __forceinline
unsigned long long gray_encode(unsigned long long x) noexcept { return x ^ (x >> 1); }

static __declspec(noalias) __forceinline
float agp_clamp_unit_open_scalar(float t) noexcept
{
	return fmaxf(0.0f, fminf(t, 0.999999940395355224609375f));
}

//============================================================
//                     RUNTIME GLOBALS
//============================================================
static const boost::mpi::environment* g_env = nullptr;
static boost::mpi::communicator* g_world = nullptr;

//============================================================
//                     TRAJECTORY COUNTER
//============================================================
alignas(64) static std::atomic<int> g_trajectoryCallCounter{ 0 };

//============================================================
//                     FORWARD DECLARATIONS / STRUCTS
//============================================================

struct MortonCachePerRank final
{
	std::vector<int, boost::alignment::aligned_allocator<int, 32u>>                  permCache;
	std::vector<unsigned long long, boost::alignment::aligned_allocator<unsigned long long, 32u>> invMaskCache;
	unsigned                                                                            baseSeed;
	bool                                                                                reverseTraversal;
};

static __declspec(noalias) __forceinline
unsigned long long agp_morton_orientation_period(const int dim) noexcept
{
	unsigned long long p = 2ull;

	int i = 2;
	while (i <= dim)
	{
		p *= static_cast<unsigned long long>(i);
		++i;
	}

	if (dim > 1)
		p <<= static_cast<unsigned>(dim - 1);

	return p;
}

static __declspec(noalias) __forceinline
void agp_setup_rank_morton_orientation(
	MortonCachePerRank& mc,
	const int           dim,
	const int           rank) noexcept
{
	const unsigned long long period = agp_morton_orientation_period(dim);
	const unsigned long long orientation = static_cast<unsigned long long>(rank) % period;
	const unsigned long long geom_count = period >> 1ull;

	const bool reverse = orientation >= geom_count;

	const unsigned long long geom =
		reverse
		? orientation - geom_count
		: orientation;

	const unsigned long long flip_mask =
		(dim > 1)
		? (geom & ((1ull << static_cast<unsigned>(dim - 1)) - 1ull))
		: 0ull;

	unsigned long long perm_index =
		(dim > 1)
		? (geom >> static_cast<unsigned>(dim - 1))
		: 0ull;

	mc.reverseTraversal = reverse;

	int i = 0;
	while (i < dim)
	{
		mc.permCache[static_cast<size_t>(i)] = i;
		mc.invMaskCache[static_cast<size_t>(i)] = 0ull;
		++i;
	}

	i = 0;
	while (i < dim)
	{
		const unsigned long long span = static_cast<unsigned long long>(dim - i);
		const size_t j = static_cast<size_t>(i) + static_cast<size_t>(perm_index % span);

		perm_index /= span;

		std::swap(mc.permCache[static_cast<size_t>(i)], mc.permCache[j]);

		++i;
	}

	int k = 1;
	while (k < dim)
	{
		if (flip_mask & (1ull << static_cast<unsigned>(k - 1)))
		{
			const int axis = mc.permCache[static_cast<size_t>(k)];
			mc.invMaskCache[static_cast<size_t>(axis)] = ~0ull;
		}

		++k;
	}
}

struct MortonND final
{
	int                                                                                 dim;
	int                                                                                 levels;
	int                                                                                 eff_levels;
	int                                                                                 extra_levels;
	int                                                                                 chunks;
	int                                                                                 cell_max_i;
	std::vector<int, boost::alignment::aligned_allocator<int, 32u>>                    chunk_bits;
	std::vector<int, boost::alignment::aligned_allocator<int, 32u>>                    chunk_inv_shift;
	std::vector<unsigned long long, boost::alignment::aligned_allocator<unsigned long long, 32u>> chunk_bases;
	std::vector<unsigned long long, boost::alignment::aligned_allocator<unsigned long long, 32u>> chunk_masks;
	std::vector<unsigned long long, boost::alignment::aligned_allocator<unsigned long long, 32u>> pextMaskChunks;
	std::vector<float, boost::alignment::aligned_allocator<float, 32u>>                chunk_basef;
	std::vector<float, boost::alignment::aligned_allocator<float, 32u>>                chunk_invBasef;
	std::vector<float, boost::alignment::aligned_allocator<float, 32u>>                pow2m1;
	unsigned long long                                                                  scale;

	__declspec(align(32)) float   low[AGP_MAX_FULL_DIM];
	__declspec(align(32)) float   high[AGP_MAX_FULL_DIM];
	__declspec(align(32)) float   step[AGP_MAX_FULL_DIM];
	__declspec(align(32)) float   invStep[AGP_MAX_FULL_DIM];
	__declspec(align(32)) float   baseOff[AGP_MAX_FULL_DIM];
	__declspec(align(32)) int     perm[AGP_MAX_FULL_DIM];
	__declspec(align(32)) unsigned long long invMask[AGP_MAX_FULL_DIM];
	__declspec(align(32)) unsigned long long pextMask[AGP_MAX_FULL_DIM];
	__declspec(align(32)) float   step_perm[AGP_MAX_FULL_DIM];
	__declspec(align(32)) float   invStep_perm[AGP_MAX_FULL_DIM];
	__declspec(align(32)) float   baseOff_perm[AGP_MAX_FULL_DIM];
	__declspec(align(32)) unsigned long long pextMask_perm[AGP_MAX_FULL_DIM];
	__declspec(align(32)) unsigned long long invMask_perm[AGP_MAX_FULL_DIM];

	float                                                                               invScaleLevel;
	bool                                                                                reverseTraversal;

	static __declspec(noalias) __forceinline int choose_eff_levels_float_safe(int D) noexcept
	{
		const int d = (D > 0) ? D : 1;
		const int hw = 63 / d;
		const int fp = 24 / d;
		return (std::max)(1, (std::min)(hw, (std::max)(1, fp)));
	}

	static __declspec(noalias) __forceinline unsigned long long make_mask(const int dim, const int Lc, const int d) noexcept
	{
		unsigned long long m = 0ull;
		unsigned long long bitpos = static_cast<unsigned long long>(d);
		int b = 0;
		while (b < Lc)
		{
			m |= (1ull << bitpos);
			bitpos += static_cast<unsigned long long>(dim);
			++b;
		}
		return m;
	}

	template<int D>
	__forceinline float block_diameter_acc(unsigned long long varying, float s2) const noexcept
	{
		float res = s2;
#pragma unroll
		for (int i = 0; i < D; ++i) {
			const int nfree_hi = static_cast<int>(_mm_popcnt_u64(varying & pextMask_perm[i]));
			const int nfree_total = nfree_hi + extra_levels;
			const float range = step_perm[i] * pow2m1[static_cast<size_t>(nfree_total)];
			res = fmaf(range, range, res);
		}
		return res;
	}

	template<int D, int C>
	__forceinline void map01ToPoint_chunks(float t, float* __restrict out) const noexcept
	{
		const float* __restrict p_chunk_basef = chunk_basef.data();
		const int* __restrict p_chunk_bits = chunk_bits.data();
		const int* __restrict p_chunk_inv_shift = chunk_inv_shift.data();
		const unsigned long long* __restrict p_chunk_masks = chunk_masks.data();
		const unsigned long long* __restrict p_pextMaskChunks = pextMaskChunks.data();

		__declspec(align(32)) unsigned long long accBits[AGP_MAX_FULL_DIM]{};

#pragma unroll
		for (int c = 0; c < C; ++c) {
			const size_t ci = static_cast<size_t>(c);
			t *= p_chunk_basef[ci];
			unsigned long long idxc = static_cast<unsigned long long>(t);
			t -= static_cast<float>(idxc);
			idxc = gray_encode(idxc);

			const int Lc = p_chunk_bits[ci];
			const int inv_shift = p_chunk_inv_shift[ci];
			const unsigned long long chunk_mask = p_chunk_masks[ci];
			const unsigned long long* __restrict masks = p_pextMaskChunks + ci * D;

#pragma unroll
			for (int i = 0; i < D; ++i) {
				unsigned long long bits = _pext_u64(idxc, masks[i]);
				bits ^= ((invMask_perm[i] >> inv_shift) & chunk_mask);
				accBits[i] = (accBits[i] << Lc) | bits;
			}
		}

#pragma unroll
		for (int i = 0; i < D; ++i) {
			out[i] = fmaf(step_perm[i], static_cast<float>(accBits[i]), baseOff_perm[i]);
		}
	}

	template<int D>
	__forceinline void map01ToPoint_general(float t, float* __restrict out) const noexcept
	{
		const float* __restrict p_chunk_basef = chunk_basef.data();
		const int* __restrict p_chunk_bits = chunk_bits.data();
		const int* __restrict p_chunk_inv_shift = chunk_inv_shift.data();
		const unsigned long long* __restrict p_chunk_masks = chunk_masks.data();
		const unsigned long long* __restrict p_pextMaskChunks = pextMaskChunks.data();

		__declspec(align(32)) unsigned long long accBits[AGP_MAX_FULL_DIM]{};

#pragma unroll
		for (int c = 0; c < chunks; ++c) {
			const size_t ci = static_cast<size_t>(c);
			t *= p_chunk_basef[ci];
			unsigned long long idxc = static_cast<unsigned long long>(t);
			t -= static_cast<float>(idxc);
			idxc = gray_encode(idxc);

			const int Lc = p_chunk_bits[ci];
			const int inv_shift = p_chunk_inv_shift[ci];
			const unsigned long long chunk_mask = p_chunk_masks[ci];
			const unsigned long long* __restrict masks = p_pextMaskChunks + ci * D;

#pragma unroll
			for (int i = 0; i < D; ++i) {
				unsigned long long bits = _pext_u64(idxc, masks[i]);
				bits ^= ((invMask_perm[i] >> inv_shift) & chunk_mask);
				accBits[i] = (accBits[i] << Lc) | bits;
			}
		}

#pragma unroll
		for (int i = 0; i < D; ++i) {
			out[i] = fmaf(step_perm[i], static_cast<float>(accBits[i]), baseOff_perm[i]);
		}
	}

	template<int D>
	__forceinline void map01ToPoint_t(float t, float* __restrict out) const noexcept
	{
		const int c = chunks;
		if (c == 1) {
			map01ToPoint_chunks<D, 1>(t, out);
		}
		else if (c == 2) {
			map01ToPoint_chunks<D, 2>(t, out);
		}
		else {
			map01ToPoint_general<D>(t, out);
		}
	}

	template<int D, int C>
	__forceinline float pointToT_chunks(const float* __restrict q) const noexcept
	{
		const int* __restrict p_perm = perm;
		const float* __restrict p_chunk_invBasef = chunk_invBasef.data();
		const int* __restrict p_chunk_inv_shift = chunk_inv_shift.data();
		const unsigned long long* __restrict p_chunk_masks = chunk_masks.data();
		const unsigned long long* __restrict p_pextMaskChunks = pextMaskChunks.data();

		__declspec(align(32)) unsigned long long cell_ordered[AGP_MAX_FULL_DIM];

#pragma unroll
		for (int i = 0; i < D; ++i) {
			const int pd = p_perm[i];
			const float v = fmaf(q[pd] - baseOff_perm[i], invStep_perm[i], 0.0f);
			int ci = _mm_cvt_ss2si(_mm_set_ss(v));
			if (ci < 0) ci = 0;
			else if (ci > cell_max_i) ci = cell_max_i;
			cell_ordered[i] = static_cast<unsigned long long>(ci);
		}

		float t = 0.0f;
#pragma unroll
		for (int c = C - 1; c >= 0; --c) {
			const size_t ci = static_cast<size_t>(c);
			const int inv_shift = p_chunk_inv_shift[ci];
			const unsigned long long chunk_mask = p_chunk_masks[ci];
			const unsigned long long* __restrict masks = p_pextMaskChunks + ci * D;

			unsigned long long idxc = 0ull;
#pragma unroll
			for (int i = 0; i < D; ++i) {
				unsigned long long bits = (cell_ordered[i] >> inv_shift) & chunk_mask;
				bits ^= ((invMask_perm[i] >> inv_shift) & chunk_mask);
				idxc |= _pdep_u64(bits, masks[i]);
			}
			idxc = gray_decode(idxc);

			t = (t + static_cast<float>(idxc)) * p_chunk_invBasef[ci];
		}
		return t;
	}

	template<int D>
	__forceinline float pointToT_general(const float* __restrict q) const noexcept
	{
		const int* __restrict p_perm = perm;
		const float* __restrict p_chunk_invBasef = chunk_invBasef.data();
		const int* __restrict p_chunk_inv_shift = chunk_inv_shift.data();
		const unsigned long long* __restrict p_chunk_masks = chunk_masks.data();
		const unsigned long long* __restrict p_pextMaskChunks = pextMaskChunks.data();

		__declspec(align(32)) unsigned long long cell_ordered[AGP_MAX_FULL_DIM];

#pragma unroll
		for (int i = 0; i < D; ++i) {
			const int pd = p_perm[i];
			const float v = fmaf(q[pd] - baseOff_perm[i], invStep_perm[i], 0.0f);
			int ci = _mm_cvt_ss2si(_mm_set_ss(v));
			if (ci < 0) ci = 0;
			else if (ci > cell_max_i) ci = cell_max_i;
			cell_ordered[i] = static_cast<unsigned long long>(ci);
		}

		float t = 0.0f;
		for (int c = chunks - 1; c >= 0; --c) {
			const size_t ci = static_cast<size_t>(c);
			const int inv_shift = p_chunk_inv_shift[ci];
			const unsigned long long chunk_mask = p_chunk_masks[ci];
			const unsigned long long* __restrict masks = p_pextMaskChunks + ci * D;

			unsigned long long idxc = 0ull;
#pragma unroll
			for (int i = 0; i < D; ++i) {
				unsigned long long bits = (cell_ordered[i] >> inv_shift) & chunk_mask;
				bits ^= ((invMask_perm[i] >> inv_shift) & chunk_mask);
				idxc |= _pdep_u64(bits, masks[i]);
			}
			idxc = gray_decode(idxc);

			t = (t + static_cast<float>(idxc)) * p_chunk_invBasef[ci];
		}
		return t;
	}

	template<int D>
	__forceinline float pointToT_t(const float* __restrict q) const noexcept
	{
		const int c = chunks;
		if (c == 1) {
			return pointToT_chunks<D, 1>(q);
		}
		else if (c == 2) {
			return pointToT_chunks<D, 2>(q);
		}
		else {
			return pointToT_general<D>(q);
		}
	}

	__declspec(noalias) __forceinline MortonND(
		int D, int L, const float* lows, const float* highs, const MortonCachePerRank& mc)
		: dim(D)
		, levels(L)
		, eff_levels(choose_eff_levels_float_safe(D))
		, extra_levels((L > eff_levels) ? (L - eff_levels) : 0)
		, chunks((extra_levels > 0) ? (1 + (extra_levels + eff_levels - 1) / eff_levels) : 1)
		, cell_max_i(static_cast<int>((1u << L) - 1u))
		, chunk_bits(static_cast<size_t>(chunks))
		, chunk_inv_shift(static_cast<size_t>(chunks))
		, chunk_bases(static_cast<size_t>(chunks))
		, chunk_masks(static_cast<size_t>(chunks))
		, pextMaskChunks(static_cast<size_t>(chunks)* static_cast<size_t>(D))
		, chunk_basef(static_cast<size_t>(chunks))
		, chunk_invBasef(static_cast<size_t>(chunks))
		, pow2m1(static_cast<size_t>(L + 1))
		, scale(0ull)
		, invScaleLevel(1.0f / static_cast<float>(static_cast<unsigned long long>(1) << L))
		, reverseTraversal(mc.reverseTraversal)
	{
		for (int d = 0; d < dim; ++d)
		{
			const float lo = lows[d];
			const float hi = highs[d];
			const float st = (hi - lo) * invScaleLevel;

			low[d] = lo;
			high[d] = hi;
			step[d] = st;
			invStep[d] = 1.0f / st;
			baseOff[d] = lo + 0.5f * st;

			const int pd = mc.permCache[static_cast<size_t>(d)];
			perm[d] = pd;
			invMask[d] = mc.invMaskCache[static_cast<size_t>(d)];

			step_perm[d] = step[pd];
			invStep_perm[d] = invStep[pd];
			baseOff_perm[d] = baseOff[pd];
			invMask_perm[d] = invMask[pd];
		}

		int remaining = levels;
		int shift_from_top = 0;
		for (int c = 0; c < chunks; ++c)
		{
			const int Lc = (std::min)(eff_levels, remaining);
			chunk_bits[static_cast<size_t>(c)] = Lc;
			remaining -= Lc;
			shift_from_top += Lc;
			chunk_inv_shift[static_cast<size_t>(c)] = levels - shift_from_top;

			const unsigned long long base = 1ull << (dim * Lc);
			chunk_bases[static_cast<size_t>(c)] = base;
			chunk_masks[static_cast<size_t>(c)] = (1ull << Lc) - 1ull;
			chunk_basef[static_cast<size_t>(c)] = static_cast<float>(base);
			chunk_invBasef[static_cast<size_t>(c)] = 1.0f / static_cast<float>(base);

			for (int d = 0; d < dim; ++d)
			{
				pextMaskChunks[static_cast<size_t>(c) * static_cast<size_t>(dim) + static_cast<size_t>(d)] =
					make_mask(dim, Lc, d);
			}
		}

		for (int d = 0; d < dim; ++d)
		{
			pextMask[d] = make_mask(dim, chunk_bits[0], d);
			pextMask_perm[d] = pextMask[d];
		}

		scale = 1ull << (dim * chunk_bits[0]);

		pow2m1[0] = 0.0f;
		for (int i = 1; i <= levels; ++i)
		{
			pow2m1[static_cast<size_t>(i)] = ldexpf(1.0f, i) - 1.0f;
		}
	}

	__declspec(noalias) __forceinline float block_diameter(unsigned long long i1, unsigned long long i2) const noexcept
	{
		const unsigned long long varying = i1 ^ i2;

		switch (dim)
		{
		case  1: return sqrtf(block_diameter_acc<1>(varying, 0.0f));
		case  2: return sqrtf(block_diameter_acc<2>(varying, 0.0f));
		case  3: return sqrtf(block_diameter_acc<3>(varying, 0.0f));
		case  4: return sqrtf(block_diameter_acc<4>(varying, 0.0f));
		case  5: return sqrtf(block_diameter_acc<5>(varying, 0.0f));
		case  6: return sqrtf(block_diameter_acc<6>(varying, 0.0f));
		case  7: return sqrtf(block_diameter_acc<7>(varying, 0.0f));
		case  8: return sqrtf(block_diameter_acc<8>(varying, 0.0f));
		case  9: return sqrtf(block_diameter_acc<9>(varying, 0.0f));
		case 10: return sqrtf(block_diameter_acc<10>(varying, 0.0f));
		case 11: return sqrtf(block_diameter_acc<11>(varying, 0.0f));
		case 12: return sqrtf(block_diameter_acc<12>(varying, 0.0f));
		case 13: return sqrtf(block_diameter_acc<13>(varying, 0.0f));
		case 14: return sqrtf(block_diameter_acc<14>(varying, 0.0f));
		case 15: return sqrtf(block_diameter_acc<15>(varying, 0.0f));
		default: return sqrtf(block_diameter_acc<16>(varying, 0.0f));
		}
	}

	__declspec(noalias) __forceinline void map01ToPoint(float t, float* __restrict out) const noexcept
	{
		if (!reverseTraversal)
		{
			switch (dim)
			{
			case  1: map01ToPoint_t< 1>(t, out); return;
			case  2: map01ToPoint_t< 2>(t, out); return;
			case  3: map01ToPoint_t< 3>(t, out); return;
			case  4: map01ToPoint_t< 4>(t, out); return;
			case  5: map01ToPoint_t< 5>(t, out); return;
			case  6: map01ToPoint_t< 6>(t, out); return;
			case  7: map01ToPoint_t< 7>(t, out); return;
			case  8: map01ToPoint_t< 8>(t, out); return;
			case  9: map01ToPoint_t< 9>(t, out); return;
			case 10: map01ToPoint_t<10>(t, out); return;
			case 11: map01ToPoint_t<11>(t, out); return;
			case 12: map01ToPoint_t<12>(t, out); return;
			case 13: map01ToPoint_t<13>(t, out); return;
			case 14: map01ToPoint_t<14>(t, out); return;
			case 15: map01ToPoint_t<15>(t, out); return;
			default: map01ToPoint_t<16>(t, out); return;
			}
		}

		switch (dim)
		{
		case  1: map01ToPoint_t< 1>(fmaf(-t, 1.0f, 1.0f), out); return;
		case  2: map01ToPoint_t< 2>(fmaf(-t, 1.0f, 1.0f), out); return;
		case  3: map01ToPoint_t< 3>(fmaf(-t, 1.0f, 1.0f), out); return;
		case  4: map01ToPoint_t< 4>(fmaf(-t, 1.0f, 1.0f), out); return;
		case  5: map01ToPoint_t< 5>(fmaf(-t, 1.0f, 1.0f), out); return;
		case  6: map01ToPoint_t< 6>(fmaf(-t, 1.0f, 1.0f), out); return;
		case  7: map01ToPoint_t< 7>(fmaf(-t, 1.0f, 1.0f), out); return;
		case  8: map01ToPoint_t< 8>(fmaf(-t, 1.0f, 1.0f), out); return;
		case  9: map01ToPoint_t< 9>(fmaf(-t, 1.0f, 1.0f), out); return;
		case 10: map01ToPoint_t<10>(fmaf(-t, 1.0f, 1.0f), out); return;
		case 11: map01ToPoint_t<11>(fmaf(-t, 1.0f, 1.0f), out); return;
		case 12: map01ToPoint_t<12>(fmaf(-t, 1.0f, 1.0f), out); return;
		case 13: map01ToPoint_t<13>(fmaf(-t, 1.0f, 1.0f), out); return;
		case 14: map01ToPoint_t<14>(fmaf(-t, 1.0f, 1.0f), out); return;
		case 15: map01ToPoint_t<15>(fmaf(-t, 1.0f, 1.0f), out); return;
		default: map01ToPoint_t<16>(fmaf(-t, 1.0f, 1.0f), out); return;
		}
	}

	__declspec(noalias) __forceinline float pointToT(const float* __restrict q) const noexcept
	{
		if (!reverseTraversal)
		{
			switch (dim)
			{
			case  1: return pointToT_t< 1>(q);
			case  2: return pointToT_t< 2>(q);
			case  3: return pointToT_t< 3>(q);
			case  4: return pointToT_t< 4>(q);
			case  5: return pointToT_t< 5>(q);
			case  6: return pointToT_t< 6>(q);
			case  7: return pointToT_t< 7>(q);
			case  8: return pointToT_t< 8>(q);
			case  9: return pointToT_t< 9>(q);
			case 10: return pointToT_t<10>(q);
			case 11: return pointToT_t<11>(q);
			case 12: return pointToT_t<12>(q);
			case 13: return pointToT_t<13>(q);
			case 14: return pointToT_t<14>(q);
			case 15: return pointToT_t<15>(q);
			default: return pointToT_t<16>(q);
			}
		}

		switch (dim)
		{
		case  1: return fmaf(-pointToT_t< 1>(q), 1.0f, 1.0f);
		case  2: return fmaf(-pointToT_t< 2>(q), 1.0f, 1.0f);
		case  3: return fmaf(-pointToT_t< 3>(q), 1.0f, 1.0f);
		case  4: return fmaf(-pointToT_t< 4>(q), 1.0f, 1.0f);
		case  5: return fmaf(-pointToT_t< 5>(q), 1.0f, 1.0f);
		case  6: return fmaf(-pointToT_t< 6>(q), 1.0f, 1.0f);
		case  7: return fmaf(-pointToT_t< 7>(q), 1.0f, 1.0f);
		case  8: return fmaf(-pointToT_t< 8>(q), 1.0f, 1.0f);
		case  9: return fmaf(-pointToT_t< 9>(q), 1.0f, 1.0f);
		case 10: return fmaf(-pointToT_t<10>(q), 1.0f, 1.0f);
		case 11: return fmaf(-pointToT_t<11>(q), 1.0f, 1.0f);
		case 12: return fmaf(-pointToT_t<12>(q), 1.0f, 1.0f);
		case 13: return fmaf(-pointToT_t<13>(q), 1.0f, 1.0f);
		case 14: return fmaf(-pointToT_t<14>(q), 1.0f, 1.0f);
		case 15: return fmaf(-pointToT_t<15>(q), 1.0f, 1.0f);
		default: return fmaf(-pointToT_t<16>(q), 1.0f, 1.0f);
		}
	}
};

//============================================================
//                     AXPY / CLAMP UTILITIES
//============================================================

static __declspec(noalias) __forceinline
void agp_axpy_clamp_avx2(const float* __restrict q_base,
	const float* __restrict dir,
	float alpha,
	const float* __restrict q_lo,
	const float* __restrict q_hi,
	float* __restrict q_out,
	int n) noexcept
{
	__declspec(align(32)) const __m256 a = _mm256_set1_ps(alpha);

	int i = 0;
	const int limit = n & ~7;

	__pragma(loop(ivdep))
		while (i < limit)
		{
			const __m256 qb = _mm256_load_ps(q_base + i);
			const __m256 di = _mm256_load_ps(dir + i);
			const __m256 lo = _mm256_load_ps(q_lo + i);
			const __m256 hi = _mm256_load_ps(q_hi + i);

			__m256 q = _mm256_fmadd_ps(a, di, qb);
			q = _mm256_max_ps(lo, _mm256_min_ps(q, hi));

			_mm256_store_ps(q_out + i, q);
			i += 8;
		}

	__pragma(loop(ivdep))
		while (i < n)
		{
			float q = fmaf(alpha, dir[i], q_base[i]);
			q = (q < q_lo[i]) ? q_lo[i] : ((q > q_hi[i]) ? q_hi[i] : q);
			q_out[i] = q;
			++i;
		}
}

//============================================================
//                     MORTON CHUNK HELPERS
//============================================================

static __declspec(noalias) __forceinline
unsigned long long agp_t_to_firstchunk_idx_open(const MortonND& map, float t) noexcept
{
	const float u = map.reverseTraversal ? fmaf(-t, 1.0f, 1.0f) : t;

	unsigned long long idx;
	if (map.scale <= 0x01000000ull)
		idx = static_cast<unsigned long long>(u * static_cast<float>(map.scale));
	else
		idx = static_cast<unsigned long long>(static_cast<double>(u) * static_cast<double>(map.scale));

	return idx;
}

static __declspec(noalias) __forceinline
bool agp_can_have_positive_diameter(const MortonND& map, const float t1, const float t2) noexcept
{
	if (!static_cast<bool>(map.extra_levels))
	{
		const unsigned long long c1 = agp_t_to_firstchunk_idx_open(map, t1);
		const unsigned long long c2 = agp_t_to_firstchunk_idx_open(map, t2);
		return c1 != c2;
	}
	return true;
}

static __declspec(noalias) __forceinline
unsigned long long agp_bit_floor_u64(unsigned long long x) noexcept
{
	unsigned long idx = 0ul;
	_BitScanReverse64(&idx, x);
	return 1ull << idx;
}

static __declspec(noalias) __forceinline
int agp_firstchunk_span_bits_open(const unsigned long long c1, const unsigned long long c2) noexcept
{
	const unsigned long long diff = c1 ^ c2;
	if (!diff) return 0;
	unsigned long msb = 0ul;
	_BitScanReverse64(&msb, diff);
	return static_cast<int>(msb) + 1;
}

static __declspec(noalias) __forceinline
float agp_block_diameter_firstchunk_exact_open_t(const MortonND& map, float t1, float t2) noexcept
{
	if (t1 == t2) return 0.0f;

	unsigned long long a0 = agp_t_to_firstchunk_idx_open(map, t1);
	unsigned long long b0 = agp_t_to_firstchunk_idx_open(map, t2);

	if (!static_cast<bool>(map.extra_levels))
		return map.block_diameter(a0, b0);

	__declspec(align(32)) unsigned long long minCell[AGP_MAX_FULL_DIM];
	__declspec(align(32)) unsigned long long maxCell[AGP_MAX_FULL_DIM];

	int d = 0;
	__pragma(loop(ivdep))
		while (d < map.dim)
		{
			minCell[d] = ~0ull;
			maxCell[d] = 0ull;
			++d;
		}

	const int inv_shift0 = map.chunk_inv_shift[0];
	const unsigned long long chunk_mask0 = map.chunk_masks[0];
	const unsigned long long extra_mask =
		(map.extra_levels > 0) ? ((1ull << map.extra_levels) - 1ull) : 0ull;

	unsigned long long cur = a0;
	while (cur <= b0)
	{
		const unsigned long long rem = b0 - cur + 1ull;

		unsigned long long block;
		if (cur != 0ull)
		{
			block = cur & (~cur + 1ull);
			while (block > rem) block >>= 1ull;
		}
		else
		{
			block = agp_bit_floor_u64(rem);
		}

		if (block == 0ull) block = 1ull;

		const unsigned long long free_interleaved_mask = block - 1ull;
		const unsigned long long g_fixed = gray_encode(cur) & ~free_interleaved_mask;

		int i = 0;
		__pragma(loop(ivdep))
			while (i < map.dim)
			{
				const int pd = map.perm[i];
				const unsigned long long maskI = map.pextMask[i];
				const unsigned long long invI =
					(map.invMask[pd] >> inv_shift0) & chunk_mask0;

				const unsigned long long fixedComp =
					_pext_u64(g_fixed, maskI) ^ invI;

				const int freeCnt =
					static_cast<int>(_mm_popcnt_u64(maskI & free_interleaved_mask));

				const unsigned long long freeCompMask =
					(freeCnt > 0) ? ((1ull << freeCnt) - 1ull) : 0ull;

				const unsigned long long blockMinChunk =
					fixedComp & ~freeCompMask;

				const unsigned long long blockMaxChunk =
					blockMinChunk | freeCompMask;

				const unsigned long long blockMinCell =
					blockMinChunk << map.extra_levels;

				const unsigned long long blockMaxCell =
					(blockMaxChunk << map.extra_levels) | extra_mask;

				if (blockMinCell < minCell[pd]) minCell[pd] = blockMinCell;
				if (blockMaxCell > maxCell[pd]) maxCell[pd] = blockMaxCell;

				++i;
			}

		cur += block;
		if (cur == 0ull) break;
	}

	float s2 = 0.0f;
	d = 0;
	while (d < map.dim)
	{
		const float range =
			map.step[d] * static_cast<float>(maxCell[d] - minCell[d]);
		s2 = fmaf(range, range, s2);
		++d;
	}

	return sqrtf(s2);
}

//============================================================
//                     INTERVAL TYPES
//============================================================

struct IntervalWire final
{
	float               x1, x2;
	float               y1, y2;
	float               N_factor;
	float               quadratic_term;
	float               M;
	float               diam;
	unsigned long long  i1, i2;
	unsigned            span_pack;

	template<typename Archive>
	__declspec(noalias) __forceinline void serialize(Archive& ar, unsigned)
	{
		ar& x1& x2& y1& y2
			& N_factor& quadratic_term& M& diam
			& i1& i2& span_pack;
	}
};

struct MultiCrossMsg final
{
	IntervalWire    intervals[AGP_MULTI_MAX_COUNT];
	unsigned        count;

	template<typename Archive>
	__declspec(noalias) __forceinline void serialize(Archive& ar, unsigned)
	{
		ar& intervals& count;
	}
};

struct BestSolutionMsg
{
	float       bestF;
	float       bestX;
	float       bestY;
	float       bestA;
	float       bestQ[AGP_MAX_FULL_DIM];
	unsigned    dim;
	unsigned    bestIndex;

	template<typename Archive>
	__declspec(noalias) __forceinline void serialize(Archive& ar, unsigned)
	{
		ar& bestF& bestX& bestY& bestA& dim& bestIndex& bestQ;
	}
};

static __forceinline bool better_indexed(unsigned lhsIndex, float lhsValue, unsigned rhsIndex, float rhsValue) noexcept
{
	return (lhsIndex > rhsIndex) || ((lhsIndex == rhsIndex) && (lhsValue < rhsValue));
}

static __forceinline void InitBestSolutionMsg(BestSolutionMsg& msg) noexcept
{
	msg.bestF = FLT_MAX;
	msg.bestX = msg.bestY = msg.bestA = 0.0f;
	msg.dim = msg.bestIndex = 0u;
	for (unsigned i = 0u; i < AGP_MAX_FULL_DIM; ++i)
		msg.bestQ[i] = 0.0f;
}

static __forceinline bool FillBestSolutionMsg(
	BestSolutionMsg& msg,
	unsigned bestIndex,
	float bestValue,
	float bestX,
	float bestY,
	float bestA,
	const std::vector<float, boost::alignment::aligned_allocator<float, 32u>>& bestQ) noexcept
{
	msg.bestF = bestValue;
	msg.bestX = bestX;
	msg.bestY = bestY;
	msg.bestA = bestA;
	msg.bestIndex = bestIndex;
	msg.dim = static_cast<unsigned>((bestQ.size() > AGP_MAX_FULL_DIM) ? AGP_MAX_FULL_DIM : bestQ.size());
	for (unsigned i = 0u; i < msg.dim; ++i)
		msg.bestQ[i] = bestQ[i];
	for (unsigned i = msg.dim; i < AGP_MAX_FULL_DIM; ++i)
		msg.bestQ[i] = 0.0f;
	return true;
}

static __forceinline bool UpdateIndexedBestFromMessage(
	const BestSolutionMsg& msg,
	unsigned fullConstraintIndex,
	unsigned& bestIndexFound,
	float& bestIndexValue,
	std::vector<float, boost::alignment::aligned_allocator<float, 32u>>& bestQIndexed,
	float& bestIndexedX,
	float& bestIndexedY,
	float& bestIndexedA,
	float& bestF,
	std::vector<float, boost::alignment::aligned_allocator<float, 32u>>& bestQ,
	float& bestX,
	float& bestY,
	float& bestA) noexcept
{
	if (!better_indexed(msg.bestIndex, msg.bestF, bestIndexFound, bestIndexValue))
		return false;

	const SIZE_T safeDim = static_cast<SIZE_T>(msg.dim) * sizeof(float);

	bestIndexFound = msg.bestIndex;
	bestIndexValue = msg.bestF;
	bestIndexedX = msg.bestX;
	bestIndexedY = msg.bestY;
	bestIndexedA = msg.bestA;
	memcpy(bestQIndexed.data(), msg.bestQ, safeDim);

	if (msg.bestIndex == fullConstraintIndex)
	{
		bestF = msg.bestF;
		bestX = msg.bestX;
		bestY = msg.bestY;
		bestA = msg.bestA;
		memcpy(bestQ.data(), msg.bestQ, safeDim);
	}
	return true;
}

//============================================================
//                     OBSTACLE / COLLISION HELPERS
//============================================================

struct SquareObstacle final
{
	float cx, cy, half, pad;
	float minX, minY, maxX, maxY;
};

static __declspec(noalias) __forceinline bool point_in_aabb(float px, float py, float minX, float minY, float maxX, float maxY) noexcept
{
	return px >= minX && px <= maxX && py >= minY && py <= maxY;
}

static __declspec(noalias) __forceinline float point_aabb_distance_sq(float px, float py, float minX, float minY, float maxX, float maxY) noexcept
{
	float dx = 0.0f;
	if (px < minX) dx = minX - px;
	else if (px > maxX) dx = px - maxX;
	float dy = 0.0f;
	if (py < minY) dy = minY - py;
	else if (py > maxY) dy = py - maxY;
	return fmaf(dx, dx, fmaf(dy, dy, 0.0f));
}

static __declspec(noalias) __forceinline float point_segment_distance_sq(float px, float py, float ax, float ay, float bx, float by) noexcept
{
	const float abx = fmaf(bx, 1.0f, -ax);
	const float aby = fmaf(by, 1.0f, -ay);
	const float apx = fmaf(px, 1.0f, -ax);
	const float apy = fmaf(py, 1.0f, -ay);
	const float ab2 = fmaf(abx, abx, fmaf(aby, aby, 0.0f));
	const float t = fmaf(1.0f / ab2, fmaf(apx, abx, fmaf(apy, aby, 0.0f)), 0.0f);
	const float clamped_t = agp_clamp_unit_open_scalar(t);
	const float qx = fmaf(clamped_t, abx, ax);
	const float qy = fmaf(clamped_t, aby, ay);
	const float dx = fmaf(px, 1.0f, -qx);
	const float dy = fmaf(py, 1.0f, -qy);
	return fmaf(dx, dx, fmaf(dy, dy, 0.0f));
}

static __declspec(noalias) __forceinline float orient2d(float ax, float ay, float bx, float by, float cx, float cy) noexcept
{
	return fmaf(bx - ax, cy - ay, -(by - ay) * (cx - ax));
}

static __declspec(noalias) __forceinline bool on_segment(float ax, float ay, float bx, float by, float px, float py) noexcept
{
	float eps = 1e-6f;
	return px >= (ax < bx ? ax : bx) - eps && px <= (ax > bx ? ax : bx) + eps &&
		py >= (ay < by ? ay : by) - eps && py <= (ay > by ? ay : by) + eps;
}

static __declspec(noalias) __forceinline bool segments_intersect(float ax, float ay, float bx, float by,
	float cx, float cy, float dx, float dy) noexcept
{
	float o1 = orient2d(ax, ay, bx, by, cx, cy);
	float o2 = orient2d(ax, ay, bx, by, dx, dy);
	float o3 = orient2d(cx, cy, dx, dy, ax, ay);
	float o4 = orient2d(cx, cy, dx, dy, bx, by);
	float eps = 1e-6f;

	if (((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
		((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps)))
		return true;

	if (fabsf(o1) <= eps && on_segment(ax, ay, bx, by, cx, cy)) return true;
	if (fabsf(o2) <= eps && on_segment(ax, ay, bx, by, dx, dy)) return true;
	if (fabsf(o3) <= eps && on_segment(cx, cy, dx, dy, ax, ay)) return true;
	if (fabsf(o4) <= eps && on_segment(cx, cy, dx, dy, bx, by)) return true;

	return false;
}

static __declspec(noalias) __forceinline float segment_segment_distance_sq(float ax, float ay, float bx, float by,
	float cx, float cy, float dx, float dy) noexcept
{
	if (segments_intersect(ax, ay, bx, by, cx, cy, dx, dy))
		return 0.0f;

	float best = point_segment_distance_sq(ax, ay, cx, cy, dx, dy);
	float d1 = point_segment_distance_sq(bx, by, cx, cy, dx, dy);
	float d2 = point_segment_distance_sq(cx, cy, ax, ay, bx, by);
	float d3 = point_segment_distance_sq(dx, dy, ax, ay, bx, by);

	if (d1 < best) best = d1;
	if (d2 < best) best = d2;
	if (d3 < best) best = d3;
	return best;
}

static __declspec(noalias) __forceinline float segment_aabb_distance_sq(float ax, float ay, float bx, float by,
	float minX, float minY, float maxX, float maxY) noexcept
{
	if (point_in_aabb(ax, ay, minX, minY, maxX, maxY) ||
		point_in_aabb(bx, by, minX, minY, maxX, maxY))
		return 0.0f;

	if (segments_intersect(ax, ay, bx, by, minX, minY, maxX, minY) ||
		segments_intersect(ax, ay, bx, by, maxX, minY, maxX, maxY) ||
		segments_intersect(ax, ay, bx, by, maxX, maxY, minX, maxY) ||
		segments_intersect(ax, ay, bx, by, minX, maxY, minX, minY))
		return 0.0f;

	float best = point_aabb_distance_sq(ax, ay, minX, minY, maxX, maxY);
	float d0 = point_aabb_distance_sq(bx, by, minX, minY, maxX, maxY);
	float d1 = segment_segment_distance_sq(ax, ay, bx, by, minX, minY, maxX, minY);
	float d2 = segment_segment_distance_sq(ax, ay, bx, by, maxX, minY, maxX, maxY);
	float d3 = segment_segment_distance_sq(ax, ay, bx, by, maxX, maxY, minX, maxY);
	float d4 = segment_segment_distance_sq(ax, ay, bx, by, minX, maxY, minX, minY);

	if (d0 < best) best = d0;
	if (d1 < best) best = d1;
	if (d2 < best) best = d2;
	if (d3 < best) best = d3;
	if (d4 < best) best = d4;
	return best;
}

static __declspec(noalias) __forceinline float polyline_square_violation(
	const float* __restrict px,
	const float* __restrict py,
	int nSeg,
	const SquareObstacle& ob,
	float clearance) noexcept
{
	__declspec(align(32)) const __m256 minX = _mm256_set1_ps(ob.minX);
	__declspec(align(32)) const __m256 minY = _mm256_set1_ps(ob.minY);
	__declspec(align(32)) const __m256 maxX = _mm256_set1_ps(ob.maxX);
	__declspec(align(32)) const __m256 maxY = _mm256_set1_ps(ob.maxY);

	auto point_in_aabb = [&](__m256 pxv, __m256 pyv) -> __m256
	{
		__m256 in_x = _mm256_and_ps(_mm256_cmp_ps(pxv, minX, _CMP_GE_OQ),
			_mm256_cmp_ps(pxv, maxX, _CMP_LE_OQ));
		__m256 in_y = _mm256_and_ps(_mm256_cmp_ps(pyv, minY, _CMP_GE_OQ),
			_mm256_cmp_ps(pyv, maxY, _CMP_LE_OQ));
		return _mm256_and_ps(in_x, in_y);
	};

	auto point_aabb_distance_sq = [&](__m256 pxv, __m256 pyv) -> __m256
	{
		__m256 cx = _mm256_max_ps(minX, _mm256_min_ps(maxX, pxv));
		__m256 cy = _mm256_max_ps(minY, _mm256_min_ps(maxY, pyv));
		__m256 dx = _mm256_sub_ps(pxv, cx);
		__m256 dy = _mm256_sub_ps(pyv, cy);
		return _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
	};

	auto orient2d = [&](__m256 ax, __m256 ay, __m256 bx, __m256 by, __m256 cx, __m256 cy) -> __m256
	{
		__m256 bx_ax = _mm256_sub_ps(bx, ax);
		__m256 by_ay = _mm256_sub_ps(by, ay);
		__m256 cx_ax = _mm256_sub_ps(cx, ax);
		__m256 cy_ay = _mm256_sub_ps(cy, ay);
		return _mm256_sub_ps(_mm256_mul_ps(bx_ax, cy_ay), _mm256_mul_ps(by_ay, cx_ax));
	};

	auto on_segment = [&](__m256 ax, __m256 ay, __m256 bx, __m256 by, __m256 px, __m256 py) -> __m256
	{
		__m256 min_x = _mm256_min_ps(ax, bx);
		__m256 max_x = _mm256_max_ps(ax, bx);
		__m256 min_y = _mm256_min_ps(ay, by);
		__m256 max_y = _mm256_max_ps(ay, by);
		__m256 in_x = _mm256_and_ps(_mm256_cmp_ps(px, _mm256_sub_ps(min_x, veps), _CMP_GE_OQ),
			_mm256_cmp_ps(px, _mm256_add_ps(max_x, veps), _CMP_LE_OQ));
		__m256 in_y = _mm256_and_ps(_mm256_cmp_ps(py, _mm256_sub_ps(min_y, veps), _CMP_GE_OQ),
			_mm256_cmp_ps(py, _mm256_add_ps(max_y, veps), _CMP_LE_OQ));
		return _mm256_and_ps(in_x, in_y);
	};

	auto segments_intersect = [&](__m256 ax, __m256 ay, __m256 bx, __m256 by,
		__m256 cx, __m256 cy, __m256 dx, __m256 dy) -> __m256
	{
		__m256 o1 = orient2d(ax, ay, bx, by, cx, cy);
		__m256 o2 = orient2d(ax, ay, bx, by, dx, dy);
		__m256 o3 = orient2d(cx, cy, dx, dy, ax, ay);
		__m256 o4 = orient2d(cx, cy, dx, dy, bx, by);

		__m256 o1_pos = _mm256_cmp_ps(o1, veps, _CMP_GT_OQ);
		__m256 o1_neg = _mm256_cmp_ps(o1, _mm256_sub_ps(VEC_ZERO, veps), _CMP_LT_OQ);
		__m256 o2_pos = _mm256_cmp_ps(o2, veps, _CMP_GT_OQ);
		__m256 o2_neg = _mm256_cmp_ps(o2, _mm256_sub_ps(VEC_ZERO, veps), _CMP_LT_OQ);
		__m256 o3_pos = _mm256_cmp_ps(o3, veps, _CMP_GT_OQ);
		__m256 o3_neg = _mm256_cmp_ps(o3, _mm256_sub_ps(VEC_ZERO, veps), _CMP_LT_OQ);
		__m256 o4_pos = _mm256_cmp_ps(o4, veps, _CMP_GT_OQ);
		__m256 o4_neg = _mm256_cmp_ps(o4, _mm256_sub_ps(VEC_ZERO, veps), _CMP_LT_OQ);

		__m256 cond1 = _mm256_or_ps(_mm256_and_ps(o1_pos, o2_neg),
			_mm256_and_ps(o1_neg, o2_pos));
		__m256 cond2 = _mm256_or_ps(_mm256_and_ps(o3_pos, o4_neg),
			_mm256_and_ps(o3_neg, o4_pos));
		__m256 cross = _mm256_and_ps(cond1, cond2);

		__m256 o1_abs = _mm256_and_ps(o1, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
		__m256 o2_abs = _mm256_and_ps(o2, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
		__m256 o3_abs = _mm256_and_ps(o3, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
		__m256 o4_abs = _mm256_and_ps(o4, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
		__m256 collinear = _mm256_and_ps(_mm256_cmp_ps(o1_abs, veps, _CMP_LE_OQ),
			_mm256_cmp_ps(o2_abs, veps, _CMP_LE_OQ));
		collinear = _mm256_and_ps(collinear, _mm256_cmp_ps(o3_abs, veps, _CMP_LE_OQ));
		collinear = _mm256_and_ps(collinear, _mm256_cmp_ps(o4_abs, veps, _CMP_LE_OQ));

		__m256 min_ab_x = _mm256_min_ps(ax, bx);
		__m256 max_ab_x = _mm256_max_ps(ax, bx);
		__m256 min_cd_x = _mm256_min_ps(cx, dx);
		__m256 max_cd_x = _mm256_max_ps(cx, dx);
		__m256 min_ab_y = _mm256_min_ps(ay, by);
		__m256 max_ab_y = _mm256_max_ps(ay, by);
		__m256 min_cd_y = _mm256_min_ps(cy, dy);
		__m256 max_cd_y = _mm256_max_ps(cy, dy);

		__m256 overlap_x = _mm256_and_ps(_mm256_cmp_ps(min_ab_x, max_cd_x, _CMP_LE_OQ),
			_mm256_cmp_ps(min_cd_x, max_ab_x, _CMP_LE_OQ));
		__m256 overlap_y = _mm256_and_ps(_mm256_cmp_ps(min_ab_y, max_cd_y, _CMP_LE_OQ),
			_mm256_cmp_ps(min_cd_y, max_ab_y, _CMP_LE_OQ));
		__m256 overlap = _mm256_and_ps(overlap_x, overlap_y);
		__m256 collinear_intersect = _mm256_and_ps(collinear, overlap);

		__m256 on_c_ab = _mm256_and_ps(_mm256_cmp_ps(o1_abs, veps, _CMP_LE_OQ),
			on_segment(ax, ay, bx, by, cx, cy));
		__m256 on_d_ab = _mm256_and_ps(_mm256_cmp_ps(o2_abs, veps, _CMP_LE_OQ),
			on_segment(ax, ay, bx, by, dx, dy));
		__m256 on_a_cd = _mm256_and_ps(_mm256_cmp_ps(o3_abs, veps, _CMP_LE_OQ),
			on_segment(cx, cy, dx, dy, ax, ay));
		__m256 on_b_cd = _mm256_and_ps(_mm256_cmp_ps(o4_abs, veps, _CMP_LE_OQ),
			on_segment(cx, cy, dx, dy, bx, by));
		__m256 endpoint_intersect = _mm256_or_ps(_mm256_or_ps(on_c_ab, on_d_ab),
			_mm256_or_ps(on_a_cd, on_b_cd));

		return _mm256_or_ps(_mm256_or_ps(cross, collinear_intersect), endpoint_intersect);
	};

	auto point_segment_distance_sq = [&](__m256 px, __m256 py,
		__m256 ax, __m256 ay,
		__m256 bx, __m256 by) -> __m256
	{
		__m256 abx = _mm256_sub_ps(bx, ax);
		__m256 aby = _mm256_sub_ps(by, ay);
		__m256 apx = _mm256_sub_ps(px, ax);
		__m256 apy = _mm256_sub_ps(py, ay);
		__m256 ab2 = _mm256_fmadd_ps(abx, abx, _mm256_mul_ps(aby, aby));
		__m256 inv_ab2 = _mm256_rcp_ps(ab2);
		inv_ab2 = _mm256_mul_ps(inv_ab2, _mm256_fnmadd_ps(ab2, inv_ab2, _mm256_set1_ps(2.0f)));
		__m256 t = _mm256_mul_ps(_mm256_fmadd_ps(apx, abx, _mm256_mul_ps(apy, aby)), inv_ab2);
		t = _mm256_max_ps(VEC_ZERO, _mm256_min_ps(VEC_COS_P0, t));
		__m256 qx = _mm256_fmadd_ps(t, abx, ax);
		__m256 qy = _mm256_fmadd_ps(t, aby, ay);
		__m256 dx = _mm256_sub_ps(px, qx);
		__m256 dy = _mm256_sub_ps(py, qy);
		return _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
	};

	auto segment_segment_distance_sq = [&](__m256 ax, __m256 ay, __m256 bx, __m256 by,
		__m256 cx, __m256 cy, __m256 dx, __m256 dy) -> __m256
	{
		__m256 intersect = segments_intersect(ax, ay, bx, by, cx, cy, dx, dy);
		__m256 d2 = _mm256_blendv_ps(vflt_max, VEC_ZERO, intersect);

		__m256 d2a = point_segment_distance_sq(ax, ay, cx, cy, dx, dy);
		__m256 d2b = point_segment_distance_sq(bx, by, cx, cy, dx, dy);
		__m256 d2c = point_segment_distance_sq(cx, cy, ax, ay, bx, by);
		__m256 d2d = point_segment_distance_sq(dx, dy, ax, ay, bx, by);
		__m256 min_d2 = _mm256_min_ps(_mm256_min_ps(d2a, d2b), _mm256_min_ps(d2c, d2d));
		return _mm256_blendv_ps(min_d2, VEC_ZERO, intersect);
	};

	auto segment_aabb_distance_sq_ = [&](__m256 ax, __m256 ay, __m256 bx, __m256 by) -> __m256
	{
		__m256 a_in = point_in_aabb(ax, ay);
		__m256 b_in = point_in_aabb(bx, by);
		__m256 in_mask = _mm256_or_ps(a_in, b_in);
		__m256 d2 = _mm256_blendv_ps(vflt_max, VEC_ZERO, in_mask);

		__m256 bottom_intersect = segments_intersect(ax, ay, bx, by, minX, minY, maxX, minY);
		__m256 right_intersect = segments_intersect(ax, ay, bx, by, maxX, minY, maxX, maxY);
		__m256 top_intersect = segments_intersect(ax, ay, bx, by, maxX, maxY, minX, maxY);
		__m256 left_intersect = segments_intersect(ax, ay, bx, by, minX, maxY, minX, minY);
		__m256 edge_intersect = _mm256_or_ps(_mm256_or_ps(bottom_intersect, right_intersect),
			_mm256_or_ps(top_intersect, left_intersect));
		d2 = _mm256_blendv_ps(d2, VEC_ZERO, edge_intersect);

		__m256 d2_a = point_aabb_distance_sq(ax, ay);
		__m256 d2_b = point_aabb_distance_sq(bx, by);
		__m256 min_end = _mm256_min_ps(d2_a, d2_b);

		__m256 d2_bottom = segment_segment_distance_sq(ax, ay, bx, by, minX, minY, maxX, minY);
		__m256 d2_right = segment_segment_distance_sq(ax, ay, bx, by, maxX, minY, maxX, maxY);
		__m256 d2_top = segment_segment_distance_sq(ax, ay, bx, by, maxX, maxY, minX, maxY);
		__m256 d2_left = segment_segment_distance_sq(ax, ay, bx, by, minX, maxY, minX, minY);

		__m256 min_edge = _mm256_min_ps(_mm256_min_ps(d2_bottom, d2_right),
			_mm256_min_ps(d2_top, d2_left));
		__m256 best = _mm256_min_ps(min_end, min_edge);
		return _mm256_blendv_ps(best, VEC_ZERO, _mm256_or_ps(in_mask, edge_intersect));
	};

	__m256 bestD2 = vflt_max;
	int i = 0;
	const int limit = nSeg & ~7;
	__pragma(loop(ivdep))
		while (i < limit)
		{
			__m256 x1 = _mm256_load_ps(px + i);
			__m256 y1 = _mm256_load_ps(py + i);
			__m256 x2 = _mm256_loadu_ps(px + i + 1);
			__m256 y2 = _mm256_loadu_ps(py + i + 1);
			__m256 d2 = segment_aabb_distance_sq_(x1, y1, x2, y2);
			bestD2 = _mm256_min_ps(bestD2, d2);
			i += 8;
		}

	__m256 min8 = bestD2;
	__m128 lo = _mm256_castps256_ps128(min8);
	__m128 hi = _mm256_extractf128_ps(min8, 1);
	__m128 min4 = _mm_min_ps(lo, hi);
	__m128 min2 = _mm_min_ps(min4, _mm_shuffle_ps(min4, min4, _MM_SHUFFLE(2, 3, 0, 1)));
	__m128 min1 = _mm_min_ss(min2, _mm_shuffle_ps(min2, min2, _MM_SHUFFLE(1, 0, 0, 0)));
	float best = _mm_cvtss_f32(min1);

	while (i < nSeg)
	{
		float d2 = segment_aabb_distance_sq(
			px[i], py[i], px[i + 1], py[i + 1],
			ob.minX, ob.minY, ob.maxX, ob.maxY
		);
		if (d2 < best) best = d2;
		if (best < 0.0f) break;
		++i;
	}

	return clearance - sqrtf(best);
}

//============================================================
//                     SLAB ALLOCATOR
//============================================================

struct Slab final
{
	char* current;

	__forceinline Slab(void* memory, size_t usable) noexcept
		: current(static_cast<char*>(memory)) {}
};

static tbb::enumerable_thread_specific<Slab*> tls([]() noexcept
	{
		void* memory = _aligned_malloc(AGP_INTERVAL_SLAB_BYTES, 64u);
		memset(memory, 0, AGP_INTERVAL_SLAB_BYTES);
		Slab* slab = static_cast<Slab*>(_aligned_malloc(sizeof(Slab), 64u));
		new (slab) Slab(memory, AGP_INTERVAL_SLAB_BYTES);
		return slab;
	});

//============================================================
//                     INTERVAL ND
//============================================================

struct __declspec(align(64)) IntervalND final
{
	float               x1, x2;
	float               y1, y2;
	float               delta_y;
	float               ordinate_factor;
	float               N_factor;
	float               quadratic_term;
	float               M;
	float               R;
	unsigned long long  i1, i2;
	float               diam;

	union
	{
		struct
		{
			unsigned short  span_level;
			unsigned char   idx1;
			unsigned char   idx2;
		};
		unsigned span_pack;
	};

	static __forceinline IntervalND* Make(const float _x1, const float _x2,
		const float _y1, const float _y2,
		const unsigned _idx1, const unsigned _idx2) noexcept
	{
		Slab* s = tls.local();
		char* r = s->current;
		s->current += 64;
		return new (r) IntervalND(_x1, _x2, _y1, _y2, _idx1, _idx2);
	}

	__declspec(noalias) __forceinline
		IntervalND(const float _x1, const float _x2, const float _y1, const float _y2, const unsigned _idx1, const unsigned _idx2) noexcept
		: x1(_x1)
		, x2(_x2)
		, y1(_y1)
		, y2(_y2)
		, delta_y(_y2 - _y1)
		, ordinate_factor(fmaf(y1, -2.0f, -2.0f * y2))
		, span_pack((static_cast<unsigned>(_idx2 & 0xFFu) << 24) |
			(static_cast<unsigned>(_idx1 & 0xFFu) << 16))
	{}

	__declspec(noalias) __forceinline void compute_span_level(const struct MortonND& map) noexcept;

	__declspec(noalias) __forceinline
		void set_metric(const float d_alpha) noexcept
	{
		N_factor = d_alpha;
		if (idx1 == idx2)
		{
			const float inv = 1.0f / N_factor;
			quadratic_term = fmaf(inv, fmaf(delta_y, delta_y, 0.0f), 0.0f);
			M = fmaf(inv, fabsf(delta_y), 0.0f);
		}
		else
		{
			quadratic_term = 0.0f;
			M = 0.0f;
		}
	}

	__declspec(noalias) __forceinline
		void ChangeCharacteristic(const float _m, const float _inv_m) noexcept
	{
		if (idx1 == idx2) R = fmaf(_inv_m, quadratic_term, fmaf(_m, N_factor, ordinate_factor));
		else if (idx1 < idx2) R = fmaf(2.0f * _m, N_factor, -4.0f * y2);
		else R = fmaf(2.0f * _m, N_factor, -4.0f * y1);
	}

	__declspec(noalias) __forceinline
		void ChangeCharacteristicConstM(const float m, const float inv_m, const float two_m) noexcept
	{
		R = idx1 == idx2 ? fmaf(inv_m, quadratic_term, fmaf(m, N_factor, ordinate_factor)) : fmaf(two_m, N_factor, -4.0f * ((idx1 < idx2) ? y2 : y1));
	}

	__declspec(noalias) __forceinline
		void ChangeCharacteristicAffine(const float GF, const float alpha) noexcept
	{
		const float m = fmaf(GF, N_factor, alpha * M);
		R = idx1 == idx2 ? fmaf(1.0f / m, quadratic_term, fmaf(m, N_factor, ordinate_factor)) : fmaf(2.0f * m, N_factor, -4.0f * ((idx1 < idx2) ? y2 : y1));
	}
};

using IntervalHeap =
std::vector<IntervalND*, boost::alignment::aligned_allocator<IntervalND*, 32u>>;

static __declspec(noalias) __forceinline
void heap_sift_up(IntervalHeap& H, size_t pos) noexcept
{
	IntervalND* v = H[pos];
	const float vR = v->R;
	while (pos > 0u)
	{
		const size_t parent = (pos - 1u) >> 1u;
		IntervalND* p = H[parent];
		if (p->R > vR) break;
		H[pos] = p;
		pos = parent;
	}
	H[pos] = v;
}

static __declspec(noalias) __forceinline
void heap_sift_down(IntervalHeap& H, size_t pos) noexcept
{
	const size_t n = H.size();
	IntervalND* v = H[pos];
	const float vR = v->R;
	size_t child = (pos << 1u) + 1u;

	while (child < n)
	{
		size_t best = child;
		const size_t right = child + 1u;
		if (right < n && H[best]->R < H[right]->R) {
			best = right;
		}
		if (H[best]->R < vR) break;
		H[pos] = H[best];
		pos = best;
		child = (pos << 1u) + 1u;
	}

	H[pos] = v;
}

static __declspec(noalias) __forceinline
void heap_make(IntervalHeap& H) noexcept
{
	size_t i = H.size() >> 1u;
	while (i > 0u) {
		--i;
		heap_sift_down(H, i);
	}
}

static __declspec(noalias) __forceinline
void heap_push(IntervalHeap& H, IntervalND* p) noexcept
{
	H.push_back(p);
	heap_sift_up(H, H.size() - 1u);
}

static __declspec(noalias) __forceinline
IntervalND* heap_pop_front(IntervalHeap& H) noexcept
{
	IntervalND* top = H[0];
	H[0] = H.back();
	H.pop_back();
	heap_sift_down(H, 0u);
	return top;
}

//============================================================
//                     INTERVAL SERIALIZATION HELPERS
//============================================================

static __declspec(noalias) __forceinline
void AgpFillIntervalWire(IntervalWire& w, const IntervalND* I) noexcept
{
	w.x1 = I->x1;
	w.x2 = I->x2;
	w.y1 = I->y1;
	w.y2 = I->y2;
	w.N_factor = I->N_factor;
	w.quadratic_term = I->quadratic_term;
	w.M = I->M;
	w.diam = I->diam;
	w.i1 = I->i1;
	w.i2 = I->i2;
	w.span_pack = I->span_pack;
}

static __declspec(noalias) __forceinline
IntervalND* AgpCreateIntervalFromWire(const IntervalWire& w) noexcept
{
	IntervalND* I = IntervalND::Make(w.x1, w.x2, w.y1, w.y2, 0u, 0u);
	I->i1 = w.i1;
	I->i2 = w.i2;
	I->diam = w.diam;
	I->N_factor = w.N_factor;
	I->quadratic_term = w.quadratic_term;
	I->M = w.M;
	I->span_pack = w.span_pack;
	return I;
}

//============================================================
//                     RECOMPUTE R VECTORIZED
//============================================================

static __forceinline void agp_store8_R(
	IntervalND** __restrict data,
	size_t i,
	__m256 r) noexcept
{
	__m128 lo = _mm256_castps256_ps128(r);
	__m128 hi = _mm256_extractf128_ps(r, 1);

	_mm_store_ss(&data[i + 0u]->R, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(0, 0, 0, 0)));
	_mm_store_ss(&data[i + 1u]->R, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 1, 1, 1)));
	_mm_store_ss(&data[i + 2u]->R, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 2, 2, 2)));
	_mm_store_ss(&data[i + 3u]->R, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(3, 3, 3, 3)));
	_mm_store_ss(&data[i + 4u]->R, _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(0, 0, 0, 0)));
	_mm_store_ss(&data[i + 5u]->R, _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(1, 1, 1, 1)));
	_mm_store_ss(&data[i + 6u]->R, _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(2, 2, 2, 2)));
	_mm_store_ss(&data[i + 7u]->R, _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(3, 3, 3, 3)));
}

static __declspec(noalias) __forceinline
void RecomputeR_ConstM_Mixed_ND(
	IntervalND** __restrict data,
	const size_t sz,
	const float m,
	const float inv_m) noexcept
{
	const __m256 vm = _mm256_set1_ps(m);
	const __m256 vinv_m = _mm256_set1_ps(inv_m);
	const __m256 vtwo_m = _mm256_set1_ps(m + m);
	const __m256 vneg4 = _mm256_set1_ps(-4.0f);

	size_t i = 0u;
	const size_t limit = sz & ~7ull;

	while (i < limit)
	{
		IntervalND* I0 = data[i + 0u];
		IntervalND* I1 = data[i + 1u];
		IntervalND* I2 = data[i + 2u];
		IntervalND* I3 = data[i + 3u];
		IntervalND* I4 = data[i + 4u];
		IntervalND* I5 = data[i + 5u];
		IntervalND* I6 = data[i + 6u];
		IntervalND* I7 = data[i + 7u];

		const __m256 vN = _mm256_set_ps(
			I7->N_factor, I6->N_factor, I5->N_factor, I4->N_factor,
			I3->N_factor, I2->N_factor, I1->N_factor, I0->N_factor);

		const __m256 vQ = _mm256_set_ps(
			I7->quadratic_term, I6->quadratic_term, I5->quadratic_term, I4->quadratic_term,
			I3->quadratic_term, I2->quadratic_term, I1->quadratic_term, I0->quadratic_term);

		const __m256 vOrd = _mm256_set_ps(
			I7->ordinate_factor, I6->ordinate_factor, I5->ordinate_factor, I4->ordinate_factor,
			I3->ordinate_factor, I2->ordinate_factor, I1->ordinate_factor, I0->ordinate_factor);

		const __m256 vy1 = _mm256_set_ps(
			I7->y1, I6->y1, I5->y1, I4->y1,
			I3->y1, I2->y1, I1->y1, I0->y1);

		const __m256 vy2 = _mm256_set_ps(
			I7->y2, I6->y2, I5->y2, I4->y2,
			I3->y2, I2->y2, I1->y2, I0->y2);

		const __m256 same =
			_mm256_fmadd_ps(vinv_m, vQ, _mm256_fmadd_ps(vm, vN, vOrd));

		const __m256 left =
			_mm256_fmadd_ps(vtwo_m, vN, _mm256_mul_ps(vneg4, vy2));

		const __m256 right =
			_mm256_fmadd_ps(vtwo_m, vN, _mm256_mul_ps(vneg4, vy1));

		const __m256i idx1 = _mm256_set_epi32(
			I7->idx1, I6->idx1, I5->idx1, I4->idx1,
			I3->idx1, I2->idx1, I1->idx1, I0->idx1);

		const __m256i idx2 = _mm256_set_epi32(
			I7->idx2, I6->idx2, I5->idx2, I4->idx2,
			I3->idx2, I2->idx2, I1->idx2, I0->idx2);

		const __m256 maskEq =
			_mm256_castsi256_ps(_mm256_cmpeq_epi32(idx1, idx2));

		const __m256 maskLt =
			_mm256_castsi256_ps(_mm256_cmpgt_epi32(idx2, idx1));

		const __m256 cross = _mm256_blendv_ps(right, left, maskLt);
		const __m256 r = _mm256_blendv_ps(cross, same, maskEq);

		agp_store8_R(data, i, r);
		i += 8u;
	}

	const float two_m = m + m;

	while (i < sz)
	{
		data[i]->ChangeCharacteristicConstM(m, inv_m, two_m);
		++i;
	}
}

static __declspec(noalias) __forceinline
void RecomputeR_AffineM_Mixed_ND(
	IntervalND** __restrict data,
	size_t sz,
	float GF,
	float alpha) noexcept
{
	const __m256 vGF = _mm256_set1_ps(GF);
	const __m256 vAlpha = _mm256_set1_ps(alpha);
	const __m256 vTwo = _mm256_set1_ps(2.0f);
	const __m256 vneg4 = _mm256_set1_ps(-4.0f);

	size_t i = 0u;
	const size_t limit = sz & ~7ull;

	while (i < limit)
	{
		IntervalND* I0 = data[i + 0u];
		IntervalND* I1 = data[i + 1u];
		IntervalND* I2 = data[i + 2u];
		IntervalND* I3 = data[i + 3u];
		IntervalND* I4 = data[i + 4u];
		IntervalND* I5 = data[i + 5u];
		IntervalND* I6 = data[i + 6u];
		IntervalND* I7 = data[i + 7u];

		const __m256 vN = _mm256_set_ps(
			I7->N_factor, I6->N_factor, I5->N_factor, I4->N_factor,
			I3->N_factor, I2->N_factor, I1->N_factor, I0->N_factor);

		const __m256 vM = _mm256_set_ps(
			I7->M, I6->M, I5->M, I4->M,
			I3->M, I2->M, I1->M, I0->M);

		const __m256 vQ = _mm256_set_ps(
			I7->quadratic_term, I6->quadratic_term, I5->quadratic_term, I4->quadratic_term,
			I3->quadratic_term, I2->quadratic_term, I1->quadratic_term, I0->quadratic_term);

		const __m256 vOrd = _mm256_set_ps(
			I7->ordinate_factor, I6->ordinate_factor, I5->ordinate_factor, I4->ordinate_factor,
			I3->ordinate_factor, I2->ordinate_factor, I1->ordinate_factor, I0->ordinate_factor);

		const __m256 vy1 = _mm256_set_ps(
			I7->y1, I6->y1, I5->y1, I4->y1,
			I3->y1, I2->y1, I1->y1, I0->y1);

		const __m256 vy2 = _mm256_set_ps(
			I7->y2, I6->y2, I5->y2, I4->y2,
			I3->y2, I2->y2, I1->y2, I0->y2);

		const __m256 vm =
			_mm256_fmadd_ps(vGF, vN, _mm256_mul_ps(vAlpha, vM));

		const __m256 same =
			_mm256_fmadd_ps(
				_mm256_div_ps(vQ, vm),
				_mm256_set1_ps(1.0f),
				_mm256_fmadd_ps(vm, vN, vOrd));

		const __m256 two_m = _mm256_mul_ps(vTwo, vm);

		const __m256 left =
			_mm256_fmadd_ps(two_m, vN, _mm256_mul_ps(vneg4, vy2));

		const __m256 right =
			_mm256_fmadd_ps(two_m, vN, _mm256_mul_ps(vneg4, vy1));

		const __m256i idx1 = _mm256_set_epi32(
			I7->idx1, I6->idx1, I5->idx1, I4->idx1,
			I3->idx1, I2->idx1, I1->idx1, I0->idx1);

		const __m256i idx2 = _mm256_set_epi32(
			I7->idx2, I6->idx2, I5->idx2, I4->idx2,
			I3->idx2, I2->idx2, I1->idx2, I0->idx2);

		const __m256 maskEq =
			_mm256_castsi256_ps(_mm256_cmpeq_epi32(idx1, idx2));

		const __m256 maskLt =
			_mm256_castsi256_ps(_mm256_cmpgt_epi32(idx2, idx1));

		const __m256 cross = _mm256_blendv_ps(right, left, maskLt);
		const __m256 r = _mm256_blendv_ps(cross, same, maskEq);

		agp_store8_R(data, i, r);
		i += 8u;
	}

	while (i < sz)
	{
		data[i]->ChangeCharacteristicAffine(GF, alpha);
		++i;
	}
}

__declspec(noalias) __forceinline
void IntervalND::compute_span_level(const MortonND& map) noexcept
{
	unsigned s = 0u;
	int d = 0;
	while (d < map.dim)
	{
		const unsigned long long varying = (i1 ^ i2) & map.pextMask[d];
		s += static_cast<unsigned>(_mm_popcnt_u64(varying));
		++d;
	}

	s += static_cast<unsigned>((map.levels - map.chunk_bits[0]) * map.dim);
	if (s > 11u) s = 11u;

	span_level = static_cast<unsigned short>(s);
}

//============================================================
//                     TRANSITION GEOMETRY SAMPLE
//============================================================

__declspec(align(32)) struct TransitionGeomSample final
{
	float q[AGP_MAX_FULL_DIM];
	float x, y, a, clearance;
};

//============================================================
//                     MANIP COST CLASS
//============================================================

__declspec(align(32)) struct ManipCost final
{
	//============================================================
	//  MEMBERS
	//============================================================
	int                 n;
	bool                variableLen;
	float               fixedLength;
	float               stretchFactor;
	float               targetX;
	float               targetY;
	float               targetAngle;
	float               maxTheta;
	unsigned            obstacleCount;
	SquareObstacle      obstacles[MAX_OBSTACLES];
	float               obstacleClearance;

	__declspec(align(32)) float referenceState[AGP_MAX_FULL_DIM];

	mutable TransitionGeomSample cached_start_sample;
	mutable bool cached_start_valid;

	float               transitionCaptureWeight;
	float               transitionEnergyWeight;
	float               transitionLengthEnergyWeight;
	float               transitionSweepPenaltyWeight;
	float               transitionSweepPenaltyWeightLeaf;

	//============================================================
	//  STATIC MATH HELPERS
	//============================================================
	static __forceinline float hsum256_ps(__m256 v) noexcept
	{
		__m128 lo = _mm256_castps256_ps128(v);
		__m128 hi = _mm256_extractf128_ps(v, 1);
		__m128 sum = _mm_add_ps(lo, hi);
		sum = _mm_hadd_ps(sum, sum);
		sum = _mm_hadd_ps(sum, sum);
		return _mm_cvtss_f32(sum);
	}

	static __forceinline __m256 wrap_pi_ps(__m256 a) noexcept
	{
		__m256 k = _mm256_floor_ps(_mm256_mul_ps(_mm256_add_ps(a, VEC_PI), INV_TWOPI));
		a = _mm256_fnmadd_ps(k, VEC_TWOPI, a);

		const __m256 gt = _mm256_cmp_ps(a, VEC_PI, _CMP_GT_OQ);
		const __m256 lt = _mm256_cmp_ps(a, vNEG_PI, _CMP_LT_OQ);

		a = _mm256_blendv_ps(a, _mm256_sub_ps(a, VEC_TWOPI), gt);
		a = _mm256_blendv_ps(a, _mm256_add_ps(a, VEC_TWOPI), lt);
		return a;
	}

	static __forceinline __m256 prefix_sum8_ps(__m256 v) noexcept
	{
		v = _mm256_add_ps(v,
			_mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v), 4)));
		v = _mm256_add_ps(v,
			_mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v), 8)));

		const __m128 lo = _mm256_castps256_ps128(v);
		__m128 hi = _mm256_extractf128_ps(v, 1);
		const __m128 carry = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(3, 3, 3, 3));
		hi = _mm_add_ps(hi, carry);

		return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
	}

	//============================================================
	//  POSE COMPUTATION TEMPLATES
	//============================================================
	template<int I, int N>
	static __forceinline void agp_pose_prefix(const float* __restrict th, float& acc, float* __restrict phi) noexcept
	{
		if constexpr (I < N)
		{
			acc += th[I];
			phi[I] = acc;
			agp_pose_prefix<I + 1, N>(th, acc, phi);
		}
	}

	template<int I, int N, bool store>
	__forceinline void agp_pose_accum_fixed(
		const float* __restrict q,
		const float* __restrict s_arr,
		const float* __restrict c_arr,
		float& x, float& y,
		float* __restrict px,
		float* __restrict py) const noexcept
	{
		if constexpr (I < N)
		{
			const float Li = fixedLength;
			x = fmaf(Li, c_arr[I], x);
			y = fmaf(Li, s_arr[I], y);
			if constexpr (store)
			{
				px[I + 1] = x;
				py[I + 1] = y;
			}
			agp_pose_accum_fixed<I + 1, N, store>(q, s_arr, c_arr, x, y, px, py);
		}
	}

	template<int I, int N, bool store>
	__forceinline void agp_pose_accum_var(
		const float* __restrict q,
		const float* __restrict s_arr,
		const float* __restrict c_arr,
		float& x, float& y,
		float* __restrict px,
		float* __restrict py) const noexcept
	{
		if constexpr (I < N)
		{
			const float Li = q[n + I];
			x = fmaf(Li, c_arr[I], x);
			y = fmaf(Li, s_arr[I], y);
			if constexpr (store)
			{
				px[I + 1] = x;
				py[I + 1] = y;
			}
			agp_pose_accum_var<I + 1, N, store>(q, s_arr, c_arr, x, y, px, py);
		}
	}

	template<int N, bool VARLEN, bool store>
	__forceinline void agp_compute_pose_fixedn(
		const float* __restrict q,
		float& out_x,
		float& out_y,
		float& out_a,
		float* __restrict px,
		float* __restrict py, float* __restrict sin_out, float* __restrict cos_out) const noexcept
	{
		__declspec(align(32)) float phi[AGP_MAX_FULL_DIM];
		float phi_acc = 0.0f, x = 0.0f, y = 0.0f;
		if constexpr (store)
		{
			px[0] = 0.0f;
			py[0] = 0.0f;
		}
		agp_pose_prefix<0, N>(q, phi_acc, phi);
		if (static_cast<bool>(sin_out))
		{
			FABE13_SINCOS(phi, sin_out, cos_out, N);
			if constexpr (VARLEN) agp_pose_accum_var<0, N, store>(q, sin_out, cos_out, x, y, px, py);
			else                  agp_pose_accum_fixed<0, N, store>(q, sin_out, cos_out, x, y, px, py);
		}
		else
		{
			__declspec(align(32)) float s_arr[AGP_MAX_FULL_DIM], c_arr[AGP_MAX_FULL_DIM];
			FABE13_SINCOS(phi, s_arr, c_arr, N);
			if constexpr (VARLEN) agp_pose_accum_var<0, N, store>(q, s_arr, c_arr, x, y, px, py);
			else                  agp_pose_accum_fixed<0, N, store>(q, s_arr, c_arr, x, y, px, py);
		}
		out_x = x;
		out_y = y;
		out_a = phi_acc;
	}

	__declspec(noalias) __forceinline void compute_pose(const float* __restrict q, float& out_x, float& out_y, float& out_a,
		float* __restrict px, float* __restrict py, float* __restrict sin_out, float* __restrict cos_out) const noexcept
	{
		const bool store = static_cast<bool>(px) && static_cast<bool>(py);
		if (variableLen)
		{
			if (store)
			{
				switch (n)
				{
				case  1: agp_compute_pose_fixedn< 1, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  2: agp_compute_pose_fixedn< 2, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  3: agp_compute_pose_fixedn< 3, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  4: agp_compute_pose_fixedn< 4, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  5: agp_compute_pose_fixedn< 5, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  6: agp_compute_pose_fixedn< 6, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  7: agp_compute_pose_fixedn< 7, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  8: agp_compute_pose_fixedn< 8, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  9: agp_compute_pose_fixedn< 9, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 10: agp_compute_pose_fixedn<10, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 11: agp_compute_pose_fixedn<11, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 12: agp_compute_pose_fixedn<12, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 13: agp_compute_pose_fixedn<13, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 14: agp_compute_pose_fixedn<14, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 15: agp_compute_pose_fixedn<15, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				default: agp_compute_pose_fixedn<16, true, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				}
			}
			else
			{
				switch (n)
				{
				case  1: agp_compute_pose_fixedn< 1, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  2: agp_compute_pose_fixedn< 2, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  3: agp_compute_pose_fixedn< 3, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  4: agp_compute_pose_fixedn< 4, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  5: agp_compute_pose_fixedn< 5, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  6: agp_compute_pose_fixedn< 6, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  7: agp_compute_pose_fixedn< 7, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  8: agp_compute_pose_fixedn< 8, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  9: agp_compute_pose_fixedn< 9, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 10: agp_compute_pose_fixedn<10, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 11: agp_compute_pose_fixedn<11, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 12: agp_compute_pose_fixedn<12, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 13: agp_compute_pose_fixedn<13, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 14: agp_compute_pose_fixedn<14, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 15: agp_compute_pose_fixedn<15, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				default: agp_compute_pose_fixedn<16, true, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				}
			}
		}
		else
		{
			if (store)
			{
				switch (n)
				{
				case  1: agp_compute_pose_fixedn< 1, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  2: agp_compute_pose_fixedn< 2, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  3: agp_compute_pose_fixedn< 3, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  4: agp_compute_pose_fixedn< 4, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  5: agp_compute_pose_fixedn< 5, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  6: agp_compute_pose_fixedn< 6, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  7: agp_compute_pose_fixedn< 7, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  8: agp_compute_pose_fixedn< 8, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  9: agp_compute_pose_fixedn< 9, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 10: agp_compute_pose_fixedn<10, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 11: agp_compute_pose_fixedn<11, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 12: agp_compute_pose_fixedn<12, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 13: agp_compute_pose_fixedn<13, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 14: agp_compute_pose_fixedn<14, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 15: agp_compute_pose_fixedn<15, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				default: agp_compute_pose_fixedn<16, false, true>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				}
			}
			else
			{
				switch (n)
				{
				case  1: agp_compute_pose_fixedn< 1, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  2: agp_compute_pose_fixedn< 2, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  3: agp_compute_pose_fixedn< 3, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  4: agp_compute_pose_fixedn< 4, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  5: agp_compute_pose_fixedn< 5, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  6: agp_compute_pose_fixedn< 6, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  7: agp_compute_pose_fixedn< 7, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  8: agp_compute_pose_fixedn< 8, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case  9: agp_compute_pose_fixedn< 9, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 10: agp_compute_pose_fixedn<10, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 11: agp_compute_pose_fixedn<11, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 12: agp_compute_pose_fixedn<12, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 13: agp_compute_pose_fixedn<13, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 14: agp_compute_pose_fixedn<14, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				case 15: agp_compute_pose_fixedn<15, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				default: agp_compute_pose_fixedn<16, false, false>(q, out_x, out_y, out_a, px, py, sin_out, cos_out); return;
				}
			}
		}

		__declspec(align(32)) float phi[AGP_MAX_FULL_DIM];

		const float* __restrict th = q;
		float phi_acc = 0.0f;

		if (store)
		{
			px[0] = 0.0f;
			py[0] = 0.0f;
		}

		int i = 0;
		__pragma(loop(ivdep))
			while (i + 4 <= n)
			{
				const float p0 = phi_acc + th[i + 0];
				const float p1 = p0 + th[i + 1];
				const float p2 = p1 + th[i + 2];
				const float p3 = p2 + th[i + 3];
				phi[i + 0] = p0;
				phi[i + 1] = p1;
				phi[i + 2] = p2;
				phi[i + 3] = p3;
				phi_acc = p3;
				i += 4;
			}
		__pragma(loop(ivdep))
			while (i < n)
			{
				phi_acc += th[i];
				phi[i] = phi_acc;
				++i;
			}

		if (static_cast<bool>(sin_out))
		{
			FABE13_SINCOS(phi, sin_out, cos_out, n);

			if (!store)
			{
				int j = 0;
				__m256 vx = _mm256_setzero_ps();
				__m256 vy = _mm256_setzero_ps();

				if (variableLen)
				{
					__pragma(loop(ivdep))
						while (j + 8 <= n)
						{
							const __m256 vL = _mm256_load_ps(q + n + j);
							const __m256 vc = _mm256_load_ps(cos_out + j);
							const __m256 vs = _mm256_load_ps(sin_out + j);
							vx = _mm256_fmadd_ps(vL, vc, vx);
							vy = _mm256_fmadd_ps(vL, vs, vy);
							j += 8;
						}

					float x = hsum256_ps(vx);
					float y = hsum256_ps(vy);

					__pragma(loop(ivdep))
						while (j < n)
						{
							const float Li = q[n + j];
							x = fmaf(Li, cos_out[j], x);
							y = fmaf(Li, sin_out[j], y);
							++j;
						}

					out_x = x;
					out_y = y;
					out_a = phi_acc;
					return;
				}
				else
				{
					__declspec(align(32)) const __m256 vL = _mm256_set1_ps(fixedLength);
					__pragma(loop(ivdep))
						while (j + 8 <= n)
						{
							const __m256 vc = _mm256_load_ps(cos_out + j);
							const __m256 vs = _mm256_load_ps(sin_out + j);
							vx = _mm256_fmadd_ps(vL, vc, vx);
							vy = _mm256_fmadd_ps(vL, vs, vy);
							j += 8;
						}

					float x = hsum256_ps(vx);
					float y = hsum256_ps(vy);

					__pragma(loop(ivdep))
						while (j < n)
						{
							x = fmaf(fixedLength, cos_out[j], x);
							y = fmaf(fixedLength, sin_out[j], y);
							++j;
						}

					out_x = x;
					out_y = y;
					out_a = phi_acc;
					return;
				}
			}

			float x = 0.0f, y = 0.0f;
			i = 0;

			if (variableLen)
			{
				__pragma(loop(ivdep))
					while (i + 4 <= n)
					{
						{
							const float L0 = q[n + i + 0];
							x = fmaf(L0, cos_out[i + 0], x);
							y = fmaf(L0, sin_out[i + 0], y);
							px[i + 1] = x; py[i + 1] = y;
						}
						{
							const float L1 = q[n + i + 1];
							x = fmaf(L1, cos_out[i + 1], x);
							y = fmaf(L1, sin_out[i + 1], y);
							px[i + 2] = x; py[i + 2] = y;
						}
						{
							const float L2 = q[n + i + 2];
							x = fmaf(L2, cos_out[i + 2], x);
							y = fmaf(L2, sin_out[i + 2], y);
							px[i + 3] = x; py[i + 3] = y;
						}
						{
							const float L3 = q[n + i + 3];
							x = fmaf(L3, cos_out[i + 3], x);
							y = fmaf(L3, sin_out[i + 3], y);
							px[i + 4] = x; py[i + 4] = y;
						}
						i += 4;
					}
				__pragma(loop(ivdep))
					while (i < n)
					{
						const float Li = q[n + i];
						x = fmaf(Li, cos_out[i], x);
						y = fmaf(Li, sin_out[i], y);
						px[i + 1] = x;
						py[i + 1] = y;
						++i;
					}
			}
			else
			{
				const float L = fixedLength;
				__pragma(loop(ivdep))
					while (i + 4 <= n)
					{
						x = fmaf(L, cos_out[i + 0], x); y = fmaf(L, sin_out[i + 0], y); px[i + 1] = x; py[i + 1] = y;
						x = fmaf(L, cos_out[i + 1], x); y = fmaf(L, sin_out[i + 1], y); px[i + 2] = x; py[i + 2] = y;
						x = fmaf(L, cos_out[i + 2], x); y = fmaf(L, sin_out[i + 2], y); px[i + 3] = x; py[i + 3] = y;
						x = fmaf(L, cos_out[i + 3], x); y = fmaf(L, sin_out[i + 3], y); px[i + 4] = x; py[i + 4] = y;
						i += 4;
					}
				__pragma(loop(ivdep))
					while (i < n)
					{
						x = fmaf(L, cos_out[i], x);
						y = fmaf(L, sin_out[i], y);
						px[i + 1] = x;
						py[i + 1] = y;
						++i;
					}
			}

			out_x = x;
			out_y = y;
			out_a = phi_acc;
		}
		else
		{
			__declspec(align(32)) float s_arr[AGP_MAX_FULL_DIM], c_arr[AGP_MAX_FULL_DIM];
			FABE13_SINCOS(phi, s_arr, c_arr, n);

			if (!store)
			{
				int j = 0;
				__m256 vx = _mm256_setzero_ps();
				__m256 vy = _mm256_setzero_ps();

				if (variableLen)
				{
					__pragma(loop(ivdep))
						while (j + 8 <= n)
						{
							const __m256 vL = _mm256_load_ps(q + n + j);
							const __m256 vc = _mm256_load_ps(c_arr + j);
							const __m256 vs = _mm256_load_ps(s_arr + j);
							vx = _mm256_fmadd_ps(vL, vc, vx);
							vy = _mm256_fmadd_ps(vL, vs, vy);
							j += 8;
						}

					float x = hsum256_ps(vx);
					float y = hsum256_ps(vy);

					__pragma(loop(ivdep))
						while (j < n)
						{
							const float Li = q[n + j];
							x = fmaf(Li, c_arr[j], x);
							y = fmaf(Li, s_arr[j], y);
							++j;
						}

					out_x = x;
					out_y = y;
					out_a = phi_acc;
					return;
				}
				else
				{
					__declspec(align(32)) const __m256 vL = _mm256_set1_ps(fixedLength);
					__pragma(loop(ivdep))
						while (j + 8 <= n)
						{
							const __m256 vc = _mm256_load_ps(c_arr + j);
							const __m256 vs = _mm256_load_ps(s_arr + j);
							vx = _mm256_fmadd_ps(vL, vc, vx);
							vy = _mm256_fmadd_ps(vL, vs, vy);
							j += 8;
						}

					float x = hsum256_ps(vx);
					float y = hsum256_ps(vy);

					__pragma(loop(ivdep))
						while (j < n)
						{
							x = fmaf(fixedLength, c_arr[j], x);
							y = fmaf(fixedLength, s_arr[j], y);
							++j;
						}

					out_x = x;
					out_y = y;
					out_a = phi_acc;
					return;
				}
			}

			float x = 0.0f, y = 0.0f;
			i = 0;

			if (variableLen)
			{
				__pragma(loop(ivdep))
					while (i + 4 <= n)
					{
						{
							const float L0 = q[n + i + 0];
							x = fmaf(L0, c_arr[i + 0], x);
							y = fmaf(L0, s_arr[i + 0], y);
							px[i + 1] = x; py[i + 1] = y;
						}
						{
							const float L1 = q[n + i + 1];
							x = fmaf(L1, c_arr[i + 1], x);
							y = fmaf(L1, s_arr[i + 1], y);
							px[i + 2] = x; py[i + 2] = y;
						}
						{
							const float L2 = q[n + i + 2];
							x = fmaf(L2, c_arr[i + 2], x);
							y = fmaf(L2, s_arr[i + 2], y);
							px[i + 3] = x; py[i + 3] = y;
						}
						{
							const float L3 = q[n + i + 3];
							x = fmaf(L3, c_arr[i + 3], x);
							y = fmaf(L3, s_arr[i + 3], y);
							px[i + 4] = x; py[i + 4] = y;
						}
						i += 4;
					}
				__pragma(loop(ivdep))
					while (i < n)
					{
						const float Li = q[n + i];
						x = fmaf(Li, c_arr[i], x);
						y = fmaf(Li, s_arr[i], y);
						px[i + 1] = x;
						py[i + 1] = y;
						++i;
					}
			}
			else
			{
				const float L = fixedLength;
				__pragma(loop(ivdep))
					while (i + 4 <= n)
					{
						x = fmaf(L, c_arr[i + 0], x); y = fmaf(L, s_arr[i + 0], y); px[i + 1] = x; py[i + 1] = y;
						x = fmaf(L, c_arr[i + 1], x); y = fmaf(L, s_arr[i + 1], y); px[i + 2] = x; py[i + 2] = y;
						x = fmaf(L, c_arr[i + 2], x); y = fmaf(L, s_arr[i + 2], y); px[i + 3] = x; py[i + 3] = y;
						x = fmaf(L, c_arr[i + 3], x); y = fmaf(L, s_arr[i + 3], y); px[i + 4] = x; py[i + 4] = y;
						i += 4;
					}
				__pragma(loop(ivdep))
					while (i < n)
					{
						x = fmaf(L, c_arr[i], x);
						y = fmaf(L, s_arr[i], y);
						px[i + 1] = x;
						py[i + 1] = y;
						++i;
					}
			}

			out_x = x;
			out_y = y;
			out_a = phi_acc;
		}
	}

	__declspec(noalias) __forceinline void copy_state_full_dim(
		const float* __restrict src,
		float* __restrict dst) const noexcept
	{
		const int total = n << 1;

		if (total == 2) {
			const __m128 v = _mm_load_ps(src);
			_mm_store_ps(dst, v);
			return;
		}
		if (total == 4) {
			const __m128 v = _mm_load_ps(src);
			_mm_store_ps(dst, v);
			return;
		}
		if (total == 6) {
			const __m128 v1 = _mm_load_ps(src);
			const __m128 v2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(src + 4));
			_mm_store_ps(dst, v1);
			_mm_storel_pi((__m64*)(dst + 4), v2);
			return;
		}
		if (total == 8) {
			const __m256 v = _mm256_load_ps(src);
			_mm256_store_ps(dst, v);
			return;
		}
		if (total == 12) {
			const __m256 v1 = _mm256_load_ps(src);
			const __m128 v2 = _mm_load_ps(src + 8);
			_mm256_store_ps(dst, v1);
			_mm_store_ps(dst + 8, v2);
			return;
		}
		if (total == 16) {
			const __m256 v1 = _mm256_load_ps(src);
			const __m256 v2 = _mm256_load_ps(src + 8);
			_mm256_store_ps(dst, v1);
			_mm256_store_ps(dst + 8, v2);
			return;
		}

		__assume((uintptr_t)src % 32 == 0);
		__assume((uintptr_t)dst % 32 == 0);
		memcpy(dst, src, static_cast<size_t>(total) * sizeof(float));
	}

	static __forceinline __m256 wrap_pi_avx(__m256 a) noexcept {
		const __m256 mask_gt = _mm256_cmp_ps(a, VEC_PI, _CMP_GT_OQ);
		const __m256 mask_lt = _mm256_cmp_ps(a, vNEG_PI, _CMP_LT_OQ);
		a = _mm256_sub_ps(a, _mm256_and_ps(mask_gt, VEC_TWOPI));
		a = _mm256_add_ps(a, _mm256_and_ps(mask_lt, VEC_TWOPI));
		return a;
	}

	static __forceinline __m128 wrap_pi_sse(__m128 a) noexcept {
		const __m128 mask_gt = _mm_cmp_ps(a, vpi4, _CMP_GT_OQ);
		const __m128 mask_lt = _mm_cmp_ps(a, vnpi4, _CMP_LT_OQ);
		a = _mm_sub_ps(a, _mm_and_ps(mask_gt, vtwo_pi4));
		a = _mm_add_ps(a, _mm_and_ps(mask_lt, vtwo_pi4));
		return a;
	}

	static __forceinline __m128 wrap_pi_ss(__m128 a) noexcept {
		const __m128 mask_gt = _mm_cmp_ss(a, vpi4, _CMP_GT_OQ);
		const __m128 mask_lt = _mm_cmp_ss(a, vnpi4, _CMP_LT_OQ);
		a = _mm_sub_ss(a, _mm_and_ps(mask_gt, vtwo_pi4));
		a = _mm_add_ss(a, _mm_and_ps(mask_lt, vtwo_pi4));
		return a;
	}

	__declspec(noalias) __forceinline void lerp_state_full_dim(
		const float* __restrict qa,
		const float* __restrict qb,
		float t,
		float* __restrict out_q) const noexcept
	{
		const int total = n << 1;

		const __m256 vt8 = _mm256_set1_ps(t);
		const __m128 vt4 = _mm_set1_ps(t);

		int i = 0;

		__pragma(loop(ivdep))
			while (i + 8 <= n)
			{
				const __m256 va = _mm256_load_ps(qa + i);
				const __m256 vb = _mm256_load_ps(qb + i);

				__m256 vd = _mm256_sub_ps(vb, va);
				vd = wrap_pi_avx(vd);

				__m256 vo = _mm256_fmadd_ps(vt8, vd, va);
				vo = wrap_pi_avx(vo);

				_mm256_store_ps(out_q + i, vo);
				i += 8;
			}

		if (i + 4 <= n)
		{
			const __m128 va = _mm_load_ps(qa + i);
			const __m128 vb = _mm_load_ps(qb + i);

			__m128 vd = _mm_sub_ps(vb, va);
			vd = wrap_pi_sse(vd);

			__m128 vo = _mm_fmadd_ps(vt4, vd, va);
			vo = wrap_pi_sse(vo);

			_mm_store_ps(out_q + i, vo);
			i += 4;
		}

		__pragma(loop(ivdep))
			while (i < n)
			{
				const __m128 va = _mm_set_ss(qa[i]);
				const __m128 vb = _mm_set_ss(qb[i]);

				__m128 vd = _mm_sub_ss(vb, va);
				vd = wrap_pi_ss(vd);

				__m128 vo = _mm_fmadd_ss(vt4, vd, va);
				vo = wrap_pi_ss(vo);

				_mm_store_ss(out_q + i, vo);
				++i;
			}

		i = n;

		if ((n & 7) == 0)
		{
			__pragma(loop(ivdep))
				while (i + 8 <= total)
				{
					const __m256 va = _mm256_load_ps(qa + i);
					const __m256 vb = _mm256_load_ps(qb + i);

					_mm256_store_ps(
						out_q + i,
						_mm256_fmadd_ps(vt8, _mm256_sub_ps(vb, va), va)
					);

					i += 8;
				}

			if (i + 4 <= total)
			{
				const __m128 va = _mm_load_ps(qa + i);
				const __m128 vb = _mm_load_ps(qb + i);

				_mm_store_ps(
					out_q + i,
					_mm_fmadd_ps(vt4, _mm_sub_ps(vb, va), va)
				);

				i += 4;
			}
		}
		else
		{
			__pragma(loop(ivdep))
				while (i + 8 <= total)
				{
					const __m256 va = _mm256_loadu_ps(qa + i);
					const __m256 vb = _mm256_loadu_ps(qb + i);

					_mm256_storeu_ps(
						out_q + i,
						_mm256_fmadd_ps(vt8, _mm256_sub_ps(vb, va), va)
					);

					i += 8;
				}

			if (i + 4 <= total)
			{
				const __m128 va = _mm_loadu_ps(qa + i);
				const __m128 vb = _mm_loadu_ps(qb + i);

				_mm_storeu_ps(
					out_q + i,
					_mm_fmadd_ps(vt4, _mm_sub_ps(vb, va), va)
				);

				i += 4;
			}
		}

		__pragma(loop(ivdep))
			while (i < total)
			{
				out_q[i] = fmaf(t, qb[i] - qa[i], qa[i]);
				++i;
			}

		const __m256 vz8 = _mm256_setzero_ps();
		const __m128 vz4 = _mm_setzero_ps();

		i = total;

		__pragma(loop(ivdep))
			while ((i & 7) != 0 && i < AGP_MAX_FULL_DIM)
			{
				out_q[i] = 0.0f;
				++i;
			}

		__pragma(loop(ivdep))
			while (i + 8 <= AGP_MAX_FULL_DIM)
			{
				_mm256_store_ps(out_q + i, vz8);
				i += 8;
			}

		if (i + 4 <= AGP_MAX_FULL_DIM)
		{
			_mm_store_ps(out_q + i, vz4);
			i += 4;
		}

		__pragma(loop(ivdep))
			while (i < AGP_MAX_FULL_DIM)
			{
				out_q[i] = 0.0f;
				++i;
			}
	}

	static __declspec(noalias) __forceinline float wrap_pi(float a) noexcept
	{
		if (a > PI) a -= TWO_PI;
		else if (a < -PI) a += TWO_PI;
		return a;
	}

	__forceinline void SetTransitionReference(const float* src) noexcept
	{
		cached_start_valid = false;
		const int total = n + (variableLen ? n : 0);
		int i = 0;
		__pragma(loop(ivdep))
			while (i < total)
			{
				referenceState[i] = src[i];
				++i;
			}
	}

	__forceinline float TransitionAngleEnergy(const float* __restrict q) const noexcept
	{
		float acc = 0.0f;
		int i = 0;
		__pragma(loop(ivdep))
			while (i < n)
			{
				float d = wrap_pi(q[i] - referenceState[i]);
				acc = fmaf(d, d, acc);
				++i;
			}
		return acc;
	}

	__forceinline float TransitionLengthEnergy(const float* __restrict q) const noexcept
	{
		if (!variableLen) return 0.0f;
		float acc = 0.0f;
		int i = 0;
		__pragma(loop(ivdep))
			while (i < n)
			{
				float d = q[n + i] - referenceState[n + i];
				acc = fmaf(d, d, acc);
				++i;
			}
		return acc;
	}

	__forceinline float TransitionEnergy(const float* __restrict q) const noexcept
	{
		return fmaf(
			transitionEnergyWeight,
			TransitionAngleEnergy(q),
			fmaf(
				transitionLengthEnergyWeight,
				TransitionLengthEnergy(q),
				0.0f
			)
		);
	}

	__declspec(noalias) __forceinline ManipCost(
		int _n, bool _variableLen, float _targetX, float _targetY, float _targetAngle, float _maxTheta,
		float _fixedLength, float _stretchFactor, float* obstacleData, unsigned _obstacleCount, int mode) noexcept
		: n(_n)
		, variableLen(_variableLen)
		, fixedLength(_fixedLength)
		, stretchFactor(_stretchFactor)
		, targetX(_targetX)
		, targetY(_targetY)
		, targetAngle(_targetAngle)
		, maxTheta(_maxTheta)
		, obstacleCount(_obstacleCount)
		, obstacleClearance(OBSTACLE_CLEARANCE)
		, transitionCaptureWeight(mode & 2 ? 5000.0f : 10000.0f)
		, transitionEnergyWeight(mode & 2 ? 10.0f : 1.0f)
		, transitionLengthEnergyWeight(mode & 2 ? 3.5f : 0.35f)
		, transitionSweepPenaltyWeight(mode & 2 ? 5.0f : 3.0f)
		, transitionSweepPenaltyWeightLeaf(mode & 2 ? 2.0f : 1.15f)
		, cached_start_valid(false)
	{
		if (obstacleData)
		{
			unsigned i = 0u;
			while (i < obstacleCount)
			{
				obstacles[i].cx = obstacleData[3u * i + 0u];
				obstacles[i].cy = obstacleData[3u * i + 1u];
				obstacles[i].half = obstacleData[3u * i + 2u];
				obstacles[i].pad = 0.0f;
				obstacles[i].minX = obstacles[i].cx - obstacles[i].half;
				obstacles[i].minY = obstacles[i].cy - obstacles[i].half;
				obstacles[i].maxX = obstacles[i].cx + obstacles[i].half;
				obstacles[i].maxY = obstacles[i].cy + obstacles[i].half;
				++i;
			}
		}
	}

	__declspec(noalias) __forceinline float link_length(const float* __restrict q, int i) const noexcept
	{
		return variableLen ? q[n + i] : fixedLength;
	}

	__declspec(noalias) __forceinline float compute_positioning_objective_from_pose(const float* __restrict q, float x, float y, float a) const noexcept
	{
		const float dx = x - targetX, dy = y - targetY, angle_err = wrap_pi(a - targetAngle);
		return fmaf(dx, dx, fmaf(dy, dy, fmaf(angle_err, angle_err, 0.0f)));
	}

	__declspec(noalias) __forceinline float compute_transition_objective_from_pose(const float* __restrict q, float x, float y, float a) const noexcept
	{
		float dx = x - targetX, dy = y - targetY, angle_err = wrap_pi(a - targetAngle);
		return fmaf(transitionCaptureWeight, fmaf(dx, dx, fmaf(dy, dy, fmaf(angle_err, angle_err, 0.0f))), TransitionEnergy(q));
	}

	__declspec(noalias) __forceinline unsigned total_constraints() const noexcept
	{
		unsigned total = static_cast<unsigned>(n);
		if (variableLen) total += static_cast<unsigned>(n);
		if (static_cast<bool>(obstacleCount)) total += obstacleCount;
		return total;
	}

	static __declspec(noalias) __forceinline float nonlinear_constraint_0(float x, float y) noexcept
	{
		float dx = x - 5.0f;
		return fmaf(dx, dx, fmaf(y, y, -25.0f));
	}
	static __declspec(noalias) __forceinline float nonlinear_constraint_1(float x, float y) noexcept
	{
		float dx = x - 8.0f, dy = y + 3.0f;
		float d2 = fmaf(dx, dx, fmaf(dy, dy, 0.0f));
		return fmaf(-1.0f, d2, 7.7f);
	}

	__declspec(noalias) __forceinline bool evaluate_joint_angle_limits_only(
		const float* __restrict q, unsigned base_index, unsigned& out_index, float& out_value) const noexcept
	{
		__declspec(align(32)) const __m256 vlimit = _mm256_set1_ps(maxTheta);
		const __m256 vabsmsk = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

		int i = 0;
		__pragma(loop(ivdep))
			while (i + 8 <= n)
			{
				const __m256 vq = _mm256_load_ps(q + i);
				const __m256 vabs = _mm256_and_ps(vq, vabsmsk);
				const __m256 vviol = _mm256_sub_ps(vabs, vlimit);
				const __m256 vcmp = _mm256_cmp_ps(vviol, VEC_ZERO, _CMP_GT_OQ);
				const unsigned mask = static_cast<unsigned>(_mm256_movemask_ps(vcmp));

				if (mask)
				{
					const unsigned first = static_cast<unsigned>(_tzcnt_u32(mask));
					__m256i perm = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, first);
					__m256 shuffled = _mm256_permutevar8x32_ps(vviol, perm);
					out_value = _mm_cvtss_f32(_mm256_castps256_ps128(shuffled));
					out_index = base_index + static_cast<unsigned>(i) + first;
					return false;
				}
				i += 8;
			}

		__pragma(loop(ivdep))
			while (i < n)
			{
				const float viol = fabsf(q[i]) - maxTheta;
				if (viol > 0.0f)
				{
					out_index = base_index + static_cast<unsigned>(i);
					out_value = viol;
					return false;
				}
				++i;
			}
		return true;
	}

	__declspec(noalias) __forceinline bool evaluate_joint_length_limits_only(
		const float* __restrict q, unsigned base_index, unsigned& out_index, float& out_value) const noexcept
	{
		const float lo = fixedLength / stretchFactor;
		const float hi = fixedLength * stretchFactor;
		__pragma(loop(ivdep))
			for (int i = 0; i < n; ++i)
			{
				const float len = q[n + i];
				if (len < lo)
				{
					out_index = base_index + static_cast<unsigned>(i);
					out_value = lo - len;
					return false;
				}
				if (len > hi)
				{
					out_index = base_index + static_cast<unsigned>(i);
					out_value = len - hi;
					return false;
				}
			}
		return true;
	}

	__declspec(noalias) __forceinline bool evaluate_pose_constraints_from_arrays(
		const float* __restrict px, const float* __restrict py,
		unsigned base_index, unsigned& out_index, float& out_value,
		float* out_geom_clearance) const noexcept
	{
		float geom_clearance = FLT_MAX;
		unsigned idx = base_index;
		unsigned j = 0u;
		__pragma(loop(ivdep))
			while (j < obstacleCount)
			{
				float viol = polyline_square_violation(px, py, n, obstacles[j], obstacleClearance);
				if (viol > 0.0f)
				{
					out_index = idx; out_value = viol;
					*out_geom_clearance = -viol;
					return false;
				}
				float clearance = -viol;
				if (clearance < geom_clearance) geom_clearance = clearance;
				++idx; ++j;
			}
		*out_geom_clearance = geom_clearance;
		return true;
	}

	template<bool HasObstacles, bool VariableLen>
	__declspec(noalias) __forceinline bool evaluate_state_without_transition_continuity(
		const float* __restrict q, float& out_x, float& out_y, float& out_a,
		unsigned& out_index, float& out_value, float* out_geom_clearance, float* __restrict sin_out = nullptr, float* __restrict cos_out = nullptr) const noexcept
	{
		unsigned base_idx = 0u;
		if (!evaluate_joint_angle_limits_only(q, base_idx, out_index, out_value))
			return false;
		base_idx += static_cast<unsigned>(n);
		if constexpr (VariableLen)
		{
			if (!evaluate_joint_length_limits_only(q, base_idx, out_index, out_value))
				return false;
			base_idx += static_cast<unsigned>(n);
		}
		__declspec(align(32)) float px[AGP_MAX_LINK_POINTS], py[AGP_MAX_LINK_POINTS];
		compute_pose(q, out_x, out_y, out_a, px, py, sin_out, cos_out);
		if constexpr (HasObstacles)
		{
			if (!evaluate_pose_constraints_from_arrays(px, py, base_idx, out_index, out_value, out_geom_clearance))
				return false;
		}
		else
		{
			*out_geom_clearance = FLT_MAX;
		}
		return true;
	}

	__declspec(noalias) __forceinline float transition_segment_motion_bound(const float* __restrict qa, const float* __restrict qb) const noexcept
	{
		float tail_reach = 0.0f, bound = 0.0f;
		int i = n - 1;
		__pragma(loop(ivdep))
			while (i >= 0)
			{
				tail_reach += fmaxf(link_length(qa, i), link_length(qb, i));
				bound = fmaf(tail_reach, fabsf(wrap_pi(qb[i] - qa[i])), bound);
				--i;
			}
		if (variableLen)
		{
			i = 0;
			__pragma(loop(ivdep))
				while (i < n)
				{
					bound = fabsf(qb[n + i] - qa[n + i]) + bound;
					++i;
				}
		}
		return bound;
	}

	template<bool HasObstacles, bool VariableLen>
	__declspec(noalias) __forceinline bool validate_transition_segment(
		const TransitionGeomSample& a,
		const TransitionGeomSample& b,
		const float motion_bound,
		unsigned& out_index,
		float& out_value) const noexcept
	{
		if (a.clearance > motion_bound && b.clearance > motion_bound)
			return true;

		const float leaf_required = fmaf(motion_bound, 0.25f, 0.0f);
		TransitionGeomSample q25, mid, q75;
		bool q25_ok, mid_ok, q75_ok;
		unsigned q25_idx, mid_idx, q75_idx;
		float q25_val, mid_val, q75_val;

		oneapi::tbb::parallel_invoke(
			[this, &a, &b, &q25, &q25_ok, &q25_idx, &q25_val]()
			{
				lerp_state_full_dim(a.q, b.q, 0.25f, q25.q);
				q25_ok = this->evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(
					q25.q, q25.x, q25.y, q25.a, q25_idx, q25_val, &q25.clearance);
			},
			[this, &a, &b, &mid, &mid_ok, &mid_idx, &mid_val]()
			{
				lerp_state_full_dim(a.q, b.q, 0.5f, mid.q);
				mid_ok = this->evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(
					mid.q, mid.x, mid.y, mid.a, mid_idx, mid_val, &mid.clearance);
			},
			[this, &a, &b, &q75, &q75_ok, &q75_idx, &q75_val]()
			{
				lerp_state_full_dim(a.q, b.q, 0.75f, q75.q);
				q75_ok = this->evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(
					q75.q, q75.x, q75.y, q75.a, q75_idx, q75_val, &q75.clearance);
			}
		);

		if (!(q25_ok && mid_ok && q75_ok))
		{
			out_index = total_constraints();
			float sweep_penalty = 0.0f;
			if (!q25_ok)
			{
				const float z = fmaf(0.5f / leaf_required, q25_val, 0.0f);
				const float sweep_penalty_shape = z > 1.0f ? fmaf(z, 2.0f, -1.0f) : fmaf(z, z, 0.0f);
				sweep_penalty += sweep_penalty_shape;
			}
			if (!mid_ok)
			{
				const float z = fmaf(1.0f / leaf_required, mid_val, 0.0f);
				const float sweep_penalty_shape = z > 1.0f ? fmaf(z, 2.0f, -1.0f) : fmaf(z, z, 0.0f);
				sweep_penalty += sweep_penalty_shape;
			}
			if (!q75_ok)
			{
				const float z = fmaf(0.5f / leaf_required, q75_val, 0.0f);
				const float sweep_penalty_shape = z > 1.0f ? fmaf(z, 2.0f, -1.0f) : fmaf(z, z, 0.0f);
				sweep_penalty += sweep_penalty_shape;
			}
			out_value = fmaf(transitionSweepPenaltyWeight, sweep_penalty, compute_transition_objective_from_pose(b.q, b.x, b.y, b.a));
			return false;
		}

		const float min_clear = std::min({
						a.clearance,
						q25.clearance,
						mid.clearance,
						q75.clearance,
						b.clearance
			});
		if (min_clear > leaf_required)
			return true;
		out_index = total_constraints();
		const float z = fmaf(-1.0f / leaf_required, min_clear, 1.0f);
		const float sweep_penalty_shape = z > 1.0f ? fmaf(z, 2.0f, -1.0f) : fmaf(z, z, 0.0f);
		out_value = fmaf(transitionSweepPenaltyWeightLeaf, sweep_penalty_shape, compute_transition_objective_from_pose(b.q, b.x, b.y, b.a));
		return false;
	}

	template<bool HasObstacles, bool VariableLen>
	__declspec(noalias) __forceinline bool validate_transition_segment_recursive(
		const TransitionGeomSample& a,
		const TransitionGeomSample& b,
		const float motion_bound,
		const int depth) const noexcept
	{
		if (a.clearance > motion_bound && b.clearance > motion_bound)
			return true;

		TransitionGeomSample mid;
		lerp_state_full_dim(a.q, b.q, 0.5f, mid.q);
		unsigned dummy_idx; float dummy_val;
		if (!evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(mid.q, mid.x, mid.y, mid.a, dummy_idx, dummy_val, &mid.clearance))
			return false;

		const float child_bound = fmaf(motion_bound, 0.5f, 0.0f);
		if (a.clearance > child_bound && mid.clearance > child_bound && b.clearance > child_bound)
			return true;

		if (!depth)
		{
			TransitionGeomSample q25, q75;
			bool q25_ok, q75_ok;
			unsigned q25_idx, q75_idx;
			float q25_val, q75_val;

			oneapi::tbb::parallel_invoke(
				[this, &a, &b, &q25, &q25_ok, &q25_idx, &q25_val]()
				{
					lerp_state_full_dim(a.q, b.q, 0.25f, q25.q);
					q25_ok = this->evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(
						q25.q, q25.x, q25.y, q25.a, q25_idx, q25_val, &q25.clearance);
				},
				[this, &a, &b, &q75, &q75_ok, &q75_idx, &q75_val]()
				{
					lerp_state_full_dim(a.q, b.q, 0.75f, q75.q);
					q75_ok = this->evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(
						q75.q, q75.x, q75.y, q75.a, q75_idx, q75_val, &q75.clearance);
				}
			);
			if (!(q25_ok && q75_ok))
				return false;

			const float min_clear = std::min({
					a.clearance,
					q25.clearance,
					mid.clearance,
					q75.clearance,
					b.clearance
				});
			return min_clear > fmaf(motion_bound, 0.25f, 0.0f);
		}

		if (depth >= 5)
		{
			bool left_ok, right_ok;
			oneapi::tbb::parallel_invoke(
				[this, &a, &mid, child_bound, depth, &left_ok]()
				{
					left_ok = this->validate_transition_segment_recursive<HasObstacles, VariableLen>(
						a, mid, child_bound, depth - 1);
				},
				[this, &mid, &b, child_bound, depth, &right_ok]()
				{
					right_ok = this->validate_transition_segment_recursive<HasObstacles, VariableLen>(
						mid, b, child_bound, depth - 1);
				}
			);
			return left_ok && right_ok;
		}
		else
		{
			if (!validate_transition_segment_recursive<HasObstacles, VariableLen>(a, mid, child_bound, depth - 1))
				return false;
			if (!validate_transition_segment_recursive<HasObstacles, VariableLen>(mid, b, child_bound, depth - 1))
				return false;
			return true;
		}
	}

	template<bool HasObstacles, bool VariableLen>
	__declspec(noalias) __forceinline bool evaluate_transition_swept_motion_indexed(
		const float* __restrict q, float final_x, float final_y, float final_a, float final_geom_clearance,
		unsigned& out_index, float& out_value,
		const float* __restrict existing_objective) const noexcept
	{
		if (!cached_start_valid)
		{
			copy_state_full_dim(referenceState, cached_start_sample.q);
			if (!evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(
				cached_start_sample.q, cached_start_sample.x, cached_start_sample.y, cached_start_sample.a, out_index, out_value, &cached_start_sample.clearance))
				return false;
			cached_start_valid = true;
		}
		const TransitionGeomSample& start = cached_start_sample;
		TransitionGeomSample finish;
		copy_state_full_dim(q, finish.q);
		finish.x = final_x;
		finish.y = final_y;
		finish.a = final_a;
		finish.clearance = final_geom_clearance;
		const float motion_bound = transition_segment_motion_bound(start.q, finish.q);
		if (!validate_transition_segment<HasObstacles, VariableLen>(start, finish, motion_bound, out_index, out_value))
			return false;
		out_index = total_constraints() + 1u;
		out_value = existing_objective ? *existing_objective : compute_transition_objective_from_pose(q, final_x, final_y, final_a);
		return true;
	}

	template<bool HasObstacles, bool VariableLen>
	__declspec(noalias) __forceinline bool evaluate_transition_swept_motion(
		const float* __restrict q, float final_x, float final_y, float final_a, float final_geom_clearance,
		unsigned& out_index, float& out_value) const noexcept
	{
		if (!cached_start_valid)
		{
			copy_state_full_dim(referenceState, cached_start_sample.q);
			if (!evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(
				cached_start_sample.q, cached_start_sample.x, cached_start_sample.y, cached_start_sample.a, out_index, out_value, &cached_start_sample.clearance))
				return false;
			cached_start_valid = true;
		}
		const TransitionGeomSample& start = cached_start_sample;
		TransitionGeomSample finish;
		copy_state_full_dim(q, finish.q);
		finish.x = final_x;
		finish.y = final_y;
		finish.a = final_a;
		finish.clearance = final_geom_clearance;
		const float motion_bound = transition_segment_motion_bound(start.q, finish.q);
		int depth;
		if (motion_bound < SWEEP_MIN_BOUND) depth = 0;
		else
		{
			const float x = fmaf(motion_bound, INV_SWEEP_MIN_BOUND, 0.0f);
			uint32_t bits;
			memcpy(&bits, &x, sizeof(bits));
			const int exp = static_cast<int>((bits >> 23) & 0xFF) - 127;
			const uint32_t mantissa = bits & 0x7FFFFF;
			depth = mantissa ? exp + 1 : exp;
		}
		if (!validate_transition_segment_recursive<HasObstacles, VariableLen>(start, finish, motion_bound, depth))
			return false;
		out_index = total_constraints() + 1u;
		out_value = compute_transition_objective_from_pose(q, final_x, final_y, final_a);
		return true;
	}

	template<bool HasObstacles, bool VariableLen, bool IsTransition>
	__declspec(noalias) __forceinline bool evaluate_indexed(
		const float* __restrict q, float& out_x, float& out_y, float& out_a,
		unsigned& out_index, float& out_value,
		float* __restrict sin_eval, float* __restrict cos_eval) const noexcept
	{
		float clearance;
		if (!evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(q, out_x, out_y, out_a, out_index, out_value, &clearance,
			sin_eval, cos_eval))
			return false;
		if constexpr (HasObstacles && IsTransition)
			return evaluate_transition_swept_motion_indexed<HasObstacles, VariableLen>(q, out_x, out_y, out_a, clearance, out_index, out_value, nullptr);
		out_index = total_constraints();
		if constexpr (IsTransition)
			out_value = compute_transition_objective_from_pose(q, out_x, out_y, out_a);
		else
			out_value = compute_positioning_objective_from_pose(q, out_x, out_y, out_a);
		return true;
	}
};

//============================================================
//                     SEED GENERATION HELPERS
//============================================================

static __declspec(noalias) __forceinline unsigned agp_splitmix32(unsigned x) noexcept
{
	x += 0x9E3779B9u;
	x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
	x = (x ^ (x >> 13)) * 0xC2B2AE35u;
	return x ^ (x >> 16);
}

static __declspec(noalias) __forceinline void agp_write_seed_point(
	int row, int stride, int dim,
	const unsigned* __restrict cur_x,
	const float* __restrict low_cache,
	const float* __restrict span_cache,
	float* __restrict S) noexcept
{
	float* __restrict row_ptr = S + static_cast<size_t>(row) * static_cast<size_t>(stride);
	int dd = 0;
	__pragma(loop(ivdep))
		while (dd < dim)
		{
			const float u = static_cast<float>(static_cast<double>(cur_x[dd]) * 2.3283064365386962890625e-10);
			row_ptr[dd] = fmaf(u, span_cache[dd], low_cache[dd]);
			++dd;
		}
	__pragma(loop(ivdep))
		while (dd < stride)
		{
			row_ptr[dd] = 0.0f;
			++dd;
		}
}

static __declspec(noalias) __forceinline int generate_sobol_seeds(
	const MortonND& map, int dim_, float* __restrict S, int stride, unsigned seed, int max_points) noexcept
{
	const int dim = dim_;

	static constexpr int AGP_SOBOL_SEED_COUNT[AGP_MAX_FULL_DIM + 1] = {
			0,
			4,
			8,
			8,
			16,
			16,
			64,
			64,
			128,
			128,
			256,
			256,
			512,
			512,
			1024,
			1024,
			2048
	};

	int ns_pow2 = AGP_SOBOL_SEED_COUNT[dim];
	if (ns_pow2 > max_points) ns_pow2 = max_points;

	static const unsigned int sobol_dir[AGP_MAX_FULL_DIM][32] =
	{
			{ 1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u, 17u, 19u, 21u, 23u, 25u, 27u, 29u, 31u,
							33u, 35u, 37u, 39u, 41u, 43u, 45u, 47u, 49u, 51u, 53u, 55u, 57u, 59u, 61u, 63u },
			{ 1u, 1u, 7u, 11u, 13u, 19u, 25u, 37u, 59u, 47u, 61u, 55u, 41u, 67u, 97u, 91u,
							109u, 103u, 115u, 131u, 193u, 137u, 217u, 197u, 229u, 199u, 241u, 229u, 257u, 307u, 277u, 313u },
			{ 1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u, 17u, 19u, 21u, 23u, 25u, 27u, 29u, 31u,
							33u, 35u, 37u, 39u, 41u, 43u, 45u, 47u, 49u, 51u, 53u, 55u, 57u, 59u, 61u, 63u },
			{ 1u, 1u, 5u, 3u, 13u, 15u, 9u, 7u, 21u, 23u, 29u, 31u, 17u, 19u, 25u, 27u,
							53u, 55u, 49u, 51u, 37u, 39u, 45u, 47u, 63u, 61u, 57u, 59u, 41u, 43u, 33u, 35u },
			{ 1u, 3u, 3u, 7u, 13u, 3u, 13u, 7u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u,
							3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u },
			{ 1u, 1u, 5u, 5u, 5u, 7u, 11u, 13u, 15u, 15u, 21u, 23u, 25u, 27u, 29u, 31u,
							31u, 37u, 39u, 41u, 43u, 45u, 47u, 49u, 51u, 53u, 55u, 57u, 59u, 61u, 63u, 63u },
			{ 1u, 3u, 5u, 5u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
							7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u },
			{ 1u, 1u, 7u, 5u, 13u, 13u, 17u, 19u, 21u, 23u, 25u, 27u, 29u, 31u, 35u, 37u,
							39u, 41u, 43u, 45u, 49u, 51u, 53u, 55u, 57u, 59u, 61u, 63u, 65u, 67u, 69u, 71u },
			{ 1u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u,
							3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u },
			{ 1u, 1u, 5u, 3u, 5u, 5u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
							7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u },
			{ 1u, 3u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u,
							5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u },
			{ 1u, 1u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
							7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u },
			{ 1u, 3u, 3u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
							7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u },
			{ 1u, 1u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u,
							5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u },
			{ 1u, 3u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
							7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u },
			{ 1u, 1u, 3u, 3u, 9u, 9u, 15u, 15u, 21u, 21u, 27u, 27u, 33u, 33u, 39u, 39u,
							45u, 45u, 51u, 51u, 57u, 57u, 63u, 63u, 69u, 69u, 75u, 75u, 81u, 81u, 87u, 87u }
	};

	__declspec(align(32)) unsigned scramble_mask[AGP_MAX_FULL_DIM];
	__declspec(align(32)) float low_cache[AGP_MAX_FULL_DIM], span_cache[AGP_MAX_FULL_DIM];
	int d = 0;
	__pragma(loop(ivdep))
		while (d < dim)
		{
			seed = agp_splitmix32(seed ^ (0x9E3779B9u + static_cast<unsigned>(d) * 0x85EBCA6Bu));
			scramble_mask[d] = seed;
			++d;
		}
	d = 0;
	__pragma(loop(ivdep))
		while (d < dim)
		{
			const float lo = map.low[d];
			low_cache[d] = lo;
			span_cache[d] = map.high[d] - lo;
			++d;
		}
	__declspec(align(32)) unsigned cur_x[AGP_MAX_FULL_DIM];
	d = 0;
	__pragma(loop(ivdep))
		while (d < dim)
		{
			cur_x[d] = scramble_mask[d];
			++d;
		}
	agp_write_seed_point(0, stride, dim, cur_x, low_cache, span_cache, S);
	unsigned long long prev_gray = 0ull;
	int j = 1;
	while (j < ns_pow2)
	{
		unsigned long long i = static_cast<unsigned long long>(j);
		unsigned long long gray = i ^ (i >> 1ull);
		unsigned long long diff = gray ^ prev_gray;
		int bit = _tzcnt_u64(diff);
		d = 0;
		while (d < dim)
		{
			cur_x[d] ^= sobol_dir[d][bit] << (31 - bit);
			++d;
		}
		prev_gray = gray;
		agp_write_seed_point(j, stride, dim, cur_x, low_cache, span_cache, S);
		++j;
	}
	return ns_pow2;
}

static __declspec(noalias) __forceinline void ccd_ik(float targetX, float targetY, const float* lengths, int n, float* angles, int max_iter) noexcept
{
	int nn = n;
	__declspec(align(32)) float x[AGP_MAX_LINK_POINTS], y[AGP_MAX_LINK_POINTS], cum_angles[AGP_MAX_FULL_DIM], s_arr[AGP_MAX_FULL_DIM], c_arr[AGP_MAX_FULL_DIM];
	int iter = 0;
	while (iter < max_iter)
	{
		float acc = 0.0f;
		int i = 0;
		while (i < nn)
		{
			acc += angles[i];
			cum_angles[i] = acc;
			++i;
		}
		FABE13_SINCOS(cum_angles, s_arr, c_arr, nn);
		float curX = 0.0f, curY = 0.0f;
		x[0] = y[0] = 0.0f;
		i = 0;
		while (i < nn)
		{
			curX = fmaf(lengths[i], c_arr[i], curX);
			curY = fmaf(lengths[i], s_arr[i], curY);
			x[i + 1] = curX;
			y[i + 1] = curY;
			++i;
		}
		i = nn - 1;
		while (i >= 0)
		{
			float toEndX = x[nn] - x[i], toEndY = y[nn] - y[i];
			float toTargetX = targetX - x[i], toTargetY = targetY - y[i];
			float dot = fmaf(toEndX, toTargetX, toEndY * toTargetY);
			float det = fmaf(toEndX, toTargetY, -toEndY * toTargetX);
			float angle = atan2f(det, dot);
			angles[i] += angle;
			--i;
		}
		++iter;
	}
}

static __declspec(noalias) __forceinline int generate_heuristic_seeds(
	const ManipCost& cost,
	const MortonND& map,
	int dim,
	float* __restrict S,
	int stride,
	unsigned seed) noexcept
{
	int n = cost.n;
	bool VL = cost.variableLen;
	float tx = cost.targetX, ty = cost.targetY, phi = atan2f(ty, tx);

	float dist_to_target = sqrtf(fmaf(tx, tx, ty * ty)), max_reach = 0.0f;
	if (VL)
	{
		for (int i = 0; i < n; ++i)
			max_reach += map.high[n + i];
	}
	else
	{
		max_reach = static_cast<float>(n) * cost.fixedLength;
	}

	float ratio = dist_to_target / max_reach;
	bool prefer_extended = (ratio > 0.7f);
	bool prefer_compact = (ratio < 0.4f);
	bool use_ik = !(ratio > 0.4f && ratio < 0.7f);

	int total_seeds = 0;

	auto emit_seed = [&](const float* q_src) noexcept -> bool
		{
			if (total_seeds >= AGP_MAX_GENERATED_SEEDS)
				return false;
			float* dst = S + static_cast<size_t>(total_seeds) * static_cast<size_t>(stride);
			memcpy(dst, q_src, static_cast<size_t>(dim) * sizeof(float));
			++total_seeds;
			return true;
		};

	{
		__declspec(align(32)) float q0[AGP_MAX_FULL_DIM];
		float rho = sqrtf(fmaf(tx, tx, ty * ty));
		float len = fmaf(1.0f / static_cast<float>(fmaxf(static_cast<float>(n), 1.0f)), rho, 0.0f);
		int i = 0;
		__pragma(loop(ivdep))
			while (i < n)
			{
				q0[i] = (1.0f / static_cast<float>(n)) * phi;
				++i;
			}
		if (VL)
		{
			i = 0;
			__pragma(loop(ivdep))
				while (i < n)
				{
					q0[n + i] = len;
					++i;
				}
		}
		emit_seed(q0);
	}

	{
		__declspec(align(32)) float q1[AGP_MAX_FULL_DIM];
		int i = 0;
		__pragma(loop(ivdep))
			while (i < n)
			{
				q1[i] = fmaf(((i & 1) ? -1.0f : 1.0f), fmaf(phi, 0.5, 0.0f), 0.0f);
				++i;
			}
		if (VL)
		{
			i = 0;
			__pragma(loop(ivdep))
				while (i < n)
				{
					q1[n + i] = fmaf(0.4f, static_cast<float>(i) / static_cast<float>(fmaxf(static_cast<float>(n), 1.0f)), 0.8f);
					++i;
				}
		}
		emit_seed(q1);
	}

	{
		__declspec(align(32)) float q2[AGP_MAX_FULL_DIM];
		float inv = (n > 1) ? 1.0f / static_cast<float>(n - 1) : 0.0f;
		int i = 0;
		__pragma(loop(ivdep))
			while (i < n)
			{
				float pr = static_cast<float>(i) * inv;
				q2[i] = fmaf(phi, fmaf(-0.3f, pr, 1.0f), 0.0f);
				++i;
			}
		if (VL)
		{
			int j = 0;
			__pragma(loop(ivdep))
				while (j < n)
				{
					float si;
					FABE13_SIN(static_cast<float>(j) * 1.5f, si);
					q2[n + j] = fmaf(0.2f, si, 1.0f);
					++j;
				}
		}
		emit_seed(q2);
	}

	if (use_ik && prefer_extended && total_seeds < AGP_MAX_GENERATED_SEEDS)
	{
		__declspec(align(32)) float q3[AGP_MAX_FULL_DIM];
		__declspec(align(32)) float angles[AGP_MAX_FULL_DIM] = { 0.0f }, lengths[AGP_MAX_FULL_DIM];
		if (VL)
		{
			const float avg_len = fmaf(map.low[n], 0.5f, fmaf(map.high[n], 0.5f, 0.0f));
			__pragma(loop(ivdep))
				for (int i = 0; i < n; ++i)
					lengths[i] = avg_len;
		}
		else
		{
			__pragma(loop(ivdep))
				for (int i = 0; i < n; ++i)
					lengths[i] = cost.fixedLength;
		}
		ccd_ik(tx, ty, lengths, n, angles, 10);
		int i = 0;
		__pragma(loop(ivdep))
			while (i < n)
			{
				q3[i] = angles[i];
				++i;
			}
		if (VL)
		{
			i = 0;
			__pragma(loop(ivdep))
				while (i < n)
				{
					q3[n + i] = lengths[i];
					++i;
				}
		}
		emit_seed(q3);
	}

	if (use_ik && prefer_compact && total_seeds < AGP_MAX_GENERATED_SEEDS)
	{
		__declspec(align(32)) float q4[AGP_MAX_FULL_DIM];
		float angles_fabrik[AGP_MAX_FULL_DIM], lengths_fabrik[AGP_MAX_FULL_DIM];
		if (VL)
		{
			float avg_len = fmaf(map.low[n], 0.5f, fmaf(map.high[n], 0.5f, 0.0f));
			__pragma(loop(ivdep))
				for (int i = 0; i < n; ++i)
					lengths_fabrik[i] = avg_len;
		}
		else
		{
			__pragma(loop(ivdep))
				for (int i = 0; i < n; ++i)
					lengths_fabrik[i] = cost.fixedLength;
		}
		float targetX_fab = tx, targetY_fab = ty;
		for (int iter_fab = 0; iter_fab < 3; ++iter_fab)
		{
			float prevX = targetX_fab, prevY = targetY_fab;
			__pragma(loop(ivdep))
				for (int j = n - 1; j >= 0; --j)
				{
					float len = lengths_fabrik[j];
					float angle_to_target = atan2f(prevY, prevX);
					angles_fabrik[j] = angle_to_target;
					float s_val, c_val;
					FABE13_SINCOS(&angle_to_target, &s_val, &c_val, 1);
					prevY = fmaf(-len, s_val, prevY);
					prevX = fmaf(-len, c_val, prevX);
				}
		}
		int i = 0;
		__pragma(loop(ivdep))
			while (i < n)
			{
				q4[i] = angles_fabrik[i];
				++i;
			}
		if (VL)
		{
			i = 0;
			__pragma(loop(ivdep))
				while (i < n)
				{
					q4[n + i] = lengths_fabrik[i];
					++i;
				}
		}
		emit_seed(q4);
	}

	int remaining = AGP_MAX_GENERATED_SEEDS - total_seeds;
	if (static_cast<bool>(remaining))
	{
		total_seeds += generate_sobol_seeds(map, dim,
			S + static_cast<size_t>(total_seeds) * static_cast<size_t>(stride),
			stride, seed, remaining);
	}

	return total_seeds;
}

//============================================================
//                     PENDING MESSAGES STORAGE
//============================================================

struct PendingMultiSend
{
	MultiCrossMsg       msg;
	boost::mpi::request req;
	PendingMultiSend(boost::mpi::communicator& comm, int partner, const MultiCrossMsg& m)
		: msg(m), req(comm.isend(partner, 0, msg)) {}
};

struct PendingBestSend
{
	BestSolutionMsg     msg;
	boost::mpi::request req;
	PendingBestSend(boost::mpi::communicator& comm, int partner, const BestSolutionMsg& m)
		: msg(m), req(comm.isend(partner, 2, msg)) {}
};

static thread_local std::deque<PendingMultiSend> g_pendingMulti;
static thread_local std::deque<PendingBestSend> g_pendingBest;

//============================================================
//                     MAIN BRANCH ALGORITHM
//============================================================

template<bool HasObstacles, bool VariableLen, bool IsTransition>
static __declspec(noalias) __forceinline void agp_run_branch_mpi(
	const MortonND& map,
	const ManipCost& cost,
	int maxIter,
	float r,
	bool adaptive,
	float eps,
	unsigned seed,
	std::vector<IntervalND*, boost::alignment::aligned_allocator<IntervalND*, 32u>>& H,
	std::vector<float, boost::alignment::aligned_allocator<float, 32u>>& bestQ,
	std::vector<float, boost::alignment::aligned_allocator<float, 32u>>& bestQIndexed,
	float& bestF,
	float& bestX,
	float& bestY,
	float& bestA,
	size_t& out_iterations,
	float& out_achieved_epsilon,
	float M_prior) noexcept
{
	__declspec(align(32)) struct TrialPoint final
	{
		float q[AGP_MAX_FULL_DIM];
		float t;
		float f;
		float x;
		float y;
		float a;
		unsigned idx;
		bool feasible;
		unsigned char _pad0[7];
		unsigned long long cell;
	};

	const int n = cost.n;
	__assume(n > 0);
	const int dim = n + (cost.variableLen ? n : 0);
	__assume(dim <= AGP_MAX_FULL_DIM);
	const float dim_f = static_cast<float>(dim);
	unsigned fullConstraintIndex = cost.total_constraints();
	if constexpr (HasObstacles && IsTransition)
		fullConstraintIndex += 1u;
	int last_send_T = 0;
	const int send_interval_T = 7;
	int last_send_best = 0;
	const int send_interval_best = 2;

	unsigned exchange_counter = 0u, exchange_counter_T = 0u;
	__declspec(align(32)) float M_by_span[12];
	__pragma(loop(ivdep))
		for (int i = 0; i < 12; ++i) M_by_span[i] = M_prior;
	float Mmax = M_prior;
	__declspec(align(32)) float sum_s[AGP_MAX_FULL_DIM], sum_c[AGP_MAX_FULL_DIM], q_try[AGP_MAX_FULL_DIM];

	float bestIndexValue = FLT_MAX, bestIndexedX = 0.0f, bestIndexedY = 0.0f, bestIndexedA = 0.0f;
	unsigned bestIndexFound = 0u;
	BestSolutionMsg lastSentBestMsg;
	InitBestSolutionMsg(lastSentBestMsg);
	int no_improve = 0;

	const float a = 0.0f, b = 0.999999940395355224609375f;
	float p = 0.0f, dmax = b - a, initial_len = dmax;

	const float A_inv_sqrt = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(dim_f + 6.75f)));
	const float A_dim = A_inv_sqrt * 5.535f;
	const float A_dim__ = A_inv_sqrt * 3.425f;
	const float B_dim = A_dim * 0.7f;
	const float B_dim__ = A_dim__ * 4.325f;
	const float log_argument = A_dim - 2.03f;
	const float log_argument__ = A_dim__ - 2.0f;
	const float C_dim = fmaf(log_argument, fmaf(log_argument, fmaf(log_argument, fmaf(log_argument, fmaf(log_argument, 0.164056f, -0.098462f), 0.240884f), -0.351834f), 0.999996f), log_argument) - B_dim;
	const float C_dim__ = fmaf(log_argument__, fmaf(log_argument__, fmaf(log_argument__, fmaf(log_argument__, fmaf(log_argument__, 0.164056f, -0.098462f), 0.240884f), -0.351834f), 0.999996f), log_argument__) - B_dim__;
	const float adaptive_coeff_addition = fmaf(C_dim, fmaf(C_dim, fmaf(C_dim, fmaf(C_dim, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
	const float adaptive_coeff_addition__ = fmaf(C_dim__, fmaf(C_dim__, fmaf(C_dim__, fmaf(C_dim__, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
	const float stagnation_seed_interval = 0.00031f * dim_f;
	const float sqrt_dim = sqrtf(dim_f);
	const float sqrt_dim_minus_1 = sqrtf(dim_f - 1.0f);
	const float first_sqrt = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(dim_f))) * 1.41421356237f;
	const float inv_sqrt = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(dim_f + 7.0f)));
	const float second_sqrt = inv_sqrt * 2.23606797750f;
	const float third_sqrt = inv_sqrt * 3.0f;
	const float fourth_sqrt = inv_sqrt * 2.54950975680f;
	float adaptive_coeff = A_dim - adaptive_coeff_addition;
	float adaptive_coeff__ = A_dim__ - adaptive_coeff_addition__;

	int it = 0, stag_boost_remaining = 0;
	float stag_r_multiplier = 0.0f;
	const int n_stag_iters = static_cast<int>(fmaf(sqrt_dim, 2.045f, 3.0f));
	const int noImproveThrDim = static_cast<int>(exp2f(-0.1f * sqrt_dim) * 7.5f);
	const int num_ik_base = 1 + static_cast<int>(sqrt_dim);

	const int rank = g_world->rank();
	const int world = g_world->size();
	const size_t comm_levels = std::bit_width(static_cast<size_t>(world - 1));
	const bool multi_start = world > 1 && cost.transitionCaptureWeight == 10000.0f;

	auto finalize_interval_geometry_from_cells = [&](IntervalND* I, unsigned long long c1, unsigned long long c2) noexcept -> bool
		{
			I->i1 = c1;
			I->i2 = c2;

			I->diam = agp_block_diameter_firstchunk_exact_open_t(map, I->x1, I->x2);

			if (!static_cast<bool>(I->diam)) return false;

			const int firstChunkSpanBits = agp_firstchunk_span_bits_open(c1, c2);
			int totalSpanBits = firstChunkSpanBits + map.extra_levels * map.dim;
			unsigned short span_level = static_cast<unsigned short>(totalSpanBits);
			if (span_level > 11u) span_level = 11u;
			I->span_level = span_level;

			I->set_metric(I->diam);
			return true;
		};

	auto make_interval_from_trials = [&](const TrialPoint& Lp, const TrialPoint& Rp) noexcept -> IntervalND*
		{
			if (Lp.t > Rp.t) return nullptr;
			if (!agp_can_have_positive_diameter(map, Lp.t, Rp.t)) return nullptr;
			IntervalND* I = IntervalND::Make(Lp.t, Rp.t, Lp.f, Rp.f, Lp.idx, Rp.idx);
			if (!finalize_interval_geometry_from_cells(
				I,
				agp_t_to_firstchunk_idx_open(map, I->x1),
				agp_t_to_firstchunk_idx_open(map, I->x2)
			)) return nullptr;
			return I;
		};

	auto update_pockets_and_Mmax = [&](IntervalND* I) noexcept
		{
			const int k = I->span_level;
			if (I->M > M_by_span[k]) M_by_span[k] = I->M;
			if (M_by_span[k] > Mmax) Mmax = M_by_span[k];
		};

	auto recompute_dmax = [&]() noexcept
		{
			float new_dmax = 0.0f;
			for (auto* pI : H)
			{
				const float Ls = pI->x2 - pI->x1;
				if (Ls > new_dmax) new_dmax = Ls;
			}
			dmax = new_dmax;
		};

	auto recompute_heap_constM = [&](const float m_cur) noexcept
		{
			const size_t sz = H.size();
			RecomputeR_ConstM_Mixed_ND(H.data(), sz, m_cur, 1.0f / m_cur);
			heap_make(H);
		};

	__declspec(align(32)) int hj_order_interleaved_distal[AGP_MAX_FULL_DIM << 1];
	{
		int pos = 0;
		if constexpr (VariableLen)
		{
			int j = n - 1;
			__pragma(loop(ivdep))
				while (j >= 0)
				{
					hj_order_interleaved_distal[pos++] = j;
					hj_order_interleaved_distal[pos++] = n + j;
					--j;
				}
		}
		else
		{
			int j = n - 1;
			__pragma(loop(ivdep))
				while (j >= 0)
				{
					hj_order_interleaved_distal[pos++] = j;
					--j;
				}
		}
	}

	__declspec(align(32)) int hj_order_proximal[AGP_MAX_FULL_DIM << 1];
	{
		int pos = 0;
		if constexpr (VariableLen)
		{
			__pragma(loop(ivdep))
				for (int j = 0; j < n; ++j)
				{
					hj_order_proximal[pos++] = j;
					hj_order_proximal[pos++] = n + j;
				}
		}
		else
		{
			__pragma(loop(ivdep))
				for (int j = 0; j < n; ++j)
				{
					hj_order_proximal[pos++] = j;
				}
		}
	}

	float tau_full_fail = tau * tau;
	if (tau_full_fail < 0.125f) tau_full_fail = 0.125f;
	if (tau_full_fail > 0.35f)  tau_full_fail = 0.35f;

	auto eval_local_model =
		[&cost](const float* q_eval,
			float& x_eval, float& y_eval, float& a_eval,
			unsigned& idx_eval, float& val_eval,
			float* clearance_eval,
			float* __restrict sin_eval, float* __restrict cos_eval) noexcept -> bool
		{
			if constexpr (HasObstacles && IsTransition)
			{
				if (!cost.evaluate_state_without_transition_continuity<true, VariableLen>(
					q_eval, x_eval, y_eval, a_eval, idx_eval, val_eval, clearance_eval,
					sin_eval, cos_eval))
					return false;
				idx_eval = cost.total_constraints();
				val_eval = cost.compute_transition_objective_from_pose(q_eval, x_eval, y_eval, a_eval);
				return true;
			}

			return cost.evaluate_indexed<HasObstacles, VariableLen, IsTransition>(q_eval, x_eval, y_eval, a_eval, idx_eval, val_eval,
				sin_eval, cos_eval);
		};

	auto confirm_local_model_if_needed =
		[&cost](const float* q_eval,
			float& x_eval, float& y_eval, float& a_eval,
			const float clearance_eval,
			unsigned& idx_eval, float& val_eval) noexcept -> bool
		{
			if constexpr (HasObstacles && IsTransition)
				return cost.evaluate_transition_swept_motion_indexed<HasObstacles, VariableLen>(
					q_eval, x_eval, y_eval, a_eval, clearance_eval, idx_eval, val_eval, &val_eval);
			return true;
		};

	auto coord_scale = [&](int d) noexcept -> float
		{
			if (d < n)
			{
				const float t01 = (n > 1) ? (static_cast<float>(d) / static_cast<float>(n - 1)) : 0.0f;
				return fmaf(1.0f - t01, hj_angle_scale_proximal, t01 * hj_angle_scale_distal);
			}
			if constexpr (VariableLen)
			{
				const int j = d - n;
				const float t01 = (n > 1) ? (static_cast<float>(j) / static_cast<float>(n - 1)) : 0.0f;
				return fmaf(1.0f - t01, hj_length_scale_proximal, t01 * hj_length_scale_distal);
			}
			return 1.0f;
		};

	__declspec(align(32)) float q_lo[AGP_MAX_FULL_DIM];
	__declspec(align(32)) float q_hi[AGP_MAX_FULL_DIM];

	__pragma(loop(ivdep))
		for (int i = 0; i < n; ++i)
		{
			q_lo[i] = -cost.maxTheta;
			q_hi[i] = cost.maxTheta;
		}

	if constexpr (VariableLen)
	{
		__pragma(loop(ivdep))
			for (int i = 0; i < n; ++i)
			{
				q_lo[n + i] = cost.fixedLength / cost.stretchFactor;
				q_hi[n + i] = cost.fixedLength * cost.stretchFactor;
			}
	}

	float hj_delta[AGP_MAX_FULL_DIM];
	__pragma(loop(ivdep))
		for (int d = 0; d < dim; ++d)
		{
			const float span = map.high[d] - map.low[d];
			const float scale = coord_scale(d);
			hj_delta[d] = fmaf(span, hj_init_delta_frac, scale);
		}

	__declspec(align(32)) float hj_curr_scale[AGP_MAX_FULL_DIM];

	const float prox_angle_prox = 1.4f, prox_angle_dist = 0.6f;
	const float prox_len_prox = 1.4f, prox_len_dist = 0.6f;
	const float distal_angle_prox = 1.3f, distal_angle_dist = 0.7f;
	const float distal_len_prox = 1.3f, distal_len_dist = 0.7f;

	bool hj_using_proximal = true;

	__pragma(loop(ivdep))
		for (int d = 0; d < dim; ++d)
		{
			float scale;
			if (d < n)
			{
				float t01 = (n > 1) ? (static_cast<float>(d) / static_cast<float>(n - 1)) : 0.0f;
				scale = fmaf(1.0f - t01, prox_angle_prox, t01 * prox_angle_dist);
			}
			else if constexpr (VariableLen)
			{
				int j = d - n;
				float t01 = (n > 1) ? (static_cast<float>(j) / static_cast<float>(n - 1)) : 0.0f;
				scale = fmaf(1.0f - t01, prox_len_prox, t01 * prox_len_dist);
			}
			else
			{
				scale = 1.0f;
			}
			hj_curr_scale[d] = scale;
		}

	auto update_best_from_trial = [&](const TrialPoint& tr) noexcept
		{
			const size_t sz = static_cast<size_t>(dim);
			if (better_indexed(tr.idx, tr.f, bestIndexFound, bestIndexValue))
			{
				bestIndexFound = tr.idx;
				bestIndexValue = tr.f;
				if (bestQIndexed.size() != sz) bestQIndexed.resize(sz);
				memcpy(bestQIndexed.data(), tr.q, sz * sizeof(float));
				bestIndexedX = tr.x;
				bestIndexedY = tr.y;
				bestIndexedA = tr.a;
			}
			if (tr.feasible)
			{
				if (tr.f < bestF)
				{
					bestF = tr.f;
					if (bestQ.size() != sz) bestQ.resize(sz);
					memcpy(bestQ.data(), tr.q, sz * sizeof(float));
					bestX = tr.x;
					bestY = tr.y;
					bestA = tr.a;
					no_improve = 0;
				}
				else
				{
					++no_improve;
				}
			}
		};

	const float refine_eta_init = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(dim_f))) * 2.0f;
	float offset;
	if (dim > 6)
	{
		const float ln_arg = fmaf(dim_f, 0.08103727715f, -1.0f);
		offset = fmaf(ln_arg,
			fmaf(ln_arg,
				fmaf(ln_arg,
					fmaf(ln_arg,
						fmaf(ln_arg,
							fmaf(ln_arg,
								fmaf(ln_arg,
									-0.02142857143f,
									0.025f),
								-0.03f),
							0.0375f),
						-0.05f),
					0.075f),
				-0.15f),
			-0.965f);
	}
	else
	{
		const uint32_t bits = std::bit_cast<uint32_t>(dim_f * 0.08103727715f);
		const float x_red = static_cast<float>(bits & 0x7FFFFF) * 0.00000011921f;
		offset = fmaf(static_cast<float>(((bits >> 23) & 0xFF) - 127), -0.10397207708f,
			fmaf(x_red,
				fmaf(x_red,
					fmaf(x_red,
						fmaf(x_red,
							fmaf(x_red,
								fmaf(x_red,
									fmaf(x_red, -0.02142857143f, 0.025f),
									-0.03f),
								0.0375f),
							-0.05f),
						0.075f),
					-0.15f),
				-0.965f));
	}
	const float offset_mult = offset + 1.0f;

	auto refine_trial = [&](float* q_inout, float& x_io, float& y_io, float& a_io, unsigned& idx_io, float& f_io,
		float* __restrict sin_arr, float* __restrict cos_arr) noexcept -> bool
		{
			if (!(f_io < fmaf(bestF, adaptive_coeff, 0.0f))) return false;

			__declspec(align(32)) float q_initial[AGP_MAX_FULL_DIM];
			memcpy(q_initial, q_inout, static_cast<size_t>(dim) * sizeof(float));

			const float progress_multiplier = fmaf(p, 0.65f, 1.0f);
			const float lbfgs_trigger = fmaf(dim_f * dim_f, -0.002525f, fmaf(sqrtf(p), -0.45f, 0.575f));
			const float weighted_len_scale = cost.transitionLengthEnergyWeight * cost.fixedLength * cost.fixedLength;
			const int max_lm_iters = static_cast<int>(progress_multiplier * 65.0f);
			const int max_lm_rejects = static_cast<int>(fmaf(progress_multiplier, -5.0f, 14.0f));
			const int max_backtrack = static_cast<int>(progress_multiplier * 20.0f);
			const int max_lbfgs_iters = static_cast<int>(progress_multiplier * 15.0f);
			const float c1 = 1.0e-4f;
			float transition_alpha_cap_local = 1.0f;
			const float f_start = f_io;

			float capture_w;
			float reg_sum;
			if constexpr (IsTransition)
			{
				capture_w = cost.transitionCaptureWeight;
				reg_sum = static_cast<float>(n) * cost.transitionEnergyWeight;
				if constexpr (VariableLen)
					reg_sum = fmaf(static_cast<float>(n), cost.transitionLengthEnergyWeight, reg_sum);
			}
			else
			{
				capture_w = 1.0f;
				reg_sum = 0.0f;
			}

			__declspec(align(32)) float jx[AGP_MAX_FULL_DIM];
			__declspec(align(32)) float jy[AGP_MAX_FULL_DIM];
			__declspec(align(32)) float ja[AGP_MAX_FULL_DIM];
			__declspec(align(32)) float grad_half[AGP_MAX_FULL_DIM];
			__declspec(align(32)) float reg_diag[AGP_MAX_FULL_DIM];
			__declspec(align(32)) float inv_rdiag[AGP_MAX_FULL_DIM];
			__declspec(align(32)) float dir_lm[AGP_MAX_FULL_DIM];

			auto solve_sym3_ldlt = [](const float a00, const float a01, const float a02,
				const float a11, const float a12, const float a22,
				const float b0, const float b1, const float b2,
				float& x0, float& x1, float& x2) noexcept
				{
					const float inv_d0 = 1.0f / a00;
					const float l10 = a01 * inv_d0;
					const float l20 = a02 * inv_d0;

					const float d1 = fmaf(-l10, a01, a11);
					const float inv_d1 = 1.0f / d1;
					const float t12 = fmaf(-l20, a01, a12);
					const float l21 = t12 * inv_d1;

					const float d2 = fmaf(-l21, t12, fmaf(-l20, a02, a22));

					const float y0 = b0;
					const float y1 = fmaf(-l10, y0, b1);
					const float y2 = fmaf(-l21, y1, fmaf(-l20, y0, b2));

					const float z0 = y0 * inv_d0;
					const float z1 = y1 * inv_d1;
					const float z2 = y2 / d2;

					x2 = z2;
					x1 = fmaf(-l21, x2, z1);
					x0 = fmaf(-l20, x2, fmaf(-l10, x1, z0));
				};

			auto build_jacobian = [&](const float* __restrict q,
				const float* __restrict s_arr, const float* __restrict c_arr,
				float& jj00, float& jj01, float& jj02,
				float& jj11, float& jj12, float& jj22) noexcept
				{
					float acc_s = 0.0f;
					float acc_c = 0.0f;
					int k = n - 1;
					while (k >= 0)
					{
						const float Lk = cost.link_length(q, k);
						acc_s = fmaf(Lk, s_arr[k], acc_s);
						acc_c = fmaf(Lk, c_arr[k], acc_c);
						sum_s[k] = acc_s;
						sum_c[k] = acc_c;
						--k;
					}

					jj00 = 0.0f; jj01 = 0.0f; jj02 = 0.0f;
					jj11 = 0.0f; jj12 = 0.0f; jj22 = 0.0f;

					int i = 0;
					__pragma(loop(ivdep))
						while (i < n)
						{
							const float xj = -sum_s[i];
							const float yj = sum_c[i];
							jx[i] = xj;
							jy[i] = yj;
							ja[i] = 1.0f;

							jj00 = fmaf(xj, xj, jj00);
							jj01 = fmaf(xj, yj, jj01);
							jj02 += xj;
							jj11 = fmaf(yj, yj, jj11);
							jj12 += yj;
							jj22 += 1.0f;
							++i;
						}

					if constexpr (VariableLen)
					{
						int j = 0;
						__pragma(loop(ivdep))
							while (j < n)
							{
								const int d = n + j;
								const float xj = c_arr[j];
								const float yj = s_arr[j];
								jx[d] = xj;
								jy[d] = yj;
								ja[d] = 0.0f;

								jj00 = fmaf(xj, xj, jj00);
								jj01 = fmaf(xj, yj, jj01);
								jj11 = fmaf(yj, yj, jj11);
								++j;
							}
					}
				};

			auto compute_full_grad = [&](const float* __restrict q, const float x, const float y, const float a,
				const float* __restrict s_arr, const float* __restrict c_arr,
				float* __restrict grad_out, float& grad_norm2_out) noexcept
				{
					float jj00, jj01, jj02, jj11, jj12, jj22;
					build_jacobian(q, s_arr, c_arr, jj00, jj01, jj02, jj11, jj12, jj22);

					const float ex = x - cost.targetX;
					const float ey = y - cost.targetY;
					const float ea = cost.wrap_pi(a - cost.targetAngle);

					grad_norm2_out = 0.0f;
					int d = 0;
					__pragma(loop(ivdep))
						while (d < dim)
						{
							float gh = capture_w * fmaf(jx[d], ex, fmaf(jy[d], ey, ja[d] * ea));
							if constexpr (IsTransition)
							{
								if (d < n)
								{
									const float r = cost.wrap_pi(q[d] - cost.referenceState[d]);
									gh = fmaf(cost.transitionEnergyWeight, r, gh);
								}
								else
								{
									const float r = q[d] - cost.referenceState[d];
									gh = fmaf(cost.transitionLengthEnergyWeight, r, gh);
								}
							}
							const float g = gh + gh;
							grad_out[d] = g;
							grad_norm2_out = fmaf(g, g, grad_norm2_out);
							++d;
						}
				};

			auto armijo_line_search = [&](const float* __restrict q_base, const float f_base,
				const float* __restrict direction, const float gtd,
				float& alpha_io, float& f_out, float& x_out, float& y_out, float& a_out, unsigned& idx_out,
				float* __restrict sin_out, float* __restrict cos_out) noexcept -> bool
				{
					float alpha = alpha_io;
					if constexpr (HasObstacles && IsTransition)
						alpha = (std::min)(alpha, transition_alpha_cap_local);

					int bt = 0;
					int full_rejects = 0;
					while (bt < max_backtrack)
					{
						agp_axpy_clamp_avx2(q_base, direction, alpha, q_lo, q_hi, q_try, dim);

						float clearance2;
						__declspec(align(32)) float sin_try[AGP_MAX_FULL_DIM];
						__declspec(align(32)) float cos_try[AGP_MAX_FULL_DIM];
						float val2;
						float x2, y2, a2;
						unsigned idx2;

						if (!eval_local_model(q_try, x2, y2, a2, idx2, val2, &clearance2, sin_try, cos_try))
						{
							alpha *= tau;
							++bt;
							continue;
						}

						const float rhs = fmaf(c1 * alpha, gtd, f_base);
						if (val2 > rhs)
						{
							alpha *= tau;
							++bt;
							continue;
						}

						if (!confirm_local_model_if_needed(q_try, x2, y2, a2, clearance2, idx2, val2) || val2 > rhs)
						{
							if constexpr (HasObstacles && IsTransition)
							{
								const float rejected_alpha = alpha;
								alpha *= tau_full_fail;
								transition_alpha_cap_local = (fminf)(transition_alpha_cap_local, rejected_alpha * tau_full_fail);
								++full_rejects;
								++bt;
								if (full_rejects > 2 && alpha < 1.0e-5f) break;
								continue;
							}
							alpha *= tau;
							++bt;
							continue;
						}

						if constexpr (HasObstacles && IsTransition)
						{
							const float grown_cap = alpha * 1.5f;
							if (grown_cap > transition_alpha_cap_local)
								transition_alpha_cap_local = (grown_cap < 1.0f) ? grown_cap : 1.0f;
						}

						memcpy(sin_out, sin_try, static_cast<size_t>(n) * sizeof(float));
						memcpy(cos_out, cos_try, static_cast<size_t>(n) * sizeof(float));

						alpha_io = alpha;
						f_out = val2;
						x_out = x2;
						y_out = y2;
						a_out = a2;
						idx_out = idx2;
						return true;
					}
					return false;
				};

			float lambda = 0.0f;
			int lm_accepts = 0;
			int lm_reject_streak = 0;

			int lm_it = 0;
			while (lm_it < max_lm_iters)
			{
				float jj00, jj01, jj02, jj11, jj12, jj22;
				build_jacobian(q_inout, sin_arr, cos_arr, jj00, jj01, jj02, jj11, jj12, jj22);

				const float ex = x_io - cost.targetX;
				const float ey = y_io - cost.targetY;
				const float ea = cost.wrap_pi(a_io - cost.targetAngle);

				if (!static_cast<bool>(lambda))
				{
					const float sqrtf_mean_diag = sqrtf(fmaf(jj00, 0.33333333333f, fmaf(jj11, 0.33333333333f, jj22 * 0.33333333333f)));
					lambda = fmaf(capture_w, fmaf(sqrtf(jj00), sqrtf_mean_diag, fmaf(sqrtf(jj11), sqrtf_mean_diag, sqrtf(jj22) * sqrtf_mean_diag)), reg_sum) / dim_f * 0.06f;
				}

				float projected_g2 = 0.0f;

				if constexpr (IsTransition)
				{
					float s00 = 1.0f / capture_w;
					float s01 = 0.0f;
					float s02 = 0.0f;
					float s11 = 1.0f / capture_w;
					float s12 = 0.0f;
					float s22 = 1.0f / capture_w;

					float v0 = 0.0f;
					float v1 = 0.0f;
					float v2 = 0.0f;

					int d = 0;
					__pragma(loop(ivdep))
						while (d < dim)
						{
							float rd;
							float rr;
							if (d < n)
							{
								rd = cost.transitionEnergyWeight;
								rr = cost.wrap_pi(q_inout[d] - cost.referenceState[d]);
							}
							else
							{
								rd = weighted_len_scale;
								rr = q_inout[d] - cost.referenceState[d];
							}
							reg_diag[d] = rd;

							const float gh = fmaf(capture_w,
								fmaf(jx[d], ex, fmaf(jy[d], ey, ja[d] * ea)),
								rd * rr);
							grad_half[d] = gh;

							const float invr = 1.0f / (rd + lambda);
							inv_rdiag[d] = invr;

							const float xj = jx[d];
							const float yj = jy[d];
							const float aj = ja[d];

							s00 = fmaf(invr * xj, xj, s00);
							s01 = fmaf(invr * xj, yj, s01);
							s02 = fmaf(invr * xj, aj, s02);
							s11 = fmaf(invr * yj, yj, s11);
							s12 = fmaf(invr * yj, aj, s12);
							s22 = fmaf(invr * aj, aj, s22);

							const float u = invr * gh;
							v0 = fmaf(xj, u, v0);
							v1 = fmaf(yj, u, v1);
							v2 = fmaf(aj, u, v2);

							float pg = gh;
							const float qd = q_inout[d];
							if ((qd <= q_lo[d] && pg > 0.0f) || (qd >= q_hi[d] && pg < 0.0f)) pg = 0.0f;
							projected_g2 = fmaf(pg, pg, projected_g2);
							++d;
						}

					float z0, z1, z2;
					solve_sym3_ldlt(s00, s01, s02, s11, s12, s22, v0, v1, v2, z0, z1, z2);

					d = 0;
					__pragma(loop(ivdep))
						while (d < dim)
						{
							const float invr = inv_rdiag[d];
							const float u = invr * grad_half[d];
							const float corr = invr * fmaf(jx[d], z0, fmaf(jy[d], z1, ja[d] * z2));
							dir_lm[d] = corr - u;
							++d;
						}
				}
				else
				{
					float z0, z1, z2;
					solve_sym3_ldlt(
						jj00 + lambda, jj01, jj02,
						jj11 + lambda, jj12, jj22 + lambda,
						ex, ey, ea,
						z0, z1, z2);

					int d = 0;
					__pragma(loop(ivdep))
						while (d < dim)
						{
							const float gh = fmaf(jx[d], ex, fmaf(jy[d], ey, ja[d] * ea));
							grad_half[d] = gh;
							reg_diag[d] = 0.0f;

							const float dd = -fmaf(jx[d], z0, fmaf(jy[d], z1, ja[d] * z2));
							dir_lm[d] = dd;

							float pg = gh;
							const float qd = q_inout[d];
							if ((qd <= q_lo[d] && pg > 0.0f) || (qd >= q_hi[d] && pg < 0.0f)) pg = 0.0f;
							projected_g2 = fmaf(pg, pg, projected_g2);
							++d;
						}
				}

				if (projected_g2 < fmaf(f_io, 1.0e-4f, 1.0e-4f))
				{
					break;
				}

				float alpha = 1.0f;
				if constexpr (HasObstacles && IsTransition)
					alpha = (std::min)(alpha, transition_alpha_cap_local);

				agp_axpy_clamp_avx2(q_inout, dir_lm, alpha, q_lo, q_hi, q_try, dim);

				float gts = 0.0f;
				float js0 = 0.0f;
				float js1 = 0.0f;
				float js2 = 0.0f;
				float reg_quad = 0.0f;

				int d = 0;
				__pragma(loop(ivdep))
					while (d < dim)
					{
						const float s = q_try[d] - q_inout[d];
						gts = fmaf(grad_half[d], s, gts);
						js0 = fmaf(jx[d], s, js0);
						js1 = fmaf(jy[d], s, js1);
						js2 = fmaf(ja[d], s, js2);
						if constexpr (IsTransition)
							reg_quad = fmaf(reg_diag[d] * s, s, reg_quad);
						++d;
					}

				float gtd_dir = 0.0f;
				__pragma(loop(ivdep))
					for (int d = 0; d < dim; ++d) {
						gtd_dir = fmaf(grad_half[d], dir_lm[d], gtd_dir);
					}

				if (gtd_dir > 0.0f) {
					lambda *= 4.0f;
					++lm_reject_streak;
					if (lm_reject_streak == max_lm_rejects)
						break;
					++lm_it;
					continue;
				}

				float f_before = f_io, f_new, x_new, y_new, a_new;
				__declspec(align(32)) float sin_try[AGP_MAX_FULL_DIM];
				__declspec(align(32)) float cos_try[AGP_MAX_FULL_DIM];
				unsigned idx_new;
				bool armijo_ok = armijo_line_search(q_inout, f_io, dir_lm, gtd_dir,
					alpha, f_new, x_new, y_new, a_new, idx_new,
					sin_try, cos_try);

				if (!armijo_ok)
				{
					lambda *= 4.0f;
					++lm_reject_streak;
					if (lm_reject_streak == max_lm_rejects)
						break;
					++lm_it;
					continue;
				}

				const float js2_norm = fmaf(js0, js0, fmaf(js1, js1, js2 * js2));
				const float model_quad = fmaf(capture_w, js2_norm, reg_quad);
				const float predicted_reduction = fmaf(2.0f, gts, model_quad);
				if (predicted_reduction < 0.0f)
				{
					memcpy(q_inout, q_try, static_cast<size_t>(dim) * sizeof(float));
					f_io = f_new;
					x_io = x_new;
					y_io = y_new;
					a_io = a_new;
					idx_io = idx_new;
					++lm_accepts;
					lm_reject_streak = 0;
				}
				else
				{
					lambda *= 4.0f;
					++lm_reject_streak;
					if (lm_reject_streak == max_lm_rejects) break;
					++lm_it;
					continue;
				}

				if ((f_start - f_io) / f_start > lbfgs_trigger)
				{
					__declspec(align(32)) float s_hist[AGP_MAX_LBFGS_HISTORY][AGP_MAX_FULL_DIM];
					__declspec(align(32)) float y_hist[AGP_MAX_LBFGS_HISTORY][AGP_MAX_FULL_DIM];
					__declspec(align(32)) float rho_hist[AGP_MAX_LBFGS_HISTORY];
					__declspec(align(32)) float alpha_hist[AGP_MAX_LBFGS_HISTORY];
					__declspec(align(32)) float gk[AGP_MAX_FULL_DIM];
					__declspec(align(32)) float g_old[AGP_MAX_FULL_DIM];
					__declspec(align(32)) float q_old[AGP_MAX_FULL_DIM];
					__declspec(align(32)) float work[AGP_MAX_FULL_DIM];
					__declspec(align(32)) float dir[AGP_MAX_FULL_DIM];

					float gk_norm2 = 0.0f;
					compute_full_grad(q_inout, x_io, y_io, a_io, sin_arr, cos_arr, gk, gk_norm2);

					int hist_size = 0;
					int hist_head = 0;
					int it_lbfgs = 0;

					while (it_lbfgs < max_lbfgs_iters)
					{
						if (gk_norm2 < fmaf(f_io, 1.0e-6f, 1.0e-6f))
							break;

						int d = 0;
						if (!static_cast<bool>(hist_size))
						{
							__pragma(loop(ivdep))
								while (d < dim)
								{
									dir[d] = -gk[d];
									++d;
								}
						}
						else
						{
							memcpy(work, gk, static_cast<size_t>(dim) * sizeof(float));

							int logical = hist_size - 1;
							while (logical >= 0)
							{
								const int h = (hist_head + logical) & (LBFGS_M - 1);
								float dot = 0.0f;
								d = 0;
								__pragma(loop(ivdep))
									while (d < dim)
									{
										dot = fmaf(s_hist[h][d], work[d], dot);
										++d;
									}
								const float ah = dot * rho_hist[h];
								alpha_hist[h] = ah;
								d = 0;
								__pragma(loop(ivdep))
									while (d < dim)
									{
										work[d] = fmaf(-ah, y_hist[h][d], work[d]);
										++d;
									}
								--logical;
							}

							const int newest = (hist_head + hist_size - 1) & (LBFGS_M - 1);
							float yy = 0.0f;
							d = 0;
							__pragma(loop(ivdep))
								while (d < dim)
								{
									yy = fmaf(y_hist[newest][d], y_hist[newest][d], yy);
									++d;
								}
							const float gamma = (1.0f / rho_hist[newest]) / yy;

							d = 0;
							__pragma(loop(ivdep))
								while (d < dim)
								{
									dir[d] = gamma * work[d];
									++d;
								}

							logical = 0;
							while (logical < hist_size)
							{
								const int h = (hist_head + logical) & (LBFGS_M - 1);
								float dot = 0.0f;
								d = 0;
								__pragma(loop(ivdep))
									while (d < dim)
									{
										dot = fmaf(y_hist[h][d], dir[d], dot);
										++d;
									}
								const float beta = dot * rho_hist[h];
								const float coeff = alpha_hist[h] - beta;
								d = 0;
								__pragma(loop(ivdep))
									while (d < dim)
									{
										dir[d] = fmaf(coeff, s_hist[h][d], dir[d]);
										++d;
									}
								++logical;
							}

							d = 0;
							__pragma(loop(ivdep))
								while (d < dim)
								{
									dir[d] = -dir[d];
									++d;
								}
						}

						float gtd = 0.0f;
						d = 0;
						__pragma(loop(ivdep))
							while (d < dim)
							{
								gtd = fmaf(gk[d], dir[d], gtd);
								++d;
							}

						if (gtd > 0.0f)
						{
							gtd = 0.0f;
							d = 0;
							__pragma(loop(ivdep))
								while (d < dim)
								{
									dir[d] = -gk[d];
									gtd = fmaf(gk[d], dir[d], gtd);
									++d;
								}
							hist_size = 0;
							hist_head = 0;
						}

						memcpy(q_old, q_inout, static_cast<size_t>(dim) * sizeof(float));
						memcpy(g_old, gk, static_cast<size_t>(dim) * sizeof(float));

						float alpha = (refine_eta_init < 1.0f) ? refine_eta_init : 1.0f;
						float f2, x2, y2, a2;
						unsigned idx2;
						if (!armijo_line_search(q_inout, f_io, dir, gtd, alpha, f2, x2, y2, a2, idx2,
							sin_arr, cos_arr))
							break;

						memcpy(q_inout, q_try, static_cast<size_t>(dim) * sizeof(float));
						f_io = f2;
						x_io = x2;
						y_io = y2;
						a_io = a2;
						idx_io = idx2;

						compute_full_grad(q_inout, x_io, y_io, a_io, sin_arr, cos_arr, gk, gk_norm2);

						float ys = 0.0f;
						float ss = 0.0f;
						float yy = 0.0f;
						__declspec(align(32)) float s_new[AGP_MAX_FULL_DIM];
						__declspec(align(32)) float y_new[AGP_MAX_FULL_DIM];

						d = 0;
						__pragma(loop(ivdep))
							while (d < dim)
							{
								const float sd = q_inout[d] - q_old[d];
								const float yd = gk[d] - g_old[d];
								s_new[d] = sd;
								y_new[d] = yd;
								ys = fmaf(yd, sd, ys);
								ss = fmaf(sd, sd, ss);
								yy = fmaf(yd, yd, yy);
								++d;
							}

						if (ys > 1.0e-6f && ys * ys > 1.0e-4f * ss * yy)
						{
							int h;
							if (hist_size < LBFGS_M)
							{
								h = (hist_head + hist_size) & (LBFGS_M - 1);
								++hist_size;
							}
							else
							{
								h = hist_head;
								hist_head = (hist_head + 1) & (LBFGS_M - 1);
							}

							rho_hist[h] = 1.0f / ys;
							memcpy(s_hist[h], s_new, static_cast<size_t>(dim) * sizeof(float));
							memcpy(y_hist[h], y_new, static_cast<size_t>(dim) * sizeof(float));
						}
						else
						{
							hist_size = 0;
							hist_head = 0;
						}

						++it_lbfgs;
					}
				}

				const float exp_arg = 1.0f / fmaf((f_io - f_before) / predicted_reduction, 1.2525f, 0.9706875f);
				if (exp_arg < 0.5f) {
					lambda *= fmaf(exp_arg,
						fmaf(exp_arg,
							fmaf(exp_arg,
								fmaf(exp_arg,
									fmaf(exp_arg,
										fmaf(exp_arg, 0.00138888889f,
											0.00833333333f),
										0.04166666667f),
									0.16666666667f),
								0.5f),
							1.0f),
						offset_mult);
				}
				else
				{
					const int n = _mm_cvt_ss2si(_mm_set_ss(exp_arg * 1.44269504089f));
					const float r = fmaf(static_cast<float>(-n), 0.69314718056f, exp_arg);
					lambda *= ldexpf(fmaf(r, fmaf(r, fmaf(r, fmaf(r, fmaf(r, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), 1.0f), n) + offset;
				}

				++lm_it;
			}

			return static_cast<bool>(memcmp(q_inout, q_initial, static_cast<size_t>(dim) * sizeof(float)));
		};

	auto evaluate_trial_from_q = [&](const float* q_seed, const bool do_refine) noexcept -> TrialPoint
		{
			TrialPoint tr;
			memcpy(tr.q, q_seed, static_cast<size_t>(dim) * sizeof(float));
			float clearance;
			__declspec(align(32)) float sin_arr[AGP_MAX_FULL_DIM], cos_arr[AGP_MAX_FULL_DIM];
			tr.feasible = cost.evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(tr.q, tr.x, tr.y, tr.a, tr.idx, tr.f, &clearance, sin_arr, cos_arr);
			bool q_changed = false;
			if (tr.feasible)
			{
				tr.idx = cost.total_constraints();
				if constexpr (IsTransition)
					tr.f = cost.compute_transition_objective_from_pose(tr.q, tr.x, tr.y, tr.a);
				else
					tr.f = cost.compute_positioning_objective_from_pose(tr.q, tr.x, tr.y, tr.a);
				if (do_refine) q_changed = refine_trial(tr.q, tr.x, tr.y, tr.a, tr.idx, tr.f, sin_arr, cos_arr);
			}
			tr.feasible = (tr.idx == fullConstraintIndex);
			tr.t = q_changed ? agp_clamp_unit_open_scalar(map.pointToT(tr.q)) : agp_clamp_unit_open_scalar(map.pointToT(q_seed));
			update_best_from_trial(tr);
			return tr;
		};

	auto evaluate_trial_from_t = [&](float t_base, float t_lo, float t_hi) noexcept -> TrialPoint
		{
			TrialPoint tr;
			map.map01ToPoint(t_base, tr.q);
			float clearance;
			__declspec(align(32)) float sin_arr[AGP_MAX_FULL_DIM], cos_arr[AGP_MAX_FULL_DIM];
			tr.feasible = cost.evaluate_state_without_transition_continuity<HasObstacles, VariableLen>(tr.q, tr.x, tr.y, tr.a, tr.idx, tr.f, &clearance, sin_arr, cos_arr);
			bool q_changed = false;
			if (tr.feasible)
			{
				tr.idx = cost.total_constraints();
				if constexpr (IsTransition)
					tr.f = cost.compute_transition_objective_from_pose(tr.q, tr.x, tr.y, tr.a);
				else
					tr.f = cost.compute_positioning_objective_from_pose(tr.q, tr.x, tr.y, tr.a);
				q_changed = refine_trial(tr.q, tr.x, tr.y, tr.a, tr.idx, tr.f, sin_arr, cos_arr);
			}
			tr.feasible = (tr.idx == fullConstraintIndex);
			tr.t = q_changed ? agp_clamp_unit_open_scalar(map.pointToT(tr.q)) : t_base;
			update_best_from_trial(tr);
			if (tr.t < t_lo) tr.t = t_lo;
			if (tr.t > t_hi) tr.t = t_hi;
			return tr;
		};

	auto inject_trial_into_queue =
		[&](const TrialPoint& tr, float r_eff_cur, float adaptive_coeff_cur) noexcept
		{
			IntervalND* src = H[0u];
			if (tr.t < src->x1 || tr.t > src->x2) return;
			const unsigned long long mid_i = agp_t_to_firstchunk_idx_open(map, tr.t);

			IntervalND* L = nullptr;
			IntervalND* Rv = nullptr;

			if (agp_can_have_positive_diameter(map, src->x1, tr.t))
			{
				L = IntervalND::Make(src->x1, tr.t, src->y1, tr.f, src->idx1, tr.idx);
				if (!finalize_interval_geometry_from_cells(L, src->i1, mid_i)) L = nullptr;
			}
			if (agp_can_have_positive_diameter(map, tr.t, src->x2))
			{
				Rv = IntervalND::Make(tr.t, src->x2, tr.f, src->y2, tr.idx, src->idx2);
				if (!finalize_interval_geometry_from_cells(Rv, mid_i, src->i2)) Rv = nullptr;
			}

			if (L == nullptr && Rv == nullptr)
			{
				H[0u] = H.back();
				H.pop_back();
				heap_sift_down(H, 0u);
				return;
			}

			if (L == nullptr || Rv == nullptr)
			{
				IntervalND* only = (L != nullptr) ? L : Rv;
				const float prevMmax = Mmax;

				update_pockets_and_Mmax(only);
				const float m_cur = r_eff_cur * Mmax;
				const float inv_m_cur = 1.0f / m_cur;
				only->ChangeCharacteristic(m_cur, inv_m_cur);

				H[0u] = only;
				heap_sift_down(H, 0u);

				if (Mmax > fmaf(adaptive_coeff_cur, prevMmax, 0.0f))
					recompute_heap_constM(m_cur);

				return;
			}

			const float prevMmax = Mmax;
			update_pockets_and_Mmax(L);
			update_pockets_and_Mmax(Rv);

			const float m_cur = r_eff_cur * Mmax;
			const float inv_m_cur = 1.0f / m_cur;
			L->ChangeCharacteristic(m_cur, inv_m_cur);
			Rv->ChangeCharacteristic(m_cur, inv_m_cur);

			H[0u] = L;
			heap_sift_down(H, 0u);
			heap_push(H, Rv);

			if (Mmax > fmaf(adaptive_coeff_cur, prevMmax, 0.0f))
				recompute_heap_constM(m_cur);

			const float len1 = tr.t - src->x1;
			const float len2 = src->x2 - tr.t;
			if (len1 + len2 == dmax)
			{
				dmax = fmaxf(len1, len2);
				for (auto* pI : H)
				{
					const float Ls = pI->x2 - pI->x1;
					if (Ls > dmax) dmax = Ls;
				}
			}
		};

	auto progress_outgoing = [&]() noexcept
		{
			for (auto it = g_pendingMulti.begin(); it != g_pendingMulti.end(); )
			{
				if (it->req.test()) it = g_pendingMulti.erase(it);
				else ++it;
			}
			for (auto it = g_pendingBest.begin(); it != g_pendingBest.end(); )
			{
				if (it->req.test()) it = g_pendingBest.erase(it);
				else ++it;
			}
		};

	auto select_hj_seed = [&](const std::vector<float, boost::alignment::aligned_allocator<float, 32u>>*& seed_vec,
		float& seed_f, unsigned& seed_idx,
		float& seed_x, float& seed_y) noexcept -> void
		{
			seed_vec = &bestQIndexed;
			seed_f = bestIndexValue;
			seed_idx = bestIndexFound;
			seed_x = bestIndexedX;
			seed_y = bestIndexedY;
		};

	auto run_hooke_jeeves_iteration = [&](float r_eff_cur, float adaptive_coeff_cur, float progress) noexcept
		{
			const bool use_proximal = (progress < 0.95f);

			if (use_proximal != hj_using_proximal)
			{
				const float a0 = use_proximal ? prox_angle_prox : distal_angle_prox;
				const float a1 = use_proximal ? prox_angle_dist : distal_angle_dist;
				const float l0 = use_proximal ? prox_len_prox : distal_len_prox;
				const float l1 = use_proximal ? prox_len_dist : distal_len_dist;

				int d = 0;
				__pragma(loop(ivdep))
					while (d < dim)
					{
						float new_scale = 1.0f;

						if (d < n)
						{
							const float t01 = (n > 1) ? (static_cast<float>(d) / static_cast<float>(n - 1)) : 0.0f;
							new_scale = fmaf(t01, a1 - a0, a0);
						}
						else if constexpr (VariableLen)
						{
							const int j = d - n;
							const float t01 = (n > 1) ? (static_cast<float>(j) / static_cast<float>(n - 1)) : 0.0f;
							new_scale = fmaf(t01, l1 - l0, l0);
						}

						const float old_scale = hj_curr_scale[d];
						if (fabsf(new_scale - old_scale) > 1.0e-7f)
						{
							hj_delta[d] = fmaxf(hj_min_delta, hj_delta[d] * (new_scale / old_scale));
							hj_curr_scale[d] = new_scale;
						}

						++d;
					}

				hj_using_proximal = use_proximal;
			}

			const int* __restrict hj_order = use_proximal ? hj_order_proximal : hj_order_interleaved_distal;

			const std::vector<float, boost::alignment::aligned_allocator<float, 32u>>* seedQ = nullptr;
			float seedF = FLT_MAX, seedX = 0.0f, seedY = 0.0f;
			unsigned seedIdx = 0u;

			select_hj_seed(seedQ, seedF, seedIdx, seedX, seedY);

			TrialPoint base;
			memcpy(base.q, seedQ->data(), static_cast<size_t>(dim) * sizeof(float));
			base.t = agp_clamp_unit_open_scalar(map.pointToT(base.q));
			base.f = seedF;
			base.idx = seedIdx;
			base.x = seedX;
			base.y = seedY;
			base.feasible = (base.idx == fullConstraintIndex);

			thread_local static __declspec(align(32)) float base_q_copy[AGP_MAX_FULL_DIM];
			thread_local static __declspec(align(32)) float current_q[AGP_MAX_FULL_DIM];
			__assume((uintptr_t)base.q % 32 == 0);
			__assume((uintptr_t)base_q_copy % 32 == 0);
			__assume((uintptr_t)current_q % 32 == 0);
			memcpy(base_q_copy, base.q, static_cast<size_t>(dim) * sizeof(float));

			bool improved_any = false;

			int ord = 0;
			while (ord < dim)
			{
				const int d = hj_order[ord];
				const float step = hj_delta[d];

				memcpy(current_q, base.q, static_cast<size_t>(dim) * sizeof(float));
				const float original_val = current_q[d];

				{
					current_q[d] = original_val + step;
					const TrialPoint tp = evaluate_trial_from_q(current_q, false);
					inject_trial_into_queue(tp, r_eff_cur, adaptive_coeff_cur);
					if (better_indexed(tp.idx, tp.f, base.idx, base.f))
					{
						base = tp;
						improved_any = true;
					}
				}

				{
					current_q[d] = original_val - step;
					const TrialPoint tp = evaluate_trial_from_q(current_q, false);
					inject_trial_into_queue(tp, r_eff_cur, adaptive_coeff_cur);
					if (better_indexed(tp.idx, tp.f, base.idx, base.f))
					{
						base = tp;
						improved_any = true;
					}
				}

				++ord;
			}

			if (improved_any)
			{
				__declspec(align(32)) float q_pattern[AGP_MAX_FULL_DIM];
				int d = 0;
				__pragma(loop(ivdep))
					while (d < dim)
					{
						q_pattern[d] = fmaf(hj_pattern_gain, base.q[d] - base_q_copy[d], base.q[d]);
						++d;
					}
				const TrialPoint pat_tp = evaluate_trial_from_q(q_pattern, false);
				if (better_indexed(pat_tp.idx, pat_tp.f, base.idx, base.f))
					base = pat_tp;
				const TrialPoint refined = evaluate_trial_from_q(base.q, true);
				if (better_indexed(refined.idx, refined.f, base.idx, base.f))
				{
					base = refined;
					inject_trial_into_queue(refined, r_eff_cur, adaptive_coeff_cur);
				}
			}
			else
			{
				int d = 0;
				__pragma(loop(ivdep))
					while (d < dim)
					{
						hj_delta[d] = fmaxf(hj_delta[d] * hj_shrink_factor, hj_min_delta);
						++d;
					}
			}
		};

	std::vector<float, boost::alignment::aligned_allocator<float, 32u>> seeds;
	seeds.resize(static_cast<size_t>(AGP_MAX_GENERATED_SEEDS) * static_cast<size_t>(AGP_SEED_STRIDE));

	int seedCnt = generate_heuristic_seeds(
		cost,
		map,
		dim,
		seeds.data(),
		AGP_SEED_STRIDE,
		seed + 7919u * static_cast<unsigned>(rank));

	static constexpr int AGP_K_VALUES[AGP_MAX_FULL_DIM + 1] = {
		0,
		10,
		8,
		7,
		5,
		2,
		2,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0
	};
	const int K = AGP_K_VALUES[dim];

	H.reserve(static_cast<size_t>(maxIter + K + seedCnt + 32u));

	const float t_seed = agp_clamp_unit_open_scalar(map.pointToT(seeds.data()));
	const float interval_size = 0.0004f * dim_f;
	const float t1b = agp_clamp_unit_open_scalar(t_seed - interval_size);
	const float t2b = agp_clamp_unit_open_scalar(t_seed + interval_size);
	const float mid = fmaf(t1b, 0.5f, fmaf(t2b, 0.5f, 0.0f));
	const float R_coeff_const = fmaf(0.01f, dim_f, 0.85f);

	TrialPoint left_tp = evaluate_trial_from_t(t1b, t1b, mid);
	TrialPoint right_tp = evaluate_trial_from_t(t2b, mid, t2b);

	IntervalND* I = make_interval_from_trials(left_tp, right_tp);
	if (static_cast<bool>(I)) {
		update_pockets_and_Mmax(I);
		const float m_curr = r * Mmax;
		const float inv_m_curr = 1.0f / m_curr;
		I->ChangeCharacteristic(m_curr, inv_m_curr);
		I->R = I->R * R_coeff_const;
		H.push_back(I);
	}

	const float t_seed1 = agp_clamp_unit_open_scalar(map.pointToT(seeds.data() + AGP_SEED_STRIDE));
	const float t1b1 = agp_clamp_unit_open_scalar(t_seed1 - interval_size);
	const float t2b1 = agp_clamp_unit_open_scalar(t_seed1 + interval_size);
	const float mid1 = fmaf(t1b1, 0.5f, fmaf(t2b1, 0.5f, 0.0f));

	left_tp = evaluate_trial_from_t(t1b1, t1b1, mid1);
	right_tp = evaluate_trial_from_t(t2b1, mid1, t2b1);

	I = make_interval_from_trials(left_tp, right_tp);
	if (static_cast<bool>(I)) {
		update_pockets_and_Mmax(I);
		const float m_curr = r * Mmax;
		const float inv_m_curr = 1.0f / m_curr;
		I->ChangeCharacteristic(m_curr, inv_m_curr);
		I->R = I->R * R_coeff_const;
		heap_push(H, I);
	}

	const float t_seed2 = agp_clamp_unit_open_scalar(map.pointToT(seeds.data() + 2u * AGP_SEED_STRIDE));
	const float t1b2 = agp_clamp_unit_open_scalar(t_seed2 - interval_size);
	const float t2b2 = agp_clamp_unit_open_scalar(t_seed2 + interval_size);
	const float mid2 = fmaf(t1b2, 0.5f, fmaf(t2b2, 0.5f, 0.0f));

	left_tp = evaluate_trial_from_t(t1b2, t1b2, mid2);
	right_tp = evaluate_trial_from_t(t2b2, mid2, t2b2);

	I = make_interval_from_trials(left_tp, right_tp);
	if (static_cast<bool>(I)) {
		update_pockets_and_Mmax(I);
		const float m_curr = r * Mmax;
		const float inv_m_curr = 1.0f / m_curr;
		I->ChangeCharacteristic(m_curr, inv_m_curr);
		I->R = I->R * R_coeff_const;
		heap_push(H, I);
	}

	const float A = (1.0f / static_cast<float>(std::max(seedCnt - 4, 1))) * log2f(0.00025f / 0.00031f);
	const float start_mult = 0.214f * dim_f;
	const float B = (1.0f / static_cast<float>(std::max(seedCnt - 4, 1))) * (-0.2984f);

	for (size_t i = 3u; i < seedCnt; ++i)
	{
		const float t_seed = agp_clamp_unit_open_scalar(map.pointToT(seeds.data() + i * AGP_SEED_STRIDE));
		const float interval_size = stagnation_seed_interval * exp2f(A * static_cast<float>(i - 3u));

		const float t1b = agp_clamp_unit_open_scalar(t_seed - interval_size);
		const float t2b = agp_clamp_unit_open_scalar(t_seed + interval_size);
		const float mid = fmaf(t1b, 0.5f, fmaf(t2b, 0.5f, 0.0f));

		left_tp = evaluate_trial_from_t(t1b, t1b, mid);
		right_tp = evaluate_trial_from_t(t2b, mid, t2b);

		I = make_interval_from_trials(left_tp, right_tp);
		if (!static_cast<bool>(I)) continue;

		update_pockets_and_Mmax(I);
		const float m_curr = r * Mmax;
		const float inv_m_curr = 1.0f / m_curr;
		I->ChangeCharacteristic(m_curr, inv_m_curr);
		I->R = I->R * start_mult * exp2f(B * static_cast<float>(i - 3u));
		heap_push(H, I);
	}

	std::vector<float, boost::alignment::aligned_allocator<float, 32u>> grid_base;
	grid_base.reserve(static_cast<size_t>(K) + 2u);
	grid_base.emplace_back(a);
	const float fraction = (1.0f / static_cast<float>(K + 1)) * (b - a);
	const float inv_world_K1 = static_cast<float>(rank) / static_cast<float>(world * (K + 1));
	for (int k = 1; k <= K; ++k)
	{
		const float t = agp_clamp_unit_open_scalar(fmaf(fraction, static_cast<float>(k), a + inv_world_K1));
		grid_base.emplace_back(t);
	}
	grid_base.emplace_back(b);

	std::vector<TrialPoint, boost::alignment::aligned_allocator<TrialPoint, 32u>> grid_trials;
	grid_trials.reserve(grid_base.size());
	for (size_t i = 0u; i < grid_base.size(); ++i)
	{
		float lo = grid_base[i];
		float hi = grid_base[i];

		if (i == 0u)
		{
			lo = hi = a;
		}
		else if (i + 1u == grid_base.size())
		{
			lo = hi = b;
		}
		else
		{
			const float tmp = fmaf(grid_base[i], 0.5f, 0.0f);
			lo = fmaf(grid_base[i - 1u], 0.5f, tmp);
			hi = fmaf(grid_base[i + 1u], 0.5f, tmp);
		}

		grid_trials.emplace_back(evaluate_trial_from_t(grid_base[i], lo, hi));
	}

	for (size_t i = 1u; i < grid_trials.size(); ++i)
	{
		IntervalND* I = make_interval_from_trials(grid_trials[i - 1u], grid_trials[i]);
		if (!static_cast<bool>(I)) continue;
		update_pockets_and_Mmax(I);
		const float m_curr = r * Mmax;
		const float inv_m_curr = 1.0f / m_curr;
		I->ChangeCharacteristic(m_curr, inv_m_curr);
		heap_push(H, I);
	}
	recompute_dmax();

	static thread_local std::vector<float, boost::alignment::aligned_allocator<float, 32u>> tl_pos_metric;
	static thread_local std::vector<float, boost::alignment::aligned_allocator<float, 32u>> tl_size_metric;
	static thread_local std::vector<float, boost::alignment::aligned_allocator<float, 32u>> tl_R_val;
	static thread_local std::vector<float, boost::alignment::aligned_allocator<float, 32u>> tl_R_norm;
	static thread_local std::vector<float, boost::alignment::aligned_allocator<float, 32u>> tl_pos_norm;
	static thread_local std::vector<float, boost::alignment::aligned_allocator<float, 32u>> tl_size_norm;
	static thread_local std::vector<int, boost::alignment::aligned_allocator<int, 32u>> tl_selected;
	static thread_local std::vector<unsigned char, boost::alignment::aligned_allocator<unsigned char, 32u>> tl_used;

	if (multi_start)
	{
		while (g_world->iprobe(boost::mpi::any_source, 0)) { MultiCrossMsg dummy; g_world->recv(boost::mpi::any_source, 0, dummy); }
		while (g_world->iprobe(boost::mpi::any_source, 2)) { BestSolutionMsg dummy; g_world->recv(boost::mpi::any_source, 2, dummy); }
		g_world->barrier();
	}

	while (true)
	{
		if (H.front()->diam < eps || ++it == maxIter)
		{
			if (multi_start)
			{
				for (auto& s : g_pendingMulti) s.req.wait();
				for (auto& s : g_pendingBest) s.req.wait();
				g_pendingMulti.clear();
				g_pendingBest.clear();
				while (g_world->iprobe(boost::mpi::any_source, 2))
				{
					BestSolutionMsg incoming;
					g_world->recv(boost::mpi::any_source, 2, incoming);
					UpdateIndexedBestFromMessage(incoming, fullConstraintIndex, bestIndexFound, bestIndexValue, bestQIndexed, bestIndexedX, bestIndexedY, bestIndexedA, bestF, bestQ, bestX, bestY, bestA);
				}
				g_world->barrier();
			}
			bool haveFeasibleBest = (bestIndexFound == fullConstraintIndex);
			if (haveFeasibleBest)
			{
				__declspec(align(32)) float q_local[AGP_MAX_FULL_DIM];
				memcpy(q_local, bestQ.data(), static_cast<size_t>(dim) * sizeof(float));
				float x_final = bestX, y_final = bestY, a_final = bestA, f_final = bestF;
				int last = n - 1;
				float bestLocF = f_final, saved = q_local[last], delta = 0.05f;
				while (delta >= 0.00625f)
				{
					int sgn = -1;
					while (sgn < 2)
					{
						float cand = fmaf(static_cast<float>(sgn), delta, saved);
						float backup = q_local[last];
						q_local[last] = cand;
						float x2_loc, y2_loc, a2_loc;
						unsigned idx2;
						float val2;
						bool feasible2 = cost.evaluate_indexed<HasObstacles, VariableLen, IsTransition>(q_local, x2_loc, y2_loc, a2_loc, idx2, val2, nullptr, nullptr);
						if (feasible2 && val2 < bestLocF)
						{
							bestLocF = val2;
							x_final = x2_loc;
							y_final = y2_loc;
							a_final = a2_loc;
							saved = cand;
						}
						q_local[last] = backup;
						sgn += 2;
					}
					delta *= 0.5f;
				}
				if (bestLocF < f_final)
				{
					q_local[last] = saved;
					bestF = bestLocF;
					bestX = x_final;
					bestY = y_final;
					bestA = a_final;
					const SIZE_T q_loc_dim = static_cast<SIZE_T>(dim) * sizeof(float);
					memcpy(bestQ.data(), q_local, q_loc_dim);
					bestIndexFound = fullConstraintIndex;
					bestIndexValue = bestF;
					bestIndexedX = bestX;
					bestIndexedY = bestY;
					bestIndexedA = bestA;
				}
			}
			else
			{
				bestF = bestIndexValue;
				bestX = bestIndexedX;
				bestY = bestIndexedY;
				bestA = bestIndexedA;
			}
			out_iterations = static_cast<size_t>(it);
			out_achieved_epsilon = H.front()->diam;
			H.clear();
			return;
		}

		p = fmaf(-1.0f / initial_len, dmax, 1.0f);
		const float tmp_multiplier = fmaf(p, 0.65f, -0.45f);
		stag_r_multiplier = fmaf(-fmaf(tmp_multiplier,
			fmaf(tmp_multiplier,
				fmaf(tmp_multiplier,
					fmaf(tmp_multiplier,
						fmaf(tmp_multiplier, 0.164056f, -0.098462f),
						0.240884f),
					-0.351834f),
				0.999996f),
			tmp_multiplier), 1.1f, 1.4f);
		const float p_arg = fmaf(p, 2.3f, -2.9775f);
		float current_r = r;
		if (stag_boost_remaining > 0)
		{
			current_r = r * stag_r_multiplier;
			--stag_boost_remaining;
		}
		const float exp_arg = fmaf(B_dim, p, 0.0f);
		const float exp_arg__ = fmaf(B_dim__, p, 0.0f);
		adaptive_coeff = fmaf(-fmaf(exp_arg, fmaf(exp_arg, fmaf(exp_arg, fmaf(exp_arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), adaptive_coeff_addition, A_dim);
		const float rr = sqrtf(fmaf(-p, 1.0f, 1.0f)), xx = p * p, tt = fmaf(500.0f, p, -486.95472f);
		const float adaptive_coeff_ = (p < 0.95f) ? fmaf(fmaf(first_sqrt, xx, 0.0f), 0.0130349902f, fmaf(-0.04f, p, fmaf(fmaf(first_sqrt, rr, 0.0f), 0.15f, 1.1f)))
			: (p < 0.97390944f) ? fmaf(second_sqrt, rr, 0.9396f)
			: (p < 0.97590944f) ? fmaf(fmaf(fmaf(fmaf(third_sqrt, tt, 0.0f), tt, 0.0f), fmaf(-2.0f, tt, 3.0f), 0.0f), fmaf(0.25f, rr, -0.0396f), fmaf(fmaf(third_sqrt, rr, 0.0f), 0.75f, 0.9396f))
			: fmaf(fourth_sqrt, rr, 0.925f);
		adaptive_coeff__ = fmaf(fmaf(exp_arg__, fmaf(exp_arg__, fmaf(exp_arg__, fmaf(exp_arg__, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), adaptive_coeff_addition__, 2.0f - A_dim__);

		float grad_norm2_best = 0.0f;
		if (no_improve > 0)
		{
			float acc_best = 0.0f;
			__declspec(align(32)) float phi_best[AGP_MAX_FULL_DIM];
			int ii_best = 0;
			while (ii_best < n)
			{
				acc_best = fmaf(bestQ[ii_best], 1.0f, acc_best);
				phi_best[ii_best] = acc_best;
				++ii_best;
			}
			__declspec(align(32)) float s_best[AGP_MAX_FULL_DIM], c_best[AGP_MAX_FULL_DIM];
			FABE13_SINCOS(phi_best, s_best, c_best, n);
			float as_best = 0.0f, ac_best = 0.0f;
			__declspec(align(32)) float sum_s_best[AGP_MAX_FULL_DIM], sum_c_best[AGP_MAX_FULL_DIM];
			int k_best = n - 1;
			while (k_best >= 0)
			{
				float Lk;
				if constexpr (VariableLen)
					Lk = bestQ[n + k_best];
				else
					Lk = cost.fixedLength;
				as_best = fmaf(Lk, s_best[k_best], as_best);
				ac_best = fmaf(Lk, c_best[k_best], ac_best);
				sum_s_best[k_best] = as_best;
				sum_c_best[k_best] = ac_best;
				--k_best;
			}
			const float dx_best = fmaf(bestX, 1.0f, -cost.targetX), dy_best = fmaf(bestY, 1.0f, -cost.targetY);
			const float dist_best = sqrtf(fmaf(dx_best, dx_best, fmaf(dy_best, dy_best, 0.0f)));
			const float inv_dist_best = 1.0f / dist_best;
			int i_best = 0;
			while (i_best < n)
			{
				float gpen_best = 0.0f, g_main_best = 0.0f;
				if constexpr (IsTransition)
				{
					const float dtheta_best = cost.wrap_pi(bestQ[i_best] - cost.referenceState[i_best]);
					gpen_best = fmaf(fmaf(2.0f, cost.transitionEnergyWeight, 0.0f), dtheta_best, 0.0f);
					g_main_best = fmaf(
						fmaf(2.0f, fmaf(cost.transitionCaptureWeight, dx_best, 0.0f), 0.0f),
						-sum_s_best[i_best],
						fmaf(fmaf(fmaf(2.0f, cost.transitionCaptureWeight, 0.0f), dy_best, 0.0f), sum_c_best[i_best], 0.0f)
					);
				}
				else
				{
					g_main_best = fmaf(fmaf(dx_best, -sum_s_best[i_best], fmaf(dy_best, sum_c_best[i_best], 0.0f)), inv_dist_best, 0.0f);
				}
				const float gi_best = g_main_best + gpen_best;
				grad_norm2_best = fmaf(gi_best, gi_best, grad_norm2_best);
				++i_best;
			}
			if constexpr (VariableLen)
			{
				int j_best = 0;
				while (j_best < n)
				{
					float gi_best = 0.0f;
					if constexpr (IsTransition)
					{
						const float two_transCapture = fmaf(cost.transitionCaptureWeight, 2.0f, 0.0f);
						const float two_transLength = fmaf(cost.transitionLengthEnergyWeight, 2.0f, 0.0f);
						gi_best = fmaf(
							fmaf(two_transCapture, dx_best, 0.0f),
							c_best[j_best],
							fmaf(
								fmaf(two_transCapture, dy_best, 0.0f),
								s_best[j_best],
								fmaf(
									two_transLength,
									bestQ[n + j_best],
									fmaf(-1.0f, fmaf(two_transLength, cost.referenceState[n + j_best], 0.0f), 0.0f)
								)
							)
						);
					}
					else
					{
						gi_best = fmaf(fmaf(dx_best, c_best[j_best], fmaf(dy_best, s_best[j_best], 0.0f)), inv_dist_best, 0.0f);
					}
					grad_norm2_best = fmaf(gi_best, gi_best, grad_norm2_best);
					++j_best;
				}
			}
		}

		bool stagnation = (no_improve > noImproveThrDim) && (grad_norm2_best < 0.5e-1f);
		float r_eff = (dim > 2) ? fmaf(-fmaf(p_arg, fmaf(p_arg, fmaf(p_arg, fmaf(p_arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), -0.05f), fmaf(sqrt_dim_minus_1, current_r, 0.0f), 0.0f)
			: fmaf(-fmaf(p_arg, fmaf(p_arg, fmaf(p_arg, fmaf(p_arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), -0.05f), current_r, 0.0f);

		if (stagnation)
		{
			stag_boost_remaining = n_stag_iters;
			float t_seeds[AGP_MAX_FULL_DIM];
			int seed_count = 0;
			int num_ik = num_ik_base;
			float dist_to_target = sqrtf(fmaf(cost.targetX, cost.targetX, cost.targetY * cost.targetY));
			float max_reach = 0.0f;
			if constexpr (VariableLen)
			{
				for (int ii = 0; ii < n; ++ii) max_reach += map.high[n + ii];
			}
			else
			{
				max_reach = cost.fixedLength * static_cast<float>(n);
			}
			float ratio = dist_to_target / max_reach;
			bool prefer_extended = (ratio > 0.7f), prefer_compact = (ratio < 0.4f), use_ik = !(ratio > 0.4f && ratio < 0.7f);
			if (!use_ik)
			{
				const int num_sobol = (std::min)(num_ik, AGP_MAX_FULL_DIM - seed_count);
				std::vector<float, boost::alignment::aligned_allocator<float, 32u>> temp_S(
					static_cast<size_t>(num_sobol) * static_cast<size_t>(AGP_SEED_STRIDE), 0.0f);
				int sobol_gen = generate_sobol_seeds(map, dim, temp_S.data(), AGP_SEED_STRIDE, seed + static_cast<unsigned>(it) + 7919u * static_cast<unsigned>(rank), num_sobol);
				for (int kk = 0; kk < num_sobol && kk < sobol_gen && seed_count < AGP_MAX_FULL_DIM; ++kk)
				{
					const float* s = temp_S.data() + static_cast<size_t>(kk) * static_cast<size_t>(64);
					t_seeds[seed_count++] = agp_clamp_unit_open_scalar(map.pointToT(s));
				}
			}
			else
			{
				__declspec(align(32)) float angles_ccd[AGP_MAX_FULL_DIM] = { 0.0f }, lengths_ccd[AGP_MAX_FULL_DIM];
				if constexpr (VariableLen)
				{
					float len_low = map.low[n], len_high = map.high[n], avg_len = fmaf(len_low, 0.5f, fmaf(len_high, 0.5f, 0.0f));
					for (int ii = 0; ii < n; ++ii) lengths_ccd[ii] = avg_len;
				}
				else
				{
					for (int ii = 0; ii < n; ++ii) lengths_ccd[ii] = cost.fixedLength;
				}
				ccd_ik(cost.targetX, cost.targetY, lengths_ccd, n, angles_ccd, 10);
				__declspec(align(32)) float angles_fabrik[AGP_MAX_FULL_DIM] = { 0.0f }, lengths_fabrik[AGP_MAX_FULL_DIM];
				if constexpr (VariableLen)
				{
					for (int ii = 0; ii < n; ++ii) lengths_fabrik[ii] = lengths_ccd[ii];
				}
				else
				{
					for (int ii = 0; ii < n; ++ii) lengths_fabrik[ii] = cost.fixedLength;
				}
				float targetX_fab = cost.targetX, targetY_fab = cost.targetY;
				for (int iter_fab = 0; iter_fab < 3; ++iter_fab)
				{
					float prevX = targetX_fab, prevY = targetY_fab;
					for (int j = n - 1; j >= 0; --j)
					{
						float len = lengths_fabrik[j];
						float angle_to_target = atan2f(prevY, prevX);
						angles_fabrik[j] = angle_to_target;
						float s_val, c_val;
						FABE13_SINCOS(&angle_to_target, &s_val, &c_val, 1);
						prevY = fmaf(-len, s_val, prevY);
						prevX = fmaf(-len, c_val, prevX);
					}
				}
				if (prefer_extended)
				{
					{
						float q_ccd[AGP_MAX_FULL_DIM];
						for (int ii = 0; ii < n; ++ii) q_ccd[ii] = angles_ccd[ii];
						if constexpr (VariableLen)
						{
							for (int ii = 0; ii < n; ++ii) q_ccd[n + ii] = lengths_ccd[ii];
						}
						t_seeds[seed_count++] = agp_clamp_unit_open_scalar(map.pointToT(q_ccd));
					}
					unsigned st_ik = seed + static_cast<unsigned>(it) + 222u;
					int remaining = num_ik - 1;
					for (int v = 0; v < remaining && seed_count < AGP_MAX_FULL_DIM; ++v)
					{
						float noisy_angles[AGP_MAX_FULL_DIM], noisy_lengths[AGP_MAX_FULL_DIM];
						for (int ii = 0; ii < n; ++ii)
						{
							XOR_STEP(st_ik);
							float rnd = fmaf(static_cast<float>(st_ik & 0xFFFFFFu), 5.9604645e-8f, 0.0f);
							noisy_angles[ii] = fmaf(rnd, 0.2f, angles_ccd[ii] - 0.1f);
						}
						if constexpr (VariableLen)
						{
							for (int ii = 0; ii < n; ++ii)
							{
								XOR_STEP(st_ik);
								float rnd = fmaf(static_cast<float>(st_ik & 0xFFFFFFu), 5.9604645e-8f, 0.0f);
								noisy_lengths[ii] = fmaf(rnd, 0.1f, lengths_ccd[ii] - 0.05f);
							}
						}
						__declspec(align(32)) float q_temp[AGP_MAX_FULL_DIM];
						for (int ii = 0; ii < n; ++ii) q_temp[ii] = noisy_angles[ii];
						if constexpr (VariableLen)
						{
							for (int ii = 0; ii < n; ++ii) q_temp[n + ii] = noisy_lengths[ii];
						}
						t_seeds[seed_count++] = agp_clamp_unit_open_scalar(map.pointToT(q_temp));
					}
				}
				else if (prefer_compact)
				{
					{
						__declspec(align(32)) float q_fabrik[AGP_MAX_FULL_DIM];
						for (int ii = 0; ii < n; ++ii) q_fabrik[ii] = angles_fabrik[ii];
						if constexpr (VariableLen)
						{
							for (int ii = 0; ii < n; ++ii) q_fabrik[n + ii] = lengths_fabrik[ii];
						}
						t_seeds[seed_count++] = agp_clamp_unit_open_scalar(map.pointToT(q_fabrik));
					}
					unsigned st_ik = seed + static_cast<unsigned>(it) + 222u;
					int remaining = num_ik - 1;
					for (int v = 0; v < remaining && seed_count < AGP_MAX_FULL_DIM; ++v)
					{
						float noisy_angles[AGP_MAX_FULL_DIM], noisy_lengths[AGP_MAX_FULL_DIM];
						for (int ii = 0; ii < n; ++ii)
						{
							XOR_STEP(st_ik);
							float rnd = fmaf(static_cast<float>(st_ik & 0xFFFFFFu), 5.9604645e-8f, 0.0f);
							noisy_angles[ii] = fmaf(rnd, 0.06f, angles_fabrik[ii] - 0.03f);
						}
						if constexpr (VariableLen)
						{
							for (int ii = 0; ii < n; ++ii)
							{
								XOR_STEP(st_ik);
								float rnd = fmaf(static_cast<float>(st_ik & 0xFFFFFFu), 5.9604645e-8f, 0.0f);
								noisy_lengths[ii] = fmaf(rnd, 0.06f, lengths_fabrik[ii] - 0.03f);
							}
						}
						__declspec(align(32)) float q_temp[AGP_MAX_FULL_DIM];
						for (int ii = 0; ii < n; ++ii) q_temp[ii] = noisy_angles[ii];
						if constexpr (VariableLen)
						{
							for (int ii = 0; ii < n; ++ii) q_temp[n + ii] = noisy_lengths[ii];
						}
						t_seeds[seed_count++] = agp_clamp_unit_open_scalar(map.pointToT(q_temp));
					}
				}
			}
			for (int s = 0; s < seed_count; ++s)
			{
				float t_center = t_seeds[s];
				float interval_size = stagnation_seed_interval;
				float t1b = agp_clamp_unit_open_scalar(fmaf(-interval_size, 0.5f, t_center));
				float t2b = agp_clamp_unit_open_scalar(fmaf(interval_size, 0.5f, t_center));
				float mid = fmaf(t1b, 0.5f, fmaf(t2b, 0.5f, 0.0f));
				TrialPoint left_tp = evaluate_trial_from_t(t1b, t1b, mid);
				TrialPoint right_tp = evaluate_trial_from_t(t2b, mid, t2b);
				IntervalND* I = make_interval_from_trials(left_tp, right_tp);
				if (!static_cast<bool>(I)) continue;
				update_pockets_and_Mmax(I);
				const float m_curr = r_eff * Mmax;
				const float inv_m_curr = 1.0f / m_curr;
				I->ChangeCharacteristic(m_curr, inv_m_curr);
				I->R = I->R * R_coeff_const;
				heap_push(H, I);
			}
			recompute_dmax();
			no_improve = 0;
		}

		const float bestFOld = bestF;
		if (adaptive && (!(it & 1))) run_hooke_jeeves_iteration(r_eff, adaptive_coeff_, p);
		else
		{
			IntervalND* cur = heap_pop_front(H);
			const float x1 = cur->x1, x2 = cur->x2, y1 = cur->y1, y2 = cur->y2;
			const float m = r_eff * Mmax;
			const float inv_m = 1.0f / m;
			float tBase;
			switch (dim) {
			case 1:  tBase = step<1>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 2:  tBase = step<2>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 3:  tBase = step<3>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 4:  tBase = step<4>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 5:  tBase = step<5>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 6:  tBase = step<6>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 7:  tBase = step<7>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 8:  tBase = step<8>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 9:  tBase = step<9>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 10: tBase = step<10>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 11: tBase = step<11>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 12: tBase = step<12>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 13: tBase = step<13>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 14: tBase = step<14>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			case 15: tBase = step<15>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			default: tBase = step<16>(m, inv_m, x1, x2, y1, y2, r_eff, cur->idx1, cur->idx2); break;
			}
			const TrialPoint trNew = evaluate_trial_from_t(tBase, x1, x2);
			const unsigned long long mid_i = agp_t_to_firstchunk_idx_open(map, trNew.t);

			IntervalND* L = nullptr;
			IntervalND* Rv = nullptr;

			if (agp_can_have_positive_diameter(map, x1, trNew.t))
			{
				L = IntervalND::Make(x1, trNew.t, y1, trNew.f, cur->idx1, trNew.idx);
				if (!finalize_interval_geometry_from_cells(L, cur->i1, mid_i)) L = nullptr;
			}
			if (agp_can_have_positive_diameter(map, trNew.t, x2))
			{
				Rv = IntervalND::Make(trNew.t, x2, trNew.f, y2, trNew.idx, cur->idx2);
				if (!finalize_interval_geometry_from_cells(Rv, mid_i, cur->i2)) Rv = nullptr;
			}

			if (L == nullptr && Rv == nullptr)
			{
			}
			else if (L == nullptr || Rv == nullptr)
			{
				IntervalND* only = (static_cast<bool>(L)) ? L : Rv;
				const float prevMmax = Mmax;
				update_pockets_and_Mmax(only);
				if ((p > 0.7f && !(it % 3) && dmax < 0.7f) || p > 0.9f)
				{
					const float alpha = p * p;
					const float beta = fmaf(-alpha, 1.0f, 2.0f);
					const float MULT = (1.0f / dmax) * Mmax;
					const float global_coeff = fmaf(MULT, r_eff, -MULT);
					const float GF = beta * global_coeff;
					const float m_curr = fmaf(GF, only->N_factor, only->M * alpha);
					const float inv_m_curr = 1.0f / m_curr;
					only->ChangeCharacteristic(m_curr, inv_m_curr);
					RecomputeR_AffineM_Mixed_ND(H.data(), H.size(), GF, alpha);
					heap_make(H);
				}
				else
				{
					const float m_curr = r_eff * Mmax;
					const float inv_m_curr = 1.0f / m_curr;
					only->ChangeCharacteristic(m_curr, inv_m_curr);
					if (only->M > adaptive_coeff_ * prevMmax) recompute_heap_constM(m_curr);
				}
				heap_push(H, only);
			}
			else
			{
				const float Mloc = fmaxf(L->M, Rv->M);
				const float prevMmax = Mmax;
				update_pockets_and_Mmax(L);
				update_pockets_and_Mmax(Rv);
				const float len1 = trNew.t - x1;
				const float len2 = x2 - trNew.t;
				if (len1 + len2 == dmax)
				{
					dmax = fmaxf(len1, len2);
					for (auto* pI : H)
					{
						const float Ls = pI->x2 - pI->x1;
						if (Ls > dmax) dmax = Ls;
					}
				}

				if ((p > 0.7f && !(it % 3) && dmax < 0.7f) || p > 0.9f)
				{
					const float alpha = fmaf(p, p, 0.0f);
					const float beta = fmaf(-alpha, 1.0f, 2.0f);
					const float MULT = fmaf((1.0f / dmax), Mmax, 0.0f);
					const float global_coeff = fmaf(MULT, r_eff, -MULT);
					const float GF = beta * global_coeff;
					const float m_curr_L = fmaf(GF, L->N_factor, L->M * alpha);
					const float m_curr_Rv = fmaf(GF, Rv->N_factor, Rv->M * alpha);
					const float inv_m_curr_L = 1.0f / m_curr_L;
					const float inv_m_curr_Rv = 1.0f / m_curr_Rv;
					L->ChangeCharacteristic(m_curr_L, inv_m_curr_L);
					Rv->ChangeCharacteristic(m_curr_Rv, inv_m_curr_Rv);
					RecomputeR_AffineM_Mixed_ND(H.data(), H.size(), GF, alpha);
					heap_make(H);
				}
				else
				{
					const float m_curr = r_eff * Mmax;
					const float inv_m_curr = 1.0f / m_curr;
					L->ChangeCharacteristic(m_curr, inv_m_curr);
					Rv->ChangeCharacteristic(m_curr, inv_m_curr);
					if (Mloc > adaptive_coeff_ * prevMmax) recompute_heap_constM(m_curr);
				}
				heap_push(H, L);
				heap_push(H, Rv);
			}
		}

		if (multi_start)
		{
			if ((stagnation || bestF < fmaf(fmaf(bestFOld, 0.6f, 0.0f), adaptive_coeff__, 0.0f)) && it - last_send_T >= send_interval_T)
			{
				last_send_T = it;
				progress_outgoing();

				const size_t scan_count = H.size();
				unsigned intervals_to_send = (dim < 12) ? static_cast<unsigned>(sqrtf(fmaf(dim_f, 5.5f, 0.0f))) : 7u;
				if (intervals_to_send > scan_count) intervals_to_send = static_cast<unsigned>(scan_count);

				auto& pos_metric = tl_pos_metric;
				auto& size_metric = tl_size_metric;
				auto& R_val = tl_R_val;
				auto& R_norm = tl_R_norm;
				auto& pos_norm = tl_pos_norm;
				auto& size_norm = tl_size_norm;
				auto& selected = tl_selected;
				auto& used = tl_used;

				pos_metric.resize(scan_count);
				size_metric.resize(scan_count);
				R_val.resize(scan_count);
				R_norm.resize(scan_count);
				pos_norm.resize(scan_count);
				size_norm.resize(scan_count);
				used.resize(scan_count);
				memset(used.data(), 0, scan_count);

				selected.clear();
				selected.reserve(intervals_to_send);

				const float alpha = 0.63f - sqrtf(fmaxf(fmaf(-p, 0.113f, 1.13f), 0.0f));
				const float beta = 1.0f - alpha;
				const float w_pos = -sqrtf(fmaf(p, 0.007f, 1.0f)) * 3.085f;
				const float w_size = 5.085f - w_pos;
				const int num_bins = (p < 1.0f - 1e-9f) ? static_cast<int>(1.0f / (1.0f - p)) : 1;

				float R_max = -FLT_MAX, R_min = FLT_MAX;
				float size_max = -FLT_MAX, size_min = FLT_MAX;
				float pos_max = -FLT_MAX, pos_min = FLT_MAX;

				for (size_t idx = 0u; idx < scan_count; ++idx)
				{
					IntervalND* I = H[idx];
					const float center = fmaf(I->x1, 0.5f, fmaf(I->x2, 0.5f, 0.0f));
					const float len = I->x2 - I->x1;
					const float size = fmaf(static_cast<float>(1 << I->span_level), len, 0.0f);

					if (p < 0.95f && num_bins > 1)
					{
						float bin = floorf(fmaf(static_cast<float>(num_bins), center, 0.0f));
						if (bin >= static_cast<float>(num_bins)) bin = static_cast<float>(num_bins - 1);
						pos_metric[idx] = bin / static_cast<float>(num_bins - 1);
					}
					else
					{
						pos_metric[idx] = center;
					}

					size_metric[idx] = size;
					R_val[idx] = I->R;

					if (pos_metric[idx] > pos_max) pos_max = pos_metric[idx];
					if (pos_metric[idx] < pos_min) pos_min = pos_metric[idx];
					if (size > size_max) size_max = size;
					if (size < size_min) size_min = size;
					if (I->R > R_max) R_max = I->R;
					if (I->R < R_min) R_min = I->R;
				}

				const float inv_R_range = (R_max > R_min) ? 1.0f / (R_max - R_min) : 0.0f;
				const float inv_pos_range = (pos_max > pos_min) ? 1.0f / (pos_max - pos_min) : 0.0f;
				const float inv_size_range = (size_max > size_min) ? 1.0f / (size_max - size_min) : 0.0f;

				for (size_t idx = 0u; idx < scan_count; ++idx)
				{
					R_norm[idx] = fmaf(-inv_R_range, R_min, fmaf(inv_R_range, R_val[idx], 0.0f));
					pos_norm[idx] = fmaf(-inv_pos_range, pos_min, fmaf(inv_pos_range, pos_metric[idx], 0.0f));
					size_norm[idx] = fmaf(-inv_size_range, size_min, fmaf(inv_size_range, size_metric[idx], 0.0f));
				}

				const float inv_sqrt_wsum = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(w_pos + w_size)));

				selected.emplace_back(0);
				used[0] = 1u;

				while (selected.size() < intervals_to_send)
				{
					int best_idx = -1;
					float best_score = -FLT_MAX;

					for (size_t idx = 0u; idx < scan_count; ++idx)
					{
						if (used[idx]) continue;

						float min_dist = FLT_MAX;
						for (int s : selected)
						{
							float d = pos_norm[idx] - pos_norm[static_cast<size_t>(s)];
							float d2 = fmaf(fmaf(w_pos, d, 0.0f), d, 0.0f);
							d = size_norm[idx] - size_norm[static_cast<size_t>(s)];
							d2 += fmaf(fmaf(w_size, d, 0.0f), d, 0.0f);
							if (d2 < min_dist) min_dist = d2;
						}

						const float score = fmaf(sqrtf(min_dist) * inv_sqrt_wsum, beta, R_norm[idx] * alpha);
						if (score > best_score)
						{
							best_score = score;
							best_idx = static_cast<int>(idx);
						}
					}

					if (best_idx < 0) break;
					selected.emplace_back(best_idx);
					used[static_cast<size_t>(best_idx)] = 1u;
				}

				MultiCrossMsg out;
				out.count = static_cast<unsigned>(selected.size());

				for (unsigned s = 0u; s < out.count; ++s)
				{
					AgpFillIntervalWire(out.intervals[s], H[static_cast<size_t>(selected[s])]);
				}

				size_t iterations = comm_levels;
				bool active = true;
				bool invert_T = static_cast<bool>((static_cast<int>(exchange_counter_T) + 1) & 1);
				size_t ii2 = 0u;
				while (ii2 < iterations && active)
				{
					size_t step = 1ULL << ii2;
					int partner = rank ^ static_cast<int>(step);
					if (partner < world)
					{
						bool am_sender = ((!!(rank & static_cast<int>(step))) ^ invert_T);
						if (am_sender)
						{
							g_pendingMulti.emplace_back(*g_world, partner, out);
							if (g_pendingMulti.size() > AGP_MULTI_MAX_COUNT)
							{
								g_pendingMulti.front().req.wait();
								g_pendingMulti.pop_front();
							}
							active = false;
						}
					}
					++ii2;
				}
				++exchange_counter_T;
			}

			bool indexedBestImprovedSinceLastSend = better_indexed(bestIndexFound, bestIndexValue, lastSentBestMsg.bestIndex, lastSentBestMsg.bestF);
			if (indexedBestImprovedSinceLastSend && it - last_send_best >= send_interval_best)
			{
				last_send_best = it;
				progress_outgoing();
				size_t iterations = comm_levels;
				bool active = true;
				bool invert_best = static_cast<bool>((static_cast<int>(exchange_counter) + 1) & 1);
				size_t ii2 = 0u;
				while (ii2 < iterations && active)
				{
					size_t step = 1ULL << ii2;
					int partner = rank ^ static_cast<int>(step);
					if (partner < world)
					{
						bool am_sender = ((!!(rank & static_cast<int>(step))) ^ invert_best);
						if (am_sender)
						{
							BestSolutionMsg outMsg;
							InitBestSolutionMsg(outMsg);
							if (FillBestSolutionMsg(outMsg, bestIndexFound, bestIndexValue, bestIndexedX, bestIndexedY, bestIndexedA, bestQIndexed) &&
								better_indexed(outMsg.bestIndex, outMsg.bestF, lastSentBestMsg.bestIndex, lastSentBestMsg.bestF))
							{
								g_pendingBest.emplace_back(*g_world, partner, outMsg);
								lastSentBestMsg = outMsg;
							}
							if (g_pendingBest.size() > AGP_MULTI_MAX_COUNT)
							{
								g_pendingBest.front().req.wait();
								g_pendingBest.pop_front();
							}
							active = false;
						}
					}
					++ii2;
				}
				++exchange_counter;
			}

			while (g_world->iprobe(boost::mpi::any_source, 0))
			{
				MultiCrossMsg in;
				g_world->recv(boost::mpi::any_source, 0, in);

				for (unsigned ii = 0u; ii < in.count; ++ii)
				{
					const IntervalWire& w = in.intervals[ii];
					IntervalND* inj = AgpCreateIntervalFromWire(w);

					const int span = inj->span_level;
					const float pocketM = (inj->M > M_by_span[span]) ? inj->M : M_by_span[span];
					const float candidateMmax = (pocketM > Mmax) ? pocketM : Mmax;
					const float m_inj = r_eff * candidateMmax;
					const float inv_m_inj = 1.0f / m_inj;

					inj->ChangeCharacteristic(m_inj, inv_m_inj);

					if (inj->R > fmaf(adaptive_coeff, H.front()->R, 0.0f) || stagnation)
					{
						const float prevMmax = Mmax;
						update_pockets_and_Mmax(inj);
						heap_push(H, inj);
						if (Mmax > fmaf(adaptive_coeff_, prevMmax, 0.0f))
							recompute_heap_constM(fmaf(r_eff, Mmax, 0.0f));
					}
				}
			}

			while (g_world->iprobe(boost::mpi::any_source, 2))
			{
				BestSolutionMsg incoming;
				g_world->recv(boost::mpi::any_source, 2, incoming);
				UpdateIndexedBestFromMessage(incoming, fullConstraintIndex, bestIndexFound, bestIndexValue, bestQIndexed, bestIndexedX, bestIndexedY, bestIndexedA, bestF, bestQ, bestX, bestY, bestA);
			}
		}
	}
}

//============================================================
//                     EXPORTED FUNCTIONS
//============================================================

extern "C" __declspec(dllexport) __declspec(noalias) void AGP_Manip2D(
	int n,
	bool variableLengths,
	float maxTheta,
	float tx,
	float ty,
	float ta,
	int maxIter,
	float r_param,
	bool adaptive,
	float eps,
	unsigned int seed,
	float baseLength,
	float stretchFactor,
	float* obstacleData,
	unsigned obstacleCount,
	float** out_bestQ,
	float* out_bestX,
	float* out_bestY,
	float* out_bestA,
	float* out_bestF,
	size_t* out_actualIterations,
	float* out_achievedEps,
	int mode,
	const float* referenceStates
) noexcept
{
	Slab* slab = tls.local();

	const int dim = n + (variableLengths ? n : 0);

	const int rank = g_world->rank();
	const int world = g_world->size();

	MortonCachePerRank mc;
	mc.baseSeed = agp_splitmix32(seed ^ (0x9E3779B9u * static_cast<unsigned>(rank + 1)));

	mc.permCache.resize(static_cast<size_t>(dim));
	mc.invMaskCache.resize(static_cast<size_t>(dim));

	agp_setup_rank_morton_orientation(mc, dim, rank);

	__declspec(align(32)) float low[AGP_MAX_FULL_DIM];
	__declspec(align(32)) float high[AGP_MAX_FULL_DIM];

	__pragma(loop(ivdep))
		for (int i = 0; i < n; ++i)
		{
			low[i] = -maxTheta;
			high[i] = maxTheta;
		}
	if (variableLengths)
	{
		const float lengthLower = baseLength / stretchFactor;
		const float lengthUpper = baseLength * stretchFactor;
		int ii = 0;
		__pragma(loop(ivdep))
			while (ii < n)
			{
				low[n + ii] = lengthLower;
				high[n + ii] = lengthUpper;
				++ii;
			}
	}

	const float dim_f = static_cast<float>(dim);
	const float exp_arg_lvls = fmaf(-dim_f, 0.455f, 0.0f);
	const int fine_lvls = static_cast<int>(fminf(
		fmaf(fmaf(exp_arg_lvls, fmaf(exp_arg_lvls,
			fmaf(exp_arg_lvls, fmaf(exp_arg_lvls, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f),
			30.0f, 6.125f), 13.0f));
	const float exp_arg_lvl0 = fmaf(-dim_f, 0.08f, 0.0f);
	const int levels0 = static_cast<int>(fminf(
		fmaf(-fminf(fmaf(fmaf(exp_arg_lvl0, fmaf(exp_arg_lvl0,
			fmaf(exp_arg_lvl0, fmaf(exp_arg_lvl0, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f),
			21.7f, -11.3f), 0.0f),
			1.0f, static_cast<float>(fine_lvls)), 8.0f));

	const int coarseIter = maxIter >> 1;
	const int fineIter = maxIter - coarseIter;

	MortonND map0(dim, levels0, low, high, mc);
	std::vector<IntervalND*, boost::alignment::aligned_allocator<IntervalND*, 32u>> H;
	H.reserve(static_cast<size_t>(fineIter) + 32u);

	size_t oi, total_oi = 0u;
	float oe, total_oe = 0.0f;

	const float M_arg = ldexpf(1.0f, -levels0);
	const float maxLinkLength = variableLengths ? baseLength * stretchFactor : baseLength;
	const float M_prior = fmaf(
		fmaf(variableLengths ? 2.8284271f : 2.0f, fmaf(static_cast<float>(n), maxLinkLength, 0.0f), 0.0f),
		fmaf(M_arg,
			fmaf(M_arg,
				fmaf(M_arg,
					fmaf(M_arg,
						fmaf(M_arg, 0.164056f, -0.098462f),
						0.240884f),
					-0.351834f),
				0.999996f),
			M_arg),
		0.0f);

	std::vector<float, boost::alignment::aligned_allocator<float, 32u>> bestQIndexed;
	std::vector<float, boost::alignment::aligned_allocator<float, 32u>> bestQ;
	bestQIndexed.reserve(static_cast<size_t>(dim));
	bestQ.reserve(static_cast<size_t>(dim));
	float bestF = FLT_MAX, bestX = 0.0f, bestY = 0.0f, bestA = 0.0f;
	const bool hasObs = static_cast<bool>(obstacleCount);
	const bool isTransition = static_cast<bool>(mode);

	ManipCost cost(n, variableLengths, tx, ty, ta, maxTheta, baseLength, stretchFactor,
		obstacleData, obstacleCount, mode);

	if (isTransition) cost.SetTransitionReference(referenceStates);
	const unsigned fullConstraintIndex = hasObs && isTransition ? cost.total_constraints() + 1u : cost.total_constraints();

	char* saved1 = slab->current;
	if (hasObs) {
		if (variableLengths) {
			if (isTransition)
				agp_run_branch_mpi<true, true, true>(map0, cost, coarseIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior);
			else
				agp_run_branch_mpi<true, true, false>(map0, cost, coarseIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior);
		}
		else {
			if (isTransition)
				agp_run_branch_mpi<true, false, true>(map0, cost, coarseIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior);
			else
				agp_run_branch_mpi<true, false, false>(map0, cost, coarseIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior);
		}
	}
	else {
		if (variableLengths) {
			if (isTransition)
				agp_run_branch_mpi<false, true, true>(map0, cost, coarseIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior);
			else
				agp_run_branch_mpi<false, true, false>(map0, cost, coarseIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior);
		}
		else {
			if (isTransition)
				agp_run_branch_mpi<false, false, true>(map0, cost, coarseIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior);
			else
				agp_run_branch_mpi<false, false, false>(map0, cost, coarseIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior);
		}
	}
	slab->current = saved1;
	total_oi += oi;
	total_oe = oe;

	unsigned bestIndexFound = 0u;
	float bestIndexValue = FLT_MAX, bestIndexedX = 0.0f, bestIndexedY = 0.0f, bestIndexedA = 0.0f;

	MortonND map1(dim, fine_lvls, low, high, mc);

	const float M_prior_fine = fmaf(1.0f / fine_lvls, fmaf(static_cast<float>(levels0), M_prior, 0.0f), 0.0f);

	char* saved2 = slab->current;
	if (hasObs) {
		if (variableLengths) {
			if (isTransition)
				agp_run_branch_mpi<true, true, true>(map1, cost, fineIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior_fine);
			else
				agp_run_branch_mpi<true, true, false>(map1, cost, fineIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior_fine);
		}
		else {
			if (isTransition)
				agp_run_branch_mpi<true, false, true>(map1, cost, fineIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior_fine);
			else
				agp_run_branch_mpi<true, false, false>(map1, cost, fineIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior_fine);
		}
	}
	else {
		if (variableLengths) {
			if (isTransition)
				agp_run_branch_mpi<false, true, true>(map1, cost, fineIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior_fine);
			else
				agp_run_branch_mpi<false, true, false>(map1, cost, fineIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior_fine);
		}
		else {
			if (isTransition)
				agp_run_branch_mpi<false, false, true>(map1, cost, fineIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior_fine);
			else
				agp_run_branch_mpi<false, false, false>(map1, cost, fineIter, r_param, adaptive, eps, seed,
					H, bestQ, bestQIndexed, bestF, bestX, bestY, bestA, oi, oe, M_prior_fine);
		}
	}
	slab->current = saved2;
	total_oi += oi;
	total_oe = oe;

	if (world > 1 && !(mode & 2))
	{
		BestSolutionMsg best;
		InitBestSolutionMsg(best);
		FillBestSolutionMsg(best, bestIndexFound, bestIndexValue,
			bestIndexedX, bestIndexedY, bestIndexedA, bestQIndexed);

		const size_t iterations = std::bit_width(static_cast<size_t>(world - 1));
		for (size_t itx = 0; itx < iterations; ++itx)
		{
			const size_t step = 1ull << itx;
			const int partner = rank ^ static_cast<int>(step);
			if (partner < world)
			{
				if (static_cast<bool>(rank & static_cast<int>(step)))
				{
					g_world->send(partner, 3, best);
					return;
				}
				else
				{
					BestSolutionMsg in;
					InitBestSolutionMsg(in);
					g_world->recv(partner, 3, in);
					if (better_indexed(in.bestIndex, in.bestF, best.bestIndex, best.bestF))
						best = in;
				}
			}
		}
		UpdateIndexedBestFromMessage(best, fullConstraintIndex,
			bestIndexFound, bestIndexValue, bestQIndexed, bestIndexedX, bestIndexedY, bestIndexedA,
			bestF, bestQ, bestX, bestY, bestA);
	}

	const size_t finalSize = static_cast<size_t>(dim);
	const SIZE_T bytes = finalSize * sizeof(float);
	const bool use_indexed = bestQ.empty();
	*out_bestQ = static_cast<float*>(CoTaskMemAlloc(bytes));
	memcpy(*out_bestQ, use_indexed ? bestQIndexed.data() : bestQ.data(), bytes);
	*out_bestX = use_indexed ? bestIndexedX : bestX;
	*out_bestY = use_indexed ? bestIndexedY : bestY;
	*out_bestA = use_indexed ? bestIndexedA : bestA;
	*out_bestF = use_indexed ? bestIndexValue : bestF;
	*out_actualIterations = total_oi;
	*out_achievedEps = total_oe;
}

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline int AgpInit() noexcept
{
	g_env = new boost::mpi::environment();
	g_world = new boost::mpi::communicator();
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	return g_world->rank();
}

__declspec(align(32)) struct RunParams final
{
	unsigned nSegments, varLen;
	float maxTheta, tx, ty, ta;
	int maxIter;
	float r;
	unsigned adaptive;
	float eps;
	unsigned seed;
	float baseLength, stretchFactor, obstacleData[MAX_OBSTACLES * 3u];
	unsigned obstacleCount;
	int mode;
	float referenceStates[AGP_MAX_FULL_DIM];
	float finalStates[AGP_MAX_FULL_DIM];

	template<typename Archive>
	__forceinline void serialize(Archive& ar, unsigned)
	{
		ar& nSegments
			& varLen
			& maxTheta
			& tx
			& ty
			& ta
			& maxIter
			& r
			& adaptive
			& eps
			& seed
			& baseLength
			& stretchFactor
			& obstacleData
			& obstacleCount
			& mode
			& referenceStates
			& finalStates;
	}
};

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline
void AgpStartManipND(
	int n,
	bool variableLengths,
	float maxTheta,
	float tx,
	float ty,
	float ta,
	int maxIter,
	float r_param,
	bool adaptive,
	float eps,
	unsigned int seed,
	float baseLength,
	float stretchFactor,
	const float* obstacleData,
	int obstacleCount,
	int mode,
	float referenceStates[AGP_MAX_FULL_DIM],
	float finalStates[AGP_MAX_FULL_DIM]) noexcept
{
	RunParams p;
	p.nSegments = static_cast<unsigned>(n);
	p.varLen = static_cast<unsigned>(variableLengths);
	p.maxTheta = maxTheta;
	p.tx = tx;
	p.ty = ty;
	p.ta = ta;
	p.maxIter = maxIter;
	p.r = r_param;
	p.adaptive = static_cast<unsigned>(adaptive);
	p.eps = eps;
	p.seed = seed;
	p.baseLength = baseLength;
	p.stretchFactor = stretchFactor;
	if (obstacleData) memcpy(p.obstacleData, obstacleData, static_cast<size_t>(obstacleCount) * 3ull * sizeof(float));
	p.obstacleCount = obstacleCount;
	p.mode = mode;
	const size_t total = static_cast<size_t>(n + (variableLengths ? n : 0));
	if (referenceStates) memcpy(p.referenceStates, referenceStates, total * sizeof(float));
	if (finalStates) memcpy(p.finalStates, finalStates, total * sizeof(float));
	const int world = g_world->size();
	int rank = 1;
	while (rank < world)
	{
		g_world->send(rank, 1, p);
		++rank;
	}
}

template<bool HasObstacles, bool VariableLen>
static void build_trajectory(
	int n, bool variableLengths, float maxTheta,
	const float* startState,
	const float* finalState,
	int maxIter, float r_param, bool adaptive, float eps, unsigned int seed,
	float baseLength, float stretchFactor,
	const float* obstacleData, int obstacleCount,
	float** outPoints, int* outPointCount,
	size_t* outTotalIterations) noexcept
{
	std::atomic_size_t totalIterationsAtomic{ 0 };
	std::atomic<float> totalEnergyAtomic{ 0.0f };

	const int rank = g_world->rank();
	const int world = g_world->size();
	const int total = n << 1;
	const int currentGen = g_trajectoryCallCounter.fetch_add(1, std::memory_order_relaxed);
	thread_local int generation = 1;

	std::vector<float, boost::alignment::aligned_allocator<float, 32u>> startStateVec(total);
	std::vector<float, boost::alignment::aligned_allocator<float, 32u>> finalStateVec(total);
	memcpy(startStateVec.data(), startState, total * sizeof(float));
	memcpy(finalStateVec.data(), finalState, total * sizeof(float));

	alignas(ManipCost) thread_local char _tempCost[sizeof(ManipCost)];
	thread_local const float* lastRef = nullptr;

	if (currentGen != generation)
	{
		generation = currentGen;
		lastRef = nullptr;
		new (_tempCost) ManipCost(n, variableLengths, 0.0f, 0.0f, 0.0f, maxTheta, baseLength, stretchFactor,
			const_cast<float*>(obstacleData), obstacleCount, 2);
	}

	__declspec(align(64)) struct TrajConfig
	{
		std::vector<float, boost::alignment::aligned_allocator<float, 32u>> state;
		float x, y, a, clearance;
		float cost;
	};

	TrajConfig endConfig;
	ManipCost& tempCost = *reinterpret_cast<ManipCost*>(_tempCost);
	endConfig.state = std::move(finalStateVec);
	tempCost.compute_pose(endConfig.state.data(), endConfig.x, endConfig.y, endConfig.a, nullptr, nullptr, nullptr, nullptr);
	if constexpr (HasObstacles)
	{
		float px[AGP_MAX_LINK_POINTS], py[AGP_MAX_LINK_POINTS];
		unsigned dummyIdx; float dummyVal;
		tempCost.evaluate_pose_constraints_from_arrays(
			px, py, 0u, dummyIdx, dummyVal, &endConfig.clearance);
	}
	else {
		endConfig.clearance = FLT_MAX;
	}
	std::vector<TrajConfig, boost::alignment::aligned_allocator<TrajConfig, 64u>> result;
	result.reserve(512);
	result.emplace_back(TrajConfig{ std::move(startStateVec) });
	result[0].clearance = FLT_MAX;

	auto build = [&](auto&& self, const TrajConfig& a, const TrajConfig& b, int depth, std::vector<TrajConfig, boost::alignment::aligned_allocator<TrajConfig, 64u>>& out) -> void
		{
			if (depth >= 8)
			{
				out.emplace_back(b);
				return;
			}
			if (currentGen != generation)
			{
				generation = currentGen;
				lastRef = nullptr;
				new (&tempCost) ManipCost(n, variableLengths, 0.0f, 0.0f, 0.0f, maxTheta, baseLength, stretchFactor,
					const_cast<float*>(obstacleData), obstacleCount, 2);
			}
			if (lastRef != a.state.data())
			{
				tempCost.SetTransitionReference(a.state.data());
				lastRef = a.state.data();
			}
			unsigned out_idx; float out_val;
			if (tempCost.evaluate_transition_swept_motion<HasObstacles, VariableLen>(b.state.data(), b.x, b.y, b.a, b.clearance, out_idx, out_val))
			{
				totalEnergyAtomic.fetch_add(b.cost, std::memory_order_relaxed);
				out.emplace_back(b);
				return;
			}

			TrajConfig mid;
			mid.state.resize(total);

			for (int i = 0; i < n; ++i)
			{
				const float diff = ManipCost::wrap_pi(b.state[i] - a.state[i]);
				mid.state[i] = fmaf(diff, 0.5f, a.state[i]);
				mid.state[n + i] = variableLengths ? fmaf(a.state[n + i], 0.5f, fmaf(b.state[n + i], 0.5f, 0.0f)) : baseLength;
			}
			tempCost.compute_pose(mid.state.data(), mid.x, mid.y, mid.a, nullptr, nullptr, nullptr, nullptr);

			float* midBestQ;
			float midBestX, midBestY, midBestA, midBestF;
			size_t midActualIterations;
			float midAchievedEps;

			AGP_Manip2D(n, variableLengths, maxTheta, mid.x, mid.y, mid.a,
				std::max(maxIter >> depth, 100), r_param, adaptive,
				eps, seed,
				baseLength, stretchFactor,
				const_cast<float*>(obstacleData), obstacleCount,
				&midBestQ, &midBestX, &midBestY, &midBestA, &midBestF, &midActualIterations, &midAchievedEps,
				2, a.state.data());
			totalIterationsAtomic.fetch_add(midActualIterations, std::memory_order_relaxed);

			TrajConfig midOpt;
			midOpt.state.resize(total);
			memcpy(midOpt.state.data(), midBestQ, total * sizeof(float));
			midOpt.x = midBestX;
			midOpt.y = midBestY;
			midOpt.a = midBestA;
			midOpt.cost = midBestF;
			CoTaskMemFree(midBestQ);

			if constexpr (HasObstacles)
			{
				float px[AGP_MAX_LINK_POINTS], py[AGP_MAX_LINK_POINTS];
				unsigned dummyIdx; float dummyVal;
				tempCost.evaluate_pose_constraints_from_arrays(
					px, py, 0u, dummyIdx, dummyVal, &midOpt.clearance);
			}
			else {
				midOpt.clearance = FLT_MAX;
			}

			std::vector<TrajConfig, boost::alignment::aligned_allocator<TrajConfig, 64u>> left, right;
			left.reserve(AGP_MAX_FULL_DIM);
			right.reserve(AGP_MAX_FULL_DIM);

			if (depth < 5)
			{
				oneapi::tbb::parallel_invoke(
					[&self, &a, &midOpt, depth, &left]() { self(self, a, midOpt, depth + 1, left); },
					[&self, &midOpt, &b, depth, &right]() { self(self, midOpt, b, depth + 1, right); }
				);
			}
			else
			{
				self(self, a, midOpt, depth + 1, left);
				self(self, midOpt, b, depth + 1, right);
			}

			out.reserve(out.size() + left.size() + right.size());
			out.insert(out.end(),
				std::make_move_iterator(left.begin()),
				std::make_move_iterator(left.end()));
			out.insert(out.end(),
				std::make_move_iterator(right.begin()),
				std::make_move_iterator(right.end()));
		};

	build(build, result[0], endConfig, 0, result);

	if (world > 1)
	{
		std::vector<char> sendBuffer, recvBuffer;
		const int maxPoints = 512;
		const size_t headerSize = sizeof(float) + (sizeof(int) << 1);
		const size_t maxBufferSize = headerSize + maxPoints * total * sizeof(float);
		if (sendBuffer.size() < maxBufferSize) sendBuffer.resize(maxBufferSize);
		if (recvBuffer.size() < maxBufferSize) recvBuffer.resize(maxBufferSize);

		auto packTrajectory = [&](std::vector<char>& buffer, const std::vector<TrajConfig, boost::alignment::aligned_allocator<TrajConfig, 64u>>& traj, float energy)
			{
				const int pointCount = (int)traj.size();
				const int dimState = total;
				char* ptr = buffer.data();
				memcpy(ptr, &energy, sizeof(float));
				ptr += sizeof(float);
				memcpy(ptr, &pointCount, sizeof(int));
				ptr += sizeof(int);
				memcpy(ptr, &dimState, sizeof(int));
				ptr += sizeof(int);
				for (const auto& cfg : traj)
				{
					memcpy(ptr, cfg.state.data(), dimState * sizeof(float));
					ptr += dimState * sizeof(float);
				}
			};

		auto unpackTrajectory = [&](const std::vector<char>& buffer, std::vector<TrajConfig, boost::alignment::aligned_allocator<TrajConfig, 64u>>& traj, float& energy)
			{
				const char* ptr = buffer.data();
				memcpy(&energy, ptr, sizeof(float));
				ptr += sizeof(float);
				int pointCount = 0;
				memcpy(&pointCount, ptr, sizeof(int));
				ptr += sizeof(int);
				int dimState = 0;
				memcpy(&dimState, ptr, sizeof(int));
				ptr += sizeof(int);
				traj.clear();
				traj.reserve(pointCount);
				for (int i = 0; i < pointCount; ++i)
				{
					TrajConfig cfg;
					cfg.state = std::vector<float, boost::alignment::aligned_allocator<float, 32u>>(
						reinterpret_cast<const float*>(ptr),
						reinterpret_cast<const float*>(ptr) + dimState
					);
					ptr += dimState * sizeof(float);
					traj.emplace_back(std::move(cfg));
				}
			};

		float localEnergy = totalEnergyAtomic.load(std::memory_order_relaxed);
		packTrajectory(sendBuffer, result, localEnergy);

		const size_t commLevels = std::bit_width(static_cast<size_t>(world - 1));
		for (size_t i = 0; i < commLevels; ++i)
		{
			const size_t step = 1ULL << i;
			const int partner = rank ^ static_cast<int>(step);
			if (partner < world)
			{
				if (static_cast<bool>(rank & static_cast<int>(step)))
				{
					g_world->send(partner, 5, sendBuffer.data(), static_cast<int>(maxBufferSize));
					return;
				}
				else
				{
					g_world->recv(partner, 5, recvBuffer.data(), static_cast<int>(maxBufferSize));
					float recvEnergy;
					std::vector<TrajConfig, boost::alignment::aligned_allocator<TrajConfig, 64u>> recvTraj;
					unpackTrajectory(recvBuffer, recvTraj, recvEnergy);
					if (recvEnergy < localEnergy)
					{
						localEnergy = recvEnergy;
						result.swap(recvTraj);
						packTrajectory(sendBuffer, result, localEnergy);
					}
				}
			}
		}
		totalEnergyAtomic.store(localEnergy, std::memory_order_relaxed);
	}

	const size_t pointCount = result.size();
	const SIZE_T bytes = static_cast<SIZE_T>(total) * sizeof(float);
	*outPointCount = pointCount;
	*outTotalIterations = totalIterationsAtomic.load(std::memory_order_relaxed);
	*outPoints = static_cast<float*>(CoTaskMemAlloc(pointCount * bytes));
	float* ptr = *outPoints;
	for (const auto& cfg : result)
	{
		memcpy(ptr, cfg.state.data(), bytes);
		ptr += total;
	}
}

extern "C" __declspec(dllexport) void __cdecl AGP_BuildTransitionTrajectory(
	int n, bool variableLengths, float maxTheta,
	const float* startState,
	const float* finalState,
	int maxIter, float r_param, bool adaptive, float eps, unsigned int seed,
	float baseLength, float stretchFactor,
	const float* obstacleData, int obstacleCount,
	float** outPoints, int* outPointCount,
	size_t* outTotalIterations) noexcept
{
	if (static_cast<bool>(obstacleCount))
	{
		if (variableLengths)
			build_trajectory<true, true>(n, variableLengths, maxTheta, startState, finalState,
				maxIter, r_param, adaptive, eps, seed, baseLength, stretchFactor,
				obstacleData, obstacleCount, outPoints, outPointCount, outTotalIterations);
		else
			build_trajectory<true, false>(n, variableLengths, maxTheta, startState, finalState,
				maxIter, r_param, adaptive, eps, seed, baseLength, stretchFactor,
				obstacleData, obstacleCount, outPoints, outPointCount, outTotalIterations);
	}
	else
	{
		if (variableLengths)
			build_trajectory<false, true>(n, variableLengths, maxTheta, startState, finalState,
				maxIter, r_param, adaptive, eps, seed, baseLength, stretchFactor,
				obstacleData, obstacleCount, outPoints, outPointCount, outTotalIterations);
		else
			build_trajectory<false, false>(n, variableLengths, maxTheta, startState, finalState,
				maxIter, r_param, adaptive, eps, seed, baseLength, stretchFactor,
				obstacleData, obstacleCount, outPoints, outPointCount, outTotalIterations);
	}
}

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline
void AgpWaitStartAndRun() noexcept
{
	while (true)
	{
		if (g_world->iprobe(0, 1))
		{
			RunParams p;
			g_world->recv(0, 1, p);
			const int n = static_cast<int>(p.nSegments);
			const bool varLen = static_cast<bool>(p.varLen);

			if (p.mode & 2)
			{
				float* outPoints = nullptr;
				int pointCount = 0;
				size_t totalIterations = 0;
				AGP_BuildTransitionTrajectory(
					n, varLen, p.maxTheta,
					p.referenceStates, p.finalStates,
					p.maxIter, p.r, static_cast<bool>(p.adaptive), p.eps, p.seed,
					p.baseLength, p.stretchFactor,
					p.obstacleData, p.obstacleCount,
					&outPoints, &pointCount, &totalIterations);
			}
			else
			{
				float* q = nullptr;
				size_t qlen = 0;
				float bx, by, ba, bf;
				size_t oi;
				float oa;
				AGP_Manip2D(
					n, varLen, p.maxTheta, p.tx, p.ty, p.ta,
					p.maxIter, p.r, static_cast<bool>(p.adaptive), p.eps, p.seed,
					p.baseLength, p.stretchFactor,
					p.obstacleData, p.obstacleCount,
					&q, &bx, &by, &ba, &bf, &oi, &oa,
					p.mode, p.referenceStates);
				CoTaskMemFree(q);
			}
		}
		Sleep(0);
	}
}
