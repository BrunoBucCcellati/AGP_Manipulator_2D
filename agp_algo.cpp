#include "pch.h"

#define XOR_RAND(state, result_var) \
    do { \
        unsigned s = (state); \
        s ^= s << 13; \
        s ^= s >> 17; \
        s ^= s << 5; \
        (state) = s; \
        float tmp = (float)((double)(s) * (1.0/4294967296.0)); \
        result_var = tmp; \
    } while (0)

#define XOR_RAND_GRSH(state, result_var) \
    do { \
        unsigned s = (state); \
        s ^= s << 13; \
        s ^= s >> 17; \
        s ^= s << 5; \
        (state) = s; \
        result_var = fmaf((float)(int)s, 0x1.0p-31f, -1.0f); \
    } while (0)

#define FABE13_COS(x, result_var) \
    do { \
        const float _ax_ = fabsf(x); \
        float _r_ = fmodf(_ax_, 6.28318530718f); \
        if (_r_ > 3.14159265359f) \
            _r_ = 6.28318530718f - _r_; \
        if (_r_ < 1.57079632679f) { \
            const float _t2_ = _r_ * _r_; \
            const float _t4_ = _t2_ * _t2_; \
            result_var = fmaf(_t4_, fmaf(_t2_, -0.0013888889f, 0.0416666667f), fmaf(_t2_, -0.5f, 1.0f)); \
        } else { \
            _r_ = 3.14159265359f - _r_; \
            const float _t2_ = _r_ * _r_; \
            const float _t4_ = _t2_ * _t2_; \
            result_var = -fmaf(_t4_, fmaf(_t2_, -0.0013888889f, 0.0416666667f), fmaf(_t2_, -0.5f, 1.0f)); \
        } \
    } while (0)

#define FABE13_SIN(x, result_var) \
    do { \
        const float _x_ = (x); \
        const float _ax_ = fabsf(_x_); \
        float _r_ = fmodf(_ax_, 6.28318530718f); \
        bool _sfl_ = _r_ > 3.14159265359f; \
        if (_sfl_) \
            _r_ = 6.28318530718f - _r_; \
        bool _cfl_ = _r_ > 1.57079632679f; \
        if (_cfl_) \
            _r_ = 3.14159265359f - _r_; \
        const float _t2_ = _r_ * _r_; \
        float _s = fmaf(_t2_, fmaf(_t2_, fmaf(_t2_, -0.0001984127f, 0.0083333333f), -0.16666666f), 1.0f) * _r_; \
        result_var = ((_x_ < 0.0f) ^ _sfl_) ? -_s : _s; \
    } while (0)

#define FABE13_SINCOS(in, sin_out, cos_out, n) \
    do { \
        int i = 0; \
        const int limit = (n) & ~7; \
        if ((n) >= 8) { \
            static __declspec(align(16)) const __m256 VEC_TWOPI = _mm256_set1_ps(6.28318530718f); \
            static __declspec(align(16)) const __m256 VEC_PI = _mm256_set1_ps(3.14159265359f); \
            static __declspec(align(16)) const __m256 VEC_PI_2 = _mm256_set1_ps(1.57079632679f); \
            static __declspec(align(16)) const __m256 INV_TWOPI = _mm256_set1_ps(0.15915494309189535f); \
            static __declspec(align(16)) const __m256 BIAS = _mm256_set1_ps(12582912.0f); \
            static __declspec(align(16)) const __m256 VEC_COS_P5 = _mm256_set1_ps(-0.0013888889f); \
            static __declspec(align(16)) const __m256 VEC_COS_P3 = _mm256_set1_ps(0.0416666667f); \
            static __declspec(align(16)) const __m256 VEC_COS_P1 = _mm256_set1_ps(-0.5f); \
            static __declspec(align(16)) const __m256 VEC_COS_P0 = _mm256_set1_ps(1.0f); \
            static __declspec(align(16)) const __m256 VEC_SIN_P5 = _mm256_set1_ps(-0.0001984127f); \
            static __declspec(align(16)) const __m256 VEC_SIN_P3 = _mm256_set1_ps(0.0083333333f); \
            static __declspec(align(16)) const __m256 VEC_SIN_P1 = _mm256_set1_ps(-0.16666666f); \
            static __declspec(align(16)) const __m256 VEC_SIN_P0 = _mm256_set1_ps(1.0f); \
            static __declspec(align(16)) const __m256 VEC_ZERO = _mm256_setzero_ps(); \
            while (i < limit) { \
                const __m256 vx = _mm256_load_ps(&(in)[i]); \
                const __m256 vax = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vx); \
                __m256 q = _mm256_fmadd_ps(vax, INV_TWOPI, BIAS); \
                q = _mm256_sub_ps(q, BIAS); \
                const __m256 r = _mm256_fnmadd_ps(VEC_TWOPI, q, vax); \
                const __m256 r1 = _mm256_min_ps(r, _mm256_sub_ps(VEC_TWOPI, r)); \
                const __m256 r2 = _mm256_min_ps(r1, _mm256_sub_ps(VEC_PI, r1)); \
                const __m256 t2 = _mm256_mul_ps(r2, r2); \
                const __m256 cosv = _mm256_fmadd_ps(t2, _mm256_fmadd_ps(t2, _mm256_fmadd_ps(t2, VEC_COS_P5, VEC_COS_P3), VEC_COS_P1), VEC_COS_P0); \
                const __m256 sinv = _mm256_mul_ps(_mm256_fmadd_ps(t2, _mm256_fmadd_ps(t2, _mm256_fmadd_ps(t2, VEC_SIN_P5, VEC_SIN_P3), VEC_SIN_P1), VEC_SIN_P0), r2); \
                const __m256 cflip = _mm256_cmp_ps(r1, VEC_PI_2, _CMP_GT_OQ); \
                const __m256 sflip = _mm256_xor_ps(_mm256_cmp_ps(vx, VEC_ZERO, _CMP_LT_OQ), _mm256_cmp_ps(r, VEC_PI, _CMP_GT_OQ)); \
                _mm256_store_ps(&(cos_out)[i], _mm256_blendv_ps(cosv, _mm256_sub_ps(VEC_ZERO, cosv), cflip)); \
                _mm256_store_ps(&(sin_out)[i], _mm256_blendv_ps(sinv, _mm256_sub_ps(VEC_ZERO, sinv), sflip)); \
                i += 8; \
            } \
        } \
        while (i < (n)) { \
            const float x = (in)[i]; \
            const float ax = fabsf(x); \
            float q = fmaf(ax, 0.15915494309189535f, 12582912.0f); \
            q -= 12582912.0f; \
            float r = fmaf(-6.28318530718f, q, ax); \
            const bool sflip = r > 3.14159265359f; \
            if (sflip) \
                r = 6.28318530718f - r; \
            const bool cflip = r > 1.57079632679f; \
            if (cflip) \
                r = 3.14159265359f - r; \
            const float t2 = r * r; \
            const float c = fmaf(t2, fmaf(t2, fmaf(t2, -0.0013888889f, 0.0416666667f), -0.5f), 1.0f); \
            const float s = fmaf(t2, fmaf(t2, fmaf(t2, -0.0001984127f, 0.0083333333f), -0.16666666f), 1.0f) * r; \
            (cos_out)[i] = cflip ? -c : c; \
            (sin_out)[i] = ((x < 0.0f) ^ sflip) ? -s : s; \
            ++i; \
        } \
    } while (0)

enum List : unsigned {
	Top = 0b00u,
	Down = 0b01u,
	Left = 0b10u,
	Right = 0b11u
};

__declspec(align(16)) struct Step final {
	const unsigned next, dx, dy;
	__forceinline Step(const unsigned n, const unsigned x, const unsigned y) noexcept : next(n), dx(x), dy(y) {}
};

__declspec(align(16)) struct InvStep final {
	const unsigned q, next;
	__forceinline InvStep(const unsigned q_val, const unsigned n) noexcept : q(q_val), next(n) {}
};

__declspec(align(16)) static const Step g_step_tbl[4][4] = {
		{ Step(Right,0u,0u), Step(Top,0u,1u), Step(Top,1u,1u), Step(Left,1u,0u) },
		{ Step(Left,1u,1u), Step(Down,1u,0u), Step(Down,0u,0u), Step(Right,0u,1u) },
		{ Step(Down,1u,1u), Step(Left,0u,1u), Step(Left,0u,0u), Step(Top,1u,0u) },
		{ Step(Top,0u,0u), Step(Right,1u,0u), Step(Right,1u,1u), Step(Down,0u,1u) }
};

__declspec(align(16)) static const InvStep g_inv_tbl[4][4] = {
		{ InvStep(0u,Right), InvStep(1u,Top), InvStep(3u,Left), InvStep(2u,Top) },
		{ InvStep(2u,Down), InvStep(3u,Right), InvStep(1u,Down), InvStep(0u,Left) },
		{ InvStep(2u,Left), InvStep(1u,Left), InvStep(3u,Top), InvStep(0u,Down) },
		{ InvStep(0u,Top), InvStep(3u,Down), InvStep(1u,Right), InvStep(2u,Right) }
};

static const boost::mpi::environment* g_env;
static const boost::mpi::communicator* g_world;
static const pybind11::scoped_interpreter* g_pyInterpreter;
static const pybind11::module_* g_pyOptimizerBridge;
static std::string g_exeDirCache;

__declspec(align(16)) struct CrossMsg final {
	float s_x1, s_x2, e_x1, e_x2, Rtop;
	template<typename Archive> __declspec(noalias) __forceinline void serialize(Archive& ar, const unsigned int) noexcept {
		ar& s_x1& s_x2& e_x1& e_x2& Rtop;
	}
};

__declspec(align(16)) struct MultiCrossMsg final {
	float intervals[15];
	unsigned count;
	template<typename Archive> __declspec(noalias) __forceinline void serialize(Archive& ar, const unsigned int) noexcept {
		ar& intervals& count;
	}
};

__declspec(align(16)) struct BestSolutionMsg final {
	float bestF, bestX, bestY, bestQ[32];
	unsigned dim;
	template<typename Archive> __declspec(noalias) __forceinline void serialize(Archive& ar, const unsigned int) noexcept {
		ar& bestF& bestX& bestY& bestQ& dim;
	}
};

__declspec(align(16)) struct Slab final sealed{
		char* const base;
		char* current;
		char* const end;
		__forceinline Slab(void* const memory, const size_t usable) noexcept :
				base((char*)memory), current(base), end(base + (usable & ~(size_t)63u)) {
		}
};

static thread_local tbb::enumerable_thread_specific<Slab*> tls([]() noexcept {
	void* memory = _aligned_malloc(16777216u, 16u);
	Slab* slab = (Slab*)_aligned_malloc(sizeof(Slab), 16u);
	new (slab) Slab(memory, 16777216u);
	char* p = slab->base;
#pragma loop ivdep
	while (p < slab->end) {
		*p = 0;
		p += 4096u;
	}
	return slab;
	});

__declspec(align(16)) struct Peano2DMap final sealed{
		const int levels;
		const float a, b, c, d;
		const float lenx, leny;
		const float inv_lenx;
		const unsigned scale;
		const unsigned start;

		__forceinline Peano2DMap(const int L, const float _a, const float _b, const float _c, const float _d, const unsigned st) noexcept
				: levels(L), a(_a), b(_b), c(_c), d(_d),
				lenx(_b - _a), leny(_d - _c), inv_lenx(1.0f / (_b - _a)),
				scale((unsigned)1u << (L << 1)), start(st) {
		}
};

static Peano2DMap gActiveMap(0, 0, 0, 0, 0, 0);

__declspec(align(16)) struct Interval1D final sealed{
		const float x1, x2, y1, y2, delta_y, ordinate_factor, N_factor, quadratic_term, M;
		float R;

		static __declspec(noalias) __forceinline void* operator new(const size_t) noexcept {
				Slab* s = tls.local();
				char* r = s->current;
				s->current += 64u;
				return r;
		}

		__declspec(noalias) __forceinline Interval1D(const float _x1, const float _x2, const float _y1, const float _y2, const float _N) noexcept
				: x1(_x1), x2(_x2), y1(_y1), y2(_y2), delta_y(_y2 - _y1), ordinate_factor(-(y1 + y2) * 2.0f),
				N_factor(_N == 1.0f ? _x2 - _x1 : sqrtf(_x2 - _x1)),
				quadratic_term(fmaf((1.0f / N_factor)* delta_y, delta_y, 0.0f)),
				M(fabsf(delta_y) / N_factor) {
		}

		__declspec(noalias) __forceinline void ChangeCharacteristic(const float _m) noexcept {
				const float inv_m = 1.0f / _m;
				R = fmaf(inv_m, quadratic_term, fmaf(_m, N_factor, ordinate_factor));
		}
};

static __declspec(noalias) __forceinline bool ComparePtr1D(const Interval1D* const a, const Interval1D* const b) noexcept {
	return a->R < b->R;
}

static __declspec(noalias) __forceinline void RecomputeR_ConstM_AVX2_1D(Interval1D* const* const arr, const size_t n, const float m) noexcept {
	const __m256 vm = _mm256_set1_ps(m);
	__m256 vinvm = _mm256_rcp_ps(vm);
	vinvm = _mm256_mul_ps(vinvm, _mm256_fnmadd_ps(vm, vinvm, _mm256_set1_ps(2.0f)));
	size_t i = 0, limit = n & ~7ull;
	alignas(16) float q[8], nf[8], od[8], out[8];
#pragma loop ivdep
	while (i < limit) {
		int k = 0;
#pragma loop ivdep
		while (k < 8) {
			const Interval1D* const p = arr[i + k];
			q[k] = p->quadratic_term;
			nf[k] = p->N_factor;
			od[k] = p->ordinate_factor;
			++k;
		}
		const __m256 vq = _mm256_load_ps(q), vnf = _mm256_load_ps(nf), vod = _mm256_load_ps(od);
		const __m256 t = _mm256_fmadd_ps(vm, vnf, vod);
		const __m256 res = _mm256_fmadd_ps(vq, vinvm, t);
		_mm256_store_ps(out, res);
		k = 0;
#pragma loop ivdep
		while (k < 8) {
			arr[i + k]->R = out[k];
			++k;
		}
		i += 8;
	}
	while (i < n) {
		arr[i]->ChangeCharacteristic(m);
		++i;
	}
}

static __declspec(noalias) __forceinline void RecomputeR_AffineM_AVX2_1D(Interval1D* const* const arr, const size_t n, const float GF, const float alpha) noexcept {
	const __m256 vGF = _mm256_set1_ps(GF), va = _mm256_set1_ps(alpha);
	size_t i = 0, limit = n & ~7ull;
	alignas(16) float ln[8], Mv[8], q[8], nf[8], od[8], out[8];
#pragma loop ivdep
	while (i < limit) {
		int k = 0;
#pragma loop ivdep
		while (k < 8) {
			const Interval1D* const p = arr[i + k];
			ln[k] = p->x2 - p->x1;
			Mv[k] = p->M;
			q[k] = p->quadratic_term;
			nf[k] = p->N_factor;
			od[k] = p->ordinate_factor;
			++k;
		}
		const __m256 vln = _mm256_load_ps(ln), vM = _mm256_load_ps(Mv), vq = _mm256_load_ps(q), vnf = _mm256_load_ps(nf), vod = _mm256_load_ps(od);
		const __m256 vm = _mm256_fmadd_ps(vGF, vln, _mm256_mul_ps(va, vM));
		__m256 vinvm = _mm256_rcp_ps(vm);
		vinvm = _mm256_mul_ps(vinvm, _mm256_fnmadd_ps(vm, vinvm, _mm256_set1_ps(2.0f)));
		const __m256 t = _mm256_fmadd_ps(vm, vnf, vod);
		const __m256 res = _mm256_fmadd_ps(vq, vinvm, t);
		_mm256_store_ps(out, res);
		k = 0;
#pragma loop ivdep
		while (k < 8) {
			arr[i + k]->R = out[k];
			++k;
		}
		i += 8;
	}
	while (i < n) {
		const Interval1D* const p = arr[i];
		const float mi = fmaf(GF, (p->x2 - p->x1), p->M * alpha);
		arr[i]->R = fmaf((1.0f / mi) * p->quadratic_term, 1.0f, fmaf(mi, p->N_factor, p->ordinate_factor));
		++i;
	}
}

__declspec(align(16)) struct IntervalND final sealed{
		const float x1, x2, y1, y2, delta_y, ordinate_factor;
		float N_factor, quadratic_term, M, R;
		unsigned long long i1, i2;
		float diam;
		int span_level;

		static __declspec(noalias) __forceinline void* operator new(const size_t) noexcept {
				Slab* s = tls.local();
				char* r = s->current;
				s->current += 64u;
				return r;
		}

		__declspec(noalias) __forceinline IntervalND(const float _x1, const float _x2, const float _y1, const float _y2) noexcept
				: x1(_x1), x2(_x2), y1(_y1), y2(_y2), delta_y(_y2 - _y1), ordinate_factor(-(y1 + y2) * 2.0f),
				N_factor(0), quadratic_term(0), M(0), R(0), i1(0), i2(0), diam(0), span_level(0) {
		}

		__declspec(noalias) __forceinline void compute_span_level(const struct MortonND& map) noexcept;
		__declspec(noalias) __forceinline void set_metric(const float d_alpha) noexcept {
				N_factor = d_alpha;
				quadratic_term = (1.0f / N_factor) * delta_y * delta_y;
				M = fabsf(delta_y) / N_factor;
		}

		__declspec(noalias) __forceinline void ChangeCharacteristic(const float _m) noexcept {
				const float inv_m = 1.0f / _m;
				R = fmaf(inv_m, quadratic_term, fmaf(_m, N_factor, ordinate_factor));
		}
};

static __declspec(noalias) __forceinline bool ComparePtrND(const IntervalND* const a, const IntervalND* const b) noexcept {
	return a->R < b->R;
}

static __declspec(noalias) __forceinline void RecomputeR_ConstM_AVX2_ND(IntervalND* const* const arr, const size_t n, const float m) noexcept {
	const __m256 vm = _mm256_set1_ps(m);
	__m256 vinvm = _mm256_rcp_ps(vm);
	vinvm = _mm256_mul_ps(vinvm, _mm256_fnmadd_ps(vm, vinvm, _mm256_set1_ps(2.0f)));
	size_t i = 0, limit = n & ~7ull;
	alignas(16) float q[8], nf[8], od[8], out[8];
#pragma loop ivdep
	while (i < limit) {
		int k = 0;
#pragma loop ivdep
		while (k < 8) {
			const IntervalND* const p = arr[i + k];
			q[k] = p->quadratic_term;
			nf[k] = p->N_factor;
			od[k] = p->ordinate_factor;
			++k;
		}
		const __m256 vq = _mm256_load_ps(q), vnf = _mm256_load_ps(nf), vod = _mm256_load_ps(od);
		const __m256 t = _mm256_fmadd_ps(vm, vnf, vod);
		const __m256 res = _mm256_fmadd_ps(vq, vinvm, t);
		_mm256_store_ps(out, res);
		k = 0;
#pragma loop ivdep
		while (k < 8) {
			arr[i + k]->R = out[k];
			++k;
		}
		i += 8;
	}
	while (i < n) {
		arr[i]->ChangeCharacteristic(m);
		++i;
	}
}

static __declspec(noalias) __forceinline void RecomputeR_AffineM_AVX2_ND(IntervalND* const* const arr, const size_t n, const float GF, const float alpha) noexcept {
	const __m256 vGF = _mm256_set1_ps(GF), va = _mm256_set1_ps(alpha);
	size_t i = 0, limit = n & ~7ull;
	alignas(16) float ln[8], Mv[8], q[8], nf[8], od[8], out[8];
#pragma loop ivdep
	while (i < limit) {
		int k = 0;
#pragma loop ivdep
		while (k < 8) {
			const IntervalND* const p = arr[i + k];
			ln[k] = p->x2 - p->x1;
			Mv[k] = p->M;
			q[k] = p->quadratic_term;
			nf[k] = p->N_factor;
			od[k] = p->ordinate_factor;
			++k;
		}
		const __m256 vln = _mm256_load_ps(ln), vM = _mm256_load_ps(Mv), vq = _mm256_load_ps(q), vnf = _mm256_load_ps(nf), vod = _mm256_load_ps(od);
		const __m256 vm = _mm256_fmadd_ps(vGF, vln, _mm256_mul_ps(va, vM));
		__m256 vinvm = _mm256_rcp_ps(vm);
		vinvm = _mm256_mul_ps(vinvm, _mm256_fnmadd_ps(vm, vinvm, _mm256_set1_ps(2.0f)));
		const __m256 t = _mm256_fmadd_ps(vm, vnf, vod);
		const __m256 res = _mm256_fmadd_ps(vq, vinvm, t);
		_mm256_store_ps(out, res);
		k = 0;
#pragma loop ivdep
		while (k < 8) {
			arr[i + k]->R = out[k];
			++k;
		}
		i += 8;
	}
	while (i < n) {
		const IntervalND* const p = arr[i];
		const float mi = fmaf(GF, (p->x2 - p->x1), p->M * alpha);
		arr[i]->R = fmaf((1.0f / mi) * p->quadratic_term, 1.0f, fmaf(mi, p->N_factor, p->ordinate_factor));
		++i;
	}
}

static __declspec(noalias) __forceinline float fast_pow_int(const float v, const int n) noexcept {
	float r;
	switch (n) {
	case 3: {
		const float v2 = v * v;
		r = v2 * v;
	} break;
	case 4: {
		const float v2 = v * v;
		r = v2 * v2;
	} break;
	case 5: {
		const float v2 = v * v;
		r = v2 * v2 * v;
	} break;
	case 6: {
		const float v2 = v * v;
		const float v4 = v2 * v2;
		r = v4 * v2;
	} break;
	case 7: {
		const float v2 = v * v;
		const float v4 = v2 * v2;
		r = v4 * v2 * v;
	} break;
	case 8: {
		const float v2 = v * v;
		const float v4 = v2 * v2;
		r = v4 * v4;
	} break;
	case 9: {
		const float v3 = v * v * v;
		const float v6 = v3 * v3;
		r = v6 * v3;
	} break;
	case 10: {
		const float v2 = v * v;
		const float v4 = v2 * v2;
		const float v8 = v4 * v4;
		r = v8 * v2;
	} break;
	case 11: {
		const float v2 = v * v;
		const float v4 = v2 * v2;
		const float v8 = v4 * v4;
		r = v8 * v2 * v;
	} break;
	case 12: {
		const float v3 = v * v * v;
		const float v6 = v3 * v3;
		r = v6 * v6;
	} break;
	case 13: {
		const float v3 = v * v * v;
		const float v6 = v3 * v3;
		r = v6 * v6 * v;
	} break;
	case 14: {
		const float v7 = v * v * v * v * v * v * v;
		r = v7 * v7;
	} break;
	case 15: {
		const float v7 = v * v * v * v * v * v * v;
		r = v7 * v7 * v;
	} break;
	default: {
		const float v2 = v * v;
		const float v4 = v2 * v2;
		const float v8 = v4 * v4;
		r = v8 * v8;
	}
	}
	return r;
}

static __declspec(noalias) __forceinline float step(const float _m, const float x1, const float x2, const float y1, const float y2, const float _N, const float _r) noexcept {
	const float diff = y2 - y1;
	const unsigned sign_mask = ((*reinterpret_cast<const unsigned*>(&diff)) & 0x80000000u) ^ 0x80000000u;
	const float sign_mult = *reinterpret_cast<const float*>(&sign_mask);
	if (_N == 1.0f)
		return fmaf(-(1.0f / _m), diff, x1 + x2) * 0.5f;
	if (_N == 2.0f)
		return fmaf((1.0f / (_m * _m)) * sign_mult * diff * diff * _r, 1.0f, x1 + x2) * 0.5f;
	return fmaf((1.0f / fast_pow_int(_m, _N)) * sign_mult * fast_pow_int(fabsf(diff), _N) * _r, 1.0f, x1 + x2) * 0.5f;
}

__declspec(align(16)) struct MortonCachePerRank final sealed{
		std::vector<int> permCache;
		std::vector<unsigned long long> invMaskCache;
		unsigned baseSeed;
};

static thread_local MortonCachePerRank g_mc;

static __declspec(noalias) __forceinline unsigned long long gray_encode(const unsigned long long x) noexcept {
	return x ^ (x >> 1);
}

static __declspec(noalias) __forceinline long long gray_decode(unsigned long long g) noexcept {
	g ^= g >> 32;
	g ^= g >> 16;
	g ^= g >> 8;
	g ^= g >> 4;
	g ^= g >> 2;
	g ^= g >> 1;
	return g;
}

__declspec(align(16)) struct MortonND final sealed{
		const int dim, levels;
		const int eff_levels;
		const int extra_levels;
		const int chunks;
		std::vector<int> chunk_bits;
		std::vector<unsigned long long> chunk_bases;
		unsigned long long scale;
		std::vector<float> low, high, step, invStep, baseOff;
		std::vector<int> perm;
		std::vector<unsigned long long> invMask;
		std::vector<unsigned long long> pextMask;
		std::vector<unsigned long long> pextMaskChunks;
		const float invScaleLevel;
		const bool use_gray;

		static __declspec(noalias) __forceinline unsigned long long make_mask(const int dim, const int Lc, const int d) noexcept {
				unsigned long long m = 0ull, bitpos = static_cast<unsigned long long>(d);
				int b = 0;
#pragma loop ivdep
				while (b < Lc) {
						m |= 1ull << bitpos;
						bitpos += static_cast<unsigned long long>(dim);
						++b;
				}
				return m;
		}

		__declspec(noalias) __forceinline MortonND(const int D, const int L, const float* const lows, const float* const highs, const MortonCachePerRank& mc)
				: dim(D), levels(L),
				eff_levels((std::max)(1, static_cast<int>(63 / (D ? D : 1)))),
				extra_levels((L > eff_levels) ? (L - eff_levels) : 0),
				chunks((extra_levels > 0) ? (1 + (extra_levels + eff_levels - 1) / eff_levels) : 1),
				low(lows, lows + D), high(highs, highs + D),
				step(D, 0.0f), invStep(D, 0.0f), baseOff(D, 0.0f),
				perm(mc.permCache.begin(), mc.permCache.begin() + D),
				invMask(mc.invMaskCache.begin(), mc.invMaskCache.begin() + D),
				invScaleLevel(1.0f / static_cast<float>(static_cast<unsigned long long>(1) << L)), use_gray(true) {

				int d = 0;
#pragma loop ivdep
				while (d < dim) {
						const float rng = high[d] - low[d];
						const float st = rng * invScaleLevel;
						step[d] = st;
						invStep[d] = 1.0f / st;
						baseOff[d] = fmaf(0.5f, st, low[d]);
						++d;
				}

				chunk_bits.resize(chunks);
				pextMaskChunks.resize(static_cast<size_t>(chunks) * static_cast<size_t>(dim));
				chunk_bases.resize(chunks);
				int remaining = levels;
				int c = 0;
				while (c < chunks) {
						const int Lc = (c == 0) ? (std::min)(eff_levels, remaining) : (std::min)(eff_levels, remaining);
						chunk_bits[c] = Lc;
						remaining -= Lc;
						const unsigned long long baseC = static_cast<unsigned long long>(1) << (dim * Lc);
						chunk_bases[c] = baseC;
						d = 0;
#pragma loop ivdep
						while (d < dim) {
								pextMaskChunks[static_cast<size_t>(c) * static_cast<size_t>(dim) + static_cast<size_t>(d)] = make_mask(dim, Lc, d);
								++d;
						}
						++c;
				}

				pextMask.resize(dim);
				d = 0;
#pragma loop ivdep
				while (d < dim) {
						pextMask[d] = make_mask(dim, chunk_bits[0], d);
						++d;
				}

				scale = static_cast<unsigned long long>(1) << (dim * chunk_bits[0]);
		}

		__declspec(noalias) __forceinline float block_diameter(unsigned long long i1, unsigned long long i2) const noexcept {
				if (i1 > i2) std::swap(i1, i2);
				float s2 = 0.0f;
				int d = 0;
#pragma loop ivdep
				while (d < dim) {
						const int pd = perm[d];
						const unsigned long long varying = (i1 ^ i2) & pextMask[d];
						const int nfree_hi = _mm_popcnt_u64(varying);
						const int nfree_total = nfree_hi + (levels - chunk_bits[0]);
						const float range = step[pd] * (ldexpf(1.0f, nfree_total) - 1.0f);
						s2 = fmaf(range, range, s2);
						++d;
				}
				return sqrtf(s2);
		}

		__declspec(noalias) __forceinline void map01ToPoint(const float t, float* const __restrict out) const noexcept {
				unsigned long long accBits[32] = { 0ull };
				int accShifted[32] = { 0 };

				int c = 0;
				while (c < chunks) {
						const int Lc = chunk_bits[c];
						const unsigned long long baseC = chunk_bases[c];
						const float scaled = t * static_cast<float>(baseC);
						unsigned long long idxc = (scaled >= static_cast<float>(baseC)) ? (baseC - 1ull) : static_cast<unsigned long long>(scaled);
						const float u = scaled - static_cast<float>(idxc);
						if (use_gray) idxc = gray_encode(idxc);

						int shift_from_top = 0;
						int k = 0;
						while (k <= c) {
								shift_from_top += chunk_bits[k];
								++k;
						}
						const int inv_shift = levels - shift_from_top;

						int d = 0;
#pragma loop ivdep
						while (d < dim) {
								const int pd = perm[d];
								const unsigned long long mask = pextMaskChunks[static_cast<size_t>(c) * static_cast<size_t>(dim) + static_cast<size_t>(d)];
								unsigned long long bits = _pext_u64(idxc, mask);
								if (inv_shift >= 0) {
										unsigned long long invMaskSegment = 0ull;
										if (chunk_bits[c] < 63) {
												const unsigned long long take = (static_cast<unsigned long long>(1) << chunk_bits[c]) - 1ull;
												invMaskSegment = (invMask[pd] >> inv_shift) & take;
										}
										bits ^= invMaskSegment;
								}
								accBits[pd] = (accBits[pd] << Lc) | bits;
								accShifted[pd] += Lc;
								++d;
						}
						++c;
				}
				int d = 0;
#pragma loop ivdep
				while (d < dim) {
						out[d] = fmaf(step[d], static_cast<float>(accBits[d]), baseOff[d]);
						++d;
				}
		}

		__declspec(noalias) __forceinline float pointToT(const float* const __restrict q) const noexcept {
				const int bitsFull = levels;
				const int bitsCoarse = chunk_bits[0];
				unsigned long long idx0 = 0ull;
				int d = 0;
#pragma loop ivdep
				while (d < dim) {
						const int pd = perm[d];
						const float v = (q[pd] - baseOff[pd]) * invStep[pd];
						const long long cell = static_cast<long long>(_mm_cvt_ss2si(_mm_round_ss(_mm_setzero_ps(), _mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
						const long long maxv = (static_cast<long long>(1) << bitsFull) - 1;
						unsigned long long b = static_cast<unsigned long long>(cell) >> (bitsFull - bitsCoarse);
						unsigned long long invMask0 = 0ull;
						if (bitsCoarse < 63) {
								const unsigned long long take = (static_cast<unsigned long long>(1) << bitsCoarse) - 1ull;
								invMask0 = (invMask[pd] >> (bitsFull - bitsCoarse)) & take;
						}
						b ^= invMask0;
						idx0 |= _pdep_u64(b, pextMask[d]);
						++d;
				}
				if (use_gray) idx0 = gray_decode(idx0);
				return (static_cast<float>(idx0) + 0.5f) / static_cast<float>(scale);
		}
};

__declspec(noalias) __forceinline void IntervalND::compute_span_level(const MortonND& map) noexcept {
	span_level = 0;
	int d = 0;
#pragma loop ivdep
	while (d < map.dim) {
		const unsigned long long varying = (i1 ^ i2) & map.pextMask[d];
		span_level += _mm_popcnt_u64(varying);
		++d;
	}
	span_level += (map.levels - map.chunk_bits[0]) * map.dim;
	span_level = (std::min)(span_level, 11);
}

__declspec(align(16)) struct ManipCost final sealed{
		const int n;
		const bool variableLen;
		const float targetX, targetY;
		const float minTheta;
		const float archBiasW, archBiasK;
		const float sharpW;

		__declspec(noalias) __forceinline ManipCost(const int _n, const bool _variableLen, const float _targetX, const float _targetY, const float _minTheta) noexcept
				: n(_n), variableLen(_variableLen), targetX(_targetX), targetY(_targetY), minTheta(_minTheta),
				archBiasW(0.02f), archBiasK(3.0f), sharpW(0.05f) {
		}

		__declspec(noalias) __forceinline float operator()(const float* const __restrict q, float& out_x, float& out_y) const noexcept {
				const float* const __restrict th = q;
				const float* const __restrict L = variableLen ? (q + n) : nullptr;
				__declspec(align(16)) float phi[32], s_arr[32], c_arr[32];
				float x = 0.0f, y = 0.0f, phi_acc = 0.0f, penC = 0.0f, archPen = 0.0f;

				int i = 0;
#pragma loop ivdep
				while (i < n) {
						phi_acc += th[i];
						phi[i] = phi_acc;
						++i;
				}
				FABE13_SINCOS(phi, s_arr, c_arr, n);

				const float sharpScale = 2.0f / (minTheta + 1.0e-6f);
				const float Lc = 1.0f;
				if (variableLen) {
						i = 0;
						while (i < n) {
								const float Li = L[i];
								x = fmaf(Li, c_arr[i], x);
								y = fmaf(Li, s_arr[i], y);
								++i;
						}
				}
				else {
						i = 0;
						while (i < n) {
								x = fmaf(Lc, c_arr[i], x);
								y = fmaf(Lc, s_arr[i], y);
								++i;
						}
				}

				i = 0;
#pragma loop ivdep
				while (i < n) {
						const float theta = th[i];
						const float ai = fabsf(theta);
						const float v = ai - minTheta;
						if (v > 0.0f) {
								const float scale = sharpScale * v;
								const float arg = scale * 0.69314718055994530941723212145818f;
								const float exp2_val = fmaf(arg, fmaf(arg, fmaf(arg, fmaf(arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
								penC = fmaf(sharpW, exp2_val - 1.0f, penC);
						}
						const float t = -theta * archBiasK;
						float sp;
						if (t > 10.0f) {
								sp = t;
						}
						else {
								const float exp_val = fmaf(t, fmaf(t, fmaf(t, fmaf(t, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
								sp = log1pf(exp_val);
						}
						archPen = fmaf(archBiasW, sp, archPen);
						++i;
				}

				const float dx = x - targetX, dy = y - targetY;
				const float dist = sqrtf(fmaf(dx, dx, dy * dy));
				out_x = x;
				out_y = y;
				return dist + penC + archPen;
		}
};

static __declspec(noalias) __forceinline void HitTest2D_analytic(const float x_param, float& out_x1, float& out_x2) noexcept {
	const float a = gActiveMap.a, inv_lenx = gActiveMap.inv_lenx;
	const unsigned scale = gActiveMap.scale, scale_minus_1 = scale - 1u;
	const float lenx = gActiveMap.lenx, leny = gActiveMap.leny, c = gActiveMap.c;
	const unsigned start = gActiveMap.start;
	const int levels = gActiveMap.levels;

	float norm = (x_param - a) * inv_lenx;
	norm = fminf(fmaxf(norm, 0.0f), 0x1.fffffep-1f);
	unsigned idx = static_cast<unsigned>(norm * static_cast<float>(scale));
	idx = idx > scale_minus_1 ? scale_minus_1 : idx;

	float sx = lenx, sy = leny;
	float x1 = a, x2 = c;
	unsigned type = start;
	int l = levels - 1;
	while (l >= 0) {
		const unsigned q = (idx >> (l * 2)) & 3u;
		const Step s = g_step_tbl[type][q];
		type = s.next;
		sx *= 0.5f;
		sy *= 0.5f;
		x1 += s.dx ? sx : 0.0f;
		x2 += s.dy ? sy : 0.0f;
		--l;
	}
	out_x1 = x1 + sx * 0.5f;
	out_x2 = x2 + sy * 0.5f;
}

static __declspec(noalias) __forceinline float FindX2D_analytic(const float px, const float py) noexcept {
	const float a = gActiveMap.a, b = gActiveMap.b, c = gActiveMap.c, d = gActiveMap.d;
	const float lenx = gActiveMap.lenx, leny = gActiveMap.leny;
	const unsigned scale = gActiveMap.scale;
	const unsigned start = gActiveMap.start;
	const int levels = gActiveMap.levels;
	const float clamped_px = fminf(fmaxf(px, a), b), clamped_py = fminf(fmaxf(py, c), d);
	float sx = lenx, sy = leny;
	float x0 = a, y0 = c;
	unsigned idx = 0u;
	unsigned type = start;
	int l = 0;
	while (l < levels) {
		sx *= 0.5f;
		sy *= 0.5f;
		const float mx = x0 + sx, my = y0 + sy;
		const unsigned tr = static_cast<unsigned>((clamped_px > mx) & (clamped_py > my));
		const unsigned tl = static_cast<unsigned>((clamped_px < mx) & (clamped_py > my));
		const unsigned dl = static_cast<unsigned>((clamped_px < mx) & (clamped_py < my));
		const unsigned none = static_cast<unsigned>(1u ^ (tr | tl | dl));
		const unsigned dd = (tr << 1) | tr | tl | (none << 1);
		const InvStep inv = g_inv_tbl[type][dd];
		type = inv.next;
		idx = (idx << 2) | inv.q;
		const unsigned dx = dd >> 1, dy = dd & 1u;
		x0 += dx ? sx : 0.0f;
		y0 += dy ? sy : 0.0f;
		++l;
	}
	const float scale_recip = 1.0f / static_cast<float>(scale);
	return fmaf(static_cast<float>(idx) * scale_recip, lenx, a);
}

static __declspec(noalias) __forceinline int generate_lhs_seeds_lite(const MortonND& map, const int dim, float* const __restrict S, const int stride, unsigned seed) noexcept {
	int temp_dim = dim;
	const int ns = --temp_dim * temp_dim;
	unsigned st = seed;
	alignas(16) int permutations[32][256];

	int d = 0;
#pragma loop ivdep
	while (d < dim) {
		int s = 0;
#pragma loop ivdep
		while (s < ns) {
			permutations[d][s] = s;
			++s;
		}
		s = ns - 1;
		while (s > 0) {
			st ^= st << 13;
			st ^= st >> 17;
			st ^= st << 5;
			const int j = static_cast<int>(st % static_cast<unsigned>(s + 1));
			const int t = permutations[d][s];
			permutations[d][s] = permutations[d][j];
			permutations[d][j] = t;
			--s;
		}
		++d;
	}

	int s2 = 0;
#pragma loop ivdep
	while (s2 < ns) {
		d = 0;
#pragma loop ivdep
		while (d < dim) {
			st ^= st << 13;
			st ^= st >> 17;
			st ^= st << 5;
			const float u = (st & 0xFFFFFF) * 5.9604645e-8f;
			const int stratum = permutations[d][s2];
			const float pos = (static_cast<float>(stratum) + u) / static_cast<float>(ns);
			const int pd = map.perm[d];
			const float lo = map.low[pd], hi = map.high[pd];
			S[s2 * stride + d] = fmaf(pos, (hi - lo), lo);
			++d;
		}
		++s2;
	}
	return ns;
}

static __declspec(noalias) __forceinline int generate_heuristic_seeds(const ManipCost& cost, const MortonND& map, const int dim, float* const __restrict S, const int stride, const unsigned seed) noexcept {
	const int n = cost.n;
	const bool VL = cost.variableLen;
	const float tx = cost.targetX, ty = cost.targetY;
	int total_seeds = 0;

	{
		float* const s0 = S + total_seeds * stride;
		float sin_phi, cos_phi;
		const float rho = sqrtf(fmaf(tx, tx, ty * ty));
		FABE13_SINCOS(&tx, &sin_phi, &cos_phi, 1);
		const float phi = (fabsf(sin_phi) > 0.0f) ? atan2f(ty, tx) : 0.0f;
		const float len = fminf(fmaxf(fmaf(1.0f / static_cast<float>(n), rho, 0.0f), 0.5f), 2.0f);
		int i = 0;
#pragma loop ivdep
		while (i < n) {
			s0[i] = (1.0f / static_cast<float>(n)) * phi;
			++i;
		}
		if (VL) {
			i = 0;
			while (i < n) {
				s0[n + i] = len;
				++i;
			}
		}
		++total_seeds;
	}

	{
		float* const s1 = S + total_seeds * stride;
		float sin_phi, cos_phi;
		FABE13_SINCOS(&tx, &sin_phi, &cos_phi, 1);
		const float phi = (fabsf(sin_phi) > 0.0f) ? atan2f(ty, tx) : 0.0f;
		int i = 0;
#pragma loop ivdep
		while (i < n) {
			s1[i] = fmaf(0.5f, phi, 0.0f) * ((i & 1) ? -1.0f : 1.0f);
			++i;
		}
		if (VL) {
			i = 0;
			while (i < n) {
				s1[n + i] = fmaf(0.4f, static_cast<float>(i) / static_cast<float>(n), 0.8f);
				++i;
			}
		}
		++total_seeds;
	}

	{
		float* const s2 = S + total_seeds * stride;
		const float inv = (n > 1) ? 1.0f / static_cast<float>(n - 1) : 0.0f;
		float sin_phi, cos_phi;
		FABE13_SINCOS(&tx, &sin_phi, &cos_phi, 1);
		const float phi = (fabsf(sin_phi) > 0.0f) ? atan2f(ty, tx) : 0.0f;
		int i = 0;
#pragma loop ivdep
		while (i < n) {
			const float pr = static_cast<float>(i) * inv;
			s2[i] = fmaf(phi, fmaf(-0.3f, pr, 1.0f), 0.0f);
			++i;
		}
		if (VL) {
			int j = 0;
			while (j < n) {
				float si;
				FABE13_SIN(fmaf(1.5f, static_cast<float>(j), 0.0f), si);
				s2[n + j] = fmaf(0.2f, si, 1.0f);
				++j;
			}
		}
		++total_seeds;
	}

	const int lhs_count = generate_lhs_seeds_lite(map, dim, S + total_seeds * stride, stride, seed);
	total_seeds += lhs_count;
	return total_seeds;
}

static __declspec(noalias) void agp_run_branch_mpi(
	const MortonND& map, const ManipCost& cost, const int maxIter, const float r, const bool adaptive, const float eps, const unsigned seed,
	std::vector<IntervalND*, boost::alignment::aligned_allocator<IntervalND*, 16u>>& H,
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>>& bestQ,
	float& bestF, float& bestX, float& bestY, size_t& out_iterations, float& out_achieved_epsilon, const float M_prior = 1e-3f)
	noexcept {
	const int n = cost.n;
	const int dim = n + (cost.variableLen ? n : 0);
	const float dim_f = static_cast<float>(dim);
	unsigned exchange_counter_500 = 0;
	unsigned exchange_counter_T = 0;

	alignas(16) float M_by_span[12];
	int msi = 0;
	while (msi < 12) {
		M_by_span[msi++] = M_prior;
	}
	float Mmax = M_prior;

	alignas(16) float q_local[32], phi[32], s_arr[32], c_arr[32], sum_s[32], sum_c[32], q_try[32];
	bestQ.reserve(static_cast<size_t>(dim));
	float x = 0.0f, y = 0.0f;
	int no_improve = 0;

	auto t_to_idx = [&](const float t) -> unsigned long long {
		unsigned long long idx = static_cast<unsigned long long>(fmaf(t, static_cast<float>(map.scale), 0.0f));
		return idx;
		};

	auto update_pockets_and_Mmax = [&](IntervalND* const I) {
		const int k = I->span_level;
		if (I->M > M_by_span[k]) M_by_span[k] = I->M;
		if (M_by_span[k] > Mmax) Mmax = M_by_span[k];
		};

	const float a = 0.0f, b = 1.0f;

	auto evalAt = [&](const float t) -> float {
		map.map01ToPoint(t, q_local);
		float f = cost(q_local, x, y);

		if (f < bestF * 1.25f) {
			float acc = 0.0f;
			int ii = 0;
			while (ii < n) {
				acc = fmaf(q_local[ii], 1.0f, acc);
				phi[ii] = acc;
				++ii;
			}
			FABE13_SINCOS(phi, s_arr, c_arr, n);
			float as = 0.0f, ac = 0.0f;
			int k = n - 1;
			while (k >= 0) {
				const float Lk = cost.variableLen ? q_local[n + k] : 1.0f;
				as = fmaf(Lk, s_arr[k], as);
				ac = fmaf(Lk, c_arr[k], ac);
				sum_s[k] = as;
				sum_c[k] = ac;
				--k;
			}
			const float dx = fmaf(x, 1.0f, -cost.targetX);
			const float dy = fmaf(y, 1.0f, -cost.targetY);
			const float dist = sqrtf(fmaf(dx, dx, dy * dy)) + 1.0e-8f;

			float eta = 0.125f;
			int stepI = 0;
			while (stepI < 3) {
				int i = 0;
#pragma loop ivdep
				while (i < n) {
					float gpen = 0.0f;
					{
						const float ai = fabsf(q_local[i]);
						const float v = ai - cost.minTheta;
						if (v > 0.0f) {
							const float scale_arg = (2.0f / fmaf(cost.minTheta, 1.0f, 1.0e-6f)) * v * 0.69314718055994530941723212145818f;
							const float exp_val = fmaf(scale_arg, fmaf(scale_arg, fmaf(scale_arg, fmaf(scale_arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
							const float dpen_dtheta = cost.sharpW * exp_val * (0.69314718055994530941723212145818f * (2.0f / (cost.minTheta + 1.0e-6f))) * copysignf(1.0f, q_local[i]);
							gpen = fmaf(dpen_dtheta, 1.0f, gpen);
						}
					}
					{
						const float tsg = fmaf(-q_local[i], cost.archBiasK, 0.0f);
						const float exp_arg = -tsg;
						const float exp_val = fmaf(exp_arg, fmaf(exp_arg, fmaf(exp_arg, fmaf(exp_arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
						const float sig = 1.0f / fmaf(exp_val, 1.0f, 1.0f);
						gpen = fmaf(-(cost.archBiasW * cost.archBiasK), sig, gpen);
					}

					const float g = fmaf(fmaf(dx, -sum_s[i], fmaf(dy, sum_c[i], 0.0f)), 1.0f / dist, gpen);
					q_try[i] = fmaf(-eta, g, q_local[i]);

					const float lo0 = -1.0471975511965977461542144610932f;
					const float hi0 = 2.6179938779914943653855361527329f;
					const float lo = -2.6179938779914943653855361527329f;
					const float hi = 2.6179938779914943653855361527329f;
					const float Lb = (i == 0) ? lo0 : lo;
					const float Hb = (i == 0) ? hi0 : hi;
					if (q_try[i] < Lb) q_try[i] = Lb;
					else if (q_try[i] > Hb) q_try[i] = Hb;
					++i;
				}
				if (cost.variableLen) {
					int j = 0;
#pragma loop ivdep
					while (j < n) {
						const float g = fmaf(fmaf(dx, c_arr[j], fmaf(dy, s_arr[j], 0.0f)), 1.0f / dist, 0.0f);
						float v = fmaf(-eta, g, q_local[n + j]);
						if (v < 0.5f) v = 0.5f;
						else if (v > 2.0f) v = 2.0f;
						q_try[n + j] = v;
						++j;
					}
				}
				float x2, y2;
				const float f2 = cost(q_try, x2, y2);
				if (f2 < f) {
					memcpy(q_local, q_try, static_cast<size_t>(dim) * sizeof(float));
					f = f2;
					x = x2;
					y = y2;
					break;
				}
				eta = fmaf(eta, 0.5f, 0.0f);
				++stepI;
			}

			const int last = n - 1;
			const float lo = (last == 0) ? -1.0471975511965977461542144610932f : -2.6179938779914943653855361527329f;
			const float hi = 2.6179938779914943653855361527329f;
			float bestLocF = f;
			float saved = q_local[last];
			float delta = 0.05f;
			while (delta >= 0.00625f) {
				int sgn = -1;
				while (sgn <= 1) {
					float cand = fmaf(static_cast<float>(sgn), delta, saved);
					if (cand < lo) cand = lo;
					else if (cand > hi) cand = hi;
					const float backup = q_local[last];
					q_local[last] = cand;
					float x2, y2;
					const float f2 = cost(q_local, x2, y2);
					if (f2 < bestLocF) {
						bestLocF = f2;
						x = x2;
						y = y2;
						saved = cand;
					}
					q_local[last] = backup;
					sgn += 2;
				}
				delta = fmaf(delta, 0.5f, 0.0f);
			}
			if (bestLocF < f) {
				q_local[last] = saved;
				f = bestLocF;
			}
		}

		if (f < bestF) {
			bestF = f;
			bestQ.assign(q_local, q_local + dim);
			bestX = x;
			bestY = y;
			no_improve = 0;
		}
		else {
			++no_improve;
		}
		return f;
		};

	const float f_a = evalAt(a), f_b = evalAt(b);
	const float Kf = fminf(fmaxf(fmaf(2.0f, dim_f, 0.0f), 8.0f), 128.0f);
	const int K = static_cast<int>(Kf);

	H.reserve(static_cast<size_t>(maxIter) + static_cast<size_t>(K) + 16u);
	const int rank = g_world->rank();
	const int world = g_world->size();

	alignas(16) float seeds[256 * 32];
	const int seedCnt = generate_heuristic_seeds(cost, map, dim, seeds, 32, fmaf(static_cast<float>(rank), 7919.0f, static_cast<float>(seed)));

	int i = 0;
	while (i < seedCnt) {
		const float* const s = seeds + static_cast<size_t>(fmaf(static_cast<float>(i), 32.0f, 0.0f));
		const float t_seed = map.pointToT(s);
		const float interval_size = (i < 3) ? fmaf(0.0004f, static_cast<float>(dim), 0.0f)
			: fmaf(fmaf(0.00031f, static_cast<float>(dim), 0.0f),
				exp2f(fmaf((1.0f / static_cast<float>(seedCnt - 4)) * log2f(fmaf(0.00025f, 1.0f / 0.00031f, 0.0f)),
					static_cast<float>(i - 3), 0.0f)),
				0.0f);
		const float t1 = fmaxf(a, fmaf(-interval_size, 1.0f, t_seed));
		const float t2 = fminf(b, fmaf(interval_size, 1.0f, t_seed));
		if (t2 > t1) {
			alignas(16) float q1[32], q2[32];
			float x1, y1, x2, y2;
			map.map01ToPoint(t1, q1);
			const float f1 = cost(q1, x1, y1);
			map.map01ToPoint(t2, q2);
			const float f2 = cost(q2, x2, y2);
			IntervalND* const I = new IntervalND(t1, t2, f1, f2);
			I->i1 = t_to_idx(t1);
			I->i2 = t_to_idx(t2);
			I->diam = map.block_diameter(I->i1, I->i2);
			I->compute_span_level(map);
			I->set_metric(I->diam);
			update_pockets_and_Mmax(I);
			I->ChangeCharacteristic(fmaf(r, Mmax, 0.0f));
			if (i < 3) {
				I->R = fmaf(I->R, fmaf(0.01f, static_cast<float>(dim), 0.85f), 0.0f);
			}
			else {
				const float start_mult = fmaf(0.214f, static_cast<float>(dim), 0.0f);
				const float end_mult = fmaf(0.174f, static_cast<float>(dim), 0.0f);
				const float mult = fmaf(start_mult,
					exp2f(fmaf((1.0f / static_cast<float>(seedCnt - 4)) * log2f(fmaf(end_mult, 1.0f / start_mult, 0.0f)),
						static_cast<float>(i - 3), 0.0f)),
					0.0f);
				I->R = fmaf(I->R, mult, 0.0f);
			}
			H.emplace_back(I);
			std::push_heap(H.begin(), H.end(), ComparePtrND);
			if (f1 < bestF) {
				bestF = f1;
				bestQ.assign(q1, q1 + dim);
				bestX = x1;
				bestY = y1;
			}
			if (f2 < bestF) {
				bestF = f2;
				bestQ.assign(q2, q2 + dim);
				bestX = x2;
				bestY = y2;
			}
		}
		++i;
	}

	float prev_t = a, prev_f = f_a;
	int k = 1;
	while (k <= K) {
		const float t = fmaf(fmaf((b - a), static_cast<float>(k) / static_cast<float>(K + 1), a),
			1.0f,
			static_cast<float>(rank) / static_cast<float>(world * (K + 1)));
		const float f = evalAt(t);
		IntervalND* const I = new IntervalND(prev_t, t, prev_f, f);
		I->i1 = t_to_idx(prev_t);
		I->i2 = t_to_idx(t);
		I->diam = map.block_diameter(I->i1, I->i2);
		I->compute_span_level(map);
		I->set_metric(I->diam);
		update_pockets_and_Mmax(I);
		I->ChangeCharacteristic(fmaf(r, Mmax, 0.0f));
		H.emplace_back(I);
		std::push_heap(H.begin(), H.end(), ComparePtrND);
		prev_t = t;
		prev_f = f;
		++k;
	}
	IntervalND* const tail = new IntervalND(prev_t, b, prev_f, f_b);
	tail->i1 = t_to_idx(prev_t);
	tail->i2 = t_to_idx(b);
	tail->diam = map.block_diameter(tail->i1, tail->i2);
	tail->compute_span_level(map);
	tail->set_metric(tail->diam);
	update_pockets_and_Mmax(tail);
	tail->ChangeCharacteristic(fmaf(r, Mmax, 0.0f));
	H.emplace_back(tail);
	std::push_heap(H.begin(), H.end(), ComparePtrND);

	float dmax = fmaf(b, 1.0f, -a);
	const float initial_len = dmax;
	const float thr03 = fmaf(0.3f, initial_len, 0.0f);
	const float inv_thr03 = 1.0f / thr03;
	int it = 0;

	float kickEveryDimf = fmaf(120.0f, exp2f(fmaf(-0.05f, dim_f, 0.0f)), 0.0f);
	if (kickEveryDimf < 60.0f) kickEveryDimf = 60.0f;
	const int kickEveryDim = static_cast<int>(kickEveryDimf);

	float noImproveThrDimf = fmaf(80.0f, exp2f(fmaf(-0.08f, dim_f, 0.0f)), 0.0f);
	if (noImproveThrDimf < 30.0f) noImproveThrDimf = 30.0f;
	const int noImproveThrDim = static_cast<int>(noImproveThrDimf);

	auto kickEveryByDim = [&](const int d) -> int {
		float z = fmaf(120.0f, exp2f(fmaf(-0.05f, static_cast<float>(d), 0.0f)), 0.0f);
		if (z < 60.0f) z = 60.0f;
		return static_cast<int>(z);
		};

	auto noImproveThrByDim = [&](const int d) -> int {
		float z = fmaf(80.0f, exp2f(fmaf(-0.08f, static_cast<float>(d), 0.0f)), 0.0f);
		if (z < 30.0f) z = 30.0f;
		return static_cast<int>(z);
		};

	while (it < maxIter) {
		if ((it % kickEveryDim) == 0 && no_improve > noImproveThrDim) {
			const float t_best = map.pointToT(bestQ.data());
			int ii = 0;
			while (ii < 2) {
				const float off = (ii == 0) ? 0.01f : -0.01f;
				const float t_seed = fminf(b, fmaxf(a, fmaf(off, 1.0f, t_best)));
				const float f_seed = evalAt(t_seed);
				IntervalND* const J = new IntervalND(fmaf(-0.005f, 1.0f, t_seed), fmaf(0.005f, 1.0f, t_seed), f_seed, f_seed);
				J->i1 = t_to_idx(fmaf(-0.005f, 1.0f, t_seed));
				J->i2 = t_to_idx(fmaf(0.005f, 1.0f, t_seed));
				J->diam = map.block_diameter(J->i1, J->i2);
				J->compute_span_level(map);
				J->set_metric(J->diam);
				update_pockets_and_Mmax(J);
				J->ChangeCharacteristic(fmaf(r, Mmax, 0.0f));
				J->R = fmaf(J->R, 0.9f, 0.0f);
				H.emplace_back(J);
				std::push_heap(H.begin(), H.end(), ComparePtrND);
				++ii;
			}
			no_improve = 0;
		}

		const float p = fmaf(-1.0f / initial_len, dmax, 1.0f);
		const bool stagnation = (no_improve > 100) && (it > 270);

		const float exp_arg = fmaf(-0.06f, dim_f, 0.0f);
		const float exp2_exp_arg = fmaf(exp_arg * 0.69314718055994530941723212145818f, fmaf(exp_arg * 0.69314718055994530941723212145818f, fmaf(exp_arg * 0.69314718055994530941723212145818f, fmaf(exp_arg * 0.69314718055994530941723212145818f, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
		const float A = fmaf(64.0f, exp2_exp_arg, 200.0f);
		const float B = fmaf(67.0f, exp2_exp_arg, 210.0f);
		const int T = static_cast<int>(fmaf(-(exp2_exp_arg - 1.0f), A, B));

		const float p_arg = fmaf(p, 2.3f, -3.0f);
		const float r_eff = fmaf(-fmaf(p_arg, fmaf(p_arg, fmaf(p_arg, fmaf(p_arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), 1.0f, 1.05f);

		std::pop_heap(H.begin(), H.end(), ComparePtrND);
		IntervalND* const cur = H.back();
		H.pop_back();

		const float x1 = cur->x1, x2 = cur->x2, y1 = cur->y1, y2 = cur->y2;
		float m = fmaf(r_eff, Mmax, 0.0f);
		float tNew = step(m, x1, x2, y1, y2, dim_f, r);
		tNew = fminf(fmaxf(tNew, a), b);
		const float fNew = evalAt(tNew);

		IntervalND* const L = new IntervalND(x1, tNew, y1, fNew);
		IntervalND* const Rv = new IntervalND(tNew, x2, fNew, y2);

		L->i1 = t_to_idx(x1);
		L->i2 = t_to_idx(tNew);
		Rv->i1 = t_to_idx(tNew);
		Rv->i2 = t_to_idx(x2);
		L->diam = map.block_diameter(L->i1, L->i2);
		Rv->diam = map.block_diameter(Rv->i1, Rv->i2);
		L->compute_span_level(map);
		Rv->compute_span_level(map);
		L->set_metric(L->diam);
		Rv->set_metric(Rv->diam);

		const float Mloc = fmaxf(L->M, Rv->M);
		update_pockets_and_Mmax(L);
		update_pockets_and_Mmax(Rv);

		const float prevMmax = Mmax;
		if (Mloc > Mmax) Mmax = Mloc;
		m = fmaf(r_eff, Mmax, 0.0f);

		if (adaptive) {
			const float len1 = fmaf(tNew, 1.0f, -x1);
			const float len2 = fmaf(x2, 1.0f, -tNew);
			if (fmaf(len1, 1.0f, len2) == dmax) {
				dmax = fmaxf(len1, len2);
				for (auto pI : H) {
					const float Ls = fmaf(pI->x2, 1.0f, -pI->x1);
					if (Ls > dmax) dmax = Ls;
				}
			}
			if ((thr03 > dmax && !(it % 3)) || (fmaf(10.0f, dmax, 0.0f) < initial_len)) {
				const float progress = fmaf(-inv_thr03, dmax, 1.0f);
				const float alpha = fmaf(progress, progress, 0.0f);
				const float beta = fmaf(-alpha, 1.0f, 2.0f);
				const float MULT = (1.0f / dmax) * Mmax;
				const float global_coeff = fmaf(MULT, r_eff, -MULT);
				const float GF = fmaf(beta, global_coeff, 0.0f);
				L->ChangeCharacteristic(fmaf(GF, len1, fmaf(L->M, alpha, 0.0f)));
				Rv->ChangeCharacteristic(fmaf(GF, len2, fmaf(Rv->M, alpha, 0.0f)));
				const size_t sz = H.size();
				RecomputeR_AffineM_AVX2_ND(H.data(), sz, GF, alpha);
				std::make_heap(H.begin(), H.end(), ComparePtrND);
			}
			else {
				if (Mloc > prevMmax) {
					L->ChangeCharacteristic(m);
					Rv->ChangeCharacteristic(m);
					if (Mloc > fmaf(1.15f, prevMmax, 0.0f)) {
						const size_t sz = H.size();
						RecomputeR_ConstM_AVX2_ND(H.data(), sz, m);
						std::make_heap(H.begin(), H.end(), ComparePtrND);
					}
				}
				else {
					L->ChangeCharacteristic(m);
					Rv->ChangeCharacteristic(m);
				}
			}
		}
		else {
			if (Mloc > prevMmax) {
				L->ChangeCharacteristic(m);
				Rv->ChangeCharacteristic(m);
				if (Mloc > fmaf(1.15f, prevMmax, 0.0f)) {
					const size_t sz = H.size();
					RecomputeR_ConstM_AVX2_ND(H.data(), sz, m);
					std::make_heap(H.begin(), H.end(), ComparePtrND);
				}
			}
			else {
				L->ChangeCharacteristic(m);
				Rv->ChangeCharacteristic(m);
			}
		}

		H.emplace_back(L);
		std::push_heap(H.begin(), H.end(), ComparePtrND);
		H.emplace_back(Rv);
		std::push_heap(H.begin(), H.end(), ComparePtrND);
		_mm_prefetch(reinterpret_cast<const char*>(H[0]), _MM_HINT_T0);
		_mm_prefetch(reinterpret_cast<const char*>(H[1]), _MM_HINT_T0);

		IntervalND* const top = H.front();
		const float interval_len = top->x2 - top->x1;

		if ((exp2f((1.0f / dim_f) * log2f(interval_len)) < eps) || (it == maxIter)) {
			out_iterations = static_cast<size_t>(it);
			out_achieved_epsilon = interval_len;
			return;
		}

		if (!(it % T)) {
			MultiCrossMsg out;
			out.count = 3;
			float* dest = out.intervals;
			IntervalND* const t1 = H[0];
			IntervalND* const t2 = H[1];
			IntervalND* const t3 = H[2];
			IntervalND* const tops[3] = { t1, t2, t3 };
			unsigned i2 = 0;
			while (i2 < 3) {
				IntervalND* const Tt = tops[i2];
				dest[0] = Tt->x1;
				dest[1] = 0.0f;
				dest[2] = Tt->x2;
				dest[3] = 0.0f;
				dest[4] = Tt->R;
				dest += 5;
				++i2;
			}
			const size_t iterations = std::bit_width(static_cast<size_t>(world - 1));
			bool active = true;
			const bool invert_T = static_cast<int>(fmaf(static_cast<float>(exchange_counter_T), 1.0f, 1.0f)) & 1;

			size_t ii = 0;
			while (ii < iterations && active) {
				const size_t step = 1ULL << ii;
				const int partner = rank ^ static_cast<int>(step);
				if (partner < world) {
					const bool am_sender = (!!(rank & static_cast<int>(step))) ^ invert_T;
					if (am_sender) {
						g_world->isend(partner, 0, out);
						active = false;
					}
				}
				++ii;
			}
			++exchange_counter_T;
		}

		if (!(it % 500)) {
			BestSolutionMsg out;
			out.bestF = bestF;
			out.bestX = bestX;
			out.bestY = bestY;
			out.dim = static_cast<unsigned>(bestQ.size());
			memcpy(out.bestQ, bestQ.data(), bestQ.size() * sizeof(float));
			const size_t iterations = std::bit_width(static_cast<size_t>(world - 1));
			bool active = true;
			const bool invert_T = static_cast<int>(fmaf(static_cast<float>(exchange_counter_500), 1.0f, 1.0f)) & 1;

			size_t ii = 0;
			while (ii < iterations && active) {
				const size_t step = 1ULL << ii;
				const int partner = rank ^ static_cast<int>(step);
				if (partner < world) {
					const bool am_sender = (!!(rank & static_cast<int>(step))) ^ invert_T;
					if (am_sender) {
						g_world->isend(partner, 2, out);
						active = false;
					}
				}
				++ii;
			}
			++exchange_counter_500;
		}

		while (g_world->iprobe(boost::mpi::any_source, 0)) {
			MultiCrossMsg in;
			g_world->recv(boost::mpi::any_source, 0, in);
			const MultiCrossMsg& mX = in;
			unsigned ii = 0;
			while (ii < mX.count) {
				const float* const d = &mX.intervals[ii * 5];
				float sx = d[0], ex = d[2];
				if (ex > sx) {
					alignas(16) float tmp[32];
					float tx, ty;
					map.map01ToPoint(sx, tmp);
					const float y1i = cost(tmp, tx, ty);
					map.map01ToPoint(ex, tmp);
					const float y2i = cost(tmp, tx, ty);
					IntervalND* const inj = new IntervalND(sx, ex, y1i, y2i);
					inj->i1 = t_to_idx(sx);
					inj->i2 = t_to_idx(ex);
					inj->diam = map.block_diameter(inj->i1, inj->i2);
					inj->compute_span_level(map);
					inj->set_metric(inj->diam);
					update_pockets_and_Mmax(inj);
					inj->ChangeCharacteristic(fmaf(r, Mmax, 0.0f));
					_mm_prefetch(reinterpret_cast<const char*>(H[0]), _MM_HINT_T0);
					_mm_prefetch(reinterpret_cast<const char*>(H[1]), _MM_HINT_T0);
					IntervalND* const topH = H.front();
					if (inj->R > fmaf(1.15f, topH->R, 0.0f)) {
						const float p2 = fmaf(-1.0f / initial_len, dmax, 1.0f);
						const float kf = (no_improve > 100 && it > 270) ? fmaf(0.5819767068693265f, (fmaf(p2 * 0.69314718055994530941723212145818f, fmaf(p2 * 0.69314718055994530941723212145818f, fmaf(p2 * 0.69314718055994530941723212145818f, fmaf(p2 * 0.69314718055994530941723212145818f, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f) - 1.0f), 0.3f)
							: fmaf(0.3491860241215959f, (fmaf(p2 * 0.69314718055994530941723212145818f, fmaf(p2 * 0.69314718055994530941723212145818f, fmaf(p2 * 0.69314718055994530941723212145818f, fmaf(p2 * 0.69314718055994530941723212145818f, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f) - 1.0f), 0.6f);
						inj->R = fmaf(d[4], kf, 0.0f);
						H.emplace_back(inj);
						std::push_heap(H.begin(), H.end(), ComparePtrND);
					}
				}
				++ii;
			}
		}
		while (g_world->iprobe(boost::mpi::any_source, 2)) {
			BestSolutionMsg bm;
			g_world->recv(boost::mpi::any_source, 2, bm);
			if (bm.bestF < fmaf(bestF, 1.15f, 0.0f)) {
				alignas(16) float tmp_q[32];
				memcpy(tmp_q, bm.bestQ, bm.dim * sizeof(float));
				const float t_best = map.pointToT(tmp_q);
				const float t1 = fmaxf(a, fmaf(-0.001f, 1.0f, t_best));
				const float t2 = fminf(b, fmaf(0.001f, 1.0f, t_best));
				if (t2 > t1) {
					alignas(16) float tq1[32], tq2[32];
					float xx1, yy1, xx2, yy2;
					map.map01ToPoint(t1, tq1);
					const float f1 = cost(tq1, xx1, yy1);
					map.map01ToPoint(t2, tq2);
					const float f2 = cost(tq2, xx2, yy2);
					IntervalND* const I = new IntervalND(t1, t2, f1, f2);
					I->i1 = t_to_idx(t1);
					I->i2 = t_to_idx(t2);
					I->diam = map.block_diameter(I->i1, I->i2);
					I->compute_span_level(map);
					I->set_metric(I->diam);
					update_pockets_and_Mmax(I);
					I->ChangeCharacteristic(fmaf(r, Mmax, 0.0f));
					I->R = fmaf(I->R, 0.90f, 0.0f);
					H.emplace_back(I);
					std::push_heap(H.begin(), H.end(), ComparePtrND);
				}
				if (bm.bestF < bestF) {
					bestF = bm.bestF;
					bestX = bm.bestX;
					bestY = bm.bestY;
					bestQ.assign(bm.bestQ, bm.bestQ + bm.dim);
				}
			}
		}
		++it;
	}
}

static __declspec(noalias) __forceinline float PivotCalculation(std::vector<IntervalND*>::iterator first, std::vector<IntervalND*>::iterator last) noexcept {
	const auto mid = first + ((last - first) >> 1);
	float pivot_value = NAN;
	if (last - first < 199) {
		pivot_value = (*mid)->R;
	}
	else {
		if ((*first)->R < (*mid)->R) {
			if ((*mid)->R < (*last)->R) {
				pivot_value = (*mid)->R;
			}
			else {
				pivot_value = std::max((*first)->R, (*last)->R);
			}
		}
		else {
			if ((*first)->R < (*last)->R) {
				pivot_value = (*first)->R;
			}
			else {
				pivot_value = std::max((*mid)->R, (*last)->R);
			}
		}
	}
	return pivot_value;
}

static __declspec(noalias) __forceinline void HoaraSort(std::vector<IntervalND*>::iterator first, std::vector<IntervalND*>::iterator last) noexcept {
	if (first >= last) {
		return;
	}
	const float pivot_value = PivotCalculation(first, last);
	auto left = first;
	auto right = last;
	do {
		while (left < last && (*left)->R < pivot_value) {
			left++;
		}
		while (right > first && (*right)->R > pivot_value) {
			right--;
		}
		if ((*left)->R == (*right)->R && left != right) {
			if ((*left)->R < (*(left + 1))->R) {
				left++;
			}
			else {
				right--;
			}
		}
		std::iter_swap(left, right);
	} while (left != right);
	if (last - first < 199) {
		HoaraSort(first, right);
		HoaraSort(left + 1, last);
	}
	else {
		oneapi::tbb::parallel_invoke([&first, &right]() { HoaraSort(first, right); },
			[&left, &last]() { HoaraSort(left + 1, last); });
	}
}

extern "C" __declspec(dllexport) __declspec(noalias)
void AGP_Manip2D(const int nSegments, const bool variableLengths, const float minTheta,
	const float targetX, const float targetY, const int peanoLevels,
	const int maxIterPerBranch, const float r, const bool adaptiveMode,
	const float epsilon, const unsigned int seed,
	const float baseLength, const float stretchFactor,
	float** const out_bestQ, size_t* const out_bestQLen, float* const out_bestX,
	float* const out_bestY, float* const out_bestF, size_t* const out_iterations,
	float* const out_achieved_epsilon) noexcept {
	Slab* const __restrict slab = tls.local();
	slab->current = slab->base;
	while (g_world->iprobe(boost::mpi::any_source, 0)) {
		MultiCrossMsg dummy;
		g_world->recv(boost::mpi::any_source, 0, dummy);
	}
	while (g_world->iprobe(boost::mpi::any_source, 2)) {
		BestSolutionMsg dummy;
		g_world->recv(boost::mpi::any_source, 2, dummy);
	}
	const int dim = nSegments + (variableLengths ? nSegments : 0);

	g_mc.permCache.resize(static_cast<size_t>(dim));
	int i = 0;
	while (i < dim) {
		g_mc.permCache[i] = i;
		++i;
	}
	unsigned s = g_mc.baseSeed;
	i = dim - 1;
	while (i > 0) {
		s ^= s << 13;
		s ^= s >> 17;
		s ^= s << 5;
		const unsigned j = s % static_cast<unsigned>(i + 1);
		std::swap(g_mc.permCache[i], g_mc.permCache[j]);
		--i;
	}
	g_mc.invMaskCache.resize(static_cast<size_t>(dim));
	int k = 0;
	while (k < dim) {
		s ^= s << 13;
		s ^= s >> 17;
		s ^= s << 5;
		g_mc.invMaskCache[k] = static_cast<unsigned long long>(s);
		++k;
	}

	std::vector<float, boost::alignment::aligned_allocator<float, 16u>> low;
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>> high;
	low.reserve(static_cast<size_t>(dim));
	high.reserve(static_cast<size_t>(dim));
	i = 0;
	while (i < nSegments) {
		low.emplace_back(i == 0 ? -1.0471975511965977461542144610932f : -2.6179938779914943653855361527329f);
		high.emplace_back(2.6179938779914943653855361527329f);
		++i;
	}
	if (variableLengths) {
		i = 0;
		const float lengthLower = baseLength / stretchFactor;
		const float lengthUpper = baseLength * stretchFactor;
		while (i < nSegments) {
			low.emplace_back(lengthLower);
			high.emplace_back(lengthUpper);
			++i;
		}
	}

	const ManipCost cost(nSegments, variableLengths, targetX, targetY, minTheta);

	const int rank = g_world->rank(), world = g_world->size();
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>> bestQ;
	float bestF = FLT_MAX, bx = 0.0f, by = 0.0f;

	const int levels0 = static_cast<int>(fminf(static_cast<float>(peanoLevels), 8.0f));
	const int maxIter0 = static_cast<int>(fmaf(static_cast<float>(maxIterPerBranch), 0.2f, 0.0f));
	const MortonND map0(dim, levels0, low.data(), high.data(), g_mc);

	std::vector<IntervalND*, boost::alignment::aligned_allocator<IntervalND*, 16u>> H_coarse;
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>> bestQ_coarse;
	float bestF_coarse = FLT_MAX, bx_coarse = 0.0f, by_coarse = 0.0f;
	size_t total_oi = 0u;
	float total_oe = 0.0f;
	size_t oi = 0u;
	float oe = 0.0f;

	const float base_M_prior_factor = fmaf(2.0f, static_cast<float>(nSegments), variableLengths ? 1.41421356f : 0.0f);

	float M_prior = fmaf(base_M_prior_factor,
		ldexpf(1.0f, -levels0),
		0.0f);

	agp_run_branch_mpi(map0, cost, maxIter0, r, adaptiveMode, epsilon, seed,
		H_coarse, bestQ_coarse, bestF_coarse, bx_coarse, by_coarse, oi, oe, M_prior);

	total_oi += oi;
	total_oe = oe;

	if (bestF_coarse < bestF) {
		bestF = bestF_coarse;
		bestQ = std::move(bestQ_coarse);
		bx = bx_coarse;
		by = by_coarse;
	}

	if (levels0 < peanoLevels) {
		while (g_world->iprobe(boost::mpi::any_source, 0)) {
			MultiCrossMsg dummy;
			g_world->recv(boost::mpi::any_source, 0, dummy);
		}
		while (g_world->iprobe(boost::mpi::any_source, 2)) {
			BestSolutionMsg dummy;
			g_world->recv(boost::mpi::any_source, 2, dummy);
		}
		const MortonND map1(dim, peanoLevels, low.data(), high.data(), g_mc);
		std::vector<IntervalND*, boost::alignment::aligned_allocator<IntervalND*, 16u>> H_fine;
		std::vector<float, boost::alignment::aligned_allocator<float, 16u>> bestQ_fine = bestQ;
		float bestF_fine = bestF, bx_fine = bx, by_fine = by;
		size_t oi_fine = 0u;
		float oe_fine = 0.0f;

		float M_prior_fine = fmaf(base_M_prior_factor,
			ldexpf(1.0f, -peanoLevels),
			0.0f);

		HoaraSort(H_coarse.begin(), H_coarse.end() - 1);
		const float inv_dim = 1.0f / static_cast<float>(dim + 1);
		size_t ci = static_cast<size_t>(fmaf(static_cast<float>(H_coarse.size()), fmaf(fmaf(inv_dim, fmaf(inv_dim, fmaf(inv_dim, fmaf(inv_dim, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), 1.0f, -0.7f), 0.0f));
		while (ci < H_coarse.size()) {
			const IntervalND* const C = H_coarse[ci];
			alignas(16) float q1[32], q2[32];
			float x1, y1, x2, y2;
			map1.map01ToPoint(C->x1, q1);
			const float f1 = cost(q1, x1, y1);
			map1.map01ToPoint(C->x2, q2);
			const float f2 = cost(q2, x2, y2);
			IntervalND* const I = new IntervalND(C->x1, C->x2, f1, f2);
			I->i1 = static_cast<unsigned long long>(fmaf(C->x1, static_cast<float>(map1.scale), 0.0f));
			I->i2 = static_cast<unsigned long long>(fmaf(C->x2, static_cast<float>(map1.scale), 0.0f));
			I->diam = map1.block_diameter(I->i1, I->i2);
			I->set_metric(I->diam);
			H_fine.emplace_back(I);
			if (f1 < bestF_fine) {
				bestF_fine = f1;
				bestQ_fine.assign(q1, q1 + dim);
				bx_fine = x1;
				by_fine = y1;
			}
			if (f2 < bestF_fine) {
				bestF_fine = f2;
				bestQ_fine.assign(q2, q2 + dim);
				bx_fine = x2;
				by_fine = y2;
			}
			++ci;
		}
		std::make_heap(H_fine.begin(), H_fine.end(), ComparePtrND);
		agp_run_branch_mpi(map1, cost, fmaf(static_cast<float>(maxIterPerBranch), 1.0f, -static_cast<float>(maxIter0)), r, adaptiveMode, epsilon, seed,
			H_fine, bestQ_fine, bestF_fine, bx_fine, by_fine, oi_fine, oe_fine, M_prior_fine);

		total_oi += oi_fine;
		total_oe = oe_fine;

		if (bestF_fine < bestF) {
			bestF = bestF_fine;
			bestQ = std::move(bestQ_fine);
			bx = bx_fine;
			by = by_fine;
		}
	}

	BestSolutionMsg best;
	best.bestF = bestF;
	best.bestX = bx;
	best.bestY = by;
	best.dim = static_cast<unsigned>(bestQ.size());
	memcpy(best.bestQ, bestQ.data(), static_cast<size_t>(best.dim) * sizeof(float));

	const size_t iterations = std::bit_width(static_cast<size_t>(world - 1));
	bool active = true;

	size_t itx = 0;
	while (itx < iterations && active) {
		const size_t step = 1ULL << itx;
		const int partner = rank ^ static_cast<int>(step);

		if (partner < world) {
			const bool am_sender = (rank & static_cast<int>(step)) != 0;
			if (am_sender) {
				g_world->isend(partner, 3, best);
				active = false;
			}
			else {
				BestSolutionMsg in;
				g_world->recv(partner, 3, in);
				if (in.bestF < best.bestF) best = in;
			}
		}
		++itx;
	}

	if (rank == 0) {
		*out_bestQLen = static_cast<size_t>(best.dim);
		*out_bestQ = static_cast<float*>(CoTaskMemAlloc(sizeof(float) * (*out_bestQLen)));
		memcpy(*out_bestQ, best.bestQ, sizeof(float) * (*out_bestQLen));
		*out_bestX = best.bestX;
		*out_bestY = best.bestY;
		*out_bestF = best.bestF;
		*out_iterations = total_oi;
		*out_achieved_epsilon = total_oe;
	}
}

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline int AgpInit(const int peanoLevel, const float a, const float b, const float c, const float d) noexcept {
	g_env = new boost::mpi::environment();
	g_world = new boost::mpi::communicator();
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	const int rank = g_world->rank();
	const int world_size = g_world->size();
	if (world_size == 4) {
		new (&gActiveMap) Peano2DMap(peanoLevel, a, b, c, d, rank & 3);
	}
	g_mc.baseSeed = fmaf(0x9E3779B9u, static_cast<float>(rank), 0x9E3779B9u);
	if (rank == 0) {
		wchar_t buf[MAX_PATH]{};
		GetModuleFileNameW(nullptr, buf, MAX_PATH);
		std::wstring ws(buf);
		auto pos = ws.find_last_of(L"\\/");
		if (pos != std::wstring::npos) ws.resize(pos);
		int n = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, nullptr, 0, nullptr, nullptr);
		g_exeDirCache.resize(n, '\0');
		WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, g_exeDirCache.data(), n, nullptr, nullptr);
		if (!g_exeDirCache.empty() && g_exeDirCache.back() == '\0') g_exeDirCache.pop_back();

		g_pyInterpreter = new pybind11::scoped_interpreter{};

		pybind11::module_ sys = pybind11::module_::import("sys");
		pybind11::list path = sys.attr("path");
		const std::string& exeDir = g_exeDirCache;

		path.attr("insert")(0, pybind11::str(exeDir + "\\env\\Lib\\site-packages"));
		path.attr("insert")(0, pybind11::str(exeDir + "\\env\\Scripts"));
		path.attr("insert")(0, pybind11::str(exeDir + "\\env"));
		path.attr("append")(pybind11::str(exeDir));

		pybind11::module_::import("warnings").attr("filterwarnings")("ignore");
		g_pyOptimizerBridge = new pybind11::module_(pybind11::module_::import("optimizer_bridge"));
	}
	return rank;
}

static __declspec(noalias) __forceinline float ShekelFunc(const float x, const float seed) noexcept {
	int i = 0;
	float st = seed, r1, r2, res = 0.0f;
#pragma loop ivdep
	while (i < 10) {
		XOR_RAND(st, r1);
		const float xp = fmaf(-r1, 10.0f, x);
		XOR_RAND(st, r1);
		XOR_RAND(st, r2);
		float d = fmaf(fmaf(r1, 20.0f, 5.0f), xp * xp, fmaf(r2, 0.2f, 1.0f));
		d = copysignf(fmaxf(fabsf(d), FLT_MIN), d);
		res -= (1.0f / d) * 1.0f;
		++i;
	}
	return res;
}

static __declspec(noalias) __forceinline float RastriginFunc(const float x1, const float x2) noexcept {
	const float t = fmaf(x1, x1, x2 * x2);
	float c1, c2;
	FABE13_COS(6.28318530717958647692f * x1, c1);
	FABE13_COS(6.28318530717958647692f * x2, c2);
	return (t - fmaf(c1 + c2, 10.0f, -14.6f)) * fmaf(-t, 0.25f, 18.42f);
}

static __declspec(noalias) __forceinline float HillFunc(const float x, const float seed) noexcept {
	int j = 0;
	__declspec(align(16)) float ang[14u];
	const float st_ang = 6.28318530717958647692f * x;
	while (j < 14) {
		ang[j] = st_ang * static_cast<float>(j + 1);
		++j;
	}
	__declspec(align(16)) float sv[14u], cv[14u];
	FABE13_SINCOS(ang, sv, cv, 14u);
	float state = seed, r1, r2;
	XOR_RAND(state, r1);
	float res = fmaf(r1, 2.0f, -1.1f);
	--j;
#pragma loop ivdep
	while (j >= 0) {
		XOR_RAND(state, r1);
		XOR_RAND(state, r2);
		res += fmaf(fmaf(r1, 2.0f, -1.1f), sv[j], fmaf(r2, 2.0f, -1.1f) * cv[j]);
		--j;
	}
	return res;
}

static __declspec(noalias) __forceinline float GrishaginFunc(const float x1, const float x2, const float seed) noexcept {
	int j = 0;
	__declspec(align(16)) float aj[8u], ak[8u];
#pragma loop ivdep
	while (j < 8) {
		const float pj = 3.14159265358979323846f * static_cast<float>(j + 1);
		aj[j] = pj * x1;
		ak[j] = pj * x2;
		++j;
	}
	__declspec(align(16)) float sj[8u], cj[8u], sk[8u], ck[8u];
	FABE13_SINCOS(aj, sj, cj, 8u);
	FABE13_SINCOS(ak, sk, ck, 8u);
	--j;
	float p1 = 0.0f, p2 = 0.0f;
	float st = seed, r1, r2;
#pragma loop ivdep
	while (j >= 0) {
		size_t k2 = 0u;
		while (k2 < 8u) {
			const float s = sj[j] * sj[j];
			const float c = ck[k2] * ck[k2];
			XOR_RAND_GRSH(st, r1);
			XOR_RAND_GRSH(st, r2);
			p1 = fmaf(r1, s, fmaf(r2, c, p1));
			XOR_RAND_GRSH(st, r1);
			XOR_RAND_GRSH(st, r2);
			p2 = fmaf(-r1, c, fmaf(r2, s, p2));
			++k2;
		}
		--j;
	}
	return -sqrtf(fmaf(p1, p1, p2 * p2));
}

extern "C" __declspec(dllexport) __declspec(noalias)
void AGP_1D(const float global_iterations, const float a, const float b, const float r, const bool mode, const float epsilon, const float seed,
	float** const out_data, size_t* const out_len) noexcept {
	Slab* const __restrict slab = tls.local();
	slab->current = slab->base;
	int counter = 0;
	const float initial_length = b - a;
	float dmax = initial_length;
	const float threshold_03 = 0.3f * initial_length, inv_threshold_03 = 1.0f / threshold_03;
	const float start_val = ShekelFunc(a, seed);
	float best_f = ShekelFunc(b, seed);
	float x_Rmax_1 = a, x_Rmax_2 = b;
	float y_Rmax_1 = start_val, y_Rmax_2 = best_f;
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>> Extr;
	std::vector<Interval1D*, boost::alignment::aligned_allocator<Interval1D*, 16u>> R;
	Extr.reserve(static_cast<size_t>(global_iterations) << 2u);
	R.reserve(static_cast<size_t>(global_iterations) << 1u);
	R.emplace_back(new Interval1D(a, b, start_val, best_f, 1.0f));
	float Mmax = R.front()->M;
	float m = r * Mmax;

	while (true) {
		const float new_point = step(m, x_Rmax_1, x_Rmax_2, y_Rmax_1, y_Rmax_2, 1.0f, r);
		const float new_value = ShekelFunc(new_point, seed);
		if (new_value < best_f) {
			best_f = new_value;
			Extr.emplace_back(best_f);
			Extr.emplace_back(new_point);
		}
		std::pop_heap(R.begin(), R.end(), ComparePtr1D);
		const Interval1D* const pro = R.back();
		const float new_x1 = pro->x1, new_x2 = pro->x2;
		const float len2 = new_x2 - new_point, len1 = new_point - new_x1;
		const float interval_len = (len1 < len2 ? len1 : len2);
		if (++counter == static_cast<int>(global_iterations) || interval_len < epsilon) {
			Extr.emplace_back(static_cast<float>(counter));
			Extr.emplace_back(interval_len);
			*out_len = Extr.size();
			*out_data = static_cast<float*>(CoTaskMemAlloc(sizeof(float) * (*out_len)));
			memcpy(*out_data, Extr.data(), sizeof(float) * (*out_len));
			return;
		}
		Interval1D* const curr = new Interval1D(new_x1, new_point, pro->y1, new_value, 1.0f);
		Interval1D* const curr1 = new Interval1D(new_point, new_x2, new_value, pro->y2, 1.0f);
		const float currM = curr->M > curr1->M ? curr->M : curr1->M;
		const size_t r_size = R.size();
		if (mode) {
			if (len2 + len1 == dmax) {
				dmax = len2 > len1 ? len2 : len1;
				for (auto p : R) {
					const float L = p->x2 - p->x1;
					if (L > dmax) dmax = L;
				}
			}
			if (threshold_03 > dmax && !(counter % 3) || 10.0f * dmax < initial_length) {
				if (currM > Mmax) {
					Mmax = currM;
					m = r * Mmax;
				}
				const float progress = fmaf(-inv_threshold_03, dmax, 1.0f);
				const float alpha = progress * progress;
				const float betta = 2.0f - alpha;
				const float MULT = (1.0f / dmax) * Mmax;
				const float global_coeff = fmaf(MULT, r, -MULT);
				const float GF = betta * global_coeff;
				curr->ChangeCharacteristic(fmaf(GF, len1, curr->M * alpha));
				curr1->ChangeCharacteristic(fmaf(GF, len2, curr1->M * alpha));
				RecomputeR_AffineM_AVX2_1D(R.data(), r_size, GF, alpha);
				std::make_heap(R.begin(), R.end(), ComparePtr1D);
			}
			else {
				if (currM > Mmax) {
					if (currM < 1.15f * Mmax) {
						Mmax = currM;
						m = r * Mmax;
						curr->ChangeCharacteristic(m);
						curr1->ChangeCharacteristic(m);
					}
					else {
						Mmax = currM;
						m = r * Mmax;
						curr->ChangeCharacteristic(m);
						curr1->ChangeCharacteristic(m);
						RecomputeR_ConstM_AVX2_1D(R.data(), r_size, m);
						std::make_heap(R.begin(), R.end(), ComparePtr1D);
					}
				}
				else {
					curr->ChangeCharacteristic(m);
					curr1->ChangeCharacteristic(m);
				}
			}
		}
		else {
			if (currM > Mmax) {
				if (currM < 1.15f * Mmax) {
					Mmax = currM;
					m = r * Mmax;
					curr->ChangeCharacteristic(m);
					curr1->ChangeCharacteristic(m);
				}
				else {
					Mmax = currM;
					m = r * Mmax;
					curr->ChangeCharacteristic(m);
					curr1->ChangeCharacteristic(m);
					RecomputeR_ConstM_AVX2_1D(R.data(), r_size, m);
					std::make_heap(R.begin(), R.end(), ComparePtr1D);
				}
			}
			else {
				curr->ChangeCharacteristic(m);
				curr1->ChangeCharacteristic(m);
			}
		}
		R.back() = curr;
		std::push_heap(R.begin(), R.end(), ComparePtr1D);
		R.emplace_back(curr1);
		std::push_heap(R.begin(), R.end(), ComparePtr1D);
		const Interval1D* const top = R.front();
		x_Rmax_1 = top->x1;
		x_Rmax_2 = top->x2;
		y_Rmax_1 = top->y1;
		y_Rmax_2 = top->y2;
	}
}

extern "C" __declspec(dllexport) __declspec(noalias)
void AGP_2D(const float N, const float global_iterations, const float a, const float b, const float c,
	const float d, const float r, const bool mode, const float epsilon, const float seed,
	float** const out_data, size_t* const out_len) noexcept {
	Slab* const __restrict slab = tls.local();
	slab->current = slab->base;
	int counter = 0, no_improve = 0;
	const int rank = g_world->rank();
	const int world_size = g_world->size();
	while (g_world->iprobe(boost::mpi::any_source, 0)) {
		MultiCrossMsg dummy;
		g_world->recv(boost::mpi::any_source, 0, dummy);
	}
	const float inv_divider = ldexpf(1.0f, -((gActiveMap.levels << 1) + 1));
	const float x_addition = (b - a) * inv_divider, y_addition = (d - c) * inv_divider;
	const float true_start = a + x_addition, true_end = b - x_addition;
	float x_Rmax_1 = true_start, x_Rmax_2 = true_end;
	const float initial_length = x_Rmax_2 - x_Rmax_1;
	float dmax = initial_length;
	const float threshold_03 = 0.3f * initial_length, inv_threshold_03 = 1.0f / threshold_03;
	const float start_val = rank % 3 ? RastriginFunc(true_end, d - y_addition) : RastriginFunc(true_start, c + y_addition);
	float best_f = rank % 2 ? RastriginFunc(true_start, d - y_addition) : RastriginFunc(true_end, c + y_addition);
	float y_Rmax_1 = start_val, y_Rmax_2 = best_f;
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>> Extr;
	std::vector<Interval1D* __restrict, boost::alignment::aligned_allocator<Interval1D* __restrict, 16u>> R;
	Extr.clear();
	Extr.reserve(static_cast<size_t>(global_iterations) << 2u);
	R.clear();
	R.reserve(static_cast<size_t>(global_iterations) << 1u);
	R.emplace_back(new Interval1D(true_start, true_end, start_val, best_f, 2.0f));
	const Interval1D* __restrict top_ptr;
	float Mmax = R.front()->M, m = r * Mmax;
	while (true) {
		const float interval_len = x_Rmax_2 - x_Rmax_1;
		const bool stagnation = no_improve > 100 && counter > 270;
		const float p = fmaf(-1.0f / initial_length, dmax, 1.0f);
		while (g_world->iprobe(boost::mpi::any_source, 0)) {
			MultiCrossMsg in;
			g_world->recv(boost::mpi::any_source, 0, in);
			const MultiCrossMsg& mX = in;
			unsigned ii = 0;
			while (ii < mX.count) {
				const float* const d2 = &mX.intervals[ii * 5];
				float sx = d2[0], ex = d2[2];
				if (ex > sx) {
					Interval1D* const __restrict injected = new Interval1D(sx, ex,
						RastriginFunc(d2[0], d2[1]), RastriginFunc(d2[2], d2[3]), 2.0f);
					injected->ChangeCharacteristic(m);
					if (injected->R > 1.15f * top_ptr->R) {
						const float k = stagnation ? fmaf(0.5819767068693265f, (fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f) - 1.0f), 0.3f) : fmaf(0.3491860241215959f, (fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f) - 1.0f), 0.6f);
						injected->R = d2[4] * k;
						R.emplace_back(injected);
						std::push_heap(R.begin(), R.end(), ComparePtr1D);
					}
				}
				++ii;
			}
		}
		const int T = static_cast<int>(fmaf(-(fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f) - 1.0f), 264.0f, 277.0f));
		const bool want_term = interval_len < epsilon || counter == static_cast<int>(global_iterations);
		if (!(++counter % T) || stagnation) {
			if (!want_term) {
				MultiCrossMsg out;
				float s_x1, s_x2, e_x1, e_x2;
				HitTest2D_analytic(top_ptr->x1, s_x1, s_x2);
				HitTest2D_analytic(top_ptr->x2, e_x1, e_x2);
				out.intervals[0] = s_x1;
				out.intervals[1] = s_x2;
				out.intervals[2] = e_x1;
				out.intervals[3] = e_x2;
				out.intervals[4] = top_ptr->R;
				out.count = 1;
				int i2 = 0;
				while (i2 < world_size) {
					if (i2 != rank) g_world->isend(i2, 0, out);
					++i2;
				}
			}
		}
		if (want_term) {
			if (!rank) {
				Extr.emplace_back(static_cast<float>(counter));
				Extr.emplace_back(interval_len);
				*out_len = Extr.size();
				*out_data = reinterpret_cast<float* __restrict>(CoTaskMemAlloc(sizeof(float) * (*out_len)));
				memcpy(*out_data, Extr.data(), sizeof(float) * (*out_len));
			}
			return;
		}
		const float new_point = step(m, x_Rmax_1, x_Rmax_2, y_Rmax_1, y_Rmax_2, 2.0f, r);
		float new_x1_val, new_x2_val;
		HitTest2D_analytic(new_point, new_x1_val, new_x2_val);
		const float new_value = RastriginFunc(new_x1_val, new_x2_val);
		if (new_value < best_f) {
			best_f = new_value;
			Extr.emplace_back(best_f);
			Extr.emplace_back(new_x1_val);
			Extr.emplace_back(new_x2_val);
			no_improve = 0;
		}
		else {
			++no_improve;
		}
		std::pop_heap(R.begin(), R.end(), ComparePtr1D);
		Interval1D* const __restrict intermediate = R.back();
		const float segment_x1 = intermediate->x1, segment_x2 = intermediate->x2;
		const float len2 = segment_x2 - new_point, len1 = new_point - segment_x1;
		Interval1D* const __restrict curr = new Interval1D(segment_x1, new_point, intermediate->y1, new_value, 2.0f);
		Interval1D* const __restrict curr1 = new Interval1D(new_point, segment_x2, new_value, intermediate->y2, 2.0f);
		const float currM = (std::max)(curr->M, curr1->M);
		const size_t r_size = R.size();
		if (mode) {
			if (len2 + len1 == dmax) {
				dmax = (std::max)(len1, len2);
				for (auto pI : R) {
					const float L = pI->x2 - pI->x1;
					if (L > dmax) dmax = L;
				}
			}
			if (threshold_03 > dmax && !(counter % 3) || 10.0f * dmax < initial_length) {
				if (currM > Mmax) {
					Mmax = currM;
					m = r * Mmax;
				}
				const float progress = fmaf(-inv_threshold_03, dmax, 1.0f);
				const float alpha = progress * progress;
				const float betta = 2.0f - alpha;
				const float MULTIPLIER = (1.0f / dmax) * Mmax;
				const float global_coeff = fmaf(MULTIPLIER, r, -MULTIPLIER);
				const float GLOBAL_FACTOR = betta * global_coeff;
				curr->ChangeCharacteristic(fmaf(GLOBAL_FACTOR, len1, curr->M * alpha));
				curr1->ChangeCharacteristic(fmaf(GLOBAL_FACTOR, len2, curr1->M * alpha));
				RecomputeR_AffineM_AVX2_1D(R.data(), r_size, GLOBAL_FACTOR, alpha);
				std::make_heap(R.begin(), R.end(), ComparePtr1D);
			}
			else {
				if (currM > Mmax) {
					if (currM < 1.15f * Mmax) {
						Mmax = currM;
						m = r * Mmax;
						curr->ChangeCharacteristic(m);
						curr1->ChangeCharacteristic(m);
					}
					else {
						Mmax = currM;
						m = r * Mmax;
						curr->ChangeCharacteristic(m);
						curr1->ChangeCharacteristic(m);
						RecomputeR_ConstM_AVX2_1D(R.data(), r_size, m);
						std::make_heap(R.begin(), R.end(), ComparePtr1D);
					}
				}
				else {
					curr->ChangeCharacteristic(m);
					curr1->ChangeCharacteristic(m);
				}
			}
		}
		else {
			if (currM > Mmax) {
				if (currM < 1.15f * Mmax) {
					Mmax = currM;
					m = r * Mmax;
					curr->ChangeCharacteristic(m);
					curr1->ChangeCharacteristic(m);
				}
				else {
					Mmax = currM;
					m = r * Mmax;
					curr->ChangeCharacteristic(m);
					curr1->ChangeCharacteristic(m);
					RecomputeR_ConstM_AVX2_1D(R.data(), r_size, m);
					std::make_heap(R.begin(), R.end(), ComparePtr1D);
				}
			}
			else {
				curr->ChangeCharacteristic(m);
				curr1->ChangeCharacteristic(m);
			}
		}
		R.back() = curr;
		std::push_heap(R.begin(), R.end(), ComparePtr1D);
		R.emplace_back(curr1);
		std::push_heap(R.begin(), R.end(), ComparePtr1D);
		top_ptr = R.front();
		x_Rmax_1 = top_ptr->x1;
		x_Rmax_2 = top_ptr->x2;
		y_Rmax_1 = top_ptr->y1;
		y_Rmax_2 = top_ptr->y2;
	}
}

__declspec(align(16)) struct RunParams final sealed{
		int nSegments;
		unsigned varLen;
		float minTheta;
		float tx, ty;
		int levels;
		int maxIter;
		float r;
		unsigned adaptive;
		float eps;
		unsigned seed;
		float baseLength;
		float stretchFactor;

		template<typename Archive>
		void serialize(Archive& ar, const unsigned int) {
				ar& nSegments& varLen& minTheta& tx& ty
						& levels& maxIter& r& adaptive& eps& seed
						& baseLength& stretchFactor;
		}
};

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline
void AgpStartManipND(const int nSegments, const bool variableLengths, const float minTheta,
	const float targetX, const float targetY, const int peanoLevels,
	const int maxIterPerBranch, const float r, const bool adaptiveMode,
	const float epsilon, const unsigned int seed,
	const float baseLength, const float stretchFactor) noexcept {
	RunParams p;
	p.nSegments = nSegments;
	p.varLen = static_cast<unsigned>(variableLengths);
	p.minTheta = minTheta;
	p.tx = targetX;
	p.ty = targetY;
	p.levels = peanoLevels;
	p.maxIter = maxIterPerBranch;
	p.r = r;
	p.adaptive = static_cast<unsigned>(adaptiveMode);
	p.eps = epsilon;
	p.seed = seed;
	p.baseLength = baseLength;
	p.stretchFactor = stretchFactor;
	int i = 1;
	const int world = g_world->size();
	while (i < world) {
		g_world->isend(i, 1, p);
		++i;
	}
}

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline void AgpWaitStartAndRun() noexcept {
	RunParams p;
	float* __restrict q;
	size_t qlen;
	float bx, by, bf;
	size_t oi;
	float oa;
	while (true) {
		if (g_world->iprobe(0, 1)) {
			g_world->recv(0, 1, p);
			AGP_Manip2D(p.nSegments, static_cast<bool>(p.varLen), p.minTheta, p.tx, p.ty,
				p.levels, p.maxIter, p.r, static_cast<bool>(p.adaptive), p.eps, p.seed,
				p.baseLength, p.stretchFactor,
				&q, &qlen, &bx, &by, &bf, &oi, &oa);
		}
		Sleep(0);
	}
}

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline void AgpWaitStartAndRun2D() noexcept {
	int dummy;
	float* __restrict buf;
	size_t len;
	while (true) {
		if (g_world->iprobe(0, 1)) {
			g_world->recv(0, 1, dummy);
			AGP_2D(2.0f, 10000.0f, -2.2f, 1.8f, -2.2f, 1.8f, 2.5f, false, 0.00001f, static_cast<float>(GetTickCount()), &buf, &len);
		}
		Sleep(0);
	}
}

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline void AgpStartWorkers() noexcept {
	int i = 1;
	const int world = g_world->size();
	while (i < world) {
		g_world->isend(i, 1, 0);
		++i;
	}
}

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline void AGP_Free(float* const p) noexcept {
	CoTaskMemFree(p);
}

extern "C" __declspec(dllexport) __declspec(noalias)
void RunPythonOptimization(const char* const backend,
	const int n_seg,
	const bool var_len,
	const float min_theta,
	const float tx,
	const float ty,
	const int levels,
	const int max_iter,
	const float r_param,
	const float eps,
	const bool adaptive,
	const float baseLength,
	const float stretchFactor,
	float* const __restrict out_bestF,
	float* const __restrict out_bestX,
	float* const __restrict out_bestY,
	float* const __restrict out_angles,
	float* const __restrict out_lengths,
	int* const __restrict out_iterations,
	float* const __restrict out_eps,
	float* const __restrict out_micros) noexcept {
	pybind11::gil_scoped_acquire gil;
	pybind11::dict result;

	if (backend && strcmp(backend, "optuna") == 0) {
		result = (*g_pyOptimizerBridge).attr("run_optuna")(n_seg,
			var_len,
			min_theta,
			tx,
			ty,
			max_iter,
			baseLength,
			stretchFactor);
	}
	else {
		result = (*g_pyOptimizerBridge).attr("run_iopt")(n_seg,
			var_len,
			min_theta,
			tx,
			ty,
			levels,
			max_iter,
			r_param,
			eps,
			adaptive,
			baseLength,
			stretchFactor);
	}

	*out_bestF = result["BEST_F"].cast<float>();
	*out_bestX = result["BEST_X"].cast<float>();
	*out_bestY = result["BEST_Y"].cast<float>();
	*out_iterations = result["ITERATIONS"].cast<int>();
	*out_eps = result["EPS"].cast<float>();
	*out_micros = result["TIME"].cast<float>();

	std::vector<float> angles_vec = result["ANGLES"].cast<std::vector<float>>();
	std::vector<float> lengths_vec = result["LENGTHS"].cast<std::vector<float>>();

	const size_t n = static_cast<size_t>(n_seg);

	if (out_angles) {
		const size_t m = angles_vec.size();
		const size_t limit = m < n ? m : n;
		const float* const __restrict src = angles_vec.data();
		size_t i = 0;
#pragma loop ivdep
		while (i < limit) {
			out_angles[i] = src[i];
			++i;
		}
	}

	if (out_lengths) {
		if (var_len) {
			const size_t mL = lengths_vec.size();
			const size_t limitL = mL < n ? mL : n;
			const float* const __restrict srcL = lengths_vec.data();
			size_t j = 0;
#pragma loop ivdep
			while (j < limitL) {
				out_lengths[j] = srcL[j];
				++j;
			}
		}
		else {
			const float one = 1.0f;
			size_t j = 0;
#pragma loop ivdep
			while (j < n) {
				out_lengths[j] = one;
				++j;
			}
		}
	}
}
