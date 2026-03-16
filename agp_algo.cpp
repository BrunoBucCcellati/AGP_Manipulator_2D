#include "pch.h"

#define XOR_RAND(state, result_var)                                            \
    do {                                                                       \
        unsigned s = state;                                                    \
        s ^= s << 13;                                                          \
        s ^= s >> 17;                                                          \
        s ^= s << 5;                                                           \
        state = s;                                                             \
        float tmp = static_cast<float>(static_cast<double>(s) *                \
                                       (1.0 / 4294967296.0));                  \
        result_var = tmp;                                                      \
    } while (0)

#define XOR_RAND_GRSH(state, result_var)                                       \
    do {                                                                       \
        unsigned s = state;                                                    \
        s ^= s << 13;                                                          \
        s ^= s >> 17;                                                          \
        s ^= s << 5;                                                           \
        state = s;                                                             \
        result_var =                                                           \
            fmaf(static_cast<float>(static_cast<int>(s)), 0x1.0p-31f, -1.0f);  \
    } while (0)

#define FABE13_COS(x, result_var)                                              \
    do {                                                                       \
        const float _ax_ = fabsf(x);                                           \
        float _r_ = fmodf(_ax_, 6.28318530717958647692f);                      \
        if (_r_ > 3.14159265359f)                                              \
            _r_ = 6.28318530717958647692f - _r_;                               \
        if (_r_ < 1.57079632679f) {                                            \
            const float _t2_ = _r_ * _r_;                                      \
            const float _t4_ = _t2_ * _t2_;                                    \
            result_var =                                                       \
                fmaf(_t4_,                                                     \
                     fmaf(_t2_, -0.0013888889f, 0.0416666667f),                \
                     fmaf(_t2_, -0.5f, 1.0f));                                 \
        } else {                                                               \
            _r_ = 3.14159265359f - _r_;                                        \
            const float _t2_ = _r_ * _r_;                                      \
            const float _t4_ = _t2_ * _t2_;                                    \
            result_var =                                                       \
                -fmaf(_t4_,                                                    \
                      fmaf(_t2_, -0.0013888889f, 0.041666667f),                \
                      fmaf(_t2_, -0.5f, 1.0f));                                \
        }                                                                      \
    } while (0)

#define FABE13_SIN(x, result_var)                                              \
    do {                                                                       \
        const float _x_ = x;                                                   \
        const float _ax_ = fabsf(_x_);                                         \
        float _r_ = fmodf(_ax_, 6.28318530717958647692f);                      \
        bool _sfl_ = _r_ > 3.14159265359f;                                     \
        if (_sfl_)                                                             \
            _r_ = 6.28318530717958647692f - _r_;                               \
        bool _cfl_ = _r_ > 1.57079632679f;                                     \
        if (_cfl_)                                                             \
            _r_ = 3.14159265359f - _r_;                                        \
        const float _t2_ = _r_ * _r_;                                          \
        float _s = fmaf(_t2_,                                                  \
                        fmaf(_t2_,                                             \
                             fmaf(_t2_, -0.0001984127f, 0.0083333333f),        \
                             -0.16666666f),                                    \
                        1.0f) *                                                \
                  _r_;                                                         \
        result_var = ((_x_ < 0.0f) ^ _sfl_) ? -_s : _s;                        \
    } while (0)

#define FABE13_SINCOS(in, sin_out, cos_out, n)                                 \
    do {                                                                       \
        int i = 0;                                                             \
        const int limit = n & ~7;                                              \
        if (n >= 8) {                                                          \
            static __declspec(align(16)) const __m256 VEC_TWOPI =              \
                _mm256_set1_ps(6.28318530717958647692f);                       \
            static __declspec(align(16)) const __m256 VEC_PI =                 \
                _mm256_set1_ps(3.14159265359f);                                \
            static __declspec(align(16)) const __m256 VEC_PI_2 =               \
                _mm256_set1_ps(1.57079632679f);                                \
            static __declspec(align(16)) const __m256 INV_TWOPI =              \
                _mm256_set1_ps(0.15915494309189535f);                          \
            static __declspec(align(16)) const __m256 BIAS =                   \
                _mm256_set1_ps(12582912.0f);                                   \
            static __declspec(align(16)) const __m256 VEC_COS_P5 =             \
                _mm256_set1_ps(-0.0013888889f);                                \
            static __declspec(align(16)) const __m256 VEC_COS_P3 =             \
                _mm256_set1_ps(0.0416666667f);                                 \
            static __declspec(align(16)) const __m256 VEC_COS_P1 =             \
                _mm256_set1_ps(-0.5f);                                         \
            static __declspec(align(16)) const __m256 VEC_COS_P0 =             \
                _mm256_set1_ps(1.0f);                                          \
            static __declspec(align(16)) const __m256 VEC_SIN_P5 =             \
                _mm256_set1_ps(-0.0001984127f);                                \
            static __declspec(align(16)) const __m256 VEC_SIN_P3 =             \
                _mm256_set1_ps(0.0083333333f);                                 \
            static __declspec(align(16)) const __m256 VEC_SIN_P1 =             \
                _mm256_set1_ps(-0.16666666f);                                  \
            static __declspec(align(16)) const __m256 VEC_SIN_P0 =             \
                _mm256_set1_ps(1.0f);                                          \
            static __declspec(align(16)) const __m256 VEC_ZERO =               \
                _mm256_setzero_ps();                                           \
            while (i < limit) {                                                \
                const __m256 vx = _mm256_load_ps(&(in)[i]);                    \
                const __m256 vax = _mm256_andnot_ps(_mm256_set1_ps(-0.0f),     \
                                                    vx);                       \
                __m256 q = _mm256_fmadd_ps(vax, INV_TWOPI, BIAS);              \
                q = _mm256_sub_ps(q, BIAS);                                    \
                const __m256 r = _mm256_fnmadd_ps(VEC_TWOPI, q, vax);          \
                const __m256 r1 =                                              \
                    _mm256_min_ps(r, _mm256_sub_ps(VEC_TWOPI, r));             \
                const __m256 r2 =                                              \
                    _mm256_min_ps(r1, _mm256_sub_ps(VEC_PI, r1));              \
                const __m256 t2 = _mm256_mul_ps(r2, r2);                       \
                const __m256 cosv =                                            \
                    _mm256_fmadd_ps(t2,                                        \
                                    _mm256_fmadd_ps(t2,                        \
                                                     _mm256_fmadd_ps(t2,       \
                                                                     VEC_COS_P5, \
                                                                     VEC_COS_P3), \
                                                     VEC_COS_P1),              \
                                    VEC_COS_P0);                               \
                const __m256 sinv =                                            \
                    _mm256_mul_ps(_mm256_fmadd_ps(t2,                          \
                                                   _mm256_fmadd_ps(t2,         \
                                                                   _mm256_fmadd_ps(t2, \
                                                                                   VEC_SIN_P5, \
                                                                                   VEC_SIN_P3), \
                                                                   VEC_SIN_P1), \
                                                   VEC_SIN_P0),                \
                                  r2);                                         \
                const __m256 cflip =                                           \
                    _mm256_cmp_ps(r1, VEC_PI_2, _CMP_GT_OQ);                   \
                const __m256 sflip =                                           \
                    _mm256_xor_ps(_mm256_cmp_ps(vx, VEC_ZERO, _CMP_LT_OQ),     \
                                  _mm256_cmp_ps(r, VEC_PI, _CMP_GT_OQ));       \
                _mm256_store_ps(&(cos_out)[i],                                 \
                                _mm256_blendv_ps(cosv,                         \
                                                  _mm256_sub_ps(VEC_ZERO,      \
                                                                cosv),         \
                                                  cflip));                     \
                _mm256_store_ps(&(sin_out)[i],                                 \
                                _mm256_blendv_ps(sinv,                         \
                                                  _mm256_sub_ps(VEC_ZERO,      \
                                                                sinv),         \
                                                  sflip));                     \
                i += 8;                                                        \
            }                                                                  \
        }                                                                      \
        while (i < n) {                                                        \
            const float x = (in)[i];                                           \
            const float ax = fabsf(x);                                         \
            float q = fmaf(ax, 0.15915494309189535f, 12582912.0f);             \
            q -= 12582912.0f;                                                  \
            float r = fmaf(-6.28318530718f, q, ax);                            \
            const bool sflip = r > 3.14159265359f;                             \
            if (sflip)                                                         \
                r = 6.28318530718f - r;                                        \
            const bool cflip = r > 1.57079632679f;                             \
            if (cflip)                                                         \
                r = 3.14159265359f - r;                                        \
            const float t2 = r * r;                                            \
            const float c = fmaf(t2,                                           \
                                 fmaf(t2,                                      \
                                      fmaf(t2, -0.0013888889f, 0.0416666667f), \
                                      -0.5f),                                  \
                                 1.0f);                                        \
            const float s = fmaf(t2,                                           \
                                 fmaf(t2,                                      \
                                      fmaf(t2, -0.0001984127f, 0.0083333333f), \
                                      -0.16666666f),                           \
                                 1.0f) *                                       \
                           r;                                                  \
            (cos_out)[i] = cflip ? -c : c;                                     \
            (sin_out)[i] = ((x < 0.0f) ^ sflip) ? -s : s;                      \
            ++i;                                                               \
        }                                                                      \
    } while (0)

enum List : unsigned {
	Top = 0b00u,
	Down = 0b01u,
	Left = 0b10u,
	Right = 0b11u
};

__declspec(align(16)) struct Step final {
	const unsigned next;
	const unsigned dx;
	const unsigned dy;

	__forceinline Step(unsigned n, unsigned x, unsigned y) noexcept
		: next(n), dx(x), dy(y) {
	}
};

__declspec(align(16)) struct InvStep final {
	const unsigned q;
	const unsigned next;

	__forceinline InvStep(unsigned q_val, unsigned n) noexcept
		: q(q_val), next(n) {
	}
};

__declspec(align(16)) static const Step g_step_tbl[4][4] = {
		{ Step(Right, 0u, 0u), Step(Top, 0u, 1u), Step(Top, 1u, 1u), Step(Left, 1u, 0u) },
		{ Step(Left, 1u, 1u), Step(Down, 1u, 0u), Step(Down, 0u, 0u), Step(Right, 0u, 1u) },
		{ Step(Down, 1u, 1u), Step(Left, 0u, 1u), Step(Left, 0u, 0u), Step(Top, 1u, 0u) },
		{ Step(Top, 0u, 0u), Step(Right, 1u, 0u), Step(Right, 1u, 1u), Step(Down, 0u, 1u) }
};

__declspec(align(16)) static const InvStep g_inv_tbl[4][4] = {
		{ InvStep(0u, Right), InvStep(1u, Top), InvStep(3u, Left), InvStep(2u, Top) },
		{ InvStep(2u, Down), InvStep(3u, Right), InvStep(1u, Down), InvStep(0u, Left) },
		{ InvStep(2u, Left), InvStep(1u, Left), InvStep(3u, Top), InvStep(0u, Down) },
		{ InvStep(0u, Top), InvStep(3u, Down), InvStep(1u, Right), InvStep(2u, Right) }
};

static const boost::mpi::environment* g_env;
static boost::mpi::communicator* g_world;
static const pybind11::scoped_interpreter* g_pyInterpreter;
static const pybind11::module_* g_pyOptimizerBridge;
static std::string g_exeDirCache;

__declspec(align(16)) struct CrossMsg final {
	float s_x1;
	float s_x2;
	float e_x1;
	float e_x2;
	float Rtop;

	template <typename Archive>
	__declspec(noalias) __forceinline void serialize(Archive& ar, unsigned int) noexcept {
		ar& s_x1& s_x2& e_x1& e_x2& Rtop;
	}
};

__declspec(align(16)) struct MultiCrossMsg final {
	float intervals[35];
	unsigned count;

	template <typename Archive>
	__declspec(noalias) __forceinline void serialize(Archive& ar, unsigned int) noexcept {
		ar& intervals& count;
	}
};

__declspec(align(16)) struct BestSolutionMsg final {
	float bestF;
	float bestX;
	float bestY;
	float bestQ[32];
	unsigned dim;

	template <typename Archive>
	__declspec(noalias) __forceinline void serialize(Archive& ar, unsigned int) noexcept {
		ar& bestF& bestX& bestY& dim& bestQ;
	}
};

struct PendingMultiSend {
	MultiCrossMsg msg;
	boost::mpi::request req;

	PendingMultiSend(boost::mpi::communicator& comm, int partner, const MultiCrossMsg& m)
		: msg(m), req(comm.isend(partner, 0, msg)) {
	}
};

struct PendingBestSend {
	BestSolutionMsg msg;
	boost::mpi::request req;

	PendingBestSend(boost::mpi::communicator& comm, int partner, const BestSolutionMsg& m)
		: msg(m), req(comm.isend(partner, 2, msg)) {
	}
};
static thread_local std::deque<PendingMultiSend> g_pendingMulti;
static thread_local std::deque<PendingBestSend> g_pendingBest;
constexpr size_t MAX_INFLIGHT_MULTI = 32;
constexpr size_t MAX_INFLIGHT_BEST = 8;

__declspec(align(16)) struct Slab final sealed{
		char* const base;
		char* current;
		char* const end;

		__forceinline Slab(void* memory, size_t usable) noexcept
				: base(static_cast<char*>(memory)), current(base),
					end(base + (usable & ~static_cast<size_t>(63u))) {
}
};

static thread_local tbb::enumerable_thread_specific<Slab*> tls([]() noexcept {
	void* memory = _aligned_malloc(16777216u, 16u);
	Slab* slab = static_cast<Slab*>(_aligned_malloc(sizeof(Slab), 16u));
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
		const float a;
		const float b;
		const float c;
		const float d;
		const float lenx;
		const float leny;
		const float inv_lenx;
		const unsigned scale;
		const unsigned start;

		__forceinline Peano2DMap(int L, float _a, float _b, float _c, float _d, unsigned st) noexcept
				: levels(L), a(_a), b(_b), c(_c), d(_d), lenx(_b - _a), leny(_d - _c),
					inv_lenx(1.0f / (_b - _a)),
					scale(static_cast<unsigned>(1u) << (L << 1)), start(st) {
}
};

static Peano2DMap gActiveMap(0, 0, 0, 0, 0, 0);

__declspec(align(16)) struct Interval1D final sealed{
		const float x1;
		const float x2;
		const float y1;
		const float y2;
		const float delta_y;
		const float ordinate_factor;
		const float N_factor;
		const float quadratic_term;
		const float M;
		float R;

		static __declspec(noalias) __forceinline void* operator new(size_t) noexcept {
				Slab* s = tls.local();
				char* r = s->current;
				s->current += 64u;
				return r;
		}

		__declspec(noalias) __forceinline Interval1D(float _x1, float _x2, float _y1, float _y2, float _N) noexcept
				: x1(_x1), x2(_x2), y1(_y1), y2(_y2), delta_y(_y2 - _y1),
					ordinate_factor(-(y1 + y2) * 2.0f),
					N_factor(_N == 1.0f ? _x2 - _x1 : sqrtf(_x2 - _x1)),
					quadratic_term(fmaf((1.0f / N_factor)* delta_y, delta_y, 0.0f)),
					M(fabsf(delta_y) / N_factor) {
}

__declspec(noalias) __forceinline void ChangeCharacteristic(float _m) noexcept {
		R = fmaf(1.0f / _m, quadratic_term, fmaf(_m, N_factor, ordinate_factor));
}
};

static __declspec(noalias) __forceinline bool ComparePtr1D(const Interval1D* a, const Interval1D* b) noexcept {
	return a->R < b->R;
}

static __declspec(noalias) __forceinline void RecomputeR_ConstM_AVX2_1D(Interval1D* const* arr, size_t n, float m) noexcept {
	const __m256 vm = _mm256_set1_ps(m);
	__m256 vinvm = _mm256_rcp_ps(vm);
	vinvm = _mm256_mul_ps(vinvm, _mm256_fnmadd_ps(vm, vinvm, _mm256_set1_ps(2.0f)));
	size_t i = 0;
	size_t limit = n & ~static_cast<size_t>(7ull);
	alignas(16) float q[8];
	alignas(16) float nf[8];
	alignas(16) float od[8];
	alignas(16) float out[8];
#pragma loop ivdep
	while (i < limit) {
		int k = 0;
#pragma loop ivdep
		while (k < 8) {
			const Interval1D* p = arr[i + k];
			q[k] = p->quadratic_term;
			nf[k] = p->N_factor;
			od[k] = p->ordinate_factor;
			++k;
		}
		const __m256 vq = _mm256_load_ps(q);
		const __m256 vnf = _mm256_load_ps(nf);
		const __m256 vod = _mm256_load_ps(od);
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

static __declspec(noalias) __forceinline void RecomputeR_AffineM_AVX2_1D(Interval1D* const* arr, size_t n, float GF, float alpha) noexcept {
	const __m256 vGF = _mm256_set1_ps(GF);
	const __m256 va = _mm256_set1_ps(alpha);
	size_t i = 0;
	size_t limit = n & ~static_cast<size_t>(7ull);
	alignas(16) float ln[8];
	alignas(16) float Mv[8];
	alignas(16) float q[8];
	alignas(16) float nf[8];
	alignas(16) float od[8];
	alignas(16) float out[8];
#pragma loop ivdep
	while (i < limit) {
		int k = 0;
#pragma loop ivdep
		while (k < 8) {
			const Interval1D* p = arr[i + k];
			ln[k] = p->x2 - p->x1;
			Mv[k] = p->M;
			q[k] = p->quadratic_term;
			nf[k] = p->N_factor;
			od[k] = p->ordinate_factor;
			++k;
		}
		const __m256 vln = _mm256_load_ps(ln);
		const __m256 vM = _mm256_load_ps(Mv);
		const __m256 vq = _mm256_load_ps(q);
		const __m256 vnf = _mm256_load_ps(nf);
		const __m256 vod = _mm256_load_ps(od);
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
		const Interval1D* p = arr[i];
		const float mi = fmaf(GF, fmaf(-p->x1, 1.0f, p->x2), fmaf(p->M, alpha, 0.0f));
		arr[i]->R = fmaf(1.0f / mi, p->quadratic_term, fmaf(mi, p->N_factor, p->ordinate_factor));
		++i;
	}
}

__declspec(align(16)) struct IntervalND final sealed{
		const float x1;
		const float x2;
		const float y1;
		const float y2;
		const float delta_y;
		const float ordinate_factor;
		float N_factor;
		float quadratic_term;
		float M;
		float R;
		unsigned long long i1;
		unsigned long long i2;
		float diam;
		int span_level;

		static __declspec(noalias) __forceinline void* operator new(size_t) noexcept {
				Slab* s = tls.local();
				char* r = s->current;
				s->current += 64u;
				return r;
		}

		__declspec(noalias) __forceinline IntervalND(float _x1, float _x2, float _y1, float _y2) noexcept
				: x1(_x1), x2(_x2), y1(_y1), y2(_y2), delta_y(fmaf(_y2, 1.0f, -_y1)),
					ordinate_factor(fmaf(fmaf(-y1, 1.0f, -y2), 2.0f, 0.0f)),
					N_factor(0), quadratic_term(0), M(0), R(0),
					i1(0), i2(0), diam(0), span_level(0) {
}

__declspec(noalias) __forceinline void compute_span_level(const struct MortonND& map) noexcept;
__declspec(noalias) __forceinline void set_metric(float d_alpha) noexcept {
		N_factor = d_alpha;
		quadratic_term = fmaf(1.0f / N_factor, fmaf(delta_y, delta_y, 0.0f), 0.0f);
		M = fmaf(1.0f / N_factor, fabsf(delta_y), 0.0f);
}
__declspec(noalias) __forceinline void ChangeCharacteristic(float _m) noexcept {
		R = fmaf(1.0f / _m, quadratic_term, fmaf(_m, N_factor, ordinate_factor));
}
};

static __declspec(noalias) __forceinline bool ComparePtrND(const IntervalND* a, const IntervalND* b) noexcept {
	return a->R < b->R;
}

static __declspec(noalias) __forceinline void RecomputeR_ConstM_AVX2_ND(IntervalND* const* arr, size_t n, float m) noexcept {
	const __m256 vm = _mm256_set1_ps(m);
	__m256 vinvm = _mm256_rcp_ps(vm);
	vinvm = _mm256_mul_ps(vinvm, _mm256_fnmadd_ps(vm, vinvm, _mm256_set1_ps(2.0f)));
	size_t i = 0;
	size_t limit = n & ~static_cast<size_t>(7ull);
	alignas(16) float q[8];
	alignas(16) float nf[8];
	alignas(16) float od[8];
	alignas(16) float out[8];
#pragma loop ivdep
	while (i < limit) {
		int k = 0;
#pragma loop ivdep
		while (k < 8) {
			const IntervalND* p = arr[i + k];
			q[k] = p->quadratic_term;
			nf[k] = p->N_factor;
			od[k] = p->ordinate_factor;
			++k;
		}
		const __m256 vq = _mm256_load_ps(q);
		const __m256 vnf = _mm256_load_ps(nf);
		const __m256 vod = _mm256_load_ps(od);
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

static __declspec(noalias) __forceinline void RecomputeR_AffineM_AVX2_ND(IntervalND* const* arr, size_t n, float GF, float alpha) noexcept {
	const __m256 vGF = _mm256_set1_ps(GF);
	const __m256 va = _mm256_set1_ps(alpha);
	size_t i = 0;
	size_t limit = n & ~static_cast<size_t>(7ull);
	alignas(16) float ln[8];
	alignas(16) float Mv[8];
	alignas(16) float q[8];
	alignas(16) float nf[8];
	alignas(16) float od[8];
	alignas(16) float out[8];
#pragma loop ivdep
	while (i < limit) {
		int k = 0;
#pragma loop ivdep
		while (k < 8) {
			const IntervalND* p = arr[i + k];
			ln[k] = p->x2 - p->x1;
			Mv[k] = p->M;
			q[k] = p->quadratic_term;
			nf[k] = p->N_factor;
			od[k] = p->ordinate_factor;
			++k;
		}
		const __m256 vln = _mm256_load_ps(ln);
		const __m256 vM = _mm256_load_ps(Mv);
		const __m256 vq = _mm256_load_ps(q);
		const __m256 vnf = _mm256_load_ps(nf);
		const __m256 vod = _mm256_load_ps(od);
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
		const IntervalND* p = arr[i];
		const float mi = fmaf(GF, fmaf(-p->x1, 1.0f, p->x2), fmaf(p->M, alpha, 0.0f));
		arr[i]->R = fmaf(1.0f / mi, p->quadratic_term, fmaf(mi, p->N_factor, p->ordinate_factor));
		++i;
	}
}

static __declspec(noalias) __forceinline float fast_pow_int(const float v, const int n) noexcept {
	float r;
	switch (n) {
	case 3: {
		const float v2 = fmaf(v, v, 0.0f);
		r = fmaf(v2, v, 0.0f);
	} break;
	case 4: {
		const float v2 = fmaf(v, v, 0.0f);
		r = fmaf(v2, v2, 0.0f);
	} break;
	case 5: {
		const float v2 = fmaf(v, v, 0.0f);
		const float v4 = fmaf(v2, v2, 0.0f);
		r = fmaf(fmaf(v4, v2, 0.0f), v, 0.0f);
	} break;
	case 6: {
		const float v2 = fmaf(v, v, 0.0f);
		const float v4 = fmaf(v2, v2, 0.0f);
		r = fmaf(v4, v2, 0.0f);
	} break;
	case 7: {
		const float v2 = fmaf(v, v, 0.0f);
		const float v4 = fmaf(v2, v2, 0.0f);
		r = fmaf(fmaf(v4, v2, 0.0f), v, 0.0f);
	} break;
	case 8: {
		const float v2 = fmaf(v, v, 0.0f);
		const float v4 = fmaf(v2, v2, 0.0f);
		r = fmaf(v4, v4, 0.0f);
	} break;
	case 9: {
		const float v3 = fmaf(fmaf(v, v, 0.0f), v, 0.0f);
		const float v6 = fmaf(v3, v3, 0.0f);
		r = fmaf(v6, v3, 0.0f);
	} break;
	case 10: {
		const float v2 = fmaf(v, v, 0.0f);
		const float v4 = fmaf(v2, v2, 0.0f);
		const float v8 = fmaf(v4, v4, 0.0f);
		r = fmaf(v8, v2, 0.0f);
	} break;
	case 11: {
		const float v2 = fmaf(v, v, 0.0f);
		const float v4 = fmaf(v2, v2, 0.0f);
		const float v8 = fmaf(v4, v4, 0.0f);
		r = fmaf(fmaf(v8, v2, 0.0f), v, 0.0f);
	} break;
	case 12: {
		const float v3 = fmaf(fmaf(v, v, 0.0f), v, 0.0f);
		const float v6 = fmaf(v3, v3, 0.0f);
		r = fmaf(v6, v6, 0.0f);
	} break;
	case 13: {
		const float v3 = fmaf(fmaf(v, v, 0.0f), v, 0.0f);
		const float v6 = fmaf(v3, v3, 0.0f);
		r = fmaf(fmaf(v6, v6, 0.0f), v, 0.0f);
	} break;
	case 14: {
		const float v7 = fmaf(fmaf(fmaf(fmaf(fmaf(fmaf(v, v, 0.0f), v, 0.0f), v, 0.0f), v, 0.0f), v, 0.0f), v, 0.0f);
		r = fmaf(v7, v7, 0.0f);
	} break;
	case 15: {
		const float v7 = fmaf(fmaf(fmaf(fmaf(fmaf(fmaf(v, v, 0.0f), v, 0.0f), v, 0.0f), v, 0.0f), v, 0.0f), v, 0.0f);
		r = fmaf(fmaf(v7, v7, 0.0f), v, 0.0f);
	} break;
	default: {
		const float v2 = fmaf(v, v, 0.0f);
		const float v4 = fmaf(v2, v2, 0.0f);
		const float v8 = fmaf(v4, v4, 0.0f);
		r = fmaf(v8, v8, 0.0f);
	}
	}
	return r;
}

static __declspec(noalias) __forceinline float step(const float _m, const float x1, const float x2, const float y1, const float y2, const float _N, const float _r) noexcept {
	const float diff = fmaf(y2, 1.0f, -y1);
	const unsigned sign_mask = ((*reinterpret_cast<const unsigned*>(&diff)) & 0x80000000u) ^ 0x80000000u;
	const float sign_mult = *reinterpret_cast<const float*>(&sign_mask);
	if (_N == 1.0f)
		return fmaf(fmaf(-(1.0f / _m), diff, x1 + x2), 0.5f, 0.0f);
	if (_N == 2.0f)
		return fmaf(fmaf(fmaf(fmaf(1.0f / fmaf(_m, _m, 0.0f), sign_mult, 0.0f), fmaf(fmaf(diff, diff, 0.0f), _r, 0.0f), 0.0f), 1.0f, fmaf(x1, 1.0f, x2)), 0.5f, 0.0f);
	return fmaf(fmaf(fmaf(fmaf(1.0f / fast_pow_int(_m, static_cast<int>(_N)), sign_mult, 0.0f), fast_pow_int(fabsf(diff), static_cast<int>(_N)), 0.0f), _r, 0.0f), 1.0f, fmaf(x1, 1.0f, x2)) * 0.5f;
}

__declspec(align(16)) struct MortonCachePerRank final sealed{
		std::vector<int> permCache;
		std::vector<unsigned long long> invMaskCache;
		unsigned baseSeed;
};

static thread_local MortonCachePerRank g_mc;

static __declspec(noalias) __forceinline unsigned long long gray_encode(unsigned long long x) noexcept {
	return x ^ (x >> 1);
}

static __declspec(noalias) __forceinline long long gray_decode(unsigned long long g) noexcept {
	g ^= g >> 32;
	g ^= g >> 16;
	g ^= g >> 8;
	g ^= g >> 4;
	g ^= g >> 2;
	g ^= g >> 1;
	return static_cast<long long>(g);
}

__declspec(align(16)) struct MortonND final sealed{
		const int dim;
		const int levels;
		const int eff_levels;
		const int extra_levels;
		const int chunks;
		std::vector<int> chunk_bits;
		std::vector<unsigned long long> chunk_bases;
		unsigned long long scale;
		std::vector<float> low;
		std::vector<float> high;
		std::vector<float> step;
		std::vector<float> invStep;
		std::vector<float> baseOff;
		std::vector<int> perm;
		std::vector<unsigned long long> invMask;
		std::vector<unsigned long long> pextMask;
		std::vector<unsigned long long> pextMaskChunks;
		const float invScaleLevel;
		const bool use_gray;

		static __declspec(noalias) __forceinline unsigned long long make_mask(int dim, int Lc, int d) noexcept {
				unsigned long long m = 0ull;
				unsigned long long bitpos = static_cast<unsigned long long>(d);
				int b = 0;
#pragma loop ivdep
				while (b < Lc) {
						m |= 1ull << bitpos;
						bitpos += static_cast<unsigned long long>(dim);
						++b;
				}
				return m;
		}

		__declspec(noalias) __forceinline MortonND(int D, int L, const float* lows, const float* highs, const MortonCachePerRank& mc)
				: dim(D), levels(L),
					eff_levels((std::max)(1, static_cast<int>(63 / (D ? D : 1)))),
					extra_levels((L > eff_levels) ? (L - eff_levels) : 0),
					chunks((extra_levels > 0) ? (1 + (extra_levels + eff_levels - 1) / eff_levels) : 1),
					low(lows, lows + D), high(highs, highs + D), step(D, 0.0f), invStep(D, 0.0f), baseOff(D, 0.0f),
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
				if (i1 > i2)
						std::swap(i1, i2);
				float s2 = 0.0f;
				int d = 0;
#pragma loop ivdep
				while (d < dim) {
						const int pd = perm[d];
						const unsigned long long varying = (i1 ^ i2) & pextMask[d];
						const int nfree_hi = _mm_popcnt_u64(varying);
						const int nfree_total = nfree_hi + levels - chunk_bits[0];
						const float range = fmaf(step[pd], fmaf(ldexpf(1.0f, nfree_total), 1.0f, -1.0f), 0.0f);
						s2 = fmaf(range, range, s2);
						++d;
				}
				return sqrtf(s2);
		}

		__declspec(noalias) __forceinline void map01ToPoint(float t, float* __restrict out) const noexcept {
				float u = t;
				unsigned long long accBits[32] = {0ull};
				int c = 0;
				while (c < chunks) {
						const int Lc = chunk_bits[c];
						const unsigned long long baseC = chunk_bases[c];
						u *= static_cast<float>(baseC);
						unsigned long long idxc = static_cast<unsigned long long>(u);
						u -= static_cast<float>(idxc);
						if (use_gray)
								idxc = gray_encode(idxc);
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
								if (inv_shift >= 0 && chunk_bits[c] < 63) {
										const unsigned long long invMaskSegment = (invMask[pd] >> inv_shift) & (static_cast<unsigned long long>(1) << chunk_bits[c]) - 1ull;
										bits ^= invMaskSegment;
								}
								accBits[pd] = (accBits[pd] << Lc) | bits;
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

__declspec(noalias) __forceinline float pointToT(const float* __restrict q) const noexcept {
		unsigned long long cell[32];
		int d = 0;
#pragma loop ivdep
		while (d < dim) {
				const int pd = perm[d];
				const float v = (q[pd] - baseOff[pd]) * invStep[pd];
				const long long c = _mm_cvt_ss2si(_mm_round_ss(_mm_setzero_ps(), _mm_set_ss(v),
																											 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
				const long long maxv = (static_cast<long long>(1) << levels) - 1;
				cell[pd] = static_cast<unsigned long long>(c < 0 ? 0 : (c > maxv ? maxv : c));
				++d;
		}
		float t = 0.0;
		int c = chunks;
		while (c > 0) {
				--c;
				const int Lc = chunk_bits[c];
				const unsigned long long baseC = chunk_bases[c];
				int shift_from_top = 0;
				int k = 0;
				while (k <= c) {
						shift_from_top += chunk_bits[k];
						++k;
				}
				const int inv_shift = levels - shift_from_top;
				unsigned long long idxc = 0ull;
				d = 0;
#pragma loop ivdep
				while (d < dim) {
						const int pd = perm[d];
						const unsigned long long mask = (static_cast<unsigned long long>(1) << Lc) - 1;
						unsigned long long bits = (cell[pd] >> inv_shift) & mask;

						if (inv_shift >= 0 && chunk_bits[c] < 63) {
								const unsigned long long invMaskSegment = (invMask[pd] >> inv_shift) & mask;
								bits ^= invMaskSegment;
						}

						const unsigned long long pdep_mask = pextMaskChunks[static_cast<size_t>(c) * dim + d];
						idxc |= _pdep_u64(bits, pdep_mask);
						++d;
				}
				if (use_gray)
						idxc = gray_decode(idxc);
				t = (t + static_cast<float>(idxc)) / static_cast<float>(baseC);
		}
		return static_cast<float>(t);
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
		const float targetX;
		const float targetY;
		const float minTheta;
		const float archBiasW;
		const float archBiasK;
		const float sharpW;

		__declspec(noalias) __forceinline ManipCost(int _n, bool _variableLen, float _targetX, float _targetY, float _minTheta) noexcept
				: n(_n), variableLen(_variableLen), targetX(_targetX), targetY(_targetY), minTheta(_minTheta),
					archBiasW(0.02f), archBiasK(3.0f), sharpW(0.05f) {
}

__declspec(noalias) __forceinline float operator()(const float* __restrict q, float& out_x, float& out_y) const noexcept {
		const float* __restrict th = q;
		const float* __restrict L = variableLen ? (q + n) : nullptr;
		alignas(16) float phi[32];
		alignas(16) float s_arr[32];
		alignas(16) float c_arr[32];
		float x = 0.0f;
		float y = 0.0f;
		float phi_acc = 0.0f;
		float penC = 0.0f;
		float archPen = 0.0f;
		int i = 0;
#pragma loop ivdep
				while (i < n) {
						phi_acc += th[i];
						phi[i] = phi_acc;
						++i;
				}
				FABE13_SINCOS(phi, s_arr, c_arr, n);
				const float sharpScale = 2.0f / (minTheta);
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
						const float v = fmaf(ai, 1.0f, -minTheta);
						if (v > 0.0f) {
								const float scale = fmaf(sharpScale, v, 0.0f);
								const float arg = fmaf(scale, 0.69314718055994530941723212145818f, 0.0f);
								const float exp2_val = fmaf(arg, fmaf(arg, fmaf(arg, fmaf(arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
								penC = fmaf(sharpW, fmaf(exp2_val, 1.0f, -1.0f), penC);
						}
						const float t = fmaf(-theta, archBiasK, 0.0f);
						float sp;
						if (t > 10.0f)
								sp = t;
						else {
								const float exp_val = fmaf(t, fmaf(t, fmaf(t, fmaf(t, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
								sp = log1pf(exp_val);
						}
						archPen = fmaf(archBiasW, sp, archPen);
						++i;
				}
				const float dx = fmaf(x, 1.0f, -targetX);
				const float dy = fmaf(y, 1.0f, -targetY);
				const float dist = sqrtf(fmaf(dx, dx, dy * dy));
				out_x = x;
				out_y = y;
				return fmaf(dist, 1.0f, fmaf(penC, 1.0f, archPen));
		}
};

static __declspec(noalias) __forceinline void HitTest2D_analytic(float x_param, float& out_x1, float& out_x2) noexcept {
	const float a = gActiveMap.a;
	const float inv_lenx = gActiveMap.inv_lenx;
	const unsigned scale = gActiveMap.scale;
	const unsigned scale_minus_1 = scale - 1u;
	const float lenx = gActiveMap.lenx;
	const float leny = gActiveMap.leny;
	const float c = gActiveMap.c;
	const unsigned start = gActiveMap.start;
	const int levels = gActiveMap.levels;
	float norm = (x_param - a) * inv_lenx;
	norm = fminf(fmaxf(norm, 0.0f), 0x1.fffffep-1f);
	unsigned idx = static_cast<unsigned>(norm * static_cast<float>(scale));
	idx = idx > scale_minus_1 ? scale_minus_1 : idx;
	float sx = lenx;
	float sy = leny;
	float x1 = a;
	float x2 = c;
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

static __declspec(noalias) __forceinline int generate_sobol_seeds(
	const MortonND& map, int dim, float* __restrict S, int stride, unsigned seed) noexcept
{
	int temp_dim = dim;
	const int ns_orig = static_cast<int>(fmaf(static_cast<float>(--temp_dim),
		fmaf(static_cast<float>(temp_dim),
			fmaf(static_cast<float>(temp_dim),
				fmaf(static_cast<float>(temp_dim), 0.00833333377f, 0.0416666679f),
				0.16666667f),
			0.5f),
		1.0f));
	int ns_pow2 = 1;
	while (ns_pow2 < ns_orig) ns_pow2 <<= 1;
	static const unsigned int sobol_dir[32][32] = {
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
																		45u, 45u, 51u, 51u, 57u, 57u, 63u, 63u, 69u, 69u, 75u, 75u, 81u, 81u, 87u, 87u },
																		{ 1u, 3u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u,
																			5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u },
																			{ 1u, 1u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
																				7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u },
																				{ 1u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u,
																					3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u },
																					{ 1u, 1u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u,
																						5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u },
																						{ 1u, 3u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
																							7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u },
																							{ 1u, 1u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u,
																								3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u },
																								{ 1u, 3u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u,
																									5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u },
																									{ 1u, 1u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
																										7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u },
																										{ 1u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u,
																											3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u },
																											{ 1u, 1u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u,
																												5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u },
																												{ 1u, 3u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
																													7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u },
																													{ 1u, 1u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u,
																														3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u },
																														{ 1u, 3u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u,
																															5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u },
																															{ 1u, 1u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u,
																																7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u },
																																{ 1u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u,
																																	3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u },
																																	{ 1u, 1u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u,
																																		5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u }
	};
	unsigned int scramble_mask[32];
	unsigned st = seed;
	int d = 0;
#pragma loop ivdep
	while (d < dim) {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		scramble_mask[d] = st;
		++d;
	}
	const unsigned long long start_idx = 1ull; 
	unsigned int cur_x[32];
	d = 0;
#pragma loop ivdep
	while (d < dim) {
		unsigned int x = 0;
		unsigned long long ii = start_idx;
		int b = 0;
		while (ii && b < 32) {
			if (ii & 1ull)
				x ^= sobol_dir[d][b];
			ii >>= 1;
			++b;
		}
		x ^= scramble_mask[d];
		cur_x[d] = x;
		++d;
	}
	d = 0;
#pragma loop ivdep
	while (d < dim) {
		float u = static_cast<float>(cur_x[d]) * 2.3283064e-10f;
		int pd = map.perm[d];
		float lo = map.low[pd];
		float hi = map.high[pd];
		S[0 * stride + d] = fmaf(u, hi - lo, lo);
		++d;
	}
	unsigned long long prev_gray = start_idx ^ (start_idx >> 1);
	int j = 1;
	while (j < ns_pow2) {
		unsigned long long i = start_idx + j;
		unsigned long long gray = i ^ (i >> 1);
		unsigned long long diff = gray ^ prev_gray;
		int bit = _tzcnt_u64(diff);  
		d = 0;
#pragma loop ivdep
		while (d < dim) {
			cur_x[d] ^= sobol_dir[d][bit];
			++d;
		}
		prev_gray = gray;
		d = 0;
#pragma loop ivdep
		while (d < dim) {
			float u = static_cast<float>(cur_x[d]) * 2.3283064e-10f;
			int pd = map.perm[d];
			float lo = map.low[pd];
			float hi = map.high[pd];
			S[j * stride + d] = fmaf(u, hi - lo, lo);
			++d;
		}
		++j;
	}

	return ns_pow2;
}
static __declspec(noalias) __forceinline void ccd_ik(
	float targetX, float targetY,
	const float* lengths, int n,
	float* angles, int max_iter) noexcept
{
	__declspec(align(16)) float x[32];
	__declspec(align(16)) float y[32];
	__declspec(align(16)) float cum_angles[32];
	__declspec(align(16)) float s_arr[32];
	__declspec(align(16)) float c_arr[32];
	int iter = 0;
	while (iter < max_iter) {
		float acc = 0.0f;
		int i = 0;
#pragma loop ivdep
		while (i < n) {
			acc += angles[i];
			cum_angles[i] = acc;
			++i;
		}
		FABE13_SINCOS(cum_angles, s_arr, c_arr, n);
		float curX = 0.0f;
		float curY = 0.0f;
		x[0] = 0.0f;
		y[0] = 0.0f;
		i = 0;
#pragma loop ivdep
		while (i < n) {
			curX = fmaf(lengths[i], c_arr[i], curX);
			curY = fmaf(lengths[i], s_arr[i], curY);
			x[i + 1] = curX;
			y[i + 1] = curY;
			++i;
		}
		i = n - 1;
		while (i >= 0) {
			float toEndX = x[n] - x[i];
			float toEndY = y[n] - y[i];
			float toTargetX = targetX - x[i];
			float toTargetY = targetY - y[i];
			float dot = fmaf(toEndX, toTargetX, toEndY * toTargetY);
			float det = fmaf(toEndX, toTargetY, -toEndY * toTargetX);
			float angle = atan2f(det, dot);
			angles[i] += angle;
			const float lo = (i == 0) ? -1.0471975511965976f : -2.6179938779914944f;
			const float hi = 2.6179938779914944f;
			if (angles[i] < lo) angles[i] = lo;
			if (angles[i] > hi) angles[i] = hi;
			--i;
		}
		++iter;
	}
}
static __declspec(noalias) __forceinline int generate_heuristic_seeds(
	const ManipCost& cost, const MortonND& map, int dim,
	float* __restrict S, int stride, unsigned seed) noexcept
{
	const int n = cost.n;
	const bool VL = cost.variableLen;
	const float tx = cost.targetX;
	const float ty = cost.targetY;
	const float phi = atan2f(ty, tx);
	int total_seeds = 0;
	float dist_to_target = sqrtf(tx * tx + ty * ty);
	float max_reach = 0.0f;
	if (VL) {
		for (int i = 0; i < n; ++i) max_reach += map.high[n + i];
	}
	else {
		max_reach = static_cast<float>(n);
	}
	float ratio = dist_to_target / max_reach;
	bool prefer_extended = (ratio > 0.7f);
	bool prefer_compact = (ratio < 0.4f);
	bool use_ik = !(ratio > 0.4f && ratio < 0.7f);
	{
		float* s0 = S + total_seeds * stride;
		const float rho = sqrtf(fmaf(tx, tx, ty * ty));
		const float len = fmaf(1.0f / static_cast<float>(n), rho, 0.0f);
		int i = 0;
#pragma loop ivdep
		while (i < n) {
			s0[i] = fminf(fmaxf((1.0f / static_cast<float>(n)) * phi, map.low[i]), map.high[i]);
			++i;
		}
		if (VL) {
			i = 0;
#pragma loop ivdep
			while (i < n) {
				s0[n + i] = fminf(fmaxf(len, map.low[n + i]), map.high[n + i]);
				++i;
			}
		}
		++total_seeds;
	}
	{
		float* s1 = S + total_seeds * stride;
		int i = 0;
#pragma loop ivdep
		while (i < n) {
			s1[i] = fminf(fmaxf(0.5f * phi * ((i & 1) ? -1.0f : 1.0f), map.low[i]), map.high[i]);
			++i;
		}
		if (VL) {
			i = 0;
#pragma loop ivdep
			while (i < n) {
				s1[n + i] = fminf(fmaxf(fmaf(0.4f, static_cast<float>(i) / static_cast<float>(n), 0.8f), map.low[n + i]), map.high[n + i]);
				++i;
			}
		}
		++total_seeds;
	}
	{
		float* s2 = S + total_seeds * stride;
		const float inv = (n > 1) ? 1.0f / static_cast<float>(n - 1) : 0.0f;
		int i = 0;
#pragma loop ivdep
		while (i < n) {
			const float pr = static_cast<float>(i) * inv;
			s2[i] = fminf(fmaxf(fmaf(phi, fmaf(-0.3f, pr, 1.0f), 0.0f), map.low[i]), map.high[i]);
			++i;
		}
		if (VL) {
			int j = 0;
#pragma loop ivdep
			while (j < n) {
				float si;
				FABE13_SIN(fmaf(1.5f, static_cast<float>(j), 0.0f), si);
				s2[n + j] = fminf(fmaxf(fmaf(0.2f, si, 1.0f), map.low[n + j]), map.high[n + j]);
				++j;
			}
		}
		++total_seeds;
	}
	if (use_ik && prefer_extended) {
		float* s3 = S + total_seeds * stride;
		float angles[32] = { 0.0f };
		float lengths[32];
		if (VL) {
			float len_low = map.low[n];
			float len_high = map.high[n];
			float avg_len = (len_low + len_high) * 0.5f;
			int i = 0;
#pragma loop ivdep
			while (i < n) {
				lengths[i] = avg_len;
				++i;
			}
		}
		else {
			int i = 0;
#pragma loop ivdep
			while (i < n) {
				lengths[i] = 1.0f;
				++i;
			}
		}
		ccd_ik(tx, ty, lengths, n, angles, 10);
		int i = 0;
#pragma loop ivdep
		while (i < n) {
			s3[i] = fminf(fmaxf(angles[i], map.low[i]), map.high[i]);
			++i;
		}
		if (VL) {
			i = 0;
#pragma loop ivdep
			while (i < n) {
				s3[n + i] = lengths[i];
				++i;
			}
		}
		++total_seeds;
	}
	if (use_ik && prefer_compact) {
		float* s4 = S + total_seeds * stride;
		float angles_fabrik[32] = { 0.0f };
		float lengths_fabrik[32];
		if (VL) {
			float len_low = map.low[n];
			float len_high = map.high[n];
			float avg_len = (len_low + len_high) * 0.5f;
			int i = 0;
#pragma loop ivdep
			while (i < n) {
				lengths_fabrik[i] = avg_len;
				++i;
			}
		}
		else {
			int i = 0;
#pragma loop ivdep
			while (i < n) {
				lengths_fabrik[i] = 1.0f;
				++i;
			}
		}
		float targetX_fab = tx;
		float targetY_fab = ty;
		for (int iter_fab = 0; iter_fab < 3; ++iter_fab) {
			float prevX = targetX_fab;
			float prevY = targetY_fab;
			for (int j = n - 1; j >= 0; --j) {
				float len = lengths_fabrik[j];
				float angle_to_target = atan2f(prevY, prevX);
				angles_fabrik[j] = angle_to_target;
				float s_val, c_val;
				FABE13_SINCOS(&angle_to_target, &s_val, &c_val, 1); 
				prevX = prevX - len * c_val;
				prevY = prevY - len * s_val;
			}
			float curX = 0.0f, curY = 0.0f;
			float angle_sum = 0.0f;
			for (int j = 0; j < n; ++j) {
				angle_sum += angles_fabrik[j];
				float s_val, c_val;
				FABE13_SINCOS(&angle_sum, &s_val, &c_val, 1);
				curX += lengths_fabrik[j] * c_val;
				curY += lengths_fabrik[j] * s_val;
			}
		}
		int i = 0;
#pragma loop ivdep
		while (i < n) {
			s4[i] = fminf(fmaxf(angles_fabrik[i], map.low[i]), map.high[i]);
			++i;
		}
		if (VL) {
			i = 0;
#pragma loop ivdep
			while (i < n) {
				s4[n + i] = lengths_fabrik[i];
				++i;
			}
		}
		++total_seeds;
	}
	const int sobol_count = generate_sobol_seeds(map, dim, S + total_seeds * stride, stride, seed);
	total_seeds += sobol_count;

	return total_seeds;
}
static __declspec(noalias) void agp_run_branch_mpi(
	const MortonND& map,
	const ManipCost& cost,
	int maxIter,
	float r,
	bool adaptive,
	float eps,
	unsigned seed,
	std::vector<IntervalND*, boost::alignment::aligned_allocator<IntervalND*, 16u>>& H,
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>>& bestQ,
	float& bestF,
	float& bestX,
	float& bestY,
	size_t& out_iterations,
	float& out_achieved_epsilon,
	float M_prior) noexcept {
	const int n = cost.n;
	const int dim = n + (cost.variableLen ? n : 0);
	const float dim_f = static_cast<float>(dim);
	unsigned exchange_counter = 0;
	unsigned exchange_counter_T = 0;
	alignas(16) float M_by_span[12];
	int msi = 0;
	while (msi < 12)
		M_by_span[msi++] = M_prior;
	float Mmax = M_prior;
	alignas(16) float q_local[32];
	alignas(16) float phi[32];
	alignas(16) float s_arr[32];
	alignas(16) float c_arr[32];
	alignas(16) float sum_s[32];
	alignas(16) float sum_c[32];
	alignas(16) float q_try[32];
	bestQ.reserve(static_cast<size_t>(dim));
	float x = 0.0f;
	float y = 0.0f;
	int no_improve = 0;
	auto t_to_idx = [&](float t) -> unsigned long long {
		unsigned long long idx = static_cast<unsigned long long>(fmaf(t, static_cast<float>(map.scale), 0.0f));
		return idx;
		};
	auto update_pockets_and_Mmax = [&](IntervalND* I) {
		const int k = I->span_level;
		if (I->M > M_by_span[k])
			M_by_span[k] = I->M;
		if (M_by_span[k] > Mmax)
			Mmax = M_by_span[k];
		};
	const float a = 0.0f;
	const float b = 1.0f;
	float p = 0.0f;
	float dmax = b - a;
	const float initial_len = dmax;
	const float A_dim = fmaf(1.0f / sqrtf(dim_f + 6.75f), 5.535f, 0.0f);
	const float A_dim__ = fmaf(1.0f / sqrtf(dim_f + 6.75f), 3.425f, 0.0f);
	const float B_dim = fmaf(A_dim, 0.7f, 0.0f);
	const float B_dim__ = fmaf(A_dim__, 4.325f, 0.0f);
	const float log_argument = A_dim - 2.03f;
	const float log_argument__ = A_dim__ - 2.0f;
	const float C_dim = fmaf(log_argument, fmaf(log_argument, fmaf(log_argument, fmaf(log_argument, fmaf(log_argument, 0.164056f, -0.098462f), 0.240884f), -0.351834f), 0.999996f), log_argument) - B_dim;
	const float C_dim__ = fmaf(log_argument__, fmaf(log_argument__, fmaf(log_argument__, fmaf(log_argument__, fmaf(log_argument__, 0.164056f, -0.098462f), 0.240884f), -0.351834f), 0.999996f), log_argument__) - B_dim__;
	const float adaptive_coeff_addition = fmaf(C_dim, fmaf(C_dim, fmaf(C_dim, fmaf(C_dim, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
	const float adaptive_coeff_addition__ = fmaf(C_dim__, fmaf(C_dim__, fmaf(C_dim__, fmaf(C_dim__, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
	float adaptive_coeff = A_dim - adaptive_coeff_addition;
	float adaptive_coeff__ = A_dim__ - adaptive_coeff_addition__;
	const float A_dim_clone = fmaf(A_dim - fmaf(-fmaf(B_dim, fmaf(B_dim, fmaf(B_dim, fmaf(B_dim, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), adaptive_coeff_addition, A_dim), 0.5f, 0.0f);
	int it = 0;
	int stag_boost_remaining = 0;
	float stag_r_multiplier = 0.0f;
	int last_send_T = 0;
	const int send_interval_T = 7;
	int last_send_best = 0;
	const int send_interval_best = 2;
	const int n_stag_iters = static_cast<int>(3.0f + 2.045f * sqrtf(dim_f)); 
	auto evalAt = [&](const float t) -> float {
		map.map01ToPoint(t, q_local);
		float f = cost(q_local, x, y);
		if (f < fmaf(bestF, adaptive_coeff, 0.0f)) {
			const float f_start = f;
			const float c1 = 1e-4f;           
			const float tau = 0.5f;              
			const int max_outer_iters = (int)(50.0f * (1.0f + 0.65f * p));
			const int max_backtrack = (int)(20.0f * (1.0f + 0.65f * p));
			const float lbfgs_trigger = 0.6f;   
			const int   m_lbfgs = 9;               
			const int max_lbfgs_iters = (int)(25.0f * (1.0f + 0.65f * p));
			const float eps_lbfgs_curv = 1e-6f;    
			const float eps_descent = 1e-12f;
			float eta = 2.0f / sqrtf(dim_f); 
			auto clampPoint = [&](float* q) {
				int i = 0;
#pragma loop ivdep
				while (i < n) {
					const float lo = (i == 0) ? -1.0471975511965977f : -2.6179938779914944f;
					const float hi = 2.6179938779914944f;
					if (q[i] < lo) q[i] = lo;
					else if (q[i] > hi) q[i] = hi;
					++i;
				}
				if (cost.variableLen) {
					i = 0;
#pragma loop ivdep
					while (i < n) {
						if (q[n + i] < 0.5f) q[n + i] = 0.5f;
						else if (q[n + i] > 2.0f) q[n + i] = 2.0f;
						++i;
					}
				}
				};
			auto computeGrad = [&](const float* q_in, const float x_in, const float y_in,
				float* grad_out, float& grad_norm2_out) {
					float acc = 0.0f;
					int ii = 0;
#pragma loop ivdep
					while (ii < n) {
						acc = fmaf(q_in[ii], 1.0f, acc);
						phi[ii] = acc;
						++ii;
					}
					FABE13_SINCOS(phi, s_arr, c_arr, n);
					float as = 0.0f;
					float ac = 0.0f;
					int k = n - 1;
					while (k >= 0) {
						const float Lk = cost.variableLen ? q_in[n + k] : 1.0f;
						as = fmaf(Lk, s_arr[k], as);
						ac = fmaf(Lk, c_arr[k], ac);
						sum_s[k] = as;
						sum_c[k] = ac;
						--k;
					}
					const float dx = fmaf(x_in, 1.0f, -cost.targetX);
					const float dy = fmaf(y_in, 1.0f, -cost.targetY);
					const float dist = sqrtf(fmaf(dx, dx, dy * dy));
					const float inv_dist = 1.0f / dist;
					grad_norm2_out = 0.0f;
					int i = 0;
#pragma loop ivdep
					while (i < n) {
						float gpen = 0.0f;
						{
							const float ai = fabsf(q_in[i]);
							const float v = fmaf(ai, 1.0f, -cost.minTheta);
							if (v > 0.0f) {
								const float scale_arg = fmaf(2.0f / cost.minTheta, v * 0.69314718f, 0.0f);
								const float exp_val = fmaf(scale_arg,
									fmaf(scale_arg,
										fmaf(scale_arg,
											fmaf(scale_arg, 0.00833333377f, 0.0416666679f),
											0.16666667f),
										0.5f),
									1.0f);
								const float dpen = fmaf(cost.sharpW, exp_val * (1.38629436f / cost.minTheta), 0.0f);
								gpen = fmaf(dpen, copysignf(1.0f, q_in[i]), gpen);
							}
						}
						{
							const float tsg = fmaf(-q_in[i], cost.archBiasK, 0.0f);
							const float exp_arg = -tsg;
							const float exp_val = fmaf(exp_arg,
								fmaf(exp_arg,
									fmaf(exp_arg,
										fmaf(exp_arg, 0.00833333377f, 0.0416666679f),
										0.16666667f),
									0.5f),
								1.0f);
							const float sig = 1.0f / (exp_val + 1.0f);
							gpen = fmaf(-cost.archBiasW * cost.archBiasK, sig, gpen);
						}
						const float g_main = fmaf(dx, -sum_s[i], dy * sum_c[i]) * inv_dist;
						float gi = g_main + gpen;
						grad_out[i] = gi;
						grad_norm2_out = fmaf(gi, gi, grad_norm2_out);
						++i;
					}
					if (cost.variableLen) {
						int j = 0;
#pragma loop ivdep
						while (j < n) {
							const float gi = fmaf(dx, c_arr[j], dy * s_arr[j]) * inv_dist;
							grad_out[n + j] = gi;
							grad_norm2_out = fmaf(gi, gi, grad_norm2_out);
							++j;
						}
					}
				};
			auto armijoLineSearch = [&](const float* q_base,
				const float f_base,
				const float x_base, const float y_base,
				const float* grad_base,
				const float* dir,
				const float gtd,
				float& alpha_io,
				float* q_out,
				float& f_out,
				float& x_out,
				float& y_out,
				bool& clipped) -> bool {
					float alpha = alpha_io;
					int backtrack = 0;
					while (backtrack < max_backtrack) {
						int i = 0;
#pragma loop ivdep
						while (i < dim) {
							q_out[i] = fmaf(alpha, dir[i], q_base[i]);
							++i;
						}
						float q_before_clamp[64];
						memcpy(q_before_clamp, q_out, dim * sizeof(float));
						clampPoint(q_out);
						clipped = false;
						i = 0;
#pragma loop ivdep
						while (i < dim) {
							if (fabsf(q_out[i] - q_before_clamp[i]) > 1e-12f) {
								clipped = true;
								break;
							}
							++i;
						}
						float x2, y2;
						float f_try = cost(q_out, x2, y2);
						if (!(f_try == f_try)) { 
							alpha *= tau;
							++backtrack;
							continue;
						}
						if (f_try <= f_base + c1 * alpha * gtd) {
							alpha_io = alpha;
							f_out = f_try;
							x_out = x2;
							y_out = y2;
							return true;
						}
						alpha *= tau;
						++backtrack;
					}
					return false;
				};
			bool lbfgs_already_tried = false;
			int outer = 0;
			while (outer < max_outer_iters) {
				float grad[64];
				float grad_norm2 = 0.0f;
				computeGrad(q_local, x, y, grad, grad_norm2);
				if (!(grad_norm2 == grad_norm2) || grad_norm2 < 1e-12f) break;
				float dir_gd[64];
				int i = 0;
#pragma loop ivdep
				while (i < dim) {
					dir_gd[i] = -grad[i];
					++i;
				}
				const float gtd_gd = -grad_norm2;
				float eta_trial = eta;
				float f_new = f, x_new = x, y_new = y;
				bool clipped_gd = false;
				bool found = armijoLineSearch(q_local, f, x, y, grad, dir_gd, gtd_gd,
					eta_trial, q_try, f_new, x_new, y_new, clipped_gd);
				if (!found) break; 
				memcpy(q_local, q_try, static_cast<size_t>(dim) * sizeof(float));
				f = f_new;
				x = x_new;
				y = y_new;
				eta = eta_trial;
				const float rel_impr = (f_start - f) / f_start;
				if (!lbfgs_already_tried && rel_impr >= lbfgs_trigger) {
					lbfgs_already_tried = true;
					float q_resume[64];
					memcpy(q_resume, q_local, static_cast<size_t>(dim) * sizeof(float));
					float f_resume = f;
					float x_resume = x;
					float y_resume = y;
					float eta_resume = eta;
					float q_best_lbfgs[64];
					memcpy(q_best_lbfgs, q_local, static_cast<size_t>(dim) * sizeof(float));
					float f_best_lbfgs = f;
					float x_best_lbfgs = x;
					float y_best_lbfgs = y;
					float s_hist[m_lbfgs][64];
					float y_hist[m_lbfgs][64];
					float rho_hist[m_lbfgs];
					float alpha_hist[m_lbfgs];
					int hist_size = 0;
					float gk[64];
					float gk_norm2 = 0.0f;
					computeGrad(q_local, x, y, gk, gk_norm2);
					bool lbfgs_ok = true;
					float alpha_k = 2.0f / sqrtf(dim_f);
					int it = 0;
					while (it < max_lbfgs_iters) {
						if (!(gk_norm2 == gk_norm2) || gk_norm2 < 1e-12f) break;
						float dir[64];
						if (hist_size == 0) {
							int d = 0;
#pragma loop ivdep
							while (d < dim) {
								dir[d] = -gk[d];
								++d;
							}
						}
						else {
							float q_vec[64];
							int d = 0;
#pragma loop ivdep
							while (d < dim) {
								q_vec[d] = gk[d];
								++d;
							}
							for (int jj = hist_size - 1; jj >= 0; --jj) {
								float dot_sq = 0.0f;
								d = 0;
#pragma loop ivdep
								while (d < dim) {
									dot_sq = fmaf(s_hist[jj][d], q_vec[d], dot_sq);
									++d;
								}
								const float a = dot_sq * rho_hist[jj];
								alpha_hist[jj] = a;
								d = 0;
#pragma loop ivdep
								while (d < dim) {
									q_vec[d] = fmaf(-a, y_hist[jj][d], q_vec[d]);
									++d;
								}
							}
							float gamma = 1.0f;
							{
								const int last = hist_size - 1;
								float yy = 0.0f;
								d = 0;
#pragma loop ivdep
								while (d < dim) {
									yy = fmaf(y_hist[last][d], y_hist[last][d], yy);
									++d;
								}
								const float ys = 1.0f / rho_hist[last];
								if (yy > 0.0f) gamma = ys / yy;
							}
							float r_vec[64];
							d = 0;
#pragma loop ivdep
							while (d < dim) {
								r_vec[d] = gamma * q_vec[d];
								++d;
							}
							for (int jj = 0; jj < hist_size; ++jj) {
								float dot_yr = 0.0f;
								d = 0;
#pragma loop ivdep
								while (d < dim) {
									dot_yr = fmaf(y_hist[jj][d], r_vec[d], dot_yr);
									++d;
								}
								const float b = dot_yr * rho_hist[jj];
								const float coeff = alpha_hist[jj] - b;
								d = 0;
#pragma loop ivdep
								while (d < dim) {
									r_vec[d] = fmaf(coeff, s_hist[jj][d], r_vec[d]);
									++d;
								}
							}
							d = 0;
#pragma loop ivdep
							while (d < dim) {
								dir[d] = -r_vec[d];
								++d;
							}
						}
						float gtd = 0.0f;
						int d = 0;
#pragma loop ivdep
						while (d < dim) {
							gtd = fmaf(gk[d], dir[d], gtd);
							++d;
						}
						if (!(gtd == gtd) || gtd >= -eps_descent) {
							d = 0;
#pragma loop ivdep
							while (d < dim) {
								dir[d] = -gk[d];
								++d;
							}
							gtd = -gk_norm2;
							if (gtd >= -eps_descent) { lbfgs_ok = false; break; }
						}
						float q_old[64];
						memcpy(q_old, q_local, static_cast<size_t>(dim) * sizeof(float));
						float g_old[64];
						memcpy(g_old, gk, static_cast<size_t>(dim) * sizeof(float));
						const float f_old = f;
						const float x_old = x;
						const float y_old = y;
						float alpha_try = alpha_k;
						float f_try, x_try, y_try;
						bool clipped_lbfgs = false;
						bool step_ok = armijoLineSearch(q_local, f, x, y, gk, dir, gtd,
							alpha_try, q_try, f_try, x_try, y_try, clipped_lbfgs);
						if (!step_ok) { lbfgs_ok = false; break; }
						memcpy(q_local, q_try, static_cast<size_t>(dim) * sizeof(float));
						f = f_try;
						x = x_try;
						y = y_try;
						alpha_k = alpha_try;
						computeGrad(q_local, x, y, gk, gk_norm2);
						if (!(gk_norm2 == gk_norm2)) { lbfgs_ok = false; break; }
						if (f < f_best_lbfgs) {
							f_best_lbfgs = f;
							memcpy(q_best_lbfgs, q_local, static_cast<size_t>(dim) * sizeof(float));
							x_best_lbfgs = x;
							y_best_lbfgs = y;
						}
						float ys = 0.0f;
						float s_new[64];
						float y_new[64];
						d = 0;
#pragma loop ivdep
						while (d < dim) {
							const float sd = q_local[d] - q_old[d];
							const float yd = gk[d] - g_old[d];
							s_new[d] = sd;
							y_new[d] = yd;
							ys = fmaf(yd, sd, ys);
							++d;
						}
						if (!(ys == ys) || ys <= eps_lbfgs_curv) {
							hist_size = 0;
						}
						else {
							if (hist_size < m_lbfgs) {
								const int idx = hist_size;
								rho_hist[idx] = 1.0f / ys;
								d = 0;
#pragma loop ivdep
								while (d < dim) {
									s_hist[idx][d] = s_new[d];
									y_hist[idx][d] = y_new[d];
									++d;
								}
								++hist_size;
							}
							else {
								for (int jj = 0; jj < m_lbfgs - 1; ++jj) {
									rho_hist[jj] = rho_hist[jj + 1];
									d = 0;
#pragma loop ivdep
									while (d < dim) {
										s_hist[jj][d] = s_hist[jj + 1][d];
										y_hist[jj][d] = y_hist[jj + 1][d];
										++d;
									}
								}
								const int idx = m_lbfgs - 1;
								rho_hist[idx] = 1.0f / ys;
								d = 0;
#pragma loop ivdep
								while (d < dim) {
									s_hist[idx][d] = s_new[d];
									y_hist[idx][d] = y_new[d];
									++d;
								}
								hist_size = m_lbfgs;
							}
						}
						if (gk_norm2 < 1e-12f) break;
						++it;
					}
					if (lbfgs_ok) {
						memcpy(q_local, q_best_lbfgs, static_cast<size_t>(dim) * sizeof(float));
						f = f_best_lbfgs;
						x = x_best_lbfgs;
						y = y_best_lbfgs;
						break; 
					}
					else {
						if (f_best_lbfgs < f_resume) {
							memcpy(q_local, q_best_lbfgs, static_cast<size_t>(dim) * sizeof(float));
							f = f_best_lbfgs;
							x = x_best_lbfgs;
							y = y_best_lbfgs;
						}
						else {
							memcpy(q_local, q_resume, static_cast<size_t>(dim) * sizeof(float));
							f = f_resume;
							x = x_resume;
							y = y_resume;
						}
						eta = eta_resume;
					}
				}

				++outer;
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
		}
		return f;
		};
	auto progress_outgoing = [&]() {
		for (auto it = g_pendingMulti.begin(); it != g_pendingMulti.end(); ) {
			if (it->req.test()) {
				it = g_pendingMulti.erase(it);
			}
			else {
				++it;
			}
		}
		for (auto it = g_pendingBest.begin(); it != g_pendingBest.end(); ) {
			if (it->req.test()) {
				it = g_pendingBest.erase(it);
			}
			else {
				++it;
			}
		}
		};
	const float f_a = evalAt(a);
	const float f_b = evalAt(b);
	const int K = static_cast<int>(fmaf(-fmaf(sqrtf(dim_f), dim_f, 0.0f), 0.725f, 10.95f));
	H.reserve(static_cast<size_t>(maxIter) + static_cast<size_t>(K) + 16u);
	const int rank = g_world->rank();
	const int world = g_world->size();
	while (g_world->iprobe(boost::mpi::any_source, 0)) {
		MultiCrossMsg dummy;
		g_world->recv(boost::mpi::any_source, 0, dummy);
	}
	while (g_world->iprobe(boost::mpi::any_source, 2)) {
		BestSolutionMsg dummy;
		g_world->recv(boost::mpi::any_source, 2, dummy);
	}
	alignas(16) float seeds[256 * 32];
	const int seedCnt = generate_heuristic_seeds(cost, map, dim, seeds, 32, static_cast<unsigned>(fmaf(static_cast<float>(rank), 7919.0f, static_cast<float>(seed))));
	int i = 0;
	while (i < seedCnt) {
		const float* s = seeds + static_cast<size_t>(fmaf(static_cast<float>(i), 32.0f, 0.0f));
		const float t_seed = map.pointToT(s);
		const float interval_size = (i < 3) ? fmaf(0.0004f, static_cast<float>(dim), 0.0f) : fmaf(fmaf(0.00031f, static_cast<float>(dim), 0.0f), exp2f((1.0f / static_cast<float>(seedCnt - 4)) * log2f(fmaf(0.00025f, 1.0f / 0.00031f, 0.0f)) * static_cast<float>(i - 3)), 0.0f);
		const float t1 = fmaf(-interval_size, 1.0f, t_seed);
		const float t2 = fmaf(interval_size, 1.0f, t_seed);
		alignas(16) float q1[32];
		alignas(16) float q2[32];
		float x1;
		float y1;
		float x2;
		float y2;
		map.map01ToPoint(t1, q1);
		const float f1 = cost(q1, x1, y1);
		map.map01ToPoint(t2, q2);
		const float f2 = cost(q2, x2, y2);
		IntervalND* I = new IntervalND(t1, t2, f1, f2);
		I->i1 = t_to_idx(t1);
		I->i2 = t_to_idx(t2);
		I->diam = map.block_diameter(I->i1, I->i2);
		I->compute_span_level(map);
		I->set_metric(I->diam);
		update_pockets_and_Mmax(I);
		I->ChangeCharacteristic(fmaf(r, Mmax, 0.0f));
		if (i < 3)
			I->R = fmaf(I->R, fmaf(0.01f, static_cast<float>(dim), 0.85f), 0.0f);
		else {
			const float start_mult = fmaf(0.214f, static_cast<float>(dim), 0.0f);
			const float end_mult = fmaf(0.174f, static_cast<float>(dim), 0.0f);
			const float mult = fmaf(start_mult, exp2f((1.0f / static_cast<float>(seedCnt - 4)) * log2f(fmaf(end_mult, 1.0f / start_mult, 0.0f)) * static_cast<float>(i - 3)), 0.0f);
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
		++i;
	}
	float prev_t = a;
	float prev_f = f_a;
	int k = 1;
	while (k <= K) {
		const float t = fmaf(fmaf(b - a, static_cast<float>(k) / static_cast<float>(K + 1), a), 1.0f, static_cast<float>(rank) / static_cast<float>(world * (K + 1)));
		const float f = evalAt(t);
		IntervalND* I = new IntervalND(prev_t, t, prev_f, f);
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
	IntervalND* tail = new IntervalND(prev_t, b, prev_f, f_b);
	tail->i1 = t_to_idx(prev_t);
	tail->i2 = t_to_idx(b);
	tail->diam = map.block_diameter(tail->i1, tail->i2);
	tail->compute_span_level(map);
	tail->set_metric(tail->diam);
	update_pockets_and_Mmax(tail);
	tail->ChangeCharacteristic(fmaf(r, Mmax, 0.0f));
	H.emplace_back(tail);
	std::push_heap(H.begin(), H.end(), ComparePtrND);
	const int noImproveThrDim = static_cast<int>(fmaf(7.5f, exp2f(-0.1f * sqrtf(dim_f)), 0.0f));
	g_world->barrier();
	while (true) {
		p = fmaf(-1.0f / initial_len, dmax, 1.0f);
		stag_r_multiplier = 1.4f - 1.1f * fmaf(0.65f * p - 0.45f,
			fmaf(0.65f * p - 0.45f,
				fmaf(0.65f * p - 0.45f,
					fmaf(0.65f * p - 0.45f,
						fmaf(0.65f * p - 0.45f, 0.164056f, -0.098462f),
						0.240884f),
					-0.351834f),
				0.999996f),
			0.65f * p - 0.45f);
		const float p_arg = fmaf(p, 2.3f, -2.9775f);
		float current_r = r;
		if (stag_boost_remaining > 0) {
			current_r = r * stag_r_multiplier;
			stag_boost_remaining--;
		}
		const float grad_threshold = 0.5e-1f; 
		float grad_norm2_best = 0.0f;
		if (no_improve > 0) {
			float acc_best = 0.0f;
			float phi_best[32];
			int ii_best = 0;
#pragma loop ivdep
			while (ii_best < n) {
				acc_best = fmaf(bestQ[ii_best], 1.0f, acc_best);
				phi_best[ii_best] = acc_best;
				++ii_best;
			}
			float s_best[32], c_best[32];
			FABE13_SINCOS(phi_best, s_best, c_best, n);
			float as_best = 0.0f, ac_best = 0.0f;
			float sum_s_best[32], sum_c_best[32];
			int k_best = n - 1;
			while (k_best >= 0) {
				const float Lk = cost.variableLen ? bestQ[n + k_best] : 1.0f;
				as_best = fmaf(Lk, s_best[k_best], as_best);
				ac_best = fmaf(Lk, c_best[k_best], ac_best);
				sum_s_best[k_best] = as_best;
				sum_c_best[k_best] = ac_best;
				--k_best;
			}
			const float dx_best = fmaf(bestX, 1.0f, -cost.targetX);
			const float dy_best = fmaf(bestY, 1.0f, -cost.targetY);
			const float dist_best = sqrtf(fmaf(dx_best, dx_best, dy_best * dy_best));
			const float inv_dist_best = 1.0f / dist_best;
			int i_best = 0;
#pragma loop ivdep
			while (i_best < n) {
				float gpen_best = 0.0f;
				{
					const float ai = fabsf(bestQ[i_best]);
					const float v = fmaf(ai, 1.0f, -cost.minTheta);
					if (v > 0.0f) {
						const float scale_arg = fmaf(2.0f / cost.minTheta, v * 0.69314718f, 0.0f);
						const float exp_val = fmaf(scale_arg,
							fmaf(scale_arg,
								fmaf(scale_arg,
									fmaf(scale_arg, 0.00833333377f, 0.0416666679f),
									0.16666667f),
								0.5f),
							1.0f);
						const float dpen = fmaf(cost.sharpW, exp_val * (1.38629436f / cost.minTheta), 0.0f);
						gpen_best = fmaf(dpen, copysignf(1.0f, bestQ[i_best]), gpen_best);
					}
				}
				{
					const float tsg = fmaf(-bestQ[i_best], cost.archBiasK, 0.0f);
					const float exp_arg = -tsg;
					const float exp_val = fmaf(exp_arg,
						fmaf(exp_arg,
							fmaf(exp_arg,
								fmaf(exp_arg, 0.00833333377f, 0.0416666679f),
								0.16666667f),
							0.5f),
						1.0f);
					const float sig = 1.0f / (exp_val + 1.0f);
					gpen_best = fmaf(-cost.archBiasW * cost.archBiasK, sig, gpen_best);
				}

				const float g_main_best = fmaf(dx_best, -sum_s_best[i_best], dy_best * sum_c_best[i_best]) * inv_dist_best;
				float gi_best = g_main_best + gpen_best;
				grad_norm2_best = fmaf(gi_best, gi_best, grad_norm2_best);
				++i_best;
			}
			if (cost.variableLen) {
				int j_best = 0;
#pragma loop ivdep
				while (j_best < n) {
					const float gi_best = fmaf(dx_best, c_best[j_best], dy_best * s_best[j_best]) * inv_dist_best;
					grad_norm2_best = fmaf(gi_best, gi_best, grad_norm2_best);
					++j_best;
				}
			}
		}
		const bool stagnation = (no_improve > noImproveThrDim) && (grad_norm2_best < grad_threshold);

		const float r_eff = dim > 2 ? fmaf(-fmaf(p_arg, fmaf(p_arg, fmaf(p_arg, fmaf(p_arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f) + 1.05f, fmaf(sqrtf(dim_f - 1), current_r, 0.0f), 0.0f) : fmaf(-fmaf(p_arg, fmaf(p_arg, fmaf(p_arg, fmaf(p_arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f) + 1.05f, current_r, 0.0f);
		if (stagnation) {
			stag_boost_remaining = n_stag_iters;
			const int num_ik = 1 + static_cast<int>(sqrtf(dim_f));
			float dist_to_target = sqrtf(cost.targetX * cost.targetX + cost.targetY * cost.targetY);
			float max_reach = 0.0f;
			if (cost.variableLen) {
				for (int i = 0; i < n; ++i) max_reach += map.high[n + i];
			}
			else {
				max_reach = static_cast<float>(n);
			}
			float ratio = dist_to_target / max_reach;
			bool prefer_extended = (ratio > 0.7f);
			bool prefer_compact = (ratio < 0.4f);
			bool use_ik = !(ratio > 0.4f && ratio < 0.7f);
			float t_seeds[32];
			int seed_count = 0;

			if (!use_ik) {

				float temp_S[32 * 32];
				int sobol_gen = generate_sobol_seeds(map, dim, temp_S, 32, seed + it);
				int num_sobol = num_ik;
				for (int k = 0; k < num_sobol && k < sobol_gen; ++k) {
					const float* s = temp_S + k * 32;
					float t_s = map.pointToT(s);
					t_seeds[seed_count++] = t_s;
				}
			}
			else {
				float angles_ccd[32] = { 0 };
				float lengths_ccd[32];
				if (cost.variableLen) {
					float len_low = map.low[n];
					float len_high = map.high[n];
					float avg_len = (len_low + len_high) * 0.5f;
					for (int i = 0; i < n; ++i) lengths_ccd[i] = avg_len;
				}
				else {
					for (int i = 0; i < n; ++i) lengths_ccd[i] = 1.0f;
				}
				ccd_ik(cost.targetX, cost.targetY, lengths_ccd, n, angles_ccd, 10);
				float angles_fabrik[32] = { 0 };
				float lengths_fabrik[32];
				if (cost.variableLen) {
					for (int i = 0; i < n; ++i) lengths_fabrik[i] = lengths_ccd[i];
				}
				else {
					for (int i = 0; i < n; ++i) lengths_fabrik[i] = 1.0f;
				}
				float targetX_fab = cost.targetX;
				float targetY_fab = cost.targetY;
				for (int iter_fab = 0; iter_fab < 3; ++iter_fab) {
					float prevX = targetX_fab;
					float prevY = targetY_fab;
					for (int j = n - 1; j >= 0; --j) {
						float len = lengths_fabrik[j];
						float angle_to_target = atan2f(prevY, prevX);
						angles_fabrik[j] = angle_to_target;
						float s_val, c_val;
						FABE13_SINCOS(&angle_to_target, &s_val, &c_val, 1);
						prevX = prevX - len * c_val;
						prevY = prevY - len * s_val;
					}
				}

				if (prefer_extended) {
					{
						float q_ccd[32];
						for (int i = 0; i < n; ++i) q_ccd[i] = angles_ccd[i];
						if (cost.variableLen) {
							for (int i = 0; i < n; ++i) q_ccd[n + i] = lengths_ccd[i];
						}
						float t_ccd = map.pointToT(q_ccd);
						t_seeds[seed_count++] = t_ccd;
					}
					unsigned st_ik = seed + it + 222;
					int remaining = num_ik - 1;
					for (int v = 0; v < remaining; ++v) {
						float noisy_angles[32];
						float noisy_lengths[32];
						for (int i = 0; i < n; ++i) {
							st_ik ^= st_ik << 13;
							st_ik ^= st_ik >> 17;
							st_ik ^= st_ik << 5;
							float rnd = static_cast<float>(st_ik & 0xFFFFFF) * 5.9604645e-8f;
							noisy_angles[i] = angles_ccd[i] + (2.0f * rnd - 1.0f) * 0.1f;
							const float lo = (i == 0) ? -1.0471975511965976f : -2.6179938779914944f;
							const float hi = 2.6179938779914944f;
							if (noisy_angles[i] < lo) noisy_angles[i] = lo;
							if (noisy_angles[i] > hi) noisy_angles[i] = hi;
						}
						if (cost.variableLen) {
							for (int i = 0; i < n; ++i) {
								st_ik ^= st_ik << 13;
								st_ik ^= st_ik >> 17;
								st_ik ^= st_ik << 5;
								float rnd = static_cast<float>(st_ik & 0xFFFFFF) * 5.9604645e-8f;
								noisy_lengths[i] = lengths_ccd[i] + (2.0f * rnd - 1.0f) * 0.05f;
								if (noisy_lengths[i] < 0.5f) noisy_lengths[i] = 0.5f;
								if (noisy_lengths[i] > 2.0f) noisy_lengths[i] = 2.0f;
							}
						}
						float q_temp[32];
						for (int i = 0; i < n; ++i) q_temp[i] = noisy_angles[i];
						if (cost.variableLen) {
							for (int i = 0; i < n; ++i) q_temp[n + i] = noisy_lengths[i];
						}
						float t_temp = map.pointToT(q_temp);
						t_seeds[seed_count++] = t_temp;
					}
				}
				else if (prefer_compact) {
					{
						float q_fabrik[32];
						for (int i = 0; i < n; ++i) q_fabrik[i] = angles_fabrik[i];
						if (cost.variableLen) {
							for (int i = 0; i < n; ++i) q_fabrik[n + i] = lengths_fabrik[i];
						}
						float t_fabrik = map.pointToT(q_fabrik);
						t_seeds[seed_count++] = t_fabrik;
					}
					unsigned st_ik = seed + it + 222;
					int remaining = num_ik - 1;
					for (int v = 0; v < remaining; ++v) {
						float noisy_angles[32];
						float noisy_lengths[32];
						for (int i = 0; i < n; ++i) {
							st_ik ^= st_ik << 13;
							st_ik ^= st_ik >> 17;
							st_ik ^= st_ik << 5;
							float rnd = static_cast<float>(st_ik & 0xFFFFFF) * 5.9604645e-8f;
							noisy_angles[i] = angles_fabrik[i] + (2.0f * rnd - 1.0f) * 0.1f;
							const float lo = (i == 0) ? -1.0471975511965976f : -2.6179938779914944f;
							const float hi = 2.6179938779914944f;
							if (noisy_angles[i] < lo) noisy_angles[i] = lo;
							if (noisy_angles[i] > hi) noisy_angles[i] = hi;
						}
						if (cost.variableLen) {
							for (int i = 0; i < n; ++i) {
								st_ik ^= st_ik << 13;
								st_ik ^= st_ik >> 17;
								st_ik ^= st_ik << 5;
								float rnd = static_cast<float>(st_ik & 0xFFFFFF) * 5.9604645e-8f;
								noisy_lengths[i] = lengths_fabrik[i] + (2.0f * rnd - 1.0f) * 0.05f;
								if (noisy_lengths[i] < 0.5f) noisy_lengths[i] = 0.5f;
								if (noisy_lengths[i] > 2.0f) noisy_lengths[i] = 2.0f;
							}
						}
						float q_temp[32];
						for (int i = 0; i < n; ++i) q_temp[i] = noisy_angles[i];
						if (cost.variableLen) {
							for (int i = 0; i < n; ++i) q_temp[n + i] = noisy_lengths[i];
						}
						float t_temp = map.pointToT(q_temp);
						t_seeds[seed_count++] = t_temp;
					}
				}
			}
			float optimized_points[32][32];
			float optimized_f[32];
			float optimized_t[32];
			for (int s = 0; s < seed_count; ++s) {
				float t_cur = t_seeds[s];
				float f_opt = evalAt(t_cur);
				memcpy(optimized_points[s], q_local, dim * sizeof(float));
				optimized_f[s] = f_opt;
				optimized_t[s] = map.pointToT(q_local);
			}
			for (int s = 0; s < seed_count; ++s) {
				float t_opt = optimized_t[s];
				const float interval_size = fmaf(0.00031f, static_cast<float>(dim), 0.0f);
				float t1 = t_opt - interval_size * 0.5f;
				float t2 = t_opt + interval_size * 0.5f;
				if (t1 < 0.0f) t1 = 0.0f;
				if (t2 > 1.0f) t2 = 1.0f;
				float q1[32], q2[32];
				float x1, y1, x2, y2;
				map.map01ToPoint(t1, q1);
				float f1 = cost(q1, x1, y1);
				map.map01ToPoint(t2, q2);
				float f2 = cost(q2, x2, y2);
				IntervalND* I = new IntervalND(t1, t2, f1, f2);
				I->i1 = t_to_idx(t1);
				I->i2 = t_to_idx(t2);
				I->diam = map.block_diameter(I->i1, I->i2);
				I->compute_span_level(map);
				I->set_metric(I->diam);
				update_pockets_and_Mmax(I);
				I->ChangeCharacteristic(fmaf(r_eff, Mmax, 0.0f));
				const float boost = fmaf(0.01f, dim_f, 0.85f);
				I->R = fmaf(I->R, boost, 0.0f);
				H.emplace_back(I);
				std::push_heap(H.begin(), H.end(), ComparePtrND);
			}
			no_improve = 0;
		}
		const float exp_arg = fmaf(B_dim, p, 0.0f);
		const float exp_arg__ = fmaf(B_dim__, p, 0.0f);
		adaptive_coeff = fmaf(-fmaf(exp_arg, fmaf(exp_arg, fmaf(exp_arg, fmaf(exp_arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), adaptive_coeff_addition, A_dim);
		float first_sqrt = sqrtf(fmaf(1.0f / dim_f, 2.0f, 0.0f));
		float second_sqrt = sqrtf(fmaf(1.0f / (dim_f + 7.0f), 5.0f, 0.0f));
		float third_sqrt = sqrtf(fmaf(1.0f / (dim_f + 7.0f), 9.0f, 0.0f));
		float fourth_sqrt = sqrtf(fmaf(1.0f / (dim_f + 7.0f), 6.5f, 0.0f));
		float rr = sqrtf(fmaf(-p, 1.0f, 1.0f)), xx = p * p, tt = fmaf(500.0f, p, -486.95472f);
		float adaptive_coeff_ = (p < 0.95f) ? fmaf(fmaf(first_sqrt, xx, 0.0f), 0.0130349902f, fmaf(-0.04f, p, fmaf(fmaf(first_sqrt, rr, 0.0f), 0.15f, 1.1f))) : (p < 0.97390944f) ? fmaf(second_sqrt, rr, 0.9396f) : (p < 0.97590944f) ? fmaf(fmaf(fmaf(fmaf(third_sqrt, tt, 0.0f), tt, 0.0f), fmaf(-2.0f, tt, 3.0f), 0.0f), fmaf(0.25f, rr, -0.0396f), fmaf(fmaf(third_sqrt, rr, 0.0f), 0.75f, 0.9396f)) : fmaf(fourth_sqrt, rr, 0.925f);
		adaptive_coeff__ = fmaf(fmaf(exp_arg__, fmaf(exp_arg__, fmaf(exp_arg__, fmaf(exp_arg__, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), adaptive_coeff_addition__, 2.0f - A_dim__);
		const float arg_for_T = fmaf(p, 4.5f, 2.475f);
		const float exp_argument = fmaf(-0.2925f, dim_f, 0.0f);
		const float exp2_exp_arg = fmaf(fmaf(exp_argument, 0.69314718055994530941723212145818f, 0.0f), fmaf(fmaf(exp_argument, 0.69314718055994530941723212145818f, 0.0f), fmaf(fmaf(exp_argument, 0.69314718055994530941723212145818f, 0.0f), fmaf(fmaf(exp_argument, 0.69314718055994530941723212145818f, 0.0f), 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f);
		const float A = fmaf(39.995f, exp2_exp_arg, 0.0f);
		const int T = static_cast<int>(fmaf(fmaf(fmaf(1.0f / fmaf(arg_for_T, fmaf(arg_for_T, fmaf(arg_for_T, fmaf(arg_for_T, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 2.0f), 1.0498f, -0.04978f), fmaf(sqrtf(dim_f), 4.88f, 0.0f), 0.0f), fmaf(-(exp2_exp_arg - 1.0f), A, A), 0.0f));
		std::pop_heap(H.begin(), H.end(), ComparePtrND);
		IntervalND* cur = H.back();
		H.pop_back();
		const float x1 = cur->x1;
		const float x2 = cur->x2;
		const float y1 = cur->y1;
		const float y2 = cur->y2;
		float m = fmaf(r_eff, Mmax, 0.0f);
		float tNew = step(m, x1, x2, y1, y2, dim_f, r_eff);
		const float bestFOld = bestF;
		const float fNew = evalAt(tNew);
		IntervalND* L = new IntervalND(x1, tNew, y1, fNew);
		IntervalND* Rv = new IntervalND(tNew, x2, fNew, y2);
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
		if (Mloc > Mmax)
			Mmax = Mloc;
		m = fmaf(r_eff, Mmax, 0.0f);
		if (adaptive) {
			const float len1 = fmaf(tNew, 1.0f, -x1);
			const float len2 = fmaf(x2, 1.0f, -tNew);
			if (fmaf(len1, 1.0f, len2) == dmax) {
				dmax = fmaxf(len1, len2);
				for (auto pI : H) {
					const float Ls = fmaf(pI->x2, 1.0f, -pI->x1);
					if (Ls > dmax)
						dmax = Ls;
				}
			}
			if ((p > 0.7f && !(it % 3) && dmax < 0.7f) || p > 0.9f) {
				const float alpha = p * p;
				const float beta = fmaf(-alpha, 1.0f, 2.0f);
				const float MULT = (1.0f / dmax) * Mmax;
				const float global_coeff = fmaf(MULT, r_eff, -MULT);
				const float GF = beta * global_coeff;
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
					if (Mloc > fmaf(adaptive_coeff_, prevMmax, 0.0f)) {
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
				if (Mloc > fmaf(adaptive_coeff_, prevMmax, 0.0f)) {
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
		_mm_prefetch((const char*)H[0], _MM_HINT_T0);
		_mm_prefetch((const char*)H[1], _MM_HINT_T0);
		if ((stagnation || bestF < fmaf(fmaf(bestFOld, 0.6f, 0.0f), adaptive_coeff__, 0.0f)) && it - last_send_T >= send_interval_T) {
			last_send_T = it;
			progress_outgoing();
			unsigned intervals_to_send = dim < 12 ? static_cast<unsigned>(sqrtf(fmaf(dim_f, 5.5f, 0.0f))) : 7u;
			const float alpha = 0.63f - sqrtf(-0.113f * (p - 1.0f));
			const float beta = 1.0f - alpha; 
			const float w_pos = 3.085f * (-sqrtf(0.007f * p) + 1.0f); 
			const float w_size = 5.085f - w_pos;  
			int num_bins = static_cast<int>(1.0f / (1.0f - p));
			std::vector<float> pos_metric(n); 
			std::vector<float> size_metric(n);   
			std::vector<float> R_val(n);
			float R_max = -FLT_MAX, R_min = FLT_MAX;
			float size_max = -FLT_MAX, size_min = FLT_MAX;
			float pos_max = -FLT_MAX, pos_min = FLT_MAX;
			for (size_t idx = 0; idx < n; ++idx) {
				IntervalND* I = H[idx];
				float center = (I->x1 + I->x2) * 0.5f;
				float len = I->x2 - I->x1;
				float size = len * (float)(1 << I->span_level);
				if (p < 0.95f) {
					float bin = floorf(center * num_bins);
					if (bin >= num_bins) bin = num_bins - 1;
					pos_metric[idx] = bin / (num_bins - 1.0f);
				}
				else {
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
			auto norm = [&](float val, float minv, float maxv) -> float {
				return (val - minv) / (maxv - minv);
				};
			std::vector<float> R_norm(n), pos_norm(n), size_norm(n);
			for (size_t idx = 0; idx < n; ++idx) {
				R_norm[idx] = norm(R_val[idx], R_min, R_max);
				pos_norm[idx] = norm(pos_metric[idx], pos_min, pos_max);
				size_norm[idx] = norm(size_metric[idx], size_min, size_max);
			}
			std::vector<int> selected;
			std::vector<bool> used(n, false);
			selected.push_back(0);
			used[0] = true;
			while (selected.size() < intervals_to_send) {
				int best_idx = -1;
				float best_score = -FLT_MAX;
				for (size_t idx = 0; idx < n; ++idx) {
					if (used[idx]) continue;
					float min_dist = FLT_MAX;
					for (int s : selected) {
						float d2 = 0.0f;
						float d = pos_norm[idx] - pos_norm[s]; d2 += w_pos * d * d;
						d = size_norm[idx] - size_norm[s]; d2 += w_size * d * d;
						if (d2 < min_dist) min_dist = d2;
					}
					float novelty = sqrtf(min_dist) / sqrtf(w_pos + w_size);
					float score = alpha * R_norm[idx] + beta * novelty;
					if (score > best_score) {
						best_score = score;
						best_idx = static_cast<int>(idx);
					}
				}
				if (best_idx == -1) break;
				selected.push_back(best_idx);
				used[best_idx] = true;
			}
			MultiCrossMsg out;
			out.count = static_cast<unsigned>(selected.size());
			float* dest = out.intervals;
			for (unsigned s = 0; s < out.count; ++s) {
				IntervalND* Tt = H[selected[s]];
				dest[0] = Tt->x1;
				dest[1] = 0.0f;
				dest[2] = Tt->x2;
				dest[3] = 0.0f;
				dest[4] = Tt->R;
				dest += 5;
			}
			const size_t iterations = std::bit_width(static_cast<size_t>(world - 1));
			bool active = true;
			const bool invert_T = (static_cast<int>(fmaf(static_cast<float>(exchange_counter_T), 1.0f, 1.0f)) & 1);
			size_t ii2 = 0u;
			while (ii2 < iterations && active) {
				const size_t step = 1ULL << ii2;
				const int partner = rank ^ static_cast<int>(step);
				if (partner < world) {
					const bool am_sender = ((!!(rank & static_cast<int>(step))) ^ invert_T);
					if (am_sender) {
						g_pendingMulti.emplace_back(*g_world, partner, out);
						if (g_pendingMulti.size() > MAX_INFLIGHT_MULTI) {
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
		if (bestF < fmaf(bestFOld, adaptive_coeff__, 0.0f) && it - last_send_best >= send_interval_best) {
			last_send_best = it;
			progress_outgoing();
			BestSolutionMsg out;
			out.bestF = bestF;
			out.bestX = bestX;
			out.bestY = bestY;
			out.dim = static_cast<unsigned>(bestQ.size());
			memcpy(out.bestQ, bestQ.data(), bestQ.size() * sizeof(float));
			const size_t iterations = std::bit_width(static_cast<size_t>(world - 1));
			bool active = true;
			const bool invert_T = (static_cast<int>(fmaf(static_cast<float>(exchange_counter), 1.0f, 1.0f)) & 1);
			size_t ii2 = 0u;
			while (ii2 < iterations && active) {
				const size_t step = 1ULL << ii2;
				const int partner = rank ^ static_cast<int>(step);
				if (partner < world) {
					const bool am_sender = ((!!(rank & static_cast<int>(step))) ^ invert_T);
					if (am_sender) {
						g_pendingBest.emplace_back(*g_world, partner, out);
						if (g_pendingBest.size() > MAX_INFLIGHT_BEST) {
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
		while (g_world->iprobe(boost::mpi::any_source, 0)) {
			MultiCrossMsg in;
			g_world->recv(boost::mpi::any_source, 0, in);
			const MultiCrossMsg& mX = in;
			unsigned ii = 0u;
			while (ii < mX.count) {
				const float* d = &mX.intervals[ii * 5];
				float sx = d[0];
				float ex = d[2];
				alignas(16) float tmp[32];
				float tx;
				float ty;
				map.map01ToPoint(sx, tmp);
				const float y1i = cost(tmp, tx, ty);
				map.map01ToPoint(ex, tmp);
				const float y2i = cost(tmp, tx, ty);
				IntervalND* inj = new IntervalND(sx, ex, y1i, y2i);
				inj->i1 = t_to_idx(sx);
				inj->i2 = t_to_idx(ex);
				inj->diam = map.block_diameter(inj->i1, inj->i2);
				inj->compute_span_level(map);
				inj->set_metric(inj->diam);
				update_pockets_and_Mmax(inj);
				inj->ChangeCharacteristic(fmaf(r_eff, Mmax, 0.0f));
				_mm_prefetch((const char*)H[0], _MM_HINT_T0);
				_mm_prefetch((const char*)H[1], _MM_HINT_T0);
				if (inj->R > fmaf(adaptive_coeff, H.front()->R, 0.0f) || stagnation) {
					const float poly = fmaf(fmaf(p, 0.69314718055994530941723212145818f, 0.0f), fmaf(fmaf(p, 0.69314718055994530941723212145818f, 0.0f), fmaf(fmaf(p, 0.69314718055994530941723212145818f, 0.0f), fmaf(fmaf(p, 0.69314718055994530941723212145818f, 0.0f), 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f) - 1.0f;
					const float kf = stagnation ? fmaf(0.5819767068693265f, poly, 0.4f) : fmaf(0.3891860241215959f, poly, 0.5f);
					const float exp_arg = fmaf(B_dim, fmaf(1.0f / static_cast<float>(mX.count - 1u), static_cast<float>(ii), 0.0f), 0.0f);
					float adaptive_coeff_clone = fmaf(-fmaf(exp_arg, fmaf(exp_arg, fmaf(exp_arg, fmaf(exp_arg, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), adaptive_coeff_addition, A_dim_clone);
					inj->R = fmaf(d[4], fmaf(kf, adaptive_coeff_clone, 0.0f), 0.0f);
					H.emplace_back(inj);
					std::push_heap(H.begin(), H.end(), ComparePtrND);
				}
				++ii;
			}
		}
		while (g_world->iprobe(boost::mpi::any_source, 2)) {
			BestSolutionMsg bm;
			g_world->recv(boost::mpi::any_source, 2, bm);
			if (bm.bestF < fmaf(bestF, adaptive_coeff__, 0.0f) || stagnation) {
				_mm_prefetch((const char*)H[0], _MM_HINT_T0);
				_mm_prefetch((const char*)H[1], _MM_HINT_T0);
				if (bm.bestF < bestF) {
					bestF = bm.bestF;
					bestX = bm.bestX;
					bestY = bm.bestY;
					bestQ.assign(bm.bestQ, bm.bestQ + bm.dim);
				}
			}
		}
		IntervalND* const top = H.front();
		const float interval_len = dim > 1 ? fmaf(top->x2, 1.0f, -top->x1) : top->diam;
		if (/*(dim > 1 ? exp2f((1.0f / dim_f) * log2f(interval_len)) : interval_len) < eps || */it == maxIter - 1) {
			memcpy(q_local, bestQ.data(), static_cast<size_t>(dim) * sizeof(float));
			float x_final = bestX;
			float y_final = bestY;
			float f_final = bestF;
			const int last = n - 1;
			const float lo = (last == 0) ? -1.0471975511965977461542144610932f : -2.6179938779914943653855361527329f;
			const float hi = 2.6179938779914943653855361527329f;
			float bestLocF = f_final;
			float saved = q_local[last];
			float delta = 0.05f;
			while (delta >= 0.00625f) {
				int sgn = -1;
				while (sgn < 2) {
					float cand = fmaf(static_cast<float>(sgn), delta, saved);
					if (cand < lo) cand = lo;
					else if (cand > hi) cand = hi;
					const float backup = q_local[last];
					q_local[last] = cand;
					float x2;
					float y2;
					const float f2 = cost(q_local, x2, y2);
					if (f2 < bestLocF) {
						bestLocF = f2;
						x_final = x2;
						y_final = y2;
						saved = cand;
					}
					q_local[last] = backup;
					sgn += 2;
				}
				delta *= 0.5f;
			}
			if (bestLocF < f_final) {
				q_local[last] = saved;
				f_final = bestLocF;
				bestF = f_final;
				bestX = x_final;
				bestY = y_final;
				bestQ.assign(q_local, q_local + dim);
			}
			out_iterations = static_cast<size_t>(it);
			out_achieved_epsilon = interval_len;
			for (auto& s : g_pendingMulti) s.req.wait();
			for (auto& s : g_pendingBest) s.req.wait();
			g_pendingMulti.clear();
			g_pendingBest.clear();
			return;
		}
		++it;
	}
}

static __declspec(noalias) __forceinline float PivotCalculation(std::vector<IntervalND*>::iterator first, std::vector<IntervalND*>::iterator last) noexcept {
	const auto mid = first + ((last - first) >> 1);
	float pivot_value = NAN;
	if (last - first < 299)
		pivot_value = (*mid)->R;
	else {
		if ((*first)->R < (*mid)->R) {
			if ((*mid)->R < (*last)->R)
				pivot_value = (*mid)->R;
			else
				pivot_value = std::max((*first)->R, (*last)->R);
		}
		else {
			if ((*first)->R < (*last)->R)
				pivot_value = (*first)->R;
			else
				pivot_value = std::max((*mid)->R, (*last)->R);
		}
	}
	return pivot_value;
}

static __declspec(noalias) __forceinline void HoaraSort(std::vector<IntervalND*>::iterator first, std::vector<IntervalND*>::iterator last) noexcept {
	if (first >= last)
		return;
	const float pivot_value = PivotCalculation(first, last);
	auto left = first;
	auto right = last;
	do {
		while (left < last && (*left)->R < pivot_value)
			++left;
		while (right > first && (*right)->R > pivot_value)
			--right;
		if ((*left)->R == (*right)->R && left != right) {
			if ((*left)->R < (*(left + 1))->R)
				++left;
			else
				--right;
		}
		std::iter_swap(left, right);
	} while (left != right);
	if (last - first < 299) {
		HoaraSort(first, right);
		HoaraSort(left + 1, last);
	}
	else {
		oneapi::tbb::parallel_invoke([&first, &right]() { HoaraSort(first, right); }, [&left, &last]() { HoaraSort(left + 1, last); });
	}
}

extern "C" __declspec(dllexport) __declspec(noalias) void AGP_Manip2D(
	int nSegments,
	bool variableLengths,
	float minTheta,
	float targetX,
	float targetY,
	int maxIterPerBranch,
	float r,
	bool adaptiveMode,
	float epsilon,
	unsigned int seed,
	float baseLength,
	float stretchFactor,
	float** out_bestQ,
	size_t* out_bestQLen,
	float* out_bestX,
	float* out_bestY,
	float* out_bestF,
	size_t* out_iterations,
	float* out_achieved_epsilon) noexcept {
	Slab* slab = tls.local();
	slab->current = slab->base;
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
	const int rank = g_world->rank();
	const int world = g_world->size();
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>> bestQ;
	float bestF = FLT_MAX;
	float bx = 0.0f;
	float by = 0.0f;
	const float exp_arg_lvls = fmaf(-static_cast<float>(dim), 0.455f, 0.0f);
	const float exp_arg_lvl0 = fmaf(-static_cast<float>(dim), 0.08f, 0.0f);
	const int fine_lvls = static_cast<int>(fminf(fmaf(fmaf(exp_arg_lvls, fmaf(exp_arg_lvls, fmaf(exp_arg_lvls, fmaf(exp_arg_lvls, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), 30.0f, 6.125f), 13.0f));
	const int levels0 = static_cast<int>(fminf(fmaf(-fminf(fmaf(fmaf(exp_arg_lvl0, fmaf(exp_arg_lvl0, fmaf(exp_arg_lvl0, fmaf(exp_arg_lvl0, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f), 21.7f, -11.3f), 0.0f), 1.0f, static_cast<float>(fine_lvls)), 8.0f));
	const MortonND map0(dim, levels0, low.data(), high.data(), g_mc);
	std::vector<IntervalND*, boost::alignment::aligned_allocator<IntervalND*, 16u>> H_coarse;
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>> bestQ_coarse;
	float bestF_coarse = FLT_MAX;
	float bx_coarse = 0.0f;
	float by_coarse = 0.0f;
	size_t total_oi = 0u;
	float total_oe = 0.0f;
	size_t oi = 0u;
	float oe = 0.0f;
	const float M_arg = ldexpf(1.0f, -levels0);
	const float M_prior = fmaf(fmaf(variableLengths ? 2.8284271f : 2.0f, static_cast<float>(nSegments), 0.0f), fmaf(M_arg, fmaf(M_arg, fmaf(M_arg, fmaf(M_arg, fmaf(M_arg, 0.164056f, -0.098462f), 0.240884f), -0.351834f), 0.999996f), M_arg), 0.0f);
	agp_run_branch_mpi(map0, cost, maxIterPerBranch >> 1, r, adaptiveMode, epsilon, seed, H_coarse, bestQ_coarse, bestF_coarse, bx_coarse, by_coarse, oi, oe, M_prior);
	total_oi += oi;
	total_oe = oe;
	if (bestF_coarse < bestF) {
		bestF = bestF_coarse;
		bestQ = std::move(bestQ_coarse);
		bx = bx_coarse;
		by = by_coarse;
	}
	const MortonND map1(dim, fine_lvls, low.data(), high.data(), g_mc);
	std::vector<IntervalND*, boost::alignment::aligned_allocator<IntervalND*, 16u>> H_fine;
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>> bestQ_fine = bestQ;
	float bestF_fine = bestF;
	float bx_fine = bx;
	float by_fine = by;
	size_t oi_fine = 0u;
	float oe_fine = 0.0f;
	const float M_prior_fine = fmaf(1.0f / static_cast<float>(fine_lvls), fmaf(static_cast<float>(levels0), M_prior, 0.0f), 0.0f);
	HoaraSort(H_coarse.begin(), H_coarse.end() - 1);
	const float inv_dim = 1.0f / static_cast<float>(dim + 1);
	size_t ci = static_cast<size_t>(fmaf(static_cast<float>(H_coarse.size()), fmaf(inv_dim, fmaf(inv_dim, fmaf(inv_dim, fmaf(inv_dim, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f) + 1.0f, -0.7f));
	while (ci < H_coarse.size()) {
		const IntervalND* C = H_coarse[ci];
		alignas(16) float q1[32];
		alignas(16) float q2[32];
		float x1;
		float y1;
		float x2;
		float y2;
		map1.map01ToPoint(C->x1, q1);
		const float f1 = cost(q1, x1, y1);
		map1.map01ToPoint(C->x2, q2);
		const float f2 = cost(q2, x2, y2);
		IntervalND* I = new IntervalND(C->x1, C->x2, f1, f2);
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
	agp_run_branch_mpi(map1, cost, maxIterPerBranch >> 1, r, adaptiveMode, epsilon, seed, H_fine, bestQ_fine, bestF_fine, bx_fine, by_fine, oi_fine, oe_fine, M_prior_fine);
	total_oi += oi_fine;
	total_oe = oe_fine;
	if (bestF_fine < bestF) {
		bestF = bestF_fine;
		bestQ = std::move(bestQ_fine);
		bx = bx_fine;
		by = by_fine;
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
				if (in.bestF < best.bestF)
					best = in;
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

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline int AgpInit(int peanoLevel, float a, float b, float c, float d) noexcept {
	g_env = new boost::mpi::environment();
	g_world = new boost::mpi::communicator();
	//pending_requests = new std::vector<boost::mpi::request>;
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	const int rank = g_world->rank();
	const int world_size = g_world->size();
	if (world_size == 4)
		new (&gActiveMap) Peano2DMap(peanoLevel, a, b, c, d, rank & 3);
	g_mc.baseSeed = 0x9E3779B9u;
	if (rank == 0) {
		wchar_t buf[MAX_PATH]{};
		GetModuleFileNameW(nullptr, buf, MAX_PATH);
		std::wstring ws(buf);
		auto pos = ws.find_last_of(L"\\/");
		if (pos != std::wstring::npos)
			ws.resize(pos);
		int n = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, nullptr, 0, nullptr, nullptr);
		g_exeDirCache.resize(n, '\0');
		WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, g_exeDirCache.data(), n, nullptr, nullptr);
		if (!g_exeDirCache.empty() && g_exeDirCache.back() == '\0')
			g_exeDirCache.pop_back();
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

static __declspec(noalias) __forceinline float ShekelFunc(float x, float seed) noexcept {
	int i = 0;
	float st = seed;
	float r1;
	float r2;
	float res = 0.0f;
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

static __declspec(noalias) __forceinline float RastriginFunc(float x1, float x2) noexcept {
	const float t = fmaf(x1, x1, x2 * x2);
	float c1;
	float c2;
	FABE13_COS(6.28318530717958647692f * x1, c1);
	FABE13_COS(6.28318530717958647692f * x2, c2);
	return (t - fmaf(c1 + c2, 10.0f, -14.6f)) * fmaf(-t, 0.25f, 18.42f);
}

static __declspec(noalias) __forceinline float HillFunc(float x, float seed) noexcept {
	int j = 0;
	alignas(16) float ang[14];
	const float st_ang = 6.28318530717958647692f * x;
	while (j < 14) {
		ang[j] = st_ang * static_cast<float>(j + 1);
		++j;
	}
	alignas(16) float sv[14];
	alignas(16) float cv[14];
	FABE13_SINCOS(ang, sv, cv, 14);
	float state = seed;
	float r1;
	float r2;
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

static __declspec(noalias) __forceinline float GrishaginFunc(float x1, float x2, float seed) noexcept {
	int j = 0;
	alignas(16) float aj[8];
	alignas(16) float ak[8];
#pragma loop ivdep
	while (j < 8) {
		const float pj = 3.14159265358979323846f * static_cast<float>(j + 1);
		aj[j] = pj * x1;
		ak[j] = pj * x2;
		++j;
	}
	alignas(16) float sj[8];
	alignas(16) float cj[8];
	alignas(16) float sk[8];
	alignas(16) float ck[8];
	FABE13_SINCOS(aj, sj, cj, 8);
	FABE13_SINCOS(ak, sk, ck, 8);
	--j;
	float p1 = 0.0f;
	float p2 = 0.0f;
	float st = seed;
	float r1;
	float r2;
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

extern "C" __declspec(dllexport) __declspec(noalias) void AGP_1D(
	float global_iterations,
	float a,
	float b,
	float r,
	bool mode,
	float epsilon,
	float seed,
	float** out_data,
	size_t* out_len) noexcept {
	Slab* slab = tls.local();
	slab->current = slab->base;
	int counter = 0;
	const float initial_length = b - a;
	float dmax = initial_length;
	const float threshold_03 = 0.3f * initial_length;
	const float inv_threshold_03 = 1.0f / threshold_03;
	const float start_val = ShekelFunc(a, seed);
	float best_f = ShekelFunc(b, seed);
	float x_Rmax_1 = a;
	float x_Rmax_2 = b;
	float y_Rmax_1 = start_val;
	float y_Rmax_2 = best_f;
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
		const Interval1D* pro = R.back();
		const float new_x1 = pro->x1;
		const float new_x2 = pro->x2;
		const float len2 = new_x2 - new_point;
		const float len1 = new_point - new_x1;
		const float interval_len = (len1 < len2 ? len1 : len2);
		if (++counter == static_cast<int>(global_iterations) || interval_len < epsilon) {
			Extr.emplace_back(static_cast<float>(counter));
			Extr.emplace_back(interval_len);
			*out_len = Extr.size();
			*out_data = static_cast<float*>(CoTaskMemAlloc(sizeof(float) * (*out_len)));
			memcpy(*out_data, Extr.data(), sizeof(float) * (*out_len));
			return;
		}
		Interval1D* curr = new Interval1D(new_x1, new_point, pro->y1, new_value, 1.0f);
		Interval1D* curr1 = new Interval1D(new_point, new_x2, new_value, pro->y2, 1.0f);
		const float currM = curr->M > curr1->M ? curr->M : curr1->M;
		const size_t r_size = R.size();
		if (mode) {
			if (len2 + len1 == dmax) {
				dmax = len2 > len1 ? len2 : len1;
				for (auto pI : R) {
					const float L = pI->x2 - pI->x1;
					if (L > dmax)
						dmax = L;
				}
			}
			if ((threshold_03 > dmax && !(counter % 3)) || 10.0f * dmax < initial_length) {
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
		const Interval1D* top = R.front();
		x_Rmax_1 = top->x1;
		x_Rmax_2 = top->x2;
		y_Rmax_1 = top->y1;
		y_Rmax_2 = top->y2;
	}
}

extern "C" __declspec(dllexport) __declspec(noalias) void AGP_2D(
	float N,
	float global_iterations,
	float a,
	float b,
	float c,
	float d,
	float r,
	bool mode,
	float epsilon,
	float seed,
	float** out_data,
	size_t* out_len) noexcept {
	Slab* slab = tls.local();
	slab->current = slab->base;
	int counter = 0;
	int no_improve = 0;
	const int rank = g_world->rank();
	const int world_size = g_world->size();
	while (g_world->iprobe(boost::mpi::any_source, 0)) {
		MultiCrossMsg dummy;
		g_world->recv(boost::mpi::any_source, 0, dummy);
	}
	const float inv_divider = ldexpf(1.0f, -((gActiveMap.levels << 1) + 1));
	const float x_addition = (b - a) * inv_divider;
	const float y_addition = (d - c) * inv_divider;
	const float true_start = a + x_addition;
	const float true_end = b - x_addition;
	float x_Rmax_1 = true_start;
	float x_Rmax_2 = true_end;
	const float initial_length = x_Rmax_2 - x_Rmax_1;
	float dmax = initial_length;
	const float threshold_03 = 0.3f * initial_length;
	const float inv_threshold_03 = 1.0f / threshold_03;
	const float start_val = (rank % 3) ? RastriginFunc(true_end, d - y_addition) : RastriginFunc(true_start, c + y_addition);
	float best_f = (rank % 2) ? RastriginFunc(true_start, d - y_addition) : RastriginFunc(true_end, c + y_addition);
	float y_Rmax_1 = start_val;
	float y_Rmax_2 = best_f;
	std::vector<float, boost::alignment::aligned_allocator<float, 16u>> Extr;
	std::vector<Interval1D* __restrict, boost::alignment::aligned_allocator<Interval1D* __restrict, 16u>> R;
	Extr.reserve(static_cast<size_t>(global_iterations) << 2u);
	R.reserve(static_cast<size_t>(global_iterations) << 1u);
	R.emplace_back(new Interval1D(true_start, true_end, start_val, best_f, 2.0f));
	const Interval1D* __restrict top_ptr = R.front();
	float Mmax = R.front()->M;
	float m = r * Mmax;
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
				const float* d2 = &mX.intervals[ii * 5];
				float sx = d2[0];
				float ex = d2[2];
				if (ex > sx) {
					Interval1D* __restrict injected = new Interval1D(sx, ex, RastriginFunc(d2[0], d2[1]), RastriginFunc(d2[2], d2[3]), 2.0f);
					injected->ChangeCharacteristic(m);
					if (injected->R > 1.15f * top_ptr->R) {
						const float poly = fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, fmaf(p * 0.69314718055994530941723212145818f, 0.00833333377f, 0.0416666679f), 0.16666667f), 0.5f), 1.0f) - 1.0f;
						const float k = stagnation ? fmaf(0.5819767068693265f, poly, 0.3f) : fmaf(0.3491860241215959f, poly, 0.6f);
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
				float s_x1;
				float s_x2;
				float e_x1;
				float e_x2;
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
					if (i2 != rank)
						g_world->isend(i2, 0, out);
					++i2;
				}
			}
		}
		if (want_term) {
			if (!rank) {
				Extr.emplace_back(static_cast<float>(counter));
				Extr.emplace_back(interval_len);
				*out_len = Extr.size();
				*out_data = static_cast<float* __restrict>(CoTaskMemAlloc(sizeof(float) * (*out_len)));
				memcpy(*out_data, Extr.data(), sizeof(float) * (*out_len));
			}
			return;
		}
		const float new_point = step(m, x_Rmax_1, x_Rmax_2, y_Rmax_1, y_Rmax_2, 2.0f, r);
		float new_x1_val;
		float new_x2_val;
		HitTest2D_analytic(new_point, new_x1_val, new_x2_val);
		const float new_value = RastriginFunc(new_x1_val, new_x2_val);
		if (new_value < best_f) {
			best_f = new_value;
			Extr.emplace_back(best_f);
			Extr.emplace_back(new_x1_val);
			Extr.emplace_back(new_x2_val);
			no_improve = 0;
		}
		else
			++no_improve;
		std::pop_heap(R.begin(), R.end(), ComparePtr1D);
		Interval1D* __restrict intermediate = R.back();
		const float segment_x1 = intermediate->x1;
		const float segment_x2 = intermediate->x2;
		const float len2 = segment_x2 - new_point;
		const float len1 = new_point - segment_x1;
		Interval1D* __restrict curr = new Interval1D(segment_x1, new_point, intermediate->y1, new_value, 2.0f);
		Interval1D* __restrict curr1 = new Interval1D(new_point, segment_x2, new_value, intermediate->y2, 2.0f);
		const float currM = (std::max)(curr->M, curr1->M);
		const size_t r_size = R.size();
		if (mode) {
			if (len2 + len1 == dmax) {
				dmax = (std::max)(len1, len2);
				for (auto pI : R) {
					const float L = pI->x2 - pI->x1;
					if (L > dmax)
						dmax = L;
				}
			}
			if ((threshold_03 > dmax && !(counter % 3)) || 10.0f * dmax < initial_length) {
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
		float tx;
		float ty;
		int maxIter;
		float r;
		unsigned adaptive;
		float eps;
		unsigned seed;
		float baseLength;
		float stretchFactor;

		template <typename Archive>
		void serialize(Archive& ar, unsigned int) {
				ar& nSegments& varLen& minTheta& tx& ty& maxIter& r& adaptive& eps& seed& baseLength& stretchFactor;
		}
};

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline void AgpStartManipND(
	int nSegments,
	bool variableLengths,
	float minTheta,
	float targetX,
	float targetY,
	int maxIterPerBranch,
	float r,
	bool adaptiveMode,
	float epsilon,
	unsigned int seed,
	float baseLength,
	float stretchFactor) noexcept {
	RunParams p;
	p.nSegments = nSegments;
	p.varLen = static_cast<unsigned>(variableLengths);
	p.minTheta = minTheta;
	p.tx = targetX;
	p.ty = targetY;
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
	float bx;
	float by;
	float bf;
	size_t oi;
	float oa;
	while (true) {
		if (g_world->iprobe(0, 1)) {
			g_world->recv(0, 1, p);
			AGP_Manip2D(p.nSegments, static_cast<bool>(p.varLen), p.minTheta, p.tx, p.ty, p.maxIter, p.r, static_cast<bool>(p.adaptive), p.eps, p.seed, p.baseLength, p.stretchFactor, &q, &qlen, &bx, &by, &bf, &oi, &oa);
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

extern "C" __declspec(dllexport) __declspec(noalias) __forceinline void AGP_Free(float* p) noexcept {
	CoTaskMemFree(p);
}

extern "C" __declspec(dllexport) __declspec(noalias) void RunPythonOptimization(
	const char* backend,
	int n_seg,
	bool var_len,
	float min_theta,
	float tx,
	float ty,
	int levels,
	int max_iter,
	float r_param,
	float eps,
	bool adaptive,
	float baseLength,
	float stretchFactor,
	float* __restrict out_bestF,
	float* __restrict out_bestX,
	float* __restrict out_bestY,
	float* __restrict out_angles,
	float* __restrict out_lengths,
	int* __restrict out_iterations,
	float* __restrict out_eps,
	float* __restrict out_micros) noexcept {
	pybind11::gil_scoped_acquire gil;
	pybind11::dict result;
	if (backend && strcmp(backend, "optuna") == 0) {
		result = (*g_pyOptimizerBridge).attr("run_optuna")(n_seg, var_len, min_theta, tx, ty, max_iter, baseLength, stretchFactor);
	}
	else {
		result = (*g_pyOptimizerBridge).attr("run_iopt")(n_seg, var_len, min_theta, tx, ty, levels, max_iter, r_param, eps, adaptive, baseLength, stretchFactor);
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
		const float* __restrict src = angles_vec.data();
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
			const float* __restrict srcL = lengths_vec.data();
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
