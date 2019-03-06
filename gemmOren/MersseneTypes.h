#pragma once
#include <cuda_runtime.h>

// this is not a finite field type, but an unsigned int type 
//to measure peak possible performance
struct SUint32
{
	typedef unsigned int basic_t;
	typedef SUint32 Accum_t;
	__host__ __device__
		SUint32() : _v(0) {}
	__host__ __device__
		SUint32(basic_t v) : _v(v) {}
	basic_t _v;

	__host__ __device__
		SUint32 operator+(const SUint32& other)
	{
		SUint32 res(_v + other._v);
		return res;
	}
	__host__ __device__
		SUint32& operator+=(const SUint32& other)
	{
		_v += other._v;
		return *this;
	}

	friend
		__host__ __device__
		bool operator==(const SUint32& v1, const SUint32& v2)
	{
		return v1._v == v2._v;
	}

	friend
		__host__ __device__
		SUint32 operator*(const SUint32& v1, const SUint32& v2)
	{
		SUint32 res(v1._v * v2._v);
		return res;
	}
};


// implemtation of Merssene31 using modulu. used for testing
struct SMersenne31Naive
{
	typedef unsigned int basic_t;
	static const basic_t p = (1U << 31) - 1;
	typedef SMersenne31Naive Accum_t;
	__host__ __device__
		SMersenne31Naive() : _v(0) {}
	__host__ __device__
		SMersenne31Naive(basic_t v) : _v(v % p) {}
	basic_t _v;

	__host__ __device__
		SMersenne31Naive operator+(const SMersenne31Naive& other)
	{
		SMersenne31Naive res(_v + other._v);
		return res;
	}
	__host__ __device__
		SMersenne31Naive& operator+=(const SMersenne31Naive& other)
	{
		_v = (_v + other._v) % p;
		return *this;
	}

	friend
		__host__ __device__
		bool operator==(const SMersenne31Naive& v1, const SMersenne31Naive& v2)
	{
		return v1._v == v2._v;
	}

	friend
		__host__ __device__
		SMersenne31Naive operator*(const SMersenne31Naive& v1, const SMersenne31Naive& v2)
	{
		uint64_t long_res = (uint64_t)(v1._v) * (uint64_t)(v2._v);
		SMersenne31Naive res((basic_t)(long_res % p));
		return res;
	}
};

// implemtation of Merssene31 using the classic acceleration
struct SMersenne31Classic
{
	typedef unsigned int basic_t;
	static const basic_t p = (1U << 31) - 1;
	typedef SMersenne31Classic Accum_t;
	__host__ __device__
		SMersenne31Classic() : _v(0) {}
	__host__ __device__
		SMersenne31Classic(basic_t v) : _v(v < p ? v : v - p) {}
	basic_t _v;

	__host__ __device__
		SMersenne31Classic operator+(const SMersenne31Classic& other)
	{
		SMersenne31Classic res(_v + other._v);
		return res;
	}
	__host__ __device__
		SMersenne31Classic& operator+=(const SMersenne31Classic& other)
	{
		basic_t uint_res = _v + other._v;
		_v = uint_res < p ? uint_res : uint_res - p;
		return *this;
	}

	friend
		__host__ __device__
		bool operator==(const SMersenne31Classic& v1, const SMersenne31Classic& v2)
	{
		return v1._v == v2._v;
	}

	friend
		__host__ __device__
		SMersenne31Classic operator*(const SMersenne31Classic& v1, const SMersenne31Classic& v2)
	{
		//SMersenne31Classic answer(v1._v + v2._v);
		//return answer;
		SMersenne31Classic answer;

		uint64_t multLong = (uint64_t)(v1._v) * (uint64_t)(v2._v);

		//get the bottom 31 bit
		basic_t bottom = multLong & p;

		//get the top 31 bits
		basic_t top = (basic_t)(multLong >> 31);

		answer._v = bottom + top;

		//maximim the value of 2p-2
		if (answer._v >= p)
			answer._v -= p;

		return answer;
	}
};
// implemtation of Merssene31 using hi-lo integer intrinsic.
/*struct SMersenne31HiLo
{
	typedef unsigned int basic_t;
	static const basic_t p = (1U << 31) - 1;
	typedef SMersenne31HiLo Accum_t;
	__host__ __device__
		SMersenne31HiLo() : _v(0) {}
	__host__ __device__
		SMersenne31HiLo(basic_t v) : _v(v < p ? v : v - p) {}
	basic_t _v;

	__host__ __device__
		SMersenne31HiLo operator+(const SMersenne31HiLo& other)
	{
		SMersenne31HiLo res(_v + other._v);
		return res;
	}
	__host__ __device__
		SMersenne31HiLo& operator+=(const SMersenne31HiLo& other)
	{
		basic_t uint_res = _v + other._v;
		_v = uint_res < p ? uint_res : uint_res - p;
		return *this;
	}

	friend
		__host__ __device__
		bool operator==(const SMersenne31HiLo& v1, const SMersenne31HiLo& v2)
	{
		return v1._v == v2._v;
	}

	friend
		__device__
		SMersenne31HiLo operator*(const SMersenne31HiLo& v1, const SMersenne31HiLo& v2)
	{
		basic_t lo = v1._v * v2._v;
		basic_t hi = __umulhi(v1._v, v2._v);
		
		if (lo >= p) lo -= p;
		basic_t sum = lo + hi * 2;
		return SMersenne31HiLo(sum);
	}
};*/

///////////////////////////////////////////////////////////////////
// merssene61 code
// implemtation of Merssene61 using modulu. used for testing
struct SMersenne61Naive
{
	typedef unsigned long long basic_t;
	static const basic_t p = (1ULL << 61) - 1;
	typedef SMersenne61Naive Accum_t;
	__host__ __device__
		SMersenne61Naive() : _v(0) {}
	__host__ __device__
		SMersenne61Naive(basic_t v) : _v(v % p) {}
	basic_t _v;

	__host__ __device__
		SMersenne61Naive operator+(const SMersenne61Naive& other)
	{
		SMersenne61Naive res(_v + other._v);
		return res;
	}
	__host__ __device__
		SMersenne61Naive& operator+=(const SMersenne61Naive& other)
	{
		_v = (_v + other._v) % p;
		return *this;
	}

	friend
		__host__ __device__
		bool operator==(const SMersenne61Naive& v1, const SMersenne61Naive& v2)
	{
		return v1._v == v2._v;
	}

	friend
		__host__ __device__
		SMersenne61Naive operator*(const SMersenne61Naive& v1, const SMersenne61Naive& v2)
	{
		basic_t high, low;
		#ifdef __CUDA_ARCH__
		low = v1._v * v2._v;
		high = __umul64hi(v1._v, v2._v);
#else
		low = _mulx_u64(v1._v, v2._v, &high);
#endif
		return SMersenne61Naive(((low % p) + (high << 3)) % p);		
	}
};

// implemtation of Merssene61 using classic acceleration. 
struct SMersenne61Classic
{
	typedef unsigned long long basic_t;
	static const basic_t p = (1ULL << 61) - 1;
	typedef SMersenne61Classic Accum_t;
	__host__ __device__
		SMersenne61Classic() : _v(0) {}
	__host__ __device__
		SMersenne61Classic(basic_t v) : _v(v < p ? v : v - p) {}
	basic_t _v;

	__host__ __device__
		SMersenne61Classic operator+(const SMersenne61Classic& other)
	{
		SMersenne61Classic res(_v + other._v);
		return res;
	}
	__host__ __device__
		SMersenne61Classic& operator+=(const SMersenne61Classic& other)
	{
		basic_t sum = _v + other._v;
		_v = sum < p ? sum : sum - p;
		return *this;
	}

	friend
		__host__ __device__
		bool operator==(const SMersenne61Classic& v1, const SMersenne61Classic& v2)
	{
		return v1._v == v2._v;
	}

	friend
		__host__ __device__
		SMersenne61Classic operator*(const SMersenne61Classic& v1, const SMersenne61Classic& v2)
	{
		basic_t high, low;
#ifdef __CUDA_ARCH__
		low = v1._v * v2._v;
		high = __umul64hi(v1._v, v2._v);
#else
		low = _mulx_u64(v1._v, v2._v, &high);
#endif

		basic_t low61 = (low & p);
		basic_t low62to64 = (low >> 61);
		basic_t highShift3 = (high << 3);

		basic_t res = low61 + low62to64 + highShift3;

		if (res >= p)
			res -= p;

		SMersenne61Classic answer;
		answer._v = res;
		return res;
	}
};

// implemtation of Merssene61 using a multiple of the prime. faster then classic
struct SMersenne61Gpu
{
	typedef unsigned long long basic_t;
	static const basic_t p = (1ULL << 61) - 1;
	static const basic_t large_p = p << 2;
	typedef SMersenne61Gpu Accum_t;
	__host__ __device__
		SMersenne61Gpu() : _v(0) {}
	__host__ __device__
		SMersenne61Gpu(basic_t v) : _v(v < large_p ? v : v - large_p) {}
	basic_t _v;

	__host__ __device__
		SMersenne61Gpu operator+(const SMersenne61Gpu& other)
	{
		SMersenne61Gpu res(_v + other._v);
		return res;
	}
	__host__ __device__
		SMersenne61Gpu& operator+=(const SMersenne61Gpu& other)
	{
		basic_t sum = _v + other._v;
		_v = sum < large_p ? sum : sum - large_p;
		return *this;
	}

	friend
		__host__ __device__
		bool operator==(const SMersenne61Gpu& v1, const SMersenne61Gpu& v2)
	{
		return v1._v == v2._v;
	}

	friend
		__host__ __device__
		SMersenne61Gpu operator*(const SMersenne61Gpu& v1, const SMersenne61Gpu& v2)
	{
		basic_t high, low;
#ifdef __CUDA_ARCH__
		low = v1._v * v2._v;
		high = __umul64hi(v1._v, v2._v);
#else
		low = _mulx_u64(v1._v, v2._v, &high);
#endif
		
		if (low >= large_p) low -= large_p;
		basic_t sum = low + (high << 3);
		return SMersenne61Gpu(sum);
		//SMersenne61Gpu answer;
		//answer._v = sum < large_p ? sum : sum - large_p;
		//return answer;


		/*basic_t low61 = (low & p);
		basic_t low62to64 = (low >> 61);
		basic_t highShift3 = (high << 3);

		basic_t res = low61 + low62to64 + highShift3;

		if (res >= p)
			res -= p;

		SMersenne61Gpu answer;
		answer._v = res;
		return answer;*/
	}
};


// DEAD CODE BELOW

/////////////////////////////////////////////////////////////////
// mixed 32/64 does not compile - dead code - do not maintain
//////////////////////////////////////////////////////////////////
struct tempStruct
{
	__host__ __device__
		tempStruct(unsigned int v) : _v(v) {}
	unsigned int _v;
};

struct myTypeUint64Mixed
{
	static const unsigned int p = (1U << 31) - 1;
	static const uint64_t m64 = ((uint64_t)(p)) << 32;
	__host__ __device__
		myTypeUint64Mixed() : _v(0) {}
	__host__ __device__
		myTypeUint64Mixed(uint64_t v) : _v(v < m64 ? v : v - m64) {}
	uint64_t _v;

	//__host__ __device__
	//	operator tempStruct() const { return tempStruct((unsigned int)(_v % p)); }


	__host__ __device__
		myTypeUint64Mixed operator+(const myTypeUint64Mixed& other)
	{
		myTypeUint64Mixed res(_v + other._v);
		return res;
	}

	__host__ __device__
		myTypeUint64Mixed& operator+=(const myTypeUint64Mixed& other)
	{
		_v += other._v;
		if (_v >= m64)
			_v -= m64;
		return *this;
	}


	friend
		__host__ __device__
		bool operator==(const myTypeUint64Mixed& v1, const myTypeUint64Mixed& v2)
	{
		return v1._v == v2._v;
	}

	/*friend
	__host__ __device__
	myTypeUint64Mixed operator*(const myTypeUint64Mixed& v1, const myTypeUint64Mixed& v2)
	{
	myTypeUint64Mixed res(v1._v * v2._v);
	return res;
	}*/

	/*friend
	__host__ __device__
	myTypeUint64Mixed operator*(const unsigned int& v1, const myTypeUint64Mixed& v2)
	{
	return v2;
	}*/

};

struct myTypeUint32Mixed
{
	static const unsigned int p = (1U << 31) - 1;
	typedef myTypeUint64Mixed Accum_t;
	__host__ __device__
		myTypeUint32Mixed() : _v(0) {}
	__host__ __device__
		myTypeUint32Mixed(unsigned int v) : _v(v < p ? v : v - p) {}
	//__host__ __device__
	//	myTypeUint32Mixed(tempStruct v) : _v(v._v) {}

	unsigned int _v;

	/*__host__ __device__
	myTypeUint32Mixed operator+(const myTypeUint32Mixed& other)
	{
	myTypeUint32Mixed res(_v + other._v);
	return res;
	}*/
	__host__ __device__
		myTypeUint32Mixed& operator+=(const myTypeUint32Mixed& other)
	{
		_v += other._v;
		if (_v >= p)
			_v = _v - p;
		return *this;
	}

	friend
		__host__ __device__
		bool operator==(const myTypeUint32Mixed& v1, const myTypeUint32Mixed& v2)
	{
		return v1._v == v2._v;
	}

	friend
		__host__ __device__
		myTypeUint64Mixed operator*(const myTypeUint32Mixed& v1, const myTypeUint32Mixed& v2)
	{
		myTypeUint64Mixed res((uint64_t)(v1._v) * (uint64_t)(v2._v));
		return res;
	}

	friend
		__host__ __device__
		myTypeUint64Mixed operator+(const myTypeUint32Mixed& v1, const myTypeUint64Mixed& v2)
	{
		myTypeUint64Mixed res(v1._v + v2._v);
		return res;
	}
};


///
