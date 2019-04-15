/*
All code in this file based on cutlass examples for float/double precision.
https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/
https://github.com/NVIDIA/cutlass

This file contains code and tests for running matrix multiplication
on merssen31/61 fields. specifically it performs C = A.t() * B
This means that the data layout for A and B is the same.

low level device functions: GemmTN31/GemmTN61. -- Do not use directly

high level host functions: recieve/return data on the host. 
divides the full matrices into tiles and uses multiple devices if given
GemmTNTiles31, GemmTNTiles61

tests: 
for low level matrix multiply : testGemmTN31, testGemmTN61
for high level matrix multiply using tiles: testGemmTNTiles31, testGemmTNTiles61
NOTE: there is a cheat mode that works faster for CPU allocation and data generation but without correctness testing

TODOS:
Write your own data generation functions. Current ones are hacky
remove the cheat mode
write some hardcodre tests with data generation on device?!

*/


// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/gemm.h"

// Defines cutlass::gemm::SgemmTraits, the structural components for single-precision GEMM
#include "cutlass/gemm/sgemm_traits.h"
#include "cutlass/gemm/dgemm_traits.h"
#include "cutlass/gemm/gemm_config.h"
#include "utils.h"
#include "DeviceMat.h"
#include "cudaGemm.h"

// SUint32 SMersenne31Naive SMersenne31Classic merssene31_tUint32Mixed SMersenne31HiLo

using namespace std;


template <
	/// The tile size for the GEMM KxNxM.
	typename OutputTile_,
	/// Tile size for thread-level GEMM (K-by-N-by-M)
	typename ThreadGemmShape_,
	/// The number of scalars per LDG for A.
	int kScalarsPerLdgA_ = 1,
	/// The number of scalars per LDG for B.
	int kScalarsPerLdgB_ = 1,
	/// Whether to specify launch bounds
	bool kLaunchBounds = true>
	struct SMersseneGemmConfig31 : public cutlass::gemm::GemmConfig <
	/// The scalar type for A.
	merssene31_t,
	/// The scalar type for B.
	merssene31_t,
	/// The scalar type for C.
	merssene31_t::Accum_t,
	/// The scalar type for D.
	merssene31_t::Accum_t,
	/// The tile size for the GEMM KxNxM.
	OutputTile_,
	/// The functor to do the math in the main loop.
	cutlass::gemm::ThreadMultiplyAdd<ThreadGemmShape_, cutlass::Shape<1, 4, 8>, merssene31_t, merssene31_t, merssene31_t::Accum_t>,
	kScalarsPerLdgA_,
	/// The number of scalars per STS for A.
	kScalarsPerLdgA_,
	/// The number of scalars per LDS for A.
	4,
	/// The number of scalars per LDG for B.
	kScalarsPerLdgB_,
	/// The number of scalars per STS for B.
	kScalarsPerLdgB_,
	/// The number of scalars per LDS for B.
	4,
	/// The number of scalars per LDG for C and STG for D.
	1,
	/// The number of scalars per STS for D.
	4,
	/// The number of scalars per LDS for D.
	1,
	/// The number of stages in shared memory.
	2,
	/// kResidueSeparate
	false,
	/// kResidueInPrologue
	true,
	/// kLaunchBounds
	kLaunchBounds> {};


template <
	/// The tile size for threadblock-level GEMM (K-by-N-by-M).
	typename OutputTile_,
	/// Tile size for thread-level GEMM (K-by-N-by-M)
	typename ThreadGemmShape_,
	/// The number of scalars per LDG for A.
	int kScalarsPerLdgA_ = 1,
	/// The number of scalars per LDG for B.
	int kScalarsPerLdgB_ = 1>
	struct SMersseneGemmConfig61
	: public cutlass::gemm::GemmConfig <
	/// The scalar type for A.
	merssene61_t,
	/// The scalar type for B.
	merssene61_t,
	/// The scalar type for C.
	merssene61_t,
	/// The scalar type for D.
	merssene61_t,
	/// The tile size for the GEMM KxNxM.
	OutputTile_,
	/// The functor to do the math in the main loop.
	cutlass::gemm::ThreadMultiplyAdd<ThreadGemmShape_, cutlass::Shape<1, 4, 8>, merssene61_t, merssene61_t, merssene61_t>,
	/// The number of scalars per LDG for A.
	kScalarsPerLdgA_,
	/// The number of scalars per STS for A.
	kScalarsPerLdgA_,
	/// The number of scalars per LDS for A.
	2,
	/// The number of scalars per LDG for B.
	kScalarsPerLdgB_,
	/// The number of scalars per STS for B.
	kScalarsPerLdgB_,
	/// The number of scalars per LDS for B.
	2,
	/// The number of scalars per LDG for C and STG for D.
	1,
	/// The number of scalars per STS for D.
	2,
	/// The number of scalars per LDS for D.
	1,
	/// The number of stages in shared memory.
	2,
	/// kResidueSeparate
	true,
	/// kResidueInPrologue
	false,
	/// kLaunchBounds
	false
	> {};

template<typename _type>
void matrixMulCPUTN(_type *C, size_t ldc, const _type *A, size_t lda, const _type *B, size_t ldb, int hA, size_t wA, size_t wB)
{
	for (size_t i = 0; i < wA; ++i)
		for (size_t  j = 0; j < wB; ++j)
		{
			//typename _type::Accum_t sum(0);
			_type sum(0);
			for (size_t  k = 0; k < hA; ++k)
			{
				_type a = A[i + k * lda];
				_type b = B[j + k * ldb];
				sum += a * b;
			}

			C[i + j * ldc] += (_type)sum;
		}
}

template<typename _readType, typename _type, typename _otherType>
void matrixMulCPUCompare(_readType *C, int ldc, const _readType *A, 
	int lda, const _readType *B, int ldb, int hA, int wA, int wB)
{
	//uint64_t m64 = _otherType::Accum_t::m64;
	unsigned int m = _otherType::p;
	for (int i = 0; i < hA; ++i)
		for (int j = 0; j < wB; ++j)
		{
			typename _type::Accum_t sum(0);
			typename _otherType::Accum_t otherSum(0);

			for (int k = 0; k < wA; ++k)
			{
				_type a(A[i * lda + k]);
				_type b (B[k * ldb + j]);
				_otherType othera (A[i * lda + k]);
				_otherType otherb (B[k * ldb + j]);

				sum += a * b;
				otherSum += othera * otherb;
				if (sum._v != (otherSum._v % m))
					printf("Error!!!\n");
			}

			C[i * ldc + j] = (_readType)(sum._v);

			_readType res = (_readType)(sum._v % _type::p);
			_readType otherRes ((_readType)(otherSum._v % _type::p));

			if (res != otherRes)
				printf("Error!!!\n");

		}
}

template<typename _type>
cudaError_t GemmTN31(
	size_t M,
	size_t N,
	size_t K,
	merssene31_t::Accum_t alpha,
	_type const *A,
	size_t lda,
	_type const *B,
	size_t ldb,
	merssene31_t::Accum_t beta,
	_type *C,
	size_t ldc,
	cudaStream_t& stream) {

	// Define type definition for single-precision CUTLASS GEMM with column-major
	// input matrices and 128x128x8 threadblock tile size.
	//
	// Note, GemmTraits<> is a generic template defined for various general matrix product
	// computations within CUTLASS. It is intended to be maximally flexible, and consequently
	// it contains numerous template arguments.
	//
	// To keep the interface manageable, several helpers are defined for plausible compositions
	// including the following example for single-precision GEMM. Typical values are used as
	// default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
	//
	//SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>> mygemmconf;
	typedef cutlass::gemm::SgemmTraits <
		cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
		cutlass::MatrixLayout::kRowMajor,   // layout of B matrix
		cutlass::Shape<8, 128, 128>,           // threadblock tile size
		cutlass::gemm::LinearScaling<merssene31_t>,
		/// Tile size for thread-level GEMM (K-by-N-by-M)
		cutlass::Shape<8, 8, 8>,
		/// The number of floats loaded in one LDG for A.
		1,
		/// The number of floats loaded in one LDG for B.
		1,
		/// The index.
		int,
		/// The SGEMM config.
		SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>>
		/// The traits class for the epilogue.
		//,cutlass::gemm::SimplifiedGemmEpilogueTraits<
		//SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>>,
		//cutlass::gemm::LinearScaling<merssene31_t::Accum_t,
		//cutlass::gemm::FragmentMultiplyAdd<merssene31_t::Accum_t, merssene31_t::Accum_t/*accumulator type*/> > >
		//
		//cutlass::gemm::myLinearScaling<merssene31_t::Accum_t>, int>
		/*typename GemmEpilogueTraits_ =
		SimplifiedGemmEpilogueTraits<GemmConfig_, EpilogueFunctor_, Index_> >
		*/
	>
		GemmTraits;

	// Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
	typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

	// Construct and initialize CUTLASS GEMM parameters object.
	//
	// One of CUTLASS's design patterns is to define parameters objects that are constructible
	// in host code and passed to kernels by value. These may include pointers, strides, scalars,
	// and other arguments needed by Gemm and its components.
	//
	// The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
	// arguments to kernels and (2.) minimized initialization overhead on kernel entry.
	//
	typename Gemm::Params params;

	//GemmTraits::Epilogue a;
	int result = params.initialize(
		N,     // GEMM M dimension
		K,     // GEMM N dimension
		M,     // GEMM K dimension
		alpha, // scalar alpha
		A,     // matrix A operand
		lda,
		B,     // matrix B operand
		ldb,
		beta,  // scalar beta
		C,     // source matrix C
		ldc,
		C,     // destination matrix C (may be different memory than source C matrix)
		ldc
	);

	if (result) {
		std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
		return cudaErrorInvalidValue;
	}


	// Launch the CUTLASS GEMM kernel.
	Gemm::launch(params, stream);

	// Return any errors associated with the launch or cudaSuccess if no error.
	return cudaGetLastError();
}

template<typename _type>
cudaError_t GemmTN61(
	int M,
	int N,
	int K,
	_type alpha,
	_type const *A,
	int lda,
	_type const *B,
	int ldb,
	_type beta,
	_type *C,
	int ldc,
	cudaStream_t& stream) {

	// Define type definition for single-precision CUTLASS GEMM with column-major
	// input matrices and 128x128x8 threadblock tile size.
	//
	// Note, GemmTraits<> is a generic template defined for various general matrix product
	// computations within CUTLASS. It is intended to be maximally flexible, and consequently
	// it contains numerous template arguments.
	//
	// To keep the interface manageable, several helpers are defined for plausible compositions
	// including the following example for single-precision GEMM. Typical values are used as
	// default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
	//
	//SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>> mygemmconf;
	typedef cutlass::gemm::DgemmTraits <
		cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
		cutlass::MatrixLayout::kRowMajor,   // layout of B matrix
		cutlass::Shape<8, 64, 128>,           // threadblock tile size
		cutlass::gemm::LinearScaling<_type>,
		/// Tile size for thread-level GEMM (K-by-N-by-M)
		cutlass::Shape<8, 8, 8>,
		/// The number of floats loaded in one LDG for A.
		1,
		/// The number of floats loaded in one LDG for B.
		1,
		/// The index.
		int,
		/// The SGEMM config.
		SMersseneGemmConfig61<cutlass::Shape<8, 64, 128>, cutlass::Shape<4, 8, 8>>
		/// The traits class for the epilogue.
		//,cutlass::gemm::SimplifiedGemmEpilogueTraits<
		//SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>>,
		//cutlass::gemm::LinearScaling<merssene31_t::Accum_t,
		//cutlass::gemm::FragmentMultiplyAdd<merssene31_t::Accum_t, merssene31_t::Accum_t/*accumulator type*/> > >
		//
		//cutlass::gemm::myLinearScaling<merssene31_t::Accum_t>, int>
		/*typename GemmEpilogueTraits_ =
		SimplifiedGemmEpilogueTraits<GemmConfig_, EpilogueFunctor_, Index_> >
		*/
	>
		GemmTraits;

	// Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
	typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

	// Construct and initialize CUTLASS GEMM parameters object.
	//
	// One of CUTLASS's design patterns is to define parameters objects that are constructible
	// in host code and passed to kernels by value. These may include pointers, strides, scalars,
	// and other arguments needed by Gemm and its components.
	//
	// The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
	// arguments to kernels and (2.) minimized initialization overhead on kernel entry.
	//
	typename Gemm::Params params;

	//GemmTraits::Epilogue a;
	int result = params.initialize(
		N,     // GEMM M dimension
		K,     // GEMM N dimension
		M,     // GEMM K dimension
		alpha, // scalar alpha
		A,     // matrix A operand
		lda,
		B,     // matrix B operand
		ldb,
		beta,  // scalar beta
		C,     // source matrix C
		ldc,
		C,     // destination matrix C (may be different memory than source C matrix)
		ldc
	);

	if (result) {
		std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
		return cudaErrorInvalidValue;
	}

	
	// Launch the CUTLASS GEMM kernel.
	Gemm::launch(params, stream);

	// Return any errors associated with the launch or cudaSuccess if no error.
	return cudaGetLastError();
}


template <typename T>
void randvec(T* ptr, size_t width, size_t height, size_t ldm, T p)
{
	static const int num_integers = 64 * 1024;
	std::vector<T> numbers(num_integers);
	for (auto& elm : numbers)
	{
		T val = 0;
		for (int i = 0; i < sizeof(T); i += 2)
			val = (val << 16) + rand();
		elm = val % p;
	}
	//printf("w, h, ldm = (%d, %d, %d)\n)", width, height, ldm);
	//return;
	//printf("data generation size (%d, %d). %d percents done", 
	//	width, height, 0);
	int rand_sum = rand();
	for (size_t h = 0; h < height; ++h)
	{
		//printf("\rdata generation size (%d, %d). %d percents done",
		//	width, height, int(h)/height);
		for (size_t w = 0; w < width; ++w)
		{
			ptr[h * ldm + w] = numbers[((w * h) ^ rand_sum) & (num_integers-1)];
			
			/*T val = 0;
			for (int i = 0; i < sizeof(T); i += 2)
				val = (val << 16) + rand();
			ptr[h * ldm + w] = T(val % p);*/
		}
	}
}

/* a test for low level matrix multiply on merssene31. 
m - number of messages. (1e6)
width_a - number of elements in matrix A
width_b - number of elements in matrix B
*/
void testGemmTN31(size_t m, size_t width_a, size_t width_b)
{
//	printf("Starting merssene 31 gemm test\n");
	cudaStream_t stream = NULL;
	size_t h_lda = width_a;
	size_t h_ldb = width_b;
	size_t h_ldc = width_a;

	std::vector<merssene31_t> h_A(h_lda * m, 1);
	std::vector<merssene31_t> h_B(h_ldb * m, 1);
	std::vector<merssene31_t> h_C(h_ldc * width_b, 0);
	std::vector<merssene31_t> h_C_ref(h_ldc * width_b, 0);

	randvec((merssene31_t::basic_t*)h_A.data(), width_a, m, width_a, merssene31_t::p);
	randvec((merssene31_t::basic_t*)h_B.data(), width_b, m, width_b, merssene31_t::p);
	randvec((merssene31_t::basic_t*)h_C.data(), width_a, width_b, width_a, merssene31_t::p);
	h_C_ref = h_C;
		
	Mat<merssene31_t> A(width_a, m); // A is width_a rows by height_a columns
	Mat<merssene31_t> B(width_b, m); // B is width_a rows by height_b columns
	Mat<merssene31_t> C(width_a, width_b); // C is height_a rows by height_b columns

	cudaSafeCall(cudaMemcpy2DAsync(A._ptr, A._ldm  * sizeof(merssene31_t), 
		h_A.data(), h_lda * sizeof(merssene31_t), width_a * sizeof(merssene31_t), 
		m, cudaMemcpyHostToDevice, stream));
	cudaSafeCall(cudaMemcpy2DAsync(B._ptr, B._ldm * sizeof(merssene31_t),
		h_B.data(), h_ldb * sizeof(merssene31_t), width_b * sizeof(merssene31_t),
		m, cudaMemcpyHostToDevice, stream));
	cudaSafeCall(cudaMemcpy2DAsync(C._ptr, C._ldm * sizeof(merssene31_t),
		h_C.data(), h_ldc * sizeof(merssene31_t), C._rows * sizeof(merssene31_t),
		C._columns, cudaMemcpyHostToDevice, stream));
	//cudaSafeCall(cudaMemcpyAsync(A._ptr, h_A.data(), h_A.size() * sizeof(merssene31_t), cudaMemcpyHostToDevice, stream));
	//cudaSafeCall(cudaMemcpyAsync(B._ptr, h_B.data(), h_B.size() * sizeof(merssene31_t), cudaMemcpyHostToDevice, stream));
	cudaEvent_t start, end;
	cudaSafeCall(cudaEventCreate(&start));
	cudaSafeCall(cudaEventCreate(&end));
	cudaSafeCall(cudaEventRecord(start));

	cudaSafeCall(GemmTN31(m, width_a, width_b, merssene31_t::Accum_t(1),
		A._ptr, A._ldm, B._ptr, B._ldm, merssene31_t::Accum_t(1), C._ptr, C._ldm, stream));

	cudaSafeCall(cudaEventRecord(end));
	cudaSafeCall(cudaEventSynchronize(end));

	float average_ms = 0;
	cudaSafeCall(cudaEventElapsedTime(&average_ms, start, end));
//	std::cout << "Mat size: M=" << m << ", N=" << width_a <<
//		", K=" << width_b << ", Time: " << average_ms << "ms\n";
	

	cudaSafeCall(cudaMemcpy2DAsync(h_C.data(), h_ldc * sizeof(merssene31_t),
		C._ptr, C._ldm * sizeof(merssene31_t), C._rows * sizeof(merssene31_t),
		C._columns, cudaMemcpyDeviceToHost, stream));

	matrixMulCPUTN(h_C_ref.data(), h_ldc, h_A.data(), h_lda,
		h_B.data(), h_ldb, m, width_a, width_b);
	
	if (h_C != h_C_ref)
	{
		UGLY_ERROR;
		/*for (size_t  i = 0; i < h_C.size(); ++i)
			if (h_C[i]._v != h_C_ref[i]._v)
				bool yyy = true;*/
	}
	else 
		printf("Test Passed\n");
}

void testGemmTN61(size_t m, size_t width_a, size_t width_b)
{
//	printf("Starting merssene 61 gemm test\n");
	cudaStream_t stream = NULL;
	size_t h_lda = width_a;
	size_t h_ldb = width_b;
	size_t h_ldc = width_a;

	std::vector<merssene61_t> h_A(h_lda * m, 1);
	std::vector<merssene61_t> h_B(h_ldb * m, 1);
	std::vector<merssene61_t> h_C(h_ldc * width_b, 0);
	std::vector<merssene61_t> h_C_ref(h_ldc * width_b, 0);

	randvec((merssene61_t::basic_t*)h_A.data(), width_a, m, width_a, merssene61_t::p);
	randvec((merssene61_t::basic_t*)h_B.data(), width_b, m, width_b, merssene61_t::p);
	randvec((merssene61_t::basic_t*)h_C.data(), width_a, width_b, width_a, merssene61_t::p);
	h_C_ref = h_C;

	Mat<merssene61_t> A(width_a, m); 
	Mat<merssene61_t> B(width_b, m); 
	Mat<merssene61_t> C(width_a, width_b); 

	cudaSafeCall(cudaMemcpy2DAsync(A._ptr, A._ldm * sizeof(merssene61_t),
		h_A.data(), h_lda * sizeof(merssene61_t), width_a * sizeof(merssene61_t),
		m, cudaMemcpyHostToDevice, stream));
	cudaSafeCall(cudaMemcpy2DAsync(B._ptr, B._ldm * sizeof(merssene61_t),
		h_B.data(), h_ldb * sizeof(merssene61_t), width_b * sizeof(merssene61_t),
		m, cudaMemcpyHostToDevice, stream));
	cudaSafeCall(cudaMemcpy2DAsync(C._ptr, C._ldm * sizeof(merssene61_t),
		h_C.data(), h_ldc * sizeof(merssene61_t), C._rows * sizeof(merssene61_t),
		C._columns, cudaMemcpyHostToDevice, stream));
	
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	cudaSafeCall(GemmTN61<merssene61_t>((int)m, (int)width_a, (int)width_b, merssene61_t(1),
		A._ptr, (int)A._ldm, B._ptr, (int)B._ldm, 
		merssene61_t(1), C._ptr, (int)C._ldm, stream));

	cudaSafeCall(cudaEventRecord(end));
	cudaSafeCall(cudaEventSynchronize(end));

	float average_ms = 0;
	cudaSafeCall(cudaEventElapsedTime(&average_ms, start, end));
//	std::cout << "Mat size: M=" << m << ", N=" << width_a << ", K=" <<
//		width_b << ", Time: " << average_ms << "ms\n";


	// (512 * 128) * (256 * 512) = (128 * 256)
	cudaSafeCall(cudaMemcpy2DAsync(h_C.data(), h_ldc * sizeof(merssene61_t),
		C._ptr, C._ldm * sizeof(merssene61_t), C._rows * sizeof(merssene61_t),
		C._columns, cudaMemcpyDeviceToHost, stream));

	matrixMulCPUTN((SMersenne61Naive*)h_C_ref.data(), h_ldc, (SMersenne61Naive*)h_A.data(), h_lda,
		(SMersenne61Naive*)h_B.data(), h_ldb, m, width_a, width_b);


	for (auto& elm : h_C)
		elm._v = elm._v % SMersenne61Naive::p;

	if (h_C != h_C_ref)
	{
		UGLY_ERROR;
		/*for (size_t i = 0; i < h_C.size(); ++i)
			if (h_C[i]._v != h_C_ref[i]._v)
				bool yyy = true;*/
	}
	else
		printf("Test Passed\n");
}

struct STile
{
	STile(size_t start, size_t size) : _size(size), _start(start) {}
	size_t _start;
	size_t _size;
};

void getTiles(size_t  m, size_t width_a, size_t width_b, size_t tile_size,
	std::vector<STile>& dst_tiles)
{
	for (size_t i = 0; i < m; i += tile_size)
	{
		dst_tiles.emplace_back(i, std::min(tile_size, m - i));
	}
}


void processTiles31(int device_id, merssene31_t* ptr_a, size_t h_lda,
	merssene31_t* ptr_b, size_t h_ldb, 
	merssene31_t* ptr_c, size_t h_ldc,
	size_t width_a, size_t width_b, size_t tile_size,
	STile* tiles, size_t num_tiles,
	std::mutex* pmutex)
{
	printf("device id %d launched\n", device_id);
	/*for (int j = 0; j < num_tiles; ++j)
	printf("tile %d. start %d, size %d\n",
	j, tiles[j]._start, tiles[j]._size);
	*/

	cudaSafeCall(cudaSetDevice(device_id));
	cudaStream_t stream;// = NULL;
	cudaSafeCall(cudaStreamCreate(&stream));

	Mat<merssene31_t> A(width_a, tile_size); // A is width_a rows by height_a columns
	Mat<merssene31_t> B(width_b, tile_size); // B is width_a rows by height_b columns
	Mat<merssene31_t> C(width_a, width_b); // C is height_a rows by height_b columns


	C.SetZero(stream);
//	printf("%d %d %d %d", C._rows, C._columns, C._ldm, h_ldc);
	for (int i = 0; i < num_tiles; ++i)
	{
		auto& tile = tiles[i];
//		std::cout << "device " << device_id << " starting work on tile " <<
//			i << ", line " << tile._start << ", size " << tile._size << "\n";

		cudaSafeCall(cudaMemcpy2DAsync(A._ptr, A._ldm * sizeof(merssene31_t),
			ptr_a + h_lda * tile._start, h_lda * sizeof(merssene31_t),
			width_a * sizeof(merssene31_t),
			tile._size, cudaMemcpyHostToDevice, stream));
		cudaSafeCall(cudaMemcpy2DAsync(B._ptr, B._ldm * sizeof(merssene31_t),
			ptr_b + h_ldb * tile._start, h_ldb * sizeof(merssene31_t), width_b * sizeof(merssene31_t),
			tile._size, cudaMemcpyHostToDevice, stream));
		//cudaSafeCall(cudaMemcpyAsync(A._ptr, h_A.data(), h_A.size() * sizeof(merssene31_t), cudaMemcpyHostToDevice, stream));
		//cudaSafeCall(cudaMemcpyAsync(B._ptr, h_B.data(), h_B.size() * sizeof(merssene31_t), cudaMemcpyHostToDevice, stream));

		cudaSafeCall(GemmTN31(tile._size, width_a, width_b, merssene31_t::Accum_t(1),
			A._ptr, A._ldm, B._ptr, B._ldm, merssene31_t::Accum_t(1), C._ptr, C._ldm, stream));
		cudaSafeCall(cudaStreamSynchronize(stream));
	}
	// (512 * 128) * (256 * 512) = (128 * 256)
	size_t local_ldc = width_a;
	std::vector<merssene31_t> h_C(local_ldc * width_b);
	cudaSafeCall(cudaMemcpy2DAsync(h_C.data(), h_ldc * sizeof(merssene31_t),
		C._ptr, C._ldm * sizeof(merssene31_t), C._rows * sizeof(merssene31_t),
		C._columns, cudaMemcpyDeviceToHost, stream));
	printf("******");
	cudaSafeCall(cudaStreamSynchronize(stream));
	printf("******");
	if (stream != nullptr)
		cudaSafeCall(cudaStreamDestroy(stream));

	std::lock_guard<std::mutex> guard(*pmutex);
	for (size_t j = 0; j < width_b; ++j)
	{
		merssene31_t* ptr_line_global = ptr_c + j * h_ldc;
		merssene31_t* ptr_line_local = h_C .data() + j * local_ldc;
		for (size_t i = 0; i < width_a; ++i)
			ptr_line_global[i] += ptr_line_local[i];
	}

}

void processTiles61(int device_id, merssene61_t* ptr_a, size_t h_lda, 
	merssene61_t* ptr_b, size_t h_ldb, 
	merssene61_t* ptr_c, size_t h_ldc,
	size_t width_a, size_t width_b, size_t tile_size,
	STile* tiles, size_t num_tiles,
	std::mutex* pmutex)
{
	printf("device id %d launched\n", device_id);
	/*for (int j = 0; j < num_tiles; ++j)
		printf("tile %d. start %d, size %d\n",
			j, tiles[j]._start, tiles[j]._size);
*/

	cudaSafeCall(cudaSetDevice(device_id));
	cudaStream_t stream;// = NULL;
	cudaSafeCall(cudaStreamCreate(&stream));

	Mat<merssene61_t> A(width_a, tile_size); // A is width_a rows by height_a columns
	Mat<merssene61_t> B(width_b, tile_size); // B is width_a rows by height_b columns
	Mat<merssene61_t> C(width_a, width_b); // C is height_a rows by height_b columns
	
	size_t local_ldc = width_a;
	std::vector<merssene61_t> h_C(local_ldc * width_b);

	C.SetZero(stream);
	for (int i = 0; i < num_tiles; ++i)
	{
		auto& tile = tiles[i];
//		std::cout << "device " << device_id << " starting work on tile " <<
//			i << ", line " << tile._start << ", size " << tile._size << "\n";

		cudaSafeCall(cudaMemcpy2DAsync(A._ptr, A._ldm * sizeof(merssene61_t),
			ptr_a + h_lda * tile._start, h_lda * sizeof(merssene61_t),
			width_a * sizeof(merssene61_t),
			tile._size, cudaMemcpyHostToDevice, stream));
		cudaSafeCall(cudaMemcpy2DAsync(B._ptr, B._ldm * sizeof(merssene61_t),
			ptr_b + h_ldb * tile._start, h_ldb * sizeof(merssene61_t), width_b * sizeof(merssene61_t),
			tile._size, cudaMemcpyHostToDevice, stream));
		
		cudaSafeCall(GemmTN61(tile._size, width_a, width_b, merssene61_t::Accum_t(1),
			A._ptr, A._ldm, B._ptr, B._ldm, merssene61_t::Accum_t(1), C._ptr, C._ldm, stream));
		cudaSafeCall(cudaStreamSynchronize(stream));
	}
	// (512 * 128) * (256 * 512) = (128 * 256)
	cudaSafeCall(cudaMemcpy2DAsync(h_C.data(), local_ldc * sizeof(merssene61_t),
		C._ptr, C._ldm * sizeof(merssene61_t), C._rows * sizeof(merssene61_t),
		C._columns, cudaMemcpyDeviceToHost, stream));

	cudaSafeCall(cudaStreamSynchronize(stream));
	if (stream != nullptr)
		cudaSafeCall(cudaStreamDestroy(stream));

	std::lock_guard<std::mutex> guard(*pmutex);

	for (size_t j = 0; j < width_b; ++j)
	{
		merssene61_t* ptr_line_global = ptr_c + j * h_ldc;
		merssene61_t* ptr_line_local = h_C.data() + j * local_ldc;
		for (size_t i = 0; i < width_a; ++i)
			ptr_line_global[i] += ptr_line_local[i];
	}

}

/////////////////////////////////////

template<typename _type>
cudaError_t GemmNN31(
        size_t M,
        size_t N,
        size_t K,
        merssene31_t::Accum_t alpha,
        _type const *A,
        size_t lda,
        _type const *B,
        size_t ldb,
        merssene31_t::Accum_t beta,
        _type *C,
        size_t ldc,
        cudaStream_t& stream) {

// Define type definition for single-precision CUTLASS GEMM with column-major
// input matrices and 128x128x8 threadblock tile size.
//
// Note, GemmTraits<> is a generic template defined for various general matrix product
// computations within CUTLASS. It is intended to be maximally flexible, and consequently
// it contains numerous template arguments.
//
// To keep the interface manageable, several helpers are defined for plausible compositions
// including the following example for single-precision GEMM. Typical values are used as
// default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
//
//SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>> mygemmconf;
    typedef cutlass::gemm::SgemmTraits <
            cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
            cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
            cutlass::Shape<8, 128, 128>,           // threadblock tile size
            cutlass::gemm::LinearScaling<merssene31_t>,
            /// Tile size for thread-level GEMM (K-by-N-by-M)
            cutlass::Shape<8, 8, 8>,
            /// The number of floats loaded in one LDG for A.
            1,
            /// The number of floats loaded in one LDG for B.
            1,
            /// The index.
            int,
            /// The SGEMM config.
            SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>>
            /// The traits class for the epilogue.
            //,cutlass::gemm::SimplifiedGemmEpilogueTraits<
            //SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>>,
            //cutlass::gemm::LinearScaling<merssene31_t::Accum_t,
            //cutlass::gemm::FragmentMultiplyAdd<merssene31_t::Accum_t, merssene31_t::Accum_t/*accumulator type*/> > >
            //
            //cutlass::gemm::myLinearScaling<merssene31_t::Accum_t>, int>
            /*typename GemmEpilogueTraits_ =
            SimplifiedGemmEpilogueTraits<GemmConfig_, EpilogueFunctor_, Index_> >
            */
    >
            GemmTraits;

// Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
    typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

// Construct and initialize CUTLASS GEMM parameters object.
//
// One of CUTLASS's design patterns is to define parameters objects that are constructible
// in host code and passed to kernels by value. These may include pointers, strides, scalars,
// and other arguments needed by Gemm and its components.
//
// The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
// arguments to kernels and (2.) minimized initialization overhead on kernel entry.
//
    typename Gemm::Params params;

//GemmTraits::Epilogue a;
    int result = params.initialize(
            M,     // GEMM M dimension
            N,     // GEMM N dimension
            K,     // GEMM K dimension
            alpha, // scalar alpha
            A,     // matrix A operand
            lda,
            B,     // matrix B operand
            ldb,
            beta,  // scalar beta
            C,     // source matrix C
            ldc,
            C,     // destination matrix C (may be different memory than source C matrix)
            ldc
    );

    if (result) {
        std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
        return cudaErrorInvalidValue;
    }


// Launch the CUTLASS GEMM kernel.
    Gemm::launch(params, stream);

// Return any errors associated with the launch or cudaSuccess if no error.
    return cudaGetLastError();
}


/////////////////////////////////////




void processNN31(merssene31_t* h_C,
                 merssene31_t* h_A, size_t rowA, size_t colA,
                 merssene31_t* h_B, size_t rowB, size_t colB,
                 int deviceID)
{


    	printf("Starting merssene 31 gemm test. deviceId = %d\n", deviceID);
    cudaSafeCall(cudaSetDevice(deviceID));
    cudaStream_t stream;// = NULL;
    cudaSafeCall(cudaStreamCreate(&stream));
    
    size_t h_lda = colA;
    size_t h_ldb = colB;
    size_t h_ldc = colA;


    Mat<merssene31_t> A(colA, rowA); // A is width_a rows by height_a columns
    Mat<merssene31_t> B(colB, rowB); // B is width_a rows by height_b columns
    Mat<merssene31_t> C(colA, rowB); // C is height_a rows by height_b columns


    cudaSafeCall(cudaMemcpy2DAsync(A._ptr, A._ldm  * sizeof(merssene31_t),
                                   h_A, h_lda * sizeof(merssene31_t), colA * sizeof(merssene31_t),
                                   rowA, cudaMemcpyHostToDevice, stream));
    cudaSafeCall(cudaMemcpy2DAsync(B._ptr, B._ldm * sizeof(merssene31_t),
                                   h_B, h_ldb * sizeof(merssene31_t), colB* sizeof(merssene31_t),
                                   rowB, cudaMemcpyHostToDevice, stream));
    cudaSafeCall(cudaMemcpy2DAsync(C._ptr, C._ldm * sizeof(merssene31_t),
                                   h_C, h_ldc * sizeof(merssene31_t), C._rows * sizeof(merssene31_t),
                                   C._columns, cudaMemcpyHostToDevice, stream));
    //cudaSafeCall(cudaMemcpyAsync(A._ptr, h_A.data(), h_A.size() * sizeof(merssene31_t), cudaMemcpyHostToDevice, stream));
    //cudaSafeCall(cudaMemcpyAsync(B._ptr, h_B.data(), h_B.size() * sizeof(merssene31_t), cudaMemcpyHostToDevice, stream));
    cudaEvent_t start, end;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&end));
    cudaSafeCall(cudaEventRecord(start));

    cudaSafeCall(GemmNN31(colA, rowB, rowA, merssene31_t::Accum_t(1),
                          A._ptr, A._ldm, B._ptr, B._ldm, merssene31_t::Accum_t(0), C._ptr, C._ldm, stream));

   cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaEventRecord(end));
    cudaSafeCall(cudaEventSynchronize(end));

    float average_ms = 0;
    cudaSafeCall(cudaEventElapsedTime(&average_ms, start, end));
	std::cout <<"Time: " << average_ms << "ms\n";


    cudaSafeCall(cudaMemcpy2DAsync(h_C, h_ldc * sizeof(merssene31_t),
                                   C._ptr, C._ldm * sizeof(merssene31_t), C._rows * sizeof(merssene31_t),
                                   C._columns, cudaMemcpyDeviceToHost, stream));
   cudaSafeCall(cudaStreamSynchronize(stream));
    cudaSafeCall(cudaStreamDestroy(stream));
}



/* performs matrix multiplication using tiles. recieves/returns data on host.
h_A - pointer to matrix A data
lda - size of leading dimension for A matrix in units of merssene31_t(NOT bytes). Typically = width_a
h_B - pointer to matrix B data
ldb - size of leading dimension for B matrix in units of merssene31_t(NOT bytes). Typically = width_b
h_C - pointer to matrix C data
ldc - size of leading dimension for C matrix in units of merssene31_t(NOT bytes). Typically = width_a
m - number of messages. (1e6)
width_a - number of elements in matrix A
width_b - number of elements in matrix B
tile_size - size of tile to use along the m dimension
devices - device Ids to use. can and should contain more then one instance for each phisycal device
cheat - recycle memory allocation. faster to run but no corretness testing
*/
void GemmTNTiles31(merssene31_t* h_A, size_t h_lda, 
	merssene31_t* h_B, size_t h_ldb,
	merssene31_t* h_C, size_t h_ldc,
	size_t m, size_t width_a, size_t width_b, size_t tile_size,
	const std::vector<int>& devices, bool cheat)
{
	cudaStream_t stream;// = NULL;
	cudaSafeCall(cudaStreamCreate(&stream));
	size_t num_threads = devices.size();
	std::vector<STile> tiles_all;
	getTiles(m, width_a, width_b, tile_size, tiles_all);
	std::vector< std::vector<STile> > tiles_groups(num_threads);
	size_t group_size = (tiles_all.size() + num_threads - 1) / num_threads;

//	std::cout << "Launching job size: (" << m << "," << width_a <<
//		"," << width_b << "). " << tiles_all.size() <<
//		" tiles of size " << tile_size << " on " << devices.size() <<
//		" devices\n";
	if (cheat)
		for (auto& elm : tiles_all)
			elm._start = 0;
	for (size_t tid = 0; tid < num_threads; ++tid)
	{
		size_t last_index = std::min(tid * group_size + group_size,
			tiles_all.size());
		tiles_groups[tid].insert(tiles_groups[tid].end(),
			tiles_all.begin() + tid * group_size,
			tiles_all.begin() + last_index);
	}
	/*for (int i = 0; i < tiles_groups.size(); ++i)
	for (int j = 0; j < tiles_groups[i].size(); ++j)
	printf("group %d, tile %d. start %d, size %d\n",
	i, j, tiles_groups[i][j]._start, tiles_groups[i][j]._size);*/


	//std::vector< std::vector<merssene31_t> >
	//	h_Cs(num_threads, std::vector<merssene31_t>(h_ldc * width_b, 0));// , h_ldc * width_b, 0);
	std::vector<merssene31_t>  h_C_all(h_ldc * width_b);
	std::vector<std::thread> threads(num_threads);
//	printf("Allocation finished\n");
	
	Mat<merssene31_t> d_C_sum(width_a, width_b);
	std::mutex mutex;


//	printf("data generation done.\n");
	auto start = std::chrono::system_clock::now();

	for (size_t tid = 0; tid < num_threads; ++tid)
	{
		if (tiles_groups[tid].size() == 0)
			continue;
		threads[tid] = std::thread(processTiles31, devices[tid], 
			(merssene31_t*)h_A, h_lda, (merssene31_t*)h_B,
			h_ldb,
			(merssene31_t*)h_C, h_ldc,
			width_a, width_b, tile_size,
			tiles_groups[tid].data(), tiles_groups[tid].size(),
			&mutex);
	}

	for (size_t tid = 0; tid < num_threads; ++tid)
		threads[tid].join();

	/*for (size_t tid = 0; tid < num_threads; ++tid)
	{
	for(size_t i = 0; i < h_C_all.size(); ++i)
	h_C_all[i] += h_Cs[tid][i];
	}
	*/
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	double elapsed_seconds = (double)(elapsed.count()) / 1e6;
	std::cout << "total time: " << elapsed_seconds << '\n';

	//cudaSafeCall(cudaMemcpyAsync(h_C.data(), C._ptr, h_C.size() * sizeof(merssene31_t), cudaMemcpyDeviceToHost, stream));

	if (stream != nullptr)
		cudaSafeCall(cudaStreamDestroy(stream));	
}

/* high level test for marix multiply.
m - number of messages. (1e6)
width_a - number of elements in matrix A
width_b - number of elements in matrix B
tile_size - size of tile to use along the m dimension
devices - device Ids to use. can and should contain more then one instance for each phisycal device
cheat - recycle memory allocation. faster to run but no corretness testing
*/
void testGemmTNTiles31(size_t m, size_t width_a, size_t width_b, size_t tile_size,
	const std::vector<int>& devices, bool cheat)
{
	size_t alloc_dim = cheat ? tile_size : m;
	cudaStream_t stream;// = NULL;
	cudaSafeCall(cudaStreamCreate(&stream));
	
	size_t h_lda = width_a;
	size_t h_ldb = width_b;
	size_t h_ldc = width_a;

	std::vector<merssene31_t> h_A(h_lda * alloc_dim);
	std::vector<merssene31_t> h_B(h_ldb * alloc_dim);
	std::vector<merssene31_t>  h_C_all(h_ldc * width_b);
	std::vector<merssene31_t> h_C_ref(h_ldc * width_b, 0);
//	printf("Allocation finished\n");
	randvec((merssene31_t::basic_t*)h_A.data(), width_a, alloc_dim, width_a, merssene31_t::p);
	randvec((merssene31_t::basic_t*)h_B.data(), width_b, alloc_dim, width_b, merssene31_t::p);


	GemmTNTiles31(h_A.data(), h_lda,
		h_B.data(), h_ldb,
		h_C_all.data(), h_ldc,
		m, width_a, width_b, tile_size,
		devices, cheat);

	
	if (stream != nullptr)
		cudaSafeCall(cudaStreamDestroy(stream));
	//return;
	matrixMulCPUTN(h_C_ref.data(), h_ldc, h_A.data(), h_lda,
		h_B.data(), h_ldb, alloc_dim, width_a, width_b);



	if (h_C_all != h_C_ref)
	{
		UGLY_ERROR;
		/*for (int i = 0; i < h_C_all.size(); ++i)
			if (h_C_all[i]._v != h_C_ref[i]._v)
				bool yyy = true;*/
	}
	else
		printf("Test Passed\n");

}

// see GemmTNTiles31
void GemmTNTiles61(merssene61_t* h_A, size_t h_lda,
	merssene61_t* h_B, size_t h_ldb,
	merssene61_t* h_C, size_t h_ldc,
	size_t m, size_t width_a, size_t width_b, size_t tile_size,
	const std::vector<int>& devices, bool cheat)
{
	cudaStream_t stream;// = NULL;
	cudaSafeCall(cudaStreamCreate(&stream));
	size_t num_threads = devices.size();
	std::vector<STile> tiles_all;
	getTiles(m, width_a, width_b, tile_size, tiles_all);
	std::vector< std::vector<STile> > tiles_groups(num_threads);
	size_t group_size = (tiles_all.size() + num_threads - 1) / num_threads;

//	std::cout << "Launching job size: (" << m << ","  << width_a <<
//		","  << width_b << "). " << tiles_all.size() <<
//		" tiles of size " << tile_size << " on " << devices.size() <<
//		" devices\n";
	if (cheat)
		for (auto& elm : tiles_all)
			elm._start = 0;
	for (size_t tid = 0; tid < num_threads; ++tid)
	{
		size_t last_index = std::min(tid * group_size + group_size,
			tiles_all.size());
		tiles_groups[tid].insert(tiles_groups[tid].end(),
			tiles_all.begin() + tid * group_size,
			tiles_all.begin() + last_index);
	}

	/*for (int i = 0; i < tiles_groups.size(); ++i)
	for (int j = 0; j < tiles_groups[i].size(); ++j)
	printf("group %d, tile %d. start %d, size %d\n",
	i, j, tiles_groups[i][j]._start, tiles_groups[i][j]._size);*/

	std::vector<std::thread> threads(num_threads);
	Mat<merssene61_t> d_C_sum(width_a, width_b);
	std::mutex mutex;

	auto start = std::chrono::system_clock::now();

	for (size_t tid = 0; tid < num_threads; ++tid)
	{
		threads[tid] = std::thread(processTiles61, devices[tid], h_A, h_lda, h_B,
			h_ldb,
			h_C, h_ldc,
			width_a, width_b, tile_size,
			tiles_groups[tid].data(), tiles_groups[tid].size(),
			&mutex);
	}

	for (size_t tid = 0; tid < num_threads; ++tid)
		threads[tid].join();

	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	double elapsed_seconds = (double)(elapsed.count()) / 1e6;
	std::cout << "total time: " << elapsed_seconds << '\n';

	if (stream != nullptr)
		cudaSafeCall(cudaStreamDestroy(stream));
}

// see testGemmTNTiles31
void testGemmTNTiles61(size_t m, size_t width_a, size_t width_b, size_t tile_size,
	const std::vector<int>& devices, bool cheat)
{
	size_t alloc_dim = cheat ? tile_size : m;
//	printf("starting Merssene61 tile test\n");
	
	size_t h_lda = width_a;
	size_t h_ldb = width_b;
	size_t h_ldc = width_a;

	std::vector<merssene61_t> h_A(h_lda * alloc_dim, 0);
	std::vector<merssene61_t> h_B(h_ldb * alloc_dim, 0);
	std::vector<merssene61_t>  h_C_all(h_ldc * width_b, 0);
	std::vector<merssene61_t> h_C_ref(h_ldc * width_b, 0);
	
	randvec((merssene61_t::basic_t*)h_A.data(), width_a, alloc_dim, width_a, merssene61_t::p);
	randvec((merssene61_t::basic_t*)h_B.data(), width_b, alloc_dim, width_b, merssene61_t::p);


	GemmTNTiles61(h_A.data(), h_lda,
		h_B.data(), h_ldb, h_C_all.data(), h_ldc,
		m, width_a, width_b, tile_size, devices, cheat);


	matrixMulCPUTN(h_C_ref.data(), h_ldc, h_A.data(), h_lda,
		h_B.data(), h_ldb, alloc_dim, width_a, width_b);

	if (h_C_all != h_C_ref)
	{
		UGLY_ERROR;
		/*for (int i = 0; i < h_C_all.size(); ++i)
			if (h_C_all[i]._v != h_C_ref[i]._v)
				bool yyy = true;*/
	}
	else
		printf("Test Passed\n");

}

// used to test tiling logic on the CPU
void testGemmTNTilesCpu(int m, int width_a, int width_b, int tile_size)
{
	std::vector<STile> tiles;
	getTiles(m, width_a, width_b, tile_size, tiles);

	int h_lda = width_a;
	int h_ldb = width_b;
	int h_ldc = width_a;

	std::vector<merssene31_t> h_A(h_lda * m, 1);
	std::vector<merssene31_t> h_B(h_ldb * m, 1);
	std::vector<merssene31_t> h_C(h_ldc * width_b, 0);
	std::vector<merssene31_t> h_C_ref(h_ldc * width_b, 0);

	for(auto& tile : tiles)
	{
		matrixMulCPUTN(h_C.data(), h_ldc, h_A.data() + h_lda * tile._start, h_lda,
			h_B.data() + h_ldb * tile._start, h_ldb, tile._size, width_a, width_b);

	}
	matrixMulCPUTN(h_C_ref.data(), h_ldc, h_A.data(), h_lda,
		h_B.data(), h_ldb, m, width_a, width_b);

	if (h_C != h_C_ref)
	{
		UGLY_ERROR
		/*for (int i = 0; i < h_C.size(); ++i)
			if (h_C[i]._v != h_C_ref[i]._v)
				bool yyy = true;*/
	}
	else
		printf("Test Passed\n");
}

int testSplit(size_t m, size_t width_a, size_t width_b);
int main2(int argc, const char *arg[]) 
{
	// Parse the command line to obtain GEMM dimensions and scalar values.
	// GEMM problem dimensions.
	//int b_height = 512; // |b|
	//int m = 1024; // 
	//int a_height = 2048; // |a|
	//int problem[3] = { m, b_height, a_height};
	int problem[3] = { 512, 128, 256};
	
	for (int i = 1; i < argc && i < 4; ++i) {
		std::stringstream ss(arg[i]);
		ss >> problem[i - 1];
	}

	// Scalars used for linear scaling the result of the matrix product.
	float scalars[2] = { 1, 0 };

	for (int i = 4; i < argc && i < 6; ++i) {
		std::stringstream ss(arg[i]);
		ss >> scalars[i - 4];
	}
	//testSplit(problem[0], problem[1], problem[2]); return 0;
	testGemmTN31(problem[0], problem[1], problem[2]);
	testGemmTN61(problem[0], problem[1], problem[2]);	
	//cudaDeviceReset();
	//return 0;
	int threads_per_device = 2;
	int num_devices = 1;
	cudaSafeCall(cudaGetDeviceCount(&num_devices));
	printf("%d devices used\n", num_devices);
	std::vector<int> devices;
	for (int device = 0; device < num_devices; ++device)
	{
		for (int i = 0; i < threads_per_device; ++i)
			devices.push_back(device);
	}
	size_t tile_size = std::min(16384ULL, (unsigned long long)problem[0] / devices.size());
	testGemmTNTiles31(problem[0], problem[1], problem[2], tile_size, devices, false); 
	testGemmTNTiles61(problem[0], problem[1], problem[2], tile_size, devices, false);
	for (int i = 0; i < num_devices; ++i)
	{
		cudaSafeCall(cudaSetDevice(i));
		cudaSafeCall(cudaDeviceReset());
	}
	return 0;
	//testGemmTNTilesCpu(problem[0], problem[1], problem[2], problem[0] / 5);; return 0;
	//
	//cudaError_t result = TestCutlassGemm(
	//	problem[0],     // GEMM M dimension
	//	problem[1],     // GEMM N dimension
	//	problem[2],     // GEMM K dimension
	//	scalars[0],     // alpha
	//	scalars[1]      // beta
	//);

	// Exit.
}



//
//
////////////////////////////////////////////////////////
//// dead code - do not maintain
///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
////
//// The source code after this point in the file is generic CUDA using the CUDA Runtime API
//// and simple CUDA kernels to initialize matrices and compute the general matrix product.
////
/////////////////////////////////////////////////////////////////////////////////////////////////////
//
///// Kernel to initialize a matrix with small integers.
//__global__ void InitializeMatrix_kernel(
//	float *matrix,
//	int ldm,
//	int rows,
//	int columns,
//	int seed = 0) {
//
//	int i = threadIdx.x + blockIdx.x * blockDim.x;
//	int j = threadIdx.y + blockIdx.y * blockDim.y;
//
//	if (i < rows && j < columns) {
//		int offset = i + j * ldm;
//
//		// Generate arbitrary elements.
//		int const k = 16807;
//		int const m = 16;
//		float value = float(((offset + seed) * k % m) - m / 2);
//
//		matrix[offset] = value;
//	}
//}
//
///// Simple function to initialize a matrix to arbitrary small integers.
//
//cudaError_t InitializeMatrix(float *matrix, int ldm, int rows, int columns, int seed = 0) {
//
//	dim3 block(16, 16);
//	dim3 grid(
//		(rows + block.x - 1) / block.x,
//		(columns + block.y - 1) / block.y
//	);
//
//	InitializeMatrix_kernel << < grid, block >> >(matrix, ldm, rows, columns, seed);
//
//	return cudaGetLastError();
//}
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
//template<typename _type>
//cudaError_t AllocateMatrix2D(_type **matrix, int rows, int columns, int &ldm, int interval = 512) {
//	cudaError_t result;
//
//	int interval_type = interval / sizeof(_type);
//	ldm = ((rows + interval_type - 1) / interval_type) * interval_type;
//
//	size_t sizeof_matrix = sizeof(_type) * ldm * columns;
//
//	// Allocate device memory.
//	result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Failed to allocate matrix: "
//			<< cudaGetErrorString(result) << std::endl;
//		return result;
//	}
//
//	// Clear the allocation.
//	result = cudaMemset(*matrix, 0, sizeof_matrix);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Failed to clear matrix device memory: "
//			<< cudaGetErrorString(result) << std::endl;
//		return result;
//	}
//
//	return result;
//}
//
//
//
///// Allocates device memory for a matrix then fills with arbitrary small integers.
//template<typename _type>
//cudaError_t AllocateMatrixAny(_type **matrix, int ldm, int rows, int columns) {
//	cudaError_t result;
//
//	size_t sizeof_matrix = sizeof(_type) * ldm * columns;
//
//	// Allocate device memory.
//	result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Failed to allocate matrix: "
//			<< cudaGetErrorString(result) << std::endl;
//		return result;
//	}
//
//	// Clear the allocation.
//	result = cudaMemset(*matrix, 0, sizeof_matrix);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Failed to clear matrix device memory: "
//			<< cudaGetErrorString(result) << std::endl;
//		return result;
//	}
//
//	return result;
//}
//
//cudaError_t AllocateMatrix(float **matrix, int ldm, int rows, int columns, int seed = 0) {
//	cudaError_t result;
//
//	size_t sizeof_matrix = sizeof(float) * ldm * columns;
//
//	// Allocate device memory.
//	result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Failed to allocate matrix: "
//			<< cudaGetErrorString(result) << std::endl;
//		return result;
//	}
//
//	// Clear the allocation.
//	result = cudaMemset(*matrix, 0, sizeof_matrix);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Failed to clear matrix device memory: "
//			<< cudaGetErrorString(result) << std::endl;
//		return result;
//	}
//
//	// Initialize matrix elements to arbitrary small integers.
//	result = InitializeMatrix(*matrix, ldm, rows, columns, seed);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Failed to initialize matrix: "
//			<< cudaGetErrorString(result) << std::endl;
//		return result;
//	}
//
//	return result;
//}
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
//
///// Naive reference GEMM computation.
//__global__ void ReferenceGemm_kernel(
//	int M,
//	int N,
//	int K,
//	float alpha,
//	float const *A,
//	int lda,
//	float const *B,
//	int ldb,
//	float beta,
//	float *C,
//	int ldc) {
//
//	int i = threadIdx.x + blockIdx.x * blockDim.x;
//	int j = threadIdx.y + blockIdx.y * blockDim.y;
//
//	if (i < M && j < N) {
//		float accumulator = 0;
//
//		for (int k = 0; k < K; ++k) {
//			accumulator += A[i + k * lda] * B[k + j * ldb];
//		}
//
//		C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
//	}
//}
//
///// Reference GEMM computation.
//cudaError_t ReferenceGemm(
//	int M,
//	int N,
//	int K,
//	float alpha,
//	float const *A,
//	int lda,
//	float const *B,
//	int ldb,
//	float beta,
//	float *C,
//	int ldc) {
//
//	dim3 block(16, 16);
//	dim3 grid(
//		(M + block.x - 1) / block.x,
//		(N + block.y - 1) / block.y
//	);
//
//	ReferenceGemm_kernel << < grid, block >> >(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
//
//	return cudaGetLastError();
//}
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
//
///// Define a CUTLASS GEMM template and launch a GEMM kernel.
//template<typename _type>
//void matrixMulCPU(_type *C, int ldc, const _type *A, int lda, const _type *B, int ldb, int hA, int wA, int wB)
//{
//	for (int i = 0; i < hA; ++i)
//		for (int j = 0; j < wB; ++j)
//		{
//			typename _type::Accum_t sum(0);
//
//			for (int k = 0; k < wA; ++k)
//			{
//				_type a = A[i * lda + k];
//				_type b = B[k * ldb + j];
//				sum += a * b;
//			}
//
//			C[i * ldc + j] = (_type)sum;
//		}
//}
//
//
//// original cutlass example for single precision float
///// Define a CUTLASS GEMM template and launch a GEMM kernel.
//cudaError_t CutlassSgemmNN(
//	int M,
//	int N,
//	int K,
//	float alpha,
//	float const *A,
//	int lda,
//	float const *B,
//	int ldb,
//	float beta,
//	float *C,
//	int ldc) {
//
//	// Define type definition for single-precision CUTLASS GEMM with column-major
//	// input matrices and 128x128x8 threadblock tile size.
//	//
//	// Note, GemmTraits<> is a generic template defined for various general matrix product
//	// computations within CUTLASS. It is intended to be maximally flexible, and consequently
//	// it contains numerous template arguments.
//	//
//	// To keep the interface manageable, several helpers are defined for plausible compositions
//	// including the following example for single-precision GEMM. Typical values are used as
//	// default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
//	//
//
//	typedef cutlass::gemm::SgemmTraits<
//		cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
//		cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
//		cutlass::Shape<8, 128, 128>            // threadblock tile size
//	>
//		GemmTraits;
//
//	// Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
//	typedef cutlass::gemm::Gemm<GemmTraits> Gemm;
//
//	// Construct and initialize CUTLASS GEMM parameters object.
//	//
//	// One of CUTLASS's design patterns is to define parameters objects that are constructible
//	// in host code and passed to kernels by value. These may include pointers, strides, scalars,
//	// and other arguments needed by Gemm and its components.
//	//
//	// The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
//	// arguments to kernels and (2.) minimized initialization overhead on kernel entry.
//	//
//	typename Gemm::Params params;
//
//	int result = params.initialize(
//		M,     // GEMM M dimension
//		N,     // GEMM N dimension
//		K,     // GEMM K dimension
//		alpha, // scalar alpha
//		A,     // matrix A operand
//		lda,
//		B,     // matrix B operand
//		ldb,
//		beta,  // scalar beta
//		C,     // source matrix C
//		ldc,
//		C,     // destination matrix C (may be different memory than source C matrix)
//		ldc
//	);
//
//	if (result) {
//		std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
//		return cudaErrorInvalidValue;
//	}
//
//	// Launch the CUTLASS GEMM kernel.
//	Gemm::launch(params);
//
//	// Return any errors associated with the launch or cudaSuccess if no error.
//	return cudaGetLastError();
//}
//
///// Allocate several matrices in GPU device memory and call a single-precision
///// CUTLASS GEMM kernel.
//cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta) {
//	cudaError_t result;
//	//
//	// Define several matrices to be used as operands to GEMM kernels.
//	//
//
//	// Compute leading dimensions for each matrix.
//	int lda = M;
//	int ldb = K;
//	int ldc = M;
//
//	// Compute size in bytes of the C matrix.
//	size_t sizeof_C = sizeof(float) * ldc * N;
//
//	// Define pointers to matrices in GPU device memory.
//	float *A;
//	float *B;
//	float *C_cutlass;
//	float *C_reference;
//
//	//
//	// Allocate matrices in GPU device memory with arbitrary seeds.
//	//
//
//	result = AllocateMatrix(&A, lda, M, K, 0);
//
//	if (result != cudaSuccess) {
//		return result;
//	}
//
//	result = AllocateMatrix(&B, ldb, K, N, 17);
//
//	if (result != cudaSuccess) {
//		cudaFree(A);
//		return result;
//	}
//
//	result = AllocateMatrix(&C_cutlass, ldc, M, N, 101);
//
//	if (result != cudaSuccess) {
//		cudaFree(A);
//		cudaFree(B);
//		return result;
//	}
//
//	result = AllocateMatrix(&C_reference, ldc, M, N, 101);
//
//	if (result != cudaSuccess) {
//		cudaFree(A);
//		cudaFree(B);
//		cudaFree(C_cutlass);
//		return result;
//	}
//
//	result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
//			<< cudaGetErrorString(result) << std::endl;
//
//		cudaFree(C_reference);
//		cudaFree(C_cutlass);
//		cudaFree(B);
//		cudaFree(A);
//
//		return result;
//	}
//
//	//
//	// Launch CUTLASS GEMM.
//	//
//
//	result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);
//
//	if (result != cudaSuccess) {
//		std::cerr << "CUTLASS GEMM kernel failed: "
//			<< cudaGetErrorString(result) << std::endl;
//
//		cudaFree(C_reference);
//		cudaFree(C_cutlass);
//		cudaFree(B);
//		cudaFree(A);
//
//		return result;
//	}
//
//	//
//	// Verify.
//	//
//
//	// Launch reference GEMM
//	result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Reference GEMM kernel failed: "
//			<< cudaGetErrorString(result) << std::endl;
//
//		cudaFree(C_reference);
//		cudaFree(C_cutlass);
//		cudaFree(B);
//		cudaFree(A);
//
//		return result;
//	}
//
//	// Copy to host and verify equivalence.
//	std::vector<float> host_cutlass(ldc * N, 0);
//	std::vector<float> host_reference(ldc * N, 0);
//
//	result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Failed to copy CUTLASS GEMM results: "
//			<< cudaGetErrorString(result) << std::endl;
//
//		cudaFree(C_reference);
//		cudaFree(C_cutlass);
//		cudaFree(B);
//		cudaFree(A);
//
//		return result;
//	}
//
//	result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);
//
//	if (result != cudaSuccess) {
//		std::cerr << "Failed to copy Reference GEMM results: "
//			<< cudaGetErrorString(result) << std::endl;
//
//		cudaFree(C_reference);
//		cudaFree(C_cutlass);
//		cudaFree(B);
//		cudaFree(A);
//
//		return result;
//	}
//
//	//
//	// Free device memory allocations.
//	//
//
//	cudaFree(C_reference);
//	cudaFree(C_cutlass);
//	cudaFree(B);
//	cudaFree(A);
//
//	//
//	// Test for bit equivalence of results.
//	//
//
//	if (host_cutlass != host_reference) {
//		std::cerr << "CUTLASS results incorrect." << std::endl;
//
//		return cudaErrorUnknown;
//	}
//
//	return cudaSuccess;
//}
//
//
//template<typename _type>
//cudaError_t MyCutlassSgemmNT(
//	int M,
//	int N,
//	int K,
//	merssene31_t::Accum_t alpha,
//	_type const *A,
//	int lda,
//	_type const *B,
//	int ldb,
//	merssene31_t::Accum_t beta,
//	_type *C,
//	int ldc) {
//
//	// Define type definition for single-precision CUTLASS GEMM with column-major
//	// input matrices and 128x128x8 threadblock tile size.
//	//
//	// Note, GemmTraits<> is a generic template defined for various general matrix product
//	// computations within CUTLASS. It is intended to be maximally flexible, and consequently
//	// it contains numerous template arguments.
//	//
//	// To keep the interface manageable, several helpers are defined for plausible compositions
//	// including the following example for single-precision GEMM. Typical values are used as
//	// default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
//	//
//	//SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>> mygemmconf;
//	typedef cutlass::gemm::SgemmTraits <
//		cutlass::MatrixLayout::kRowMajor,   // layout of A matrix
//		cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
//		cutlass::Shape<8, 128, 128>,           // threadblock tile size
//		cutlass::gemm::LinearScaling<merssene31_t>,
//		/// Tile size for thread-level GEMM (K-by-N-by-M)
//		cutlass::Shape<8, 8, 8>,
//		/// The number of floats loaded in one LDG for A.
//		1,
//		/// The number of floats loaded in one LDG for B.
//		1,
//		/// The index.
//		int,
//		/// The SGEMM config.
//		SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>>
//		/// The traits class for the epilogue.
//		//,cutlass::gemm::SimplifiedGemmEpilogueTraits<
//		//SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>>,
//		//cutlass::gemm::LinearScaling<merssene31_t::Accum_t,
//		//cutlass::gemm::FragmentMultiplyAdd<merssene31_t::Accum_t, merssene31_t::Accum_t/*accumulator type*/> > >
//		//
//		//cutlass::gemm::myLinearScaling<merssene31_t::Accum_t>, int>
//		/*typename GemmEpilogueTraits_ =
//		SimplifiedGemmEpilogueTraits<GemmConfig_, EpilogueFunctor_, Index_> >
//		*/
//	>
//		GemmTraits;
//
//	// Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
//	typedef cutlass::gemm::Gemm<GemmTraits> Gemm;
//
//	// Construct and initialize CUTLASS GEMM parameters object.
//	//
//	// One of CUTLASS's design patterns is to define parameters objects that are constructible
//	// in host code and passed to kernels by value. These may include pointers, strides, scalars,
//	// and other arguments needed by Gemm and its components.
//	//
//	// The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
//	// arguments to kernels and (2.) minimized initialization overhead on kernel entry.
//	//
//	typename Gemm::Params params;
//
//	//GemmTraits::Epilogue a;
//	int result = params.initialize(
//		N,     // GEMM M dimension
//		K,     // GEMM N dimension
//		M,     // GEMM K dimension
//		alpha, // scalar alpha
//		A,     // matrix A operand
//		lda,
//		B,     // matrix B operand
//		ldb,
//		beta,  // scalar beta
//		C,     // source matrix C
//		ldc,
//		C,     // destination matrix C (may be different memory than source C matrix)
//		ldc
//	);
//
//	if (result) {
//		std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
//		return cudaErrorInvalidValue;
//	}
//
//	cudaEvent_t start, end;
//	cudaEventCreate(&start);
//	cudaEventCreate(&end);
//	cudaEventRecord(start);
//
//	// Launch the CUTLASS GEMM kernel.
//	Gemm::launch(params);
//	cudaEventRecord(end);
//	cudaEventSynchronize(end);
//
//	float average_ms = 0;
//	cudaEventElapsedTime(&average_ms, start, end);
//	printf("Mat size: M=%d, N=%d, K=%d, Time: %fms\n", M, N, K, average_ms);
//
//
//	// Return any errors associated with the launch or cudaSuccess if no error.
//	return cudaGetLastError();
//}
//
//
//template<typename _type>
//void matrixMulCPUNT(_type *C, int ldc, const _type *A, int lda, const _type *B, int ldb, int hA, int wA, int hB)
//{
//	for (int i = 0; i < hA; ++i)
//		for (int j = 0; j < hB; ++j)
//		{
//			typename _type::Accum_t sum(0);
//
//			for (int k = 0; k < wA; ++k)
//			{
//				_type a = A[i * lda + k];
//				_type b = B[j * ldb + k];
//				sum += a * b;
//			}
//
//			C[i + j * ldc] = (_type)sum;
//		}
//}
//
//
//void testGemmNT(int width_a /*m*/, int height_a, int height_b)
//{
//	cudaStream_t stream = NULL;
//	int h_lda = width_a;
//	int h_ldb = width_a;
//	int h_ldc = height_a;
//
//	std::vector<merssene31_t> h_A(h_lda * height_a, 1);
//	std::vector<merssene31_t> h_B(h_ldb * height_b, 2);
//	std::vector<merssene31_t> h_C(h_ldc * height_b, 0);
//	std::vector<merssene31_t> h_C_ref(h_ldc * height_b, 0);
//
//	randvec((merssene31_t::basic_t*)h_A.data(), width_a, height_a, width_a, merssene31_t::p);
//	randvec((merssene31_t::basic_t*)h_B.data(), width_a, height_b, width_a, merssene31_t::p);
//
//
//	Mat<merssene31_t> A(width_a, height_a); // A is width_a rows by height_a columns
//	Mat<merssene31_t> B(width_a, height_b); // B is width_a rows by height_b columns
//	Mat<merssene31_t> C(height_a, height_b); // C is height_a rows by height_b columns
//
//
//	cudaSafeCall(cudaMemcpyAsync(A._ptr, h_A.data(), h_A.size() * sizeof(merssene31_t), cudaMemcpyHostToDevice, stream));
//	cudaSafeCall(cudaMemcpyAsync(B._ptr, h_B.data(), h_B.size() * sizeof(merssene31_t), cudaMemcpyHostToDevice, stream));
//
//	cudaSafeCall(MyCutlassSgemmNT(width_a, height_a, height_b, merssene31_t::Accum_t(1),
//		A._ptr, A._ldm, B._ptr, B._ldm, merssene31_t::Accum_t(0), C._ptr, C._ldm));
//
//	cudaSafeCall(cudaMemcpyAsync(h_C.data(), C._ptr, h_C.size() * sizeof(merssene31_t), cudaMemcpyDeviceToHost, stream));
//
//	matrixMulCPUNT(h_C_ref.data(), h_ldc, h_A.data(), h_lda,
//		h_B.data(), h_ldb, height_a, width_a, height_b);
//
//
//	if (h_C != h_C_ref)
//	{
//		printf("Error\n");
//		for (int i = 0; i < h_C.size(); ++i)
//			if (h_C[i]._v != h_C_ref[i]._v)
//				bool yyy = true;
//	}
//}
//
//
//template<typename _type>
//cudaError_t MyCutlassSgemmNN(
//	int M,
//	int N,
//	int K,
//	merssene31_t::Accum_t alpha,
//	_type const *A,
//	int lda,
//	_type const *B,
//	int ldb,
//	merssene31_t::Accum_t beta,
//	_type *C,
//	int ldc) {
//
//	// Define type definition for single-precision CUTLASS GEMM with column-major
//	// input matrices and 128x128x8 threadblock tile size.
//	//
//	// Note, GemmTraits<> is a generic template defined for various general matrix product
//	// computations within CUTLASS. It is intended to be maximally flexible, and consequently
//	// it contains numerous template arguments.
//	//
//	// To keep the interface manageable, several helpers are defined for plausible compositions
//	// including the following example for single-precision GEMM. Typical values are used as
//	// default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
//	//
//	//SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>> mygemmconf;
//	typedef cutlass::gemm::SgemmTraits <
//		cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
//		cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
//		cutlass::Shape<8, 128, 128>,           // threadblock tile size
//		cutlass::gemm::LinearScaling<merssene31_t>,
//		/// Tile size for thread-level GEMM (K-by-N-by-M)
//		cutlass::Shape<8, 8, 8>,
//		/// The number of floats loaded in one LDG for A.
//		1,
//		/// The number of floats loaded in one LDG for B.
//		1,
//		/// The index.
//		int,
//		/// The SGEMM config.
//		SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>>
//		/// The traits class for the epilogue.
//		//,cutlass::gemm::SimplifiedGemmEpilogueTraits<
//		//SMersseneGemmConfig31<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>>,
//		//cutlass::gemm::LinearScaling<merssene31_t::Accum_t,
//		//cutlass::gemm::FragmentMultiplyAdd<merssene31_t::Accum_t, merssene31_t::Accum_t/*accumulator type*/> > >
//		//
//		//cutlass::gemm::myLinearScaling<merssene31_t::Accum_t>, int>
//		/*typename GemmEpilogueTraits_ =
//		SimplifiedGemmEpilogueTraits<GemmConfig_, EpilogueFunctor_, Index_> >
//		*/
//	>
//		GemmTraits;
//
//	// Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
//	typedef cutlass::gemm::Gemm<GemmTraits> Gemm;
//
//	// Construct and initialize CUTLASS GEMM parameters object.
//	//
//	// One of CUTLASS's design patterns is to define parameters objects that are constructible
//	// in host code and passed to kernels by value. These may include pointers, strides, scalars,
//	// and other arguments needed by Gemm and its components.
//	//
//	// The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
//	// arguments to kernels and (2.) minimized initialization overhead on kernel entry.
//	//
//	typename Gemm::Params params;
//
//	//GemmTraits::Epilogue a;
//	int result = params.initialize(
//		M,     // GEMM M dimension
//		N,     // GEMM N dimension
//		K,     // GEMM K dimension
//		alpha, // scalar alpha
//		A,     // matrix A operand
//		lda,
//		B,     // matrix B operand
//		ldb,
//		beta,  // scalar beta
//		C,     // source matrix C
//		ldc,
//		C,     // destination matrix C (may be different memory than source C matrix)
//		ldc
//	);
//
//	if (result) {
//		std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
//		return cudaErrorInvalidValue;
//	}
//
//	cudaEvent_t start, end;
//	cudaEventCreate(&start);
//	cudaEventCreate(&end);
//	cudaEventRecord(start);
//
//	// Launch the CUTLASS GEMM kernel.
//	Gemm::launch(params);
//	cudaEventRecord(end);
//	cudaEventSynchronize(end);
//
//	float average_ms = 0;
//	cudaEventElapsedTime(&average_ms, start, end);
//	printf("Mat size: M=%d, N=%d, K=%d, Time: %fms\n", M, N, K, average_ms);
//
//
//	// Return any errors associated with the launch or cudaSuccess if no error.
//	return cudaGetLastError();
//}
//
//
//template <typename T>
//void randvec(T* ptr, size_t size)
//{
//	for (size_t i = 0; i < size; ++i)
//	{
//		ptr[i] = 1;// T(rand()*rand());
//	}
//}
//
//
///// CUTLASS GEMM5 kernel.
//cudaError_t TestCutlassGemmMine(int M, int N, int K, float alpha, float beta)
//{
//	typedef SMersenne31Naive testType;
//	cudaStream_t stream = NULL;
//	//
//	// Define several matrices to be used as operands to GEMM kernels.
//	//
//
//	// Compute leading dimensions for each matrix.
//	int h_lda = M;
//	int h_ldb = K;
//	int h_ldc = M;
//
//	std::vector<merssene31_t> h_A(h_lda * K, 1);
//	std::vector<merssene31_t> h_B(h_ldb * N, 2);
//	std::vector<merssene31_t> h_C(h_ldc * N, 0);
//	std::vector<merssene31_t> h_C_ref(h_ldc * N, 0);
//
//	randvec(h_A.data(), h_A.size());
//	randvec(h_B.data(), h_B.size());
//
//	// Define pointers to matrices in GPU device memory.
//	merssene31_t *A;
//	merssene31_t *B;
//	merssene31_t *C;
//
//	//
//	// Allocate matrices in GPU device memory with arbitrary seeds.
//	//
//	int lda = 0, ldb = 0, ldc = 0;
//
//	cudaSafeCall(AllocateMatrix2D(&A, M, K, lda));
//	cudaSafeCall(AllocateMatrix2D(&B, K, N, ldb));
//	cudaSafeCall(AllocateMatrix2D(&C, M, N, ldc));
//
//	//cudaMemcpy2DAsync(A, lda * sizeof(merssene31_t), h_A.data(), h_lda * sizeof(merssene31_t), M, K, cudaMemcpyHostToDevice, stream);
//	cudaSafeCall(cudaMemcpyAsync(A, h_A.data(), h_A.size() * sizeof(merssene31_t), cudaMemcpyHostToDevice, stream));
//	cudaSafeCall(cudaMemcpyAsync(B, h_B.data(), h_B.size() * sizeof(merssene31_t), cudaMemcpyHostToDevice, stream));
//
//	////
//	//// Launch CUTLASS GEMM.
//	////
//
//	cudaSafeCall(MyCutlassSgemmNN(M, N, K, merssene31_t::Accum_t(1),
//		A, lda, B, ldb, merssene31_t::Accum_t(0), C, ldc));
//
//
//	cudaSafeCall(cudaMemcpyAsync(h_C.data(), C, h_C.size() * sizeof(merssene31_t), cudaMemcpyDeviceToHost, stream));
//	//
//	// Verify.
//	//
//	/*int lda = M;
//	int ldb = K;
//	int ldc = M;
//
//	cudaSafeCall(AllocateMatrixAny(&A, lda, M, K));
//	cudaSafeCall(AllocateMatrixAny(&B, ldb, K, N));
//	cudaSafeCall(AllocateMatrixAny(&C, ldc, M, N));
//	*/
//	//void matrixMulCPU(int hA, int wA, int wB)
//	matrixMulCPU((testType*)h_C_ref.data(), ldc, (testType*)h_B.data(), ldb, (testType*)h_A.data(), lda,
//		N, K, M);
//
//	matrixMulCPU((testType*)h_C_ref.data(), ldc, (testType*)h_A.data(), lda, (testType*)h_B.data(), ldb,
//		K, M, N);
//
//
//	matrixMulCPUCompare<unsigned int, SMersenne31Naive, myTypeUint32Mixed>
//		((unsigned int*)h_C_ref.data(), ldc, (unsigned int*)h_B.data(),
//			ldb, (unsigned int*)h_A.data(), lda,
//			N, K, M);
//
//
//	printf("%s\n", h_C_ref == h_C ? "SUCCESS" : "ERROR");
//
//
//	return cudaSuccess;
//}
