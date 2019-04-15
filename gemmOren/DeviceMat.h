#pragma once
#include "utils.h"

template<typename _type, size_t interval = 512>
struct Mat
{
	Mat(size_t rows, size_t columns)
	{
		_rows = rows;
		_columns = columns;

		size_t  leading_dimension = rows;
		size_t interval_type = interval / sizeof(_type);
		_ldm = ((leading_dimension + interval_type - 1) / interval_type) * interval_type;

		size_t sizeof_matrix = sizeof(_type) * _ldm * columns;

		// Allocate device memory.
		cudaSafeCall(cudaMalloc(reinterpret_cast<void **>(&_ptr), sizeof_matrix));

	}

	~Mat(){
		cudaSafeCall(cudaFree(_ptr));
	}
	void SetZero(cudaStream_t& stream)
	{
		cudaSafeCall(cudaMemset2DAsync(_ptr, _ldm * sizeof(_type), 0, _rows * sizeof(_type), _columns, stream));
	}
	size_t  _rows;
	size_t _columns;
	size_t _ldm; // of _type
	_type* _ptr;
};
