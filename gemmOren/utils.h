#pragma once
// a macro to deal with cuda error values. should wrp avery cuda API
#define cudaSafeCall(a) {\
cudaError_t err = a; \
if (err != cudaSuccess)\
{\
printf("Cuda Error in line %d: %s\n", __LINE__, cudaGetErrorString(err));\
exit(0);\
}\
}


#define UGLY_ERROR {\
printf("//////////////////////////////////////////////////////////\n");\
printf("Error in line %d, function %s\n", __LINE__, __FUNCTION__);\
printf("//////////////////////////////////////////////////////////\n"); \
}
