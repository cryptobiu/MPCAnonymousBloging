//
// Created by liork on 3/6/19.
//

#ifndef MPCANONYMOUSBLOGING_CUDAGEMM_H
#define MPCANONYMOUSBLOGING_CUDAGEMM_H

#include "MersseneTypes.h"
typedef SMersenne31Classic merssene31_t;
typedef SMersenne61Gpu merssene61_t;




void GemmTNTiles31(merssene31_t* h_A, size_t h_lda,
                   merssene31_t* h_B, size_t h_ldb,
                   merssene31_t* h_C, size_t h_ldc,
                   size_t m, size_t width_a, size_t width_b, size_t tile_size,
                   const std::vector<int>& devices, bool cheat);


void processNN31(merssene31_t* h_C,
                 merssene31_t* h_A, size_t rowA, size_t colA,
                 merssene31_t* h_B, size_t rowB, size_t colB,
                 int devicesID);

#endif //MPCANONYMOUSBLOGING_CUDAGEMM_H
