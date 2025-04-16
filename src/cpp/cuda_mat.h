#ifndef CUDA_MAT_H_DEFINED
#define CUDA_MAT_H_DEFINED

#ifdef __cplusplus
extern "C" {
#endif

void calc_tau_cublas(int n_nodes, const double *t, double *tau,
                     double coeff_theta, int n_iterations, int debug_level);

#ifdef __cplusplus
}
#endif

#endif // !CUDA_MAT_H_DEFINED
