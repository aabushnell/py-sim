#ifndef SIMS_H_DEFINED
#define SIMS_H_DEFINED

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

double **alloc_model(int n_nodes);

void dealloc_model(double **model_ptrs);

void calc_p_pi(int n_nodes, double *tau, double *A, double *L, double *P,
               double *Pi, double coeff_theta, double diff_limit,
               int debug_level, bool relative_diff);

void calc_Y(int n_nodes, double *A, double *L, double *Y, double *Pi,
            double coeff_theta, int debug_level);
void calc_X(int n_nodes, double *tau, double *X, double *Y, double *P,
            double *Pi, double coeff_theta, int debug_level);
void calc_Xi(int n_nodes, double *t, double *Xi, double *P, double *Pi,
             double coeff_theta, int debug_level, bool allow_internal_trade);

void update_A(int n_nodes, double *Xi, double *A, double coeff_eta,
              double coeff_beta, double coeff_sigma, int debug_level,
              bool normalized, bool log_A, bool translate_A);
void update_L(int n_nodes, double *L, double *B, double *Y, double *P,
              double coeff_a, double coeff_b, double coeff_f, double coeff_d,
              double coeff_xi, double coeff_lambda, int debug_level);
void update_t(int n_nodes, double *t, double *Xi, double *Xi_prev,
              double coeff_chi, int debug_level);

#ifdef __cplusplus
}
#endif

#endif // SIMS_H_DEFINED
