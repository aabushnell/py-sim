#include "sims.h"
// #include "cuda_mat.h"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>

void print_array(int n_nodes, double *arr) {
  for (auto i = 0; i < n_nodes; i++) {
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

double **alloc_model(int n_nodes) {
  // number of arrays in model
  int model_ptr_size = 12;
  double **model_ptrs = (double **)malloc(sizeof(double *) * model_ptr_size);

  // number of size n arrays
  int n_count = 6;
  // number of size n*n arrays
  int n_2_count = 6;
  size_t data_size = n_nodes * n_nodes * n_2_count + n_nodes * n_count;
  double *model_data = (double *)malloc(sizeof(double) * data_size);
  // initialize all data to zero
  for (size_t i = 0; i < data_size; i++) {
    model_data[i] = 0;
  }

  // t [n^2]
  model_ptrs[0] = model_data + n_nodes * n_nodes * 0 + n_nodes * 0;
  // tau [n^2]
  model_ptrs[1] = model_data + n_nodes * n_nodes * 1 + n_nodes * 0;
  // X [n^2]
  model_ptrs[2] = model_data + n_nodes * n_nodes * 2 + n_nodes * 0;
  // Xi [n^2]
  model_ptrs[3] = model_data + n_nodes * n_nodes * 3 + n_nodes * 0;
  // Xi_prev [n^2]
  model_ptrs[4] = model_data + n_nodes * n_nodes * 4 + n_nodes * 0;
  // Xi_temp [n^2]
  model_ptrs[5] = model_data + n_nodes * n_nodes * 5 + n_nodes * 0;
  // A [n]
  model_ptrs[6] = model_data + n_nodes * n_nodes * 6 + n_nodes * 0;
  // L [n]
  model_ptrs[7] = model_data + n_nodes * n_nodes * 6 + n_nodes * 1;
  // B [n]
  model_ptrs[8] = model_data + n_nodes * n_nodes * 6 + n_nodes * 2;
  // Y [n]
  model_ptrs[9] = model_data + n_nodes * n_nodes * 6 + n_nodes * 3;
  // P [n]
  model_ptrs[10] = model_data + n_nodes * n_nodes * 6 + n_nodes * 4;
  // Pi [n]
  model_ptrs[11] = model_data + n_nodes * n_nodes * 6 + n_nodes * 5;

  return model_ptrs;
}

void dealloc_model(double **model_ptrs) {
  double *model_data = model_ptrs[0];
  free(model_data);
  free(model_ptrs);
}

void calc_p_pi(int n_nodes, double *tau, double *A, double *L, double *P,
               double *Pi, double coeff_theta, double diff_limit,
               int debug_level, bool relative_diff) {
  // note** at least for now only calculates P and Pi as a single vector for
  // simplicity

  // allocate temporary buffers for intermediate values
  double *P_temp = new double[n_nodes];
  double *Pi_temp = new double[n_nodes];
  double *x = new double[n_nodes];

  double max_diff = 1;
  int cycles = 0;

  while (max_diff > diff_limit) {
    max_diff = 0;
    for (auto j = 0; j < n_nodes; j++) {
      // start x_max at -inf to find largest x_i value
      double x_max = -std::numeric_limits<double>::infinity();
      for (auto i = 0; i < n_nodes; i++) {
        // formula for x_i
        double x_i =
            (coeff_theta / (coeff_theta + 1)) *
                (std::log(A[i]) + std::log(L[i])) +
            (coeff_theta * coeff_theta / (coeff_theta + 1)) * std::log(Pi[i]) -
            coeff_theta * std::log(tau[i * n_nodes + j]);
        if (x_i > x_max) {
          x_max = x_i;
        }
        x[i] = x_i;
      }
      if (debug_level > 2) {
        std::cout << "[" << j << "] x_max = " << x_max << std::endl;
      }
      double sum = 0;
      // logsumexponent calculation
      for (auto i = 0; i < n_nodes; i++) {
        sum += std::exp(x[i] - x_max);
      }
      if (debug_level > 2) {
        std::cout << "[" << j << "] sum = " << sum << std::endl;
      }
      // full P value (in absolute terms)
      double P_new = std::exp((-1 / coeff_theta) * (x_max + std::log(sum)));
      if (debug_level > 2) {
        std::cout << "[" << j << "] P = " << P_new << std::endl;
      }
      // compare to previous value and find maximum difference
      double diff = 0;
      if (relative_diff) {
        diff = std::abs(P[j] - P_new) / P[j];
      } else {
        diff = std::abs(P[j] - P_new);
      }
      P_temp[j] = P_new;
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
    // refresh P and Pi buffers with new values
    for (auto j = 0; j < n_nodes; j++) {
      P[j] = P_temp[j];
      Pi[j] = P_temp[j];
    }
    if (debug_level > 1) {
      std::cout << "# " << cycles << ": " << max_diff << std::endl;
    }
    cycles++;
  }

  if (debug_level > 0) {
    std::cout << "# Pi and P arrays calculated #" << std::endl;
    if (debug_level > 1) {
      std::cout << "***** Number of cycles required: " << cycles << std::endl;
      if (debug_level > 2) {
        std::cout << "*** Pi ***" << std::endl;
        print_array(n_nodes, Pi);
        std::cout << "*** P ***" << std::endl;
        print_array(n_nodes, P);
      }
    }
  }

  // free temporary buffers
  delete[] P_temp;
  delete[] Pi_temp;
  delete[] x;
}

void calc_Y(int n_nodes, double *A, double *L, double *Y, double *Pi,
            double coeff_theta, int debug_level) {
  // Y is derived from A, L, and Pi
  for (auto i = 0; i < n_nodes; i++) {
    Y[i] = std::pow(A[i] * L[i] / Pi[i], coeff_theta / (coeff_theta + 1));
  }

  if (debug_level > 0) {
    std::cout << "# Y array calculated #" << std::endl;
    if (debug_level > 2) {
      std::cout << "*** Y ***" << std::endl;
      print_array(n_nodes, Y);
    }
  }
}

void calc_X(int n_nodes, double *tau, double *X, double *Y, double *P,
            double *Pi, double coeff_theta, int debug_level) {
  for (auto i = 0; i < n_nodes; i++) {
    for (auto j = 0; j < n_nodes; j++) {
      X[i * n_nodes + j] = std::pow(tau[i * n_nodes + j], -1 * coeff_theta) *
                           Y[i] * std::pow(Pi[i], coeff_theta) * Y[j] *
                           std::pow(P[j], coeff_theta);
    }
  }

  if (debug_level > 0) {
    std::cout << "# X matrix calculated #" << std::endl;
    if (debug_level > 1) {
      std::cout << "*** X ***" << std::endl;
      // not implemented
      // print_matrix(n_nodes, X);
    }
  }
}

void calc_Xi(int n_nodes, double *t, double *Xi, double *P, double *Pi,
             double coeff_theta, int debug_level, bool allow_internal_trade) {

  for (auto k = 0; k < n_nodes; k++) {
    for (auto l = 0; l < n_nodes; l++) {
      // allows cells to 'trade' with themselves
      if (k == l && allow_internal_trade) {
        Xi[k * n_nodes + l] = std::pow(P[k], -1 * coeff_theta) *
                              std::pow(Pi[l], -1 * coeff_theta);
      } else if (t[k * n_nodes + l] == 0) {
        Xi[k * n_nodes + l] = 0;
      } else {
        Xi[k * n_nodes + l] = std::pow(t[k * n_nodes + l], -1 * coeff_theta) *
                              std::pow(P[k], -1 * coeff_theta) *
                              std::pow(Pi[l], -1 * coeff_theta);
      }
    }
  }

  if (debug_level > 0) {
    std::cout << "# Xi matrix calculated #" << std::endl;
    if (debug_level > 1) {
      std::cout << "*** Xi_ij ***" << std::endl;
      // not implemented
      // print_matrix(n_nodes, Xi);
    }
    std::cout << "#### Cycle Simulated ####" << std::endl;
  }
}

void update_A(int n_nodes, double *Xi, double *A, double coeff_eta,
              double coeff_beta, double coeff_sigma, int debug_level,
              bool normalized, bool log_A, bool translate_A) {

  double *A_temp = new double[n_nodes];

  double A_min = std::numeric_limits<double>::infinity();
  if (translate_A) {
    for (int i = 0; i < n_nodes; i++) {
      if (A[i] < A_min) {
        A_min = A[i];
      }
    }
  }

  for (int i = 0; i < n_nodes; i++) {
    if (A[i] == 0) {
      continue;
    }
    double Xi_sum = 0;
    if (normalized) {
      for (int l = 0; l < n_nodes; l++) {
        Xi_sum += Xi[l * n_nodes + i];
      }
    }
    double sum = 0;
    for (int j = 0; j < n_nodes; j++) {
      double A_transformed = A[j];
      if (translate_A) {
        A_transformed -= A_min;
      }
      double Xi_transformed = Xi[j * n_nodes + i];
      if (normalized) {
        Xi_transformed = std::pow(Xi_transformed / Xi_sum, coeff_sigma);
      }
      sum += Xi_transformed * std::pow(A_transformed, coeff_beta);
    }
    if (log_A) {
      double log_A_i = std::log(A[i]);
      log_A_i += coeff_eta * sum / A[i];
      A_temp[i] = std::exp(log_A_i);
    } else {
      A_temp[i] = A[i] + coeff_eta * sum;
    }
  }

  for (int i = 0; i < n_nodes; i++) {
    A[i] = A_temp[i];
  }
  delete[] A_temp;

  if (debug_level > 2) {
    print_array(n_nodes, A);
  }
}

void update_L(int n_nodes, double *L, double *B, double *Y, double *P,
              double coeff_a, double coeff_b, double coeff_f, double coeff_d,
              double coeff_xi, double coeff_lambda, int debug_level) {

  for (int i = 0; i < n_nodes; i++) {
    double log_L_i = std::log(L[i]);
    double y = (std::pow(B[i], coeff_xi) *
                std::pow(Y[i] / (P[i] * L[i]), 1 - coeff_xi)) /
               100000.0;
    double delta_log_L =
        coeff_lambda * (coeff_f * (1 - std::exp(-1 * coeff_a * y)) -
                        (std::pow(y, -1 * coeff_b) - coeff_d));
    log_L_i += delta_log_L;
    L[i] = std::exp(log_L_i);
  }
}

void update_t(int n_nodes, double *t, double *Xi, double *Xi_prev,
              double coeff_chi, int debug_level) {

  // T = 1
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_nodes; j++) {
      if (t[i * n_nodes + j] > 0) {
        double log_t = std::log(t[i * n_nodes + j]);
        log_t += -1 * coeff_chi *
                 std::log(Xi[i * n_nodes + j] / Xi_prev[i * n_nodes + j]);
        t[i * n_nodes + j] = std::max(std::exp(log_t), 2.0);
      }
    }
  }
}
