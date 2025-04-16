#include "sims.hpp"
#include "cuda_mat.hpp"
#include "cuda_vec.h"
#include "matrix.hpp"
#include "parse_img.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

void print_array(int dim, double *arr) {
  for (auto i = 0; i < dim; i++) {
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

void build_array(double val, int dim, double *arr) {
  for (auto i = 0; i < dim; i++) {
    arr[i] = val;
  }
}

void print_edge_weights(double *edges, int node_count) {
  for (auto j = 0; j < node_count; j++) {
    for (auto i = 0; i < node_count; i++) {
      printf("%2.2f ", edges[j * node_count + i]);
    }
    std::cout << std::endl;
  }
}

void search_nodes(int start_node, int x, int y, const int *graph, int width,
                  int height, int node_count, bool *crawled,
                  double *edge_weights, cost_modifiers costs,
                  double distance_travelled) {
  crawled[y * width + x] = true;
  bool next_to_node = false;
  int start_graph_val = graph[y * width + x];
  for (auto dx = -1; dx < 2; dx++) {
    for (auto dy = -1; dy < 2; dy++) {
      if ((dx == 0 && dy == 0) || x + dx < 0 || x + dx == width || y + dy < 0 ||
          y + dy == height) {
        continue;
      }
      int graph_val = graph[(y + dy) * width + x + dx];
      bool crawled_node = crawled[(y + dy) * width + x + dx];
      if (graph_val >= 0 && crawled_node == false) {
        next_to_node = true;
        double add_distance = std::pow(std::abs(dx) + std::abs(dy), 0.5);
        if (start_graph_val == TILE_SEA) {
          add_distance *= costs.sea_cost;
        }
        double cost = costs.base_cost * (distance_travelled + add_distance);
        double min_cost = edge_weights[start_node * node_count + graph_val];
        if (min_cost == 0 || cost < min_cost) {
          // std::cout << "cost = " << cost << std::endl;
          edge_weights[start_node * node_count + graph_val] = cost;
          edge_weights[graph_val * node_count + start_node] = cost;
        }
      }
    }
  }
  for (auto dx = -1; dx < 2; dx++) {
    for (auto dy = -1; dy < 2; dy++) {
      if ((dx == 0 && dy == 0) || x + dx < 0 || x + dx == width || y + dy < 0 ||
          y + dy == height) {
        continue;
      }
      int graph_val = graph[(y + dy) * width + x + dx];
      bool crawled_node = crawled[(y + dy) * width + x + dx];
      if ((graph_val == TILE_LAND || graph_val == TILE_SEA) &&
          crawled_node == false && next_to_node == false) {
        double add_distance = std::pow(std::abs(dx) + std::abs(dy), 0.5);
        if (graph_val == TILE_SEA || start_graph_val == TILE_SEA) {
          add_distance *= costs.sea_cost;
        }
        search_nodes(start_node, x + dx, y + dy, graph, width, height,
                     node_count, crawled, edge_weights, costs,
                     distance_travelled + add_distance);
      }
    }
  }
  return;
}

int crawl_graph(const int *graph, int width, int height, int node_count,
                bool *crawled, double *edge_weights, cost_modifiers costs,
                bool debug) {
  if (debug) {
    std::cout << "Crawling graph of " << node_count << " nodes..." << std::endl;
  }
  for (auto i = 0; i < node_count * node_count; i++) {
    edge_weights[i] = 0;
  }
  // Crawls the entire graph to build the edge weight array
  int node_index;
  for (auto y = 0; y < height; y++) {
    for (auto x = 0; x < width; x++) {
      node_index = graph[y * width + x];
      // Searches for connected nodes from each starting node
      if (node_index >= 0) {
        for (auto i = 0; i < width * height; i++) {
          crawled[i] = false;
        }
        search_nodes(node_index, x, y, graph, width, height, node_count,
                     crawled, edge_weights, costs, 0.0);
      }
    }
  }

  if (debug) {
    std::cout << "*** Edge Weights ***" << std::endl;
    print_edge_weights(edge_weights, node_count);
  }

  return 0;
}

int run_cycle(int t, int dim, const double *t_ij, double *tau_ij,
              const double *A, const double *L, double *Y, double *Pi,
              double *P, double *X_ij, double *Xi_ij, double theta,
              bool update_t, int price_iterations, int debug_level) {

  bool allow_internal_trade = true;

  int iterations = 0;
  if (debug_level > 0) {
    std::cout << "#### Simulating Cycle ####" << std::endl;
  }
  // Derive tau_ij from t_ij -- tau_ij = t_ij^{-\theta}
  if (t == 0 || update_t) {
    generate_tau_ij_cublas(dim, t_ij, tau_ij, theta, 60, debug_level);
    // generate_tau_ij(dim, t_ij, tau_ij, theta, debug_level);
  }

  if (t > 0) {
    price_iterations = 10;
  } else {
    // Initialize intermediates
    for (auto i = 0; i < dim; i++) {
      // Y[i] = 1;
      // Y[i] = std::pow(A[i] * L[i] / Pi[i], theta / (theta + 1));
      // Y[i] = A[i] * L[i];
    }
    // Initial seed value of Pi
    for (auto i = 0; i < dim; i++) {
      // Pi[i] = A[i] * L[i] * std::pow(Y[i], -1 * (theta + 1) / theta);
      // P[i] = A[i] * L[i] * std::pow(Y[i], -1 * (theta + 1) / theta);
      // Pi[i] = 1;
      // P[i] = 1;
      // Pi[i] = std::min(1000 + std::log(tau_ij[1418 * dim + i] + 1) * 10000,
      //                  1000000.0);
      // std::cout << "Pi[" << i << "] = " << Pi[i] << std::endl;
    }
  }

  double *P_temp = new double[dim];
  double *Pi_temp = new double[dim];
  double *Y_temp = new double[dim];
  double *B_ij = new double[dim * dim];
  double *Z = new double[dim];
  double *Z_temp = new double[dim];
  double *x = new double[dim];

  double max_diff_outer = 1;
  double max_diff_inner = 1;
  int cycles_outer = 0;
  int cycles_inner = 0;

  int solution;
  if (t == 0) {
    solution = 4;
  } else {
    solution = 4;
  }
  double diff_limit = std::pow(10.0, -3);
  // Numerically derive equilibrium P and Pi values
  // for (auto cycle = 0; cycle < price_iterations; cycle++) {
  // while (cycles < 100) {
  // cycles = calc_p_pi(dim, P, Pi, tau_ij, A, L, theta, 0.001);
  if (solution == 0) {
    for (auto i = 0; i < dim; i++) {
      for (auto j = 0; j < dim; j++) {
        B_ij[i * dim + j] =
            std::exp((theta / (1 + theta)) * (std::log(A[j]) + std::log(L[j])) -
                     theta * std::log(tau_ij[i * dim + j]));
      }
      Z[i] = Pi[i];
    }

    while (max_diff_outer > diff_limit) {
      max_diff_outer = 0;
      for (auto j = 0; j < dim; j++) {
        double sum = 0;
        for (auto i = 0; i < dim; i++) {
          sum += B_ij[i * dim + j] * Z[j];
        }
        double Z_new = std::pow(sum, -1 * theta / (1 + theta));
        double diff = std::abs(Z[j] - Z_new);
        Z_temp[j] = Z_new;
        if (diff > max_diff_outer) {
          max_diff_outer = diff;
        }
      }
      for (auto j = 0; j < dim; j++) {
        Z[j] = Z_temp[j];
      }
      if (debug_level > 1) {
        std::cout << "# " << cycles_outer << ": " << max_diff_outer
                  << std::endl;
      }
      cycles_outer++;
    }

    for (auto i = 0; i < dim; i++) {
      P[i] = std::pow(Z[i], (1 + theta) / std::pow(theta, 2));
      Pi[i] = P[i];
      Y[i] = std::pow(A[i] * L[i] / Pi[i], theta / (theta + 1));
    }
    cycles_inner = 1;
  }
  if (solution == 1) {
    while (max_diff_outer > diff_limit) {
      cycles_inner = 0;
      max_diff_inner = 1;
      while (max_diff_inner > std::max(max_diff_outer / 10, diff_limit)) {
        max_diff_inner = 0;
        for (auto j = 0; j < dim; j++) {
          double x_max = -std::numeric_limits<double>::infinity();
          for (auto i = 0; i < dim; i++) {
            double x_i =
                theta * (std::log(A[i]) + std::log(L[i]) - std::log(Y[i])) +
                std::log(tau_ij[i * dim + j]);
            if (x_i > x_max) {
              x_max = x_i;
            }
            x[i] = x_i;
          }
          double sum = 0;
          for (auto i = 0; i < dim; i++) {
            sum += std::exp(x[i] - x_max);
          }
          double P_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
          double diff = std::abs(P[j] - P_new) / P[j];
          P_temp[j] = P_new;
          if (diff > max_diff_inner) {
            max_diff_inner = diff;
          }
        }
        for (auto j = 0; j < dim; j++) {
          P[j] = P_temp[j];
        }
        for (auto i = 0; i < dim; i++) {
          double x_max = -std::numeric_limits<double>::infinity();
          for (auto j = 0; j < dim; j++) {
            double x_j = std::log(Y[j]) + theta * (std::log(P[j])) +
                         std::log(tau_ij[i * dim + j]);
            if (x_j > x_max) {
              x_max = x_j;
            }
            x[j] = x_j;
          }
          double sum = 0;
          for (auto j = 0; j < dim; j++) {
            sum += std::exp(x[j] - x_max);
          }
          double Y_new = std::exp((1 / (1 + theta)) *
                                  (theta * (std::log(A[i]) + std::log(L[i])) +
                                   x_max + std::log(sum)));
          double diff = std::abs(Y[i] - Y_new) / Y[i];
          Y_temp[i] = Y_new;
          if (diff > max_diff_inner) {
            max_diff_inner = diff;
          }
        }
        for (auto i = 0; i < dim; i++) {
          Y[i] = Y_temp[i];
        }
        std::cout << "# " << cycles_inner << ": " << max_diff_inner
                  << std::endl;
        cycles_inner++;
      }
      for (auto i = 0; i < dim; i++) {
        Pi_temp[i] = std::exp(std::log(A[i]) + std::log(L[i]) +
                              (-1 * (theta + 1) / theta) * std::log(Y[i]));
      }
      for (auto i = 0; i < dim; i++) {
        Pi[i] = Pi_temp[i];
      }
      max_diff_outer = 0;
      for (auto j = 0; j < dim; j++) {
        double x_max = -std::numeric_limits<double>::infinity();
        for (auto i = 0; i < dim; i++) {
          double x_i =
              theta * (std::log(A[i]) + std::log(L[i]) - std::log(Y[i])) +
              std::log(tau_ij[i * dim + j]);
          if (x_i > x_max) {
            x_max = x_i;
          }
          x[i] = x_i;
        }
        double sum = 0;
        for (auto i = 0; i < dim; i++) {
          sum += std::exp(x[i] - x_max);
        }
        double P_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
        double diff = std::abs(P[j] - P_new) / P[j];
        if (diff > max_diff_outer) {
          max_diff_outer = diff;
        }
      }
      for (auto i = 0; i < dim; i++) {
        double x_max = -std::numeric_limits<double>::infinity();
        for (auto j = 0; j < dim; j++) {
          double x_j = std::log(Y[j]) + theta * (std::log(P[j])) +
                       std::log(tau_ij[i * dim + j]);
          if (x_j > x_max) {
            x_max = x_j;
          }
          x[j] = x_j;
        }
        double sum = 0;
        for (auto j = 0; j < dim; j++) {
          sum += std::exp(x[j] - x_max);
        }
        double Pi_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
        double diff = std::abs(Pi[i] - Pi_new) / Pi[i];
        if (diff > max_diff_outer) {
          max_diff_outer = diff;
        }
      }
      std::cout << "~~ Outer Loop # " << cycles_outer << ": " << max_diff_outer
                << std::endl;
      cycles_outer++;
    }
  }
  if (solution == 2) {
    while (max_diff_outer > diff_limit) {
      cycles_inner = 0;
      max_diff_inner = 1;
      for (auto i = 0; i < dim; i++) {
        Y[i] = std::pow(A[i] * L[i] / Pi[i], theta / (theta + 1));
      }
      while (max_diff_inner > diff_limit) {
        max_diff_inner = 0;
        for (auto j = 0; j < dim; j++) {
          double x_max = -std::numeric_limits<double>::infinity();
          for (auto i = 0; i < dim; i++) {
            double x_i =
                std::log(Y[i]) +
                theta * (std::log(Pi[i]) - std::log(tau_ij[i * dim + j]));
            if (x_i > x_max) {
              x_max = x_i;
            }
            x[i] = x_i;
          }
          double sum = 0;
          for (auto i = 0; i < dim; i++) {
            sum += std::exp(x[i] - x_max);
          }
          double P_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
          double diff = std::abs(P[j] - P_new) / P[j];
          P_temp[j] = P_new;
          if (diff > max_diff_inner) {
            max_diff_inner = diff;
          }
        }
        for (auto j = 0; j < dim; j++) {
          P[j] = P_temp[j];
        }
        for (auto i = 0; i < dim; i++) {
          double x_max = -std::numeric_limits<double>::infinity();
          for (auto j = 0; j < dim; j++) {
            double x_j =
                std::log(Y[j]) +
                theta * (std::log(P[j]) - std::log(tau_ij[i * dim + j]));
            if (x_j > x_max) {
              x_max = x_j;
            }
            x[j] = x_j;
          }
          double sum = 0;
          for (auto j = 0; j < dim; j++) {
            sum += std::exp(x[j] - x_max);
          }
          double Pi_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
          double diff = std::abs(Pi[i] - Pi_new) / Pi[i];
          Pi_temp[i] = Pi_new;
          if (diff > max_diff_inner) {
            max_diff_inner = diff;
          }
        }
        for (auto i = 0; i < dim; i++) {
          Pi[i] = Pi_temp[i];
        }
        std::cout << "# " << cycles_inner << ": " << max_diff_inner
                  << std::endl;
        cycles_inner++;
      }
      for (auto i = 0; i < dim; i++) {
        Y[i] = std::pow(A[i] * L[i] / Pi[i], theta / (theta + 1));
      }
      max_diff_outer = 0;
      for (auto j = 0; j < dim; j++) {
        double x_max = -std::numeric_limits<double>::infinity();
        for (auto i = 0; i < dim; i++) {
          double x_i = std::log(Y[i]) + theta * (std::log(Pi[i]) -
                                                 std::log(tau_ij[i * dim + j]));
          if (x_i > x_max) {
            x_max = x_i;
          }
          x[i] = x_i;
        }
        double sum = 0;
        for (auto i = 0; i < dim; i++) {
          sum += std::exp(x[i] - x_max);
        }
        double P_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
        double diff = std::abs(P[j] - P_new) / P[j];
        if (diff > max_diff_outer) {
          max_diff_outer = diff;
        }
      }
      for (auto i = 0; i < dim; i++) {
        double x_max = -std::numeric_limits<double>::infinity();
        for (auto j = 0; j < dim; j++) {
          double x_j = std::log(Y[j]) +
                       theta * (std::log(P[j]) - std::log(tau_ij[i * dim + j]));
          if (x_j > x_max) {
            x_max = x_j;
          }
          x[j] = x_j;
        }
        double sum = 0;
        for (auto j = 0; j < dim; j++) {
          sum += std::exp(x[j] - x_max);
        }
        double Pi_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
        double diff = std::abs(Pi[i] - Pi_new) / Pi[i];
        if (diff > max_diff_outer) {
          max_diff_outer = diff;
        }
      }
      std::cout << "~~ Outer Loop # " << cycles_outer << ": " << max_diff_outer
                << std::endl;
      cycles_outer++;
    }
  }
  if (solution == 3) {
    while (max_diff_outer > diff_limit) {
      cycles_inner = 0;
      max_diff_inner = 1;
      for (auto i = 0; i < dim; i++) {
        double A_L = A[i] * L[i];
        double A_L_Pi = A_L / Pi[i];
        double exponent = theta / (theta + 1);
        Y[i] = std::pow(A_L_Pi, exponent);
      }
      while (max_diff_inner > diff_limit) {
        max_diff_inner = 0;
        for (auto j = 0; j < dim; j++) {
          double x_max = -std::numeric_limits<double>::infinity();
          for (auto i = 0; i < dim; i++) {
            double x_i =
                theta * (std::log(A[i]) + std::log(L[i]) - std::log(Y[i]) -
                         std::log(tau_ij[i * dim + j]));
            if (x_i > x_max) {
              x_max = x_i;
            }
            x[i] = x_i;
          }
          double sum = 0;
          for (auto i = 0; i < dim; i++) {
            sum += std::exp(x[i] - x_max);
          }
          double P_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
          double diff = std::abs(P[j] - P_new);
          P_temp[j] = P_new;
          if (diff > max_diff_inner) {
            max_diff_inner = diff;
          }
        }
        for (auto j = 0; j < dim; j++) {
          P[j] = P_temp[j];
          Pi[j] = P_temp[j];
        }
        std::cout << "# " << cycles_inner << ": " << max_diff_inner
                  << std::endl;
        cycles_inner++;
      }
      for (auto i = 0; i < dim; i++) {
        Y[i] = std::pow(A[i] * L[i] / Pi[i], theta / (theta + 1));
      }
      max_diff_outer = 0;
      for (auto j = 0; j < dim; j++) {
        double x_max = -std::numeric_limits<double>::infinity();
        for (auto i = 0; i < dim; i++) {
          double x_i = theta * (std::log(A[i]) + std::log(L[i]) -
                                std::log(Y[i]) - std::log(tau_ij[i * dim + j]));
          if (x_i > x_max) {
            x_max = x_i;
          }
          x[i] = x_i;
        }
        double sum = 0;
        for (auto i = 0; i < dim; i++) {
          sum += std::exp(x[i] - x_max);
        }
        double P_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
        double diff = std::abs(P[j] - P_new);
        if (diff > max_diff_outer) {
          max_diff_outer = diff;
        }
      }
      std::cout << "~~ Outer Loop # " << cycles_outer << ": " << max_diff_outer
                << std::endl;
      cycles_outer++;
    }
  }
  if (solution == 4) {
    while (max_diff_outer > diff_limit) {
      cycles_inner = 0;
      max_diff_inner = 1;
      while (max_diff_inner > diff_limit) {
        max_diff_inner = 0;
        for (auto j = 0; j < dim; j++) {
          double x_max = -std::numeric_limits<double>::infinity();
          for (auto i = 0; i < dim; i++) {
            double x_i =
                (theta / (theta + 1)) * (std::log(A[i]) + std::log(L[i])) +
                (theta * theta / (theta + 1)) * std::log(Pi[i]) -
                theta * std::log(tau_ij[i * dim + j]);
            if (x_i > x_max) {
              x_max = x_i;
            }
            x[i] = x_i;
          }
          double sum = 0;
          for (auto i = 0; i < dim; i++) {
            sum += std::exp(x[i] - x_max);
          }
          double P_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
          double diff = std::abs(P[j] - P_new);
          P_temp[j] = P_new;
          if (diff > max_diff_inner) {
            max_diff_inner = diff;
          }
        }
        for (auto j = 0; j < dim; j++) {
          P[j] = P_temp[j];
          Pi[j] = P_temp[j];
        }
        std::cout << "# " << cycles_inner << ": " << max_diff_inner
                  << std::endl;
        cycles_inner++;
      }
      for (auto i = 0; i < dim; i++) {
        Y[i] = std::pow(A[i] * L[i] / Pi[i], theta / (theta + 1));
      }
      max_diff_outer = 0;
      for (auto j = 0; j < dim; j++) {
        double x_max = -std::numeric_limits<double>::infinity();
        for (auto i = 0; i < dim; i++) {
          double x_i =
              (theta / (theta + 1)) * (std::log(A[i]) + std::log(L[i])) +
              (theta * theta / (theta + 1)) * std::log(Pi[i]) -
              theta * std::log(tau_ij[i * dim + j]);
          if (x_i > x_max) {
            x_max = x_i;
          }
          x[i] = x_i;
        }
        double sum = 0;
        for (auto i = 0; i < dim; i++) {
          sum += std::exp(x[i] - x_max);
        }
        double P_new = std::exp((-1 / theta) * (x_max + std::log(sum)));
        double diff = std::abs(P[j] - P_new);
        if (diff > max_diff_outer) {
          max_diff_outer = diff;
        }
      }
      std::cout << "~~ Outer Loop # " << cycles_outer << ": " << max_diff_outer
                << std::endl;
      cycles_outer++;
    }
  }
  std::cout << "***** Number of cycles required: "
            << cycles_outer * cycles_inner << std::endl;

  if (debug_level > 0) {
    std::cout << "# Pi and P arrays calculated #" << std::endl;
    if (debug_level > 1) {
      std::cout << "*** Pi_i ***" << std::endl;
      print_array(dim, Pi);
      std::cout << "*** P_j ***" << std::endl;
      print_array(dim, P);
    }
  }

  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      X_ij[i * dim + j] = std::pow(tau_ij[i * dim + j], -1 * theta) * Y[i] *
                          std::pow(Pi[i], theta) * Y[j] * std::pow(P[j], theta);
    }
  }

  if (debug_level > 0) {
    std::cout << "# X matrix calculated #" << std::endl;
    if (debug_level > 1) {
      std::cout << "*** X_ij ***" << std::endl;
      print_matrix(dim, X_ij);
    }
  }

  // Write Xi_ij = t_kl^{\theta} * P_k^{-\theta} * \Pi_l^{-\theta} to array
  for (auto k = 0; k < dim; k++) {
    for (auto l = 0; l < dim; l++) {
      if (k == l && allow_internal_trade) {
        Xi_ij[k * dim + l] =
            std::pow(P[k], -1 * theta) * std::pow(Pi[l], -1 * theta);
      } else if (t_ij[k * dim + l] == 0) {
        Xi_ij[k * dim + l] = 0;
      } else {
        Xi_ij[k * dim + l] = std::pow(t_ij[k * dim + l], -1 * theta) *
                             std::pow(P[k], -1 * theta) *
                             std::pow(Pi[l], -1 * theta);
      }
    }
  }

  if (debug_level > 0) {
    std::cout << "# Xi matrix calculated #" << std::endl;
    if (debug_level > 1) {
      std::cout << "*** Xi_ij ***" << std::endl;
      print_matrix(dim, Xi_ij);
    }
    std::cout << "#### Cycle Simulated ####" << std::endl;
  }

  delete[] P_temp;
  delete[] Pi_temp;
  delete[] Y_temp;
  delete[] B_ij;
  delete[] Z;
  delete[] Z_temp;
  delete[] x;
  return 0;
}

model_ptrs *build_model(int *graph, int width, int height, int node_count,
                        cost_modifiers costs, bool debug) {
  size_t size = node_count * (5 * node_count + 7);
  double *model_data = new double[size];
  model_base *model_start = new model_base;
  model *model_current = new model;
  model_ptrs *models = new model_ptrs;

  model_start->edge_weights = model_data + 0;
  model_start->A = model_data + node_count * node_count;
  model_start->L = model_data + node_count * (node_count + 1);

  model_current->edge_weights = model_data + node_count * (node_count + 2);
  model_current->tau = model_data + node_count * (2 * node_count + 2);
  model_current->X = model_data + node_count * (3 * node_count + 2);
  model_current->Xi = model_data + node_count * (4 * node_count + 2);
  model_current->A = model_data + node_count * (5 * node_count + 2);
  model_current->L = model_data + node_count * (5 * node_count + 3);
  model_current->Y = model_data + node_count * (5 * node_count + 4);
  model_current->Pi = model_data + node_count * (5 * node_count + 5);
  model_current->P = model_data + node_count * (5 * node_count + 6);

  models->addr_start = model_start;
  models->addr_current = model_current;

  bool crawled[width * height];
  int error = crawl_graph(graph, width, height, node_count, crawled,
                          model_start->edge_weights, costs, debug);

  return models;
}

model_ptrs *load_model(const char filename_edges[], const char filename_A[],
                       const char filename_L[], const char filename_B[],
                       int node_count, int debug_level) {
  size_t size = node_count * (5 * node_count + 9);
  double *model_data = new double[size];
  for (auto i = 0; i < size; i++) {
    model_data[i] = 0;
  }
  model_base *model_start = new model_base;
  model *model_current = new model;
  model_ptrs *models = new model_ptrs;

  model_start->edge_weights = model_data + 0;
  model_start->A = model_data + node_count * node_count;
  model_start->L = model_data + node_count * (node_count + 1);
  model_start->B = model_data + node_count * (node_count + 2);

  model_current->edge_weights = model_data + node_count * (node_count + 3);
  model_current->tau = model_data + node_count * (2 * node_count + 3);
  model_current->X = model_data + node_count * (3 * node_count + 3);
  model_current->Xi = model_data + node_count * (4 * node_count + 3);
  model_current->A = model_data + node_count * (5 * node_count + 3);
  model_current->L = model_data + node_count * (5 * node_count + 4);
  model_current->Y = model_data + node_count * (5 * node_count + 5);
  model_current->Pi = model_data + node_count * (5 * node_count + 6);
  model_current->P = model_data + node_count * (5 * node_count + 7);
  model_current->B = model_data + node_count * (5 * node_count + 8);

  models->addr_start = model_start;
  models->addr_current = model_current;

  std::string line;
  std::string delim = ",";
  std::string entry;
  std::ifstream model_file(filename_edges);
  size_t pos;
  size_t indx;
  int i;
  int j;
  double distance;
  double pop;
  double knowledge;
  double fertility;
  double min_distance = INFINITY;
  double max_distance = 0;
  double min_pop = INFINITY;
  double max_pop = 0;
  double min_knowledge = INFINITY;
  double max_knowledge = 0;
  long double distance_total = 0;

  double *t_temp = new double[node_count * node_count];
  bool ignore_zero = true;

  if (model_file.is_open()) {
    for (int x = 0; x < node_count * node_count; x++) {
      model_start->edge_weights[x] = 0;
      t_temp[x] = 0;
    }
    // lines < 18497
    // std::getline(model_file, line);
    // std::cout << line << std::endl;
    for (int lines = 0; lines < node_count * 8; lines++) {
      pos = 0;
      std::getline(model_file, line);

      pos = line.find(delim);
      entry = line.substr(0, pos);
      i = std::stoi(entry, &indx);
      line.erase(0, pos + delim.length());

      pos = line.find(delim);
      entry = line.substr(0, pos);
      j = std::stoi(entry, &indx);
      line.erase(0, pos + delim.length());

      pos = line.find(delim);
      entry = line.substr(0, pos);
      distance = stod(entry, &indx);

      if (distance < min_distance && distance > 0) {
        min_distance = distance;
      }
      if (distance > max_distance && distance < INFINITY) {
        max_distance = distance;
      }
      if (j >= 0 && (distance > 0 || ignore_zero == true)) {
        t_temp[i * node_count + j] = std::max(1.1 + distance / 5.0, 2.0);
      }
    }
  }

  bool average = true;
  for (int i = 0; i < node_count; i++) {
    for (int j = 0; j < node_count; j++) {
      if (average) {
        if (t_temp[i * node_count + j] > 0 || t_temp[j * node_count + i] > 0) {

          model_start->edge_weights[i * node_count + j] = std::max(
              (t_temp[i * node_count + j] + t_temp[j * node_count + i]) / 2.0,
              1.3);
        } else {

          model_start->edge_weights[i * node_count + j] = 0;
        }
      } else {

        model_start->edge_weights[i * node_count + j] =
            t_temp[i * node_count + j];
      }
    }
  }

  std::ifstream A_file(filename_A);
  if (A_file.is_open()) {
    for (int lines = 0; lines < node_count; lines++) {
      pos = 0;
      std::getline(A_file, line);

      pos = line.find(delim);
      entry = line.substr(0, pos);
      knowledge = std::stod(entry, &indx);
      if (knowledge < min_knowledge) {
        min_knowledge = knowledge;
      }
      if (knowledge > max_knowledge) {
        max_knowledge = knowledge;
      }
      model_start->A[lines] = std::max(knowledge, 1.0);
      // model_start->A[lines] = std::log(1 + knowledge) + 1;
    }
  }
  std::ifstream L_file(filename_L);
  if (L_file.is_open()) {
    for (int lines = 0; lines < node_count; lines++) {
      pos = 0;
      std::getline(L_file, line);

      pos = line.find(delim);
      entry = line.substr(0, pos);
      pop = std::stod(entry, &indx);
      if (pop < min_pop) {
        min_pop = pop;
      }
      if (pop > max_pop) {
        max_pop = pop;
      }
      model_start->L[lines] = pop;
      // model_start->L[lines] = std::log(1 + knowledge) + 1;
    }
  }
  std::ifstream B_file(filename_B);
  if (B_file.is_open()) {
    for (int lines = 0; lines < node_count; lines++) {
      pos = 0;
      std::getline(B_file, line);

      pos = line.find(delim);
      entry = line.substr(0, pos);
      fertility = std::stod(entry, &indx);
      model_start->B[lines] = std::max(fertility, 1.0);
    }
  }

  delete[] t_temp;

  return models;
}

void test_model(int node_count, model *current_model, model_coefficients coeff,
                int debug_level) {
  int iterations = 10;

  double *tau_ij_cu = new double[node_count * node_count];
  double *tau_ij_inv = new double[node_count * node_count];
  double *t_ij = current_model->edge_weights;
  generate_tau_ij(node_count, t_ij, tau_ij_inv, coeff.theta, debug_level);

  // std::cout << "tau_ij_inv::" << std::endl;
  // print_matrix(node_count, tau_ij_inv);

  while (iterations < 101) {
    generate_tau_ij_cublas(node_count, t_ij, tau_ij_cu, coeff.theta, iterations,
                           debug_level);
    // std::cout << "tau :" << std::endl;
    // print_matrix(node_count, tau_ij_cu);

    double max_diff = 0;
    for (int i = 0; i < node_count; i++) {
      for (int j = 0; j < node_count; j++) {
        int index = i * node_count + j;
        double diff =
            std::abs(tau_ij_cu[index] - tau_ij_inv[index]) / tau_ij_inv[index];
        // std::cout << diff << std::endl;
        if (diff > max_diff) {
          std::cout << diff << " -> " << tau_ij_cu[index] << " -> "
                    << tau_ij_inv[index] << std::endl;
          max_diff = diff;
        }
      }
    }

    double max_diff_near = 0;
    int dist = 2;
    for (int i = 0; i < node_count; i++) {
      for (int di = -1 * dist; di <= dist; di++) {
        if (di != 0) {
          int index = i * node_count + (i + di);
          double diff = std::abs(tau_ij_cu[index] - tau_ij_inv[index]);
          // std::cout << diff << std::endl;
          if (diff > max_diff_near && diff < INFINITY) {
            max_diff_near = diff;
          }
        }
      }
    }

    std::cout << "Iter: #" << iterations << std::endl;
    std::cout << "Max Diff: " << max_diff << std::endl;
    std::cout << "Max Diff Near: " << max_diff_near << std::endl;
    iterations += 10;
  }
}

int update_model(int t, int node_count, model *current_model,
                 model_coefficients coeff, bool update_A, bool update_t,
                 bool update_L, int price_iterations, int debug_level) {

  if (debug_level > 0) {
    std::cout << "##### Updating model for t = " << t << " ######" << std::endl;
    if (update_A) {
      std::cout << "#### Updating A array ####" << std::endl;
    }
    if (update_t) {
      std::cout << "#### Updating t matrix ####" << std::endl;
    }
    if (update_L) {
      std::cout << "#### Updating L array ####" << std::endl;
    }
  }

  double *Xi_ij_t_min_1 = new double[node_count * node_count];
  std::memcpy(Xi_ij_t_min_1, current_model->Xi,
              node_count * node_count * sizeof(double));

  // run a cycle
  run_cycle(t, node_count, current_model->edge_weights, current_model->tau,
            current_model->A, current_model->L, current_model->Y,
            current_model->Pi, current_model->P, current_model->X,
            current_model->Xi, coeff.theta, update_t, price_iterations,
            debug_level);

  double *A = current_model->A;
  double *L = current_model->L;
  double *B = current_model->B;
  double *Y = current_model->Y;
  double *P = current_model->P;
  double *t_ij = current_model->edge_weights;
  double *Xi_ij = current_model->Xi;

  bool normalized = false;
  bool log_A = true;
  bool translate_A = true;

  if (t > 0) {

    if (update_A) {

      double *A_temp = new double[node_count];

      double A_min = std::numeric_limits<double>::infinity();
      if (translate_A) {
        for (int i = 0; i < node_count; i++) {
          if (A[i] < A_min) {
            A_min = A[i];
          }
        }
      }

      for (int i = 0; i < node_count; i++) {
        if (A[i] == 0) {
          continue;
        }
        double Xi_sum = 0;
        if (normalized) {
          for (int l = 0; l < node_count; l++) {
            Xi_sum += Xi_ij[l * node_count + i];
          }
        }
        double sum = 0;
        for (int j = 0; j < node_count; j++) {
          double A_transformed = A[j];
          if (translate_A) {
            A_transformed -= A_min;
          }
          double Xi_transformed = Xi_ij[j * node_count + i];
          if (normalized) {
            Xi_transformed = std::pow(Xi_transformed / Xi_sum, coeff.sigma);
          }
          sum += Xi_transformed * std::pow(A_transformed, coeff.beta);
        }
        if (log_A) {
          double log_A_i = std::log(A[i]);
          log_A_i += coeff.eta * sum / A[i];
          A_temp[i] = std::exp(log_A_i);
        } else {
          A_temp[i] = A[i] + coeff.eta * sum;
        }
      }

      for (int i = 0; i < node_count; i++) {
        A[i] = A_temp[i];
      }
      delete[] A_temp;

      if (debug_level > 1) {
        print_array(node_count, A);
      }
    }

    if (update_t) {
      if (t > 0)
        // T = 1
        for (int i = 0; i < node_count; i++) {
          for (int j = 0; j < node_count; j++) {
            if (t_ij[i * node_count + j] > 0) {
              double log_t_ij = std::log(t_ij[i * node_count + j]);
              log_t_ij += -1 * coeff.chi *
                          std::log(Xi_ij[i * node_count + j] /
                                   Xi_ij_t_min_1[i * node_count + j]);
              t_ij[i * node_count + j] = std::max(std::exp(log_t_ij), 2.0);
            }
          }
        }
    }

    delete[] Xi_ij_t_min_1;

    if (update_L) {
      for (int i = 0; i < node_count; i++) {
        double log_L_i = std::log(L[i]);
        double y = (std::pow(B[i], coeff.xi) *
                    std::pow(Y[i] / (P[i] * L[i]), 1 - coeff.xi)) /
                   100000.0;
        double delta_log_L =
            coeff.lambda * (coeff.f * (1 - std::exp(-1 * coeff.a * y)) -
                            (std::pow(y, -1 * coeff.b) - coeff.d));
        log_L_i += delta_log_L;
        L[i] = std::exp(log_L_i);
      }
    }
  }

  if (debug_level > 0) {
    std::cout << "### Model Updated ###" << std::endl;
  }

  return 0;
}

void reset_model(model_ptrs *models, int node_count) {

  std::memcpy(models->addr_current->edge_weights,
              models->addr_start->edge_weights,
              node_count * node_count * sizeof(double));
  std::memcpy(models->addr_current->A, models->addr_start->A,
              node_count * sizeof(double));
  std::memcpy(models->addr_current->L, models->addr_start->L,
              node_count * sizeof(double));
  std::memcpy(models->addr_current->B, models->addr_start->B,
              node_count * sizeof(double));
  for (auto i = 0; i < node_count; i++) {
    models->addr_current->Y[i] = 0;
    models->addr_current->Pi[i] = 0;
    models->addr_current->P[i] = 0;
  }
  for (auto i = 0; i < node_count * node_count; i++) {
    models->addr_current->X[i] = 0;
    models->addr_current->Xi[i] = 0;
  }
}
