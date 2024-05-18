#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <time.h>

//#include "oneapi/mkl/spblas.h"
#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"

#undef NDEBUG

using complex_t = std::complex<double>;
const complex_t h_one = 1.0;

using spindex_t = std::int32_t;

struct problem {
  int nrows;
  int nnz;
  int nrhs;
  spindex_t *row_ptr;
  spindex_t *col_ind;
  complex_t *val;
  complex_t *rhs;
  complex_t *sol1;
  complex_t *sol2;
  complex_t *sol_out;
};

struct problem read_problem(sycl::queue &q) {
  struct problem p;
  size_t n;

  std::ifstream f("sparse_matrix.dat", std::fstream::in);

  f >> p.nrows;
  std::cout << "nrows " << p.nrows << std::endl;
  n = (p.nrows + 1);
  p.row_ptr = sycl::malloc_shared<spindex_t>(n, q);
  for (int i = 0; i < p.nrows + 1; i++) {
    f >> p.row_ptr[i];
  }
  f >> p.nnz;
  std::cout << "nnz " << p.nnz << std::endl;
  n = p.nnz;
  p.col_ind = sycl::malloc_shared<spindex_t>(n, q);
  n = p.nnz;
  p.val = sycl::malloc_shared<complex_t>(n, q);
  for (int i = 0; i < p.nnz; i++) {
    f >> p.col_ind[i];
  }
  for (int i = 0; i < p.nnz; i++) {
    f >> p.val[i];
  }
  f >> p.nrhs;
  std::cout << "nrhs " << p.nrhs << std::endl;
  n = p.nnz * p.nrhs;
  p.rhs = sycl::malloc_shared<complex_t>(n, q);
  p.sol1 = sycl::malloc_shared<complex_t>(n, q);
  p.sol2 = sycl::malloc_shared<complex_t>(n, q);
  p.sol_out = sycl::malloc_shared<complex_t>(n, q);
  for (int i = 0; i < p.nnz * p.nrhs; i++) {
    f >> p.rhs[i];
  }

  return p;
}

struct problem problem_to_device(sycl::queue &q, struct problem p) {
  struct problem d_p;
  size_t n;

  d_p.row_ptr = sycl::malloc_device<spindex_t>(p.nrows + 1, q);
  q.copy(p.row_ptr, d_p.row_ptr, p.nrows + 1);

  d_p.col_ind = sycl::malloc_device<spindex_t>(p.nnz, q);
  q.copy(p.col_ind, d_p.col_ind, p.nnz);

  d_p.val = sycl::malloc_device<complex_t>(p.nnz, q);
  q.copy(p.val, d_p.val, p.nnz);

  n = p.nnz * p.nrhs;
  d_p.rhs = sycl::malloc_device<complex_t>(n, q);
  q.copy(p.rhs, d_p.rhs, n);

  d_p.sol1 = sycl::malloc_device<complex_t>(n, q);
  d_p.sol2 = sycl::malloc_device<complex_t>(n, q);
  d_p.sol_out = sycl::malloc_shared<complex_t>(n, q);

  q.wait();

  return d_p;
}

void solve(sycl::queue &q, struct problem p, bool copy_out = false)
{
  struct timespec start, end;
  double total, elapsed;

  oneapi::mkl::sparse::matrix_handle_t mat_h;
  oneapi::mkl::sparse::init_matrix_handle(&mat_h);
  oneapi::mkl::sparse::set_csr_data(mat_h, p.nrows, p.nrows,
                                    oneapi::mkl::index_base::one, p.row_ptr,
                                    p.col_ind, p.val);

  // oneapi::mkl::sparse::optimize_trsv(q, oneapi::mkl::uplo::lower,
  //                                    oneapi::mkl::transpose::nontrans,
  //                                    oneapi::mkl::diag::unit, mat_h);
  // oneapi::mkl::sparse::optimize_trsv(q, oneapi::mkl::uplo::upper,
  //                                    oneapi::mkl::transpose::nontrans,
  //                                    oneapi::mkl::diag::nonunit, mat_h);

  /* step 1: solve L * Y = B */
  clock_gettime(CLOCK_MONOTONIC, &start);

  auto e_lower = oneapi::mkl::sparse::trsv(
      q, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
      oneapi::mkl::diag::unit, mat_h, p.rhs, p.sol1);
  e_lower.wait();

  /* step 2: solve U * X = Y */
  auto e_upper = oneapi::mkl::sparse::trsv(
      q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
      oneapi::mkl::diag::nonunit, mat_h, p.sol1, p.sol2);
  e_upper.wait();

  if (copy_out) {
    q.copy(p.sol2, p.sol_out, p.nnz * p.nrhs);
  }

  q.wait();
  clock_gettime(CLOCK_MONOTONIC, &end);
  // count++;
  elapsed =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
  // total += elapsed;
  // std::cout << "sparse " << count << " " << elapsed << " " << total / count
  // << std::endl;
  std::cout << "sparse " << elapsed << std::endl;
}

inline auto get_exception_handler() {
  static auto exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const &e) {
        std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                  << e.what() << std::endl;
        abort();
      }
    }
  };
  return exception_handler;
}

int main(int argc, char *argv[]) {
  struct problem p;

  auto q = cl::sycl::queue{get_exception_handler()};
  auto dev = q.get_device();
  std::string type;
  if (dev.is_cpu()) {
    type = "CPU  ";
  } else if (dev.is_gpu()) {
    type = "GPU  ";
  } else if (dev.is_host()) {
    type = "HOST ";
  } else {
    type = "OTHER";
  }
  std::cout << "[" << type << "] "
            << dev.get_info<cl::sycl::info::device::name>() << " {"
            << dev.get_info<cl::sycl::info::device::vendor>() << "}"
            << std::endl;

  p = read_problem(q);

  std::cout << "nrows " << p.nrows << std::endl;
  std::cout << "nnz " << p.nnz << std::endl;
  std::cout << "nrhs " << p.nrhs << std::endl;

  std::cout << "managed warmup run" << std::endl;
  solve(q, p);

  std::cout << "managed memory runs" << std::endl;
  solve(q, p);
  // std::cout << p.rhs[0] << std::endl;
  solve(q, p);
  // std::cout << p.rhs[0] << std::endl;
  solve(q, p);
  // std::cout << p.rhs[0] << std::endl;
  solve(q, p);

  auto d_p = problem_to_device(q, p);

  std::cout << "device warmup run" << std::endl;
  solve(q, d_p, true);
  std::cout << "device memory runs" << std::endl;
  solve(q, d_p, true);
  solve(q, d_p, true);
  solve(q, d_p, true);
  solve(q, d_p, true);

  return 0;
}
