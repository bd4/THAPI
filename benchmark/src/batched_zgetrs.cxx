#include <complex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <string>
#include <typeinfo>
#include <time.h>

#define NRUNS 10

#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"

using index_t = std::int64_t;

template <typename T>
inline void read_carray(std::ifstream& f, int n, std::complex<T>* Adata) {
    for (int i=0; i < n; i++) {
        //std::cout << i << " " << std::endl;
        f >> Adata[i];
    }
}

inline void read_iarray(std::ifstream& f, int n, int *data) {
    for (int i=0; i < n; i++) {
        f >> data[i];
    }
}

template <typename T>
void test(cl::sycl::queue q, index_t n=140, index_t nrhs=1, index_t batch_size=384) {

    index_t lda, ldb;
    int Aptr_count, Bptr_count, Adata_count, Bdata_count, piv_count;
    std::complex<T> **h_Aptr, **d_Aptr, **h_Bptr, **d_Bptr;
    std::complex<T> *h_Adata, *d_Adata, *h_Bdata, *d_Bdata;
    index_t **h_piv_ptr, **d_piv_ptr;
    index_t *h_piv, *d_piv;

    using CT = std::complex<T>;

#ifdef READ_INPUT
    std::ifstream f("zgetrs.txt", std::ifstream::in);

    f >> n;
    f >> nrhs;
    f >> lda;
    f >> ldb;
    f >> batch_size;

#else
    lda = n;
    ldb = n;
#endif

    std::cout << "n    = " << n    << std::endl;
    std::cout << "nrhs = " << nrhs << std::endl;
    std::cout << "lda  = " << lda  << std::endl;
    std::cout << "ldb  = " << ldb  << std::endl;
    std::cout << "batch_size = " << batch_size << std::endl;

    Aptr_count = Bptr_count = batch_size;
    Adata_count = n * n * batch_size;
    Bdata_count = n * nrhs * batch_size;
    piv_count = n * batch_size;

    std::cout << "Aptr count  = " << Aptr_count  << std::endl;
    std::cout << "Bptr count  = " << Bptr_count  << std::endl;
    std::cout << "Adata count = " << Adata_count << std::endl;
    std::cout << "Bdata count = " << Bdata_count << std::endl;
    std::cout << "piv_count   = " << piv_count << std::endl;

    h_Aptr = cl::sycl::malloc_host<CT*>(Aptr_count, q);
    h_Bptr = cl::sycl::malloc_host<CT*>(Bptr_count, q);
    d_Aptr = cl::sycl::malloc_device<CT*>(Aptr_count, q);
    d_Bptr = cl::sycl::malloc_device<CT*>(Bptr_count, q);

    h_Adata = cl::sycl::malloc_host<CT>(Adata_count, q);
    h_Bdata = cl::sycl::malloc_host<CT>(Bdata_count, q);
    d_Adata = cl::sycl::malloc_device<CT>(Adata_count, q);
    d_Bdata = cl::sycl::malloc_device<CT>(Bdata_count, q);

    h_piv_ptr = cl::sycl::malloc_host<index_t*>(batch_size, q);
    d_piv_ptr = cl::sycl::malloc_device<index_t*>(batch_size, q);
    h_piv = cl::sycl::malloc_host<index_t>(piv_count, q);
    d_piv = cl::sycl::malloc_device<index_t>(piv_count, q);

    std::cout << "malloc done" << std::endl;

#ifdef READ_INPUT
    read_carray(f, n*n*batch_size, h_Adata);
    read_carray(f, n*nrhs*batch_size, h_Bdata);
    read_iarray(f, n*batch_size, h_piv);
    f.close();
#else
    std::ostringstream ss;
    std::string run_label;

    ss << "n=" << n << ";nrhs=" << nrhs << ";batches=" << batch_size
       << ";t=" << typeid(T).name();
    run_label = ss.str();
    
    q.fill<CT>(h_Adata, CT(0.0, 0.0), Adata_count);
    q.fill<CT>(h_Bdata, CT(0.0, 0.0), Bdata_count);
    q.wait();

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            h_Adata[b*n*n + i*n + i] = CT(1.0, 0.0);
            for (int j = 0; j < nrhs; j++) {
                h_Bdata[b*n*nrhs + i*nrhs + j] = CT(i / (j+1) * b, i * j / (b+1));
            }
            h_piv[b*n + i] = i+1;
        }
    }
#endif

    for (int i = 0; i < batch_size; i++) {
        h_Aptr[i] = d_Adata + (n*n*i);
        h_Bptr[i] = d_Bdata + (n*nrhs*i);
        h_piv_ptr[i] = d_piv + (i*n);
    }

    std::cout << "read/init done" << std::endl;

    q.copy(h_Aptr, d_Aptr, Aptr_count);
    q.copy(h_Adata, d_Adata, Adata_count);
    q.copy(h_Bptr, d_Bptr, Bptr_count);
    q.copy(h_Bdata, d_Bdata, Bdata_count);
    q.copy(h_piv_ptr, d_piv_ptr, batch_size);
    q.copy(h_piv, d_piv, piv_count);
    q.wait();

    std::cout << "memcpy done" << std::endl;

    struct timespec start, end;
    double elapsed, total = 0.0;
    double total_strided = 0.0;
    int *info, info_sum;

    auto trans_op = oneapi::mkl::transpose::nontrans;
    auto scratch_count = oneapi::mkl::lapack::getrs_batch_scratchpad_size<CT>(
          q, &trans_op, &n, &nrhs, &lda, &ldb, 1, &batch_size);
    auto scratch = cl::sycl::malloc_device<CT>(scratch_count, q);

    auto scratch_count2 = oneapi::mkl::lapack::getrs_batch_scratchpad_size<CT>(
          q, trans_op, n, nrhs, lda, n*n, n, ldb, n*nrhs, batch_size);
    auto scratch2 = cl::sycl::malloc_device<CT>(scratch_count2, q);


    for (int i=0; i<NRUNS; i++) {
        // std::cout << "run [" << i << "]: start" << std::endl;
        clock_gettime(CLOCK_MONOTONIC, &start);

        auto e = oneapi::mkl::lapack::getrs_batch(
          q, &trans_op, &n, &nrhs, h_Aptr, &lda, h_piv_ptr, h_Bptr, &ldb,
          1, &batch_size, scratch, scratch_count);
        e.wait();

        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
        if (i > 0)
            total += elapsed;
        std::cout << "run group  [" << i << "]: " << elapsed << std::endl;

        clock_gettime(CLOCK_MONOTONIC, &start);

        try {
        auto e2 = oneapi::mkl::lapack::getrs_batch(
          q, trans_op, n, nrhs, d_Adata, lda, n*n, d_piv, n, d_Bdata, ldb,
          n*nrhs, batch_size, scratch2, scratch_count2);
        e2.wait();
        } catch(oneapi::mkl::lapack::invalid_argument const &e) {
            std::cerr << e.what() << " arg #: "
                      << e.info() << std::endl;
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
        if (i > 0)
            total_strided += elapsed;
        std::cout << "run stride [" << i << "]: " << elapsed << std::endl;
    }

    std::cout << "zgetrs done" << std::endl;
    std::cout << run_label << " avg group  " << total / (NRUNS-1)         << std::endl;
    std::cout << run_label << " avg stride " << total_strided / (NRUNS-1) << std::endl;

#ifndef READ_INPUT
    // check result
    q.copy(d_Bdata, h_Bdata, Bdata_count);
    q.wait();
    bool ok = true;
    CT err = CT(0.0, 0.0);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                err = h_Bdata[b*n*nrhs + i*nrhs + j] - CT(i / (j+1) * b, i * j / (b+1));
                if (std::abs(err) > 0.0) {
                    std::cout << "err of " << err
                              << " at [" << b << ", " << i
                              <<", " << j << "]" << std::endl;
                    ok = false;
                    break;
                }
            }
            if (!ok)
                break;
        }
        if (!ok)
            break;
    }
#endif

    cl::sycl::free(scratch, q);

    cl::sycl::free(h_Aptr, q);
    cl::sycl::free(h_Bptr, q);
    cl::sycl::free(d_Aptr, q);
    cl::sycl::free(d_Bptr, q);

    cl::sycl::free(h_Adata, q);
    cl::sycl::free(h_Bdata, q);
    cl::sycl::free(d_Adata, q);
    cl::sycl::free(d_Bdata, q);

    cl::sycl::free(h_piv_ptr, q);
    cl::sycl::free(d_piv_ptr, q);
    cl::sycl::free(h_piv, q);
    cl::sycl::free(d_piv, q);

    std::cout << "free done" << std::endl;
}

inline auto get_exception_handler()
{
  static auto exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const& e) {
        std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                  << e.what() << std::endl;
        abort();
      }
    }
  };
  return exception_handler;
}

int main(int argc, char **argv) {
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
              << dev.get_info<cl::sycl::info::device::name>()
              << " {" << dev.get_info<cl::sycl::info::device::vendor>() << "}"
              << std::endl;

    index_t n = 140;
    index_t nrhs = 1;
    index_t batch_size = 384;

    if (argc > 1) {
        n = std::stoi(argv[1]);
    }
    if (argc > 2) {
        nrhs = std::stoi(argv[2]);
    }
    if (argc > 2) {
        batch_size = std::stoi(argv[3]);
    }

    std::cout << "==== float  ====" << std::endl;
    test<float>(q, n, nrhs, batch_size);
    std::cout << "==== double  ====" << std::endl;
    test<double>(q, n, nrhs, batch_size);
}
