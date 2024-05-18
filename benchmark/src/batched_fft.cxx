#include <sycl/sycl.hpp>
#include <complex>
#include <ctime>
#include <iostream>
#include <oneapi/mkl.hpp>

constexpr double PI = 3.141592653589793;

template <typename T>
inline void _expect_near_array(const char* file, int line, const char* xname,
                               T* x, const char* yname, T* y, int n,
                               double max_err = -1.0) {
    bool equal = true;
    int i;
    double err;

    if (max_err == -1.0) {
        max_err = 1e-10;
    }

    for (i = 0; i < n; i++) {
        err = std::abs(x[i] - y[i]);
        if (err > max_err) {
            equal = false;
            break;
        }
    }

    if (!equal) {
        // const int max_view = 10;
        std::cerr << "Arrays not close (max " << max_err << ") at " << file
                  << ":" << line << std::endl
                  << " err " << err << " at [" << i << "]" << std::endl
                  << " " << xname << ":" << std::endl
                  << x[i] << std::endl
                  << " " << yname << ":" << std::endl
                  << y[i] << std::endl;
    }
}

#define GT_EXPECT_NEAR_ARRAY(x, y, n) \
    _expect_near_array(__FILE__, __LINE__, #x, x, #y, y, n)

#define GT_EXPECT_NEAR_ARRAY_ERR(x, y, n, max_err) \
    _expect_near_array(__FILE__, __LINE__, #x, x, #y, y, n, max_err)

template <typename E, typename Desc>
void test_fft_r2c_1d_many_batches(sycl::queue& q) {
    constexpr int Nx = 48;
    constexpr int Nxc = Nx / 2 + 1;
    constexpr int batch_size = 1024 * 256;
    using T = std::complex<E>;

    E* h_A = sycl::malloc_host<E>(Nx * batch_size, q);
    q.fill(h_A, 0, Nx * batch_size);
    E* d_A = sycl::malloc_device<E>(Nx * batch_size, q);

    E* h_A2 = sycl::malloc_host<E>(Nx * batch_size, q);
    q.fill(h_A2, 0, Nx * batch_size);
    E* d_A2 = sycl::malloc_device<E>(Nx * batch_size, q);

    T* h_B = sycl::malloc_host<T>(Nxc * batch_size, q);
    T* h_B0_expected = sycl::malloc_host<T>(Nxc, q);
    T* d_B = sycl::malloc_device<T>(Nxc * batch_size, q);

    double x, y;

    struct timespec start, end;
    double elapsed;

    std::cout << "init h_A" << std::endl;
    // Set up periodic domain with frequency 4 and 8 for batch 0 and 1
    // m = [sin(2*pi*x) for x in -2:4/Nx:2-4/Nx]
    for (int i = 0; i < Nx; i++) {
        x = -2.0 + 4.0 * i / static_cast<E>(Nx);
        y = sin(2 * PI * x);
        for (int b = 0; b < batch_size; b++) {
            h_A[i + b * Nx] = y;
        }
    }

    std::cout << "copy h_A -> d_A" << std::endl;
    q.copy(h_A, d_A, Nx * batch_size).wait();

    std::int64_t rstrides[2] = {0, 1};
    std::int64_t cstrides[2] = {0, 1};

    std::cout << "init plan" << std::endl;
    Desc plan(Nx);
    plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                   batch_size);
    plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, rstrides);
    plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, cstrides);
    plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, Nx);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, Nxc);
    plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                   DFTI_NOT_INPLACE);
    plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                   DFTI_COMPLEX_COMPLEX);
    plan.commit(q);

    std::cout << "compute forward fft" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);
    auto e =
        oneapi::mkl::dft::compute_forward(plan, d_A, reinterpret_cast<E*>(d_B));
    e.wait();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
    std::cout << "fft time(s): " << elapsed << std::endl;

    std::cout << "copy fft result to host" << std::endl;
    q.copy(d_B, h_B, Nxc * batch_size).wait();

    // NB: allow greater error than for other tests
    double max_err = 0.0005;

    std::cout << "init expected result" << std::endl;
    // Expect denormalized -0.5i at positions 4 for all batches
    for (int i = 0; i < Nxc; i++) {
        if (i == 4) {
            h_B0_expected[i] = T(0, -0.5 * Nx);
        } else {
            h_B0_expected[i] = T(0, 0);
        }
    }

    std::cout << "check result" << std::endl;
    for (int b = 0; b < batch_size; b++) {
        GT_EXPECT_NEAR_ARRAY_ERR(h_B0_expected, h_B + b * Nxc, Nxc, max_err);
    }

    /*
    std::cout << "round trip test" << std::endl;
    // test roundtripping data, with normalization
    plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, cstrides);
    plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, rstrides);
    plan.commit(q);
    e = oneapi::mkl::dft::compute_backward(plan, reinterpret_cast<E*>(d_B),
                                           d_A2);
    e.wait();
    q.copy(d_A2, h_A2, Nx * batch_size);
    for (int i = 0; i < Nx * batch_size; i++) {
        h_A2[i] /= E(Nx);
    }
    GT_EXPECT_NEAR_ARRAY(h_A, h_A2, Nx * batch_size);
    */

    sycl::free(h_A, q);
    sycl::free(h_A2, q);
    sycl::free(d_A, q);
    sycl::free(h_B, q);
    sycl::free(h_B0_expected, q);
    sycl::free(d_B, q);
}

inline auto get_exception_handler() {
    static auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl;
                abort();
            }
        }
    };
    return exception_handler;
}

int main(int argc, char** argv) {
    auto q = sycl::queue{get_exception_handler()};
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
              << dev.get_info<sycl::info::device::name>() << " {"
              << dev.get_info<sycl::info::device::vendor>() << "}"
              << std::endl;

    using Desc =
        oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                     oneapi::mkl::dft::domain::REAL>;
    test_fft_r2c_1d_many_batches<double, Desc>(q);
}
