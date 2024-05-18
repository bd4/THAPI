SYCL_CXX ?= $(ONEAPI_ROOT)/compiler/latest/bin/icpx
MKL_ROOT ?= $(ONEAPI_ROOT)/mkl/latest
SYCL_CXX_FLAGS = -fsycl -std=c++17 -O2 -g
MKL_FLAGS = -I$(MKL_ROOT)/include -L$(MKL_ROOT)/lib/intel64 -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

# See
# https://github.com/jinge90/llvm/blob/sycl/sycl/doc/extensions/C-CXX-StandardLibrary/DeviceLibExtensions.rst
# and
# https://github.com/jinge90/llvm/blob/sycl/sycl/test/devicelib/std_complex_math_fp64_test.cpp
#DPCPP_LIBDIR ?= /opt/intel/inteloneapi/compiler/latest/linux/lib
#DPCPP_OBJS = libsycl-complex-fp64.o libsycl-cmath-fp64.o
#LIBS = $(addprefix $(DPCPP_LIBDIR)/,$(DPCPP_OBJS))

BUILD_DIR=build-oneapi

SYCL_SOURCES = $(wildcard src/*.cxx)
EXES = $(addprefix $(BUILD_DIR)/,$(basename $(notdir $(SYCL_SOURCES))))

.PHONY: all
all: $(EXES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/% : src/%.cxx | $(BUILD_DIR)
	@echo "Compiling "$<
	$(SYCL_CXX) $(SYCL_CXX_FLAGS) $(MKL_FLAGS) -o $@ $< $(LIBS)

.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/*
