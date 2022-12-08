#--------------------------------------------------------------------
# Cori
#--------------------------------------------------------------------
builddir = build

CXX=/global/homes/m/millerk1/kokkos/bin/nvcc_wrapper #Nvidia compiler
# Ripper
# CXXFLAGS = -I/usr/local/cuda-11.5/include
# KOKKOS_ARCH = Maxwell52
# Perlmutter
CXXFLAGS =
KOKKOS_ARCH = Ampere80

# Production
CXXFLAGS += -O3 -use_fast_math -extra-device-vectorization
# Debug
# CXXFLAGS += -O0 -G -g


# Kokkos flags
KOKKOS_PATH = /global/homes/m/millerk1/kokkos
CUDA_PATH = $(CUDA_HOME)
KOKKOS_DEVICES=Cuda
KOKKOS_CXX_STANDARD = c++14


# SRC = cuda-code.cu
SRC = kokkos-code.cu

OBJ := $(patsubst %.cu,$(builddir)/%.o,$(SRC))

default: cuda-code.e
include $(KOKKOS_PATH)/Makefile.kokkos

$(builddir)/%.o: %.cu $(KOKKOS_CPP_DEPENDS)
	@$(CXX) $(CXXFLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -I. -c $< -o $@

cuda-code.e: $(builddir) $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(CXX) -o $@ $(OBJ) $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

$(builddir):
	@mkdir -p $@

clean:
	@rm -rf $(builddir)

clean-all: kokkos-clean clean