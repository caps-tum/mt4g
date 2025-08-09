# === Configuration ===
HIPCC         := hipcc
CXXFLAGS      := -std=c++20 -O3 -g
INC_DIRS      := -I$(CURDIR)/include \
                 -isystem $(CURDIR)/external/cxxopts/include \
                 -isystem $(CURDIR)/external/json/single_include \
                 -isystem $(HIP_PATH)/include

SRC_BASE      := src
TARGET        := mt4g
BUILD_DIR     := build
GPU_TARGET_ARCH ?=
d             ?= 0

# === Validate input ===
ifneq (,$(filter clean fetch-cxxopts fetch-json,$(MAKECMDGOALS)))
else
  ifndef GPU_TARGET_ARCH
    $(error Please set GPU_TARGET_ARCH, e.g., make GPU_TARGET_ARCH=sm_70 or gfx90a)
  endif
endif

ifndef HIP_PATH
  $(error HIP_PATH is not set in environment)
endif

# Determine platform
ifeq ($(findstring sm_,$(GPU_TARGET_ARCH)),sm_)
  PLATFORM := nvidia
  INC_DIRS += -I$(CUDA_PATH)/include
else ifeq ($(findstring gfx,$(GPU_TARGET_ARCH)),gfx)
  PLATFORM := amd
  INC_DIRS += -isystem /opt/rocm/include 
else
  ifneq (,$(filter clean fetch-cxxopts fetch-json,$(MAKECMDGOALS)))
    PLATFORM := dummy
  else
    $(error Invalid GPU_TARGET_ARCH '$(GPU_TARGET_ARCH)'. Must start with sm_ or gfx)
  endif
endif

export HIP_PLATFORM := $(PLATFORM)

# === GPUFLAGS setup ===
ifeq ($(PLATFORM),nvidia)
  GPUFLAGS := --gpu-architecture=$(GPU_TARGET_ARCH) -x cu -Wno-deprecated-gpu-targets
else ifeq ($(PLATFORM),amd)
  GPUFLAGS := --offload-arch=$(GPU_TARGET_ARCH) -Wall -Wextra -Wpedantic -Wnull-dereference 
endif

# === Source gathering ===
CPP_SOURCES := $(shell find $(SRC_BASE) -type f -name '*.cpp')
OBJECTS     := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))

# === Default target ===
all: fetch-cxxopts fetch-json $(TARGET)

# === Link ===
ifeq ($(PLATFORM),amd)
  LIB_DIRS := -L/opt/rocm/lib -lrocm_smi64 -lhsa-runtime64 -pthread
else
  LIB_DIRS := -L$(CUDA_PATH)/lib64 -lcudart
endif

$(TARGET): $(OBJECTS)
	@echo "Linking $@ for $(PLATFORM) with arch $(GPU_TARGET_ARCH)"
	$(HIPCC) $(CXXFLAGS) $(OBJECTS) $(LIB_DIRS) -o $@
	@printf "\033[1;32mCompiled successfully ✔\033[0m\n"

# === Compile C++ sources (with debug intermediates if d=1) ===
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(CXXFLAGS) $(INC_DIRS) $(GPUFLAGS) -c $< -o $@
	@if [ "$(d)" = "1" ]; then \
		echo "Dropping all temp-files for $< into dbg/"; \
		mkdir -p dbg; \
		cd dbg && \
		if [ "$(PLATFORM)" = "amd" ]; then \
		    hipcc $(CXXFLAGS) $(INC_DIRS) $(GPUFLAGS) -x hip --save-temps -c ../$< -o /dev/null 2>/dev/null; \
		else \
		    hipcc $(CXXFLAGS) $(INC_DIRS) $(GPUFLAGS) -x cu --save-temps -c ../$< -o /dev/null 2>/dev/null; \
		fi; \
	fi

# === Fetch cxxopts if missing ===
fetch-cxxopts:
	@if [ ! -d external/cxxopts ]; then \
		git clone --depth=1 --branch v3.2.0 https://github.com/jarro2783/cxxopts.git external/cxxopts; \
	else \
		echo "cxxopts already present"; \
	fi

# === Fetch nlohmann/json if missing ===
fetch-json:
	@if [ ! -d external/json ]; then \
		git clone --depth=1 --branch v3.11.2 https://github.com/nlohmann/json.git external/json; \
	else \
		echo "nlohmann/json already present"; \
	fi

# === Clean ===
clean:
	rm -rf $(BUILD_DIR) $(TARGET) dbg

# === Info ===
info:
	@echo "HIP_PATH=$(HIP_PATH)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "ROCM_PATH=$(ROCM_PATH)"
	@echo "GPU_TARGET_ARCH=$(GPU_TARGET_ARCH)"
	@echo "Platform=$(PLATFORM)"
