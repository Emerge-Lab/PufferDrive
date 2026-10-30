# Demo-only renderer. Built on demand, never by setup.py, so the training extension stays free of
# raylib (which is what #593 removed). Build with:  make -C pufferlib/ocean/drive/render
ROOT := ../../../..
PYTHON ?= $(ROOT)/.venv/bin/python

EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PY_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
TARGET := $(ROOT)/pufferlib/ocean/drive/render/drive_render$(EXT_SUFFIX)

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
RAYLIB_DIR := $(ROOT)/extern/raylib-5.5_macos
RAYLIB_LIB := $(RAYLIB_DIR)/lib/libraylib.a
RAYLIB_INCLUDE := $(RAYLIB_DIR)/include
PLATFORM_LDFLAGS := -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL \
	-undefined dynamic_lookup
else
# Linux: expects a system raylib (apt install libraylib-dev, or a source build on the PATH).
RAYLIB_LIB := -lraylib
RAYLIB_INCLUDE := /usr/local/include
PLATFORM_LDFLAGS := -lGL -lm -lpthread -ldl -lrt -lX11
endif

CFLAGS := -O2 -std=c99 -D_GNU_SOURCE -fPIC -Wall -Wextra \
	-Wno-unused-function -Wno-unused-parameter \
	-I$(ROOT) -I$(RAYLIB_INCLUDE) -I$(PY_INCLUDE)

SOURCES := render_binding.c
HEADERS := render.h $(wildcard $(ROOT)/pufferlib/ocean/drive/*.h)

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -shared $(SOURCES) $(RAYLIB_LIB) $(PLATFORM_LDFLAGS) -lm -o $@
	@echo "built $@"

clean:
	rm -f $(TARGET)
