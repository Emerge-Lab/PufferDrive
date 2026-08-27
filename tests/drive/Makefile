CC ?= gcc
ROOT := ../..
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
RAYLIB_NAME := raylib-5.5_macos
LDLIBS_DRIVE := $(ROOT)/extern/$(RAYLIB_NAME)/lib/libraylib.a -framework Cocoa -framework OpenGL -framework IOKit -lm
else
RAYLIB_NAME := raylib-5.5_linux_amd64
LDLIBS_DRIVE := $(ROOT)/extern/$(RAYLIB_NAME)/lib/libraylib.a -lGL -ldl -lpthread -lrt -lX11 -lm
ifneq ($(wildcard /usr/include/EGL/egl.h),)
LDLIBS_DRIVE += -lEGL
endif
endif

BUILD_DIR := build
CFLAGS := -O0 -g -std=c99 -DPLATFORM_DESKTOP -D_GNU_SOURCE -I$(ROOT) -I$(ROOT)/extern/$(RAYLIB_NAME)/include \
	-Wall -Wextra -Wno-unused-function -Wno-unused-variable -Wno-unused-parameter -Wno-implicit-function-declaration

UNIT_SRCS := \
	test_error.c \
	test_drive_geometry.c \
	test_drive_observations_rewards.c \
	test_drive_goals.c \
	test_drive_metrics.c \
	test_drive_red_light.c \
	test_drive_eval_training_render.c \
	test_drive_dynamics.c \
	test_drive_map_cache.c \
	test_drive_idm.c
SMOKE_SRCS := test_drive_env_smoke.c
RENDER_SRCS := test_render_smoke.c
PERF_SRCS := test_perf.c

UNIT_BINS := $(UNIT_SRCS:%.c=$(BUILD_DIR)/%)
SMOKE_BINS := $(SMOKE_SRCS:%.c=$(BUILD_DIR)/%)
RENDER_BINS := $(RENDER_SRCS:%.c=$(BUILD_DIR)/%)
PERF_BINS := $(PERF_SRCS:%.c=$(BUILD_DIR)/%)

# Sim headers are compiled into every test binary; without them as
# prerequisites, editing drive.h leaves stale test binaries behind.
DRIVE_HEADERS := $(wildcard $(ROOT)/pufferlib/ocean/drive/*.h)

$(BUILD_DIR)/%: %.c include/test.h include/drive_fixture.h $(DRIVE_HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(LDLIBS_DRIVE) -o $@

unit: $(UNIT_BINS)
	@for bin in $(UNIT_BINS); do (cd $(ROOT) && tests/drive/$$bin) || exit $$?; done

smoke: $(SMOKE_BINS)
	@for bin in $(SMOKE_BINS); do (cd $(ROOT) && tests/drive/$$bin) || exit $$?; done

test: unit smoke

render: $(RENDER_BINS)
	@cd $(ROOT) && xvfb-run -a tests/drive/$(RENDER_BINS)

perf: $(PERF_BINS)
	@cd $(ROOT) && tests/drive/$(PERF_BINS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: unit smoke test render perf clean
