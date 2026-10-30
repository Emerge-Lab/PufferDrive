CC ?= gcc
ROOT := ../..
LDLIBS_DRIVE := -lm

BUILD_DIR := build
CFLAGS := -O0 -g -std=c99 -D_GNU_SOURCE -I$(ROOT) \
	-Wall -Wextra -Wno-unused-function -Wno-unused-variable -Wno-unused-parameter -Wno-implicit-function-declaration

UNIT_SRCS := \
	test_error.c \
	test_drive_geometry.c \
	test_drive_observations_rewards.c \
	test_drive_goals.c \
	test_drive_metrics.c \
	test_drive_red_light.c \
	test_drive_stop_sign.c \
	test_drive_traffic_lights.c \
	test_drive_dynamics.c \
	test_drive_map_cache.c \
	test_drive_idm.c \
	test_drive_speed_limits.c \
	test_drive_cosim_setters.c \
	test_drive_human_control.c \
	test_drive_human_powertrain.c \
	test_drive_substep.c \
	test_drive_steering_convention.c
SMOKE_SRCS := test_drive_env_smoke.c
PERF_SRCS := test_perf.c

UNIT_BINS := $(UNIT_SRCS:%.c=$(BUILD_DIR)/%)
SMOKE_BINS := $(SMOKE_SRCS:%.c=$(BUILD_DIR)/%)
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

# Renderer maths. Needs raylib headers, so it is kept out of `test` -- the sim suite must keep
# building on a machine with no raylib. Run with: make -C tests/drive render
RENDER_SRCS := test_drive_render.c
RENDER_BINS := $(RENDER_SRCS:%.c=$(BUILD_DIR)/%)

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
RAYLIB_DIR := $(ROOT)/extern/raylib-5.5_macos
RAYLIB_LIB := $(RAYLIB_DIR)/lib/libraylib.a
RAYLIB_INCLUDE := $(RAYLIB_DIR)/include
RAYLIB_LDFLAGS := -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL
else
RAYLIB_LIB := -lraylib
RAYLIB_INCLUDE := /usr/local/include
RAYLIB_LDFLAGS := -lGL -lpthread -ldl -lrt -lX11
endif

RENDER_HEADERS := $(wildcard $(ROOT)/pufferlib/ocean/drive/render/*.h)

$(BUILD_DIR)/test_drive_render: test_drive_render.c include/test.h include/drive_fixture.h \
		$(DRIVE_HEADERS) $(RENDER_HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RAYLIB_INCLUDE) $< $(RAYLIB_LIB) $(RAYLIB_LDFLAGS) $(LDLIBS_DRIVE) -o $@

render: $(RENDER_BINS)
	@for bin in $(RENDER_BINS); do (cd $(ROOT) && tests/drive/$$bin) || exit $$?; done

perf: $(PERF_BINS)
	@cd $(ROOT) && tests/drive/$(PERF_BINS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: unit smoke test render perf clean
