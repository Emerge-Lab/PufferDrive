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
	test_drive_pedestrians.c \
	test_drive_env_smoke.c
PERF_SRCS := test_perf.c

UNIT_BINS := $(UNIT_SRCS:%.c=$(BUILD_DIR)/%)
PERF_BINS := $(PERF_SRCS:%.c=$(BUILD_DIR)/%)

# Prerequisite of every test binary, so editing a sim header rebuilds them all
DRIVE_HEADERS := $(wildcard $(ROOT)/opengigaflow/ocean/drive/*.h)

$(BUILD_DIR)/%: %.c include/test.h include/drive_fixture.h $(DRIVE_HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(LDLIBS_DRIVE) -o $@

unit: $(UNIT_BINS)
	@for bin in $(UNIT_BINS); do (cd $(ROOT) && tests/drive/$$bin) || exit $$?; done

test: unit

perf: $(PERF_BINS)
	@cd $(ROOT) && tests/drive/$(PERF_BINS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: unit test perf clean
