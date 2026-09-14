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
	test_drive_dynamics.c \
	test_drive_map_cache.c \
	test_drive_idm.c
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

perf: $(PERF_BINS)
	@cd $(ROOT) && tests/drive/$(PERF_BINS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: unit smoke test perf clean
