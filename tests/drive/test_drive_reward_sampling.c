#include "include/drive_fixture.h"
#include "include/test.h"

#define SAMPLE_COUNT 200000
#define MAX_DECADES 8

static int decade_histogram(float min_val, float max_val, int *counts, int *decade_count_out) {
    Rng rng;
    rng_seed(&rng, 12345);
    int decade_lo, decade_hi;
    decade_bounds(min_val, max_val, &decade_lo, &decade_hi);
    int decade_count = decade_hi - decade_lo + 1;
    EXPECT_TRUE(decade_count <= MAX_DECADES);
    for (int i = 0; i < decade_count; i++) {
        counts[i] = 0;
    }
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float sample = sample_decade_uniform(&rng, min_val, max_val);
        EXPECT_TRUE(sample >= min_val && sample <= max_val);
        int decade = (int) floorf(log10f(sample));
        if (decade < decade_lo) {
            decade = decade_lo;
        }
        if (decade > decade_hi) {
            decade = decade_hi;
        }
        counts[decade - decade_lo]++;
    }
    *decade_count_out = decade_count;
    return 0;
}

// Each decade gets equal mass; that is the whole point of the scheme.
static int test_decades_are_equally_weighted(void) {
    const float bounds[][2] = {{2.5e-4f, 2.5e-2f}, {2.5e-4f, 7.5e-3f}, {1e-5f, 0.1f}};
    for (int b = 0; b < 3; b++) {
        int counts[MAX_DECADES];
        int decade_count = 0;
        if (decade_histogram(bounds[b][0], bounds[b][1], counts, &decade_count) != 0) {
            return 1;
        }
        float expected = (float) SAMPLE_COUNT / (float) decade_count;
        for (int i = 0; i < decade_count; i++) {
            EXPECT_NEAR((float) counts[i] / expected, 1.0f, 0.05f);
        }
    }
    return 0;
}

// A max_val that is itself a power of ten must not collapse the top decade to a point.
static int test_power_of_ten_max_is_not_a_point_mass(void) {
    Rng rng;
    rng_seed(&rng, 999);
    int at_max = 0;
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        if (sample_decade_uniform(&rng, 1e-5f, 0.1f) >= 0.0999f) {
            at_max++;
        }
    }
    EXPECT_TRUE(at_max < SAMPLE_COUNT / 100);
    return 0;
}

// The conditioning signal is the quantile, so it must be uniform over [0, 1].
static int test_quantile_is_uniform(void) {
    Rng rng;
    rng_seed(&rng, 4242);
    const float min_val = 2.5e-4f;
    const float max_val = 2.5e-2f;
    int bins[10] = {0};
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float sample = sample_decade_uniform(&rng, min_val, max_val);
        float quantile = decade_uniform_quantile(sample, min_val, max_val);
        EXPECT_TRUE(quantile >= 0.0f && quantile <= 1.0f);
        int bin = (int) (quantile * 10.0f);
        if (bin > 9) {
            bin = 9;
        }
        bins[bin]++;
    }
    float expected = (float) SAMPLE_COUNT / 10.0f;
    for (int i = 0; i < 10; i++) {
        EXPECT_NEAR((float) bins[i] / expected, 1.0f, 0.05f);
    }
    return 0;
}

static int test_quantile_spans_endpoints_and_is_monotonic(void) {
    const float min_val = 2.5e-4f;
    const float max_val = 7.5e-3f;
    EXPECT_NEAR(decade_uniform_quantile(min_val, min_val, max_val), 0.0f, 1e-5f);
    EXPECT_NEAR(decade_uniform_quantile(max_val, min_val, max_val), 1.0f, 1e-5f);

    float previous = -1.0f;
    for (int i = 0; i <= 100; i++) {
        float coef = min_val + (max_val - min_val) * ((float) i / 100.0f);
        float quantile = decade_uniform_quantile(coef, min_val, max_val);
        EXPECT_TRUE(quantile >= previous);
        previous = quantile;
    }
    return 0;
}

// Uniform sampling starves the low end of a wide range; that is the bug being fixed.
static int test_low_end_is_reachable_unlike_uniform(void) {
    Rng rng;
    rng_seed(&rng, 7);
    const float min_val = 2.5e-4f;
    const float max_val = 2.5e-2f;
    int decade_below_uniform = 0;
    int decade_below_decadewise = 0;
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        if (sample_uniform(&rng, min_val, max_val) < 1e-3f) {
            decade_below_uniform++;
        }
        if (sample_decade_uniform(&rng, min_val, max_val) < 1e-3f) {
            decade_below_decadewise++;
        }
    }
    EXPECT_TRUE(decade_below_decadewise > 8 * decade_below_uniform);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_decades_are_equally_weighted);
    RUN_TEST(test_power_of_ten_max_is_not_a_point_mass);
    RUN_TEST(test_quantile_is_uniform);
    RUN_TEST(test_quantile_spans_endpoints_and_is_monotonic);
    RUN_TEST(test_low_end_is_reachable_unlike_uniform);
    return test_summary(failures);
}
