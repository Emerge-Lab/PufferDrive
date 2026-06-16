#ifndef DRIVE_TEST_H
#define DRIVE_TEST_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPECT_TRUE(expr)                                                                                              \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);                                                     \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

#define EXPECT_EQ_INT(actual, expected)                                                                                \
    do {                                                                                                               \
        int actual_value = (actual);                                                                                   \
        int expected_value = (expected);                                                                               \
        if (actual_value != expected_value) {                                                                          \
            printf("FAIL %s:%d: %s == %d, expected %d\n", __FILE__, __LINE__, #actual, actual_value, expected_value);  \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define EXPECT_NEAR(actual, expected, tol)                                                                             \
    do {                                                                                                               \
        float actual_value = (actual);                                                                                 \
        float expected_value = (expected);                                                                             \
        if (fabsf(actual_value - expected_value) > (tol)) {                                                            \
            printf("FAIL %s:%d: %s == %f, expected %f\n", __FILE__, __LINE__, #actual, actual_value, expected_value);  \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define EXPECT_FINITE(value)                                                                                           \
    do {                                                                                                               \
        float finite_value = (value);                                                                                  \
        if (!isfinite(finite_value)) {                                                                                 \
            printf("FAIL %s:%d: %s is not finite: %f\n", __FILE__, __LINE__, #value, finite_value);                    \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define RUN_TEST(test_fn)                                                                                              \
    do {                                                                                                               \
        int failed = test_fn();                                                                                        \
        if (failed) {                                                                                                  \
            failures++;                                                                                                \
        } else {                                                                                                       \
            printf("PASS %s\n", #test_fn);                                                                             \
        }                                                                                                              \
    } while (0)

static inline int test_summary(int failures) {
    if (failures) {
        printf("%d test groups failed\n", failures);
        return 1;
    }
    printf("All test groups passed\n");
    return 0;
}

#endif
