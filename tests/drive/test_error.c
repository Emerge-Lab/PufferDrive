#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/wait.h>
#include "pufferlib/ocean/drive/error.h"

// Test helper to run functions that call exit() in child processes
int test_exit_function(void (*test_func)(void)) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process - run the test function that exits
        test_func();
        // Should never reach here
        exit(0);
    } else if (pid > 0) {
        // Parent process - wait for child and check exit status
        int status;
        wait(&status);
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        }
        return -1;
    } else {
        // Fork failed
        perror("fork failed");
        return -1;
    }
}

// Test functions that call the error macros
void test_file_error(void) {
    printf("Testing RAISE_FILE_ERROR...\n");
    RAISE_FILE_ERROR("/path/to/missing/file.txt");
}

void test_bounds_error(void) {
    printf("Testing RAISE_BOUNDS_ERROR...\n");
    RAISE_BOUNDS_ERROR();
}

void test_bounds_error_with_bounds(void) {
    printf("Testing RAISE_BOUNDS_ERROR_WITH_BOUNDS...\n");
    RAISE_BOUNDS_ERROR_WITH_BOUNDS(10, 0, 5);
}

void test_null_error(void) {
    printf("Testing RAISE_NULL_ERROR...\n");
    RAISE_NULL_ERROR();
}

void test_null_error_with_name(void) {
    printf("Testing RAISE_NULL_ERROR_WITH_NAME...\n");
    RAISE_NULL_ERROR_WITH_NAME("env");
}

void test_memory_error(void) {
    printf("Testing RAISE_MEMORY_ERROR...\n");
    RAISE_MEMORY_ERROR();
}

void test_memory_error_with_size(void) {
    printf("Testing RAISE_MEMORY_ERROR_WITH_SIZE...\n");
    RAISE_MEMORY_ERROR_WITH_SIZE(1024);
}

void test_invalid_arg_error(void) {
    printf("Testing RAISE_INVALID_ARG_ERROR...\n");
    RAISE_INVALID_ARG_ERROR();
}

void test_invalid_arg_error_with_arg(void) {
    printf("Testing RAISE_INVALID_ARG_ERROR_WITH_ARG...\n");
    RAISE_INVALID_ARG_ERROR_WITH_ARG("agent_idx", -1);
}

void test_raise_error_direct(void) {
    printf("Testing raise_error directly...\n");
    raise_error(ERROR_UNKNOWN);
}

void test_raise_error_with_message_direct(void) {
    printf("Testing raise_error_with_message directly...\n");
    raise_error_with_message(ERROR_INITIALIZATION_FAILED, "failed to initialize component '%s' with value %d",
                             "neural_network", 42);
}

// Test the error_type_to_string function (doesn't exit)
void test_error_type_to_string(void) {
    printf("\n=== Testing error_type_to_string function ===\n");

    printf("ERROR_NONE: %s\n", error_type_to_string(ERROR_NONE));
    printf("ERROR_NULL_POINTER: %s\n", error_type_to_string(ERROR_NULL_POINTER));
    printf("ERROR_INVALID_ARGUMENT: %s\n", error_type_to_string(ERROR_INVALID_ARGUMENT));
    printf("ERROR_OUT_OF_BOUNDS: %s\n", error_type_to_string(ERROR_OUT_OF_BOUNDS));
    printf("ERROR_MEMORY_ALLOCATION: %s\n", error_type_to_string(ERROR_MEMORY_ALLOCATION));
    printf("ERROR_FILE_NOT_FOUND: %s\n", error_type_to_string(ERROR_FILE_NOT_FOUND));
    printf("ERROR_INITIALIZATION_FAILED: %s\n", error_type_to_string(ERROR_INITIALIZATION_FAILED));
    printf("ERROR_UNKNOWN: %s\n", error_type_to_string(ERROR_UNKNOWN));
    printf("Invalid error type (99): %s\n", error_type_to_string(99));
}

int main(void) {
    printf("=== Error Handling Unit Tests ===\n\n");

    // Test error_type_to_string (doesn't exit)
    test_error_type_to_string();

    printf("\n=== Testing Error Macros (each runs in separate process) ===\n");

    // Test each error macro in a separate process
    struct {
        const char *name;
        void (*func)(void);
    } tests[] = {{"RAISE_FILE_ERROR", test_file_error},
                 {"RAISE_BOUNDS_ERROR", test_bounds_error},
                 {"RAISE_BOUNDS_ERROR_WITH_BOUNDS", test_bounds_error_with_bounds},
                 {"RAISE_NULL_ERROR", test_null_error},
                 {"RAISE_NULL_ERROR_WITH_NAME", test_null_error_with_name},
                 {"RAISE_MEMORY_ERROR", test_memory_error},
                 {"RAISE_MEMORY_ERROR_WITH_SIZE", test_memory_error_with_size},
                 {"RAISE_INVALID_ARG_ERROR", test_invalid_arg_error},
                 {"RAISE_INVALID_ARG_ERROR_WITH_ARG", test_invalid_arg_error_with_arg},
                 {"raise_error (direct)", test_raise_error_direct},
                 {"raise_error_with_message (direct)", test_raise_error_with_message_direct}};

    int num_tests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;

    for (int i = 0; i < num_tests; i++) {
        printf("\n--- Test %d: %s ---\n", i + 1, tests[i].name);

        int exit_status = test_exit_function(tests[i].func);

        if (exit_status == EXIT_FAILURE) {
            printf("PASS: Function exited with EXIT_FAILURE as expected\n");
            passed++;
        } else {
            printf("FAIL: Function exited with status %d (expected %d)\n", exit_status, EXIT_FAILURE);
        }
    }

    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", num_tests);
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", num_tests - passed);

    if (passed == num_tests) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("Some tests failed!\n");
        return 1;
    }
}
