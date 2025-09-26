#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

// Error types enumeration
typedef enum {
    ERROR_NONE = 0,
    ERROR_NULL_POINTER,
    ERROR_INVALID_ARGUMENT,
    ERROR_OUT_OF_BOUNDS,
    ERROR_MEMORY_ALLOCATION,
    ERROR_FILE_NOT_FOUND,
    ERROR_INITIALIZATION_FAILED,
    ERROR_UNKNOWN
} ErrorType;

const char* error_type_to_string(ErrorType type) {
    switch (type) {
        case ERROR_NONE: return "No Error";
        case ERROR_NULL_POINTER: return "Null Pointer";
        case ERROR_INVALID_ARGUMENT: return "Invalid Argument";
        case ERROR_OUT_OF_BOUNDS: return "Out of Bounds";
        case ERROR_MEMORY_ALLOCATION: return "Memory Allocation Failed";
        case ERROR_FILE_NOT_FOUND: return "File Not Found";
        case ERROR_INITIALIZATION_FAILED: return "Initialization Failed";
        default: return "Unknown Error";
    }
}

// Enhanced error function with custom message support
void raise_error_with_message(ErrorType type, const char* format, ...) {
    printf("Error occurred: %s", error_type_to_string(type));

    if (format != NULL) {
        printf(" - ");
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
    printf("\n");
    exit(EXIT_FAILURE);
}

// Simple error function (backward compatibility)
void raise_error(ErrorType type) {
    raise_error_with_message(type, NULL);
}

// Convenience macros for common error patterns
#define RAISE_FILE_ERROR(path) \
    raise_error_with_message(ERROR_FILE_NOT_FOUND, "at path: %s", path)

#define RAISE_BOUNDS_ERROR() \
    raise_error(ERROR_OUT_OF_BOUNDS)

#define RAISE_BOUNDS_ERROR_WITH_BOUNDS(index, min, max) \
    raise_error_with_message(ERROR_OUT_OF_BOUNDS, "index %d exceeds minimum of %d and maximum %d", index, min, max)

#define RAISE_NULL_ERROR() \
    raise_error(ERROR_NULL_POINTER)

#define RAISE_NULL_ERROR_WITH_NAME(var_name) \
    raise_error_with_message(ERROR_NULL_POINTER, "variable '%s' is null", var_name)

#define RAISE_MEMORY_ERROR() \
    raise_error(ERROR_MEMORY_ALLOCATION)

#define RAISE_MEMORY_ERROR_WITH_SIZE(size) \
    raise_error_with_message(ERROR_MEMORY_ALLOCATION, "failed to allocate %zu bytes", size)

#define RAISE_INVALID_ARG_ERROR() \
    raise_error(ERROR_INVALID_ARGUMENT)

#define RAISE_INVALID_ARG_ERROR_WITH_ARG(arg_name, value) \
    raise_error_with_message(ERROR_INVALID_ARGUMENT, "invalid value for '%s': %d", arg_name, value)
#endif
