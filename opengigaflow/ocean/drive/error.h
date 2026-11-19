#ifndef OPENGIGAFLOW_OCEAN_DRIVE_ERROR_H
#define OPENGIGAFLOW_OCEAN_DRIVE_ERROR_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum {
    ERROR_INVALID_ARGUMENT,
    ERROR_MEMORY_ALLOCATION,
    ERROR_INITIALIZATION_FAILED,
} ErrorType;

const char *error_type_to_string(ErrorType type) {
    switch (type) {
    case ERROR_INVALID_ARGUMENT:
        return "Invalid Argument";
    case ERROR_MEMORY_ALLOCATION:
        return "Memory Allocation Failed";
    case ERROR_INITIALIZATION_FAILED:
        return "Initialization Failed";
    default:
        return "Unknown Error";
    }
}

void raise_error_with_message(ErrorType type, const char *format, ...) {
    printf("Error occurred: %s - ", error_type_to_string(type));
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
    exit(EXIT_FAILURE);
}

#endif
