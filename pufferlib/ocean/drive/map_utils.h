#ifndef MAP_UTILS_H
#define MAP_UTILS_H

#include <dirent.h>
#include <stdlib.h>
#include <string.h>

// Structure to hold discovered map files
typedef struct {
    char **filenames;
    int count;
} MapFileList;

// Compare function for qsort to sort filenames alphabetically
static int compare_map_strings(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

// Scan directory for .bin files and return sorted list
static MapFileList scan_map_files(const char *map_dir) {
    MapFileList result = {NULL, 0};
    DIR *dir = opendir(map_dir);
    if (!dir) {
        return result;
    }

    // First pass: count .bin files
    struct dirent *entry;
    int count = 0;
    while ((entry = readdir(dir)) != NULL) {
        const char *name = entry->d_name;
        size_t len = strlen(name);
        if (len > 4 && strcmp(name + len - 4, ".bin") == 0) {
            count++;
        }
    }

    if (count == 0) {
        closedir(dir);
        return result;
    }

    // Allocate array
    result.filenames = malloc(count * sizeof(char *));
    if (!result.filenames) {
        closedir(dir);
        return result;
    }

    // Second pass: collect filenames
    rewinddir(dir);
    int idx = 0;
    while ((entry = readdir(dir)) != NULL && idx < count) {
        const char *name = entry->d_name;
        size_t len = strlen(name);
        if (len > 4 && strcmp(name + len - 4, ".bin") == 0) {
            // Allocate full path
            size_t path_len = strlen(map_dir) + 1 + len + 1;
            result.filenames[idx] = malloc(path_len);
            if (!result.filenames[idx]) {
                // Clean up already allocated strings
                for (int j = 0; j < idx; j++) {
                    free(result.filenames[j]);
                }
                free(result.filenames);
                result.filenames = NULL;
                closedir(dir);
                return result;
            }
            snprintf(result.filenames[idx], path_len, "%s/%s", map_dir, name);
            idx++;
        }
    }
    closedir(dir);

    // Update count to actual number found (handles race condition if files were deleted)
    result.count = idx;

    if (idx == 0) {
        free(result.filenames);
        result.filenames = NULL;
        return result;
    }

    // Sort for deterministic ordering
    qsort(result.filenames, result.count, sizeof(char *), compare_map_strings);

    return result;
}

static void free_map_file_list(MapFileList *list) {
    if (list->filenames) {
        for (int i = 0; i < list->count; i++) {
            free(list->filenames[i]);
        }
        free(list->filenames);
        list->filenames = NULL;
        list->count = 0;
    }
}

#endif // MAP_UTILS_H
