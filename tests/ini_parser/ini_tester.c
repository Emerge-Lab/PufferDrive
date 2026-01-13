#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ini.h"

#define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0

typedef struct s_drive_config {
    int num_agents;
    const char *action_type;
    int num_maps;
    int input_size;
    int hidden_size;
    int key1;
    float key2;
    int key3;
    float key31;
    char *key4;
    char *key5;
    int key6;
    char *key7;
} drive_config;

// Generic handler
static int handler(void *config, const char *section, const char *name, const char *value) {
    drive_config *env_config = (drive_config *)config;
    if (MATCH("env", "num_agents")) {
        env_config->num_agents = atoi(value);
    } else if (MATCH("env", "action_type")) {
        env_config->action_type = strdup(value);
    } else if (MATCH("env", "num_maps")) {
        env_config->num_maps = atoi(value);
    } else if (MATCH("policy", "input_size")) {
        env_config->input_size = atoi(value);
    } else if (MATCH("policy", "hidden_size")) {
        env_config->hidden_size = atoi(value);
    } else if (MATCH("test", "key1")) {
        env_config->key1 = atoi(value);
    } else if (MATCH("test", "key2")) {
        env_config->key2 = atof(value);
    } else if (MATCH("test", "key3")) {
        env_config->key3 = atoi(value);
    } else if (MATCH("test", "key31")) {
        env_config->key31 = atof(value);
    } else if (MATCH("test", "key4")) {
        env_config->key4 = strdup(value);
    } else if (MATCH("test", "key5")) {
        env_config->key5 = strdup(value);
    } else if (MATCH("test", "key6")) {
        env_config->key6 = atoi(value);
    } else if (MATCH("test", "key7")) {
        env_config->key7 = strdup(value);
    } else {
        return 0;
    }
    return 1;
}

void free_configurator(drive_config *config) {
    if (config->action_type)
        free((void *)config->action_type);
    if (config->key4)
        free((void *)config->key4);
    if (config->key5)
        free((void *)config->key5);
    if (config->key7)
        free((void *)config->key7);
}

int test_values() {
    drive_config config;
    if (ini_parse("test_drive.ini", handler, &config) < 0)
        return 1;
    assert(config.num_agents == 512);
    assert(strcmp(config.action_type, "discrete") == 0);
    assert(config.num_maps == 1);
    assert(config.input_size == 64);
    assert(config.hidden_size == 256);
    free_configurator(&config);
    return 0;
}

int test_full_line_comment() {
    drive_config config;
    if (ini_parse("test_drive.ini", handler, &config) < 0)
        return 1;
    assert(config.key1 == 1);
    assert(config.key2 == 2.5);
    assert(config.key3 != 3);
    assert(config.key31 != 3.1);
    free_configurator(&config);
    return 0;
}

int test_inline_comment() {
    drive_config config;
    if (ini_parse("test_drive.ini", handler, &config) < 0)
        return 1;
    assert(strcmp(config.key5, "five") == 0);
    assert(config.key6 == 6);
    free_configurator(&config);
    return 0;
}

int test_problematic_inline_comment() {
    drive_config config;
    if (ini_parse("test_drive.ini", handler, &config) < 0)
        return 1;
    // assert(strcmp(config.key4, "four") == 0); // should pass if comments where eluded
    assert(strcmp(config.key4, "four # and more") == 0); // # comments are interpreted as content
    // assert(strcmp(config.key7, "seven ; seven") == 0); // was expected to pass, actually fails
    assert(strcmp(config.key7, "seven") == 0); // ; comments are interpreted properly
    free_configurator(&config);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <test_name>\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "test_values") == 0) {
        return test_values();
    } else if (strcmp(argv[1], "test_full_line_comment") == 0) {
        return test_full_line_comment();
    } else if (strcmp(argv[1], "test_inline_comment") == 0) {
        return test_inline_comment();
    } else if (strcmp(argv[1], "test_problematic_inline_comment") == 0) {
        return test_problematic_inline_comment();
    }
    fprintf(stderr, "Unknown test: %s\n", argv[1]);
    return 1;
}
