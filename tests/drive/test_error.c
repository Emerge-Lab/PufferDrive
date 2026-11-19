#include "include/test.h"
#include "opengigaflow/ocean/drive/error.h"

#include <sys/wait.h>
#include <unistd.h>

static int test_raise_error_with_message_exits_with_failure(void) {
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        raise_error_with_message(
            ERROR_INITIALIZATION_FAILED,
            "failed to initialize component '%s' with value %d",
            "neural_network",
            42);
        exit(0);
    }
    EXPECT_TRUE(pid > 0);
    int status;
    wait(&status);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ_INT(WEXITSTATUS(status), EXIT_FAILURE);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_raise_error_with_message_exits_with_failure);
    return test_summary(failures);
}
