/*
 * ffmpeg_pipe.c — popen-based RGBA → MP4 streaming.
 *
 * The ffmpeg invocation matches what visualize.c uses on the live
 * raylib path, with the same -preset and -crf so output sizes are
 * comparable. Single-pass libx264, yuv420p (the most compatible pixel
 * format for downstream players).
 *
 * Notes on robustness:
 *   - We pipe raw rgba in row-major order, no padding (width * 4 bytes
 *     per row, height rows). The renderer uses HOST_COHERENT readback
 *     buffers tightly packed at width*4 stride, so no row-pitch
 *     conversion is needed here.
 *   - We rely on the OS to write a full frame to ffmpeg's stdin in one
 *     fwrite call. The pipe buffer size on Linux is typically 64 KiB
 *     and a 1280×720 RGBA frame is 3.6 MiB, so fwrite will internally
 *     loop on a blocking pipe — that's the correct behavior.
 */

/* F_SETPIPE_SZ is a Linux extension behind _GNU_SOURCE on glibc. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ffmpeg_pipe.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#endif

/* Forward decls so ffmpeg_pipe_open can pthread_create the writer. */
static void *writer_thread_main(void *arg);
static int do_blocking_write(FfmpegPipe *p, const void *data, size_t bytes);

int ffmpeg_pipe_open(FfmpegPipe *p, int width, int height, int fps, const char *out_mp4) {
    if (!p || !out_mp4)
        return -1;
    memset(p, 0, sizeof(*p));
    p->fd = -1;
    p->width = width;
    p->height = height;
    p->fps = fps;
    snprintf(p->path, sizeof(p->path), "%s", out_mp4);

    const char *ffmpeg_bin = getenv("TRAJVIZ_FFMPEG");
    if (!ffmpeg_bin || !*ffmpeg_bin)
        ffmpeg_bin = "ffmpeg";

    /* TRAJVIZ_NO_FFMPEG=1 → bypass ffmpeg entirely for benchmarking the
     * pure Vulkan path. We sink raw RGBA bytes to /dev/null via cat,
     * which removes the libx264 encode cost from the timing loop. */
    int bypass = 0;
    const char *no_ff = getenv("TRAJVIZ_NO_FFMPEG");
    if (no_ff && *no_ff && *no_ff != '0')
        bypass = 1;

    /* Build the ffmpeg command line. We single-quote the output path so
     * shell metacharacters in user-supplied paths don't blow us up — but
     * a single-quote in the path itself would still break, so we reject
     * paths containing one. */
    if (strchr(out_mp4, '\'') != NULL) {
        fprintf(stderr, "[trajviz] output path contains single quote: %s\n", out_mp4);
        return -1;
    }

    char cmd[2048];
    int n;
    if (bypass) {
        n = snprintf(cmd, sizeof(cmd), "cat > /dev/null");
    } else {
        n = snprintf(cmd, sizeof(cmd),
                     "%s -y -hide_banner -loglevel error "
                     "-f rawvideo -pix_fmt rgba "
                     "-s %dx%d -r %d -i - "
                     "-c:v libx264 -pix_fmt yuv420p "
                     "-preset veryfast -crf 20 "
                     "'%s'",
                     ffmpeg_bin, width, height, fps, out_mp4);
    }
    if (n < 0 || n >= (int)sizeof(cmd)) {
        fprintf(stderr, "[trajviz] ffmpeg command too long\n");
        return -1;
    }

    p->fp = popen(cmd, "w");
    if (!p->fp) {
        fprintf(stderr, "[trajviz] popen(\"%s\") failed\n", cmd);
        return -1;
    }

#ifdef __linux__
    /* Cache the underlying file descriptor — we use raw write() in the
     * hot path because libc fwrite chunks our 3.6 MB frame through its
     * default ~8 KB stdio buffer (450+ syscalls per frame), which is
     * the actual single biggest bottleneck on this path. Disable any
     * stdio buffering as a paranoia measure in case anything ever does
     * touch p->fp. */
    p->fd = fileno(p->fp);
    if (p->fd >= 0) {
        setvbuf(p->fp, NULL, _IONBF, 0);
    }

    /* Also bump the kernel pipe buffer up to whatever the per-process
     * limit allows — ideally enough to fit multiple full frames so the
     * producer can race ahead of the consumer. Default 64 KB → 1 MB
     * unprivileged → 8+ MB with sudo sysctl. */
    if (p->fd >= 0) {
        long want_one_frame = (long)width * (long)height * 4;
        long tries[] = {
            16L << 20, /* 16 MB — needs sudo sysctl fs.pipe-max-size=16777216 */
            8L << 20,  4L << 20, 2L << 20, 1L << 20, 512L << 10, 256L << 10,
        };
        int got = 0;
        for (size_t i = 0; i < sizeof(tries) / sizeof(tries[0]); ++i) {
            if (fcntl(p->fd, F_SETPIPE_SZ, (int)tries[i]) >= 0) {
                got = (int)tries[i];
                break;
            }
        }
        static int warned_small_pipe = 0;
        if (got > 0 && got < want_one_frame && !warned_small_pipe) {
            fprintf(stderr,
                    "[trajviz] pipe size %d B < frame size %ld B — fwrites "
                    "may block. Raise /proc/sys/fs/pipe-max-size for better "
                    "throughput (sudo sysctl fs.pipe-max-size=16777216).\n",
                    got, want_one_frame);
            warned_small_pipe = 1;
        }
    }
#endif

    /* Spin up the writer thread. From here on, all writes go through
     * submit_frame → cv_go → writer_thread_main → write() → cv_done. */
    if (pthread_mutex_init(&p->mu, NULL) != 0 || pthread_cond_init(&p->cv_go, NULL) != 0 ||
        pthread_cond_init(&p->cv_done, NULL) != 0) {
        fprintf(stderr, "[trajviz] failed to init writer thread sync for %s\n", p->path);
        pclose(p->fp);
        p->fp = NULL;
        return -1;
    }
    if (pthread_create(&p->thread, NULL, writer_thread_main, p) != 0) {
        fprintf(stderr, "[trajviz] pthread_create failed for %s\n", p->path);
        pthread_mutex_destroy(&p->mu);
        pthread_cond_destroy(&p->cv_go);
        pthread_cond_destroy(&p->cv_done);
        pclose(p->fp);
        p->fp = NULL;
        return -1;
    }
    p->thread_started = 1;

    return 0;
}

/* The actual blocking write — used by the writer thread. Loops on
 * EINTR + partial writes. Returns 0 on success, -1 on error. */
static int do_blocking_write(FfmpegPipe *p, const void *data, size_t bytes) {
    static int no_write_cached = -1;
    if (no_write_cached < 0) {
        const char *e = getenv("TRAJVIZ_NO_WRITE");
        no_write_cached = (e && *e && *e != '0') ? 1 : 0;
    }
    if (no_write_cached)
        return 0;

#ifdef __linux__
    const uint8_t *buf = (const uint8_t *)data;
    size_t left = bytes;
    while (left > 0) {
        ssize_t n = write(p->fd, buf, left);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "[trajviz] write() failed (%s) for %s\n", strerror(errno), p->path);
            return -1;
        }
        if (n == 0) {
            fprintf(stderr, "[trajviz] write() returned 0 — pipe closed for %s\n", p->path);
            return -1;
        }
        buf += (size_t)n;
        left -= (size_t)n;
    }
    return 0;
#else
    size_t got = fwrite(data, 1, bytes, p->fp);
    if (got != bytes) {
        fprintf(stderr, "[trajviz] short write to ffmpeg pipe (%zu/%zu) for %s\n", got, bytes, p->path);
        return -1;
    }
    return 0;
#endif
}

/* Background writer thread main loop. Sleeps on cv_go, wakes up when
 * the main thread submits work, does the write outside the lock so
 * other threads can proceed in parallel, then signals cv_done. */
static void *writer_thread_main(void *arg) {
    FfmpegPipe *p = (FfmpegPipe *)arg;

    pthread_mutex_lock(&p->mu);
    for (;;) {
        while (!p->work_pending && !p->stop) {
            pthread_cond_wait(&p->cv_go, &p->mu);
        }
        if (p->stop) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }

        /* Snapshot the work and release the lock so other writers can
         * be submitted to in parallel while we write. */
        const void *data = p->pending_data;
        size_t bytes = p->pending_bytes;
        pthread_mutex_unlock(&p->mu);

        int err = do_blocking_write(p, data, bytes);

        pthread_mutex_lock(&p->mu);
        p->work_error = err;
        p->work_pending = 0;
        pthread_cond_signal(&p->cv_done);
    }
}

int ffmpeg_pipe_submit_frame(FfmpegPipe *p, const void *rgba_bytes) {
    if (!p || !rgba_bytes)
        return -1;
    if (!p->thread_started) {
        /* No writer thread — fall back to synchronous write. */
        return do_blocking_write(p, rgba_bytes, (size_t)p->width * (size_t)p->height * 4);
    }

    pthread_mutex_lock(&p->mu);
    /* Drain any in-flight write before submitting a new one. The caller
     * is supposed to call wait() between frames so this should normally
     * be a no-op, but we guard against misuse. */
    while (p->work_pending) {
        pthread_cond_wait(&p->cv_done, &p->mu);
    }
    p->pending_data = rgba_bytes;
    p->pending_bytes = (size_t)p->width * (size_t)p->height * 4;
    p->work_pending = 1;
    p->work_error = 0;
    pthread_cond_signal(&p->cv_go);
    pthread_mutex_unlock(&p->mu);
    return 0;
}

int ffmpeg_pipe_wait(FfmpegPipe *p) {
    if (!p)
        return -1;
    if (!p->thread_started)
        return 0; /* sync mode — already done */

    pthread_mutex_lock(&p->mu);
    while (p->work_pending) {
        pthread_cond_wait(&p->cv_done, &p->mu);
    }
    int err = p->work_error;
    pthread_mutex_unlock(&p->mu);
    return err;
}

int ffmpeg_pipe_write_frame(FfmpegPipe *p, const void *rgba_bytes) {
    /* Sync wrapper: submit + wait. Used by the single-episode path. */
    int rc = ffmpeg_pipe_submit_frame(p, rgba_bytes);
    if (rc != 0)
        return rc;
    return ffmpeg_pipe_wait(p);
}

int ffmpeg_pipe_close(FfmpegPipe *p) {
    if (!p)
        return 0;

    /* Drain any in-flight write, then signal the writer to exit and
     * join it. After this, no thread is touching p->fp / p->fd, so we
     * can safely pclose. */
    if (p->thread_started) {
        ffmpeg_pipe_wait(p);
        pthread_mutex_lock(&p->mu);
        p->stop = 1;
        pthread_cond_signal(&p->cv_go);
        pthread_mutex_unlock(&p->mu);
        pthread_join(p->thread, NULL);
        pthread_mutex_destroy(&p->mu);
        pthread_cond_destroy(&p->cv_go);
        pthread_cond_destroy(&p->cv_done);
        p->thread_started = 0;
    }

    if (!p->fp)
        return 0;
    int status = pclose(p->fp);
    p->fp = NULL;
    p->fd = -1;
    return status;
}
