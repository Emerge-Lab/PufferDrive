/*
 * ffmpeg_pipe.h — write rendered RGBA frames to an ffmpeg subprocess.
 *
 * Each FfmpegPipe is a unidirectional handle to an ffmpeg process whose
 * stdin we feed raw RGBA pixels. ffmpeg encodes them to H.264 in an MP4
 * via libx264. We use popen() so we don't have to manage fork/exec/dup2
 * by hand — at the cost of going through /bin/sh, which is fine because
 * the output paths are caller-supplied and the rest of the args are
 * static.
 *
 * One pipe per output MP4. The orchestrator opens (up to) two pipes per
 * episode — one for top-down, one for BEV — writes one frame at a time
 * to each, and closes both at episode end.
 *
 * Error model: write returns 0 on success, non-zero if fwrite fails (e.g.
 * ffmpeg crashed or the disk filled up). The pipe is left open; the
 * caller should close it.
 */

#ifndef FFMPEG_PIPE_H
#define FFMPEG_PIPE_H

#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

typedef struct FfmpegPipe {
    FILE *fp; /* popen handle (kept so pclose can close it) */
    int fd;   /* cached fileno(fp) — we use raw write() */
    int width;
    int height;
    int fps;
    char path[1024]; /* output mp4 path, kept for error messages */

    /* Writer thread + signaling. Each pipe gets its own background
     * thread that does the blocking write() in parallel with the main
     * thread + the other pipes' threads, so vk_batch_renderer's per-
     * frame "submit all → wait all" loop costs max(single write) per
     * frame instead of sum-of-writes. */
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv_go;   /* main → writer: new work pending */
    pthread_cond_t cv_done; /* writer → main: write completed */
    int thread_started;
    int stop;
    int work_pending;
    int work_error;
    const void *pending_data; /* borrowed, valid between submit/wait */
    size_t pending_bytes;
} FfmpegPipe;

/* Spawn ffmpeg writing to out_mp4. Returns 0 on success, non-zero on
 * popen failure. The ffmpeg binary path is taken from $TRAJVIZ_FFMPEG if
 * set, else "ffmpeg". */
int ffmpeg_pipe_open(FfmpegPipe *p, int width, int height, int fps, const char *out_mp4);

/* Write one frame's worth of RGBA bytes (width*height*4). SYNC: blocks
 * until the write completes. Internally implemented as
 * submit + wait — used by the single-episode path that doesn't need
 * fan-out parallelism. */
int ffmpeg_pipe_write_frame(FfmpegPipe *p, const void *rgba_bytes);

/* ASYNC: hand off a frame to the pipe's writer thread and return
 * immediately. The buffer pointer must stay valid until ffmpeg_pipe_wait
 * returns for the same pipe. Will block briefly if a previous submit on
 * this pipe is still running. */
int ffmpeg_pipe_submit_frame(FfmpegPipe *p, const void *rgba_bytes);

/* Wait for the most recent submit_frame on this pipe to complete.
 * Returns 0 on success or the error code from the writer thread's
 * write() call. Idempotent — safe to call when no submit is pending. */
int ffmpeg_pipe_wait(FfmpegPipe *p);

/* Close the pipe and wait for ffmpeg to flush. Idempotent. Returns the
 * exit status of ffmpeg (0 = success). */
int ffmpeg_pipe_close(FfmpegPipe *p);

#endif
