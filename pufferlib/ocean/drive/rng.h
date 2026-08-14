#ifndef PUFFERLIB_OCEAN_DRIVE_RNG_H
#define PUFFERLIB_OCEAN_DRIVE_RNG_H

#include <stdint.h>

// xoshiro256++ (Blackman & Vigna, public domain), state passed explicitly like rand_r.

typedef struct {
    uint64_t s[4];
} Rng;

static inline uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// splitmix64 expansion keeps the state valid (never all-zero) and well mixed for any seed.
static inline void rng_seed(Rng *rng, uint64_t seed) {
    uint64_t splitmix_state = seed;
    rng->s[0] = splitmix64_next(&splitmix_state);
    rng->s[1] = splitmix64_next(&splitmix_state);
    rng->s[2] = splitmix64_next(&splitmix_state);
    rng->s[3] = splitmix64_next(&splitmix_state);
}

static inline uint64_t rng_rotl64(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(Rng *rng) {
    uint64_t *s = rng->s;
    const uint64_t result = rng_rotl64(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rng_rotl64(s[3], 45);
    return result;
}

// [0, 1) from the top 24 bits (float mantissa width); never returns 1.0f.
static inline float rng_uniform_f32(Rng *rng) {
    return (float) (rng_next(rng) >> 40) * (1.0f / 16777216.0f);
}

// [0, upper_exclusive), unbiased: Lemire's debiased multiply-shift, mirroring Rust rand's
// UniformInt::sample. Terminates: rejection probability < 0.5 per draw, and the full-period
// generator cannot emit rejected values forever.
static inline int rng_below(Rng *rng, int upper_exclusive) {
    uint32_t range = (uint32_t) upper_exclusive;
    uint32_t reject_below = (uint32_t) (0u - range) % range; // (2^32 - range) % range
    while (1) {
        uint64_t m = (uint64_t) (uint32_t) (rng_next(rng) >> 32) * (uint64_t) range;
        if ((uint32_t) m >= reject_below) {
            return (int) (m >> 32);
        }
    }
}

#endif
