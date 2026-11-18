/* PocketTTS.cpp — C API (FFI / embedding).
 * Implemented in pocket_tts.cpp; link pocket_tts_static (CMake) or the shared library.
 * Audio: float PCM, 24 kHz, mono.
 */
#ifndef POCKET_TTS_H
#define POCKET_TTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Creates an engine. Returns NULL on error (message on stderr).
 * precision: "int8" or "fp32". num_threads: 0 = half the cores. */
void* ptt_create(const char* models_dir, const char* voices_dir, const char* tokenizer_path,
                 const char* precision, float temperature, int lsd_steps, int num_threads);

/* All engine options. Initialise with ptt_config_default(), then change what you need. */
typedef struct ptt_config {
    const char* models_dir;      /* default "models" */
    const char* voices_dir;      /* default "voices" */
    const char* tokenizer_path;  /* default "models/tokenizer.model" */
    const char* precision;       /* "int8" (default) or "fp32" */
    float temperature;           /* default 0.7 */
    int   lsd_steps;             /* default 1 */
    int   num_threads;           /* default 0 = half the cores */
    float eos_threshold;         /* end-of-speech logit threshold, default -4.0 */
    int   eos_extra_frames;      /* frames generated after end of speech; -1 = auto (default) */
    float noise_clamp;           /* 0 = off (default) */
    int   voice_cache;           /* 1 = cache voice embeddings / KV state on disk (default) */
    const char* cache_dir;       /* voice cache folder; NULL = "<voices_dir>/.cache" (default).
                                    Files: <stem>-<hash of the voice path>.emb / .kv */
} ptt_config;

ptt_config ptt_config_default(void);

/* Creates an engine from a full configuration. Returns NULL on error (message on stderr). */
void* ptt_create_ex(const ptt_config* config);

/* Runs one short synthesis so the first real line is fast. Returns seconds, or -1 on error. */
double ptt_warmup(void* handle);

void ptt_destroy(void* handle);
void ptt_free_audio(float* samples);

/* Streaming synthesis on a background thread.
 * voice: a file name in voices_dir, or a path to a WAV/MP3/FLAC sample. */
void* ptt_stream_start(void* handle, const char* text, const char* voice);

/* Blocks until a chunk is ready. Returns 1 with a chunk (free it with ptt_free_audio),
 * 0 when the stream is finished, -1 on bad arguments. */
int ptt_stream_read(void* stream, float** out_samples, int* out_len);

/* Stops (if still running) and frees the stream. */
void ptt_stream_end(void* stream);

#ifdef __cplusplus
}
#endif

#endif /* POCKET_TTS_H */
