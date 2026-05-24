/* tinynn_backend_metal.m — Metal backend init.
 *
 * Lives in its own .o so libtinynn_ggml_metal.a contains only the
 * symbols that bridge to ggml-metal — no overlap with libtinynn_ggml.a.
 * The common-side tnn_engine_get (in tinynn_ggml.c) calls
 * tnn_backend_metal_init_internal through a weak reference, so CPU-only
 * programs link cleanly without this archive and Metal programs pull
 * the needed symbols.
 *
 * Compiled only into libtinynn_ggml_metal.a (rule in Makefile). The
 * file is .m (Objective-C) because the Metal frameworks are an ObjC
 * API; ggml-metal itself does the heavy lifting and the only thing
 * we need at this layer is the C-callable init plus the cleanup
 * trampoline.
 *
 * No BYO-pointer hook: ggml-metal doesn't expose a public
 * buffer_from_ptr API. mmap'd weights through the BYO path would
 * fall through to ggml_backend_cpu_buffer_from_ptr (the existing
 * else branch in tnn_session_attach_weight_mmap) and the ggml-metal
 * scheduler crashes when fed CPU-resident weight tensors as kernel
 * inputs. lib/transformer_lm_metal.rb sidesteps this by taking the
 * copy-load path (realize_for) for the Metal backend.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

ggml_backend_t tnn_backend_metal_init_internal(void)
{
    return ggml_backend_metal_init();
}

/* ggml-metal asserts on a non-empty residency set at process exit, via
 * the destructor of a static std::vector<unique_ptr<ggml_metal_device>>
 * registered with __cxa_atexit during ggml-metal init. The cached
 * engine pattern keeps the backend alive for the program's lifetime;
 * tearing it down via tnn_shutdown_engines does NOT drain the device's
 * residency set (the dispatch queue keeping buffers alive runs on a
 * separate thread with no public wait API).
 *
 * Rather than fight the lifecycle, expose `_exit` so single-shot
 * inference programs can skip cxa_finalize entirely — the device
 * vector's destructor never runs, the assert never fires, the OS
 * reclaims everything cleanly. Long-running processes (servers)
 * should think harder; none of our Metal callers fit that profile
 * today.
 *
 * Callers: replace `exit(rc)` with `TinyNNMetal.tnn_force_exit(rc)`
 * at end-of-main on Metal builds. */
void tnn_force_exit(int status)
{
    /* _exit skips cxa_finalize, which means libc never flushes stdio
     * buffers — Ruby/spinel-side puts that landed in the userspace
     * FILE buffer would be lost. Flush every open stream first so the
     * user sees the output they expect. */
    fflush(NULL);
    _exit(status);
}

/* Forcing-reference symbol. Pass `-Wl,-u,_tnn_metal_force_link` to the
 * linker (note the leading underscore for the macOS ABI) so this
 * object — and transitively libggml-metal.a — gets pulled in from
 * libtinynn_ggml_metal.a. Without this, the weak
 * tnn_backend_metal_init_internal fallback in tinynn_ggml.c satisfies
 * the symbol table and the strong override here never gets linked,
 * silently downgrading a "Metal" binary to CPU. */
void tnn_metal_force_link(void)
{
    volatile void *p = (void *)&ggml_backend_metal_init;
    (void)p;
}
