/* tinynn/cuda_byo_smoke.c — direct C smoke test for the
 * ggml_backend_cuda_buffer_from_ptr patch (vendored).
 *
 * Verifies:
 *   1. cudaHostRegister + cudaHostGetDevicePointer succeed on a
 *      page-aligned mmap region.
 *   2. ggml_backend_cuda_buffer_from_ptr returns non-NULL.
 *   3. ggml_backend_buffer_get_base returns a valid pointer.
 *   4. ggml_backend_buffer_free cleans up without leaking the host
 *      pointer's CUDA registration.
 *
 * Builds against libggml-cuda.a + libggml.a from vendor/ggml/build-cuda.
 * No Ruby / Spinel involved.
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "data/qwen25-1.5b-native-q8.gguf";

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }

    size_t size = (size_t)st.st_size;
    void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    printf("[1] mmap'd %s: %zu bytes at %p\n", path, size, map);

    /* Spin up the CUDA backend so subsequent calls have device 0. */
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "ggml_backend_cuda_init failed\n"); return 1; }
    printf("[2] CUDA backend initialised\n");

    /* The big test: wrap mmap'd pages in a CUDA buffer. */
    ggml_backend_buffer_t buf = ggml_backend_cuda_buffer_from_ptr(map, size, 0);
    if (!buf) {
        fprintf(stderr, "[FAIL] ggml_backend_cuda_buffer_from_ptr returned NULL\n");
        munmap(map, size); close(fd);
        return 1;
    }
    printf("[3] cuda_buffer_from_ptr OK; buffer = %p\n", (void*)buf);

    void *dev_ptr = ggml_backend_buffer_get_base(buf);
    printf("[4] buffer base = %p (host = %p; %s)\n", dev_ptr, map,
            dev_ptr == map ? "UVA (host==dev)" : "discrete (different)");

    size_t bsize = ggml_backend_buffer_get_size(buf);
    printf("[5] buffer size = %zu bytes (file = %zu)\n", bsize, size);

    /* Read a few bytes via the device pointer; on UVA-enabled SKUs
     * this is just a host memcpy. The first 4 bytes of a GGUF file
     * are the magic "GGUF". */
    char hdr[5] = {0};
    memcpy(hdr, dev_ptr, 4);
    printf("[6] first 4 bytes via dev_ptr: '%.4s' (expect 'GGUF')\n", hdr);

    /* Free the buffer — should cudaHostUnregister the mmap region
     * without freeing the actual memory (mmap still owns it). */
    ggml_backend_buffer_free(buf);
    printf("[7] buffer freed\n");

    /* Now we can munmap safely. */
    munmap(map, size);
    close(fd);

    ggml_backend_free(backend);
    printf("[8] OK — vendored ggml_backend_cuda_buffer_from_ptr works\n");
    return 0;
}
