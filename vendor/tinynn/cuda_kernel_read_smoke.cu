/* tinynn/cuda_kernel_read_smoke.cu — does a CUDA kernel actually
 * read correct bytes via a UVA-mapped file-backed mmap region?
 *
 * The earlier C smoke test (cuda_byo_smoke.c) only did host-side
 * memcpy through dev_ptr. That always works on UVA because the host
 * pointer is valid in host context.
 *
 * Here we launch an actual kernel that reads via dev_ptr and writes
 * to a device-side output buffer, then copy the output back to host
 * for comparison. If the kernel-read result matches what's in the
 * file, BYO-pointer kernel reads work on this SKU. If not, we've
 * confirmed the bug.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    return 1; } } while (0)

// Kernel: read N bytes starting at src + offset, copy to dst.
__global__ void copy_bytes(const unsigned char *src, unsigned char *dst, size_t offset, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[offset + i];
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "data/qwen25-1.5b-native.gguf";

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t size = (size_t)st.st_size;

    void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    printf("[1] mmap'd %s: %zu bytes at host_ptr=%p\n", path, size, map);

    // Touch first 64 bytes on host to force pages in.
    unsigned char host_first64[64];
    memcpy(host_first64, map, 64);
    printf("[2] host memcpy: first 4 bytes = '%.4s' (expect 'GGUF')\n",
            (char *)host_first64);

    // Register + get device pointer.
    cudaError_t err = cudaHostRegister(map, size, cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "[3] cudaHostRegister: %s\n", cudaGetErrorString(err));
        return 1;
    }
    void *dev_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, map, 0));
    printf("[3] cudaHostRegister OK; dev_ptr=%p (UVA: %s host)\n",
            dev_ptr, dev_ptr == map ? "==" : "!=");

    // Allocate a device buffer for the kernel to write into.
    unsigned char *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, 64));

    // Launch kernel: read first 64 bytes via dev_ptr, write to d_out.
    copy_bytes<<<1, 64>>>((const unsigned char *)dev_ptr, d_out, 0, 64);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back to host.
    unsigned char kernel_first64[64];
    CUDA_CHECK(cudaMemcpy(kernel_first64, d_out, 64, cudaMemcpyDeviceToHost));
    printf("[4] kernel-read: first 4 bytes = '%.4s' (expect 'GGUF')\n",
            (char *)kernel_first64);

    // Byte-for-byte compare.
    int diffs = 0;
    for (int i = 0; i < 64; i++) {
        if (kernel_first64[i] != host_first64[i]) diffs++;
    }
    printf("[5] byte-diff in first 64: %d (expect 0)\n", diffs);

    if (diffs == 0) {
        printf("[6] OK — kernel-side UVA reads work\n");
    } else {
        printf("[6] FAIL — kernel reads return different bytes than host\n");
        printf("    First mismatch in detail:\n");
        for (int i = 0; i < 64; i++) {
            if (kernel_first64[i] != host_first64[i]) {
                printf("      byte %d: host=0x%02x kernel=0x%02x\n",
                        i, host_first64[i], kernel_first64[i]);
                break;
            }
        }
    }

    // Also probe a "deep" offset — a real weight tensor. For
    // qwen25-1.5b-native.gguf, the GGUF data section starts somewhere
    // after the header; tensor offsets are GB-scale. Pick a known
    // offset in the middle.
    if (size > 1000000) {
        size_t off = size / 2;   // middle of the file
        unsigned char host_mid[64];
        memcpy(host_mid, (char *)map + off, 64);

        copy_bytes<<<1, 64>>>((const unsigned char *)dev_ptr, d_out, off, 64);
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned char kernel_mid[64];
        CUDA_CHECK(cudaMemcpy(kernel_mid, d_out, 64, cudaMemcpyDeviceToHost));

        int mid_diffs = 0;
        for (int i = 0; i < 64; i++) {
            if (kernel_mid[i] != host_mid[i]) mid_diffs++;
        }
        printf("[7] mid-file offset %zu: byte-diff = %d (expect 0)\n",
                off, mid_diffs);
        if (mid_diffs > 0) {
            for (int i = 0; i < 64; i++) {
                if (kernel_mid[i] != host_mid[i]) {
                    printf("      byte %d: host=0x%02x kernel=0x%02x\n",
                            i, host_mid[i], kernel_mid[i]);
                    break;
                }
            }
        }
    }

    cudaFree(d_out);
    cudaHostUnregister(map);
    munmap(map, size);
    close(fd);
    return 0;
}
