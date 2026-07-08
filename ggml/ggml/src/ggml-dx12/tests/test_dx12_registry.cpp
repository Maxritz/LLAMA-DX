#include <cstdio>
#include <cstring>

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-dx12.h"

static int n_pass = 0;
static int n_fail = 0;

#define TEST(name) do { printf("  %-48s", name); } while(0)
#define PASS() do { printf("PASS\n"); n_pass++; } while(0)
#define FAIL(msg) do { printf("FAIL (%s)\n", msg); n_fail++; } while(0)

int main() {
    printf("\n=== DX12 Three-Tier Hierarchy Tests ===\n\n");

    /* Tier 1: Registration */
    printf("-- Tier 1: ggml_backend_reg --\n");
    ggml_backend_reg_t reg = ggml_backend_dx12_reg();

    TEST("reg_get_name");
    const char* reg_name = reg->iface.get_name(reg);
    printf("  name=%s\n", reg_name);
    if (reg_name && strcmp(reg_name, "DX12") == 0) PASS(); else FAIL("wrong reg name");

    TEST("reg_api_version");
    printf("  api=%d\n", reg->api_version);
    if (reg->api_version == GGML_BACKEND_API_VERSION) PASS(); else FAIL("wrong api version");

    TEST("reg_get_device_count");
    size_t dev_count = reg->iface.get_device_count(reg);
    printf("  count=%zu\n", dev_count);
    if (dev_count > 0) PASS(); else FAIL("no devices");

    /* Tier 2: Device */
    printf("\n-- Tier 2: ggml_backend_device --\n");
    if (dev_count == 0) {
        printf("  No devices to test, skipping Tier 2+3\n");
        goto done;
    }

    ggml_backend_dev_t dev = reg->iface.get_device(reg, 0);

    TEST("dev_reg_pointer");
    if (dev->reg == reg) PASS(); else FAIL("dev.reg != reg");

    TEST("dev_get_name");
    const char* dev_name = dev->iface.get_name(dev);
    printf("  name=%s\n", dev_name ? dev_name : "(null)");
    if (dev_name && strlen(dev_name) > 0) PASS(); else FAIL("empty name");

    TEST("dev_get_description");
    const char* dev_desc = dev->iface.get_description(dev);
    printf("  desc=%s\n", dev_desc ? dev_desc : "(null)");
    if (dev_desc && strlen(dev_desc) > 0) PASS(); else FAIL("empty desc");

    TEST("dev_get_memory");
    size_t mem_free = 0, mem_total = 0;
    dev->iface.get_memory(dev, &mem_free, &mem_total);
    printf("  total=%zu MB, free=%zu MB\n", mem_total/(1024*1024), mem_free/(1024*1024));
    if (mem_total > 0) PASS(); else FAIL("VRAM=0");

    TEST("dev_get_type");
    int dev_type = (int)dev->iface.get_type(dev);
    printf("  type=%d (1=GPU)\n", dev_type);
    if (dev_type == 1) PASS(); else FAIL("not GPU");

    TEST("dev_supports_op");
    if (dev->iface.supports_op) PASS(); else FAIL("null ptr");

    TEST("dev_supports_buft");
    if (dev->iface.supports_buft) PASS(); else FAIL("null ptr");

    TEST("dev_offload_op");
    if (dev->iface.offload_op) PASS(); else FAIL("null ptr");

    TEST("dev_get_buffer_type");
    ggml_backend_buffer_type_t buft = dev->iface.get_buffer_type(dev);
    printf("  buft=%p\n", (void*)buft);
    if (buft) PASS(); else FAIL("no buffer type");

    if (buft) {
        TEST("buft_get_name");
        const char* buft_name = buft->iface.get_name(buft);
        printf("  name=%s\n", buft_name ? buft_name : "(null)");
        if (buft_name && strcmp(buft_name, "DX12") == 0) PASS(); else FAIL("wrong name");

        TEST("buft_get_alignment");
        size_t align = buft->iface.get_alignment(buft);
        printf("  align=%zu\n", align);
        if (align >= 256) PASS(); else FAIL("alignment too small");

        TEST("buft_is_host");
        bool is_host = buft->iface.is_host(buft);
        printf("  is_host=%d\n", is_host);
        if (!is_host) PASS(); else FAIL("should not be host");
    }

    /* Tier 3: Backend (create via device init) */
    printf("\n-- Tier 3: ggml_backend --\n");

    TEST("dev_init_backend");
    ggml_backend_t be = dev->iface.init_backend(dev, nullptr);
    if (be) PASS(); else { FAIL("init returned null"); goto done; }

    TEST("backend_device_pointer");
    if (be->device == dev) PASS(); else FAIL("be.device != dev");

    TEST("backend_get_name");
    const char* be_name = be->iface.get_name(be);
    printf("  name=%s\n", be_name ? be_name : "(null)");
    if (be_name && strstr(be_name, "DX12") != nullptr) PASS(); else FAIL("wrong name");

    TEST("backend_synchronize");
    if (be->iface.synchronize) {
        be->iface.synchronize(be);
        PASS();
    } else { FAIL("null ptr"); }

    TEST("backend_graph_compute");
    if (be->iface.graph_compute) PASS(); else FAIL("null ptr");

    TEST("backend_free");
    be->iface.free(be);
    PASS();

done:
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
