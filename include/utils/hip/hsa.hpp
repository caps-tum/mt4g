#pragma once

#include <optional>
#include <cstdint>
#include <string>
#include <fstream>
#include <filesystem>
#include <hip/hip_runtime.h>
#ifdef __HIP_PLATFORM_AMD__
#include <rocm_smi/rocm_smi.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

#ifdef __HIP_PLATFORM_AMD__
inline hsa_agent_t getCurrentHsaAgent() {
    static hsa_agent_t cached = []() {
        hsa_agent_t result{};
        int dev = 0;
        util::hipCheck(hipGetDevice(&dev));
        hipDeviceProp_t prop{};
        util::hipCheck(hipGetDeviceProperties(&prop, dev));
        const uint32_t want_domain = static_cast<uint32_t>(prop.pciDomainID);
        const int want_bus = prop.pciBusID;
        const int want_dev = prop.pciDeviceID;
        if (hsa_init() != HSA_STATUS_SUCCESS) return result;
        auto cb_bdf = +[](hsa_agent_t a, void* data)->hsa_status_t {
            uint64_t* d = reinterpret_cast<uint64_t*>(data);
            const uint32_t want_domain = static_cast<uint32_t>(d[0]);
            const int want_bus = static_cast<int>(d[1]);
            const int want_dev = static_cast<int>(d[2]);
            hsa_device_type_t type{};
            if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS ||
                type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
            uint32_t dom = 0, bdf = 0;
            if (hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &dom) != HSA_STATUS_SUCCESS)
                return HSA_STATUS_SUCCESS;
            if (hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf) != HSA_STATUS_SUCCESS)
                return HSA_STATUS_SUCCESS;
            const int bus = (bdf >> 8) & 0xFF;
            const int dev = (bdf >> 3) & 0x1F;
            if (dom == want_domain && bus == want_bus && dev == want_dev) {
                *reinterpret_cast<hsa_agent_t*>(&d[3]) = a;
                d[5] = 1;
                return HSA_STATUS_INFO_BREAK;
            }
            return HSA_STATUS_SUCCESS;
        };
        uint64_t data_bdf[6] = { want_domain, (uint64_t)want_bus, (uint64_t)want_dev, 0, 0, 0 };
        hsa_status_t st = hsa_iterate_agents(cb_bdf, data_bdf);
        if (st == HSA_STATUS_INFO_BREAK) st = HSA_STATUS_SUCCESS;
        if (st == HSA_STATUS_SUCCESS && data_bdf[5] == 1) {
            result = *reinterpret_cast<hsa_agent_t*>(&data_bdf[3]);
            return result;
        }
        auto cb_ord = +[](hsa_agent_t a, void* data)->hsa_status_t {
            int* d = reinterpret_cast<int*>(data);
            hsa_device_type_t type{};
            if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS &&
                type == HSA_DEVICE_TYPE_GPU) {
                if (d[1] == d[0]) { *reinterpret_cast<hsa_agent_t*>(&d[3]) = a; return HSA_STATUS_INFO_BREAK; }
                ++d[1];
            }
            return HSA_STATUS_SUCCESS;
        };
        int data_ord[5] = { dev, 0, 0, 0, 0 };
        st = hsa_iterate_agents(cb_ord, data_ord);
        if (st == HSA_STATUS_INFO_BREAK) st = HSA_STATUS_SUCCESS;
        if (st == HSA_STATUS_SUCCESS) {
            result = *reinterpret_cast<hsa_agent_t*>(&data_ord[3]);
        }
        return result;
    }();
    return cached;
}

static inline uint32_t queryCacheLevelBytes(hsa_agent_t agent, uint32_t level) {
    uint32_t ctx[2] = { level, 0 };
    auto cb = +[](hsa_cache_t c, void* data)->hsa_status_t {
        uint32_t* d = reinterpret_cast<uint32_t*>(data);
        uint32_t lvl=0, sz=0;
        if (hsa_cache_get_info(c, HSA_CACHE_INFO_LEVEL, &lvl) != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;
        if (hsa_cache_get_info(c, HSA_CACHE_INFO_SIZE,  &sz)  != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;
        if (lvl == d[0] && sz > d[1]) d[1] = sz;
        return HSA_STATUS_SUCCESS;
    };
    hsa_status_t st = hsa_agent_iterate_caches(agent, cb, ctx);
    (void)st;
    return ctx[1];
}

inline std::optional<size_t> getKfdCachelineBytesForLevel(uint32_t level) {
    if (level < 1 || level > 4) return std::nullopt;
    hsa_agent_t agent = getCurrentHsaAgent();
    if (!agent.handle) return std::nullopt;
    uint32_t node = 0;
    if (hsa_agent_get_info(agent,
          (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_NODE_ID, &node) != HSA_STATUS_SUCCESS)
        return std::nullopt;
    namespace fs = std::filesystem;
    const std::string base = "/sys/class/kfd/kfd/topology/nodes/" + std::to_string(node) + "/caches";
    std::error_code ec;
    if (!fs::exists(base, ec)) return std::nullopt;
    size_t best = 0;
    for (const auto& d : fs::directory_iterator(base, ec)) {
        if (ec) return std::nullopt;
        const fs::path prop = d.path() / "properties";
        if (!fs::is_regular_file(prop)) continue;
        std::ifstream f(prop);
        if (!f) continue;
        int lvl = -1;
        long long cls = -1;
        std::string key; std::string val;
        while (f >> key >> val) {
            if (key == "level")            lvl = std::stoi(val);
            else if (key == "cache_line_size") cls = std::stoll(val);
        }
        if (lvl == static_cast<int>(level) && cls > 0) {
            if (static_cast<size_t>(cls) > best) best = static_cast<size_t>(cls);
        }
    }
    if (best == 0) return std::nullopt;
    return best;
}
#endif

