#pragma once

// Specter-owned host-side launch metadata. CUDA function attributes belong to
// a device and kernel, not to a stream. Never lower a previously granted limit.
// CUDA context reset/unload while this extension is live is not supported.
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace specter {
struct DeviceLimits {
  int optin_shared_memory;
  int multiprocessors;
};
struct KernelKey {
  int device;
  const void* kernel;
  bool operator==(const KernelKey& other) const {
    return device == other.device && kernel == other.kernel;
  }
};
struct KernelKeyHash {
  size_t operator()(const KernelKey& key) const {
    return std::hash<const void*>{}(key.kernel) ^
           (std::hash<int>{}(key.device) << 1);
  }
};
struct LaunchMetadata {
  std::mutex mutex;
  std::unordered_map<int, DeviceLimits> devices;
  std::unordered_map<KernelKey, int, KernelKeyHash> capacities;
  std::unordered_map<KernelKey, int, KernelKeyHash> register_counts;
  // Cold-path diagnostics only; no counter increments on the launch hot path.
  uint64_t device_queries = 0;
  uint64_t function_queries = 0;
  uint64_t attribute_sets = 0;
};
inline LaunchMetadata& launch_metadata() {
  static LaunchMetadata metadata;
  return metadata;
}
inline DeviceLimits device_limits(int device) {
  thread_local std::unordered_map<int, DeviceLimits> local;
  auto hit = local.find(device);
  if (hit != local.end()) return hit->second;
  auto& metadata = launch_metadata();
  std::lock_guard<std::mutex> lock(metadata.mutex);
  auto global = metadata.devices.find(device);
  if (global == metadata.devices.end()) {
    DeviceLimits limits;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &limits.optin_shared_memory,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &limits.multiprocessors, cudaDevAttrMultiProcessorCount, device));
    metadata.device_queries += 2;
    global = metadata.devices.emplace(device, limits).first;
  }
  local.emplace(device, global->second);
  return global->second;
}
inline int kernel_register_count(int device, const void* kernel) {
  const KernelKey key{device, kernel};
  thread_local std::unordered_map<KernelKey, int, KernelKeyHash> local;
  auto hit = local.find(key);
  if (hit != local.end()) return hit->second;
  auto& metadata = launch_metadata();
  std::lock_guard<std::mutex> lock(metadata.mutex);
  auto global = metadata.register_counts.find(key);
  if (global == metadata.register_counts.end()) {
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    ++metadata.function_queries;
    global = metadata.register_counts.emplace(key, attributes.numRegs).first;
    metadata.capacities.emplace(key, attributes.maxDynamicSharedSizeBytes);
  }
  local.emplace(key, global->second);
  return global->second;
}
inline void ensure_dynamic_shared_memory(int device, const void* kernel,
                                         int requested_bytes) {
  const KernelKey key{device, kernel};
  thread_local std::unordered_map<KernelKey, int, KernelKeyHash> local;
  auto hit = local.find(key);
  if (hit != local.end() && hit->second >= requested_bytes) return;

  // First use per thread or a larger request. Serialize only this cold path:
  // another host thread may already have raised the kernel's capacity.
  auto& metadata = launch_metadata();
  std::lock_guard<std::mutex> lock(metadata.mutex);
  auto global = metadata.capacities.find(key);
  if (global == metadata.capacities.end() || global->second < requested_bytes) {
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    if (global == metadata.capacities.end()) {
      cudaFuncAttributes attributes;
      C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
      ++metadata.function_queries;
      metadata.register_counts.emplace(key, attributes.numRegs);
      // The function-specific default already accounts for static shared mem.
      // Small alignment kernels therefore need no opt-in attribute write.
      global = metadata.capacities.emplace(
          key, attributes.maxDynamicSharedSizeBytes).first;
    }
    if (global->second < requested_bytes) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, requested_bytes));
      ++metadata.attribute_sets;
      global->second = requested_bytes;
    }
  }
  local[key] = global->second;
}
inline std::array<uint64_t, 3> shared_memory_cache_stats() {
  auto& metadata = launch_metadata();
  std::lock_guard<std::mutex> lock(metadata.mutex);
  return {metadata.device_queries, metadata.function_queries,
          metadata.attribute_sets};
}
}  // namespace specter
