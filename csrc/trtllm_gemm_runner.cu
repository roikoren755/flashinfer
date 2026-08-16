/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>

#include <algorithm>
#include <sstream>
#include <string>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/common.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/Enums.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/GemmInterface.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/trtllm/gen/SfLayoutDecl.h"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

namespace {
static thread_local gemm::gemm::GemmInterface::ModuleCache globalTrtllmGenGemmModuleCache;
}  // namespace

namespace flashinfer {

namespace {

// The trtllm-gen cubin manifest is a downloaded artifact, so which architectures
// it actually covers is not knowable at compile time. Encode only the
// cubin-arch -> SM-version compatibility rules here and let `config.mSm` decide
// what is available. Unknown cubin families are rejected so that a newly shipped
// one fails loudly instead of being mis-dispatched onto hardware that cannot run
// it (see #4107).
bool isArchCompatible(int smVersion, gemm::trtllm::gen::CudaArch cubinArch) {
  using CudaArch = gemm::trtllm::gen::CudaArch;
  switch (cubinArch) {
    case CudaArch::Sm100a:
      return smVersion == 100;
    case CudaArch::Sm100f:
      return smVersion == 100 || smVersion == 103;
    case CudaArch::Sm103a:
      return smVersion == 103;
#ifdef TLLM_RUBIN_FEATURES
    // CudaArch::Sm107a only exists in the Rubin cubin pin's generated headers,
    // which is also the only build that defines TLLM_RUBIN_FEATURES. sm107 is
    // unreachable in the non-Rubin module: get_trtllm_gemm_module() picks the
    // Rubin variant by device compute capability before this runs.
    case CudaArch::Sm107a:
      return smVersion == 107;
#endif
    default:
      return false;
  }
}

}  // namespace

enum class TrtllmGemmOperandLayout : int64_t {
  Standard = 0,
  ShuffledTransposed = 1,
  UnshuffledTransposed = 2,
};

struct TrtllmGenGemmRunnerOptions {
  gemm::trtllm::gen::Dtype eltType;
  gemm::trtllm::gen::Dtype outputType;
  gemm::gemm::EltwiseActType activationType;
  gemm::trtllm::gen::SfLayout sfLayoutB;
  TrtllmGemmOperandLayout operandLayout;
};

int64_t select_kernel_fp8(int32_t M, int32_t N, int32_t K,
                          const gemm::gemm::GemmInterface& interface) {
  static constexpr const char* KERNEL_NAME_HIGH_N_K_RATIO_SM100F =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s6_et64x8_m64x8x32_c1x1x1_rM_TN_"
      "transOut_noShfl_dsFp8_schPd2x2x1x3_sm100f";
  static constexpr const char* KERNEL_NAME_LOW_N_K_RATIO_SM100F =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_rM_TN_"
      "transOut_noShfl_dsFp8_schedS_sm100f";
  static constexpr const char* KERNEL_NAME_LARGE_N_SM100F =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_rM_TN_"
      "transOut_noShfl_dsFp8_schPd2x2x1x3_sm100f";
  static constexpr const char* KERNEL_NAME_DEFAULT_SM100F =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_TN_"
      "transOut_noShfl_dsFp8_schedS_sm100f";

  static constexpr const char* KERNEL_NAME_HIGH_N_K_RATIO_SM107A =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s6_et64x8_m64x8x32_c1x1x1_rM_TN_"
      "transOut_noShfl_dsFp8_schPd2x2x1x3_sm107a";
  static constexpr const char* KERNEL_NAME_LOW_N_K_RATIO_SM107A =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_rM_TN_"
      "transOut_noShfl_dsFp8_schedS_sm107a";
  static constexpr const char* KERNEL_NAME_LARGE_N_SM107A =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_rM_TN_"
      "transOut_noShfl_dsFp8_schPd2x2x1x3_sm107a";
  static constexpr const char* KERNEL_NAME_DEFAULT_SM107A =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_TN_"
      "transOut_noShfl_dsFp8_schedS_sm107a";

  bool const is_sm107 = getSMVersion() == 107;
  const char* const kHighNK =
      is_sm107 ? KERNEL_NAME_HIGH_N_K_RATIO_SM107A : KERNEL_NAME_HIGH_N_K_RATIO_SM100F;
  const char* const kLowNK =
      is_sm107 ? KERNEL_NAME_LOW_N_K_RATIO_SM107A : KERNEL_NAME_LOW_N_K_RATIO_SM100F;
  const char* const kLargeN = is_sm107 ? KERNEL_NAME_LARGE_N_SM107A : KERNEL_NAME_LARGE_N_SM100F;
  const char* const kDefault = is_sm107 ? KERNEL_NAME_DEFAULT_SM107A : KERNEL_NAME_DEFAULT_SM100F;

  double const n_k_ratio = static_cast<double>(N) / static_cast<double>(K);

  std::string kernel_name;
  if (n_k_ratio >= 32) {
    kernel_name = kHighNK;
  } else if (n_k_ratio <= 2.0) {
    kernel_name = kLowNK;
  } else if (N >= 20000) {
    kernel_name = kLargeN;
  } else {
    kernel_name = kDefault;
  }

  auto const& configs = interface.getGemmConfigs();
  size_t const num_configs = interface.getNumGemmConfigs();

  for (size_t i = 0; i < num_configs; ++i) {
    if (std::string(configs[i].mFunctionName) == kernel_name) {
      return static_cast<int64_t>(i);
    }
  }

  TVM_FFI_ICHECK(false) << "Kernel not found";
}

class TrtllmGenGemmRunner {
 public:
  explicit TrtllmGenGemmRunner(TrtllmGenGemmRunnerOptions const& options) : mOptions(options) {
    // Select a GEMM kernel config to use
    auto const gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    mPassingConfigIndices.clear();
    int const sv = getSMVersion();

    bool const transposeMmaOutput =
        mOptions.operandLayout != TrtllmGemmOperandLayout::Standard;
    bool const useShuffledMatrix =
        mOptions.operandLayout == TrtllmGemmOperandLayout::ShuffledTransposed;
    bool const useDeepSeekFp8 =
        mOptions.operandLayout == TrtllmGemmOperandLayout::UnshuffledTransposed;

    for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i) {
      auto const options = configs[i].mOptions;

      if (options.mDtypeA == mOptions.eltType && options.mDtypeB == mOptions.eltType &&
          options.mDtypeC == mOptions.outputType &&
          options.mEltwiseActType == mOptions.activationType &&
          options.mTransposeMmaOutput == transposeMmaOutput &&
          options.mUseShuffledMatrix == useShuffledMatrix &&
          options.mUseDeepSeekFp8 == useDeepSeekFp8 && options.mSfLayoutB == mOptions.sfLayoutB &&
          options.mLayoutA == gemm::gemm::MatrixLayout::MajorK) {  // FIXME(siyuanf): expose matrix layout to user
        if (mOptions.eltType == gemm::trtllm::gen::Dtype::E2m1 ||
            mOptions.eltType == gemm::trtllm::gen::Dtype::MxE4m3) {
          int32_t const blockSize =
              mOptions.eltType == gemm::trtllm::gen::Dtype::E2m1 ? 16 : 32;
          if (options.mSfLayoutA != gemm::trtllm::gen::SfLayout::R128c4 ||
              options.mSfBlockSizeA != blockSize || options.mSfBlockSizeB != blockSize) {
            continue;
          }
        }
        if (mOptions.outputType == gemm::trtllm::gen::Dtype::E2m1 &&
            (options.mSfLayoutC != gemm::trtllm::gen::SfLayout::R128c4 ||
             options.mSfBlockSizeC != 16 || options.mDtypeSfC != gemm::trtllm::gen::Dtype::E4m3)) {
          continue;
        }
        if (!isArchCompatible(sv, configs[i].mSm)) continue;
        mPassingConfigIndices.push_back(i);
      }
    }

    if (mPassingConfigIndices.empty()) {
      // Distinguish "this GPU has no cubins at all" from "no cubin matches these GEMM
      // options". The former is the common failure on unsupported hardware, and the
      // option dump below would send users looking in entirely the wrong place.
      bool anyArchCompatible = false;
      for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i) {
        if (isArchCompatible(sv, configs[i].mSm)) {
          anyArchCompatible = true;
          break;
        }
      }
      if (!anyArchCompatible) {
        std::ostringstream arch_msg;
        arch_msg << "The trtllm-gen GEMM cubin manifest contains no kernels runnable on sm" << sv
                 << "; this backend currently ships cubins for sm100, sm103 and sm107.";
        FLASHINFER_ERROR(arch_msg.str());
      }
    }

    FLASHINFER_CHECK(mPassingConfigIndices.size() > 0,
                     "No valid tactic found for the given options",
                     "mDtypeA: ", gemm::trtllm::gen::dtypeToString(mOptions.eltType),
                     "mDtypeC: ", gemm::trtllm::gen::dtypeToString(mOptions.outputType),
                     "mTransposeMmaOutput: ", transposeMmaOutput,
                     "mSfLayoutB: ", gemm::trtllm::gen::sfLayoutToString(mOptions.sfLayoutB),
                     "mEltwiseActType: ", static_cast<int64_t>(mOptions.activationType),
                     "operandLayout: ", static_cast<int64_t>(mOptions.operandLayout));
  }

  void checkPassingConfigIndex(int64_t tactic) const {
    auto it = std::find(mPassingConfigIndices.begin(), mPassingConfigIndices.end(), tactic);
    TVM_FFI_ICHECK(it != mPassingConfigIndices.end())
        << "Tactic " << tactic
        << " is not in this runner's compatible config set (device architecture or GEMM options "
           "mismatch)";
  }

  int64_t getWorkspaceSizeInBytes(int64_t m, int64_t n, int64_t k, int64_t tactic) {
    auto gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();
    FLASHINFER_CHECK(tactic >= 0 && tactic < gemm.getNumGemmConfigs(),
                     "Invalid tactic in getWorkspaceSizeInBytes");
    checkPassingConfigIndex(tactic);
    auto const config = configs[tactic];

    gemm::gemm::GemmData gemmData;
    bool const transposeMmaOutput =
        mOptions.operandLayout != TrtllmGemmOperandLayout::Standard;
    gemmData.mProblemDimensions.mM = transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
    gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
    gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    return gemm.getWorkspaceSizeInBytes(config, gemmData);
  }

  void run(int64_t m, int64_t n, int64_t k, void const* a, void const* aScale, void const* b,
           void const* bScale, void* c, void* cScale, void* cScalePtr, void* scaleAct,
           void* workspace, CUstream stream, int32_t device_index, int64_t tactic) {
    auto gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();
    TVM_FFI_ICHECK(tactic >= 0 && tactic < gemm.getNumGemmConfigs()) << "Invalid tactic id in run";
    checkPassingConfigIndex(tactic);
    auto const& config = configs[tactic];
    TVM_FFI_ICHECK(config.mOptions.mSfLayoutB == mOptions.sfLayoutB) << "Invalid sf layout in run";

    gemm::gemm::GemmData gemmData;
    bool const transposeMmaOutput =
        mOptions.operandLayout != TrtllmGemmOperandLayout::Standard;
    // Dims
    gemmData.mProblemDimensions.mM = transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
    gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
    gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
    gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
    gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;

    // Inputs
    gemmData.mInputBuffers.mPtrA = transposeMmaOutput ? b : a;
    gemmData.mInputBuffers.mPtrSfA = transposeMmaOutput ? bScale : aScale;
    gemmData.mInputBuffers.mPtrB = transposeMmaOutput ? a : b;
    gemmData.mInputBuffers.mPtrSfB = transposeMmaOutput ? aScale : bScale;
    gemmData.mInputBuffers.mPtrScaleC = cScale;
    gemmData.mInputBuffers.mPtrScaleAct = scaleAct;

    // Outputs
    gemmData.mOutputBuffers.mPtrC = c;
    gemmData.mOutputBuffers.mPtrSfC = cScalePtr;

    TVM_FFI_ICHECK(gemm.isValidConfig(config, gemmData)) << "unsupported tactic id in run";

    const int32_t multiProcessorCount = [device_index]() {
      static thread_local int32_t cached_multi_processor_count = -1;
      static thread_local int cached_device_index = -1;

      if (device_index == cached_device_index && cached_multi_processor_count != -1) {
        return cached_multi_processor_count;
      } else {
        int32_t count;
        cudaError_t cudaStatus =
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_index);
        TVM_FFI_ICHECK(cudaStatus == cudaSuccess)
            << "Failed to get device attribute: " << cudaGetErrorString(cudaStatus);
        cached_multi_processor_count = count;
        cached_device_index = device_index;
        return count;
      }
    }();

    TVM_FFI_ICHECK(gemm.run(config, workspace, gemmData, static_cast<void*>(stream),
                            multiProcessorCount, true, globalTrtllmGenGemmModuleCache) == 0)
        << "Error occurred when running GEMM!";
  }

  std::vector<int64_t> getValidTactics(int64_t m, int64_t n, int64_t k) const {
    auto const gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    gemm::gemm::GemmData gemmData;
    bool const transposeMmaOutput =
        mOptions.operandLayout != TrtllmGemmOperandLayout::Standard;
    // Dims
    gemmData.mProblemDimensions.mM = transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
    gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
    gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
    gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
    gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;

    std::vector<int64_t> sortedIndices = mPassingConfigIndices;
    std::sort(sortedIndices.begin(), sortedIndices.end(), [&configs](int64_t idx0, int64_t idx1) {
      auto const& optionsA = configs[idx0].mOptions;
      auto const& optionsB = configs[idx1].mOptions;

      // Sort by tileK sizes first
      if (optionsA.mTileK != optionsB.mTileK) {
        return optionsA.mTileK > optionsB.mTileK;
      }

      // Then by splitK sizes
      if (optionsA.mNumSlicesForSplitK != optionsB.mNumSlicesForSplitK) {
        return optionsA.mNumSlicesForSplitK > optionsB.mNumSlicesForSplitK;
      }

      // Then by unroll loop 2x for mma
      if (optionsA.mUseUnrollLoop2xForMma != optionsB.mUseUnrollLoop2xForMma) {
        return optionsA.mUseUnrollLoop2xForMma;
      }

      return false;
    });

    bool findLoop2xMma = false;
    std::vector<int64_t> validTactics;
    for (auto const& configIndex : sortedIndices) {
      auto const& config = configs[configIndex];
      if (gemm.isValidConfig(config, gemmData)) {
        validTactics.push_back(configIndex);

        // when loop2x mma is found, only add the tactic that has loop2x mma
        if (!findLoop2xMma) {
          if (config.mOptions.mUseUnrollLoop2xForMma) {
            findLoop2xMma = true;
          }
        } else {
          if (!config.mOptions.mUseUnrollLoop2xForMma) {
            break;
          }
        }
      }
    }
    return validTactics;
  }

  int64_t selectHeuristic(int64_t m, int64_t n, int64_t k) const {
    if (mOptions.operandLayout == TrtllmGemmOperandLayout::UnshuffledTransposed) {
      return select_kernel_fp8(m, n, k, gemm::gemm::GemmInterface());
    } else {
      auto sortedIndices = getValidTactics(m, n, k);
      TVM_FFI_ICHECK(!sortedIndices.empty()) << "No valid tactic found";

      // the getValidTactics is sorted by priority, so the first one is the best one
      return sortedIndices[0];
    }
  }

 private:
  TrtllmGenGemmRunnerOptions mOptions;
  std::vector<int64_t> mPassingConfigIndices;
};

using tvm::ffi::Array;
using tvm::ffi::Optional;

void trtllm_gemm(
    int64_t input_dtype_, int64_t output_dtype_, int64_t activation_type_,
    TensorView workspace_buffer, TensorView a, TensorView b, Optional<TensorView> a_scale,
    Optional<TensorView> b_scale,
    Optional<TensorView> pre_activation_scale, Optional<TensorView> accumulator_scale,
    Optional<TensorView> output_quant_scale, TensorView out, Optional<TensorView> out_scale,
    bool use_8x4_sf_layout, int64_t operand_layout_, int64_t tactic) {
  auto input_dtype = static_cast<gemm::trtllm::gen::Dtype>(input_dtype_);
  auto output_dtype = static_cast<gemm::trtllm::gen::Dtype>(output_dtype_);
  auto activation_type = static_cast<gemm::gemm::EltwiseActType>(activation_type_);
  auto operand_layout = static_cast<TrtllmGemmOperandLayout>(operand_layout_);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, out);
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(out);
  CHECK_INPUT(workspace_buffer);
  TVM_FFI_ICHECK_EQ(workspace_buffer.ndim(), 1);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  TVM_FFI_ICHECK_EQ(a.dtype(), b.dtype());
  TVM_FFI_ICHECK(a.dtype() == dl_float8_e4m3fn || a.dtype() == dl_uint8)
      << "a must be a Float8 or Byte(e2m1) tensor";
  bool is_fp8 = a.dtype() == dl_float8_e4m3fn;
  if (!is_fp8) {
    TVM_FFI_ICHECK(a_scale.has_value() && b_scale.has_value())
        << "E2M1 input requires a_scale and b_scale";
    CHECK_INPUT(a_scale.value());
    CHECK_INPUT(b_scale.value());
  }
  if (pre_activation_scale.has_value()) {
    CHECK_INPUT(pre_activation_scale.value());
  }
  if (accumulator_scale.has_value()) {
    CHECK_INPUT(accumulator_scale.value());
  }
  if (output_quant_scale.has_value()) {
    CHECK_INPUT(output_quant_scale.value());
  }
  if (out_scale.has_value()) {
    CHECK_DEVICE(a, out_scale.value());
    CHECK_INPUT(out_scale.value());
  }

  int32_t m = a.size(0);
  int32_t k = is_fp8 ? a.size(1) : a.size(1) * 2;
  int32_t n = b.size(0);
  TVM_FFI_ICHECK_EQ(b.size(1), a.size(1)) << "Matrix dimensions don't match for multiplication";
  if (output_dtype == gemm::trtllm::gen::Dtype::E2m1) {
    TVM_FFI_ICHECK_EQ(out.dtype(), dl_uint8);
    TVM_FFI_ICHECK(out.size(0) == m && out.size(1) * 2 == n)
        << "Output tensor has wrong dimensions";
    TVM_FFI_ICHECK(output_quant_scale.has_value() && out_scale.has_value())
        << "E2M1 output requires output_quant_scale and out_scale";
  } else {
    TVM_FFI_ICHECK(out.size(0) == m && out.size(1) == n)
        << "Output tensor has wrong dimensions";
    if (output_dtype == gemm::trtllm::gen::Dtype::E4m3) {
      TVM_FFI_ICHECK_EQ(out.dtype(), dl_float8_e4m3fn);
      TVM_FFI_ICHECK(output_quant_scale.has_value() && !out_scale.has_value())
          << "E4M3 output requires output_quant_scale and has no scale sidecar";
    }
  }

  auto runner = flashinfer::TrtllmGenGemmRunner(flashinfer::TrtllmGenGemmRunnerOptions{
      .eltType = input_dtype,
      .outputType = output_dtype,
      .activationType = activation_type,
      .sfLayoutB = use_8x4_sf_layout ? gemm::trtllm::gen::SfLayout::R8c4
                                     : gemm::trtllm::gen::SfLayout::R128c4,
      .operandLayout = operand_layout,
  });

  if (tactic == -1) {
    tactic = runner.selectHeuristic(m, n, k);
  }

  void* scaleAct =
      pre_activation_scale.has_value() ? pre_activation_scale.value().data_ptr() : nullptr;
  void* scaleC = accumulator_scale.has_value()
                     ? accumulator_scale.value().data_ptr()
                     : (output_quant_scale.has_value() ? output_quant_scale.value().data_ptr()
                                                       : nullptr);
  auto stream = get_stream(a.device());

  auto runKernel = [&](void* workspace) {
    runner.run(m, n, k, a.data_ptr(),
               a_scale.has_value() ? a_scale.value().data_ptr() : nullptr, b.data_ptr(),
               b_scale.has_value() ? b_scale.value().data_ptr() : nullptr, out.data_ptr(), scaleC,
               out_scale.has_value() ? out_scale.value().data_ptr() : nullptr, scaleAct, workspace,
               stream, a.device().device_id, tactic);
  };

  int64_t const required_workspace_size = runner.getWorkspaceSizeInBytes(m, n, k, tactic);
  int64_t const provided_workspace_size =
      workspace_buffer.numel() * get_element_size(workspace_buffer);
  if (provided_workspace_size < required_workspace_size) {
    Tensor new_workspace = alloc_tensor({required_workspace_size}, dl_int8, a.device());
    runKernel(new_workspace.data_ptr());
  } else {
    runKernel(workspace_buffer.data_ptr());
  }
}

Array<int64_t> trtllm_gemm_tactics(
    int64_t m, int64_t n, int64_t k, int64_t input_dtype_, int64_t output_dtype_,
    int64_t activation_type_, bool use_8x4_sf_layout, int64_t operand_layout_) {
  auto input_dtype = static_cast<gemm::trtllm::gen::Dtype>(input_dtype_);
  auto output_dtype = static_cast<gemm::trtllm::gen::Dtype>(output_dtype_);
  auto activation_type = static_cast<gemm::gemm::EltwiseActType>(activation_type_);
  auto operand_layout = static_cast<TrtllmGemmOperandLayout>(operand_layout_);

  TVM_FFI_CHECK(input_dtype == gemm::trtllm::gen::Dtype::E4m3 ||
                    input_dtype == gemm::trtllm::gen::Dtype::MxE4m3 ||
                    input_dtype == gemm::trtllm::gen::Dtype::E2m1,
                "Unsupported input dtype");
  TVM_FFI_CHECK(output_dtype == gemm::trtllm::gen::Dtype::Bfloat16 ||
                    output_dtype == gemm::trtllm::gen::Dtype::E4m3 ||
                    output_dtype == gemm::trtllm::gen::Dtype::E2m1,
                "Unsupported output dtype");
  TVM_FFI_CHECK(activation_type == gemm::gemm::EltwiseActType::None ||
                    activation_type == gemm::gemm::EltwiseActType::Relu2,
                "Unsupported activation type");
  TVM_FFI_CHECK(operand_layout == TrtllmGemmOperandLayout::Standard ||
                    operand_layout == TrtllmGemmOperandLayout::ShuffledTransposed ||
                    operand_layout == TrtllmGemmOperandLayout::UnshuffledTransposed,
                "Unsupported operand layout");

  auto runner = flashinfer::TrtllmGenGemmRunner(flashinfer::TrtllmGenGemmRunnerOptions{
      .eltType = input_dtype,
      .outputType = output_dtype,
      .activationType = activation_type,
      .sfLayoutB = use_8x4_sf_layout ? gemm::trtllm::gen::SfLayout::R8c4
                                     : gemm::trtllm::gen::SfLayout::R128c4,
      .operandLayout = operand_layout,
  });

  return runner.getValidTactics(m, n, k);
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_gemm, flashinfer::trtllm_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_gemm_tactics, flashinfer::trtllm_gemm_tactics);
