
#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "src/model/resnet18_model_data.h"
#include "src/model/resnet18_ops_resolver.h"
#include "src/model/resnet18_test_image.h"

namespace {

constexpr std::size_t kTensorArenaSize = 2 * 1024 * 1024;

alignas(16) std::uint8_t tensor_arena[kTensorArenaSize];

}  // namespace

int main() {
  MicroPrintf("=== ResNet-18 Inference ===");

  // Load and validate model.
  const tflite::Model* model =
      tflite::GetModel(g_resnet18_model_data);

  if (model == nullptr ||
      model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("ERROR: Invalid model or schema version");
    return 1;
  }

  // Register only the operators used by this model.
  ResNet18OpResolver resolver;

  if (RegisterResNet18Ops(resolver) != kTfLiteOk) {
    MicroPrintf("ERROR: Operator registration failed");
    return 1;
  }

  // Standard interpreter; no recording/profiling allocator.
  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("ERROR: AllocateTensors failed");
    return 1;
  }

  MicroPrintf("AllocateTensors: PASS");

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  if (input == nullptr || output == nullptr) {
    MicroPrintf("ERROR: Missing input/output tensor");
    return 1;
  }

  MicroPrintf("Input tensor bytes: %d",
              static_cast<int>(input->bytes));
  MicroPrintf("Input scale x1e6: %d",
              static_cast<int>(input->params.scale * 1000000.0f));
  MicroPrintf("Input zero point: %d",
              input->params.zero_point);

  MicroPrintf("Output tensor bytes: %d",
              static_cast<int>(output->bytes));
  MicroPrintf("Output scale x1e6: %d",
              static_cast<int>(output->params.scale * 1000000.0f));
  MicroPrintf("Output zero point: %d",
              output->params.zero_point);


  if (input == nullptr ||
      input->type != kTfLiteInt8 ||
      input->data.int8 == nullptr ||
      input->bytes != g_resnet18_test_image_data_len) {
    MicroPrintf("ERROR: Input tensor mismatch");
    return 1;
  }

  for (std::size_t i = 0; i < input->bytes; ++i) {
    input->data.int8[i] = g_resnet18_test_image_data[i];
  }

  // Run the model.
  MicroPrintf("Starting Invoke()...");

  if (interpreter.Invoke() != kTfLiteOk) {
    MicroPrintf("ERROR: Invoke failed");
    return 1;
  }

  MicroPrintf("Invoke: PASS");

  if (output == nullptr ||
      output->type != kTfLiteInt8 ||
      output->data.int8 == nullptr ||
      output->bytes == 0) {
    MicroPrintf("ERROR: Invalid output tensor");
    return 1;
  }

  std::size_t best_index = 0;

  for (std::size_t i = 1; i < output->bytes; ++i) {
    if (output->data.int8[i] > output->data.int8[best_index]) {
      best_index = i;
    }
  }

  MicroPrintf("Output elements: %d",
              static_cast<int>(output->bytes));
  MicroPrintf("Top-1 class index: %d",
              static_cast<int>(best_index));
  MicroPrintf("Top-1 quantized score: %d",
              static_cast<int>(output->data.int8[best_index]));

  // Dump the complete INT8 output vector for host-side validation.
  // Format: OUTPUT_INT8 <index> <signed_value>
  for (std::size_t i = 0; i < output->bytes; ++i) {
    MicroPrintf("OUTPUT_INT8 %d %d",
                static_cast<int>(i),
                static_cast<int>(output->data.int8[i]));
  }


  MicroPrintf("=== Inference completed successfully ===");
  return 0;
}


/*

export TFLM_ROOT="$HOME/Desktop/EEproject/workspace/tflite-micro"
export TOOLCHAIN_BIN="$HOME/Desktop/EEproject/workspace/NN-accelerator/src/lib/gnu-toolchain/bin"

"$TOOLCHAIN_BIN/riscv64-unknown-elf-g++" \
  -O2 -std=c++17 \
  -march=rv64imc -mabi=lp64 \
  -I. \
  -I"$TFLM_ROOT" \
  -I"$TFLM_ROOT/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include" \
  -I"$TFLM_ROOT/tensorflow/lite/micro/tools/make/downloads/gemmlowp" \
  src/tools/resnet18_b0_inference.cc \
  src/tools/debug_log.cc \
  -Wl,--allow-multiple-definition \
  src/lib/TFLM/riscv64_generic_x86_64_default_gcc/lib/libtensorflow-microlite.a \
  -o src/tools/resnet18_inference.elf


spike pk src/tools/resnet18_inference.elf 2>&1 | tee results/resnet18_inference_spike_output.txt 

*/
