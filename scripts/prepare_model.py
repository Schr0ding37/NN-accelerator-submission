
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "src/model/resnet18_int8_channelwise.tflite"
)
IMAGE_PATH = os.path.join(PROJECT_ROOT, "src/img/dog.jpg")

MODEL_H_PATH = os.path.join(
    PROJECT_ROOT, "src/model/resnet18_model_data.h"
)
IMAGE_H_PATH = os.path.join(
    PROJECT_ROOT, "src/model/resnet18_test_image.h"
)
OPS_H_PATH = os.path.join(
    PROJECT_ROOT, "src/model/resnet18_ops_resolver.h"
)

# This model uses the standard ImageNet preprocessing associated
# with the PyTorch Vision ResNet-18 weights.
IMAGE_SIZE = 224
RESIZE_SHORT_SIDE = 256

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
).reshape(1, 1, 3)

OP_MAPPING = {
    "ADD": "AddAdd",
    "AVERAGE_POOL_2D": "AddAveragePool2D",
    "CONV_2D": "AddConv2D",
    "FULLY_CONNECTED": "AddFullyConnected",
    "MAX_POOL_2D": "AddMaxPool2D",
    "PAD": "AddPad",
    "TRANSPOSE": "AddTranspose",
}


def preprocess_image(image_path, input_scale, input_zero_point):
    """Create a quantized INT8 NCHW tensor for this ResNet-18 model."""

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    width, height = image.size

    # Resize while preserving aspect ratio: the shorter side becomes 256.
    if width < height:
        resized_width = RESIZE_SHORT_SIDE
        resized_height = int(
            round(height * RESIZE_SHORT_SIDE / width)
        )
    else:
        resized_height = RESIZE_SHORT_SIDE
        resized_width = int(
            round(width * RESIZE_SHORT_SIDE / height)
        )

    image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    # Center crop to 224 x 224.
    left = (resized_width - IMAGE_SIZE) // 2
    top = (resized_height - IMAGE_SIZE) // 2

    image = image.crop(
        (left, top, left + IMAGE_SIZE, top + IMAGE_SIZE)
    )

    # RGB uint8 -> float32 [0, 1].
    x = np.asarray(image, dtype=np.float32) / 255.0

    # ImageNet normalization.
    x = (x - IMAGENET_MEAN) / IMAGENET_STD

    # Quantize real values using the actual TFLite input parameters.
    x = np.rint(x / input_scale + input_zero_point)
    x = np.clip(x, -128, 127).astype(np.int8)

    # HWC -> CHW, then add batch dimension: [1, 3, 224, 224].
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)

    return np.ascontiguousarray(x)


def write_byte_array_header(
    path, guard, variable_name, data, c_type
):
    """Write a C++ header containing a constant array."""

    flat = np.asarray(data).reshape(-1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write("#include <cstdint>\n\n")

        f.write(
            f"alignas(16) const {c_type} {variable_name}[] = {{\n"
        )

        for start in range(0, len(flat), 12):
            chunk = flat[start:start + 12]

            if c_type == "uint8_t":
                values = ", ".join(
                    f"0x{int(v):02x}" for v in chunk
                )
            else:
                values = ", ".join(
                    str(int(v)) for v in chunk
                )

            f.write(f"  {values},\n")

        f.write("};\n")
        f.write(
            f"const unsigned int {variable_name}_len = "
            f"{len(flat)};\n\n"
        )
        f.write(f"#endif  // {guard}\n")


def write_model_header():
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()

    write_byte_array_header(
        MODEL_H_PATH,
        "RESNET18_MODEL_DATA_H_",
        "g_resnet18_model_data",
        np.frombuffer(model_bytes, dtype=np.uint8),
        "uint8_t",
    )

    print(
        f"[OK] Model header: {MODEL_H_PATH} "
        f"({len(model_bytes)} bytes)"
    )


def write_resolver_header(unique_ops):
    unsupported = [
        op for op in unique_ops if op not in OP_MAPPING
    ]

    if unsupported:
        raise RuntimeError(
            "No TFLM resolver mapping for: "
            + ", ".join(unsupported)
        )

    registrations = [
        OP_MAPPING[op] for op in unique_ops
    ]

    with open(OPS_H_PATH, "w", encoding="utf-8") as f:
        f.write("#ifndef RESNET18_OPS_RESOLVER_H_\n")
        f.write("#define RESNET18_OPS_RESOLVER_H_\n\n")
        f.write(
            '#include "tensorflow/lite/micro/'
            'micro_mutable_op_resolver.h"\n'
        )
        f.write(
            '#include "tensorflow/lite/micro/c/common.h"\n\n'
        )

        f.write(
            "using ResNet18OpResolver = "
            f"tflite::MicroMutableOpResolver<{len(registrations)}>;\n\n"
        )

        f.write(
            "inline TfLiteStatus RegisterResNet18Ops("
            "ResNet18OpResolver& resolver) {\n"
        )

        for registration in registrations:
            f.write(
                f"  TF_LITE_ENSURE_STATUS("
                f"resolver.{registration}());\n"
            )

        f.write("  return kTfLiteOk;\n")
        f.write("}\n\n")
        f.write("#endif  // RESNET18_OPS_RESOLVER_H_\n")

    print(
        f"[OK] Resolver header: {OPS_H_PATH} "
        f"({len(registrations)} operators)"
    )


def main():
    if not os.path.isfile(MODEL_PATH):
        sys.exit(f"ERROR: Model not found: {MODEL_PATH}")

    if not os.path.isfile(IMAGE_PATH):
        sys.exit(f"ERROR: Image not found: {IMAGE_PATH}")

    # Disable delegates: inspect the actual model operators.
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[],
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    expected_shape = [1, 3, IMAGE_SIZE, IMAGE_SIZE]

    input_shape = input_details["shape"].tolist()
    if input_shape != expected_shape:
        sys.exit(
            f"ERROR: Expected input shape {expected_shape}, "
            f"got {input_shape}"
        )

    if input_details["dtype"] != np.int8:
        sys.exit(
            f"ERROR: Expected INT8 input, "
            f"got {input_details['dtype']}"
        )

    input_scale, input_zero_point = input_details["quantization"]

    if input_scale <= 0:
        sys.exit("ERROR: Invalid input quantization scale")

    if output_details["dtype"] != np.int8:
        sys.exit(
            f"ERROR: Expected INT8 output, "
            f"got {output_details['dtype']}"
        )

    print("=== ResNet-18 INT8 Model ===")
    print(f"Input shape: {input_shape}")
    print(f"Input dtype: {input_details['dtype']}")
    print(
        f"Input quantization: scale={input_scale}, "
        f"zero_point={input_zero_point}"
    )
    print(
        f"Output shape: {output_details['shape'].tolist()}"
    )
    print(f"Output dtype: {output_details['dtype']}")

    # Inspect builtin operators without delegates.
    ops_details = interpreter._get_ops_details()
    unique_ops = sorted({
        op["op_name"]
        for op in ops_details
        if op["op_name"] != "DELEGATE"
    })

    print(f"Operators ({len(unique_ops)}): {unique_ops}")

    # Fail rather than silently omit an unsupported operator.
    write_resolver_header(unique_ops)
    write_model_header()

    # Generate correctly preprocessed, quantized input data.
    image_tensor = preprocess_image(
        IMAGE_PATH,
        input_scale,
        input_zero_point,
    )

    if list(image_tensor.shape) != expected_shape:
        sys.exit(
            f"ERROR: Generated image shape is {image_tensor.shape}"
        )

    write_byte_array_header(
        IMAGE_H_PATH,
        "RESNET18_TEST_IMAGE_H_",
        "g_resnet18_test_image_data",
        image_tensor,
        "int8_t",
    )

    print(f"[OK] Test image header: {IMAGE_H_PATH}")
    print(f"[OK] Input tensor bytes: {image_tensor.nbytes}")
    print("=== Preparation completed ===")


if __name__ == "__main__":
    main()


