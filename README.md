# Quantized Neural-Network Acceleration on RV64IMC

## 1. Project Overview

This project explores architectural extensions for accelerating INT8 quantized neural-network inference on a RISC-V RV64IMC processor.

The target workload is quantized ResNet-18 using TensorFlow Lite for Microcontrollers (TFLM).

Spike is used for ISA-level functional evaluation and dynamic instruction analysis. CV-Wally is intended for subsequent RTL-level hardware evaluation.

The project investigates the effectiveness of a custom packed INT8 multiply-accumulate instruction, Xqmac8, and evaluates different software strategies for preparing packed INT8 data.

### Current Progress

| Configuration | Description | Status |
|---|---|---|
| B0 | Baseline RV64IMC | Completed |
| B1 | RV64IMC + 4-way loop unrolling | Completed |
| X1.1 | RV64IMC + Xqmac8 + software data packing loop | Completed |
| X1.2 | Xqmac8 + unrolled software data packing | Completed |
| X1.3 | Xqmac8 + 64-bit load (Zicclsm enabled) | Completed |
| CV-Wally RTL | Hardware-level evaluation | Pending |

---

## 2. Repository Structure

```text
.
├── README.md
├── results
│   ├── alignmentCheck.txt
│   ├── b0_summary.txt
│   ├── b1_summary.txt
│   ├── slides
│   │   └── Midterm.pptx
│   ├── tensor_shape_summary.txt
│   ├── x1.1_xqmac8_summary.txt
│   ├── x1.2_xqmac8_summary.txt
│   └── Zicclsm_X1.3_summary.txt
├── scripts
│   ├── compare_versions.py
│   └── prepare_model.py
└── src
    ├── model
    │   ├── imagenet_classes.txt
    │   ├── resnet18_int8_channelwise.tflite
    │   ├── resnet18_model_data.h
    │   ├── resnet18_ops_resolver.h
    │   └── resnet18_test_image.h
    ├── resnet18_b0_inference.cc
    └── xqmac8
        ├── xqmac8.h
        └── XQMAC8_SPEC.md
```

### Directory and File Description

**src/ — Inference implementation and model files**

| File | Description |
|---|---|
| `resnet18_b0_inference.cc` | ResNet-18 inference benchmark source. |
| `model/resnet18_int8_channelwise.tflite` | INT8 quantized ResNet-18 model. |
| `model/resnet18_model_data.h` | Embedded model data generated from the TFLite model. |
| `model/resnet18_ops_resolver.h` | TFLM operator resolver configuration. |
| `model/resnet18_test_image.h` | Embedded test image. |
| `model/imagenet_classes.txt` | ImageNet class labels. |
| `xqmac8/xqmac8.h` | Xqmac8 instruction implementation for Spike. |
| `xqmac8/XQMAC8_SPEC.md` | Custom Xqmac8 instruction specification. |

**scripts/ — Model preparation and analysis tools**

| File | Description |
|---|---|
| `prepare_model.py` | Reads the TFLite model and generates header files required by the inference program. |
| `compare_versions.py` | Compares Spike instruction histogram results between different configurations. |

**results/ — Experimental results and analysis**

| File | Description |
|---|---|
| `b0_summary.txt` | B0 baseline histogram analysis. |
| `b1_summary.txt` | B1 software optimization results. |
| `x1.1_xqmac8_summary.txt` | X1.1 Xqmac8 results. |
| `x1.2_xqmac8_summary.txt` | X1.2 Xqmac8 results. |
| `Zicclsm_X1.3_summary.txt` | X1.3 results with Zicclsm enabled. |
| `alignmentCheck.txt` | Runtime memory-address alignment analysis for packed INT8 loads. |
| `tensor_shape_summary.txt` | Tensor shape analysis of the quantized TFLite model. |
| `slides/Midterm.pptx` | Midterm project presentation. |

---

## 3. Experimental Configuration

| Configuration | Value |
|---|---|
| Model | Quantized ResNet-18 |
| Weight / Activation | INT8 / INT8 (W8A8) |
| ISA baseline | RV64IMC |
| ABI | LP64 |
| Compiler | GCC 16.1.0 |
| Compiler optimization | -O2 |
| Simulator | Spike 1.1.1-dev |
| Runtime | TensorFlow Lite for Microcontrollers |
| Execution environment | Spike + Proxy Kernel (pk) |
| Tensor Arena | 2 MiB |

Spike is used for functional ISA-level simulation and dynamic instruction analysis. It is not a cycle-accurate simulator, so the reported instruction counts must not be interpreted as hardware cycle counts or execution-time speedups.

---

## 4. Optimization Progress

### 4.1 B0 — Baseline RV64IMC

B0 is the original software implementation using scalar INT8 multiply-accumulate operations.

It serves as the baseline for evaluating subsequent software optimizations and custom ISA extensions.

### 4.2 B1 — 4-way Loop Unrolling

The convolution implementation was optimized using 4-way loop unrolling.

The purpose is to reduce loop-control overhead, including branch and loop-index update instructions, while preserving the original scalar MAC computation.

### 4.3 X1.1 — Xqmac8 with Software Data Packing

A custom instruction, `xqmac8`, was implemented in Spike.

The instruction performs eight signed INT8 multiplications and accumulates their sum into a 64-bit destination register.

```asm
xqmac8 rd, rs1, rs2
```

The two source registers contain eight packed signed INT8 values each.

The initial TFLM integration uses a software packing loop to collect eight activation values and eight weight values before executing Xqmac8.

Although the custom instruction reduces the number of MAC instructions, software data-packing overhead increases the overall dynamic instruction count.

### 4.4 X1.2 — Unrolled Software Data Packing

The X1.1 packing loop was manually unrolled to eliminate its inner-loop control overhead.

This reduces the overhead of preparing packed INT8 operands, while retaining the same Xqmac8 computation.

However, the packing operations themselves remain, and the resulting instruction count is still slightly higher than B1.

### 4.5 X1.3 — 64-bit Load for Packed INT8 Data

X1.3 replaces byte-by-byte software packing with 64-bit loads.

Since eight consecutive INT8 values occupy exactly 64 bits, a single `LD` can load an entire packed operand.

This significantly reduces the software packing overhead.

However, some weight pointers are not naturally 8-byte aligned. Under the base RV64IMC execution environment, misaligned 64-bit loads may raise exceptions and require software emulation through the execution environment.

Therefore, the current X1.3 experiment enables the Zicclsm extension to support misaligned loads and stores in the Spike simulation environment.

**Important:** X1.3 is currently evaluated under RV64IMC + Zicclsm, not pure RV64IMC. Its instruction count must therefore be interpreted separately from configurations evaluated under RV64IMC.

---

## 5. Evaluation

### 5.1 Dynamic Instruction Count

The following table summarizes the measured dynamic instruction counts for each configuration.

| Configuration | Optimization | Dynamic Instructions |
|---|---|---:|
| B0 | Original scalar MAC | 82.93B |
| B1 | 4-way loop unrolling | 64.08B |
| X1.1 | Xqmac8 + software packing loop | 93.52B |
| X1.2 | Xqmac8 + unrolled software packing | 65.08B |
| X1.3 | Xqmac8 + 64-bit LD (Zicclsm) | 25.64B |

### 5.2 Comparison

Relative to B0:

| Configuration | Dynamic Instruction Reduction |
|---|---:|
| B1 | 22.7% |
| X1.1 | -12.8% |
| X1.2 | 21.5% |
| X1.3 | 69.1% |

Relative to B1:

| Configuration | Dynamic Instruction Reduction |
|---|---:|
| X1.1 | -46.0% |
| X1.2 | -1.6% |
| X1.3 | 60.0% |

A negative reduction indicates an increase in dynamic instruction count.

### 5.3 Observations

- B1 demonstrates that software loop unrolling can effectively reduce loop-control overhead.
- X1.1 shows that introducing a custom MAC instruction does not automatically improve overall instruction efficiency. Software operand-packing overhead can outweigh the benefits of the custom instruction.
- X1.2 reduces packing-loop overhead through manual unrolling, bringing its instruction count close to B1.
- X1.3 demonstrates the benefit of directly loading packed INT8 operands using 64-bit loads, reducing the dynamic instruction count substantially.
- The current X1.3 result relies on Zicclsm support for misaligned memory accesses and is not yet a pure RV64IMC result.

These measurements represent dynamic instruction counts, not execution cycles or hardware speedups.

---

## 6. Functional Correctness

Functional correctness is evaluated by comparing the quantized inference outputs across different configurations.

For the tested input image, the following configurations produced identical results:

- B0
- B1
- X1.1
- X1.2
- X1.3

The comparison covers all 1,000 output class scores, not only the Top-1 prediction.

| Metric | Result |
|---|---|
| Number of output classes | 1,000 |
| Top-1 class index | Identical across configurations |
| Top-1 quantized score | Identical across configurations |
| Output score mismatches | 0 |
| Identical output scores | 1,000 / 1,000 |


These results demonstrate numerical equivalence for the tested input and successful execution of the basic Xqmac8 instruction tests.

**Note:** This is a numerical-equivalence and functional-correctness test. It does not represent a complete ImageNet validation accuracy measurement.

---

## 7. Current Limitations

- X1.3 currently relies on Zicclsm for misaligned 64-bit loads.
- Pure RV64IMC execution of X1.3 may incur substantial software-emulation overhead for misaligned memory accesses. A complete performance measurement under this configuration has not been obtained.
- Functional equivalence has been verified using the tested inference input, but broader multi-image validation remains future work.
- ImageNet Top-1 and Top-5 accuracy of the W8A8 model have not been independently measured.
- Spike is not cycle-accurate, so CPI, hardware cycle count, clock frequency, and actual hardware speedup have not been measured.
- CV-Wally RTL integration and hardware-level evaluation have not yet been completed.

---

## 8. Future Work

| Configuration / Task | Description | Status |
|---|---|---|
| X1.4 | Alignment-aware data layout to enable 64-bit loads under pure RV64IMC | Planned |
| Correctness validation | Test numerical equivalence across additional input images | Planned |
| Model accuracy | Evaluate W8A8 model using labeled validation data | Planned |
| X2 | Additional architectural extension | Pending |
| CV-Wally RTL | Integrate and evaluate a representative custom instruction kernel | Pending |

Future evaluation will focus on:

- Dynamic instruction reduction.
- MACs per retired instruction.
- Load/store instructions per MAC.
- Custom instruction frequency.
- Code-size changes.
- Hardware cycle count and CPI after RTL integration.

The overall objective is to investigate how custom ISA extensions and software/data-layout optimizations can improve quantized neural-network inference on resource-constrained RISC-V processors.

