# Quantized Neural-Network Acceleration on RV64IMC

## 1. Project Overview

This project explores architectural extensions for accelerating INT8 quantized neural-network inference on a bare-metal RV64IMC RISC-V processor using the CV-Wally core.

Spike is used for ISA-level analysis, while CV-Wally RTL is intended for hardware-level evaluation.

### Current Progress

- B0: Baseline RV64IMC workload characterization — Completed.
- B1: Software-optimized RV64IMC — Not implemented.
- X1: RV64IMC + Xqmac — Not implemented.
- CV-Wally RTL evaluation — Not completed.

The current submission focuses on B0 analysis using quantized ResNet-18.

---

## 2. Repository Structure

    .
    ├── README.md
    ├── results
    │   ├── B0_measurement
    │   │   ├── histogram_analysis
    │   │   │   ├── summary.txt
    │   │   │   └── top_pcs.csv
    │   │   ├── instruction_mix_summary.txt
    │   │   ├── model_mac_summary.md
    │   │   ├── operator_contribution.txt
    │   │   └── resnet18_b0_pc_histogram.txt
    │   └── slides
    ├── scripts
    │   ├── analyze_instruction_mix.py
    │   ├── analyze_model_macs.py
    │   ├── analyze_operator_contribution.py
    │   ├── analyze_pc_histogram.py
    │   └── prepare_model.py
    └── src
        ├── model
        │   ├── imagenet_classes.txt
        │   ├── resnet18_int8_channelwise.tflite
        │   ├── resnet18_model_data.h
        │   ├── resnet18_ops_resolver.h
        │   └── resnet18_test_image.h
        ├── resnet18_b0_inference.cc
        └── resnet18_b0_inference.elf

### Directory and File Description

**src/ — Inference implementation and model files**

| File | Description |
|---|---|
| `resnet18_b0_inference.cc` | C++ B0 inference benchmark. |
| `resnet18_b0_inference.elf` | Compiled RISC-V inference executable. |
| `model/resnet18_int8_channelwise.tflite` | INT8 quantized ResNet-18 model. |
| `model/resnet18_model_data.h` | Embedded model data. |
| `model/resnet18_ops_resolver.h` | TFLM operator resolver configuration. |
| `model/resnet18_test_image.h` | Embedded test image. |
| `model/imagenet_classes.txt` | ImageNet class labels. |

**scripts/ — Model preparation and analysis tools**

| File | Description |
|---|---|
| `prepare_model.py` | Model preparation. |
| `analyze_pc_histogram.py` | PC histogram analysis. |
| `analyze_instruction_mix.py` | Dynamic instruction-mix analysis. |
| `analyze_model_macs.py` | Theoretical model MAC estimation. |
| `analyze_operator_contribution.py` | Operator contribution analysis. |

**results/B0_measurement/ — B0 measurement results**

| File | Description |
|---|---|
| `resnet18_b0_pc_histogram.txt` | Raw Spike PC histogram. |
| `instruction_mix_summary.txt` | Dynamic instruction distribution. |
| `model_mac_summary.md` | Model MAC analysis. |
| `operator_contribution.txt` | Operator contribution estimates. |
| `histogram_analysis/summary.txt` | Histogram statistics. |
| `histogram_analysis/top_pcs.csv` | Most frequently executed PCs. |

**results/slides/** — Project presentation slides.

---

## 3. B0 Measurement Results

### Experimental Configuration

| Configuration | Value |
|---|---|
| Model | Quantized ResNet-18 |
| Input / Output | INT8 |
| ISA | RV64IMC |
| ABI | LP64 |
| Compiler | GCC 16.1.0 |
| Optimization | -O2 |
| Simulator | Spike 1.1.1-dev |
| Runtime | TensorFlow Lite for Microcontrollers |
| Execution | Spike + pk |
| Tensor Arena | 2 MiB |

### 3.1 Inference Results

| Metric | Result |
|---|---:|
| AllocateTensors | PASS |
| Invoke | PASS |
| Top-1 Class Index | 245 |
| Top-1 Quantized Score | 40 |

### 3.2 Dynamic Instruction Count

Measured using `rdinstret` immediately before and after `interpreter.Invoke()`.

| Metric | Result |
|---|---:|
| Retired Instructions | 82,907,465,609 |
| Equivalent Scale | ~82.907 billion |

This is the primary B0 dynamic-instruction baseline.

### 3.3 Dynamic Instruction Mix

Based on Spike PC histogram and disassembly analysis.

| Instruction Category | Dynamic Count | Percentage |
|---|---:|---:|
| Load | 17,301,635,179 | 20.870% |
| Store | 178,384,500 | 0.215% |
| Integer ALU | 21,041,446,086 | 25.382% |
| Multiply | 11,859,266,659 | 14.305% |
| Branch / Control Transfer | 32,519,371,645 | 39.227% |
| **Total Matched** | **82,900,104,069** | **100.000%** |

Note: 3,848 histogram PCs (27,180,642 dynamic counts) could not be matched to disassembly instructions. Percentages above are calculated from matched instructions only.

### 3.4 Model MAC Analysis

| Operator | Operator Count | MACs | Percentage |
|---|---:|---:|---:|
| CONV_2D | 20 | 1,813,561,344 | 99.9718% |
| FULLY_CONNECTED | 1 | 512,000 | 0.0282% |
| **Total** | **21** | **1,814,073,344** | **100%** |

| Metric | Result |
|---|---:|
| Theoretical MAC Count | 1,814,073,344 |
| Retired Instructions | 82,907,465,609 |
| MACs per Retired Instruction | 0.021881 |
| Retired Instructions per MAC | ~45.70 |

MAC count is statically estimated from the TFLite model graph and tensor shapes.

### 3.5 Operator Contribution

Preliminary symbol-based attribution results:

| Function / Operator | Dynamic Count |
|---|---:|
| ConvPerChannel() clone | 82,606,450,546 |
| MultiplyByQuantizedMultiplier | 91,014,458 |
| EvalAddQuantized | 90,605,848 |
| MaxPool | 69,684,988 |
| PadImpl | 27,541,091 |

Convolution accounts for approximately 99.65% of the attributed dynamic instruction count.

These results are preliminary estimates based on disassembly symbols and operator grouping.

### 3.6 Code Size

| Metric | Result |
|---|---:|
| ELF File Size | 12,622,552 bytes |
| .text Section Size | 179,694 bytes |
| Total Executable Section Size | 179,694 bytes |

The executable-section size is used as the primary code-size metric.

---

## 4. Current Limitations and Future Work

### Current Limitations

- Only B0 baseline characterization has been completed.
- 3,848 histogram PCs remain unmatched to disassembly instructions.
- Operator contribution analysis is preliminary.
- No custom ISA instructions have been implemented.
- No CV-Wally RTL performance evaluation has been conducted.
- Cycle count, CPI, clock frequency, and hardware resource overhead have not been measured.

Spike is used for ISA-level analysis and is not a cycle-accurate representation of CV-Wally.

### Future Work

| Configuration | Description | Status |
|---|---|---|
| B0 | Baseline RV64IMC | Completed |
| B1 | Software-optimized RV64IMC | Pending |
| X1 | RV64IMC + Xqmac | Pending |
| X2 | Additional architectural extension | Pending |
| CV-Wally | RTL evaluation of a representative kernel | Pending |

Future evaluation will focus on:

- Dynamic instruction reduction.
- MACs per instruction.
- Load/store instructions per MAC.
- Custom-instruction frequency.
- Code-size changes.
- Hardware cycle count, CPI, and speedup.

The objective is to evaluate the effectiveness of architectural extensions for quantized neural-network inference.

