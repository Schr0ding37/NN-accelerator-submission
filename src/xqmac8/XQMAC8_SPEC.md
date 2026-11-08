# XQMAC8 Specification

## 1. Overview

Xqmac8 is a custom RISC-V instruction designed to accelerate
quantized neural-network inference.

The instruction performs eight signed INT8 multiplication
operations and accumulates their sum into a 64-bit destination
register.

Target architecture:
- RV64IMC
- XLEN = 64
- Signed INT8 operands
- 64-bit accumulator

## 2. Instruction Format

Instruction name:
    xqmac8 rd, rs1, rs2

Instruction format:
    R-type

Bit layout:

    31       25 24   20 19   15 14  12 11    7 6      0
    +----------+-------+-------+------+-------+--------+
    |  funct7  |  rs2  |  rs1  |funct3|  rd   | opcode |
    +----------+-------+-------+------+-------+--------+

Encoding:

    opcode = CUSTOM-0 = 0x0B
    funct3 = 0b000
    funct7 = 0b0000000

Instruction encoding:

    MATCH_XQMAC8 = 0x0000000B
    MASK_XQMAC8  = 0xFE00707F

Note:
    rs1, rs2, and rd are variable register fields.

## 3. Operand Definition

rs1 contains eight packed signed INT8 activation values.

rs2 contains eight packed signed INT8 weight values.

Each 64-bit register is interpreted as eight
8-bit signed integers.

    rs1 = {A7, A6, A5, A4, A3, A2, A1, A0}
    rs2 = {W7, W6, W5, W4, W3, W2, W1, W0}

Where:

    Ai = sign_extend(rs1[8*i +: 8])
    Wi = sign_extend(rs2[8*i +: 8])

Each operand is in the range:

    -128 <= Ai <= 127
    -128 <= Wi <= 127

## 4. Instruction Semantics

The instruction performs eight signed INT8
multiplication operations and accumulates the results.

Mathematical definition:

    result = X[rd] + SUM(Ai * Wi), for i = 0 to 7

Pseudocode:

    int64_t accumulator = X[rd];

    for (int i = 0; i < 8; i++) {
        int8_t a = sign_extend(rs1[8*i +: 8]);
        int8_t w = sign_extend(rs2[8*i +: 8]);

        accumulator += (int64_t)a * (int64_t)w;
    }

    X[rd] = accumulator;

The original value of rd is used as the accumulator.

## 5. Signedness

All INT8 operands are signed two's-complement integers.

Each INT8 operand is sign-extended before multiplication.

Each multiplication produces a signed integer result.

The multiplication result is accumulated into a 64-bit
integer accumulator.

## 6. Accumulator and Overflow Behavior

Accumulator width:

    64 bits

Overflow behavior:

    Arithmetic is performed modulo 2^64.

Only the lower 64 bits of the final result are retained.

No saturation is performed.

No exception is raised due to arithmetic overflow.

## 7. Register Constraints

- rs1: Packed INT8 activation operand.
- rs2: Packed INT8 weight operand.
- rd: 64-bit accumulator and destination.

The instruction reads the old value of rd before
writing the updated result.

The instruction supports rd = rs1 or rd = rs2
according to normal RISC-V register semantics.

## 8. Example

Initial values:

    A = {1, 2, 3, 4, 5, 6, 7, 8}
    W = {1, 1, 1, 1, 1, 1, 1, 1}
    accumulator = 100

Execution:

    xqmac8 rd, rs1, rs2

Calculation:

    result = 100
           + 1*1 + 2*1 + 3*1 + 4*1
           + 5*1 + 6*1 + 7*1 + 8*1

    result = 136

After execution:

    X[rd] = 136

## 9. Implementation Scope

The initial implementation targets Spike ISA simulation.

Required implementation:
- Custom instruction encoding.
- Instruction decoding.
- Signed INT8 multiplication.
- Eight-way multiply-accumulate.
- 64-bit accumulation.
- Register write-back.

Hardware implementation and physical design are
outside the scope of this specification.

## 10. Compiler Integration

The initial software interface uses GNU inline assembly.

The compiler does not need native knowledge of Xqmac8.

The instruction will be emitted using its custom
32-bit instruction encoding.

## 11. Performance Evaluation

The instruction will be evaluated using:

- Dynamic instruction count.
- Xqmac8 execution count.
- MUL instruction reduction.
- ADD instruction reduction.
- MAC operations per retired instruction.
- Kernel-level performance improvement.
- Output correctness against the B1 implementation.
