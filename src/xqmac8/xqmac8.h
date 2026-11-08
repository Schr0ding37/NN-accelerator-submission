
// Read the source registers and the original destination register.
//
// a   : rs1, containing 8 packed signed INT8 activations.
// b   : rs2, containing 8 packed signed INT8 weights.
// acc : rd, used as the initial 64-bit accumulator.
//
// Read all register values before write-back to support register aliasing,
// such as rd == rs1 or rd == rs2.
reg_t a = RS1;
reg_t b = RS2;
reg_t acc = RD;

// Process 8 pairs of INT8 values (8 lanes).
// Each lane occupies 8 bits in the 64-bit source register.
for (int i = 0; i < 8; i++) {

    // Extract the i-th byte from each source register.
    //
    // Right shift moves the target byte to the lowest 8 bits.
    // Bitwise AND with 0xFF keeps only those 8 bits.
    //
    // The extracted values are unsigned bytes (0 to 255).
    uint8_t a_lane = (a >> (8 * i)) & 0xFF;
    uint8_t b_lane = (b >> (8 * i)) & 0xFF;

    // Convert the unsigned byte into a signed INT8 value.
    //
    // In two's complement representation, bit 7 is the sign bit.
    // If bit 7 is 1, subtract 256 (2^8) to obtain the signed value.
    // Otherwise, the value is already non-negative.
    //
    // Use int64_t to ensure sign extension before multiplication.
    int64_t ai = (a_lane & 0x80)
        ? (int64_t)a_lane - 256
        : (int64_t)a_lane;

    int64_t bi = (b_lane & 0x80)
        ? (int64_t)b_lane - 256
        : (int64_t)b_lane;

    // Perform signed INT8 multiplication.
    // The product is safely representable as an int64_t value.
    int64_t product = ai * bi;

    // Accumulate the product into the 64-bit accumulator.
    //
    // Cast product to reg_t (unsigned 64-bit on RV64) before addition.
    // Unsigned arithmetic wraps modulo 2^64, matching the Xqmac8
    // specification and avoiding C++ signed integer overflow.
    acc += (reg_t)product;
}

// Write the final accumulated result back to the destination register.
// Only the lower 64 bits are retained.
WRITE_RD(acc);



