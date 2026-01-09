#pragma once
#include <stdint.h>

typedef struct {
    int      pred;        // 0..9
    int32_t  logits[10];  // (acc_raw >> 12),
    int64_t  acc_raw[10];
} sw_result_t;

// input_img: 28x28 uint8 row-major (0..255)
sw_result_t mnist_sw_infer_run(const uint8_t *input_img);
