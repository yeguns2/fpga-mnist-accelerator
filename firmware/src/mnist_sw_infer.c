#include "mnist_sw_infer.h"
#include <stdint.h>

#include "weights.h"     // conv1_params, conv2_params, fc_w_packed, fc_b

// -----------------------
// HW-Equivalent constants
// -----------------------
#define CONV1_POOL_SHIFT 10   // RTL: final_max >>> 10 then clip to 0..255
#define FC_LOGIT_SHIFT   12   // RTL: logits32 = acc >>> 12

// Shapes
#define C1_OUT 8
#define C2_OUT 16
#define IMG_W  28
#define IMG_H  28

// conv1_params layout (export_weight.py):
// for f in 0..7: [w(3*3*1=9) ...] then [b]  -> 10 per filter, total 80
#define CONV1_STRIDE (9 + 1)

// conv2_params layout:
// for f in 0..15: [w(3*3*8=72) ...] then [b] -> 73 per filter, total 1168
#define CONV2_STRIDE (72 + 1)

// FC packed layout (export_weight.py):
// 10 outputs, each has 784 weights (7*7*16), packed 4 weights per int32 word
// => 784/4 = 196 words per output => 1960 total words
#define FC_WORDS_PER_OUT 196
#define FC_WEIGHTS_PER_OUT 784

// -----------------------
// Intermediate buffers (match RTL behavior)
// -----------------------
static uint8_t c1_pool_u8[C1_OUT][14][14];    // after conv1 relu + pool + >>>10 + clip
static int32_t c2_pool_i32[C2_OUT][7][7];     // after conv2 relu + pool (raw int32)

// -----------------------
// Helpers
// -----------------------
static inline uint32_t u32_max(uint32_t a, uint32_t b) { return (a > b) ? a : b; }
static inline int32_t  i32_max(int32_t a, int32_t b)   { return (a > b) ? a : b; }

// conv1 weight/bias access (weights-first then bias per filter)  :contentReference[oaicite:2]{index=2}
static inline int8_t conv1_w(int oc, int ky, int kx) {
    // oc:0..7, ky/kx:0..2
    int base = oc * CONV1_STRIDE;
    int widx = (ky * 3 + kx);          // 0..8 (ic=1 only)
    return (int8_t)conv1_params[base + widx];
}
static inline int32_t conv1_b(int oc) {
    int base = oc * CONV1_STRIDE;
    return conv1_params[base + 9];
}

// conv2 weight/bias access (weights-first then bias per filter) :contentReference[oaicite:3]{index=3}
static inline int8_t conv2_w(int oc, int ic, int ky, int kx) {
    // oc:0..15, ic:0..7, ky/kx:0..2
    int base = oc * CONV2_STRIDE;
    int widx = ((ky * 3 + kx) * 8) + ic;   // because export transposes to (Out, Row, Col, In)
    return (int8_t)conv2_params[base + widx];
}
static inline int32_t conv2_b(int oc) {
    int base = oc * CONV2_STRIDE;
    return conv2_params[base + 72];
}

// FC packed weight accessor:
// idx in [0..783], word = idx>>2, byte = idx&3 (little-endian packing) :contentReference[oaicite:4]{index=4}
static inline int8_t fc_w_packed_i8(int out, int idx) {
    int word_index = out * FC_WORDS_PER_OUT + (idx >> 2);
    uint32_t wword = (uint32_t)fc_w_packed[word_index];
    int byte_sel = idx & 3;
    uint8_t u = (wword >> (8 * byte_sel)) & 0xFF;
    return (int8_t)u; // sign-extend as int8
}

// ----------------------------------------
// Conv1: 28x28 SAME(pad=1), ReLU, 2x2 maxpool stride2, >>>10, clip to 0..255
// RTL-equivalent: ReLU uses lower 32-bit of acc when acc>=0
// ----------------------------------------
static void conv1_relu_pool_shiftclip(const uint8_t *img) {
    for (int oc = 0; oc < C1_OUT; oc++) {
        // store post-ReLU conv results as uint32 for pooling (matches RTL truncation to [31:0])
        uint32_t conv_u32[28][28];

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int64_t acc = (int64_t)conv1_b(oc);

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int iy = y + ky;
                        int ix = x + kx;

                        uint8_t px = 0;
                        if ((unsigned)iy < 28u && (unsigned)ix < 28u) {
                            px = img[iy * 28 + ix];
                        }
                        int8_t w = conv1_w(oc, ky + 1, kx + 1);
                        acc += (int64_t)((int32_t)px * (int32_t)w);
                    }
                }

                // RTL: (acc < 0) ? 0 : acc[31:0]
                if (acc < 0) conv_u32[y][x] = 0;
                else         conv_u32[y][x] = (uint32_t)acc; // low 32 bits
            }
        }

        // pool 2x2 stride2: 28 -> 14
        for (int oy = 0; oy < 14; oy++) {
            for (int ox = 0; ox < 14; ox++) {
                int y = oy * 2;
                int x = ox * 2;

                uint32_t a = conv_u32[y][x];
                uint32_t b = conv_u32[y][x + 1];
                uint32_t c = conv_u32[y + 1][x];
                uint32_t d = conv_u32[y + 1][x + 1];

                uint32_t m = u32_max(u32_max(a, b), u32_max(c, d));

                uint32_t scaled = (m >> CONV1_POOL_SHIFT);
                if (scaled > 255u) c1_pool_u8[oc][oy][ox] = 255u;
                else               c1_pool_u8[oc][oy][ox] = (uint8_t)scaled;
            }
        }
    }
}

// ----------------------------------------
// Conv2: 14x14 SAME(pad=1), ReLU, 2x2 maxpool stride2 -> 7x7 (raw int32)
// ----------------------------------------
static void conv2_relu_pool_raw32(void) {
    for (int oc = 0; oc < C2_OUT; oc++) {
        int32_t conv_out[14][14];

        for (int y = 0; y < 14; y++) {
            for (int x = 0; x < 14; x++) {
                int32_t acc = conv2_b(oc);

                for (int ic = 0; ic < C1_OUT; ic++) {
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int iy = y + ky;
                            int ix = x + kx;

                            int32_t px = 0;
                            if ((unsigned)iy < 14u && (unsigned)ix < 14u) {
                                px = (int32_t)c1_pool_u8[ic][iy][ix]; // 0..255
                            }
                            int8_t w = conv2_w(oc, ic, ky + 1, kx + 1);
                            acc += (int32_t)(px * (int32_t)w);
                        }
                    }
                }

                if (acc < 0) acc = 0; // ReLU
                conv_out[y][x] = acc;
            }
        }

        // pool 2x2 stride2: 14 -> 7
        for (int oy = 0; oy < 7; oy++) {
            for (int ox = 0; ox < 7; ox++) {
                int y = oy * 2;
                int x = ox * 2;

                int32_t a = conv_out[y][x];
                int32_t b = conv_out[y][x + 1];
                int32_t c = conv_out[y + 1][x];
                int32_t d = conv_out[y + 1][x + 1];

                int32_t m = i32_max(i32_max(a, b), i32_max(c, d));
                c2_pool_i32[oc][oy][ox] = m;
            }
        }
    }
}

// ----------------------------------------
// FC: flatten order = (row-major 7x7) then channel 0..15
// idx = (row*7 + col)*16 + ch  => matches export packing order :contentReference[oaicite:5]{index=5}
// acc_raw[k] = fc_b[k] + sum(feat * w)
// logits[k] = acc_raw[k] >> 12 (RTL)
// argmax on acc_raw
// ----------------------------------------
static void fc_and_argmax(sw_result_t *r) {
    int64_t acc[10];
    for (int k = 0; k < 10; k++) acc[k] = (int64_t)fc_b[k];

    int idx = 0;
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 7; col++) {
            for (int ch = 0; ch < 16; ch++) {
                int32_t feat = c2_pool_i32[ch][row][col]; // >=0
                // weight for each output class at this flattened idx
                for (int k = 0; k < 10; k++) {
                    int8_t w = fc_w_packed_i8(k, idx);
                    acc[k] += (int64_t)feat * (int64_t)w;
                }
                idx++;
            }
        }
    }

    // argmax over raw acc (equivalent to argmax over shifted logits)
    int best_i = 0;
    int64_t best_v = acc[0];
    for (int k = 1; k < 10; k++) {
        if (acc[k] > best_v) { best_v = acc[k]; best_i = k; }
    }

    r->pred = best_i;
    for (int k = 0; k < 10; k++) {
        r->acc_raw[k] = acc[k];
        r->logits[k]  = (int32_t)(acc[k] >> FC_LOGIT_SHIFT);
    }
}

sw_result_t mnist_sw_infer_run(const uint8_t *input_img) {
    sw_result_t r;

    conv1_relu_pool_shiftclip(input_img);
    conv2_relu_pool_raw32();
    fc_and_argmax(&r);

    return r;
}
