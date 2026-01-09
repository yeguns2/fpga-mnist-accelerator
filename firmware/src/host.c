/*
 * =============================================================================
 * File: host.c
 * Author: Eric Shim
 * Date: Jan 8, 2026
 * -----------------------------------------------------------------------------
 * Summary
 *   MicroBlaze firmware for MNIST inference using a custom CNN accelerator.
 *
 *   Flow:
 *     1) Copy Conv1/Conv2/FC weights + biases into BRAM (CPU writes)
 *     2) Run "Weight Load Mode" once after boot to preload accelerator caches
 *     3) In an infinite loop:
 *        - Receive a 28x28 image over UART (784 bytes)
 *        - Flush cache so BRAM sees the new image
 *        - Kick accelerator in inference mode
 *        - Poll done flag
 *        - Read cycles + predicted label via AXI-Lite
 *        - Invalidate cache and read logits from BRAM
 *        - Send a packet back to the PC: magic + cycles + label + logits[10]
 *
 * Notes
 *   - BRAM is memory-mapped at BRAM_BASEADDR.
 *   - Accelerator is controlled through AXI4-Lite registers at ACCEL_BASEADDR.
 * =============================================================================
 */

#include <stdint.h>
#include <string.h>

#include "platform.h"
#include "xparameters.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xtmrctr.h"

#include "weights.h"

/* ===================== ADDED FOR SW INFERENCE BENCHMARK ===================== */
#include "mnist_sw_infer.h"
/* =========================================================================== */

#define N_LOGITS 10

// ============================================================================
// Base Addresses
// ============================================================================
#define ACCEL_BASEADDR     0x44A00000
#define BRAM_BASEADDR      0xC0000000

// ============================================================================
// BRAM Memory Map (64KB region assumed in this design)
//
// Static regions written by MicroBlaze (weights/image)
//   - The accelerator reads these through its AXI4 master
// ============================================================================
#define WEIGHT_OFFSET      0x00001000 // Conv1 weights/bias blob
#define CONV2_OFFSET       0x00002008 // Conv2 weights/bias blob (aligned in memory map)
#define FC_W_OFFSET        0x0000A0B0 // FC packed weights (10 * 196 words = 7840B)
#define FC_B_OFFSET        0x0000D000 // FC bias (10 words = 40B)
#define IMG_OFFSET         0x00004000 // Input image buffer (784B)

// Results written by HW (avoid overlap with other regions)
#define RESULT_OFFSET      0x0000FC00 // 10 logits (40B)

// ============================================================================
// Transfer Sizes
// ============================================================================
#define IMG_SIZE           784

// ============================================================================
// AXI-Lite Register Map (must match RTL: case(awaddr_q[5:2]))
// ============================================================================
#define REG_CTRL           0x00 // bit0: start pulse, bit1: clear done, bit2: weight-load mode
#define REG_IMG_BASE       0x04
#define REG_WEIGHT_BASE    0x08
#define REG_WEIGHT2_BASE   0x0C
#define REG_FC_W_BASE      0x10
#define REG_FC_B_BASE      0x14

#define REG_STATUS         0x18 // bit0: busy, bit1: done
#define REG_CYCLES         0x1C
#define REG_RESULT_LABEL   0x24
#define REG_RESULT_BASE    0x28


static XTmrCtr TimerInst;


// ============================================================================
// UART Helpers
// ============================================================================
static inline void uart_write_byte(uint8_t data) { outbyte(data); }
static inline uint8_t uart_read_byte(void)       { return inbyte(); }

static inline void uart_write_word(uint32_t data) {
    uart_write_byte((uint8_t)(data & 0xFF));
    uart_write_byte((uint8_t)((data >> 8) & 0xFF));
    uart_write_byte((uint8_t)((data >> 16) & 0xFF));
    uart_write_byte((uint8_t)((data >> 24) & 0xFF));
}

// ============================================================================
// Main
// ============================================================================
int main(void) {
    init_platform();
    // ------------------------------------------------------------------------
        // 1. AXI Timer 초기화 (XTmrCtr 함수 사용)
        // ------------------------------------------------------------------------
        int status;
        // XPAR_AXI_TIMER_0_DEVICE_ID는 xparameters.h에 자동 생성되어 있습니다.
        status = XTmrCtr_Initialize(&TimerInst, XPAR_AXI_TIMER_0_DEVICE_ID);
        if (status != XST_SUCCESS) {
            xil_printf("ERROR: Timer Init failed\r\n");
            return 1;
        }

        // 타이머 옵션 설정 (자동 리로드 등 필요하면 설정, 기본값도 OK)
        // 0번 타이머를 사용합니다.
        XTmrCtr_SetOptions(&TimerInst, 0, 0);
        XTmrCtr_Reset(&TimerInst, 0);
        XTmrCtr_Start(&TimerInst, 0);

    // ------------------------------------------------------------------------
    // 1) Initialize BRAM contents (CPU writes weights/biases once at boot)
    // ------------------------------------------------------------------------
    xil_printf("Initializing BRAM...\r\n");

    // Conv1 parameters
    int32_t *bram_w_ptr = (int32_t *)(BRAM_BASEADDR + WEIGHT_OFFSET);
    memcpy(bram_w_ptr, conv1_params, sizeof(conv1_params));

    // Conv2 parameters
    int32_t *bram_w2_ptr = (int32_t *)(BRAM_BASEADDR + CONV2_OFFSET);
    memcpy(bram_w2_ptr, conv2_params, sizeof(conv2_params));
    uint32_t conv2_addr = (uint32_t)(BRAM_BASEADDR + CONV2_OFFSET);

    // FC weights and bias
    int32_t *fc_w_ptr = (int32_t *)(BRAM_BASEADDR + FC_W_OFFSET);
    memcpy(fc_w_ptr, fc_w_packed, sizeof(fc_w_packed));

    int32_t *fc_b_ptr = (int32_t *)(BRAM_BASEADDR + FC_B_OFFSET);
    memcpy(fc_b_ptr, fc_b, sizeof(fc_b));

    // Cache flush: ensure all CPU writes are committed to BRAM before HW reads.
    // Flushing a generous range keeps the firmware robust against map tweaks.
    Xil_DCacheFlushRange(BRAM_BASEADDR + WEIGHT_OFFSET, 65536);

    // ------------------------------------------------------------------------
    // 2) HW Stage 1: Weight Load Mode (run once after boot)
    //    The accelerator fetches weights from BRAM and stores them internally.
    // ------------------------------------------------------------------------
    xil_printf("Loading Weights into Accelerator...\r\n");

    // Program base addresses for each weight/bias region
    Xil_Out32(ACCEL_BASEADDR + REG_WEIGHT_BASE,  (uint32_t)(BRAM_BASEADDR + WEIGHT_OFFSET));
    Xil_Out32(ACCEL_BASEADDR + REG_WEIGHT2_BASE, conv2_addr);
    Xil_Out32(ACCEL_BASEADDR + REG_FC_W_BASE,    (uint32_t)(BRAM_BASEADDR + FC_W_OFFSET));
    Xil_Out32(ACCEL_BASEADDR + REG_FC_B_BASE,    (uint32_t)(BRAM_BASEADDR + FC_B_OFFSET));

    // Start Weight Load Mode (bit2=1), then drop to 0 to create a pulse
    Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x04);
    Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x00);

    // Poll for done (REG_STATUS bit1)
    while ((Xil_In32(ACCEL_BASEADDR + REG_STATUS) & 0x02) == 0) { }

    // Clear done flag (bit1=1), then drop back to 0
    Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x02);
    Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x00);

    xil_printf("Weights Loaded! Ready for Inference.\r\n");

    // ------------------------------------------------------------------------
    // 3) HW Stage 2: Inference loop
    // ------------------------------------------------------------------------
    const uint32_t result_base_addr = (uint32_t)(BRAM_BASEADDR + RESULT_OFFSET);

    // Magic value for framing on the PC side (packet sync)
    const uint32_t magic = 0xAABBCCDD;

    while (1) {
        // --------------------------------------------------------------------
        // A) Receive image over UART and store into BRAM
        // --------------------------------------------------------------------
        volatile uint8_t *img_ptr = (volatile uint8_t *)(BRAM_BASEADDR + IMG_OFFSET);
        for (int i = 0; i < IMG_SIZE; i++) {
            img_ptr[i] = uart_read_byte();
        }

        // Flush the image region so HW sees the newest pixels
        Xil_DCacheFlushRange(BRAM_BASEADDR + IMG_OFFSET, 1024);

        /* --------------------------------------------------------------------
         * B0) SW compute-only inference (Conv1 -> Conv2 -> FC -> Argmax)
         *     This uses the exact same quantized model and arithmetic
         *     as the hardware accelerator, and measures compute-only cycles.
         * ------------------------------------------------------------------ */
        uint32_t t_start = XTmrCtr_GetValue(&TimerInst, 0);

                sw_result_t sw = mnist_sw_infer_run((const uint8_t *)img_ptr);

                uint32_t t_end = XTmrCtr_GetValue(&TimerInst, 0);

                uint32_t sw_cycles = t_end - t_start;

        // --------------------------------------------------------------------
        // B) Run accelerator (image load + CNN compute)
        // --------------------------------------------------------------------

        // Clear done flag defensively in case it is still set
        Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x02);
        Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x00);

        // Set image base and result base addresses
        Xil_Out32(ACCEL_BASEADDR + REG_IMG_BASE,    (uint32_t)(BRAM_BASEADDR + IMG_OFFSET));
        Xil_Out32(ACCEL_BASEADDR + REG_RESULT_BASE, result_base_addr);

        // Start inference (bit0=1 pulse)
        Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x01);
        Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x00);

        // Wait until inference completes
        while ((Xil_In32(ACCEL_BASEADDR + REG_STATUS) & 0x02) == 0) { }

        // --------------------------------------------------------------------
        // C) Read back results and send to PC
        // --------------------------------------------------------------------
        uint32_t cycles     = Xil_In32(ACCEL_BASEADDR + REG_CYCLES);
        uint32_t fpga_label = Xil_In32(ACCEL_BASEADDR + REG_RESULT_LABEL);

        // HW wrote logits into BRAM; invalidate CPU cache before reading them
        Xil_DCacheInvalidateRange(result_base_addr, 64);
        volatile int32_t *logits_ptr = (volatile int32_t *)result_base_addr;

        // Packet:
        //   magic
        //   HW cycles, HW label, HW logits[10]
        //   SW cycles, SW label
        uart_write_word(magic);

        // HW results
        uart_write_word(cycles);
        uart_write_word(fpga_label);
        for (int i = 0; i < N_LOGITS; i++) {
            uart_write_word((uint32_t)logits_ptr[i]);
        }

        // SW results (compute-only reference)
        uart_write_word(sw_cycles);
        uart_write_word((uint32_t)sw.pred);
    }

    cleanup_platform();
    return 0;
}
