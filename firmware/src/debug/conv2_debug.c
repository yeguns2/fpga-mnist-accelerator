/*
 * ============================================================================
 * File: conv2_debug.c
 * Author: Eric Shim
 * Date: Jan 8, 2026
 *
 * Purpose:
 *   Minimal UART inference loop for CONV2 debug comparison.
 *   - Receives a 28x28 MNIST image (784 bytes) over UART
 *   - Runs the accelerator
 *   - Reads the CONV2 debug dump (7*7*16 = 784 int32 words) from BRAM
 *   - Sends a binary UART frame to the host for comparison
 *
 * UART Frame Format (host_conv2_comp.py expectation):
 *   MAGIC (4B) + cycles (4B) + conv2_dump[784] (784 * 4B)
 *
 * ============================================================================
 */
/*
#include <stdint.h>
#include <string.h>

#include "platform.h"
#include "xil_io.h"
#include "xil_cache.h"
#include "weights.h"

// ===========================================================================
// Address / Register Map
// ===========================================================================
#define ACCEL_BASEADDR     0x44A00000u
#define BRAM_BASEADDR      0xC0000000u

// ---------------------------------------------------------------------------
// BRAM Memory Map (64KB) - CPU-written regions (weights/image)
// ---------------------------------------------------------------------------
#define WEIGHT_OFFSET      0x00001000u  // Conv1 weights+bias block
#define CONV2_OFFSET       0x00002008u  // Conv2 weights+bias block (aligned in your setup)
#define FC_W_OFFSET        0x0000A0B0u  // FC packed weights (still loaded because RTL may run FC)
#define FC_B_OFFSET        0x0000D000u  // FC bias
#define IMG_OFFSET         0x00004000u  // 28x28 image bytes

// ---------------------------------------------------------------------------
// HW debug dump address (written relative to reg_img_base in RTL)
//   conv2 dump: reg_img_base + 0xA000
//   if reg_img_base = BRAM + IMG_OFFSET => BRAM + IMG_OFFSET + 0xA000
// ---------------------------------------------------------------------------
#define HW_CONV2_DUMP_ADDR  (BRAM_BASEADDR + IMG_OFFSET + 0x0000A000u)

// Sizes
#define IMG_SIZE           784
#define CONV2_DUMP_WORDS   784   // 7 * 7 * 16

// ---------------------------------------------------------------------------
// AXI-Lite registers (must match RTL case(awaddr_q[5:2]))
// ---------------------------------------------------------------------------
#define REG_CTRL           0x00u
#define REG_IMG_BASE       0x04u
#define REG_WEIGHT_BASE    0x08u
#define REG_WEIGHT2_BASE   0x0Cu
#define REG_FC_W_BASE      0x10u
#define REG_FC_B_BASE      0x14u
#define REG_STATUS         0x18u  // read
#define REG_CYCLES         0x1Cu  // read

// UART binary framing
static inline void uart_write_byte(uint8_t data) { outbyte(data); }
static inline uint8_t uart_read_byte(void)       { return inbyte(); }

static inline void uart_write_word(uint32_t data) {
    uart_write_byte((uint8_t)( data        & 0xFFu));
    uart_write_byte((uint8_t)((data >>  8) & 0xFFu));
    uart_write_byte((uint8_t)((data >> 16) & 0xFFu));
    uart_write_byte((uint8_t)((data >> 24) & 0xFFu));
}

int main(void) {
    init_platform();

    // -----------------------------------------------------------------------
    // 0) Preload weights into BRAM (once at boot)
    // -----------------------------------------------------------------------
    {
        int32_t *bram_c1_ptr = (int32_t *)(BRAM_BASEADDR + WEIGHT_OFFSET);
        memcpy(bram_c1_ptr, conv1_params, sizeof(conv1_params));

        int32_t *bram_c2_ptr = (int32_t *)(BRAM_BASEADDR + CONV2_OFFSET);
        memcpy(bram_c2_ptr, conv2_params, sizeof(conv2_params));

        int32_t *bram_fc_w_ptr = (int32_t *)(BRAM_BASEADDR + FC_W_OFFSET);
        memcpy(bram_fc_w_ptr, fc_w_packed, sizeof(fc_w_packed));

        int32_t *bram_fc_b_ptr = (int32_t *)(BRAM_BASEADDR + FC_B_OFFSET);
        memcpy(bram_fc_b_ptr, fc_b, sizeof(fc_b));

        // Ensure CPU-written BRAM contents are committed (write-back cache flush)
        Xil_DCacheFlushRange(BRAM_BASEADDR + WEIGHT_OFFSET, 65536);
    }

    const uint32_t conv2_addr = (uint32_t)(BRAM_BASEADDR + CONV2_OFFSET);
    const uint32_t magic      = 0xAABBCCDDu;

    // -----------------------------------------------------------------------
    // 1) Main loop: image RX -> run accelerator -> UART dump CONV2 results
    // -----------------------------------------------------------------------
    while (1) {
        // A) Receive image bytes from host over UART into BRAM
        volatile uint8_t *img_ptr = (volatile uint8_t *)(BRAM_BASEADDR + IMG_OFFSET);
        for (int i = 0; i < IMG_SIZE; i++) {
            img_ptr[i] = uart_read_byte();
        }

        // Push received image bytes to BRAM (flush write-back cache lines)
        Xil_DCacheFlushRange(BRAM_BASEADDR + IMG_OFFSET, 1024);

        // B) Clear done flag (CTRL[1]) in case it is sticky
        Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x02u);
        Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x00u);

        // C) Program base addresses (absolute addresses)
        Xil_Out32(ACCEL_BASEADDR + REG_IMG_BASE,     (uint32_t)(BRAM_BASEADDR + IMG_OFFSET));
        Xil_Out32(ACCEL_BASEADDR + REG_WEIGHT_BASE,  (uint32_t)(BRAM_BASEADDR + WEIGHT_OFFSET));
        Xil_Out32(ACCEL_BASEADDR + REG_WEIGHT2_BASE, conv2_addr);
        Xil_Out32(ACCEL_BASEADDR + REG_FC_W_BASE,    (uint32_t)(BRAM_BASEADDR + FC_W_OFFSET));
        Xil_Out32(ACCEL_BASEADDR + REG_FC_B_BASE,    (uint32_t)(BRAM_BASEADDR + FC_B_OFFSET));

        // D) Start pulse (CTRL[0])
        Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x01u);
        Xil_Out32(ACCEL_BASEADDR + REG_CTRL, 0x00u);

        // E) Wait until done (STATUS[1])
        while ((Xil_In32(ACCEL_BASEADDR + REG_STATUS) & 0x00000002u) == 0u) { ; }

        // F) Read performance counter (cycles)
        uint32_t cycles = Xil_In32(ACCEL_BASEADDR + REG_CYCLES);

        // G) Read CONV2 dump from BRAM and transmit a binary frame
        //    HW wrote -> invalidate CPU cache before reading
        const uint32_t conv2_dump_addr = (uint32_t)HW_CONV2_DUMP_ADDR;
        Xil_DCacheInvalidateRange(conv2_dump_addr, 4096);

        volatile int32_t *conv2_ptr = (volatile int32_t *)conv2_dump_addr;

        uart_write_word(magic);
        uart_write_word(cycles);
        for (int i = 0; i < CONV2_DUMP_WORDS; i++) {
            uart_write_word((uint32_t)conv2_ptr[i]);
        }
    }

    cleanup_platform();
    return 0;
}
*/
