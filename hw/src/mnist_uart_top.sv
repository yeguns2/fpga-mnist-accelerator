`timescale 1ns / 1ps

/* =============================================================================
 * Top Module: mnist_uart_top
 * Author : Yegun Shim
 * Date   : 2025-12-10
 * -----------------------------------------------------------------------------
 * Description:
 *   This is the absolute top-level module for the Urbana FPGA board design.
 *   It provides the board-level clock and reset wiring and exposes the UART
 *   interface to the outside world.
 *
 *   All application logic, including:
 *     - AXI-based MNIST CNN accelerator
 *     - UART command/response handling
 *     - BRAM / DDR memory interfaces
 *     - Control FSMs and computation pipelines
 *   is encapsulated inside the generated `mnist_wrapper` module.
 *
 *   This module exists purely as a thin integration layer that connects
 *   physical board I/O (clock, reset, UART pins) to the internal system.
 *
 * Clocking:
 *   - Clk : 100 MHz system clock provided by the Urbana board
 *
 * Reset:
 *   - reset_rtl_0 : Active-high external reset
 *     Internally inverted to match the active-low reset convention
 *     used inside the design.
 *
 * Interfaces:
 *   - UART RX/TX connected directly to the internal MNIST system wrapper
 *
 * Notes:
 *   - No computation or control logic is implemented here.
 *   - Modifications should be made inside `mnist_wrapper` or lower-level modules,
 *     not in this top file.
 * ============================================================================= */

module mnist_uart_top (
    // -------------------------------------------------------------------------
    // Clock & Reset (Board-Level Signals)
    // -------------------------------------------------------------------------
    input  logic Clk,             // 100 MHz system clock
    input  logic reset_rtl_0,      // Active-high external reset
    
    // -------------------------------------------------------------------------
    // UART Interface (Board Pins)
    // -------------------------------------------------------------------------
    input  logic uart_rtl_0_rxd,   // UART receive from host
    output logic uart_rtl_0_txd    // UART transmit to host
);

    // -------------------------------------------------------------------------
    // System Wrapper Instantiation
    // -------------------------------------------------------------------------
    // `mnist_wrapper` contains the complete system:
    //   - AXI interconnect
    //   - MNIST CNN accelerator
    //   - UART protocol logic
    //   - Memory-mapped control and result handling
    //
    // This top module only forwards board-level signals.
    // -------------------------------------------------------------------------
    mnist_wrapper mnist_i (
        .clk_100MHz (Clk),                 // Board clock
        .reset_rtl_0 (~reset_rtl_0),       // Convert to active-low reset

        // UART passthrough
        .uart_rxd (uart_rtl_0_rxd),
        .uart_txd (uart_rtl_0_txd)
    );

endmodule
