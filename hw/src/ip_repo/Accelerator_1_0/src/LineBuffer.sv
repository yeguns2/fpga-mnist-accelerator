`timescale 1ns/1ps

/* =============================================================================
 * Module: LineBuffer
 * Author: Yegun Shim
 * Date:   2026-01-01
 * -----------------------------------------------------------------------------
 * Summary
 *   Sliding-window line buffer for streaming convolution.
 *
 *   - Accepts CHANNEL parallel samples per cycle (e.g., 8 channels) with valid/ready.
 *   - Maintains (K_SIZE x K_SIZE) window (e.g., 3x3) over an image stream.
 *   - Uses internal line buffers to provide the previous rows needed for the window.
 *   - Buffers produced windows in a small 2-entry FIFO to decouple generation
 *     from downstream backpressure (win_ready).
 *
 * Handshake
 *   - Input side:  in_valid / in_ready
 *   - Output side: win_valid / win_ready
 *
 * Notes
 *   - "flush" is used to restart window formation (e.g., at the beginning of a new frame/stream).
 * ============================================================================= */

module LineBuffer #(
    parameter integer CHANNEL   = 8,
    parameter integer K_SIZE    = 3,
    parameter integer IMG_WIDTH = 14,
    parameter integer DW        = 9
)(
    input  logic clk,
    input  logic rst_n,
    input  logic flush,

    // Input stream: CHANNEL samples per cycle
    input  logic                 in_valid,
    output logic                 in_ready,
    input  logic signed [DW-1:0] ch_in [0:CHANNEL-1],

    // Output window stream: a K_SIZE x K_SIZE x CHANNEL window
    output logic                 win_valid,
    input  logic                 win_ready,
    output logic signed [DW-1:0] window [0:K_SIZE-1][0:K_SIZE-1][0:CHANNEL-1]
);

    // -------------------------------------------------------------------------
    // Data path storage
    // -------------------------------------------------------------------------
    // line_buf holds K_SIZE-1 previous rows (shift-register style across IMG_WIDTH)
    (* ram_style = "distributed" *) logic signed [DW-1:0] line_buf [0:K_SIZE-2][0:IMG_WIDTH-1][0:CHANNEL-1];

    // w_reg is the current window register; w_next is the combinational next-state
    (* ram_style = "distributed" *) logic signed [DW-1:0] w_reg  [0:K_SIZE-1][0:K_SIZE-1][0:CHANNEL-1];
    (* ram_style = "distributed" *) logic signed [DW-1:0] w_next [0:K_SIZE-1][0:K_SIZE-1][0:CHANNEL-1];

    // -------------------------------------------------------------------------
    // Counters
    // -------------------------------------------------------------------------
    // START_CYCLES defines how many input samples are needed before the first valid window
    localparam integer START_CYCLES = (K_SIZE-1)*IMG_WIDTH + (K_SIZE-1);
    logic [$clog2(START_CYCLES+2)-1:0] acc_cnt;
    logic [$clog2(IMG_WIDTH)-1:0] col_cnt;

    // -------------------------------------------------------------------------
    // Small 2-entry FIFO to buffer generated windows (handles backpressure cleanly)
    // -------------------------------------------------------------------------
    logic [1:0] count;
    logic rd_ptr, wr_ptr;
    logic pop, push;
    (* ram_style = "distributed" *) logic signed [DW-1:0] fifo_mem [0:1][0:K_SIZE-1][0:K_SIZE-1][0:CHANNEL-1];

    // Output-valid if FIFO is non-empty
    assign win_valid = (count != 0);

    // Pop when downstream is ready and FIFO has data
    assign pop       = win_valid && win_ready;

    // Input is ready if FIFO has space, or if we are popping this cycle (freeing space)
    assign in_ready  = (count < 2) || pop;

    // Input accept condition
    wire in_fire = in_valid && in_ready;

    // -------------------------------------------------------------------------
    // Window generation enable
    // -------------------------------------------------------------------------
    // gen_fire asserts only when:
    //  - we accepted an input sample (in_fire)
    //  - we have collected enough history for a full KxK window
    //  - we are past the initial warm-up region (acc_cnt) and at a valid column (col_cnt)
    logic gen_fire;

    always_comb begin
        gen_fire = 1'b0;
        if (in_fire) begin
            if ((acc_cnt >= START_CYCLES[$bits(acc_cnt)-1:0]) &&
                (col_cnt >= (K_SIZE-1))) begin
                gen_fire = 1'b1;
            end
        end
    end

    // Push into FIFO when a new window is generated
    assign push = gen_fire;

    // -------------------------------------------------------------------------
    // Output logic (FIFO head -> window)
    // -------------------------------------------------------------------------
    always_comb begin
        integer r,c,ch;
        if (count == 0) begin
            for (r=0; r<K_SIZE; r=r+1)
                for (c=0; c<K_SIZE; c=c+1)
                    for (ch=0; ch<CHANNEL; ch=ch+1)
                        window[r][c][ch] = '0;
        end else begin
            for (r=0; r<K_SIZE; r=r+1)
                for (c=0; c<K_SIZE; c=c+1)
                    for (ch=0; ch<CHANNEL; ch=ch+1)
                        window[r][c][ch] = fifo_mem[rd_ptr][r][c][ch];
        end
    end

    // -------------------------------------------------------------------------
    // Compute w_next (window shift logic)
    // -------------------------------------------------------------------------
    // Shifts the window left each cycle and injects new samples into the bottom-right.
    // The top rows are supplied from the line buffers (previous rows).
    always_comb begin
        integer r,c,ch;

        // default: hold
        for (r=0; r<K_SIZE; r=r+1)
            for (c=0; c<K_SIZE; c=c+1)
                for (ch=0; ch<CHANNEL; ch=ch+1)
                    w_next[r][c][ch] = w_reg[r][c][ch];

        if (in_fire) begin
            // Shift window columns left
            for (r=0; r<K_SIZE; r=r+1)
                for (c=0; c<K_SIZE-1; c=c+1)
                    for (ch=0; ch<CHANNEL; ch=ch+1)
                        w_next[r][c][ch] = w_reg[r][c+1][ch];

            // Insert new rightmost column:
            // bottom row comes from current input; upper rows come from line buffers.
            for (ch=0; ch<CHANNEL; ch=ch+1) begin
                w_next[K_SIZE-1][K_SIZE-1][ch] = ch_in[ch];
                for (r=0; r<K_SIZE-1; r=r+1)
                   w_next[r][K_SIZE-1][ch] = line_buf[(K_SIZE-2) - r][0][ch];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Sequential state updates
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        integer i,x,r,c,ch;
        if (!rst_n || flush) begin
            // Reset counters and FIFO pointers; internal RAM contents do not need explicit clearing
            acc_cnt <= 0;
            col_cnt <= 0;
            count  <= 0;
            rd_ptr <= 0;
            wr_ptr <= 0;

        end else begin
            // Pop: advance read pointer
            if (pop) rd_ptr <= rd_ptr ^ 1'b1;

            // On input accept:
            //  - update counters
            //  - shift line buffers
            //  - update window registers
            if (in_fire) begin
                if (acc_cnt != {($bits(acc_cnt)){1'b1}})
                    acc_cnt <= acc_cnt + 1'b1;

                if (col_cnt == IMG_WIDTH-1) col_cnt <= '0;
                else                        col_cnt <= col_cnt + 1'b1;

                // Shift line buffers horizontally
                for (i=0; i<K_SIZE-1; i++)
                    for (x=0; x<IMG_WIDTH-1; x++)
                        for (ch=0; ch<CHANNEL; ch++)
                            line_buf[i][x][ch] <= line_buf[i][x+1][ch];

                // Cascade rows (end of line_buf[i-1] feeds line_buf[i])
                for (i=1; i<K_SIZE-1; i++)
                    for (ch=0; ch<CHANNEL; ch++)
                        line_buf[i][IMG_WIDTH-1][ch] <= line_buf[i-1][0][ch];

                // New sample enters the first line buffer row
                for (ch=0; ch<CHANNEL; ch++)
                    line_buf[0][IMG_WIDTH-1][ch] <= ch_in[ch];

                // Commit window registers
                for (r=0; r<K_SIZE; r++)
                    for (c=0; c<K_SIZE; c++)
                        for (ch=0; ch<CHANNEL; ch++)
                            w_reg[r][c][ch] <= w_next[r][c][ch];
            end

            // Push generated window into FIFO
            if (push) begin
                for (r=0; r<K_SIZE; r++)
                    for (c=0; c<K_SIZE; c++)
                        for (ch=0; ch<CHANNEL; ch++)
                            fifo_mem[wr_ptr][r][c][ch] <= w_next[r][c][ch];
                wr_ptr <= wr_ptr ^ 1'b1;
            end

            // FIFO occupancy update
            unique case ({push, pop})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: count <= count;
            endcase
        end
    end

endmodule
