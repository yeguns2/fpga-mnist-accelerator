`timescale 1ns/1ps

/* =============================================================================
 * Module: ComputeEngine
 * Author : Yegun Shim
 * Date   : 2026-01-06
 * -----------------------------------------------------------------------------
 * Overview
 *   Conv2 compute core that consumes 3x3 windows (with IN_CH channels) from a
 *   LineBuffer, applies per-filter MAC + bias + ReLU, then performs 2x2 maxpool.
 *
 * Interfaces
 *   - Input (from LineBuffer):
 *       valid_in + (row_idx, col_idx) + window[3][3][IN_CH]
 *       ready indicates this engine can accept a new window "token".
 *
 *   - Weight/Bias:
 *       w_we/w_data streams signed 9-bit weights into internal w_reg.
 *       bias[OUT_CH] provides 32-bit signed bias per output channel.
 *
 *   - Output:
 *       valid_out pulses when a pooled output sample is produced.
 *       data_out is the pooled ReLU result (32-bit signed).
 *       out_ch/out_row/out_col describe which pooled coordinate this sample maps to.
 *
 * Pipeline
 *   Stage 0 : accept a window and seed the per-filter loop (one filter per cycle)
 *   Stage 1 : multiply window * weights
 *   Stage 2 : sum across IN_CH for each (r,c)
 *   Stage 3 : sum across (r,c), add bias, apply ReLU
 *   Stage 4 : 2x2 maxpool (horizontal temp + vertical line buffer)
 *
 * Notes
 *   - The per-filter loop is implemented by iterating f_cnt across OUT_CH.
 *   - The ready/valid handshake is intentionally conservative: ready is asserted
 *     only in IDLE (i.e., accepts the next token once per OUT_CH cycles).
 * ============================================================================= */

module ComputeEngine #(
    parameter int unsigned IN_CH      = 8,
    parameter int unsigned OUT_CH     = 16,
    parameter int unsigned K_SIZE     = 3,
    parameter int unsigned IMG_WIDTH  = 14,
    parameter int unsigned DATA_WIDTH = 9
)(
    input  logic clk,
    input  logic rst_n,

    // -------------------------------------------------------------------------
    // Control Interface (Handshake with LineBuffer)
    // -------------------------------------------------------------------------
    output logic ready,
    output logic busy,

    // -------------------------------------------------------------------------
    // Data Input Interface
    // -------------------------------------------------------------------------
    input  logic valid_in,
    input  logic [5:0] col_idx,
    input  logic [5:0] row_idx,
    input  logic signed [DATA_WIDTH-1:0] window [0:K_SIZE-1][0:K_SIZE-1][0:IN_CH-1],

    // -------------------------------------------------------------------------
    // Weight & Bias Interface
    // -------------------------------------------------------------------------
    input  logic w_we,
    input  logic signed [DATA_WIDTH-1:0] w_data,
    input  logic signed [31:0] bias [0:OUT_CH-1],

    // -------------------------------------------------------------------------
    // Output Interface
    // -------------------------------------------------------------------------
    output logic valid_out,
    output logic signed [31:0] data_out,
    output logic [4:0] out_ch,      // 0..15
    output logic [2:0] out_row,     // 0..6 (row >> 1)
    output logic [2:0] out_col      // 0..6 (col >> 1)
);

    // =========================================================================
    // 0) Local State
    // =========================================================================
    typedef enum logic [1:0] {IDLE, COMPUTE} state_t;
    state_t state;

    // =========================================================================
    // 1) Handshake Logic
    // =========================================================================
    // This engine only accepts a new (row,col,window) token when IDLE.
    // Once accepted, it iterates over all OUT_CH filters internally.
    assign ready = (state == IDLE);

    // "fire" means we latched a new window token this cycle.
    wire fire = valid_in && ready;

    // =========================================================================
    // 2) Weight Storage (streamed programming interface)
    // =========================================================================
    // w_reg[f][r][c][ch] holds the Conv2 weight for output filter f,
    // at spatial tap (r,c) and input channel ch.
    (* ram_style = "distributed" *)
    logic signed [DATA_WIDTH-1:0] w_reg
        [0:OUT_CH-1][0:K_SIZE-1][0:K_SIZE-1][0:IN_CH-1];

    // Sequential programming pointers for the weight stream.
    // Order: ch -> c -> r -> f (i.e., fastest: input-channel).
    logic [4:0] ptr_f;
    logic [3:0] ptr_r;
    logic [3:0] ptr_c;
    logic [4:0] ptr_ch;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ptr_f  <= 0;
            ptr_r  <= 0;
            ptr_c  <= 0;
            ptr_ch <= 0;
        end else if (w_we) begin
            // Store one signed weight element per cycle.
            w_reg[ptr_f][ptr_r][ptr_c][ptr_ch] <= w_data;

            // Advance pointers through the 4D weight tensor.
            if (ptr_ch == IN_CH - 1) begin
                ptr_ch <= 0;
                if (ptr_c == K_SIZE - 1) begin
                    ptr_c <= 0;
                    if (ptr_r == K_SIZE - 1) begin
                        ptr_r <= 0;
                        if (ptr_f == OUT_CH - 1) ptr_f <= 0;
                        else                     ptr_f <= ptr_f + 1;
                    end else begin
                        ptr_r <= ptr_r + 1;
                    end
                end else begin
                    ptr_c <= ptr_c + 1;
                end
            end else begin
                ptr_ch <= ptr_ch + 1;
            end
        end
    end

    // =========================================================================
    // 3) Pipeline Control & Registers
    // =========================================================================
    // f_cnt iterates output filters while processing a single input token.
    logic [4:0] f_cnt;

    // Pipeline valid flags (Stage 1..4).
    logic val_s1, val_s2, val_s3, val_s4;

    // Pipeline coordinates & filter indices (carried through stages).
    logic [5:0] col_s1, row_s1, col_s2, row_s2, col_s3, row_s3, col_s4, row_s4;
    logic [4:0] f_idx_s1, f_idx_s2, f_idx_s3, f_idx_s4;

    // Pipeline data:
    // - px_pipe: latched input window token (Stage 0)
    // - mult_res: per-tap, per-channel multiplication results (Stage 1)
    // - part_sum: per-tap sum across channels (Stage 2)
    // - total_sum/relu_out: spatial sum + bias + activation (Stage 3)
    logic signed [DATA_WIDTH-1:0] px_pipe  [0:K_SIZE-1][0:K_SIZE-1][0:IN_CH-1];
    logic signed [31:0]           mult_res [0:K_SIZE-1][0:K_SIZE-1][0:IN_CH-1];
    logic signed [31:0]           part_sum [0:K_SIZE-1][0:K_SIZE-1];
    logic signed [31:0]           relu_out;
    logic signed [31:0]           total_sum;

    // =========================================================================
    // 4) Max Pooling Internal Memories
    // =========================================================================
    // Horizontal temp: holds max of (col even, col odd) pair for each filter.
    (* ram_style = "distributed" *)
    logic signed [31:0] pool_horz_buf [0:OUT_CH-1];

    // Vertical line buffer (after horizontal max):
    // Stores the max from the even row to combine with odd row.
    (* ram_style = "distributed" *)
    logic signed [31:0] pool_line_buf [0:(IMG_WIDTH >> 1)-1][0:OUT_CH-1];

    // =========================================================================
    // 5) Main Pipeline (single always_ff)
    // =========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            // Control
            state     <= IDLE;
            busy      <= 0;

            // Output
            valid_out <= 0;
            data_out  <= 0;
            out_ch    <= 0;
            out_row   <= 0;
            out_col   <= 0;

            // Per-filter iteration
            f_cnt     <= 0;

            // Pipeline valids / indices init
            val_s1    <= 0;
            val_s2    <= 0;
            val_s3    <= 0;
            val_s4    <= 0;
            f_idx_s1  <= 0;

            // Pooling memories reset
            for (int i=0; i<OUT_CH; i++) pool_horz_buf[i] <= '0;
            for (int c=0; c<(IMG_WIDTH>>1); c++)
                for (int i=0; i<OUT_CH; i++)
                    pool_line_buf[c][i] <= '0;

        end else begin
            // Default: output valid is a pulse, so clear unless explicitly set.
            valid_out <= 0;

            // -----------------------------------------------------------------
            // STAGE 0: State machine & per-filter iteration control
            // -----------------------------------------------------------------
            case (state)
                IDLE: begin
                    busy <= 0;

                    // Accept one window token. This seeds the pipeline and starts
                    // iterating across OUT_CH filters (one filter per cycle).
                    if (fire) begin
                        busy  <= 1;
                        state <= COMPUTE;

                        // Start from filter 0 now, then proceed with filter 1..(OUT_CH-1).
                        f_cnt    <= 1;
                        val_s1   <= 1;

                        // Latch the window token and its coordinates.
                        px_pipe  <= window;
                        col_s1   <= col_idx;
                        row_s1   <= row_idx;
                        f_idx_s1 <= 0;
                    end else begin
                        val_s1 <= 0;
                    end
                end

                COMPUTE: begin
                    // While in COMPUTE, we continuously feed Stage 1 with the same
                    // window token but different filter index each cycle.
                    val_s1   <= 1;
                    f_idx_s1 <= f_cnt;

                    // Coordinates remain associated with the original token.
                    col_s1 <= col_s1;
                    row_s1 <= row_s1;

                    // End after the last filter is issued.
                    if (f_cnt == OUT_CH - 1) begin
                        state <= IDLE;
                        busy  <= 0;
                    end else begin
                        f_cnt <= f_cnt + 1;
                        busy  <= 1;
                    end
                end
            endcase

            // -----------------------------------------------------------------
            // STAGE 1: Multiply (window taps x weights)
            // -----------------------------------------------------------------
            if (val_s1) begin
                for (int r=0; r<K_SIZE; r++) begin
                    for (int c=0; c<K_SIZE; c++) begin
                        for (int ch=0; ch<IN_CH; ch++) begin
                            // Multiply uses explicit 32-bit casting to encourage DSP inference.
                            mult_res[r][c][ch] <= 32'(px_pipe[r][c][ch]) * 32'(w_reg[f_idx_s1][r][c][ch]);
                        end
                    end
                end

                // Carry metadata
                f_idx_s2 <= f_idx_s1;
                col_s2   <= col_s1;
                row_s2   <= row_s1;
                val_s2   <= 1;
            end else begin
                val_s2 <= 0;
            end

            // -----------------------------------------------------------------
            // STAGE 2: Sum across IN_CH for each (r,c)
            // -----------------------------------------------------------------
            if (val_s2) begin
                for (int r=0; r<K_SIZE; r++) begin
                    for (int c=0; c<K_SIZE; c++) begin
                        // psum accumulates across channels for one spatial tap.
                        automatic int psum = 0;
                        for (int ch=0; ch<IN_CH; ch++) begin
                            psum += mult_res[r][c][ch];
                        end
                        part_sum[r][c] <= psum;
                    end
                end

                // Carry metadata
                f_idx_s3 <= f_idx_s2;
                col_s3   <= col_s2;
                row_s3   <= row_s2;
                val_s3   <= 1;
            end else begin
                val_s3 <= 0;
            end

            // -----------------------------------------------------------------
            // STAGE 3: Spatial sum + bias + ReLU
            // -----------------------------------------------------------------
            if (val_s3) begin
                total_sum = 0;
                for (int r=0; r<K_SIZE; r++)
                    for (int c=0; c<K_SIZE; c++)
                        total_sum += part_sum[r][c];

                // Add per-filter bias.
                total_sum += bias[f_idx_s3];

                // ReLU activation.
                if (total_sum < 0) relu_out <= 0;
                else               relu_out <= total_sum;

                // Carry metadata
                f_idx_s4 <= f_idx_s3;
                col_s4   <= col_s3;
                row_s4   <= row_s3;
                val_s4   <= 1;
            end else begin
                val_s4 <= 0;
            end

            // -----------------------------------------------------------------
            // STAGE 4: 2x2 Max Pooling (no extra wait-states)
            // -----------------------------------------------------------------
            if (val_s4) begin
                logic signed [31:0] h_max;
                logic signed [31:0] final_max;

                if (col_s4[0] == 0) begin
                    // Even column: store ReLU result for horizontal compare.
                    pool_horz_buf[f_idx_s4] <= relu_out;
                end else begin
                    // Odd column: take max across the 2 horizontal samples.
                    h_max = (relu_out > pool_horz_buf[f_idx_s4]) ? relu_out : pool_horz_buf[f_idx_s4];

                    if (row_s4[0] == 0) begin
                        // Even row: store horizontal max for vertical compare.
                        pool_line_buf[col_s4[5:1]][f_idx_s4] <= h_max;
                    end else begin
                        // Odd row: take max across the 2 vertical samples => pooled output.
                        final_max =
                            (h_max > pool_line_buf[col_s4[5:1]][f_idx_s4]) ?
                            h_max : pool_line_buf[col_s4[5:1]][f_idx_s4];

                        valid_out <= 1;
                        data_out  <= final_max;

                        // Pooled coordinate mapping (divide by 2 via shift).
                        out_ch  <= f_idx_s4;
                        out_row <= row_s4[3:1];
                        out_col <= col_s4[3:1];
                    end
                end
            end
        end
    end

endmodule
