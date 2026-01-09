`timescale 1ns/1ps

/* =============================================================================
 * Module: BramPaddedStreamer
 * Author: Yegun Shim
 * Date  : 2026-01-01
 * -----------------------------------------------------------------------------
 * ASSUMPTION (MANDATORY):
 *   Synchronous-read BRAM with EXACTLY 1-cycle latency:
 *     - mem_addr is sampled at cycle N (when we "issue" a read)
 *     - mem_rdata[*] becomes valid at cycle N+1
 *
 * PURPOSE:
 *   Stream a zero-padded frame with optional backpressure (out_ready).
 *     - BRAM stores: IMG_W x IMG_W (interior image)
 *     - Output stream: EXT_W x EXT_W where EXT_W = IMG_W + 2*PAD
 *
 * HANDSHAKE (downstream backpressure):
 *   - Output interface: out_valid / out_ready / out_data[]
 *   - When out_valid=1 and out_ready=0, THIS MODULE HOLDS:
 *       out_data, x_ext/y_ext position, mem_addr, and internal alignment state.
 *
 * ============================================================================= */

module BramPaddedStreamer #(
    parameter int unsigned N_BANKS = 8,
    parameter int unsigned MEM_DW  = 32,
    parameter int unsigned OUT_DW  = 9,
    parameter int unsigned ADDR_W  = 9,
    parameter int unsigned IMG_W   = 14,
    parameter int unsigned PAD     = 1,
    parameter logic [ADDR_W-1:0] BASE_ADDR = '0,
    parameter int unsigned STRIDE = 1,
    parameter bit SATURATE = 1
)(
    input  logic clk,
    input  logic rst_n,

    input  logic start,
    output logic busy,
    output logic done,

    // BRAM sync-read (1-cycle latency)
    output logic mem_en,
    output logic [ADDR_W-1:0] mem_addr,
    input  logic [MEM_DW-1:0] mem_rdata [0:N_BANKS-1],

    // Stream output with backpressure
    output logic                    out_valid,
    input  logic                    out_ready,
    output logic signed [OUT_DW-1:0] out_data [0:N_BANKS-1]
);

    localparam int unsigned EXT_W      = IMG_W + 2*PAD;
    localparam int unsigned TOTAL_REAL = IMG_W * IMG_W;
    localparam int unsigned TOTAL_EXT  = EXT_W * EXT_W;

    // Extended coordinate counters (current output position in padded space)
    logic [5:0] x_ext, y_ext;

    // Lookahead coordinates (next output position if we advance)
    logic [5:0] x_next, y_next;

    // Counters
    logic [$clog2(TOTAL_EXT):0]  ext_cnt;   // 0..TOTAL_EXT-1 (accepted outputs)
    logic [$clog2(TOTAL_REAL):0] real_idx;  // 0..TOTAL_REAL   (issued interior reads)

    // 1-cycle delayed marker:
    // issue_d == 1 means "mem_rdata corresponds to the current interior position"
    logic issue_d;

    // Address hold register:
    // For synchronous BRAM, mem_addr must remain stable during stalls.
    logic [ADDR_W-1:0] last_addr;

    // -------------------------------------------------------------------------
    // Convert MEM_DW -> OUT_DW (signed), optional saturation
    // -------------------------------------------------------------------------
    function automatic logic signed [OUT_DW-1:0] conv_width(input logic [MEM_DW-1:0] x);
        logic signed [MEM_DW-1:0] xs;
        logic signed [MEM_DW-1:0] maxv, minv;
        begin
            xs = $signed(x);

            if (!SATURATE) begin
                conv_width = xs[OUT_DW-1:0];
            end else begin
                maxv = $signed({1'b0, {(OUT_DW-1){1'b1}}});
                minv = $signed({1'b1, {(OUT_DW-1){1'b0}}});
                if (xs > maxv)      conv_width = maxv[OUT_DW-1:0];
                else if (xs < minv) conv_width = minv[OUT_DW-1:0];
                else                conv_width = xs[OUT_DW-1:0];
            end
        end
    endfunction

    // -------------------------------------------------------------------------
    // Address mapping: BASE + (idx * STRIDE)
    // - Works for STRIDE != 1
    // - Avoids div/mod
    // - Truncates to ADDR_W (ADDR_W must be sized sufficiently in the system)
    // -------------------------------------------------------------------------
    function automatic logic [ADDR_W-1:0] addr_for(input logic [$clog2(TOTAL_REAL):0] i);
        longint unsigned prod;
        logic [ADDR_W-1:0] offs;
        begin
            prod = longint'($unsigned(i)) * longint'(STRIDE);
            offs = prod[ADDR_W-1:0];
            addr_for = BASE_ADDR + offs;
        end
    endfunction

    // Padding detection in extended coordinate space
    function automatic logic is_pad_pos(input logic [5:0] xx, input logic [5:0] yy);
        begin
            is_pad_pos = (xx < PAD) || (xx >= (PAD + IMG_W)) ||
                         (yy < PAD) || (yy >= (PAD + IMG_W));
        end
    endfunction

    // -------------------------------------------------------------------------
    // Lookahead coordinate computation (no div/mod)
    // -------------------------------------------------------------------------
    always_comb begin
        if (x_ext == EXT_W-1) begin
            x_next = 0;
            if (y_ext == EXT_W-1) y_next = 0;
            else                  y_next = y_ext + 1;
        end else begin
            x_next = x_ext + 1;
            y_next = y_ext;
        end
    end

    // Current/next padding flags
    logic is_pad_cur, is_pad_nxt;

    always_comb begin
        is_pad_cur = is_pad_pos(x_ext,  y_ext);
        is_pad_nxt = is_pad_pos(x_next, y_next);
    end

    // -------------------------------------------------------------------------
    // Handshake step:
    // - out_valid is asserted whenever busy=1
    // - We "advance" only when downstream accepts (out_ready=1)
    // -------------------------------------------------------------------------
    wire out_fire = busy && out_ready;

    // Prefetch decision:
    // Issue a BRAM read only when:
    //   (1) downstream accepts current output (out_fire)
    //   (2) NEXT output position is interior (not padding)
    //   (3) we still have interior pixels remaining to read
    logic issue_fire;
    always_comb begin
        issue_fire = out_fire && (!is_pad_nxt) && (real_idx < TOTAL_REAL);
    end

    assign mem_en = issue_fire;

    // -------------------------------------------------------------------------
    // mem_addr behavior (stall-safe):
    // For synchronous BRAM, mem_addr must be stable unless issuing a new read.
    // We hold last_addr across stalls so BRAM output alignment remains correct.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            last_addr <= BASE_ADDR;
        end else if (!busy && start) begin
            last_addr <= BASE_ADDR;
        end else if (issue_fire) begin
            last_addr <= addr_for(real_idx);
        end
        // else: HOLD last_addr
    end

    always_comb begin
        if (issue_fire) mem_addr = addr_for(real_idx);
        else            mem_addr = last_addr;
    end

    // -------------------------------------------------------------------------
    // Output logic (combinational):
    // Must remain stable when stalled because x_ext/y_ext/issue_d/mem_addr hold.
    // -------------------------------------------------------------------------
    always_comb begin
        out_valid = busy;

        if (!busy) begin
            for (int b=0; b<N_BANKS; b++) out_data[b] = '0;

        end else if (is_pad_cur) begin
            for (int b=0; b<N_BANKS; b++) out_data[b] = '0;

        end else begin
            // Interior position:
            // mem_rdata is valid only if we issued a read for this position in the prior cycle.
            if (issue_d) begin
                for (int b=0; b<N_BANKS; b++) out_data[b] = conv_width(mem_rdata[b]);
            end else begin
                // Safety: If alignment ever breaks, output zeros (should not happen in correct use).
                for (int b=0; b<N_BANKS; b++) out_data[b] = '0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Sequential logic:
    // Advance all internal state only on out_fire (downstream acceptance).
    // When stalled, EVERYTHING is held to guarantee stable output + BRAM alignment.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy     <= 1'b0;
            done     <= 1'b0;
            x_ext    <= '0;
            y_ext    <= '0;
            ext_cnt  <= '0;
            real_idx <= '0;
            issue_d  <= 1'b0;

        end else begin
            done <= 1'b0;

            // Start a new frame
            if (!busy && start) begin
                busy     <= 1'b1;
                x_ext    <= 0;
                y_ext    <= 0;
                ext_cnt  <= 0;
                real_idx <= 0;
                issue_d  <= 1'b0;
            end

            if (busy) begin
                if (out_fire) begin
                    // (1) Update real_idx only when we issued a read (prefetch for NEXT interior)
                    if (issue_fire) begin
                        real_idx <= real_idx + 1'b1;
                    end

                    // (2) Pipeline marker: indicates whether mem_rdata is valid next cycle
                    issue_d <= issue_fire;

                    // (3) Advance extended scan position (x_ext, y_ext)
                    if (x_ext == EXT_W-1) begin
                        x_ext <= 0;
                        if (y_ext == EXT_W-1) y_ext <= 0;
                        else                  y_ext <= y_ext + 1'b1;
                    end else begin
                        x_ext <= x_ext + 1'b1;
                    end

                    // (4) Finish after TOTAL_EXT accepted outputs
                    if (ext_cnt == (TOTAL_EXT-1)) begin
                        busy    <= 1'b0;
                        done    <= 1'b1;   // 1-cycle pulse
                        ext_cnt <= '0;
                        issue_d <= 1'b0;   // clean for next start
                    end else begin
                        ext_cnt <= ext_cnt + 1'b1;
                    end
                end
                // else HOLD EVERYTHING
            end
        end
    end

endmodule
