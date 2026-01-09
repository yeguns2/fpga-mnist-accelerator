`timescale 1ns/1ps

/* =============================================================================
 * Module: Accelerator_v1_0
 * Author : Yegun Shim
 * Date   : 2026-01-08
 * -----------------------------------------------------------------------------
 * Summary
 *   AXI4-Lite Slave (control/status) + AXI4 Master (memory-mapped BRAM (or DDR) reads & writes)
 *   Implements a small CNN pipeline:
 *     1) Load 28x28 image (uint8 packed into 32-bit words) into local cache
 *     2) Load Conv1 weights/bias (8 filters, 3x3) into local caches
 *     3) Load Conv2 weights/bias (16 filters, 8ch󫢫, weights streamed as 8-bit) into ComputeEngine
 *     4) Conv1 compute: (Conv -> ReLU -> 2x2 MaxPool) producing 8ch�14�14
 *     5) Stream padded 16�16 view (PAD=1 around 14�14) into Conv2 linebuffer/engine
 *     6) Conv2 compute produces 16ch󬱟 and writes results back via AXI write bursts
 *
 * Clocking
 *   - s00_axi_aclk : AXI4-Lite clock domain (register interface)
 *   - m00_axi_aclk : AXI4 Master + compute + BRAM banks clock domain
 *
 * Notes / Assumptions
 *   - BRAM banks are inferred as synchronous-read (1-cycle latency) memories.
 *   - BramPaddedStreamer_1clk_clean is designed for 1-cycle BRAM latency + backpressure.
 *   - Conv1 input image is stored in img_cache[] as 196 x 32-bit words (784 bytes total).
 *   - Output BRAM banks store:
 *       * bank[0..7], addr 0..195 : Conv1 pooled output 8ch�14�14 (one pixel per address)
 *       * bank[0..15], addr 256..(256+48) : Conv2 pooled output 16ch󬱟 (one pixel per address)
 * ============================================================================= */

module Accelerator_v1_0 #(
    parameter int unsigned C_S00_AXI_DATA_WIDTH = 32,
    parameter int unsigned C_S00_AXI_ADDR_WIDTH = 6,
    parameter int unsigned C_M00_AXI_TARGET_SLAVE_BASE_ADDR = 32'hC000_0000,
    parameter int unsigned C_M00_AXI_ID_WIDTH     = 1,
    parameter int unsigned C_M00_AXI_ADDR_WIDTH   = 32,
    parameter int unsigned C_M00_AXI_DATA_WIDTH   = 32
)(
    // -------------------------------------------------------------------------
    // S00_AXI (AXI4-Lite Slave Interface) : Control / Status Registers
    // -------------------------------------------------------------------------
    input  logic s00_axi_aclk, s00_axi_aresetn,
    input  logic [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_awaddr,
    input  logic s00_axi_awvalid, output logic s00_axi_awready,
    input  logic [C_S00_AXI_DATA_WIDTH-1:0] s00_axi_wdata,
    input  logic [3:0] s00_axi_wstrb, input logic s00_axi_wvalid, output logic s00_axi_wready,
    output logic [1:0] s00_axi_bresp, output logic s00_axi_bvalid, input logic s00_axi_bready,
    input  logic [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_araddr,
    input  logic s00_axi_arvalid, output logic s00_axi_arready,
    output logic [C_S00_AXI_DATA_WIDTH-1:0] s00_axi_rdata,
    output logic [1:0] s00_axi_rresp, output logic s00_axi_rvalid, input logic s00_axi_rready,

    // -------------------------------------------------------------------------
    // M00_AXI (AXI4 Master Interface) : External Memory Transactions
    //   - Uses AR/R to fetch image + weights
    //   - Uses AW/W/B to write back final results
    // -------------------------------------------------------------------------
    input  logic m00_axi_aclk, m00_axi_aresetn,
    output logic [C_M00_AXI_ID_WIDTH-1:0] m00_axi_awid,
    output logic [C_M00_AXI_ADDR_WIDTH-1:0] m00_axi_awaddr,
    output logic [7:0] m00_axi_awlen, output logic [2:0] m00_axi_awsize, output logic [1:0] m00_axi_awburst,
    output logic m00_axi_awlock, output logic [3:0] m00_axi_awcache,
    output logic [2:0] m00_axi_awprot, output logic [3:0] m00_axi_awqos, output logic m00_axi_awvalid,
    input  logic m00_axi_awready,
    output logic [C_M00_AXI_DATA_WIDTH-1:0] m00_axi_wdata, output logic [3:0] m00_axi_wstrb, output logic m00_axi_wlast, output logic m00_axi_wvalid,
    input  logic m00_axi_wready,
    input  logic [C_M00_AXI_ID_WIDTH-1:0] m00_axi_bid, input logic [1:0] m00_axi_bresp, input logic m00_axi_bvalid, output logic m00_axi_bready,

    // -------------------------------------------------------------------------
    // Master Read Ports (AR/R) : Used for image + weights load
    // -------------------------------------------------------------------------
    output logic [C_M00_AXI_ID_WIDTH-1:0] m00_axi_arid, output logic [C_M00_AXI_ADDR_WIDTH-1:0] m00_axi_araddr,
    output logic [7:0] m00_axi_arlen, output logic [2:0] m00_axi_arsize, output logic [1:0] m00_axi_arburst,
    output logic m00_axi_arlock, output logic [3:0] m00_axi_arcache,
    output logic [2:0] m00_axi_arprot, output logic [3:0] m00_axi_arqos, output logic m00_axi_arvalid,
    input  logic m00_axi_arready,
    input  logic [C_M00_AXI_ID_WIDTH-1:0] m00_axi_rid, input logic [C_M00_AXI_DATA_WIDTH-1:0] m00_axi_rdata,
    input  logic [1:0] m00_axi_rresp, input logic m00_axi_rlast, input logic m00_axi_rvalid, output logic m00_axi_rready
);

    // =========================================================================
    //  FSM State Encoding
    // =========================================================================
    typedef enum logic [4:0] {
        IDLE, LOAD_IMG_CMD, LOAD_IMG_DATA, LOAD_W_CMD, LOAD_W_DATA,
        LOAD_W2_CMD, LOAD_W2_DATA, LOAD_FC_BIAS_CMD, LOAD_FC_BIAS_DATA, LOAD_FC_W_CMD, LOAD_FC_W_DATA,
        RUN_CONV1, RUN_CONV2, RUN_FC,
        FC_CHECK_MAX,
        WRITE_FC_AW, WRITE_FC_W, WRITE_FC_B,

        /* DEBUG :
         * Conv2 result dump / profiling (writes intermediate results back to memory block)
         *
         * WRITE_AW, WRITE_PREFETCH, WRITE_PREFETCH_2, WRITE_W, WRITE_B,
         */

        DONE
    } state_t;

    state_t state;

    // =========================================================================
    //  AXI4-Lite Register Map (Control / Status)
    // =========================================================================
    // reg_ctrl[0] : start (edge-detected internally)
    // reg_ctrl[1] : clear done flag (clears reg_status[1])
    // reg_ctrl[2] : weight-load-only mode (enters weight-loading sequence from IDLE)
    //
    // reg_status[0] : busy flag
    // reg_status[1] : done flag (sticky until cleared)
    //
    // reg_img_base     : base address of packed MNIST input (196 words)
    // reg_weight_base  : base address of Conv1 weights/bias
    // reg_weight2_base : base address of Conv2 weights/bias
    // reg_fc_w_base    : base address of FC weights (10 x 196 words)
    // reg_fc_b_base    : base address of FC bias (10 words)
    // reg_result_base  : base address of output logits (10 words)
    // reg_cycles       : compute-cycle counter (performance/benchmark)
    // reg_res_label    : predicted label (argmax over logits)
    logic [31:0] reg_ctrl, reg_status, reg_img_base, reg_weight_base, reg_weight2_base, reg_fc_w_base, reg_fc_b_base, reg_result_base;
    logic [31:0] reg_res_label, reg_cycles;

    logic [C_S00_AXI_ADDR_WIDTH-1:0] awaddr_q, araddr_q;
    logic aw_en, ar_inflight;

    // Generic counters used across bursts / loops
    logic [7:0]  cnt;               // burst beat counter (0..195 etc.)
    logic [3:0]  cur_filter_load;   // weight-bank / class index during loading (0..15 or 0..9)

    //debug reg
    // logic [31:0] reg_debug;

    // -------------------------------------------------------------------------
    // AXI4-Lite single-beat write + read handling (minimal register file)
    // -------------------------------------------------------------------------
    always_ff @(posedge s00_axi_aclk) begin
        if (!s00_axi_aresetn) begin
            s00_axi_awready <= 0; s00_axi_wready <= 0; s00_axi_bvalid <= 0; aw_en <= 1;
            reg_ctrl <= 0; reg_img_base <= 0; reg_weight_base <= 0; reg_weight2_base <= 0;
            s00_axi_arready <= 0; s00_axi_rvalid <= 0; ar_inflight <= 0;
            reg_fc_w_base <= 0; reg_fc_b_base <= 0; reg_result_base <= 0;
        end else begin
            // Write address handshake (accept when W also valid; single outstanding)
            if (~s00_axi_awready & s00_axi_awvalid & s00_axi_wvalid & aw_en) begin
                s00_axi_awready <= 1; awaddr_q <= s00_axi_awaddr; aw_en <= 0;
            end else s00_axi_awready <= 0;

            // Write data handshake (paired with AW)
            if (~s00_axi_wready & s00_axi_wvalid & s00_axi_awvalid & aw_en) s00_axi_wready <= 1;
            else s00_axi_wready <= 0;

            // Register write on AW+W accept
            if (s00_axi_awready & s00_axi_wready) begin
                case(awaddr_q[5:2])
                    0:  reg_ctrl       <= s00_axi_wdata;
                    1:  reg_img_base   <= s00_axi_wdata;
                    2:  reg_weight_base<= s00_axi_wdata;
                    3:  reg_weight2_base<= s00_axi_wdata;
                    4:  reg_fc_w_base  <= s00_axi_wdata;
                    5:  reg_fc_b_base  <= s00_axi_wdata;
                    10: reg_result_base<= s00_axi_wdata;
                    default: ;
                endcase
            end

            // Write response channel
            if (~s00_axi_bvalid & s00_axi_bready & s00_axi_awready & s00_axi_wready) s00_axi_bvalid <= 1;
            else if (s00_axi_bvalid & s00_axi_bready) begin s00_axi_bvalid <= 0; aw_en <= 1; end

            // Read address handshake (single inflight)
            if (~ar_inflight & s00_axi_arvalid) begin
                s00_axi_arready <= 1; araddr_q <= s00_axi_araddr; ar_inflight <= 1;
            end else s00_axi_arready <= 0;

            // Read data return (combinational mux into registered R channel)
            if (ar_inflight & ~s00_axi_rvalid) begin
                s00_axi_rvalid <= 1;
                case(araddr_q[5:2])
                    0:  s00_axi_rdata <= reg_ctrl;
                    6:  s00_axi_rdata <= reg_status;
                    7:  s00_axi_rdata <= reg_cycles;
                  //8:  s00_axi_rdata <= reg_debug;
                    9:  s00_axi_rdata <= reg_res_label;
                    default: s00_axi_rdata <= 0;
                endcase
            end else if (s00_axi_rvalid & s00_axi_rready) begin
                s00_axi_rvalid <= 0; ar_inflight <= 0;
            end
        end
    end

    // =========================================================================
    //  Core Local Storage (BRAM / Registers)
    // =========================================================================

    // Local image cache: 196 x 32-bit words = 784 bytes (MNIST 28x28 uint8)
    (* ram_style = "block" *) logic [31:0] img_cache [0:195];

    // Conv1 parameters (8 filters): 3x3 weights + bias
    logic signed [31:0] w_cache [0:7][0:8];
    logic signed [31:0] b_cache [0:7];

    // Conv2 bias (16 filters). Conv2 weights stream into ComputeEngine.
    logic signed [31:0] b2_cache [0:15];

    // Conv1 pooling helper buffers:
    // - horz_temp[] holds horizontal max across (ox even/odd)
    // - line_buf[][] holds vertical partial max across (oy even/odd)
    logic signed [31:0] line_buf [0:13][0:7];

    // Fully-connected parameters:
    // - fc_b_cache[] : 10 biases
    // - fc weight RAMs: 10 independent banks (one bank per class)
    logic signed [31:0] fc_b_cache [0:9];
    logic [7:0]  fc_rd_addr;                 // 0..195
    logic [31:0] fc_w_rdata [0:9];

    // FC weight RAM banks (10 banks, synchronous 1-cycle read)
    genvar gi;
    generate
      for (gi = 0; gi < 10; gi++) begin : FC_BANKS
        (* ram_style="block" *) logic [31:0] ram [0:195];
        logic [31:0] rd_q;

        // WRITE port: only during LOAD_FC_W_DATA and only for selected class
        always_ff @(posedge m00_axi_aclk) begin
          if (state == LOAD_FC_W_DATA && m00_axi_rvalid && m00_axi_rready &&
              (cur_filter_load == gi)) begin
            ram[cnt[7:0]] <= m00_axi_rdata;     // cnt is 0..195
          end
        end

        // READ port: synchronous 1-cycle
        always_ff @(posedge m00_axi_aclk) begin
          rd_q <= ram[fc_rd_addr];
        end

        assign fc_w_rdata[gi] = rd_q;
      end
    endgenerate

    // =========================================================================
    //  Output BRAM Banks (16 banks)
    // =========================================================================
    // Each bank is a 512-depth synchronous-read RAM with 1-cycle latency.
    // - Writes: controlled by out_buf_we[] + out_buf_addr_write + out_buf_wdata[]
    // - Read port address mux:
    //     * RUN_CONV2: streamer reads Conv1 pooled output from banks 0..7
    //     * otherwise: FSM reads banks 0..15 (FC reads Conv2-pooled data using fsm_addr_read)
    logic [15:0] out_buf_we;
    logic [8:0]  out_buf_addr_write;
    logic [31:0] out_buf_wdata [0:15];

    logic [8:0]  out_buf_addr_read;
    logic [31:0] out_buf_rdata [0:15];

    logic [8:0] fsm_addr_read;           // FSM-driven read address
    logic streamer_mem_en;               // streamer read enable (observability)
    logic [8:0] streamer_addr_read;      // streamer read address

    // Read-port address mux
    assign out_buf_addr_read = (state == RUN_CONV2) ? streamer_addr_read : fsm_addr_read;

    genvar i;
    generate
        for (i = 0; i < 16; i++) begin : BRAM_BANKS
            (* ram_style = "block" *) logic signed [31:0] ram [0:511];

            // Write port (synchronous)
            always_ff @(posedge m00_axi_aclk) begin
                if (out_buf_we[i]) ram[out_buf_addr_write] <= out_buf_wdata[i];
            end

            // Read port (synchronous, 1-cycle latency)
            logic [31:0] ram_out_reg;
            always_ff @(posedge m00_axi_aclk) ram_out_reg <= ram[out_buf_addr_read];
            assign out_buf_rdata[i] = ram_out_reg;
        end
    endgenerate

    // =========================================================================
    //  Streamer Integration: Conv1 pooled output -> padded stream -> Conv2
    // =========================================================================
    logic streamer_start, streamer_busy, streamer_done;
    logic streamer_out_valid, streamer_out_ready;
    logic signed [8:0] streamer_out_data [0:7];

    // Provide streamer the read data from banks 0..7 only
    logic [31:0] streamer_read_rdata [0:7];
    always_comb begin
        for(int k=0; k<8; k++) streamer_read_rdata[k] = out_buf_rdata[k];
    end

    BramPaddedStreamer #(
        .N_BANKS(8),
        .MEM_DW(32),
        .OUT_DW(9),
        .ADDR_W(9),
        .IMG_W(14),
        .PAD(1),
        .BASE_ADDR('0),
        .STRIDE(1),
        .SATURATE(1)
    ) cv2_bramStreamer (
        .clk        (m00_axi_aclk),
        .rst_n      (m00_axi_aresetn),
        .start      (streamer_start),
        .busy       (streamer_busy),
        .done       (streamer_done),

        .mem_en     (streamer_mem_en),
        .mem_addr   (streamer_addr_read),
        .mem_rdata  (streamer_read_rdata),

        .out_valid  (streamer_out_valid),
        .out_ready  (streamer_out_ready),
        .out_data   (streamer_out_data)
    );

    // =========================================================================
    //  Conv2 Pipeline: LineBuffer + ComputeEngine
    // =========================================================================
    logic c2_busy, c2_valid_out, c2_ready;
    logic signed [31:0] c2_data_out;
    logic        c2_w_we;
    logic [8:0]  c2_w_data;
    logic [4:0]  c2_out_ch;
    logic [2:0]  c2_out_row, c2_out_col;

    logic        lb_valid_out;
    logic signed [8:0] lb_window [0:2][0:2][0:7];
    logic [3:0]  in_col_14, in_row_14;

    LineBuffer #(
        .CHANNEL   (8),
        .K_SIZE    (3),
        .IMG_WIDTH (16),
        .DW        (9)
    ) cv2_lineBuffer (
        .clk        (m00_axi_aclk),
        .rst_n      (m00_axi_aresetn),
        .flush      (streamer_start),
        .in_valid   (streamer_out_valid),
        .in_ready   (streamer_out_ready),
        .ch_in      (streamer_out_data),
        .win_valid  (lb_valid_out),
        .win_ready  (c2_ready),
        .window     (lb_window)
    );

    ComputeEngine #(
        .IN_CH      (8),
        .OUT_CH     (16),
        .K_SIZE     (3),
        .IMG_WIDTH  (14),
        .DATA_WIDTH (9)
    ) cv2_computeEngine (
        .clk        (m00_axi_aclk),
        .rst_n      (m00_axi_aresetn),
        .busy       (c2_busy),
        .ready      (c2_ready),
        .valid_in   (lb_valid_out),
        .col_idx    (in_col_14),
        .row_idx    (in_row_14),
        .window     (lb_window),

        .w_we       (c2_w_we),
        .w_data     (c2_w_data),
        .bias       (b2_cache),

        .valid_out  (c2_valid_out),
        .data_out   (c2_data_out),
        .out_ch     (c2_out_ch),
        .out_row    (c2_out_row),
        .out_col    (c2_out_col)
    );

    // Input coordinate tracking for ComputeEngine (advances when a window is accepted)
    wire c2_fire = lb_valid_out && c2_ready;
    always_ff @(posedge m00_axi_aclk) begin
        if (!m00_axi_aresetn) begin
            in_col_14 <= 0;
            in_row_14 <= 0;
        end else if (streamer_start) begin
            in_col_14 <= 0;
            in_row_14 <= 0;
        end else if (c2_fire) begin
            if (in_col_14 == 13) begin
                in_col_14 <= 0;
                in_row_14 <= in_row_14 + 1;
            end else begin
                in_col_14 <= in_col_14 + 1;
            end
        end
    end

    // =========================================================================
    //  Conv1 Address Calculation (3x3 with padding) for packed uint8 image
    // =========================================================================
    logic [3:0]  k_idx;
    logic [4:0]  oy, ox;

    logic signed [2:0]  ky, kx;
    logic signed [8:0]  iy, ix;
    logic [5:0]  iy_u, ix_u;
    logic [9:0]  addr10;

    always_comb begin
        case (k_idx)
            0: begin ky=0; kx=0; end 1: begin ky=0; kx=1; end 2: begin ky=0; kx=2; end
            3: begin ky=1; kx=0; end 4: begin ky=1; kx=1; end 5: begin ky=1; kx=2; end
            6: begin ky=2; kx=0; end 7: begin ky=2; kx=1; end 8: begin ky=2; kx=2; end
            default: begin ky=0; kx=0; end
        endcase

        iy = $signed({1'b0, oy}) - 9'sd1 + ky;
        ix = $signed({1'b0, ox}) - 9'sd1 + kx;

        if (iy < 0 || iy > 27 || ix < 0 || ix > 27) begin
            addr10 = 10'd784; // sentinel => use 0
        end else begin
            iy_u = $unsigned(iy);
            ix_u = $unsigned(ix);
            // 28*iy + ix via shifts: (iy<<5) - (iy<<2) + ix
            addr10 = ({4'b0, iy_u} << 5) - ({4'b0, iy_u} << 2) + {4'b0, ix_u};
        end
    end

    // =========================================================================
    //  Conv1 / FC Runtime Registers
    // =========================================================================
    logic signed [31:0] px_val_s1;
    logic [3:0]  k_idx_s1;
    logic valid_s1, valid_s2;

    // Accumulators:
    // - acc[0..7] : Conv1 per-filter accumulation
    // - acc[0..9] : reused as FC accumulators when entering RUN_FC
    logic signed [63:0] acc [0:9];

    logic signed [31:0] horz_temp [0:7];
    logic signed [31:0] val_relu, h_max;
    logic signed [31:0] final_max;
    logic signed [31:0] scaled_val;

    // Start pulse generation (edge detect on reg_ctrl[0])
    logic start_p, prev_start;

    // Conv1 staged fetch bookkeeping for packed image
    logic        img_vld_r;
    logic [1:0]  lane_r;
    logic [3:0]  k_idx_r;
    logic [31:0] img_word_r;

    // FC pipeline temporary registers (timing-friendly partial-sum stage)
    logic signed [63:0] partial_save [0:9];
    logic signed [31:0] fc_logits32 [0:9];
    logic signed [63:0] fc_best;

    // AXI writebeat index for result writeback
    logic [3:0] beat;

    /* =========================================================================
     * DEBUG 
     *   The following debug signals/states were used
     *   to validate Conv2 correctness by dumping intermediate Conv2 results
     *   back to external memory for software-side checking.
     *
     * logic [9:0] write_cnt;
     * logic [7:0] fc_dump_w_idx;
     * logic [3:0] dbg_fc_class_r;
     * ========================================================================= */

    // =========================================================================
    //  Main FSM + Compute Sequencing (m00_axi_aclk domain)
    // =========================================================================
    always_ff @(posedge m00_axi_aclk) begin
        if (!m00_axi_aresetn) begin
            // Global reset for compute + AXI master control
            state <= IDLE; reg_cycles <= 0;

            // AXI master control outputs default
            m00_axi_arvalid<=0; m00_axi_rready<=0; m00_axi_awvalid<=0; m00_axi_wvalid<=0; m00_axi_bready<=0;

            // Pulse detector state
            prev_start <= 0;

            // BRAM write controls and counters
            out_buf_we <= 0;

            // Streamer / Conv2 weight streamer defaults
            streamer_start <= 0;
            c2_w_we <= 0;

            // BRAM read address default for FSM path
            fsm_addr_read <= 0;

            // Status flags
            reg_status <= 0;

            // Conv1 pipeline regs
            img_vld_r <= 0;
            valid_s1 <= 0;
            valid_s2 <= 0;
            k_idx_r <= 0;

            start_p <= 0;
        end else begin
            // Start pulse detection (rising edge of reg_ctrl[0])
            start_p <= (reg_ctrl[0] & ~prev_start);
            prev_start <= reg_ctrl[0];

            // Optional clear of done flag (software-controlled)
            if (reg_ctrl[1]) reg_status[1] <= 0;

            // Default deassertions each cycle (explicit enable style)
            out_buf_we <= 0;
            c2_w_we <= 0;
            streamer_start <= 0;

            case(state)

                // -------------------------------------------------------------
                // IDLE: wait for start pulse; initialize runtime state
                // -------------------------------------------------------------
                IDLE: begin
                    reg_status[0] <= 0;

                    // Continuous reset of loop counters / pipeline controls
                    cnt <= 0;
                    valid_s1 <= 0; valid_s2 <= 0;
                    oy <= 0; ox <= 0; k_idx <= 0;
                    cur_filter_load <= 0;

                    beat <= 0;

                    // Mode select:
                    // - reg_ctrl[2] = weight-load-only path
                    // - start_p triggers full inference path (image->conv->fc->writeback)
                    if (reg_ctrl[2]) begin
                        state <= LOAD_W_CMD;
                        reg_status[0] <= 1; // busy
                        reg_status[1] <= 0; // done=0
                    end
                    else if (start_p) begin
                        state <= LOAD_IMG_CMD;
                        reg_res_label <= 0;
                        reg_status[0] <= 1; // busy
                        reg_status[1] <= 0; // done=0
                        reg_cycles <= 0;
                    end
                end

                // -------------------------------------------------------------
                // 1) LOAD CONV1 WEIGHTS: 8 filters, each 10 beats (9 weights + bias)
                //    Address stride per filter = 40 bytes (10 words)
                // -------------------------------------------------------------
                LOAD_W_CMD: begin
                    m00_axi_arvalid <= 1; m00_axi_araddr <= reg_weight_base + (cur_filter_load * 40);
                    m00_axi_arlen <= 9; m00_axi_arsize <= 2; m00_axi_arburst <= 1;
                    state <= LOAD_W_DATA; cnt <= 0;
                end
                LOAD_W_DATA: begin
                    if (m00_axi_arready) m00_axi_arvalid <= 0;
                    m00_axi_rready <= 1;
                    if (m00_axi_rvalid) begin
                        if (cnt < 9) w_cache[cur_filter_load][cnt] <= m00_axi_rdata;
                        else         b_cache[cur_filter_load]       <= m00_axi_rdata;
                        cnt <= cnt + 1;
                        if (m00_axi_rlast) begin
                            m00_axi_rready <= 0;
                            if (cur_filter_load == 7) begin
                                state <= LOAD_W2_CMD; cnt <= 0; cur_filter_load <= 0;
                            end else begin
                                cur_filter_load <= cur_filter_load + 1;
                                state <= LOAD_W_CMD;
                            end
                        end
                    end
                end

                // -------------------------------------------------------------
                // 2) LOAD CONV2 WEIGHTS + BIAS:
                //    Per filter: 72 weight words (only [7:0] used) + 1 bias word
                //    Total per filter = 73 words = 292 bytes
                //    Weights are streamed into ComputeEngine via (c2_w_we, c2_w_data)
                // -------------------------------------------------------------
                LOAD_W2_CMD: begin
                    m00_axi_arvalid <= 1; m00_axi_araddr <=  reg_weight2_base + (cur_filter_load * 292);
                    m00_axi_arlen <= 8'd72;
                    m00_axi_arsize <= 2; m00_axi_arburst <= 1;
                    state <= LOAD_W2_DATA; cnt <= 0;
                end
                LOAD_W2_DATA: begin
                    if (m00_axi_arready) m00_axi_arvalid <= 0;
                    m00_axi_rready <= 1;
                    if (m00_axi_rvalid) begin
                        if (cnt < 72) begin
                            c2_w_we <= 1;
                            c2_w_data <= $signed({m00_axi_rdata[7], m00_axi_rdata[7:0]});
                        end else begin
                            c2_w_we <= 0;
                            b2_cache[cur_filter_load] <= m00_axi_rdata;
                        end
                        cnt <= cnt + 1;
                        if (m00_axi_rlast) begin
                            m00_axi_rready <= 0;
                            if (cur_filter_load == 15) begin
                                state <= LOAD_FC_BIAS_CMD;
                                cur_filter_load <= 0;
                            end else begin
                                cur_filter_load <= cur_filter_load + 1;
                                state <= LOAD_W2_CMD;
                            end
                        end
                    end else begin
                        c2_w_we <= 0;
                    end
                end

                // -------------------------------------------------------------
                // 3) FC Bias CMD: 10-word burst
                // -------------------------------------------------------------
                LOAD_FC_BIAS_CMD: begin
                    m00_axi_arvalid <= 1;
                    m00_axi_araddr  <= reg_fc_b_base;
                    m00_axi_arlen   <= 9;
                    m00_axi_arsize  <= 2;
                    m00_axi_arburst <= 1;

                    state   <= LOAD_FC_BIAS_DATA;
                    cnt <= 0;
                end

                LOAD_FC_BIAS_DATA: begin
                    if (m00_axi_arready && m00_axi_arvalid) m00_axi_arvalid <= 0;
                    m00_axi_rready <= 1;

                    if (m00_axi_rvalid) begin
                        fc_b_cache[cnt] <= m00_axi_rdata;
                        cnt <= cnt + 1;

                        if (m00_axi_rlast) begin
                            m00_axi_rready <= 0;
                            state <= LOAD_FC_W_CMD;
                            cur_filter_load <= 0;
                            cnt <= 0;
                        end
                    end
                end

                // -------------------------------------------------------------
                // 3) FC Weight CMD: 10 classes * 196 beats each
                // -------------------------------------------------------------
                LOAD_FC_W_CMD: begin
                    m00_axi_arvalid <= 1;
                    m00_axi_araddr  <= reg_fc_w_base + (cur_filter_load * 784);
                    m00_axi_arlen   <= 195;
                    m00_axi_arsize  <= 2;
                    m00_axi_arburst <= 1;
                    cnt <= 0;
                    state <= LOAD_FC_W_DATA;
                end

                LOAD_FC_W_DATA: begin
                    if (m00_axi_arvalid && m00_axi_arready) m00_axi_arvalid <= 1'b0;
                    m00_axi_rready <= 1'b1;
                    if (m00_axi_rvalid && m00_axi_rready) begin
                        if (m00_axi_rlast) begin
                            m00_axi_rready <= 0;
                            if (cur_filter_load == 9) begin
                                state <= IDLE;
                                reg_status[1] <= 1;  // "weight loading done" flag
                            end else begin
                                cur_filter_load <= cur_filter_load + 1;
                                state <= LOAD_FC_W_CMD;
                            end
                        end else begin
                            cnt <= cnt + 1;
                        end
                    end
                end

                // -------------------------------------------------------------
                // 4) LOAD IMAGE: burst read 196 beats (784 bytes) into img_cache[]
                // -------------------------------------------------------------
                LOAD_IMG_CMD: begin
                    m00_axi_arvalid <= 1; m00_axi_araddr <= reg_img_base;
                    m00_axi_arlen <= 195; m00_axi_arsize <= 2; m00_axi_arburst <= 1;
                    state <= LOAD_IMG_DATA; cnt <= 0;
                end
                LOAD_IMG_DATA: begin
                    if (m00_axi_arready & m00_axi_arvalid) m00_axi_arvalid <= 0;
                    m00_axi_rready <= 1;
                    if (m00_axi_rvalid) begin
                        img_cache[cnt] <= m00_axi_rdata;
                        cnt <= cnt + 1;
                        if (m00_axi_rlast) begin
                            state <= RUN_CONV1;
                            m00_axi_rready <= 0;
                            for (int i=0; i<8; i++) acc[i] <= b_cache[i];
                        end
                    end
                end

                // -------------------------------------------------------------
                // 5) RUN_CONV1:
                //    28x28 Conv -> ReLU -> 2x2 MaxPool => 8ch x 14x14 written to banks[0..7]
                // -------------------------------------------------------------
                RUN_CONV1: begin
                    reg_cycles <= reg_cycles + 1;

                    // Stage 0: issue next tap fetch (k_idx 0..8)
                    if (k_idx < 9) begin
                        img_vld_r <= 1'b1;
                        lane_r    <= addr10[1:0];
                        k_idx_r   <= k_idx;

                        // Read packed image word (or 0 if padding)
                        if (addr10 >= 784) img_word_r <= 32'b0;
                        else               img_word_r <= img_cache[addr10[9:2]];

                        k_idx <= k_idx + 1;
                    end else begin
                        img_vld_r <= 1'b0;
                    end

                    // Stage 1: byte extract
                    valid_s1 <= img_vld_r;
                    k_idx_s1 <= k_idx_r;

                    if (img_vld_r) begin
                        case (lane_r)
                            2'd0: px_val_s1 <= {24'b0, img_word_r[7:0]};
                            2'd1: px_val_s1 <= {24'b0, img_word_r[15:8]};
                            2'd2: px_val_s1 <= {24'b0, img_word_r[23:16]};
                            2'd3: px_val_s1 <= {24'b0, img_word_r[31:24]};
                        endcase
                    end else begin
                        px_val_s1 <= 0;
                    end

                    // Stage 2: accumulate per-filter
                    if (valid_s1) begin
                        valid_s2 <= 1;
                        for(int i=0; i<8; i++) acc[i] <= acc[i] + (px_val_s1 * w_cache[i][k_idx_s1]);
                    end else valid_s2 <= 0;

                    // Finalize when all taps issued and pipeline drained
                    if (k_idx == 9 && !valid_s1 && !valid_s2) begin
                        for(int i=0; i<8; i++) begin
                            val_relu = (acc[i] < 0) ? 0 : acc[i][31:0];

                            // 2x2 maxpool (tile over ox/oy parity)
                            if (ox[0] == 0) horz_temp[i] <= val_relu;
                            else begin
                                h_max = (val_relu > horz_temp[i]) ? val_relu : horz_temp[i];
                                if (oy[0] == 0) line_buf[ox[4:1]][i] <= h_max;
                                else begin
                                    final_max = (h_max > line_buf[ox[4:1]][i]) ? h_max : line_buf[ox[4:1]][i];
                                    // Write pooled Conv1 output into bank i at address (oy>>1, ox>>1)
                                    out_buf_we[i] <= 1;
                                    out_buf_addr_write <= (oy[4:1])*14 + (ox[4:1]);
                                
                                    // Clip to 0..255 (stored in 32b, used downstream as small magnitude)
                                    scaled_val = final_max >>> 10;
                                    if (scaled_val > 255)      out_buf_wdata[i] <= 32'd255;
                                    else if (scaled_val < 0)   out_buf_wdata[i] <= 32'd0;
                                    else                       out_buf_wdata[i] <= scaled_val;
                                end
                            end
                        end

                        // Reset accumulators for next pixel
                        for(int i=0; i<8; i++) acc[i] <= b_cache[i];
                        k_idx <= 0;

                        // Advance scan over 28x28
                        if (ox < 27) ox <= ox + 1;
                        else begin
                            ox <= 0;
                            if (oy < 27) oy <= oy + 1;
                            else begin
                                state <= RUN_CONV2;
                                streamer_start <= 1;
                            end
                        end
                    end
                end

                // -------------------------------------------------------------
                // 6) RUN_CONV2:
                //    Stream padded Conv1 output into Conv2; write 16ch pooled results
                //    into banks[0..15] at addr = 256 + (row*7 + col)
                // -------------------------------------------------------------
                RUN_CONV2: begin
                    reg_cycles <= reg_cycles + 1;

                    if (c2_valid_out) begin
                        out_buf_we <= 0;
                        out_buf_we[c2_out_ch] <= 1;

                        out_buf_addr_write <= 9'd256 + (c2_out_row * 7) + c2_out_col;
                        out_buf_wdata[c2_out_ch] <= c2_data_out;

                        // Last sample => transition to FC
                        if (c2_out_row == 6 && c2_out_col == 6 && c2_out_ch == 15) begin
                            state <= RUN_FC;
                            cnt <= 0;
                            for(int i=0; i<10; i++) acc[i] <= fc_b_cache[i];
                            fsm_addr_read <= 9'd256;
                            fc_rd_addr <= 0;
                        end
                    end else begin
                        out_buf_we <= 0;
                    end
                end

                // -------------------------------------------------------------
                // 7) RUN_FC: Fully Connected (timing-pipelined)
                //    Stage0: issue BRAM reads for feature + weight
                //    Stage1: compute partial sum (4 MACs) into partial_save[]
                //    Stage2: accumulate partial_save[] into acc[]
                // -------------------------------------------------------------
                RUN_FC: begin
                    reg_cycles <= reg_cycles + 1;

                    // [Stage 0] Issue read
                    if (cnt < 195) begin
                          fsm_addr_read <= 9'd256 + ((cnt + 1) >> 2);
                          fc_rd_addr    <= cnt + 1;
                    end

                    // [Stage 1] Multiply + partial sum (registered)
                    if (cnt > 0 && cnt <= 196) begin
                        logic [7:0] word_idx;
                        logic [3:0] base;
                        logic [31:0] w_pack;
                        logic signed [24:0] f0,f1,f2,f3;
                        logic signed [8:0]  w0,w1,w2,w3;
                        logic signed [33:0] p0, p1, p2, p3;
                        logic signed [35:0] calc_temp;

                        word_idx = cnt - 1;
                        base = (word_idx[1:0] << 2);

                        for (int k = 0; k < 10; k++) begin
                            w_pack = fc_w_rdata[k];
    
                            f0 = $signed(out_buf_rdata[base + 0][24:0]);
                            f1 = $signed(out_buf_rdata[base + 1][24:0]);
                            f2 = $signed(out_buf_rdata[base + 2][24:0]);
                            f3 = $signed(out_buf_rdata[base + 3][24:0]);
    
                            w0 = $signed({w_pack[7],   w_pack[7:0]});
                            w1 = $signed({w_pack[15],  w_pack[15:8]});
                            w2 = $signed({w_pack[23],  w_pack[23:16]});
                            w3 = $signed({w_pack[31],  w_pack[31:24]});
    
                            // Encourage DSP inference (tool-dependent)
                            (* use_dsp = "yes" *) p0 = f0 * w0;
                            (* use_dsp = "yes" *) p1 = f1 * w1;
                            (* use_dsp = "yes" *) p2 = f2 * w2;
                            (* use_dsp = "yes" *) p3 = f3 * w3;
    
                            calc_temp = p0 + p1 + p2 + p3;
                            partial_save[k] <= {{(28){calc_temp[35]}}, calc_temp};
                        end
                    end

                    // [Stage 2] Accumulate
                    if (cnt > 1 && cnt <= 197) begin
                        for (int k = 0; k < 10; k++) begin
                            acc[k] <= acc[k] + partial_save[k];
                        end
                    end

                    // End condition: includes pipeline latency (2 cycles)
                    if (cnt == 197) begin
                        state <= FC_CHECK_MAX;

                        fc_best <= acc[0];
                        reg_res_label <= 0;
                    end else begin
                        cnt <= cnt + 1;
                    end
                end

                // -------------------------------------------------------------
                // 8) FC_CHECK_MAX:
                //    - Convert acc[] to 32-bit logits (shift) once
                //    - Serial argmax over 10 classes to produce reg_res_label
                // -------------------------------------------------------------
                FC_CHECK_MAX: begin
                    reg_cycles <= reg_cycles + 1;
                
                    if (beat == 0) begin
                        for (int t=0; t<10; t++) fc_logits32[t] <= $signed(acc[t] >>> 12);
                        beat <= 1;

                        cnt <= 1;
                        fc_best <= acc[0];
                        reg_res_label <= 0;
                    end else begin
                        if (cnt < 10) begin
                            if (acc[cnt] > fc_best) begin
                                fc_best <= acc[cnt];
                                reg_res_label <= cnt[3:0];
                            end
                            cnt <= cnt + 1;
                        end else begin
                            state <= WRITE_FC_AW;
                            beat  <= 0;
                        end
                    end
                end

                // -------------------------------------------------------------
                // 9) WRITE_FC_*: Burst write 10 logits (32-bit) to reg_result_base
                // -------------------------------------------------------------
                WRITE_FC_AW: begin
                    m00_axi_awvalid <= 1;
                    m00_axi_awaddr  <= reg_result_base;
                    m00_axi_awlen   <= 8'd9;
                    m00_axi_awsize  <= 3'b010;
                    m00_axi_awburst <= 2'b01;
                    beat <= 0;

                    if (m00_axi_awvalid && m00_axi_awready) begin
                        m00_axi_awvalid <= 0;
                        m00_axi_wvalid  <= 1;
                        state <= WRITE_FC_W;
                    end
                end

                WRITE_FC_W: begin
                    if (m00_axi_wvalid && m00_axi_wready) begin
                        if (beat == 4'd9) begin
                            m00_axi_wvalid <= 0;
                            m00_axi_bready <= 1;
                            state <= WRITE_FC_B;
                        end else begin
                            beat <= beat + 1;
                        end
                    end
                end

                WRITE_FC_B: begin
                    if (m00_axi_bvalid && m00_axi_bready) begin
                        m00_axi_bready <= 0;
                        state <= DONE;
                    end
                end

                /* =================================================================
                 * DEBUG STATES (Disabled by default)
                 *   Purpose:
                 *     Dump Conv2 results (which are already written into BRAM banks)
                 *     back to external memory for software-side verification.
                 *
                 *   NOTE: This block is intentionally commented out in normal mode.
                 * ================================================================= */
                /*
                WRITE_AW: begin
                    m00_axi_awvalid <= 1;
                    m00_axi_awaddr <= reg_img_base + 32'hA000 + (write_cnt * 4);
                    m00_axi_awlen <= 8'd15; // 16 beats (all 16 channels per pixel)
                    m00_axi_awsize <= 3'b010;
                    m00_axi_awburst <= 2'b01;

                    beat <= 0;
                    fsm_addr_read <= 9'd256 + (write_cnt >> 4);

                    if (m00_axi_awvalid && m00_axi_awready) begin
                        m00_axi_awvalid <= 0;
                        state <= WRITE_PREFETCH;
                    end
                end

                WRITE_PREFETCH: begin
                  state <= WRITE_PREFETCH_2;
                end

                WRITE_PREFETCH_2: begin
                  state <= WRITE_W;
                  m00_axi_wvalid <= 1;
                end

                WRITE_W: begin
                    if (m00_axi_wready && m00_axi_wvalid) begin
                        beat <= beat + 1;
                        write_cnt <= write_cnt + 1;
                        if (beat == 4'd15) begin
                            m00_axi_wvalid <= 0;
                            m00_axi_bready <= 1;
                            state <= WRITE_B;
                        end
                    end
                end

                WRITE_B: begin
                    if (m00_axi_bvalid && m00_axi_bready) begin
                        m00_axi_bready <= 0;
                        if (write_cnt == 784) begin
                            state <= DONE;
                            reg_status[1] <= 1;
                        end else state <= WRITE_AW;
                    end
                end
                */

                // -------------------------------------------------------------
                // DONE: finalize status and return to IDLE
                // -------------------------------------------------------------
                DONE: begin
                    state <= IDLE;
                    reg_status[1] <= 1; // done sticky until cleared by reg_ctrl[1]
                    reg_status[0] <= 0; // busy=0
                end

                default: state <= IDLE;
            endcase
        end
    end

    // =========================================================================
    //  AXI Write Data / Last
    // =========================================================================
    assign m00_axi_wlast = (state == WRITE_FC_W) && (beat == 4'd9);
    assign m00_axi_wdata = (state == WRITE_FC_W) ? $unsigned(fc_logits32[beat]) : 32'd0;

    /* debug
    assign m00_axi_wlast =
        ((state == WRITE_W)      && m00_axi_wvalid && (beat == 4'd15)) ||
        ((state == WRITE_FC_W)      && m00_axi_wvalid && (beat == 4'd9));

    assign m00_axi_wdata =
        (state == WRITE_W)     ? out_buf_rdata[beat[3:0]] :
        (state == WRITE_FC_W)     ? $unsigned(fc_logits32[beat[3:0]]) :
                                    32'd0;
    */

    // =========================================================================
    //  Constant / Unused AXI Master assignments (static fields)
    // =========================================================================
    assign m00_axi_awid=0; assign m00_axi_awlock=0; assign m00_axi_awcache=0; assign m00_axi_awprot=0; assign m00_axi_awqos=0;
    assign m00_axi_wstrb=4'hF; assign m00_axi_arid=0; assign m00_axi_arlock=0; assign m00_axi_arcache=3; assign m00_axi_arprot=0; assign m00_axi_arqos=0;

    // =========================================================================
    //  AXI4-Lite response defaults (always OKAY)
    // =========================================================================
    assign s00_axi_bresp = 2'b00;
    assign s00_axi_rresp = 2'b00;

endmodule
