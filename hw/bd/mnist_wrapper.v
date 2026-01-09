//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
//Date        : Thu Jan  8 22:52:00 2026
//Host        : shimsen running 64-bit major release  (build 9200)
//Command     : generate_target mnist_wrapper.bd
//Design      : mnist_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module mnist_wrapper
   (clk_100MHz,
    reset_rtl_0,
    uart_rxd,
    uart_txd);
  input clk_100MHz;
  input reset_rtl_0;
  input uart_rxd;
  output uart_txd;

  wire clk_100MHz;
  wire reset_rtl_0;
  wire uart_rxd;
  wire uart_txd;

  mnist mnist_i
       (.clk_100MHz(clk_100MHz),
        .reset_rtl_0(reset_rtl_0),
        .uart_rxd(uart_rxd),
        .uart_txd(uart_txd));
endmodule
