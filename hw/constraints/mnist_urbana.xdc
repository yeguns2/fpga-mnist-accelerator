# -------------------------------------------------------------------------
# Clock Signal (100MHz)
# -------------------------------------------------------------------------
set_property PACKAGE_PIN N15 [get_ports Clk]
set_property IOSTANDARD LVCMOS33 [get_ports Clk]
create_clock -period 10.000 -name clk_100 -waveform {0.000 5.000} [get_ports Clk]

# -------------------------------------------------------------------------
# Reset Signal (Active High Button)
# -------------------------------------------------------------------------
# J2 is usually the "RESET" button on Urbana
set_property PACKAGE_PIN J2 [get_ports reset_rtl_0]
set_property IOSTANDARD LVCMOS25 [get_ports reset_rtl_0]

# -------------------------------------------------------------------------
# UART Interface (USB-UART)
# -------------------------------------------------------------------------
# UART RX (Host -> FPGA)
set_property PACKAGE_PIN B16 [get_ports uart_rtl_0_rxd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rtl_0_rxd]

# UART TX (FPGA -> Host)
set_property PACKAGE_PIN A16 [get_ports uart_rtl_0_txd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rtl_0_txd]