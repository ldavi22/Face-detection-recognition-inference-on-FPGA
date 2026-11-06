# Platform Designer component for the conv1x1 accelerator.
#
# Put this next to the rtl/ directory and point Platform Designer's IP
# search path at it (Tools > Options > IP Search Paths). The component
# then appears in the IP Catalog and can be dropped into the system and
# wired to the lightweight HPS-to-FPGA bridge.
#
# Connections to make in Platform Designer:
#   clk          -> the system clock (100 MHz is a reasonable target)
#   reset        -> the system reset
#   avalon_slave -> hps.h2f_lw_axi_master
#
# Then note the base address Platform Designer assigns the slave. That
# number, minus 0xFF200000, is the component_offset the HPS driver needs.

package require -exact qsys 16.0

set_module_property NAME conv1x1
set_module_property DISPLAY_NAME "1x1 Convolution Accelerator"
set_module_property VERSION 1.0
set_module_property GROUP "CNN"
set_module_property AUTHOR ""
set_module_property DESCRIPTION \
    "INT8 1x1 convolution with per-channel requantization and leaky relu"
set_module_property EDITABLE true
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property VALIDATION_CALLBACK validate

add_fileset synth QUARTUS_SYNTH generate_synth
set_fileset_property synth TOP_LEVEL conv1x1

proc generate_synth {name} {
    add_fileset_file conv1x1.v VERILOG PATH rtl/conv1x1.v TOP_LEVEL_FILE
    add_fileset_file mac4.v    VERILOG PATH rtl/mac4.v
    add_fileset_file requant.v VERILOG PATH rtl/requant.v
}

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

add_parameter C_IN INTEGER 256
set_parameter_property C_IN DISPLAY_NAME "Input channels (multiple of 4)"
set_parameter_property C_IN HDL_PARAMETER true

add_parameter C_OUT INTEGER 128
set_parameter_property C_OUT DISPLAY_NAME "Output channels"
set_parameter_property C_OUT HDL_PARAMETER true

add_parameter POS INTEGER 169
set_parameter_property POS DISPLAY_NAME "Spatial positions (W*H)"
set_parameter_property POS HDL_PARAMETER true

add_parameter LEAKY INTEGER 1
set_parameter_property LEAKY DISPLAY_NAME "Leaky ReLU (0 = linear)"
set_parameter_property LEAKY HDL_PARAMETER true

proc validate {} {
    set c_in [get_parameter_value C_IN]
    if {[expr {$c_in % 4}] != 0} {
        send_message error "C_IN must be a multiple of 4 (one 32-bit word\
                            holds four int8 channels)"
    }
}

# ---------------------------------------------------------------------
# Clock and reset
# ---------------------------------------------------------------------

add_interface clk clock end
add_interface_port clk clk clk Input 1

add_interface reset reset end
set_interface_property reset associatedClock clk
set_interface_property reset synchronousEdges DEASSERT
add_interface_port reset rst_n reset_n Input 1

# ---------------------------------------------------------------------
# Avalon-MM slave, 32-bit, word addressed
#
# 19 address bits: [18:16] select the region, [15:0] the offset inside it.
# That is 2^19 words = 2 MB, which is exactly the lightweight bridge
# window -- so this component fills it. If you add another component to
# the same bridge later, shrink the address width here first.
# ---------------------------------------------------------------------

add_interface avalon_slave avalon end
set_interface_property avalon_slave associatedClock clk
set_interface_property avalon_slave associatedReset reset
set_interface_property avalon_slave addressUnits WORDS
set_interface_property avalon_slave readLatency 1
set_interface_property avalon_slave maximumPendingReadTransactions 0

add_interface_port avalon_slave avs_address   address   Input  18
add_interface_port avalon_slave avs_write     write     Input   1
add_interface_port avalon_slave avs_writedata writedata Input  32
add_interface_port avalon_slave avs_read      read      Input   1
add_interface_port avalon_slave avs_readdata  readdata  Output 32
