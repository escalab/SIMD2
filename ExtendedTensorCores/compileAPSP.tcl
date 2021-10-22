analyze -format sverilog {./src/etc.v ./src/etcEx.v ./src/min.v ./src/etcAPSP.v ./src/etcL2.v}
elaborate etcAPSP
link
set TCLK 1.9
set TCU 0.1
uniquify
compile -only_design_rule
compile_ultra -no_autoungroup
compile -inc -only_hold_time
set_fix_multiple_port_nets-all -buffer_constants [get_designs *]
remove_unconnected_ports-blast_buses [get_cells -hier]
report_timing -path full -delay min -max_paths 10 > etcAPSP.holdtiming
report_timing -path full -delay max -max_paths 10 > etcAPSP.setuptiming
report_area -hierarchy > etcAPSP.area
report_power -hier -hier_level 2 > etcAPSP.power
