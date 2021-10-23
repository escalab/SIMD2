analyze -format sverilog {./src/etc.v ./src/etcEx.v ./src/min.v ./src/etcAPSP.v ./src/etcL2.v}
elaborate etc
link
set TCLK 1.9
set TCU 0.1
uniquify
compile -only_design_rule
compile_ultra -no_autoungroup
compile -inc -only_hold_time
set_fix_multiple_port_nets-all -buffer_constants [get_designs *]
remove_unconnected_ports-blast_buses [get_cells -hier]
report_timing -path full -delay min -max_paths 10 > etc.holdtiming
report_timing -path full -delay max -max_paths 10 > etc.setuptiming
report_area -hierarchy > etc.area
report_power -hier -hier_level 2 > etc.power
elaborate etcEX
link
set TCLK 1.9
set TCU 0.1
uniquify
compile -only_design_rule
compile_ultra -no_autoungroup
compile -inc -only_hold_time
set_fix_multiple_port_nets-all -buffer_constants [get_designs *]
remove_unconnected_ports-blast_buses [get_cells -hier]
report_timing -path full -delay min -max_paths 10 > etcEX.holdtiming
report_timing -path full -delay max -max_paths 10 > etcEX.setuptiming
report_area -hierarchy > etcEX.area
report_power -hier -hier_level 2 > etcEX.power
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
elaborate etcL2
link
set TCLK 1.9
set TCU 0.1
uniquify
compile -only_design_rule
compile_ultra -no_autoungroup
compile -inc -only_hold_time
set_fix_multiple_port_nets-all -buffer_constants [get_designs *]
remove_unconnected_ports-blast_buses [get_cells -hier]
report_timing -path full -delay min -max_paths 10 > etcL2.holdtiming
report_timing -path full -delay max -max_paths 10 > etcL2.setuptiming
report_area -hierarchy > etcL2.area
report_power -hier -hier_level 2 > etcL2.power
