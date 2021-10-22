`timescale 1ns / 1ps

/*
 * CSE141L Lab1: Tools of the Trade
 * University of California, San Diego
 * 
 * Written by Matt DeVuyst, 3/30/2010
 * Modified by Vikram Bhatt, 30/3/2010
 * Modified by Adrian Caulfield, 1/8/2012
 */

//
// NOTE: This verilog is non-synthesizable.
// You can only use constructs like "initial", "#10", "forever"
// inside your test bench! Do not use it in your actual design.
//

module test_etc#(parameter W = 16);

   reg          clk;
   reg  [3:0][3:0][W-1:0] a_r;
   reg  [3:0][3:0][W-1:0] b_r;
   wire [3:0][3:0][W-1:0] sum;
   reg [1:0] operation;
   integer i;

   // The design under test is our adder
   etcEX dut (   .clk(clk)
            ,.op(operation)
	        ,.inA(a_r)
	        ,.inB(b_r)
           ,.out(sum)
             );

   // Toggle the clock every 10 ns

   initial
     begin
        clk = 0;
        forever #10 clk = !clk;
     end

   // Test with a variety of inputs.
   // Introduce new stimulus on the falling clock edge so that values
   // will be on the input wires in plenty of time to be read by
   // registers on the subsequent rising clock edge.
   initial
     begin
        operation = 2'b00;
        for(int i = 0; i < 4; i++) begin
            a_r[0][i] = 0;
            b_r[0][i] = 0;
        end
        for(int i = 0; i < 4; i++) begin
            a_r[1][i] = 0;
            b_r[1][i] = 0;
        end
        for(int i = 0; i < 4; i++) begin
            a_r[2][i] = 0;
            b_r[2][i] = 0;
        end
        for(int i = 0; i < 4; i++) begin
            a_r[3][i] = 0;
            b_r[3][i] = 0;
        end
        a_r[0][0] = 1;
        b_r[0][0] = 1;
        @(negedge clk);
        a_r[0][1] = 1;
        b_r[1][0] = 1;
        @(negedge clk);
        a_r[0][2] = 5;
        b_r[2][0] = 6;
        @(negedge clk);
        a_r[0][3] = 2;
        b_r[3][0] = 2;
        @(negedge clk);
        a_r[1][0] = 3;
        b_r[0][1] = 3;
        @(negedge clk);
        a_r[2][0] = 1;
        b_r[0][2] = 8;
        @(negedge clk);
//        #2320700;
               $display("At time %t, sum[0][0] = %h (%0d)", $time, sum[0][0], sum[0][0]);
               $display("At time %t, sum[0][1] = %h (%0d)", $time, sum[0][1], sum[0][1]);
               $display("At time %t, sum[0][2] = %h (%0d)", $time, sum[0][2], sum[0][2]);
               $display("At time %t, sum[0][3] = %h (%0d)", $time, sum[0][3], sum[0][3]);
               $display("At time %t, sum[1][0] = %h (%0d)", $time, sum[1][0], sum[1][0]);
               $display("At time %t, sum[1][1] = %h (%0d)", $time, sum[1][1], sum[1][1]);
               $display("At time %t, sum[1][2] = %h (%0d)", $time, sum[1][2], sum[1][2]);
               $display("At time %t, sum[1][3] = %h (%0d)", $time, sum[1][3], sum[1][3]);
               $display("At time %t, sum[2][0] = %h (%0d)", $time, sum[2][0], sum[2][0]);
               $display("At time %t, sum[2][1] = %h (%0d)", $time, sum[2][1], sum[2][1]);
               $display("At time %t, sum[2][2] = %h (%0d)", $time, sum[2][2], sum[2][2]);
               $display("At time %t, sum[2][3] = %h (%0d)", $time, sum[2][3], sum[2][3]);
               $display("At time %t, sum[3][0] = %h (%0d)", $time, sum[3][0], sum[3][0]);
               $display("At time %t, sum[3][1] = %h (%0d)", $time, sum[3][1], sum[3][1]);
               $display("At time %t, sum[3][2] = %h (%0d)", $time, sum[3][2], sum[3][2]);
               $display("At time %t, sum[3][3] = %h (%0d)", $time, sum[3][3], sum[3][3]);
        @(negedge clk);
        operation = 2'b10;
         @(negedge clk);
               $display("At time %t, sum[0][0] = %h (%0d)", $time, sum[0][0], sum[0][0]);
               $display("At time %t, sum[0][1] = %h (%0d)", $time, sum[0][1], sum[0][1]);
               $display("At time %t, sum[0][2] = %h (%0d)", $time, sum[0][2], sum[0][2]);
               $display("At time %t, sum[0][3] = %h (%0d)", $time, sum[0][3], sum[0][3]);
               $display("At time %t, sum[1][0] = %h (%0d)", $time, sum[1][0], sum[1][0]);
               $display("At time %t, sum[1][1] = %h (%0d)", $time, sum[1][1], sum[1][1]);
               $display("At time %t, sum[1][2] = %h (%0d)", $time, sum[1][2], sum[1][2]);
               $display("At time %t, sum[1][3] = %h (%0d)", $time, sum[1][3], sum[1][3]);
               $display("At time %t, sum[2][0] = %h (%0d)", $time, sum[2][0], sum[2][0]);
               $display("At time %t, sum[2][1] = %h (%0d)", $time, sum[2][1], sum[2][1]);
               $display("At time %t, sum[2][2] = %h (%0d)", $time, sum[2][2], sum[2][2]);
               $display("At time %t, sum[2][3] = %h (%0d)", $time, sum[2][3], sum[2][3]);
               $display("At time %t, sum[3][0] = %h (%0d)", $time, sum[3][0], sum[3][0]);
               $display("At time %t, sum[3][1] = %h (%0d)", $time, sum[3][1], sum[3][1]);
               $display("At time %t, sum[3][2] = %h (%0d)", $time, sum[3][2], sum[3][2]);
               $display("At time %t, sum[3][3] = %h (%0d)", $time, sum[3][3], sum[3][3]);
   
         operation = 2'b01;
         @(negedge clk);
               $display("At time %t, sum[0][0] = %h (%0d)", $time, sum[0][0], sum[0][0]);
               $display("At time %t, sum[0][1] = %h (%0d)", $time, sum[0][1], sum[0][1]);
               $display("At time %t, sum[0][2] = %h (%0d)", $time, sum[0][2], sum[0][2]);
               $display("At time %t, sum[0][3] = %h (%0d)", $time, sum[0][3], sum[0][3]);
               $display("At time %t, sum[1][0] = %h (%0d)", $time, sum[1][0], sum[1][0]);
               $display("At time %t, sum[1][1] = %h (%0d)", $time, sum[1][1], sum[1][1]);
               $display("At time %t, sum[1][2] = %h (%0d)", $time, sum[1][2], sum[1][2]);
               $display("At time %t, sum[1][3] = %h (%0d)", $time, sum[1][3], sum[1][3]);
               $display("At time %t, sum[2][0] = %h (%0d)", $time, sum[2][0], sum[2][0]);
               $display("At time %t, sum[2][1] = %h (%0d)", $time, sum[2][1], sum[2][1]);
               $display("At time %t, sum[2][2] = %h (%0d)", $time, sum[2][2], sum[2][2]);
               $display("At time %t, sum[2][3] = %h (%0d)", $time, sum[2][3], sum[2][3]);
               $display("At time %t, sum[3][0] = %h (%0d)", $time, sum[3][0], sum[3][0]);
               $display("At time %t, sum[3][1] = %h (%0d)", $time, sum[3][1], sum[3][1]);
               $display("At time %t, sum[3][2] = %h (%0d)", $time, sum[3][2], sum[3][2]);
               $display("At time %t, sum[3][3] = %h (%0d)", $time, sum[3][3], sum[3][3]);
   
         $finish;
     end // initial begin
endmodule // test_adder
