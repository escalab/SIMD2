`timescale 1ns / 1ps

module mulOp#(parameter W = 16)
(
input clk,
input [1:0] op,
input [3:0][3:0][W-1:0] inA, inB,
output [3:0][3:0][3:0][W-1:0] out 
);

logic [3:0][3:0][W-1:0] regTemp00;
logic [3:0][3:0][W-1:0] regTemp01;
logic [3:0][3:0][W-1:0] regTemp10;
logic [3:0][3:0][W-1:0] regTemp11;