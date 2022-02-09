`timescale 1ns / 1ps
/*
 * Extended Tensor Cores
 */
 
module etcMaxMin#(parameter W = 16)
(
 input clk,
 input [1:0] op,
 input [3:0][3:0][W-1:0] inA, inB,
 output [3:0][3:0][W-1:0] out
);

integer i,j;
reg [3:0][3:0][W-1:0] regA, regB;
reg [3:0][3:0][W-1:0] regOut;
wire [3:0][3:0][W-1:0] wireOut;
wire [3:0][3:0][W-1:0] wireOutMaxMin;

//MMA logic
assign wireOut[0][0] = regA[0][0] * regB[0][0] + regA[0][1] * regB[1][0] + regA[0][2] * regB[2][0] + regA[0][3] * regB[3][0];
assign wireOut[0][1] = regA[0][0] * regB[0][1] + regA[0][1] * regB[1][1] + regA[0][2] * regB[2][1] + regA[0][3] * regB[3][1];
assign wireOut[0][2] = regA[0][0] * regB[0][2] + regA[0][1] * regB[1][2] + regA[0][2] * regB[2][2] + regA[0][3] * regB[3][2];
assign wireOut[0][3] = regA[0][0] * regB[0][3] + regA[0][1] * regB[1][3] + regA[0][2] * regB[2][3] + regA[0][3] * regB[3][3];
assign wireOut[1][0] = regA[1][0] * regB[0][0] + regA[1][1] * regB[1][0] + regA[1][2] * regB[2][0] + regA[1][3] * regB[3][0];
assign wireOut[1][1] = regA[1][0] * regB[0][1] + regA[1][1] * regB[1][1] + regA[1][2] * regB[2][1] + regA[1][3] * regB[3][1];
assign wireOut[1][2] = regA[1][0] * regB[0][2] + regA[1][1] * regB[1][2] + regA[1][2] * regB[2][2] + regA[1][3] * regB[3][2];
assign wireOut[1][3] = regA[1][0] * regB[0][3] + regA[1][1] * regB[1][3] + regA[1][2] * regB[2][3] + regA[1][3] * regB[3][3];
assign wireOut[2][0] = regA[2][0] * regB[0][0] + regA[2][1] * regB[1][0] + regA[2][2] * regB[2][0] + regA[2][3] * regB[3][0];
assign wireOut[2][1] = regA[2][0] * regB[0][1] + regA[2][1] * regB[1][1] + regA[2][2] * regB[2][1] + regA[2][3] * regB[3][1];
assign wireOut[2][2] = regA[2][0] * regB[0][2] + regA[2][1] * regB[1][2] + regA[2][2] * regB[2][2] + regA[2][3] * regB[3][2];
assign wireOut[2][3] = regA[2][0] * regB[0][3] + regA[2][1] * regB[1][3] + regA[2][2] * regB[2][3] + regA[2][3] * regB[3][3];
assign wireOut[3][0] = regA[3][0] * regB[0][0] + regA[3][1] * regB[1][0] + regA[3][2] * regB[2][0] + regA[3][3] * regB[3][0];
assign wireOut[3][1] = regA[3][0] * regB[0][1] + regA[3][1] * regB[1][1] + regA[3][2] * regB[2][1] + regA[3][3] * regB[3][1];
assign wireOut[3][2] = regA[3][0] * regB[0][2] + regA[3][1] * regB[1][2] + regA[3][2] * regB[2][2] + regA[3][3] * regB[3][2];
assign wireOut[3][3] = regA[3][0] * regB[0][3] + regA[3][1] * regB[1][3] + regA[3][2] * regB[2][3] + regA[3][3] * regB[3][3];

// maxmin logic
maxMin2 maxmin00(regA[0][0] , regB[0][0] , regA[0][1] , regB[1][0],  regA[0][2] , regB[2][0] , regA[0][3] , regB[3][0], wireOutMaxMin[0][0]); 
maxMin2 maxmin01(regA[0][0] , regB[0][1] , regA[0][1] , regB[1][1] , regA[0][2] , regB[2][1] , regA[0][3] , regB[3][1], wireOutMaxMin[0][1]); 
maxMin2 maxmin02(regA[0][0] , regB[0][2] , regA[0][1] , regB[1][2] , regA[0][2] , regB[2][2] , regA[0][3] , regB[3][2], wireOutMaxMin[0][2]); 
maxMin2 maxmin03(regA[0][0] , regB[0][3] , regA[0][1] , regB[1][3] , regA[0][2] , regB[2][3] , regA[0][3] , regB[3][3], wireOutMaxMin[0][3]); 
maxMin2 maxmin10(regA[1][0] , regB[0][0] , regA[1][1] , regB[1][0] , regA[1][2] , regB[2][0] , regA[1][3] , regB[3][0], wireOutMaxMin[1][0]); 
maxMin2 maxmin11(regA[1][0] , regB[0][1] , regA[1][1] , regB[1][1] , regA[1][2] , regB[2][1] , regA[1][3] , regB[3][1], wireOutMaxMin[1][1]); 
maxMin2 maxmin12(regA[1][0] , regB[0][2] , regA[1][1] , regB[1][2] , regA[1][2] , regB[2][2] , regA[1][3] , regB[3][2], wireOutMaxMin[1][2]); 
maxMin2 maxmin13(regA[1][0] , regB[0][3] , regA[1][1] , regB[1][3] , regA[1][2] , regB[2][3] , regA[1][3] , regB[3][3], wireOutMaxMin[1][3]); 
maxMin2 maxMin20(regA[2][0] , regB[0][0] , regA[2][1] , regB[1][0] , regA[2][2] , regB[2][0] , regA[2][3] , regB[3][0], wireOutMaxMin[2][0]); 
maxMin2 maxMin21(regA[2][0] , regB[0][1] , regA[2][1] , regB[1][1] , regA[2][2] , regB[2][1] , regA[2][3] , regB[3][1], wireOutMaxMin[2][1]); 
maxMin2 maxMin22(regA[2][0] , regB[0][2] , regA[2][1] , regB[1][2] , regA[2][2] , regB[2][2] , regA[2][3] , regB[3][2], wireOutMaxMin[2][2]); 
maxMin2 maxMin23(regA[2][0] , regB[0][3] , regA[2][1] , regB[1][3] , regA[2][2] , regB[2][3] , regA[2][3] , regB[3][3], wireOutMaxMin[2][3]); 
maxMin2 maxmin30(regA[3][0] , regB[0][0] , regA[3][1] , regB[1][0] , regA[3][2] , regB[2][0] , regA[3][3] , regB[3][0], wireOutMaxMin[3][0]); 
maxMin2 maxmin31(regA[3][0] , regB[0][1] , regA[3][1] , regB[1][1] , regA[3][2] , regB[2][1] , regA[3][3] , regB[3][1], wireOutMaxMin[3][1]); 
maxMin2 maxmin32(regA[3][0] , regB[0][2] , regA[3][1] , regB[1][2] , regA[3][2] , regB[2][2] , regA[3][3] , regB[3][2], wireOutMaxMin[3][2]); 
maxMin2 maxmin33(regA[3][0] , regB[0][3] , regA[3][1] , regB[1][3] , regA[3][2] , regB[2][3] , regA[3][3] , regB[3][3], wireOutMaxMin[3][3]);

assign out = regOut;
//end
always@(posedge clk)
begin
    regA[0][0] <= inA[0][0];
    regA[0][1] <= inA[0][1];
    regA[0][2] <= inA[0][2];
    regA[0][3] <= inA[0][3];
    regA[1][0] <= inA[1][0];
    regA[1][1] <= inA[1][1];
    regA[1][2] <= inA[1][2];
    regA[1][3] <= inA[1][3];
    regA[2][0] <= inA[2][0];
    regA[2][1] <= inA[2][1];
    regA[2][2] <= inA[2][2];
    regA[2][3] <= inA[2][3];
    regA[3][0] <= inA[3][0];
    regA[3][1] <= inA[3][1];
    regA[3][2] <= inA[3][2];
    regA[3][3] <= inA[3][3];

    regB[0][0] <= inB[0][0];
    regB[0][1] <= inB[0][1];
    regB[0][2] <= inB[0][2];
    regB[0][3] <= inB[0][3];
    regB[1][0] <= inB[1][0];
    regB[1][1] <= inB[1][1];
    regB[1][2] <= inB[1][2];
    regB[1][3] <= inB[1][3];
    regB[2][0] <= inB[2][0];
    regB[2][1] <= inB[2][1];
    regB[2][2] <= inB[2][2];
    regB[2][3] <= inB[2][3];
    regB[3][0] <= inB[3][0];
    regB[3][1] <= inB[3][1];
    regB[3][2] <= inB[3][2];
    regB[3][3] <= inB[3][3];

    if(op==0) begin
    regOut[0][0] <= wireOut[0][0];
    regOut[0][1] <= wireOut[0][1];
    regOut[0][2] <= wireOut[0][2];
    regOut[0][3] <= wireOut[0][3];
    regOut[1][0] <= wireOut[1][0];
    regOut[1][1] <= wireOut[1][1];
    regOut[1][2] <= wireOut[1][2];
    regOut[1][3] <= wireOut[1][3];
    regOut[2][0] <= wireOut[2][0];
    regOut[2][1] <= wireOut[2][1];
    regOut[2][2] <= wireOut[2][2];
    regOut[2][3] <= wireOut[2][3];
    regOut[3][0] <= wireOut[3][0];
    regOut[3][1] <= wireOut[3][1];
    regOut[3][2] <= wireOut[3][2];
    regOut[3][3] <= wireOut[3][3];
    end
    else begin
    regOut[0][0] <= wireOutMaxMin[0][0];
    regOut[0][1] <= wireOutMaxMin[0][1];
    regOut[0][2] <= wireOutMaxMin[0][2];
    regOut[0][3] <= wireOutMaxMin[0][3];
    regOut[1][0] <= wireOutMaxMin[1][0];
    regOut[1][1] <= wireOutMaxMin[1][1];
    regOut[1][2] <= wireOutMaxMin[1][2];
    regOut[1][3] <= wireOutMaxMin[1][3];
    regOut[2][0] <= wireOutMaxMin[2][0];
    regOut[2][1] <= wireOutMaxMin[2][1];
    regOut[2][2] <= wireOutMaxMin[2][2];
    regOut[2][3] <= wireOutMaxMin[2][3];
    regOut[3][0] <= wireOutMaxMin[3][0];
    regOut[3][1] <= wireOutMaxMin[3][1];
    regOut[3][2] <= wireOutMaxMin[3][2];
    regOut[3][3] <= wireOutMaxMin[3][3];
    end
end
endmodule