`timescale 1ns / 1ps
/*
 * Extended Tensor Cores
 */
 
module etcMinMax#(parameter W = 16)
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
wire [3:0][3:0][W-1:0] wireOutMinMax;

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

// MinMax logic
minMax2 minmax00(regA[0][0] , regB[0][0] , regA[0][1] , regB[1][0],  regA[0][2] , regB[2][0] , regA[0][3] , regB[3][0], wireOutMinMax[0][0]); 
minMax2 minmax01(regA[0][0] , regB[0][1] , regA[0][1] , regB[1][1] , regA[0][2] , regB[2][1] , regA[0][3] , regB[3][1], wireOutMinMax[0][1]); 
minMax2 minmax02(regA[0][0] , regB[0][2] , regA[0][1] , regB[1][2] , regA[0][2] , regB[2][2] , regA[0][3] , regB[3][2], wireOutMinMax[0][2]); 
minMax2 minmax03(regA[0][0] , regB[0][3] , regA[0][1] , regB[1][3] , regA[0][2] , regB[2][3] , regA[0][3] , regB[3][3], wireOutMinMax[0][3]); 
minMax2 minmax10(regA[1][0] , regB[0][0] , regA[1][1] , regB[1][0] , regA[1][2] , regB[2][0] , regA[1][3] , regB[3][0], wireOutMinMax[1][0]); 
minMax2 minmax11(regA[1][0] , regB[0][1] , regA[1][1] , regB[1][1] , regA[1][2] , regB[2][1] , regA[1][3] , regB[3][1], wireOutMinMax[1][1]); 
minMax2 minmax12(regA[1][0] , regB[0][2] , regA[1][1] , regB[1][2] , regA[1][2] , regB[2][2] , regA[1][3] , regB[3][2], wireOutMinMax[1][2]); 
minMax2 minmax13(regA[1][0] , regB[0][3] , regA[1][1] , regB[1][3] , regA[1][2] , regB[2][3] , regA[1][3] , regB[3][3], wireOutMinMax[1][3]); 
minMax2 minmax20(regA[2][0] , regB[0][0] , regA[2][1] , regB[1][0] , regA[2][2] , regB[2][0] , regA[2][3] , regB[3][0], wireOutMinMax[2][0]); 
minMax2 minmax21(regA[2][0] , regB[0][1] , regA[2][1] , regB[1][1] , regA[2][2] , regB[2][1] , regA[2][3] , regB[3][1], wireOutMinMax[2][1]); 
minMax2 minmax22(regA[2][0] , regB[0][2] , regA[2][1] , regB[1][2] , regA[2][2] , regB[2][2] , regA[2][3] , regB[3][2], wireOutMinMax[2][2]); 
minMax2 minmax23(regA[2][0] , regB[0][3] , regA[2][1] , regB[1][3] , regA[2][2] , regB[2][3] , regA[2][3] , regB[3][3], wireOutMinMax[2][3]); 
minMax2 minmax30(regA[3][0] , regB[0][0] , regA[3][1] , regB[1][0] , regA[3][2] , regB[2][0] , regA[3][3] , regB[3][0], wireOutMinMax[3][0]); 
minMax2 minmax31(regA[3][0] , regB[0][1] , regA[3][1] , regB[1][1] , regA[3][2] , regB[2][1] , regA[3][3] , regB[3][1], wireOutMinMax[3][1]); 
minMax2 minmax32(regA[3][0] , regB[0][2] , regA[3][1] , regB[1][2] , regA[3][2] , regB[2][2] , regA[3][3] , regB[3][2], wireOutMinMax[3][2]); 
minMax2 minmax33(regA[3][0] , regB[0][3] , regA[3][1] , regB[1][3] , regA[3][2] , regB[2][3] , regA[3][3] , regB[3][3], wireOutMinMax[3][3]);

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
    regOut[0][0] <= wireOutMinMax[0][0];
    regOut[0][1] <= wireOutMinMax[0][1];
    regOut[0][2] <= wireOutMinMax[0][2];
    regOut[0][3] <= wireOutMinMax[0][3];
    regOut[1][0] <= wireOutMinMax[1][0];
    regOut[1][1] <= wireOutMinMax[1][1];
    regOut[1][2] <= wireOutMinMax[1][2];
    regOut[1][3] <= wireOutMinMax[1][3];
    regOut[2][0] <= wireOutMinMax[2][0];
    regOut[2][1] <= wireOutMinMax[2][1];
    regOut[2][2] <= wireOutMinMax[2][2];
    regOut[2][3] <= wireOutMinMax[2][3];
    regOut[3][0] <= wireOutMinMax[3][0];
    regOut[3][1] <= wireOutMinMax[3][1];
    regOut[3][2] <= wireOutMinMax[3][2];
    regOut[3][3] <= wireOutMinMax[3][3];
    end
end
endmodule