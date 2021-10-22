`timescale 1ns / 1ps
/*
 * Extended Tensor Cores
 * University of California, Riverside
 * 
 * Written by Hung-Wei Tseng, 6/9/2021
 */
 
module etcEX#(parameter W = 16)
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
wire [3:0][3:0][W-1:0] wireOutAPSP;
wire [3:0][3:0][W-1:0] wireOutL2D;
reg [W-1:0] reg010, reg011, reg012, reg020, reg021, reg022, reg030, reg031, reg032, reg120, reg121, reg122, reg130, reg131, reg132, reg230, reg231, reg232;
// Logic for L2D
    min m00(regA[0][0] + regB[0][0] , regA[0][1] + regB[1][0],  regA[0][2] + regB[2][0] , regA[0][3] + regB[3][0], regA[0][0], wireOut[0][0]); 
    min m01(regA[0][0] + regB[0][1] , regA[0][1] + regB[1][1] , regA[0][2] + regB[2][1] , regA[0][3] + regB[3][1], regA[0][1], wireOut[0][1]); 
    min m02(regA[0][0] + regB[0][2] , regA[0][1] + regB[1][2] , regA[0][2] + regB[2][2] , regA[0][3] + regB[3][2], regA[0][2], wireOut[0][2]); 
    min m03(regA[0][0] + regB[0][3] , regA[0][1] + regB[1][3] , regA[0][2] + regB[2][3] , regA[0][3] + regB[3][3], regA[0][3], wireOut[0][3]); 
    min m10(regA[1][0] + regB[0][0] , regA[1][1] + regB[1][0] , regA[1][2] + regB[2][0] , regA[1][3] + regB[3][0], regA[1][0], wireOut[1][0]); 
    min m11(regA[1][0] + regB[0][1] , regA[1][1] + regB[1][1] , regA[1][2] + regB[2][1] , regA[1][3] + regB[3][1], regA[1][1], wireOut[1][1]); 
    min m12(regA[1][0] + regB[0][2] , regA[1][1] + regB[1][2] , regA[1][2] + regB[2][2] , regA[1][3] + regB[3][2], regA[1][2], wireOut[1][2]); 
    min m13(regA[1][0] + regB[0][3] , regA[1][1] + regB[1][3] , regA[1][2] + regB[2][3] , regA[1][3] + regB[3][3], regA[1][3], wireOut[1][3]); 
    min m20(regA[2][0] + regB[0][0] , regA[2][1] + regB[1][0] , regA[2][2] + regB[2][0] , regA[2][3] + regB[3][0], regA[2][0], wireOut[2][0]); 
    min m21(regA[2][0] + regB[0][1] , regA[2][1] + regB[1][1] , regA[2][2] + regB[2][1] , regA[2][3] + regB[3][1], regA[2][1], wireOut[2][1]); 
    min m22(regA[2][0] + regB[0][2] , regA[2][1] + regB[1][2] , regA[2][2] + regB[2][2] , regA[2][3] + regB[3][2], regA[2][2], wireOut[2][2]); 
    min m23(regA[2][0] + regB[0][3] , regA[2][1] + regB[1][3] , regA[2][2] + regB[2][3] , regA[2][3] + regB[3][3], regA[2][3], wireOut[2][3]); 
    min m30(regA[3][0] + regB[0][0] , regA[3][1] + regB[1][0] , regA[3][2] + regB[2][0] , regA[3][3] + regB[3][0], regA[3][0], wireOut[3][0]); 
    min m31(regA[3][0] + regB[0][1] , regA[3][1] + regB[1][1] , regA[3][2] + regB[2][1] , regA[3][3] + regB[3][1], regA[3][1], wireOut[3][1]); 
    min m32(regA[3][0] + regB[0][2] , regA[3][1] + regB[1][2] , regA[3][2] + regB[2][2] , regA[3][3] + regB[3][2], regA[3][2], wireOut[3][2]); 
    min m33(regA[3][0] + regB[0][3] , regA[3][1] + regB[1][3] , regA[3][2] + regB[2][3] , regA[3][3] + regB[3][3], regA[3][3], wireOut[3][3]); 
// MAC Logic
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
// L2D Logic
    assign reg010 = regA[0][0] - regB[1][0];
    assign reg011 = regA[0][1] - regB[1][1];
    assign reg012 = regA[0][2] - regB[1][2];
    assign reg020 = regA[0][0] - regB[2][0];
    assign reg021 = regA[0][1] - regB[2][1];
    assign reg022 = regA[0][2] - regB[2][2];
    assign reg030 = regA[0][0] - regB[3][0];
    assign reg031 = regA[0][1] - regB[3][1];
    assign reg032 = regA[0][2] - regB[3][2];
    assign reg120 = regA[1][0] - regB[2][0];
    assign reg121 = regA[1][1] - regB[2][1];
    assign reg122 = regA[1][2] - regB[2][2];
    assign reg130 = regA[1][0] - regB[3][0];
    assign reg131 = regA[1][1] - regB[3][1];
    assign reg132 = regA[1][2] - regB[3][2];
    assign reg230 = regA[2][0] - regB[3][0];
    assign reg231 = regA[2][1] - regB[3][1];
    assign reg232 = regA[2][2] - regB[3][2];
    assign wireOutL2D[0][0] = 0;
    assign wireOutL2D[0][1] = reg010*reg010 + reg011*reg011 + reg012*reg012;
    assign wireOutL2D[0][2] = reg020*reg020 + reg021*reg021 + reg022*reg022;
    assign wireOutL2D[0][3] = reg030*reg030 + reg021*reg031 + reg032*reg032;
    assign wireOutL2D[1][0] = 0;
    assign wireOutL2D[1][1] = 0;
    assign wireOutL2D[1][2] = reg120*reg120 + reg121*reg121 + reg122*reg122;
    assign wireOutL2D[1][3] = reg130*reg130 + reg131*reg131 + reg132*reg132;
    assign wireOutL2D[2][0] = 0;
    assign wireOutL2D[2][1] = 0;
    assign wireOutL2D[2][2] = 0;
    assign wireOutL2D[2][3] = reg230*reg230 + reg231*reg231 + reg232*reg232;
    assign wireOutL2D[3][0] = 0;
    assign wireOutL2D[3][1] = 0;
    assign wireOutL2D[3][2] = 0;
    assign wireOutL2D[3][3] = 0;
assign out = regOut;
//end
always@(posedge clk)
begin
// regOut[16*8-1:0] <= {wireOut[0][7],wireOut[0][6],wireOut[0][5],wireOut[0][4],wireOut[0][3],wireOut[0][2],wireOut[0][1],wireOut[0][0]}};
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
//        $monitor("At time %t, regOut[0][0] = %h (%0d)", $time, regOut[0][0], regOut[0][0]);
    end
    else if(op==2'b01) begin
    regOut[0][0] <= wireOutL2D[0][0];
    regOut[0][1] <= wireOutL2D[0][1];
    regOut[0][2] <= wireOutL2D[0][2];
    regOut[0][3] <= wireOutL2D[0][3];
    regOut[1][0] <= wireOutL2D[1][0];
    regOut[1][1] <= wireOutL2D[1][1];
    regOut[1][2] <= wireOutL2D[1][2];
    regOut[1][3] <= wireOutL2D[1][3];
    regOut[2][0] <= wireOutL2D[2][0];
    regOut[2][1] <= wireOutL2D[2][1];
    regOut[2][2] <= wireOutL2D[2][2];
    regOut[2][3] <= wireOutL2D[2][3];
    regOut[3][0] <= wireOutL2D[3][0];
    regOut[3][1] <= wireOutL2D[3][1];
    regOut[3][2] <= wireOutL2D[3][2];
    regOut[3][3] <= wireOutL2D[3][3];
    end
    else begin
    regOut[0][0] <= wireOutAPSP[0][0];
    regOut[0][1] <= wireOutAPSP[0][1];
    regOut[0][2] <= wireOutAPSP[0][2];
    regOut[0][3] <= wireOutAPSP[0][3];
    regOut[1][0] <= wireOutAPSP[1][0];
    regOut[1][1] <= wireOutAPSP[1][1];
    regOut[1][2] <= wireOutAPSP[1][2];
    regOut[1][3] <= wireOutAPSP[1][3];
    regOut[2][0] <= wireOutAPSP[2][0];
    regOut[2][1] <= wireOutAPSP[2][1];
    regOut[2][2] <= wireOutAPSP[2][2];
    regOut[2][3] <= wireOutAPSP[2][3];
    regOut[3][0] <= wireOutAPSP[3][0];
    regOut[3][1] <= wireOutAPSP[3][1];
    regOut[3][2] <= wireOutAPSP[3][2];
    regOut[3][3] <= wireOutAPSP[3][3];
        
    end
end

endmodule
