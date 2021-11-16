`timescale 1ns / 1ps
/*
 * Extended Tensor Cores
 * University of California, Riverside
 * 
 * Written by Hung-Wei Tseng, 6/9/2021
 */
 
module etcEX2#(parameter W = 16)
(
 input clk,
 input [3:0] op,
 input [3:0][3:0][W-1:0] inA, inB,
 output [3:0][3:0][W-1:0] out
);
integer i,j;
reg [3:0][3:0][W-1:0] regA, regB;
reg [3:0][3:0][W-1:0] regOut;
wire [3:0][3:0][W-1:0] wireOut;
wire [3:0][3:0][W-1:0] wireOutMax;
wire [3:0][3:0][W-1:0] wireOutAPSP;
wire [3:0][3:0][W-1:0] wireOutL2D;
wire [3:0][3:0][W-1:0] wireOutOrAnd;
reg [W-1:0] reg010, reg011, reg012, reg020, reg021, reg022, reg030, reg031, reg032, reg120, reg121, reg122, reg130, reg131, reg132, reg230, reg231, reg232;
// Logic for MinPlus/MinMul

    minEX m00(regA[0][0] , regB[0][0] , regA[0][1] , regB[1][0],  regA[0][2] , regB[2][0] , regA[0][3] , regB[3][0], regA[0][0], wireOut[0][0], op[0]); 
    minEX m01(regA[0][0] , regB[0][1] , regA[0][1] , regB[1][1] , regA[0][2] , regB[2][1] , regA[0][3] , regB[3][1], regA[0][1], wireOut[0][1], op[0]); 
    minEX m02(regA[0][0] , regB[0][2] , regA[0][1] , regB[1][2] , regA[0][2] , regB[2][2] , regA[0][3] , regB[3][2], regA[0][2], wireOut[0][2], op[0]); 
    minEX m03(regA[0][0] , regB[0][3] , regA[0][1] , regB[1][3] , regA[0][2] , regB[2][3] , regA[0][3] , regB[3][3], regA[0][3], wireOut[0][3], op[0]); 
    minEX m10(regA[1][0] , regB[0][0] , regA[1][1] , regB[1][0] , regA[1][2] , regB[2][0] , regA[1][3] , regB[3][0], regA[1][0], wireOut[1][0], op[0]); 
    minEX m11(regA[1][0] , regB[0][1] , regA[1][1] , regB[1][1] , regA[1][2] , regB[2][1] , regA[1][3] , regB[3][1], regA[1][1], wireOut[1][1], op[0]); 
    minEX m12(regA[1][0] , regB[0][2] , regA[1][1] , regB[1][2] , regA[1][2] , regB[2][2] , regA[1][3] , regB[3][2], regA[1][2], wireOut[1][2], op[0]); 
    minEX m13(regA[1][0] , regB[0][3] , regA[1][1] , regB[1][3] , regA[1][2] , regB[2][3] , regA[1][3] , regB[3][3], regA[1][3], wireOut[1][3], op[0]); 
    minEX m20(regA[2][0] , regB[0][0] , regA[2][1] , regB[1][0] , regA[2][2] , regB[2][0] , regA[2][3] , regB[3][0], regA[2][0], wireOut[2][0], op[0]); 
    minEX m21(regA[2][0] , regB[0][1] , regA[2][1] , regB[1][1] , regA[2][2] , regB[2][1] , regA[2][3] , regB[3][1], regA[2][1], wireOut[2][1], op[0]); 
    minEX m22(regA[2][0] , regB[0][2] , regA[2][1] , regB[1][2] , regA[2][2] , regB[2][2] , regA[2][3] , regB[3][2], regA[2][2], wireOut[2][2], op[0]); 
    minEX m23(regA[2][0] , regB[0][3] , regA[2][1] , regB[1][3] , regA[2][2] , regB[2][3] , regA[2][3] , regB[3][3], regA[2][3], wireOut[2][3], op[0]); 
    minEX m30(regA[3][0] , regB[0][0] , regA[3][1] , regB[1][0] , regA[3][2] , regB[2][0] , regA[3][3] , regB[3][0], regA[3][0], wireOut[3][0], op[0]); 
    minEX m31(regA[3][0] , regB[0][1] , regA[3][1] , regB[1][1] , regA[3][2] , regB[2][1] , regA[3][3] , regB[3][1], regA[3][1], wireOut[3][1], op[0]); 
    minEX m32(regA[3][0] , regB[0][2] , regA[3][1] , regB[1][2] , regA[3][2] , regB[2][2] , regA[3][3] , regB[3][2], regA[3][2], wireOut[3][2], op[0]); 
    minEX m33(regA[3][0] , regB[0][3] , regA[3][1] , regB[1][3] , regA[3][2] , regB[2][3] , regA[3][3] , regB[3][3], regA[3][3], wireOut[3][3], op[0]); 
/*
// Logic for MinPlus
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
// Logic for MinMul
    min minmul00(regA[0][0] * regB[0][0] , regA[0][1] * regB[1][0],  regA[0][2] * regB[2][0] , regA[0][3] * regB[3][0], regA[0][0], wireOutMinMul[0][0]); 
    min minmul01(regA[0][0] * regB[0][1] , regA[0][1] * regB[1][1] , regA[0][2] * regB[2][1] , regA[0][3] * regB[3][1], regA[0][1], wireOutMinMul[0][1]); 
    min minmul02(regA[0][0] * regB[0][2] , regA[0][1] * regB[1][2] , regA[0][2] * regB[2][2] , regA[0][3] * regB[3][2], regA[0][2], wireOutMinMul[0][2]); 
    min minmul03(regA[0][0] * regB[0][3] , regA[0][1] * regB[1][3] , regA[0][2] * regB[2][3] , regA[0][3] * regB[3][3], regA[0][3], wireOutMinMul[0][3]); 
    min minmul10(regA[1][0] * regB[0][0] , regA[1][1] * regB[1][0] , regA[1][2] * regB[2][0] , regA[1][3] * regB[3][0], regA[1][0], wireOutMinMul[1][0]); 
    min minmul11(regA[1][0] * regB[0][1] , regA[1][1] * regB[1][1] , regA[1][2] * regB[2][1] , regA[1][3] * regB[3][1], regA[1][1], wireOutMinMul[1][1]); 
    min minmul12(regA[1][0] * regB[0][2] , regA[1][1] * regB[1][2] , regA[1][2] * regB[2][2] , regA[1][3] * regB[3][2], regA[1][2], wireOutMinMul[1][2]); 
    min minmul13(regA[1][0] * regB[0][3] , regA[1][1] * regB[1][3] , regA[1][2] * regB[2][3] , regA[1][3] * regB[3][3], regA[1][3], wireOutMinMul[1][3]); 
    min minmul20(regA[2][0] * regB[0][0] , regA[2][1] * regB[1][0] , regA[2][2] * regB[2][0] , regA[2][3] * regB[3][0], regA[2][0], wireOutMinMul[2][0]); 
    min minmul21(regA[2][0] * regB[0][1] , regA[2][1] * regB[1][1] , regA[2][2] * regB[2][1] , regA[2][3] * regB[3][1], regA[2][1], wireOutMinMul[2][1]); 
    min minmul22(regA[2][0] * regB[0][2] , regA[2][1] * regB[1][2] , regA[2][2] * regB[2][2] , regA[2][3] * regB[3][2], regA[2][2], wireOutMinMul[2][2]); 
    min minmul23(regA[2][0] * regB[0][3] , regA[2][1] * regB[1][3] , regA[2][2] * regB[2][3] , regA[2][3] * regB[3][3], regA[2][3], wireOutMinMul[2][3]); 
    min minmul30(regA[3][0] * regB[0][0] , regA[3][1] * regB[1][0] , regA[3][2] * regB[2][0] , regA[3][3] * regB[3][0], regA[3][0], wireOutMinMul[3][0]); 
    min minmul31(regA[3][0] * regB[0][1] , regA[3][1] * regB[1][1] , regA[3][2] * regB[2][1] , regA[3][3] * regB[3][1], regA[3][1], wireOutMinMul[3][1]); 
    min minmul32(regA[3][0] * regB[0][2] , regA[3][1] * regB[1][2] , regA[3][2] * regB[2][2] , regA[3][3] * regB[3][2], regA[3][2], wireOutMinMul[3][2]); 
    min minmul33(regA[3][0] * regB[0][3] , regA[3][1] * regB[1][3] , regA[3][2] * regB[2][3] , regA[3][3] * regB[3][3], regA[3][3], wireOutMinMul[3][3]); 
*/
/*
    maxEX max00(regA[0][0] , regB[0][0] , regA[0][1] , regB[1][0],  regA[0][2] , regB[2][0] , regA[0][3] , regB[3][0], regA[0][0], wireOutMax[0][0], op[0]); 
    maxEX max01(regA[0][0] , regB[0][1] , regA[0][1] , regB[1][1] , regA[0][2] , regB[2][1] , regA[0][3] , regB[3][1], regA[0][1], wireOutMax[0][1], op[0]); 
    maxEX max02(regA[0][0] , regB[0][2] , regA[0][1] , regB[1][2] , regA[0][2] , regB[2][2] , regA[0][3] , regB[3][2], regA[0][2], wireOutMax[0][2], op[0]); 
    maxEX max03(regA[0][0] , regB[0][3] , regA[0][1] , regB[1][3] , regA[0][2] , regB[2][3] , regA[0][3] , regB[3][3], regA[0][3], wireOutMax[0][3], op[0]); 
    maxEX max10(regA[1][0] , regB[0][0] , regA[1][1] , regB[1][0] , regA[1][2] , regB[2][0] , regA[1][3] , regB[3][0], regA[1][0], wireOutMax[1][0], op[0]); 
    maxEX max11(regA[1][0] , regB[0][1] , regA[1][1] , regB[1][1] , regA[1][2] , regB[2][1] , regA[1][3] , regB[3][1], regA[1][1], wireOutMax[1][1], op[0]); 
    maxEX max12(regA[1][0] , regB[0][2] , regA[1][1] , regB[1][2] , regA[1][2] , regB[2][2] , regA[1][3] , regB[3][2], regA[1][2], wireOutMax[1][2], op[0]); 
    maxEX max13(regA[1][0] , regB[0][3] , regA[1][1] , regB[1][3] , regA[1][2] , regB[2][3] , regA[1][3] , regB[3][3], regA[1][3], wireOutMax[1][3], op[0]); 
    maxEX max20(regA[2][0] , regB[0][0] , regA[2][1] , regB[1][0] , regA[2][2] , regB[2][0] , regA[2][3] , regB[3][0], regA[2][0], wireOutMax[2][0], op[0]); 
    maxEX max21(regA[2][0] , regB[0][1] , regA[2][1] , regB[1][1] , regA[2][2] , regB[2][1] , regA[2][3] , regB[3][1], regA[2][1], wireOutMax[2][1], op[0]); 
    maxEX max22(regA[2][0] , regB[0][2] , regA[2][1] , regB[1][2] , regA[2][2] , regB[2][2] , regA[2][3] , regB[3][2], regA[2][2], wireOutMax[2][2], op[0]); 
    maxEX max23(regA[2][0] , regB[0][3] , regA[2][1] , regB[1][3] , regA[2][2] , regB[2][3] , regA[2][3] , regB[3][3], regA[2][3], wireOutMax[2][3], op[0]); 
    maxEX max30(regA[3][0] , regB[0][0] , regA[3][1] , regB[1][0] , regA[3][2] , regB[2][0] , regA[3][3] , regB[3][0], regA[3][0], wireOutMax[3][0], op[0]); 
    maxEX max31(regA[3][0] , regB[0][1] , regA[3][1] , regB[1][1] , regA[3][2] , regB[2][1] , regA[3][3] , regB[3][1], regA[3][1], wireOutMax[3][1], op[0]); 
    maxEX max32(regA[3][0] , regB[0][2] , regA[3][1] , regB[1][2] , regA[3][2] , regB[2][2] , regA[3][3] , regB[3][2], regA[3][2], wireOutMax[3][2], op[0]); 
    maxEX max33(regA[3][0] , regB[0][3] , regA[3][1] , regB[1][3] , regA[3][2] , regB[2][3] , regA[3][3] , regB[3][3], regA[3][3], wireOutMax[3][3], op[0]); 
*/
// Logic for MaxMul/Maxplus

    max maxmul00(regA[0][0] * regB[0][0] , regA[0][1] * regB[1][0],  regA[0][2] * regB[2][0] , regA[0][3] * regB[3][0], regA[0][0], wireOutMaxMul[0][0]); 
    max maxmul01(regA[0][0] * regB[0][1] , regA[0][1] * regB[1][1] , regA[0][2] * regB[2][1] , regA[0][3] * regB[3][1], regA[0][1], wireOutMaxMul[0][1]); 
    max maxmul02(regA[0][0] * regB[0][2] , regA[0][1] * regB[1][2] , regA[0][2] * regB[2][2] , regA[0][3] * regB[3][2], regA[0][2], wireOutMaxMul[0][2]); 
    max maxmul03(regA[0][0] * regB[0][3] , regA[0][1] * regB[1][3] , regA[0][2] * regB[2][3] , regA[0][3] * regB[3][3], regA[0][3], wireOutMaxMul[0][3]); 
    max maxmul10(regA[1][0] * regB[0][0] , regA[1][1] * regB[1][0] , regA[1][2] * regB[2][0] , regA[1][3] * regB[3][0], regA[1][0], wireOutMaxMul[1][0]); 
    max maxmul11(regA[1][0] * regB[0][1] , regA[1][1] * regB[1][1] , regA[1][2] * regB[2][1] , regA[1][3] * regB[3][1], regA[1][1], wireOutMaxMul[1][1]); 
    max maxmul12(regA[1][0] * regB[0][2] , regA[1][1] * regB[1][2] , regA[1][2] * regB[2][2] , regA[1][3] * regB[3][2], regA[1][2], wireOutMaxMul[1][2]); 
    max maxmul13(regA[1][0] * regB[0][3] , regA[1][1] * regB[1][3] , regA[1][2] * regB[2][3] , regA[1][3] * regB[3][3], regA[1][3], wireOutMaxMul[1][3]); 
    max maxmul20(regA[2][0] * regB[0][0] , regA[2][1] * regB[1][0] , regA[2][2] * regB[2][0] , regA[2][3] * regB[3][0], regA[2][0], wireOutMaxMul[2][0]); 
    max maxmul21(regA[2][0] * regB[0][1] , regA[2][1] * regB[1][1] , regA[2][2] * regB[2][1] , regA[2][3] * regB[3][1], regA[2][1], wireOutMaxMul[2][1]); 
    max maxmul22(regA[2][0] * regB[0][2] , regA[2][1] * regB[1][2] , regA[2][2] * regB[2][2] , regA[2][3] * regB[3][2], regA[2][2], wireOutMaxMul[2][2]); 
    max maxmul23(regA[2][0] * regB[0][3] , regA[2][1] * regB[1][3] , regA[2][2] * regB[2][3] , regA[2][3] * regB[3][3], regA[2][3], wireOutMaxMul[2][3]); 
    max maxmul30(regA[3][0] * regB[0][0] , regA[3][1] * regB[1][0] , regA[3][2] * regB[2][0] , regA[3][3] * regB[3][0], regA[3][0], wireOutMaxMul[3][0]); 
    max maxmul31(regA[3][0] * regB[0][1] , regA[3][1] * regB[1][1] , regA[3][2] * regB[2][1] , regA[3][3] * regB[3][1], regA[3][1], wireOutMaxMul[3][1]); 
    max maxmul32(regA[3][0] * regB[0][2] , regA[3][1] * regB[1][2] , regA[3][2] * regB[2][2] , regA[3][3] * regB[3][2], regA[3][2], wireOutMaxMul[3][2]); 
    max maxmul33(regA[3][0] * regB[0][3] , regA[3][1] * regB[1][3] , regA[3][2] * regB[2][3] , regA[3][3] * regB[3][3], regA[3][3], wireOutMaxMul[3][3]); 
// Logic for MaxPlus
    max maxplus00(regA[0][0] + regB[0][0] , regA[0][1] + regB[1][0],  regA[0][2] + regB[2][0] , regA[0][3] + regB[3][0], regA[0][0], wireOutMaxPlus[0][0]); 
    max maxplus01(regA[0][0] + regB[0][1] , regA[0][1] + regB[1][1] , regA[0][2] + regB[2][1] , regA[0][3] + regB[3][1], regA[0][1], wireOutMaxPlus[0][1]); 
    max maxplus02(regA[0][0] + regB[0][2] , regA[0][1] + regB[1][2] , regA[0][2] + regB[2][2] , regA[0][3] + regB[3][2], regA[0][2], wireOutMaxPlus[0][2]); 
    max maxplus03(regA[0][0] + regB[0][3] , regA[0][1] + regB[1][3] , regA[0][2] + regB[2][3] , regA[0][3] + regB[3][3], regA[0][3], wireOutMaxPlus[0][3]); 
    max maxplus10(regA[1][0] + regB[0][0] , regA[1][1] + regB[1][0] , regA[1][2] + regB[2][0] , regA[1][3] + regB[3][0], regA[1][0], wireOutMaxPlus[1][0]); 
    max maxplus11(regA[1][0] + regB[0][1] , regA[1][1] + regB[1][1] , regA[1][2] + regB[2][1] , regA[1][3] + regB[3][1], regA[1][1], wireOutMaxPlus[1][1]); 
    max maxplus12(regA[1][0] + regB[0][2] , regA[1][1] + regB[1][2] , regA[1][2] + regB[2][2] , regA[1][3] + regB[3][2], regA[1][2], wireOutMaxPlus[1][2]); 
    max maxplus13(regA[1][0] + regB[0][3] , regA[1][1] + regB[1][3] , regA[1][2] + regB[2][3] , regA[1][3] + regB[3][3], regA[1][3], wireOutMaxPlus[1][3]); 
    max maxplus20(regA[2][0] + regB[0][0] , regA[2][1] + regB[1][0] , regA[2][2] + regB[2][0] , regA[2][3] + regB[3][0], regA[2][0], wireOutMaxPlus[2][0]); 
    max maxplus21(regA[2][0] + regB[0][1] , regA[2][1] + regB[1][1] , regA[2][2] + regB[2][1] , regA[2][3] + regB[3][1], regA[2][1], wireOutMaxPlus[2][1]); 
    max maxplus22(regA[2][0] + regB[0][2] , regA[2][1] + regB[1][2] , regA[2][2] + regB[2][2] , regA[2][3] + regB[3][2], regA[2][2], wireOutMaxPlus[2][2]); 
    max maxplus23(regA[2][0] + regB[0][3] , regA[2][1] + regB[1][3] , regA[2][2] + regB[2][3] , regA[2][3] + regB[3][3], regA[2][3], wireOutMaxPlus[2][3]); 
    max maxplus30(regA[3][0] + regB[0][0] , regA[3][1] + regB[1][0] , regA[3][2] + regB[2][0] , regA[3][3] + regB[3][0], regA[3][0], wireOutMaxPlus[3][0]); 
    max maxplus31(regA[3][0] + regB[0][1] , regA[3][1] + regB[1][1] , regA[3][2] + regB[2][1] , regA[3][3] + regB[3][1], regA[3][1], wireOutMaxPlus[3][1]); 
    max maxplus32(regA[3][0] + regB[0][2] , regA[3][1] + regB[1][2] , regA[3][2] + regB[2][2] , regA[3][3] + regB[3][2], regA[3][2], wireOutMaxPlus[3][2]); 
    max maxplus33(regA[3][0] + regB[0][3] , regA[3][1] + regB[1][3] , regA[3][2] + regB[2][3] , regA[3][3] + regB[3][3], regA[3][3], wireOutMaxPlus[3][3]); 

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

// OrAnd Logic
    assign wireOutOrAnd[0][0] = (regA[0][0]  | regB[0][0]) & (regA[0][1]  | regB[1][0]) & (regA[0][2]  | regB[2][0]) & (regA[0][3]  | regB[3][0]);
    assign wireOutOrAnd[0][1] = (regA[0][0]  | regB[0][1]) & (regA[0][1]  | regB[1][1]) & (regA[0][2]  | regB[2][1]) & (regA[0][3]  | regB[3][1]);
    assign wireOutOrAnd[0][2] = (regA[0][0]  | regB[0][2]) & (regA[0][1]  | regB[1][2]) & (regA[0][2]  | regB[2][2]) & (regA[0][3]  | regB[3][2]);
    assign wireOutOrAnd[0][3] = (regA[0][0]  | regB[0][3]) & (regA[0][1]  | regB[1][3]) & (regA[0][2]  | regB[2][3]) & (regA[0][3]  | regB[3][3]);
    assign wireOutOrAnd[1][0] = (regA[1][0]  | regB[0][0]) & (regA[1][1]  | regB[1][0]) & (regA[1][2]  | regB[2][0]) & (regA[1][3]  | regB[3][0]);
    assign wireOutOrAnd[1][1] = (regA[1][0]  | regB[0][1]) & (regA[1][1]  | regB[1][1]) & (regA[1][2]  | regB[2][1]) & (regA[1][3]  | regB[3][1]);
    assign wireOutOrAnd[1][2] = (regA[1][0]  | regB[0][2]) & (regA[1][1]  | regB[1][2]) & (regA[1][2]  | regB[2][2]) & (regA[1][3]  | regB[3][2]);
    assign wireOutOrAnd[1][3] = (regA[1][0]  | regB[0][3]) & (regA[1][1]  | regB[1][3]) & (regA[1][2]  | regB[2][3]) & (regA[1][3]  | regB[3][3]);
    assign wireOutOrAnd[2][0] = (regA[2][0]  | regB[0][0]) & (regA[2][1]  | regB[1][0]) & (regA[2][2]  | regB[2][0]) & (regA[2][3]  | regB[3][0]);
    assign wireOutOrAnd[2][1] = (regA[2][0]  | regB[0][1]) & (regA[2][1]  | regB[1][1]) & (regA[2][2]  | regB[2][1]) & (regA[2][3]  | regB[3][1]);
    assign wireOutOrAnd[2][2] = (regA[2][0]  | regB[0][2]) & (regA[2][1]  | regB[1][2]) & (regA[2][2]  | regB[2][2]) & (regA[2][3]  | regB[3][2]);
    assign wireOutOrAnd[2][3] = (regA[2][0]  | regB[0][3]) & (regA[2][1]  | regB[1][3]) & (regA[2][2]  | regB[2][3]) & (regA[2][3]  | regB[3][3]);
    assign wireOutOrAnd[3][0] = (regA[3][0]  | regB[0][0]) & (regA[3][1]  | regB[1][0]) & (regA[3][2]  | regB[2][0]) & (regA[3][3]  | regB[3][0]);
    assign wireOutOrAnd[3][1] = (regA[3][0]  | regB[0][1]) & (regA[3][1]  | regB[1][1]) & (regA[3][2]  | regB[2][1]) & (regA[3][3]  | regB[3][1]);
    assign wireOutOrAnd[3][2] = (regA[3][0]  | regB[0][2]) & (regA[3][1]  | regB[1][2]) & (regA[3][2]  | regB[2][2]) & (regA[3][3]  | regB[3][2]);
    assign wireOutOrAnd[3][3] = (regA[3][0]  | regB[0][3]) & (regA[3][1]  | regB[1][3]) & (regA[3][2]  | regB[2][3]) & (regA[3][3]  | regB[3][3]);

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
    else if(op==4'b001) begin
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

    else if(op==4'b010) begin
    regOut[0][0] <= wireOutMinMul[0][0];
    regOut[0][1] <= wireOutMinMul[0][1];
    regOut[0][2] <= wireOutMinMul[0][2];
    regOut[0][3] <= wireOutMinMul[0][3];
    regOut[1][0] <= wireOutMinMul[1][0];
    regOut[1][1] <= wireOutMinMul[1][1];
    regOut[1][2] <= wireOutMinMul[1][2];
    regOut[1][3] <= wireOutMinMul[1][3];
    regOut[2][0] <= wireOutMinMul[2][0];
    regOut[2][1] <= wireOutMinMul[2][1];
    regOut[2][2] <= wireOutMinMul[2][2];
    regOut[2][3] <= wireOutMinMul[2][3];
    regOut[3][0] <= wireOutMinMul[3][0];
    regOut[3][1] <= wireOutMinMul[3][1];
    regOut[3][2] <= wireOutMinMul[3][2];
    regOut[3][3] <= wireOutMinMul[3][3];
    end

    else if(op==4'b011) begin
    regOut[0][0] <= wireOutMaxPlus[0][0];
    regOut[0][1] <= wireOutMaxPlus[0][1];
    regOut[0][2] <= wireOutMaxPlus[0][2];
    regOut[0][3] <= wireOutMaxPlus[0][3];
    regOut[1][0] <= wireOutMaxPlus[1][0];
    regOut[1][1] <= wireOutMaxPlus[1][1];
    regOut[1][2] <= wireOutMaxPlus[1][2];
    regOut[1][3] <= wireOutMaxPlus[1][3];
    regOut[2][0] <= wireOutMaxPlus[2][0];
    regOut[2][1] <= wireOutMaxPlus[2][1];
    regOut[2][2] <= wireOutMaxPlus[2][2];
    regOut[2][3] <= wireOutMaxPlus[2][3];
    regOut[3][0] <= wireOutMaxPlus[3][0];
    regOut[3][1] <= wireOutMaxPlus[3][1];
    regOut[3][2] <= wireOutMaxPlus[3][2];
    regOut[3][3] <= wireOutMaxPlus[3][3];
    end
    else if(op==4'b100) begin
    regOut[0][0] <= wireOutMaxMul[0][0];
    regOut[0][1] <= wireOutMaxMul[0][1];
    regOut[0][2] <= wireOutMaxMul[0][2];
    regOut[0][3] <= wireOutMaxMul[0][3];
    regOut[1][0] <= wireOutMaxMul[1][0];
    regOut[1][1] <= wireOutMaxMul[1][1];
    regOut[1][2] <= wireOutMaxMul[1][2];
    regOut[1][3] <= wireOutMaxMul[1][3];
    regOut[2][0] <= wireOutMaxMul[2][0];
    regOut[2][1] <= wireOutMaxMul[2][1];
    regOut[2][2] <= wireOutMaxMul[2][2];
    regOut[2][3] <= wireOutMaxMul[2][3];
    regOut[3][0] <= wireOutMaxMul[3][0];
    regOut[3][1] <= wireOutMaxMul[3][1];
    regOut[3][2] <= wireOutMaxMul[3][2];
    regOut[3][3] <= wireOutMaxMul[3][3];
    end
    else if(op==4'b101) begin
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
    else if(op==4'b111) begin
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
    else if(op==4'b1000) begin
    regOut[0][0] <= wireOutOrAnd[0][0];
    regOut[0][1] <= wireOutOrAnd[0][1];
    regOut[0][2] <= wireOutOrAnd[0][2];
    regOut[0][3] <= wireOutOrAnd[0][3];
    regOut[1][0] <= wireOutOrAnd[1][0];
    regOut[1][1] <= wireOutOrAnd[1][1];
    regOut[1][2] <= wireOutOrAnd[1][2];
    regOut[1][3] <= wireOutOrAnd[1][3];
    regOut[2][0] <= wireOutOrAnd[2][0];
    regOut[2][1] <= wireOutOrAnd[2][1];
    regOut[2][2] <= wireOutOrAnd[2][2];
    regOut[2][3] <= wireOutOrAnd[2][3];
    regOut[3][0] <= wireOutOrAnd[3][0];
    regOut[3][1] <= wireOutOrAnd[3][1];
    regOut[3][2] <= wireOutOrAnd[3][2];
    regOut[3][3] <= wireOutOrAnd[3][3];
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
