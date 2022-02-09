`timescale 1ns / 1ps
/*
 * Extended Tensor Cores version 3
 */

module etcEX4#(parameter W = 16)
(
input clk,
input [4:0] op,
input [3:0][3:0][W-1:0] inA, inB,
output [3:0][3:0][W-1:0] out 
);

logic [3:0][3:0][W-1:0] regA, regB;
logic [3:0][3:0][W-1:0] regOut;

logic [3:0][3:0][W-1:0] regTemp00;
logic [3:0][3:0][W-1:0] regTemp01;
logic [3:0][3:0][W-1:0] regTemp10;
logic [3:0][3:0][W-1:0] regTemp11;

genvar i,j;

// No re-use logics

wire [3:0][3:0][W-1:0] wireOutMinMax;
wire [3:0][3:0][W-1:0] wireOutMaxMin;
wire [3:0][3:0][W-1:0] wireOutOrAnd;
wire [3:0][3:0][W-1:0] wireOutL2D;
reg [W-1:0] reg010, reg011, reg012, reg020, reg021, reg022, reg030, reg031, reg032, reg120, reg121, reg122, reg130, reg131, reg132, reg230, reg231, reg232;

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

maxMin2 maxmin00(regA[0][0] , regB[0][0] , regA[0][1] , regB[1][0],  regA[0][2] , regB[2][0] , regA[0][3] , regB[3][0], wireOutMaxMin[0][0]); 
maxMin2 maxmin01(regA[0][0] , regB[0][1] , regA[0][1] , regB[1][1] , regA[0][2] , regB[2][1] , regA[0][3] , regB[3][1], wireOutMaxMin[0][1]); 
maxMin2 maxmin02(regA[0][0] , regB[0][2] , regA[0][1] , regB[1][2] , regA[0][2] , regB[2][2] , regA[0][3] , regB[3][2], wireOutMaxMin[0][2]); 
maxMin2 maxmin03(regA[0][0] , regB[0][3] , regA[0][1] , regB[1][3] , regA[0][2] , regB[2][3] , regA[0][3] , regB[3][3], wireOutMaxMin[0][3]); 
maxMin2 maxmin10(regA[1][0] , regB[0][0] , regA[1][1] , regB[1][0] , regA[1][2] , regB[2][0] , regA[1][3] , regB[3][0], wireOutMaxMin[1][0]); 
maxMin2 maxmin11(regA[1][0] , regB[0][1] , regA[1][1] , regB[1][1] , regA[1][2] , regB[2][1] , regA[1][3] , regB[3][1], wireOutMaxMin[1][1]); 
maxMin2 maxmin12(regA[1][0] , regB[0][2] , regA[1][1] , regB[1][2] , regA[1][2] , regB[2][2] , regA[1][3] , regB[3][2], wireOutMaxMin[1][2]); 
maxMin2 maxmin13(regA[1][0] , regB[0][3] , regA[1][1] , regB[1][3] , regA[1][2] , regB[2][3] , regA[1][3] , regB[3][3], wireOutMaxMin[1][3]); 
maxMin2 maxmin20(regA[2][0] , regB[0][0] , regA[2][1] , regB[1][0] , regA[2][2] , regB[2][0] , regA[2][3] , regB[3][0], wireOutMaxMin[2][0]); 
maxMin2 maxmin21(regA[2][0] , regB[0][1] , regA[2][1] , regB[1][1] , regA[2][2] , regB[2][1] , regA[2][3] , regB[3][1], wireOutMaxMin[2][1]); 
maxMin2 maxmin22(regA[2][0] , regB[0][2] , regA[2][1] , regB[1][2] , regA[2][2] , regB[2][2] , regA[2][3] , regB[3][2], wireOutMaxMin[2][2]); 
maxMin2 maxmin23(regA[2][0] , regB[0][3] , regA[2][1] , regB[1][3] , regA[2][2] , regB[2][3] , regA[2][3] , regB[3][3], wireOutMaxMin[2][3]); 
maxMin2 maxmin30(regA[3][0] , regB[0][0] , regA[3][1] , regB[1][0] , regA[3][2] , regB[2][0] , regA[3][3] , regB[3][0], wireOutMaxMin[3][0]); 
maxMin2 maxmin31(regA[3][0] , regB[0][1] , regA[3][1] , regB[1][1] , regA[3][2] , regB[2][1] , regA[3][3] , regB[3][1], wireOutMaxMin[3][1]); 
maxMin2 maxmin32(regA[3][0] , regB[0][2] , regA[3][1] , regB[1][2] , regA[3][2] , regB[2][2] , regA[3][3] , regB[3][2], wireOutMaxMin[3][2]); 
maxMin2 maxmin33(regA[3][0] , regB[0][3] , regA[3][1] , regB[1][3] , regA[3][2] , regB[2][3] , regA[3][3] , regB[3][3], wireOutMaxMin[3][3]);

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

//OP 2 wires
logic [3:0][3:0][W-1:0] wireOut_2_Mul_00;
logic [3:0][3:0][W-1:0] wireOut_2_Mul_01;
logic [3:0][3:0][W-1:0] wireOut_2_Mul_10;
logic [3:0][3:0][W-1:0] wireOut_2_Mul_11;

logic [3:0][3:0][W-1:0] wireOut_2_Plus_00;
logic [3:0][3:0][W-1:0] wireOut_2_Plus_01;
logic [3:0][3:0][W-1:0] wireOut_2_Plus_10;
logic [3:0][3:0][W-1:0] wireOut_2_Plus_11;


//OP 1 wires
logic [3:0][3:0][W-1:0] wireOut_1_Plus;
logic [3:0][3:0][W-1:0] wireOut_1_Min;
logic [3:0][3:0][W-1:0] wireOut_1_Max;

// OP 2 Logic

// connect output wire with reg out
assign out = regOut;

generate  
for (i  = 0; i < 4; i++) begin
    for (j = 0; j < 4; j++) begin
        // op2 = mul
        assign wireOut_2_Mul_00[i][j] = regA[i][0] * regB[0][j];
        assign wireOut_2_Mul_01[i][j] = regA[i][1] * regB[1][j];
        assign wireOut_2_Mul_10[i][j] = regA[i][2] * regB[2][j];
        assign wireOut_2_Mul_11[i][j] = regA[i][3] * regB[3][j];

        // op2 = plus
        assign wireOut_2_Plus_00[i][j] = regA[i][0] + regB[0][j];
        assign wireOut_2_Plus_01[i][j] = regA[i][1] + regB[1][j];
        assign wireOut_2_Plus_10[i][j] = regA[i][2] + regB[2][j];
        assign wireOut_2_Plus_11[i][j] = regA[i][3] + regB[3][j];
    
        // op1 = plus
        assign wireOut_1_Plus[i][j] = regTemp00[i][j] + regTemp01[i][j] + regTemp10[i][j] + regTemp11[i][j];
        // op1 = max
        max maxmod(regTemp00[i][j] , regTemp01[i][j] ,  regTemp10[i][j] , regTemp11[i][j], regA[i][j], wireOut_1_Max[i][j]);
        // op1 = min
        min minmod(regTemp00[i][j] , regTemp01[i][j] ,  regTemp10[i][j] , regTemp11[i][j], regA[i][j], wireOut_1_Min[i][j]);


        always@(posedge clk) begin
            regA[i][j] <= inA[i][j];
            regB[i][j] <= inB[i][j];
            
            if(op[2:0]==3'b000) begin
                regTemp00[i][j] <= wireOut_2_Mul_00[i][j];
                regTemp01[i][j] <= wireOut_2_Mul_01[i][j];
                regTemp10[i][j] <= wireOut_2_Mul_10[i][j];
                regTemp11[i][j] <= wireOut_2_Mul_11[i][j];
            end
            else if(op[2:0]== 3'b001) begin
                regTemp00[i][j] <= wireOut_2_Plus_00[i][j];
                regTemp01[i][j] <= wireOut_2_Plus_01[i][j];
                regTemp10[i][j] <= wireOut_2_Plus_10[i][j];
                regTemp11[i][j] <= wireOut_2_Plus_11[i][j];
            end
        end
        always@(negedge clk) begin
            if (op == 5'b00010) begin // l2d
                regOut[i][j] <= wireOutL2D[i][j];
            end
            else if (op == 5'b01011) begin // minmax
                regOut[i][j] <= wireOutMinMax[i][j];
            end
            else if (op == 5'b10100) begin // maxmin
                regOut[i][j] <= wireOutMaxMin[i][j];
            end
            else if (op == 5'b11101) begin // orand
                regOut[i][j] <= wireOutOrAnd[i][j];
            end
            else if(op[4:3]==2'b00) begin
                regOut[i][j] <= wireOut_1_Plus[i][j];
            end
            else if(op[4:3]==2'b01) begin
                regOut[i][j] <= wireOut_1_Min[i][j];
            end
            else if(op[4:3]==2'b10) begin
                regOut[i][j] <= wireOut_1_Max[i][j];
            end
        end
    end
end
endgenerate
endmodule