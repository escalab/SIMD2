`timescale 1ns / 1ps
/*
 * Extended Tensor Cores version 3
 */

module etcEX3#(parameter W = 16)
(
input clk,
input [4:0] op,
input [3:0][3:0][W-1:0] inA, inB,
output [3:0][3:0][W-1:0] out 
);

reg [3:0][3:0][W-1:0] regA, regB;
reg [3:0][3:0][W-1:0] regOut;

reg [3:0][3:0][W-1:0] regTemp00;
reg [3:0][3:0][W-1:0] regTemp01;
reg [3:0][3:0][W-1:0] regTemp10;
reg [3:0][3:0][W-1:0] regTemp11;

genvar i,j;
//OP 2 wires
wire [3:0][3:0][W-1:0] wireOut_2_Mul_00;
wire [3:0][3:0][W-1:0] wireOut_2_Mul_01;
wire [3:0][3:0][W-1:0] wireOut_2_Mul_10;
wire [3:0][3:0][W-1:0] wireOut_2_Mul_11;

wire [3:0][3:0][W-1:0] wireOut_2_Plus_00;
wire [3:0][3:0][W-1:0] wireOut_2_Plus_01;
wire [3:0][3:0][W-1:0] wireOut_2_Plus_10;
wire [3:0][3:0][W-1:0] wireOut_2_Plus_11;

wire [3:0][3:0][W-1:0] wireOut_2_Max_00;
wire [3:0][3:0][W-1:0] wireOut_2_Max_01;
wire [3:0][3:0][W-1:0] wireOut_2_Max_10;
wire [3:0][3:0][W-1:0] wireOut_2_Max_11;

wire [3:0][3:0][W-1:0] wireOut_2_Min_00;
wire [3:0][3:0][W-1:0] wireOut_2_Min_01;
wire [3:0][3:0][W-1:0] wireOut_2_Min_10;
wire [3:0][3:0][W-1:0] wireOut_2_Min_11;

wire [3:0][3:0][W-1:0] wireOut_2_L2_00;
wire [3:0][3:0][W-1:0] wireOut_2_L2_01;
wire [3:0][3:0][W-1:0] wireOut_2_L2_10;
wire [3:0][3:0][W-1:0] wireOut_2_L2_11;

wire [3:0][3:0][W-1:0] wireOut_2_And_00;
wire [3:0][3:0][W-1:0] wireOut_2_And_01;
wire [3:0][3:0][W-1:0] wireOut_2_And_10;
wire [3:0][3:0][W-1:0] wireOut_2_And_11;

//OP 1 wires
wire [3:0][3:0][W-1:0] wireOut_1_Plus;
wire [3:0][3:0][W-1:0] wireOut_1_Min;
wire [3:0][3:0][W-1:0] wireOut_1_Max;
wire [3:0][3:0][W-1:0] wireOut_1_Or;

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

        // op2 = max
        assign wireOut_2_Max_00[i][j] = regA[i][0] >= regB[0][j] ? regA[i][0] : regB[0][j];
        assign wireOut_2_Max_01[i][j] = regA[i][1] >= regB[1][j] ? regA[i][1] : regB[1][j];
        assign wireOut_2_Max_10[i][j] = regA[i][2] >= regB[2][j] ? regA[i][2] : regB[2][j];
        assign wireOut_2_Max_11[i][j] = regA[i][3] >= regB[3][j] ? regA[i][3] : regB[3][j];

        // op2 = min
        assign wireOut_2_Min_00[i][j] = regA[i][0] <= regB[0][j] ? regA[i][0] : regB[0][j];
        assign wireOut_2_Min_01[i][j] = regA[i][0] <= regB[0][j] ? regA[i][0] : regB[0][j];
        assign wireOut_2_Min_10[i][j] = regA[i][0] <= regB[0][j] ? regA[i][0] : regB[0][j];
        assign wireOut_2_Min_11[i][j] = regA[i][0] <= regB[0][j] ? regA[i][0] : regB[0][j];

        // op2 = l2
        assign wireOut_2_L2_00[i][j] = (regA[i][0] - regB[0][j]) * (regA[i][0] - regB[0][j]);
        assign wireOut_2_L2_01[i][j] = (regA[i][1] - regB[1][j]) * (regA[i][1] - regB[1][j]);
        assign wireOut_2_L2_10[i][j] = (regA[i][2] - regB[2][j]) * (regA[i][2] - regB[2][j]);
        assign wireOut_2_L2_11[i][j] = (regA[i][3] - regB[3][j]) * (regA[i][3] - regB[3][j]);

        // op2 = and
        assign wireOut_2_And_00[i][j] = regA[i][0] & regB[0][j];
        assign wireOut_2_And_01[i][j] = regA[i][1] & regB[1][j];
        assign wireOut_2_And_10[i][j] = regA[i][2] & regB[2][j];
        assign wireOut_2_And_11[i][j] = regA[i][3] & regB[3][j];

        // op1 = plus
        assign wireOut_1_Plus[i][j] = regTemp00[i][j] + regTemp01[i][j] + regTemp10[i][j] + regTemp11[i][j];
        // op1 = or
        assign wireOut_1_Or[i][j] = regTemp00[i][j] | regTemp01[i][j] | regTemp10[i][j] | regTemp11[i][j] | regB[i][j];

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
            else if(op[2:0]== 3'b010) begin
                regTemp00[i][j] <= wireOut_2_L2_00[i][j];
                regTemp01[i][j] <= wireOut_2_L2_01[i][j];
                regTemp10[i][j] <= wireOut_2_L2_10[i][j];
                regTemp11[i][j] <= wireOut_2_L2_11[i][j];
            end
            else if(op[2:0]== 3'b001) begin
                 regTemp00[i][j] <= wireOut_2_Plus_00[i][j];
                regTemp01[i][j] <= wireOut_2_Plus_01[i][j];
                regTemp10[i][j] <= wireOut_2_Plus_10[i][j];
                regTemp11[i][j] <= wireOut_2_Plus_11[i][j];
            end
            else if(op[2:0]== 3'b011) begin
                regTemp00[i][j] <= wireOut_2_Max_00[i][j];
                regTemp01[i][j] <= wireOut_2_Max_01[i][j];
                regTemp10[i][j] <= wireOut_2_Max_10[i][j];
                regTemp11[i][j] <= wireOut_2_Max_11[i][j];
            end
            else if(op[2:0]== 3'b100) begin
                regTemp00[i][j] <= wireOut_2_Min_00[i][j];
                regTemp01[i][j] <= wireOut_2_Min_01[i][j];
                regTemp10[i][j] <= wireOut_2_Min_10[i][j];
                regTemp11[i][j] <= wireOut_2_Min_11[i][j];
            end
            else begin
                regTemp00[i][j] <= wireOut_2_And_00[i][j];
                regTemp01[i][j] <= wireOut_2_And_01[i][j];
                regTemp10[i][j] <= wireOut_2_And_10[i][j];
                regTemp11[i][j] <= wireOut_2_And_11[i][j];
            end
        end
        always@(negedge clk) begin
            if(op[4:3]==2'b00) begin
                regOut[i][j] <= wireOut_1_Plus[i][j];
            end
            else if(op[4:3]==2'b01) begin
                regOut[i][j] <= wireOut_1_Min[i][j];
            end
            else if(op[4:3]==2'b10) begin
                regOut[i][j] <= wireOut_1_Max[i][j];
            end
            else begin
                regOut[i][j] <= wireOut_1_Or[i][j];
            end
        end
    end
end
endgenerate
endmodule