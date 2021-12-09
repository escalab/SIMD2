module minMax#(parameter W = 16) 
           (input [W-1:0] a1, 
            input [W-1:0] a2, 
            input [W-1:0] b1, 
            input [W-1:0] b2,
            input [W-1:0] c1, 
            input [W-1:0] c2, 
            input [W-1:0] d1, 
            input [W-1:0] d2, 
            input [W-1:0] e,
            output reg [W-1:0] out);
wire [W-1:0] ab;
wire [W-1:0] cd;
wire [W-1:0] a;
wire [W-1:0] b;
wire [W-1:0] c;
wire [W-1:0] d;
reg [W-1:0] abe;
        assign a =  a1<=a2 ? a1 : a2;
        assign b =  b1<=b2 ? b1 : b2;
        assign c =  c1<=c2 ? c1 : c2;
        assign d =  d1<=d2 ? d1 : d2;
        assign ab = a >= b ? a : b;
        assign cd = c >= d? c : d;
        assign abe = ab<=e ? ab : e;
        assign out = abe<=cd ? abe : cd;
//assign out = a;
endmodule