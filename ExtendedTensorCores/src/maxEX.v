module maxEX#(parameter W = 16) 
           (input [W-1:0] a1, 
            input [W-1:0] a2, 
            input [W-1:0] b1, 
            input [W-1:0] b2,
            input [W-1:0] c1, 
            input [W-1:0] c2, 
            input [W-1:0] d1, 
            input [W-1:0] d2, 
            input [W-1:0] e,
            output reg [W-1:0] out, input func);
wire [W-1:0] ab;
wire [W-1:0] cd;
wire [W-1:0] abp;
wire [W-1:0] cdp;
wire [W-1:0] abm;
wire [W-1:0] cdm;
reg [W-1:0] abe;
        assign abp = (a1+a2)>=(b1+b2) ? (a1+a2) : (b1+b2);
        assign cdp = (c1+c2)>=(d1+d2) ? (c1+c2) : (d1+d2);
//        assign abe = ab<=e ? ab : e;
//        assign out = abe<=cd ? abe : cd;

        assign abm = (a1*a2)>=(b1*b2) ? (a1*a2) : (b1*b2);
        assign cdm = (c1*c2)>=(d1*d2) ? (c1*c2) : (d1*d2);
        assign ab = func ? abm : abp;
        assign cd = func ? cdm : cdp;
        assign abe = ab>=e ? ab : e;
        assign out = abe>=cd ? abe : cd;
//assign out = a;
endmodule