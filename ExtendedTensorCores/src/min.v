module min#(parameter W = 16) 
           (input [W-1:0] a, 
            input [W-1:0] b, 
            input [W-1:0] c, 
            input [W-1:0] d,
            input [W-1:0] e, 
            output reg [W-1:0] out);
wire [W-1:0] ab;
wire [W-1:0] cd;
wire [W-1:0] abe;

assign ab = a<=b ? a : b;
assign cd = c<=d ? c : d;
assign abe = ab<=e ? ab : e;
assign out = abe<=cd ? abe : cd;

//assign out = a;
endmodule