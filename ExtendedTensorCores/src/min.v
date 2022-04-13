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

module min0#(parameter W = 16) 
           (input [W-1:0] a, 
            input [W-1:0] b, 
            input [W-1:0] c,
            output reg [W-1:0] out);
wire [W-1:0] ab;

assign ab = a<=b ? a : b;
assign out = ab<=c ? ab : c;

endmodule

module min2#(parameter W = 16) 
           (input [W-1:0] a, 
            input [W-1:0] b, 
            input [W-1:0] c, 
            input [W-1:0] d,
            input [W-1:0] e, 
            input [W-1:0] f, 
            input [W-1:0] g, 
            input [W-1:0] h, 
            input [W-1:0] i,
            output reg [W-1:0] out);
wire [W-1:0] ab;
wire [W-1:0] cd;
wire [W-1:0] ef;
wire [W-1:0] gh;
wire [W-1:0] abcd;
wire [W-1:0] efgh;
wire [W-1:0] abcdi;

assign ab = a<=b ? a : b;
assign cd = c<=d ? c : d;
assign ef = e<=f ? e : f;
assign gh = g<=h ? g : h;
assign abcd = ab<=cd ? ab : cd;
assign efgh = ef<=gh ? ef : gh;
assign abcdi = abcd<=i ? abcd : i;
assign out = abcdi<=efgh ? abcdi : efgh;

endmodule

module min3#(parameter W = 16) 
           (input [W-1:0] a, 
            input [W-1:0] b, 
            input [W-1:0] c, 
            input [W-1:0] d,
            input [W-1:0] e, 
            input [W-1:0] f, 
            input [W-1:0] g, 
            input [W-1:0] h, 
            input [W-1:0] i, 
            input [W-1:0] j, 
            input [W-1:0] k, 
            input [W-1:0] l,
            input [W-1:0] m, 
            input [W-1:0] n, 
            input [W-1:0] o, 
            input [W-1:0] p,
            input [W-1:0] r,
            output reg [W-1:0] out);
wire [W-1:0] ab;
wire [W-1:0] cd;
wire [W-1:0] ef;
wire [W-1:0] gh;
wire [W-1:0] abcd;
wire [W-1:0] efgh;

wire [W-1:0] ij;
wire [W-1:0] kl;
wire [W-1:0] mn;
wire [W-1:0] op;
wire [W-1:0] ijkl;
wire [W-1:0] mnop;

wire [W-1:0] all;

assign ab = a<=b ? a : b;
assign cd = c<=d ? c : d;
assign ef = e<=f ? e : f;
assign gh = g<=h ? g : h;
assign abcd = ab<=cd ? ab : cd;
assign efgh = ef<=gh ? ef : gh;

assign ij = i<=j ? i : j;
assign kl = k<=l ? k : l;
assign mn = m<=n ? m : n;
assign op = o<=p ? o : p;
assign ijkl = ij<=kl ? ij : kl;
assign mnop = mn<=op ? mn : op;

assign all = mnop<=efgh ? mnop : efgh;

assign out = all<=r ? all : r;

endmodule