module maxMin#(parameter W = 16) 
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
        assign a =  a1>=a2 ? a1 : a2;
        assign b =  b1>=b2 ? b1 : b2;
        assign c =  c1>=c2 ? c1 : c2;
        assign d =  d1>=d2 ? d1 : d2;
        assign ab = a <= b ? a : b;
        assign cd = c <= d? c : d;
        assign abe = ab <=e ? ab : e;
        assign out = abe <=cd ? abe : cd;
//assign out = a;
endmodule

module maxMin0#(parameter W = 16) 
           (input [W-1:0] a1, 
            input [W-1:0] a2, 
            input [W-1:0] b1, 
            input [W-1:0] b2,
            output reg [W-1:0] out);
wire [W-1:0] a;
wire [W-1:0] b;
        assign a =  a1<=a2 ? a1 : a2;
        assign b =  b1<=b2 ? b1 : b2;
        assign out = a>=b ? a : b;
//assign out = a;
endmodule

module maxMin2#(parameter W = 16) 
           (input [W-1:0] a1, 
            input [W-1:0] a2, 
            input [W-1:0] b1, 
            input [W-1:0] b2,
            input [W-1:0] c1, 
            input [W-1:0] c2, 
            input [W-1:0] d1, 
            input [W-1:0] d2, 
            output reg [W-1:0] out);
wire [W-1:0] ab;
wire [W-1:0] cd;
wire [W-1:0] a;
wire [W-1:0] b;
wire [W-1:0] c;
wire [W-1:0] d;
        assign a =  a1>=a2 ? a1 : a2;
        assign b =  b1>=b2 ? b1 : b2;
        assign c =  c1>=c2 ? c1 : c2;
        assign d =  d1>=d2 ? d1 : d2;
        assign ab = a <= b ? a : b;
        assign cd = c <= d ? c : d;
        assign out = ab <=cd ? ab : cd;
//assign out = a;
endmodule

module maxMin3#(parameter W = 16) 
           (input [W-1:0] a1, 
            input [W-1:0] a2, 
            input [W-1:0] b1, 
            input [W-1:0] b2,
            input [W-1:0] c1, 
            input [W-1:0] c2, 
            input [W-1:0] d1, 
            input [W-1:0] d2, 
            input [W-1:0] e1, 
            input [W-1:0] e2, 
            input [W-1:0] f1, 
            input [W-1:0] f2,
            input [W-1:0] g1, 
            input [W-1:0] g2, 
            input [W-1:0] h1, 
            input [W-1:0] h2, 
            output reg [W-1:0] out);
wire [W-1:0] ab;
wire [W-1:0] cd;
wire [W-1:0] ef;
wire [W-1:0] gh;
wire [W-1:0] abcd;
wire [W-1:0] efgh;
wire [W-1:0] a;
wire [W-1:0] b;
wire [W-1:0] c;
wire [W-1:0] d;
wire [W-1:0] e;
wire [W-1:0] f;
wire [W-1:0] g;
wire [W-1:0] h;
        assign a =  a1>=a2 ? a1 : a2;
        assign b =  b1>=b2 ? b1 : b2;
        assign c =  c1>=c2 ? c1 : c2;
        assign d =  d1>=d2 ? d1 : d2;
        assign e =  e1>=e2 ? e1 : e2;
        assign f =  f1>=f2 ? f1 : f2;
        assign g =  g1>=g2 ? g1 : g2;
        assign h =  h1>=h2 ? h1 : h2;
        assign ab = a <= b ? a : b;
        assign cd = c <= d ? c : d;
        assign ef = e <= f ? e : f;
        assign gh = g <= h ? g : h;
        assign abcd = ab <= cd ? ab : cd;
        assign efgh = ef <= gh ? ef : gh;
        assign out = abcd <=efgh ? abcd : efgh;
//assign out = a;
endmodule


module maxMin4#(parameter W = 16) 
           (input [W-1:0] a1, 
            input [W-1:0] a2, 
            input [W-1:0] b1, 
            input [W-1:0] b2,
            input [W-1:0] c1, 
            input [W-1:0] c2, 
            input [W-1:0] d1, 
            input [W-1:0] d2, 
            input [W-1:0] e1, 
            input [W-1:0] e2, 
            input [W-1:0] f1, 
            input [W-1:0] f2,
            input [W-1:0] g1, 
            input [W-1:0] g2, 
            input [W-1:0] h1, 
            input [W-1:0] h2,
            input [W-1:0] i1, 
            input [W-1:0] j1, 
            input [W-1:0] k1, 
            input [W-1:0] l1,
            input [W-1:0] m1, 
            input [W-1:0] n1, 
            input [W-1:0] o1, 
            input [W-1:0] p1,
            input [W-1:0] i2, 
            input [W-1:0] j2, 
            input [W-1:0] k2, 
            input [W-1:0] l2,
            input [W-1:0] m2, 
            input [W-1:0] n2, 
            input [W-1:0] o2, 
            input [W-1:0] p2,
            output reg [W-1:0] out);
wire [W-1:0] ab;
wire [W-1:0] cd;
wire [W-1:0] ef;
wire [W-1:0] gh;
wire [W-1:0] abcd;
wire [W-1:0] efgh;
wire [W-1:0] a;
wire [W-1:0] b;
wire [W-1:0] c;
wire [W-1:0] d;
wire [W-1:0] e;
wire [W-1:0] f;
wire [W-1:0] g;
wire [W-1:0] h;

wire [W-1:0] ij;
wire [W-1:0] kl;
wire [W-1:0] mn;
wire [W-1:0] op;
wire [W-1:0] ijkl;
wire [W-1:0] mnop;
wire [W-1:0] i; 
wire [W-1:0] j; 
wire [W-1:0] k; 
wire [W-1:0] l;
wire [W-1:0] m; 
wire [W-1:0] n; 
wire [W-1:0] o; 
wire [W-1:0] p;
wire [W-1:0] all1;
wire [W-1:0] all2;

        assign a =  a1>=a2 ? a1 : a2;
        assign b =  b1>=b2 ? b1 : b2;
        assign c =  c1>=c2 ? c1 : c2;
        assign d =  d1>=d2 ? d1 : d2;
        assign e =  e1>=e2 ? e1 : e2;
        assign f =  f1>=f2 ? f1 : f2;
        assign g =  g1>=g2 ? g1 : g2;
        assign h =  h1>=h2 ? h1 : h2;
        assign ab = a <= b ? a : b;
        assign cd = c <= d ? c : d;
        assign ef = e <= f ? e : f;
        assign gh = g <= h ? g : h;
        assign abcd = ab <= cd ? ab : cd;
        assign efgh = ef <= gh ? ef : gh;

        assign i =  i1>=i2 ? i1 : i2;
        assign j =  j1>=j2 ? j1 : j2;
        assign k =  k1>=k2 ? k1 : k2;
        assign l =  l1>=l2 ? l1 : l2;
        assign m =  m1>=m2 ? m1 : m2;
        assign n =  n1>=n2 ? n1 : n2;
        assign o =  o1>=o2 ? o1 : o2;
        assign p =  p1>=p2 ? p1 : p2;
        assign ij = i<=j ? i : j;
        assign kl = k<=l ? k : l;
        assign mn = m<=n ? m : n;
        assign op = o<=p ? o : p;
        assign ijkl = ij<=kl ? ij : kl;
        assign mnop = mn<=op ? mn : op;


        assign all1 = abcd <=efgh ? abcd : efgh;
        assign all2 = ijkl <=mnop ? ijkl : mnop;
        assign out = all1 <=all2 ? all1 : all2;
//assign out = a;
endmodule
