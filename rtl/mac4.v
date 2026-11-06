module mac4 (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clear,
    input  wire               en,
    input  wire signed [31:0] init,
    input  wire        [31:0] act,
    input  wire        [31:0] wgt,
    output reg  signed [31:0] acc
);

  wire signed [7:0] a0 = act[7:0];
  wire signed [7:0] a1 = act[15:8];
  wire signed [7:0] a2 = act[23:16];
  wire signed [7:0] a3 = act[31:24];

  wire signed [7:0] w0 = wgt[7:0];
  wire signed [7:0] w1 = wgt[15:8];
  wire signed [7:0] w2 = wgt[23:16];
  wire signed [7:0] w3 = wgt[31:24];

  reg signed [15:0] p0, p1, p2, p3;
  reg               s1_en, s1_clear;
  reg signed [31:0] s1_init;

  always @(posedge clk) begin
    p0 <= a0 * w0;
    p1 <= a1 * w1;
    p2 <= a2 * w2;
    p3 <= a3 * w3;
    s1_en    <= en;
    s1_clear <= clear;
    s1_init  <= init;
  end

  reg signed [31:0] sum_r;
  reg               s2_en, s2_clear;
  reg signed [31:0] s2_init;

  always @(posedge clk) begin
    sum_r <= {{16{p0[15]}}, p0} + {{16{p1[15]}}, p1} +
             {{16{p2[15]}}, p2} + {{16{p3[15]}}, p3};
    s2_en    <= s1_en;
    s2_clear <= s1_clear;
    s2_init  <= s1_init;
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      acc <= 32'sd0;
    else if (s2_clear)
      acc <= s2_init;
    else if (s2_en)
      acc <= acc + sum_r;
  end

endmodule
