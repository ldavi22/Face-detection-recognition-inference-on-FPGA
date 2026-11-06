module requant (
    input  wire               clk,
    input  wire        [31:0] acc,
    input  wire signed [31:0] mult,
    input  wire        [ 5:0] shift,
    input  wire               leaky_en,
    output reg  signed [ 7:0] out
);

  localparam signed [39:0] LEAKY_MUL = 40'sd13;
  localparam LEAKY_SHIFT = 7;

  wire signed [31:0] acc_s = $signed(acc);

  wire signed [39:0] leaky_prod = acc_s * LEAKY_MUL;
  wire signed [39:0] leaky_sh = leaky_prod >>> LEAKY_SHIFT;

  reg signed [31:0] s1_a;
  reg signed [31:0] s1_mult;
  reg        [ 5:0] s1_shift;

  always @(posedge clk) begin
    s1_a     <= (leaky_en && acc_s[31]) ? leaky_sh[31:0] : acc_s;
    s1_mult  <= mult;
    s1_shift <= shift;
  end

  reg signed [63:0] s2_prod;
  reg        [ 5:0] s2_shift;

  always @(posedge clk) begin
    s2_prod  <= s1_a * s1_mult;
    s2_shift <= s1_shift;
  end

  wire signed [63:0] round_c = (s2_shift == 6'd0)
                             ? 64'sd0
                             : (64'sd1 <<< (s2_shift - 6'd1));

  wire signed [63:0] prod_r = (s2_prod[63] == 1'b0) ? (s2_prod + round_c)
                                                    : (s2_prod - round_c);

  reg signed [63:0] s3_sh;

  always @(posedge clk) begin
    s3_sh <= prod_r >>> s2_shift;
  end

  always @(posedge clk) begin
    out <= (s3_sh >  64'sd127) ?  8'sd127
         : (s3_sh < -64'sd127) ? -8'sd127
                               :  s3_sh[7:0];
  end

endmodule
