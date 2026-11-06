/* Testbench for the 1x1 accelerator.
 *
 * Loads activations, weights and parameters from hex files produced by
 * gen_vectors, runs the engine, and compares every output byte against
 * the expected file. The comparison must be EXACT -- both sides are
 * doing the same integer arithmetic, so any difference is a bug, not
 * rounding.
 *
 *   iverilog -g2012 -o tb tb/tb_conv1x1.v rtl/conv1x1.v rtl/mac4.v rtl/requant.v
 *   vvp tb
 */

`timescale 1ns / 1ps

module tb_conv1x1;

  localparam C_IN = 256;
  localparam C_OUT = 128;
  localparam POS = 169;

  localparam CHUNKS = C_IN / 4;
  localparam ACT_WORDS = POS * CHUNKS;
  localparam WGT_WORDS = C_OUT * CHUNKS;
  localparam PRM_WORDS = C_OUT * 3;
  localparam OUT_BYTES = C_OUT * POS;

  reg clk = 0;
  reg rst_n = 0;

  reg [17:0] addr;
  reg write;
  reg read;
  reg [31:0] wdata;
  wire [31:0] rdata;

  always #5 clk = ~clk; /* 100 MHz */

  conv1x1 #(
      .C_IN (C_IN),
      .C_OUT(C_OUT),
      .POS  (POS),
      .LEAKY(1)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .avs_address(addr),
      .avs_write(write),
      .avs_writedata(wdata),
      .avs_read(read),
      .avs_readdata(rdata)
  );

  reg [31:0] act_mem[0:ACT_WORDS-1];
  reg [31:0] wgt_mem[0:WGT_WORDS-1];
  reg [31:0] prm_mem[0:PRM_WORDS-1];
  reg [ 7:0] exp_mem[0:OUT_BYTES-1];

  integer i, errors, first_err;
  reg [31:0] got_word;
  reg [7:0] got, want;

  task bus_write(input [17:0] a, input [31:0] d);
    begin
      @(posedge clk);
      addr  <= a;
      wdata <= d;
      write <= 1'b1;
      @(posedge clk);
      write <= 1'b0;
    end
  endtask

  task bus_read(input [17:0] a);
    begin
      @(posedge clk);
      addr <= a;
      read <= 1'b1;
      @(posedge clk);
      read <= 1'b0;
      @(posedge clk);
      got_word = rdata;
    end
  endtask

  initial begin
    write = 0;
    read = 0;
    addr = 0;
    wdata = 0;
    errors = 0;
    first_err = -1;

    $readmemh("vectors/act.hex", act_mem);
    $readmemh("vectors/wgt.hex", wgt_mem);
    $readmemh("vectors/prm.hex", prm_mem);
    $readmemh("vectors/exp.hex", exp_mem);

    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (4) @(posedge clk);

    $display("loading %0d activation words...", ACT_WORDS);
    for (i = 0; i < ACT_WORDS; i = i + 1)
      bus_write({3'd1, i[14:0]}, act_mem[i]);

    $display("loading %0d weight words...", WGT_WORDS);
    for (i = 0; i < WGT_WORDS; i = i + 1)
      bus_write({3'd2, i[14:0]}, wgt_mem[i]);

    $display("loading %0d param words...", PRM_WORDS);
    for (i = 0; i < PRM_WORDS; i = i + 1)
      bus_write({3'd3, i[14:0]}, prm_mem[i]);

    $display("starting...");
    bus_write({3'd0, 15'd0}, 32'd1);

    /* poll STATUS until done */
    got_word = 0;
    while (got_word[0] !== 1'b1) begin
      bus_read({3'd0, 15'd1});
      if ($time > 200_000_000) begin
        $display("TIMEOUT");
        $finish;
      end
    end

    bus_read({3'd0, 15'd2});
    $display("done in %0d cycles", got_word);

    $display("checking %0d output bytes...", OUT_BYTES);
    for (i = 0; i < OUT_BYTES; i = i + 1) begin
      bus_read({3'd4, i[14:0]});
      got  = got_word[7:0];
      want = exp_mem[i];

      if (got !== want) begin
        errors = errors + 1;
        if (first_err < 0) first_err = i;
        if (errors <= 10)
          $display("  MISMATCH at %0d (f=%0d p=%0d): got %0d want %0d",
                   i, i / POS, i % POS, $signed(got), $signed(want));
      end
    end

    if (errors == 0)
      $display("PASS -- all %0d bytes match", OUT_BYTES);
    else
      $display("FAIL -- %0d of %0d bytes differ, first at %0d", errors,
               OUT_BYTES, first_err);

    $finish;
  end

endmodule
