module conv1x1 #(
    parameter C_IN  = 256,
    parameter C_OUT = 128,
    parameter POS   = 169,
    parameter LEAKY = 1
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [17:0] avs_address,
    input  wire        avs_write,
    input  wire [31:0] avs_writedata,
    input  wire        avs_read,
    output reg  [31:0] avs_readdata
);

  localparam CHUNKS    = C_IN / 4;
  localparam ACT_WORDS = POS * CHUNKS;
  localparam WGT_WORDS = C_OUT * CHUNKS;
  localparam PRM_WORDS = C_OUT * 3;
  localparam OUT_BYTES = C_OUT * POS;

  localparam MAC_LAT = 3;
  localparam RQ_LAT  = 4;
  localparam LAST    = CHUNKS + MAC_LAT + RQ_LAT;

  localparam REG_CTRL = 3'd0;
  localparam REG_ACT  = 3'd1;
  localparam REG_WGT  = 3'd2;
  localparam REG_PRM  = 3'd3;
  localparam REG_OUT  = 3'd4;

  wire [2:0]  region = avs_address[17:15];
  wire [14:0] offset = avs_address[14:0];

  reg start, busy, done;
  reg [31:0] cycles;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      start <= 1'b0;
    else
      start <= (avs_write && region == REG_CTRL && offset == 15'd0 &&
                avs_writedata[0]);
  end

  localparam S_IDLE = 3'd0;
  localparam S_P0   = 3'd1;
  localparam S_P1   = 3'd2;
  localparam S_P2   = 3'd3;
  localparam S_P3   = 3'd4;
  localparam S_P4   = 3'd5;
  localparam S_RUN  = 3'd6;

  reg [2:0]  state;
  reg [15:0] f;
  reg [15:0] p;
  reg [15:0] cnt;

  reg [15:0] p_base;
  reg [15:0] f_base;
  reg [15:0] o_base;
  reg [15:0] prm_addr;

  reg signed [31:0] bias_r;
  reg signed [31:0] mult_r;
  reg        [ 5:0] shift_r;

  wire mac_clear = (state == S_RUN) && (cnt == 16'd0);
  wire mac_en    = (state == S_RUN) && (cnt >= 16'd1) &&
                                       (cnt <= CHUNKS[15:0]);
  wire do_write  = (state == S_RUN) && (cnt == LAST[15:0]);

  wire [15:0] k_rd = (cnt < CHUNKS[15:0]) ? cnt : (CHUNKS[15:0] - 16'd1);
  wire [15:0] act_addr = p_base + k_rd;
  wire [15:0] wgt_addr = f_base + k_rd;
  wire [15:0] out_addr = o_base + p;

  reg [31:0] act_ram[0:ACT_WORDS-1];
  reg [31:0] wgt_ram[0:WGT_WORDS-1];
  reg [31:0] prm_ram[0:PRM_WORDS-1];
  reg [ 7:0] out_ram[0:OUT_BYTES-1];

  reg [31:0] act_q, wgt_q, prm_q;
  reg [ 7:0] out_q;

  wire signed [31:0] acc;
  wire signed [ 7:0] q_out;

  always @(posedge clk) begin
    if (avs_write && region == REG_ACT)
      act_ram[offset] <= avs_writedata;
    act_q <= act_ram[act_addr];
  end

  always @(posedge clk) begin
    if (avs_write && region == REG_WGT)
      wgt_ram[offset] <= avs_writedata;
    wgt_q <= wgt_ram[wgt_addr];
  end

  always @(posedge clk) begin
    if (avs_write && region == REG_PRM)
      prm_ram[offset] <= avs_writedata;
    prm_q <= prm_ram[prm_addr];
  end

  always @(posedge clk) begin
    if (do_write)
      out_ram[out_addr] <= q_out;
    out_q <= out_ram[offset];
  end

  reg [2:0]  region_r;
  reg [14:0] offset_r;

  always @(posedge clk) begin
    region_r <= region;
    offset_r <= offset;
  end

  always @* begin
    case (region_r)
      REG_CTRL:
        avs_readdata = (offset_r == 15'd1) ? {30'd0, busy, done}
                     : (offset_r == 15'd2) ? cycles
                                           : 32'd0;
      REG_OUT: avs_readdata = {24'd0, out_q};
      default: avs_readdata = 32'd0;
    endcase
  end

  mac4 u_mac (
      .clk  (clk),
      .rst_n(rst_n),
      .clear(mac_clear),
      .en   (mac_en),
      .init (bias_r),
      .act  (act_q),
      .wgt  (wgt_q),
      .acc  (acc)
  );

  requant u_rq (
      .clk     (clk),
      .acc     (acc),
      .mult    (mult_r),
      .shift   (shift_r),
      .leaky_en(LEAKY[0]),
      .out     (q_out)
  );

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    <= S_IDLE;
      busy     <= 1'b0;
      done     <= 1'b0;
      f        <= 16'd0;
      p        <= 16'd0;
      cnt      <= 16'd0;
      p_base   <= 16'd0;
      f_base   <= 16'd0;
      o_base   <= 16'd0;
      prm_addr <= 16'd0;
      cycles   <= 32'd0;
      bias_r   <= 32'sd0;
      mult_r   <= 32'sd0;
      shift_r  <= 6'd0;
    end else begin

      if (busy) cycles <= cycles + 32'd1;

      case (state)

        S_IDLE: begin
          if (start) begin
            busy     <= 1'b1;
            done     <= 1'b0;
            cycles   <= 32'd0;
            f        <= 16'd0;
            p        <= 16'd0;
            cnt      <= 16'd0;
            p_base   <= 16'd0;
            f_base   <= 16'd0;
            o_base   <= 16'd0;
            prm_addr <= 16'd0;
            state    <= S_P0;
          end
        end

        S_P0: state <= S_P1;

        S_P1: begin
          prm_addr <= prm_addr + 16'd1;
          state    <= S_P2;
        end

        S_P2: begin
          bias_r   <= prm_q;
          prm_addr <= prm_addr + 16'd1;
          state    <= S_P3;
        end

        S_P3: begin
          mult_r <= prm_q;
          state  <= S_P4;
        end

        S_P4: begin
          shift_r  <= prm_q[5:0];
          prm_addr <= prm_addr + 16'd1;
          cnt      <= 16'd0;
          state    <= S_RUN;
        end

        S_RUN: begin
          if (cnt == LAST[15:0]) begin
            cnt <= 16'd0;
            if (p == POS[15:0] - 16'd1) begin
              p      <= 16'd0;
              p_base <= 16'd0;
              if (f == C_OUT[15:0] - 16'd1) begin
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= S_IDLE;
              end else begin
                f      <= f + 16'd1;
                f_base <= f_base + CHUNKS[15:0];
                o_base <= o_base + POS[15:0];
                state  <= S_P0;
              end
            end else begin
              p      <= p + 16'd1;
              p_base <= p_base + CHUNKS[15:0];
            end
          end else begin
            cnt <= cnt + 16'd1;
          end
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule
