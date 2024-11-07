module spi_master(
  input        clock,
  input        reset,
  input        io_cpol,
  input        io_cpha,
  output       io_mosi,
  input        io_miso,
  output       io_sclk,
  output       io_cs,
  input  [7:0] io_slaveData,
  input        io_slaveDataValid,
  output [7:0] io_masterData
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
  reg [31:0] _RAND_3;
  reg [31:0] _RAND_4;
  reg [31:0] _RAND_5;
  reg [31:0] _RAND_6;
`endif // RANDOMIZE_REG_INIT
  reg [1:0] state; // @[spi_master.scala 27:22]
  reg [2:0] bitIndex; // @[spi_master.scala 32:25]
  reg [7:0] regout; // @[spi_master.scala 33:23]
  reg [7:0] regin; // @[spi_master.scala 34:22]
  reg [1:0] count; // @[spi_master.scala 37:22]
  reg  cpolReg; // @[spi_master.scala 39:24]
  reg  cphaReg; // @[spi_master.scala 40:24]
  wire [1:0] _count_T_1 = 2'h2 - 2'h1; // @[spi_master.scala 54:24]
  wire  _T_2 = count > 2'h0; // @[spi_master.scala 66:18]
  wire [1:0] _count_T_3 = count - 2'h1; // @[spi_master.scala 67:24]
  wire [1:0] _GEN_5 = count > 2'h0 ? _count_T_3 : _count_T_1; // @[spi_master.scala 66:24 67:15 69:15]
  wire [7:0] _regin_T_1 = {regin[6:0],io_miso}; // @[Cat.scala 33:92]
  wire [1:0] _GEN_8 = _T_2 ? state : 2'h3; // @[spi_master.scala 27:22 75:24 79:15]
  wire [7:0] _GEN_9 = _T_2 ? regin : _regin_T_1; // @[spi_master.scala 34:22 75:24 80:15]
  wire [2:0] _bitIndex_T_1 = bitIndex - 3'h1; // @[spi_master.scala 90:32]
  wire [1:0] _GEN_10 = bitIndex > 3'h0 ? _count_T_1 : count; // @[spi_master.scala 88:29 89:17 37:22]
  wire [2:0] _GEN_11 = bitIndex > 3'h0 ? _bitIndex_T_1 : bitIndex; // @[spi_master.scala 88:29 90:20 32:25]
  wire [1:0] _GEN_12 = bitIndex > 3'h0 ? 2'h2 : 2'h0; // @[spi_master.scala 88:29 91:17 93:17]
  wire [1:0] _GEN_13 = _T_2 ? _count_T_3 : _GEN_10; // @[spi_master.scala 85:24 86:15]
  wire [2:0] _GEN_14 = _T_2 ? bitIndex : _GEN_11; // @[spi_master.scala 85:24 32:25]
  wire [1:0] _GEN_15 = _T_2 ? state : _GEN_12; // @[spi_master.scala 27:22 85:24]
  wire [1:0] _GEN_16 = 2'h3 == state ? _GEN_13 : count; // @[spi_master.scala 46:16 37:22]
  wire [2:0] _GEN_17 = 2'h3 == state ? _GEN_14 : bitIndex; // @[spi_master.scala 46:16 32:25]
  wire [1:0] _GEN_18 = 2'h3 == state ? _GEN_15 : state; // @[spi_master.scala 46:16 27:22]
  wire  _io_sclk_T = ~io_cs; // @[spi_master.scala 110:18]
  wire  _io_sclk_T_5 = state == 2'h2 ? cphaReg ^ cpolReg : ~(cphaReg ^ cpolReg); // @[spi_master.scala 110:29]
  wire [7:0] _io_mosi_T_3 = regout >> bitIndex; // @[spi_master.scala 119:50]
  assign io_mosi = _io_sclk_T & state != 2'h0 & _io_mosi_T_3[0]; // @[spi_master.scala 119:17]
  assign io_sclk = ~io_cs ? _io_sclk_T_5 : cpolReg; // @[spi_master.scala 110:116]
  assign io_cs = state == 2'h0; // @[spi_master.scala 43:18]
  assign io_masterData = regin; // @[spi_master.scala 100:17]
  always @(posedge clock) begin
    if (reset) begin // @[spi_master.scala 27:22]
      state <= 2'h0; // @[spi_master.scala 27:22]
    end else if (2'h0 == state) begin // @[spi_master.scala 46:16]
      if (io_slaveDataValid) begin // @[spi_master.scala 51:30]
        if (io_cpha) begin // @[spi_master.scala 56:22]
          state <= 2'h1; // @[spi_master.scala 57:17]
        end else begin
          state <= 2'h2; // @[spi_master.scala 59:17]
        end
      end
    end else if (2'h1 == state) begin // @[spi_master.scala 46:16]
      if (!(count > 2'h0)) begin // @[spi_master.scala 66:24]
        state <= 2'h2; // @[spi_master.scala 70:15]
      end
    end else if (2'h2 == state) begin // @[spi_master.scala 46:16]
      state <= _GEN_8;
    end else begin
      state <= _GEN_18;
    end
    if (reset) begin // @[spi_master.scala 32:25]
      bitIndex <= 3'h0; // @[spi_master.scala 32:25]
    end else if (2'h0 == state) begin // @[spi_master.scala 46:16]
      if (io_slaveDataValid) begin // @[spi_master.scala 51:30]
        bitIndex <= 3'h7; // @[spi_master.scala 53:18]
      end
    end else if (!(2'h1 == state)) begin // @[spi_master.scala 46:16]
      if (!(2'h2 == state)) begin // @[spi_master.scala 46:16]
        bitIndex <= _GEN_17;
      end
    end
    if (reset) begin // @[spi_master.scala 33:23]
      regout <= 8'h0; // @[spi_master.scala 33:23]
    end else if (2'h0 == state) begin // @[spi_master.scala 46:16]
      if (io_slaveDataValid) begin // @[spi_master.scala 51:30]
        regout <= io_slaveData; // @[spi_master.scala 52:16]
      end
    end
    if (reset) begin // @[spi_master.scala 34:22]
      regin <= 8'h0; // @[spi_master.scala 34:22]
    end else if (2'h0 == state) begin // @[spi_master.scala 46:16]
      regin <= 8'h0; // @[spi_master.scala 62:13]
    end else if (!(2'h1 == state)) begin // @[spi_master.scala 46:16]
      if (2'h2 == state) begin // @[spi_master.scala 46:16]
        regin <= _GEN_9;
      end
    end
    if (reset) begin // @[spi_master.scala 37:22]
      count <= 2'h0; // @[spi_master.scala 37:22]
    end else if (2'h0 == state) begin // @[spi_master.scala 46:16]
      if (io_slaveDataValid) begin // @[spi_master.scala 51:30]
        count <= _count_T_1; // @[spi_master.scala 54:15]
      end
    end else if (2'h1 == state) begin // @[spi_master.scala 46:16]
      count <= _GEN_5;
    end else if (2'h2 == state) begin // @[spi_master.scala 46:16]
      count <= _GEN_5;
    end else begin
      count <= _GEN_16;
    end
    if (reset) begin // @[spi_master.scala 39:24]
      cpolReg <= 1'h0; // @[spi_master.scala 39:24]
    end else if (2'h0 == state) begin // @[spi_master.scala 46:16]
      cpolReg <= io_cpol; // @[spi_master.scala 48:15]
    end
    if (reset) begin // @[spi_master.scala 40:24]
      cphaReg <= 1'h0; // @[spi_master.scala 40:24]
    end else if (2'h0 == state) begin // @[spi_master.scala 46:16]
      cphaReg <= io_cpha; // @[spi_master.scala 49:15]
    end
  end
// Register and memory initialization
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  state = _RAND_0[1:0];
  _RAND_1 = {1{`RANDOM}};
  bitIndex = _RAND_1[2:0];
  _RAND_2 = {1{`RANDOM}};
  regout = _RAND_2[7:0];
  _RAND_3 = {1{`RANDOM}};
  regin = _RAND_3[7:0];
  _RAND_4 = {1{`RANDOM}};
  count = _RAND_4[1:0];
  _RAND_5 = {1{`RANDOM}};
  cpolReg = _RAND_5[0:0];
  _RAND_6 = {1{`RANDOM}};
  cphaReg = _RAND_6[0:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
