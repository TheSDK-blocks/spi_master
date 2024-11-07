module spi_slave( // @[:@12.2]
  input        clock, // @[:@13.4]
  input        reset, // @[:@14.4]
  input        io_mosi, // @[:@15.4]
  input        io_cs, // @[:@15.4]
  output       io_miso, // @[:@15.4]
  input        io_sclk, // @[:@15.4]
  output [3:0] io_config_out, // @[:@15.4]
  input  [3:0] io_monitor_in // @[:@15.4]
);
  wire  async_set_register_set; // @[spi_slave.scala 36:31:@26.4]
  wire  async_set_register_clock; // @[spi_slave.scala 36:31:@26.4]
  wire  async_set_register_Q; // @[spi_slave.scala 36:31:@26.4]
  wire  async_set_register_D; // @[spi_slave.scala 36:31:@26.4]
  wire  _T_17; // @[spi_slave.scala 22:30:@17.4]
  wire  _T_20; // @[spi_slave.scala 22:21:@19.4]
  wire  inv_sclk; // @[spi_slave.scala 22:52:@20.4]
  reg [4:0] shiftingConfig; // @[spi_slave.scala 27:50:@21.4]
  reg [31:0] _RAND_0;
  reg [4:0] stateConfig; // @[spi_slave.scala 28:26:@22.4]
  reg [31:0] _RAND_1;
  reg [3:0] shiftingMonitor; // @[spi_slave.scala 29:51:@23.4]
  reg [31:0] _RAND_2;
  reg  misoPosEdgeBuffer; // @[spi_slave.scala 30:52:@24.4]
  reg [31:0] _RAND_3;
  wire [4:0] configMask; // @[Cat.scala 30:58:@36.4]
  wire  _T_29; // @[spi_slave.scala 44:41:@37.4]
  wire [3:0] monitorMask; // @[Cat.scala 30:58:@38.4]
  wire [3:0] _T_31; // @[spi_slave.scala 49:59:@39.4]
  wire [4:0] _GEN_5; // @[spi_slave.scala 49:41:@40.4]
  wire [4:0] nextShiftingConfig; // @[spi_slave.scala 49:41:@40.4]
  wire [2:0] _T_32; // @[spi_slave.scala 50:60:@41.4]
  wire [3:0] _GEN_6; // @[spi_slave.scala 50:41:@42.4]
  wire [3:0] monitorRegShifted; // @[spi_slave.scala 50:41:@42.4]
  wire  _T_35; // @[spi_slave.scala 53:29:@44.4]
  wire  spiFirstCycle; // @[spi_slave.scala 35:29:@25.4 spi_slave.scala 38:19:@32.4]
  wire  monitorMuxControl; // @[spi_slave.scala 53:43:@46.4]
  wire [3:0] nextShiftingMonitor; // @[spi_slave.scala 54:34:@47.4]
  wire  _T_40; // @[spi_slave.scala 61:47:@53.6]
  reg  _T_43; // @[spi_slave.scala 19:44:@59.6]
  reg [31:0] _RAND_4;
  wire  _T_45; // @[spi_slave.scala 19:36:@61.6]
  wire  _T_46; // @[spi_slave.scala 19:33:@62.6]
  wire [4:0] _GEN_0; // @[spi_slave.scala 65:40:@63.6]
  async_set_register #(.n(1)) async_set_register ( // @[spi_slave.scala 36:31:@26.4]
    .set(async_set_register_set),
    .clock(async_set_register_clock),
    .Q(async_set_register_Q),
    .D(async_set_register_D)
  );
  assign _T_17 = $unsigned(io_sclk); // @[spi_slave.scala 22:30:@17.4]
  assign _T_20 = _T_17 == 1'h0; // @[spi_slave.scala 22:21:@19.4]
  assign inv_sclk = _T_20; // @[spi_slave.scala 22:52:@20.4]
  assign configMask = {io_mosi,4'h0}; // @[Cat.scala 30:58:@36.4]
  assign _T_29 = shiftingConfig[0]; // @[spi_slave.scala 44:41:@37.4]
  assign monitorMask = {_T_29,3'h0}; // @[Cat.scala 30:58:@38.4]
  assign _T_31 = shiftingConfig[4:1]; // @[spi_slave.scala 49:59:@39.4]
  assign _GEN_5 = {{1'd0}, _T_31}; // @[spi_slave.scala 49:41:@40.4]
  assign nextShiftingConfig = configMask | _GEN_5; // @[spi_slave.scala 49:41:@40.4]
  assign _T_32 = shiftingMonitor[3:1]; // @[spi_slave.scala 50:60:@41.4]
  assign _GEN_6 = {{1'd0}, _T_32}; // @[spi_slave.scala 50:41:@42.4]
  assign monitorRegShifted = monitorMask | _GEN_6; // @[spi_slave.scala 50:41:@42.4]
  assign _T_35 = io_cs == 1'h0; // @[spi_slave.scala 53:29:@44.4]
  assign spiFirstCycle = async_set_register_Q; // @[spi_slave.scala 35:29:@25.4 spi_slave.scala 38:19:@32.4]
  assign monitorMuxControl = _T_35 & spiFirstCycle; // @[spi_slave.scala 53:43:@46.4]
  assign nextShiftingMonitor = monitorMuxControl ? io_monitor_in : monitorRegShifted; // @[spi_slave.scala 54:34:@47.4]
  assign _T_40 = nextShiftingMonitor[0]; // @[spi_slave.scala 61:47:@53.6]
  assign _T_45 = _T_43 == 1'h0; // @[spi_slave.scala 19:36:@61.6]
  assign _T_46 = io_cs & _T_45; // @[spi_slave.scala 19:33:@62.6]
  assign _GEN_0 = _T_46 ? shiftingConfig : stateConfig; // @[spi_slave.scala 65:40:@63.6]
  assign io_miso = _T_35 ? misoPosEdgeBuffer : 1'h0; // @[spi_slave.scala 62:15:@55.6 spi_slave.scala 68:16:@66.6]
  assign io_config_out = stateConfig[4:1]; // @[spi_slave.scala 72:19:@69.4]
  assign async_set_register_set = io_cs; // @[spi_slave.scala 40:23:@35.4]
  assign async_set_register_clock = _T_20; // @[spi_slave.scala 39:25:@33.4]
  assign async_set_register_D = 1'h0; // @[spi_slave.scala 37:21:@31.4]
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
`ifdef RANDOMIZE
  integer initvar;
  initial begin
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      #0.002 begin end
    `endif
  `ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  shiftingConfig = _RAND_0[4:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{`RANDOM}};
  stateConfig = _RAND_1[4:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {1{`RANDOM}};
  shiftingMonitor = _RAND_2[3:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{`RANDOM}};
  misoPosEdgeBuffer = _RAND_3[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_4 = {1{`RANDOM}};
  _T_43 = _RAND_4[0:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge inv_sclk) begin
    if (_T_35) begin
      shiftingConfig <= nextShiftingConfig;
    end
    if (monitorMuxControl) begin
      shiftingMonitor <= io_monitor_in;
    end else begin
      shiftingMonitor <= monitorRegShifted;
    end
  end
  always @(posedge clock) begin
    if (!(_T_35)) begin
      if (_T_46) begin
        stateConfig <= shiftingConfig;
      end
    end
    _T_43 <= io_cs;
  end
  always @(posedge io_sclk) begin
    if (_T_35) begin
      misoPosEdgeBuffer <= _T_40;
    end
  end
endmodule
