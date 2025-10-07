// =============================================================
// File: pe_array.v
// Module: PE Array (Systolic Tile Engine)
// Purpose: Performs micro-GEMM on tiles; MAC engines arranged in systolic grid
// =============================================================
module pe_array #(
  parameter DATAWIDTH = 8,     // INT8 data
  parameter ACCWIDTH  = 32,    // INT32 accumulator
  parameter PE_ROWS   = 16,    // number of rows (tile height)
  parameter PE_COLS   = 16     // number of columns (tile width)
)(
  // ----------------------------
  // Interface / Control
  // ----------------------------
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire                     start,       // start operation
  input  wire [1:0]               mode,        // 00=LOAD, 01=MATMUL, 10=DRAIN
  input  wire [7:0]               tile_dim,    // tile dimension (TS)
  input  wire                     valid_in,

  // ----------------------------
  // Data Interfaces
  // ----------------------------
  input  wire [PE_COLS*DATAWIDTH-1:0] A_in,        // streamed activations (per column)
  input  wire [PE_ROWS*DATAWIDTH-1:0] B_in,        // streamed weights (per row)
  input  wire [PE_ROWS*DATAWIDTH-1:0] weight_load, // initial weight load bus

  // ----------------------------
  // Outputs
  // ----------------------------
  output wire [PE_ROWS*ACCWIDTH-1:0]  C_out,       // accumulated results (per row)
  output wire                         done         // computation complete
);

  // =============================================================
  // Internal Systolic Connections
  // =============================================================
  wire [DATAWIDTH-1:0] a_wire [0:PE_ROWS-1][0:PE_COLS-1];
  wire [DATAWIDTH-1:0] b_wire [0:PE_ROWS-1][0:PE_COLS-1];
  wire                 valid_wire [0:PE_ROWS-1][0:PE_COLS-1];
  wire [ACCWIDTH-1:0]  c_wire [0:PE_ROWS-1][0:PE_COLS-1];

  // =============================================================
  // PE Grid Instantiation
  // =============================================================
  genvar i, j;
  generate
    for (i = 0; i < PE_ROWS; i = i + 1) begin: ROW
      for (j = 0; j < PE_COLS; j = j + 1) begin: COL
        // Input selection for each PE
        wire [DATAWIDTH-1:0] a_in_elem = (j == 0) ? A_in[i*DATAWIDTH +: DATAWIDTH]
                                                  : a_wire[i][j-1];
        wire [DATAWIDTH-1:0] b_in_elem = (i == 0) ? B_in[j*DATAWIDTH +: DATAWIDTH]
                                                  : b_wire[i-1][j];

        pe #(
          .DATAWIDTH(DATAWIDTH),
          .ACCWIDTH(ACCWIDTH)
        ) u_pe (
          .clk(clk),
          .rst_n(rst_n),
          .mode(mode),
          .valid_in((i==0 && j==0) ? valid_in :
                    (i==0) ? valid_wire[i][j-1] :
                    (j==0) ? valid_wire[i-1][j] :
                              valid_wire[i-1][j-1]),
          .a_in(a_in_elem),
          .b_in(b_in_elem),
          .valid_out(valid_wire[i][j]),
          .a_out(a_wire[i][j]),
          .b_out(b_wire[i][j]),
          .c_out(c_wire[i][j])
        );

        // Flatten each PE’s C output into the row output bus
        assign C_out[(i*ACCWIDTH) +: ACCWIDTH] = c_wire[i][j];
      end
    end
  endgenerate

  // =============================================================
  // Done flag (bottom-right PE drains)
  // =============================================================
  assign done = (mode == 2'b10) && valid_wire[PE_ROWS-1][PE_COLS-1];

endmodule


// =============================================================
// File: pe.v
// Description: Processing Element (Weight-Stationary, INT8 MAC)
// =============================================================
module pe #(
  parameter DATAWIDTH = 8,   // INT8
  parameter ACCWIDTH  = 32   // INT32
)(
  input  wire clk,
  input  wire rst_n,

  input  wire [1:0] mode,      // 00=LOAD, 01=MATMUL, 10=DRAIN
  input  wire       valid_in,
  input  wire [DATAWIDTH-1:0] a_in,  // activation stream
  input  wire [DATAWIDTH-1:0] b_in,  // weight or dummy for matmul

  output reg  valid_out,
  output reg  [DATAWIDTH-1:0] a_out,
  output reg  [DATAWIDTH-1:0] b_out,
  output reg  [ACCWIDTH-1:0]  c_out
);

  // Local registers
  reg [DATAWIDTH-1:0] weight_reg;
  reg [ACCWIDTH-1:0]  acc_reg;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      weight_reg <= 0;
      acc_reg    <= 0;
      c_out      <= 0;
      a_out      <= 0;
      b_out      <= 0;
      valid_out  <= 0;
    end else begin
      valid_out <= valid_in;
      a_out     <= a_in;
      b_out     <= b_in;

      case (mode)
        // -----------------------------
        // LOAD WEIGHTS
        // -----------------------------
        2'b00: begin
          if (valid_in) begin
            weight_reg <= b_in; // store local weight
            acc_reg    <= 0;    // clear accumulator
          end
        end

        // -----------------------------
        // MATRIX MULTIPLY
        // -----------------------------
        2'b01: begin
          if (valid_in)
            acc_reg <= acc_reg + $signed(a_in) * $signed(weight_reg);
        end

        // -----------------------------
        // DRAIN (output accumulated sum)
        // -----------------------------
        2'b10: begin
          c_out <= acc_reg;
        end

        default: ;
      endcase
    end
  end
endmodule
