// High-Performance Pipelined GELU (gelu_pipelined.sv)
`timescale 1ns / 1ps

module gelu_pipelined #(
    parameter int N = 64,
    parameter int D = 768, // MLP dimension
    parameter string GELU_LUT = "C:/Users/user/Downloads/vit_new/export_quantized_new/gelu_lut.mem"
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [63:0] x_in [N][D],
    input  logic signed [31:0] m_idx,
    input  logic signed [31:0] s_idx,
    output logic signed [31:0] x_out [N][D],
    output logic done
);

    typedef enum logic [1:0] {IDLE, COMPUTE, DONE_STATE} state_t;
    state_t state;
    logic [15:0] i_cnt, d_cnt;
    
    logic signed [31:0] gelu_lut [512];
    initial begin
        for (int i=0; i<512; i++) gelu_lut[i] = 32'h0;
        $readmemh(GELU_LUT, gelu_lut);
        if (gelu_lut[256] === 32'hx) $display("WARNING: GELU LUT loading failed");
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            i_cnt <= 0;
            d_cnt <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= COMPUTE;
                        i_cnt <= 0;
                        d_cnt <= 0;
                    end
                end
                
                COMPUTE: begin
                    if (i_cnt < N) begin
                        if (d_cnt < D) begin
                            automatic logic signed [127:0] val128 = $signed(x_in[i_cnt][d_cnt]);
                            automatic int idx_offset = int'( (val128 * $signed(128'(m_idx))) >>> s_idx );
                            automatic int idx = 256 + idx_offset;
                            if (idx < 0) idx = 0; if (idx > 511) idx = 511;
                            
                            // Output high-precision LUT value directly
                            x_out[i_cnt][d_cnt] <= gelu_lut[idx];
                            
                            d_cnt <= d_cnt + 1;
                        end else begin
                            d_cnt <= 0;
                            i_cnt <= i_cnt + 1;
                        end
                    end else begin
                        state <= DONE_STATE;
                    end
                end
                
                DONE_STATE: begin
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
