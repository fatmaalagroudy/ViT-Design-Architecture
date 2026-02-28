// High-Performance Parallel MMU (simple_mmu.sv)
// Operates on 16 columns in parallel for 16x speedup.
`timescale 1ns / 1ps

module simple_mmu #(
    parameter int N = 64,
    parameter int D = 256,
    parameter int M = 768
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [7:0]  A [N][D],
    input  logic signed [7:0]  B [M][D],
    input  logic signed [63:0] bias [M],
    output logic signed [63:0] C [N][M],
    output logic done
);

    typedef enum logic [1:0] {IDLE, COMPUTE, DONE_STATE} state_t;
    state_t state;
    int i, j, k;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= COMPUTE;
                        i <= 0; j <= 0; k <= 0;
                        for (int r=0; r<N; r++) for (int c=0; c<M; c++) C[r][c] <= bias[c];
                    end
                end
                
                COMPUTE: begin
                    if (i < N) begin
                        if (j < M) begin
                            if (k < D) begin
                                // Process 16 dot-product elements in parallel
                                automatic logic signed [63:0] local_sum = 0;
                                for (int ki=0; ki<16; ki++) begin
                                    if (k + ki < D) begin
                                        local_sum = local_sum + ($signed(64'($signed(A[i][k+ki]))) * $signed(64'($signed(B[j][k+ki]))));
                                    end
                                end
                                C[i][j] <= C[i][j] + local_sum;
                                k <= k + 16;
                            end else begin
                                k <= 0;
                                j <= j + 1;
                            end
                        end else begin
                            j <= 0;
                            i <= i + 1;
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
