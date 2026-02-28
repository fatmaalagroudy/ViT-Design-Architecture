`timescale 1ns / 1ps

module simple_mmu #(
    parameter int N = 64,
    parameter int D = 256,
    parameter int M = 768,
    parameter string VERSION = "last_version"
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

    logic [15:0] i, j, k;
    logic signed [63:0] acc;
    typedef enum logic [1:0] {IDLE, COMPUTE, DONE_STATE} state_t;
    state_t state;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            i <= 0; j <= 0; k <= 0;
            acc <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= COMPUTE;
                        i <= 0; j <= 0; k <= 0;
                        acc <= 0;
                        if (i == 0 && j == 0) $display("[%0t] MMU STARTING - VERSION: %s", $time, VERSION);
                    end
                end
                
                COMPUTE: begin
                    if (i < N) begin
                        if (j < M) begin
                            if (k < D) begin
                                automatic logic signed [63:0] local_sum;
                                local_sum = 0;
                                for (int ki=0; ki<16; ki++) begin
                                    if (k + ki < D) begin
                                        if (A[i][k+ki] === 8'hx || B[j][k+ki] === 8'hx) begin
                                            $display("[%0t] CRITICAL ERROR: 'x' detected in MMU inputs at i=%0d, j=%0d, k=%0d", $time, i, j, k+ki);
                                        end
                                        local_sum = local_sum + ($signed(A[i][k+ki]) * $signed(B[j][k+ki]));
                                    end
                                end
                                acc <= acc + local_sum;
                                k <= k + 16;
                            end else begin
                                C[i][j] <= acc + bias[j];
                                if (i == 0 && j == 0) begin
                                    $display("[%0t] MMU Trace: C[0][0] = %d (acc=%d, bias=%d)", $time, acc + bias[j], acc, bias[j]);
                                end
                                acc <= 0;
                                k <= 0;
                                j <= j + 1;
                            end
                        end else begin
                            j <= 0;
                            i <= i + 1;
                            if (i % 8 == 0) $display("[%0t] MMU Progress: %0d/%0d rows done (Head/Patch processing)", $time, i, N);
                        end
                    end else begin
                        state <= DONE_STATE;
                    end
                end
                
                DONE_STATE: begin
                    done <= 1;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule
