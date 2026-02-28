// High-Performance Pipelined Softmax (softmax_pipelined.sv)
`timescale 1ns / 1ps

module softmax_pipelined #(
    parameter int N = 64,
    parameter string EXP_LUT = "C:/Users/user/Downloads/vit_new/export_quantized_new/exp_lut.mem",
    parameter string INV_SQRT_LUT = "C:/Users/user/Downloads/vit_new/export_quantized_new/inv_sqrt_lut.mem"
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [63:0] x_in [N],
    input  logic signed [31:0] m_idx,
    input  logic signed [31:0] s_idx,
    output logic signed [7:0] x_out [N],
    output logic done
);

    typedef enum logic [2:0] {IDLE, MAX, EXP_SUM, RECIP, NORM_STATE, DONE_STATE} state_t;
    state_t state;
    logic [15:0] cnt;
    
    logic signed [63:0] max_val;
    logic signed [63:0] exp_val [N];
    logic signed [63:0] sum_exp;
    logic signed [63:0] recip_m;
    int recip_s_signed;

    // Persisted calculation registers
    logic [6:0] blen;
    logic signed [6:0] shift_val;
    logic [63:0] norm_v;
    int lut_idx;
    logic signed [63:0] is_m;
    logic [6:0] is_s;

    logic [31:0] exp_lut [1024];
    logic [31:0] inv_sqrt_lut [1024];
    initial begin
        for (int i=0; i<1024; i++) begin
            exp_lut[i] = 32'h0;
            inv_sqrt_lut[i] = 32'h0;
        end
        $readmemh(EXP_LUT, exp_lut);
        $readmemh(INV_SQRT_LUT, inv_sqrt_lut);
        if (exp_lut[0] === 32'hx) $display("WARNING: Softmax EXP LUT loading failed");
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            cnt <= 0;
            sum_exp <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= MAX;
                        cnt <= 0;
                        max_val <= -64'h7FFFFFFFFFFFFFFF;
                    end
                end
                
                MAX: begin
                    if (cnt < N) begin
                        if (x_in[cnt] > max_val) max_val <= x_in[cnt];
                        cnt <= cnt + 1;
                    end else begin
                        state <= EXP_SUM;
                        cnt <= 0;
                        sum_exp <= 0;
                    end
                end
                
                EXP_SUM: begin
                    if (cnt < N) begin
                        automatic logic signed [63:0] shifted = x_in[cnt] - max_val;
                        // Use explicit 128-bit product for safety and match Python parity
                        automatic logic signed [127:0] prod128 = $signed(shifted) * $signed(m_idx);
                        automatic int idx_offset = int'(prod128 >>> s_idx);
                        automatic int idx = 1023 + idx_offset;
                        
                        if (idx < 0) idx = 0; if (idx > 1023) idx = 1023;
                        exp_val[cnt] <= 64'(exp_lut[idx]);
                        sum_exp <= sum_exp + 64'(exp_lut[idx]);
                        cnt <= cnt + 1;
                    end else begin
                        state <= RECIP;
                        cnt <= 0;
                    end
                end
                
                RECIP: begin
                    // Python-equivalent hw_reciprocal (using blen-based shift)
                    automatic logic [63:0] safe_var = (sum_exp <= 0) ? 64'd1 : sum_exp;

                    blen = 0;
                    for (int b=63; b>=0; b--) if (safe_var[b]) begin blen = b + 1; break; end
                    
                    if (blen % 2 == 0) shift_val = blen - 30;
                    else               shift_val = blen - 31;
                    
                    if (shift_val > 0) norm_v = safe_var >> shift_val;
                    else               norm_v = safe_var << (-shift_val);
                    
                    lut_idx = ($signed(128'(norm_v)) - $signed(128'd536870912)) * $signed(128'd1023) / $signed(128'd1610612736);
                    if (lut_idx < 0) lut_idx = 0; if (lut_idx > 1023) lut_idx = 1023;
                    
                    is_m = 64'(inv_sqrt_lut[lut_idx]);
                    is_s = 45 + (shift_val >>> 1);
                    
                    if (sum_exp <= 0) begin is_m = 64'd1073741824; is_s = 0; end
                    
                    // hw_reciprocal: m_sq = (m*m)>>30, s_out = 2*s - 30
                    recip_m = ($signed(is_m) * $signed(is_m)) >>> 30;
                    recip_s_signed = (2 * int'(is_s)) - 30;
                    
                    state <= NORM_STATE;
                    cnt <= 0;
                    $display("[SOFTMAX TRACE] SumExp=%d, blen=%d, shift=%d, l_idx=%d, is_m=%d, is_s=%d", 
                             sum_exp, blen, shift_val, lut_idx, is_m, is_s);
                end
                
                NORM_STATE: begin
                    if (cnt < N) begin
                        // CRITICAL FIX: Explicitly sign-extend to 128 bits
                        automatic logic signed [127:0] s_exp = $signed(exp_val[cnt]);
                        automatic logic signed [127:0] prod128 = s_exp * 128'd127;
                        
                        automatic logic signed [127:0] s_recip = $signed(recip_m);
                        automatic logic signed [127:0] final_prod128 = prod128 * s_recip;
                        
                        automatic logic signed [127:0] res;
                        
                        if (recip_s_signed >= 0) res = final_prod128 >>> recip_s_signed;
                        else                    res = final_prod128 <<< (-recip_s_signed);
                        
                        if ($signed(res) > $signed(128'sd127)) x_out[cnt] <= 8'sd127;
                        else if ($signed(res) < $signed(-128'sd128)) x_out[cnt] <= -8'sd128;
                        else x_out[cnt] <= res[7:0];
                        
                        cnt <= cnt + 1;
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
