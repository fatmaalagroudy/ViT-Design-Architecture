module LayerNorm #(
    parameter int EMBED_DIM = 384
)(
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [15:0]  vec_in,
    input  logic         vec_in_v,
    input  logic [15:0]  gamma,
    input  logic [15:0]  beta,
    output logic [15:0]  vec_out,
    output logic         vec_out_v,
    output logic         busy,
    output logic         done
);
  
    // Parameters / Local Types
    typedef enum logic [1:0] {
        IDLE,
        PASS1_MEAN,
        PASS2_VAR,
        PASS3_NORM
    } state_t;

    state_t state, next_state;

    logic [$clog2(EMBED_DIM):0] count;
    logic [31:0] sum_accum;     // FP32 placeholder
    logic [31:0] mean_val;
    logic [31:0] var_accum;
    logic [31:0] var_val;
    logic [31:0] inv_std;

 
    logic [15:0] buffer [0:EMBED_DIM-1];



    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // FSM: Next State Logic
    always_comb begin
     
        next_state = state;
        busy = 1'b0;
        done = 1'b0;

        case (state)
            IDLE: begin
                if (start)
                    next_state = PASS1_MEAN;
            end

            PASS1_MEAN: begin
                busy = 1'b1;
                if (count == EMBED_DIM)
                    next_state = PASS2_VAR;
            end

            PASS2_VAR: begin
                busy = 1'b1;
                if (count == EMBED_DIM)
                    next_state = PASS3_NORM;
            end

            PASS3_NORM: begin
                busy = 1'b1;
                if (count == EMBED_DIM)
                    next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase

        if (state == PASS3_NORM && count == EMBED_DIM)
            done = 1'b1;
    end

 
    // Datapath / Control Logic
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count      <= '0;
            sum_accum  <= 32'd0;
            var_accum  <= 32'd0;
            mean_val   <= 32'd0;
            var_val    <= 32'd0;
            inv_std    <= 32'd0;
            vec_out_v  <= 1'b0;
        end
        else begin
            vec_out_v <= 1'b0;

            case (state)
               
                IDLE: begin
                    count <= 0;
                    sum_accum <= 0;
                    var_accum <= 0;
                end

                PASS1_MEAN: begin
                    busy <= 1'b1;
                    if (vec_in_v) begin
                        buffer[count] <= vec_in;
                        // convert FP16->FP32 and accumulate
                        sum_accum <= sum_accum + $bitstoshortreal(vec_in);
                        count <= count + 1;
                    end
                    if (count == EMBED_DIM) begin
                        mean_val <= sum_accum / EMBED_DIM;
                        count <= 0;
                    end
                end

               
                PASS2_VAR: begin
                    busy <= 1'b1;
                    // Compute variance
                    var_accum <= var_accum + (($bitstoshortreal(buffer[count]) - mean_val) *
                                              ($bitstoshortreal(buffer[count]) - mean_val));
                    count <= count + 1;
                    if (count == EMBED_DIM) begin
                        var_val <= var_accum / EMBED_DIM;
                        inv_std <= 1.0 / ($sqrt(var_val + 1e-5));  // epsilon
                        count <= 0;
                    end
                end

                PASS3_NORM: begin
                    busy <= 1'b1;
                    // Normalize and apply gamma/beta
                    vec_out_v <= 1'b1;
                    vec_out <= $shortrealtobits(
                        (($bitstoshortreal(buffer[count]) - mean_val) * inv_std) *
                        $bitstoshortreal(gamma) + $bitstoshortreal(beta)
                    );
                    count <= count + 1;
                end
            endcase
        end
    end

endmodule
