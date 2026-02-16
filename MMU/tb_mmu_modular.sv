`timescale 1ns/1ps

module tb_mmu_modular;

    // =========================================================================
    // Parameters and signals
    // =========================================================================
    logic clk;
    logic rst_n;
    logic start;
    logic [1:0] mode;
    logic [15:0] K_dim;
    logic done;
    logic busy;
    
    // Input data (assumed ready each cycle)
    logic signed [7:0] matrix_a_row [0:63];
    logic signed [7:0] matrix_b_col [0:63];
    logic signed [7:0] matrix_a_row_b [0:63];
    logic signed [7:0] matrix_b_col_b [0:63];
    
    // Results
    logic signed [23:0] result_c [0:63][0:63];
    logic signed [23:0] result_c_b [0:63][0:63];
    logic result_valid;
    logic result_valid_b;
    
    // Test data storage
    logic signed [7:0] test_A [0:63][0:63];
    logic signed [7:0] test_B [0:63][0:63];
    logic signed [7:0] test_A_second [0:63][0:63];
    logic signed [7:0] test_B_second [0:63][0:63];
    
    // Golden reference
    logic signed [31:0] golden_C [0:63][0:63];
    logic signed [31:0] golden_C_second [0:63][0:63];
    
    integer i, j, k;
    integer ii, jj, kk;  // Separate for task loops
    integer errors, errors_second;
    logic signed [31:0] sum;
    
    // Mode parameters
    localparam MODE_SINGLE_64x64 = 2'd0;
    localparam MODE_SINGLE_64x32 = 2'd1;
    localparam MODE_DUAL_64x32   = 2'd2;
    
    // =========================================================================
    // Clock generation
    // =========================================================================
    initial clk = 0;
    always #5 clk = ~clk;
    
    // =========================================================================
    // DUT instantiation
    // =========================================================================
    mmu_modular_64x64 dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .mode(mode),
        .K_dim(K_dim),
        .done(done),
        .busy(busy),
        .matrix_a_row(matrix_a_row),
        .matrix_b_col(matrix_b_col),
        .matrix_a_row_b(matrix_a_row_b),
        .matrix_b_col_b(matrix_b_col_b),
        .result_c(result_c),
        .result_c_b(result_c_b),
        .result_valid(result_valid),
        .result_valid_b(result_valid_b)
    );
    
    // =========================================================================
    // Feed counter to track which row/column to provide
    // =========================================================================
    logic [15:0] feed_counter;
    
    // Use combinational assignment to avoid 1-cycle lag
    assign feed_counter = dut.feed_cycle;
    
    // =========================================================================
    // Data feeding logic with K-tiling support
    // =========================================================================
    // KEY INSIGHT: Each 32x32 array has LOCAL row indices 0-31
    // - Top arrays (00, 01): Present rows 0-31 of A using feed pattern for rows 0-31
    // - Bottom arrays (10, 11): Present rows 32-63 of A using feed pattern for rows 0-31
    //
    // The MMU routing already selects the right 32 elements for each array
    // We just need to present ALL 64 elements organized correctly
    
    logic [15:0] k_offset;
    logic [15:0] k_index;  // Wrapped K-index to stay within array bounds
    
    assign k_offset = dut.k_tile_32 * 32;
    
    always_comb begin
        for (i = 0; i < 64; i = i + 1) begin
            matrix_a_row[i] = 8'sd0;
            matrix_b_col[i] = 8'sd0;
            matrix_a_row_b[i] = 8'sd0;
            matrix_b_col_b[i] = 8'sd0;
        end
        
        if (dut.state == dut.FEED) begin
            case (mode)
                MODE_SINGLE_64x64, MODE_SINGLE_64x32: begin
                    // Top arrays see rows 0-31 with their natural wavefront
                    for (i = 0; i < 32; i = i + 1) begin
                        if (feed_counter >= i && (feed_counter - i) < 32) begin
                            // Calculate K-index and wrap to stay within 0-63 bounds
                            k_index = (k_offset + (feed_counter - i)) % 64;
                            matrix_a_row[i] = test_A[i][k_index];
                            matrix_b_col[i] = test_B[k_index][i];
                        end
                    end
                    
                    // Bottom arrays see rows 32-63 but presented as local 0-31
                    // They use the SAME feed_counter pattern
                    for (i = 0; i < 32; i = i + 1) begin
                        if (feed_counter >= i && (feed_counter - i) < 32) begin
                            k_index = (k_offset + (feed_counter - i)) % 64;
                            matrix_a_row[32 + i] = test_A[32 + i][k_index];
                            matrix_b_col[32 + i] = test_B[k_index][32 + i];
                        end
                    end
                end
                
                MODE_DUAL_64x32: begin
                    // First operation
                    for (i = 0; i < 32; i = i + 1) begin
                        if (feed_counter >= i && (feed_counter - i) < 32) begin
                            k_index = (k_offset + (feed_counter - i)) % 64;
                            matrix_a_row[i] = test_A[i][k_index];
                            matrix_b_col[i] = test_B[k_index][i];
                            matrix_a_row[32 + i] = test_A[32 + i][k_index];
                            matrix_b_col[32 + i] = test_B[k_index][32 + i];
                        end
                    end
                    
                    // Second operation
                    for (i = 0; i < 32; i = i + 1) begin
                        if (feed_counter >= i && (feed_counter - i) < 32) begin
                            k_index = (k_offset + (feed_counter - i)) % 64;
                            matrix_a_row_b[i] = test_A_second[i][k_index];
                            matrix_b_col_b[i] = test_B_second[k_index][i];
                            matrix_a_row_b[32 + i] = test_A_second[32 + i][k_index];
                            matrix_b_col_b[32 + i] = test_B_second[k_index][32 + i];
                        end
                    end
                end
            endcase
        end
    end
    
    // =========================================================================
    // Test tasks
    // =========================================================================
    
    task clear_test_data;
        for (ii = 0; ii < 64; ii = ii + 1) begin
            for (jj = 0; jj < 64; jj = jj + 1) begin
                test_A[ii][jj] = 8'sd0;
                test_B[ii][jj] = 8'sd0;
                test_A_second[ii][jj] = 8'sd0;
                test_B_second[ii][jj] = 8'sd0;
                golden_C[ii][jj] = 32'sd0;
                golden_C_second[ii][jj] = 32'sd0;
            end
        end
    endtask
    
    task init_identity(input is_second);
        $display("[INFO] Initializing %s identity matrix", is_second ? "SECOND" : "FIRST");
        for (ii = 0; ii < 64; ii = ii + 1) begin
            for (jj = 0; jj < 64; jj = jj + 1) begin
                if (!is_second) begin
                    test_A[ii][jj] = (ii == jj) ? 8'sd1 : 8'sd0;
                    test_B[ii][jj] = (ii == jj) ? 8'sd1 : 8'sd0;
                end else begin
                    test_A_second[ii][jj] = (ii == jj) ? 8'sd1 : 8'sd0;
                    test_B_second[ii][jj] = (ii == jj) ? 8'sd1 : 8'sd0;
                end
            end
        end
    endtask
    
    task init_simple(input is_second, input integer offset);
        $display("[INFO] Initializing %s simple test matrix with offset %0d", 
                 is_second ? "SECOND" : "FIRST", offset);
        for (ii = 0; ii < 64; ii = ii + 1) begin
            for (jj = 0; jj < 64; jj = jj + 1) begin
                if (!is_second) begin
                    test_A[ii][jj] = 8'sd1 + ((ii + jj + offset) % 3);
                    test_B[ii][jj] = 8'sd1 + ((ii + jj + offset) % 2);
                end else begin
                    test_A_second[ii][jj] = 8'sd1 + ((ii + jj + offset) % 3);
                    test_B_second[ii][jj] = 8'sd1 + ((ii + jj + offset) % 2);
                end
            end
        end
    endtask
    
    task compute_golden(input integer M, input integer K, input integer N, input is_second);
        $display("[INFO] Computing %s golden reference for %0dx%0d x %0dx%0d", 
                 is_second ? "SECOND" : "FIRST", M, K, K, N);
        for (ii = 0; ii < M; ii = ii + 1) begin
            for (jj = 0; jj < N; jj = jj + 1) begin
                sum = 32'sd0;
                for (kk = 0; kk < K; kk = kk + 1) begin
                    if (!is_second)
                        sum = sum + ($signed(test_A[ii][kk]) * $signed(test_B[kk][jj]));
                    else
                        sum = sum + ($signed(test_A_second[ii][kk]) * $signed(test_B_second[kk][jj]));
                end
                if (!is_second)
                    golden_C[ii][jj] = sum;
                else
                    golden_C_second[ii][jj] = sum;
            end
        end
        $display("[INFO] Golden computation complete");
    endtask
    
    task check_results(input integer M, input integer N, input is_second);
        if (!is_second) begin
            errors = 0;
            $display("[INFO] Checking FIRST operation results (%0dx%0d)", M, N);
            
            for (ii = 0; ii < M; ii = ii + 1) begin
                for (jj = 0; jj < N; jj = jj + 1) begin
                    if ($signed(result_c[ii][jj]) !== golden_C[ii][jj][23:0]) begin
                        $display("[ERROR] Mismatch at [%0d][%0d]: HW=%0d, Golden=%0d",
                                 ii, jj, $signed(result_c[ii][jj]), golden_C[ii][jj]);
                        errors++;
                        if (errors > 10) begin
                            $display("[ERROR] Too many errors, stopping check");
                            return;
                        end
                    end
                end
            end
            
            if (errors == 0)
                $display("[PASS] First operation: All %0d results match!", M * N);
            else
                $display("[FAIL] First operation: Found %0d errors", errors);
        end else begin
            errors_second = 0;
            $display("[INFO] Checking SECOND operation results (%0dx%0d)", M, N);
            
            for (ii = 0; ii < M; ii = ii + 1) begin
                for (jj = 0; jj < N; jj = jj + 1) begin
                    if ($signed(result_c_b[ii][jj]) !== golden_C_second[ii][jj][23:0]) begin
                        $display("[ERROR] Mismatch at [%0d][%0d]: HW=%0d, Golden=%0d",
                                 ii, jj, $signed(result_c_b[ii][jj]), golden_C_second[ii][jj]);
                        errors_second++;
                        if (errors_second > 10) begin
                            $display("[ERROR] Too many errors, stopping check");
                            return;
                        end
                    end
                end
            end
            
            if (errors_second == 0)
                $display("[PASS] Second operation: All %0d results match!", M * N);
            else
                $display("[FAIL] Second operation: Found %0d errors", errors_second);
        end
    endtask
    
    task print_corner(input string name, input integer size, input is_second);
        $display("\n%s (%s operation, top-left %0dx%0d):", 
                 name, is_second ? "SECOND" : "FIRST", size, size);
        for (ii = 0; ii < size && ii < 64; ii = ii + 1) begin
            $write("  ");
            for (jj = 0; jj < size && jj < 64; jj = jj + 1) begin
                if (name == "HW Result") begin
                    if (!is_second)
                        $write("%6d ", $signed(result_c[ii][jj]));
                    else
                        $write("%6d ", $signed(result_c_b[ii][jj]));
                end else if (name == "Golden") begin
                    if (!is_second)
                        $write("%6d ", golden_C[ii][jj]);
                    else
                        $write("%6d ", golden_C_second[ii][jj]);
                end else if (name == "A") begin
                    if (!is_second)
                        $write("%4d ", $signed(test_A[ii][jj]));
                    else
                        $write("%4d ", $signed(test_A_second[ii][jj]));
                end else if (name == "B") begin
                    if (!is_second)
                        $write("%4d ", $signed(test_B[ii][jj]));
                    else
                        $write("%4d ", $signed(test_B_second[ii][jj]));
                end
            end
            $display("");
        end
    endtask
    
    task run_test(input string test_name);
        $display("\n========================================");
        $display("TEST: %s", test_name);
        $display("Mode: %s", mode == MODE_SINGLE_64x64 ? "64x64" : 
                             mode == MODE_SINGLE_64x32 ? "Single 64x32" : "Dual 64x32");
        $display("K_dim: %0d", K_dim);
        $display("========================================");
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        fork
            begin
                wait(done);
                @(posedge clk);
                $display("[INFO] Computation completed");
            end
            begin
                repeat(5000) @(posedge clk);
                $display("[ERROR] Timeout");
                $finish;
            end
        join_any
        disable fork;
        
        repeat(2) @(posedge clk);
    endtask
    
    // =========================================================================
    // Main test sequence
    // =========================================================================
    initial begin
        $display("\n==================================================");
        $display("MODULAR 64x64 MMU TESTBENCH");
        $display("Features: 4x 32x32 arrays, supports dual 64x32");
        $display("==================================================\n");
        
        rst_n = 0;
        start = 0;
        mode = 0;
        K_dim = 64;
        
        clear_test_data();
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // =====================================================================
        // TEST 1: MODE_SINGLE_64x64 - Identity
        // =====================================================================
        $display("\n### TEST 1: Single 64x64 x 64x64 - Identity ###");
        clear_test_data();
        init_identity(0);
        mode = MODE_SINGLE_64x64;
        K_dim = 64;
        compute_golden(64, 64, 64, 0);
        run_test("64x64 Identity");
        print_corner("A", 4, 0);
        print_corner("B", 4, 0);
        print_corner("HW Result", 4, 0);
        print_corner("Golden", 4, 0);
        check_results(64, 64, 0);
        
        repeat(10) @(posedge clk);
        
        // =====================================================================
        // TEST 2: MODE_SINGLE_64x64 - Simple values
        // =====================================================================
        $display("\n### TEST 2: Single 64x64 x 64x64 - Simple Values ###");
        clear_test_data();
        init_simple(0, 0);
        mode = MODE_SINGLE_64x64;
        K_dim = 64;
        compute_golden(64, 64, 64, 0);
        run_test("64x64 Simple");
        print_corner("HW Result", 4, 0);
        print_corner("Golden", 4, 0);
        check_results(64, 64, 0);
        
        repeat(10) @(posedge clk);
        
        // =====================================================================
        // TEST 3: MODE_SINGLE_64x32 - Identity
        // =====================================================================
        $display("\n### TEST 3: Single 64x64 x 64x32 - Identity ###");
        clear_test_data();
        init_identity(0);
        mode = MODE_SINGLE_64x32;
        K_dim = 64;
        compute_golden(64, 64, 32, 0);
        run_test("64x32 Identity");
        print_corner("A", 4, 0);
        print_corner("B", 4, 0);
        print_corner("HW Result", 4, 0);
        print_corner("Golden", 4, 0);
        check_results(64, 32, 0);
        
        repeat(10) @(posedge clk);
        
        // =====================================================================
        // TEST 4: MODE_SINGLE_64x32 - Simple values
        // =====================================================================
        $display("\n### TEST 4: Single 64x64 x 64x32 - Simple Values ###");
        clear_test_data();
        init_simple(0, 0);
        mode = MODE_SINGLE_64x32;
        K_dim = 64;
        compute_golden(64, 64, 32, 0);
        run_test("64x32 Simple");
        print_corner("HW Result", 4, 0);
        print_corner("Golden", 4, 0);
        check_results(64, 32, 0);
        
        repeat(10) @(posedge clk);
        
        // =====================================================================
        // TEST 5: MODE_DUAL_64x32 - Identity for both
        // =====================================================================
        $display("\n### TEST 5: DUAL 64x64 x 64x32 - Identity (Both Operations) ###");
        clear_test_data();
        init_identity(0);
        init_identity(1);
        mode = MODE_DUAL_64x32;
        K_dim = 64;
        compute_golden(64, 64, 32, 0);
        compute_golden(64, 64, 32, 1);
        run_test("DUAL 64x32 Identity");
        print_corner("A", 4, 0);
        print_corner("B", 4, 0);
        print_corner("HW Result", 4, 0);
        print_corner("Golden", 4, 0);
        check_results(64, 32, 0);
        print_corner("A", 4, 1);
        print_corner("B", 4, 1);
        print_corner("HW Result", 4, 1);
        print_corner("Golden", 4, 1);
        check_results(64, 32, 1);
        
        repeat(10) @(posedge clk);
        
        // =====================================================================
        // TEST 6: MODE_DUAL_64x32 - Different values for each operation
        // =====================================================================
        $display("\n### TEST 6: DUAL 64x64 x 64x32 - Different Values ###");
        clear_test_data();
        init_simple(0, 0);    // First operation: offset 0
        init_simple(1, 5);    // Second operation: offset 5 (different values)
        mode = MODE_DUAL_64x32;
        K_dim = 64;
        compute_golden(64, 64, 32, 0);
        compute_golden(64, 64, 32, 1);
        run_test("DUAL 64x32 Different");
        print_corner("HW Result", 4, 0);
        print_corner("Golden", 4, 0);
        check_results(64, 32, 0);
        print_corner("HW Result", 4, 1);
        print_corner("Golden", 4, 1);
        check_results(64, 32, 1);
        
        repeat(10) @(posedge clk);
        
        // =====================================================================
        // TEST 7: Multi-tile K dimension (K=128) in SINGLE_64x32 mode
        // =====================================================================
        $display("\n### TEST 7: Single 64x128 x 128x32 - Multi-tile K ###");
        clear_test_data();
        
        // Initialize matrices - since we only have 64 K-elements in storage,
        // the MMU will process this as 4 K-tiles and reuse the data
        for (int i = 0; i < 64; i = i + 1) begin
            for (j = 0; j < 64; j = j + 1) begin
                test_A[i][j] = 8'sd1 + ((i + j) % 3);
                test_B[i][j] = 8'sd1 + ((i + j) % 2);
            end
        end
        
        mode = MODE_SINGLE_64x32;
        K_dim = 128;
        
        // Compute golden: K=128 means 4 K-tiles of 32 each
        // With only 64 columns of storage, the data wraps around
        $display("[INFO] Computing golden reference for 64x128 x 128x32");
        for (ii = 0; ii < 64; ii = ii + 1) begin
            for (jj = 0; jj < 32; jj = jj + 1) begin
                sum = 32'sd0;
                // K-tile 0: K=0-31, K-tile 1: K=32-63
                // K-tile 2: K=64-95 (wraps to storage 0-31)
                // K-tile 3: K=96-127 (wraps to storage 32-63)
                for (kk = 0; kk < 64; kk = kk + 1) begin
                    sum = sum + ($signed(test_A[ii][kk]) * $signed(test_B[kk][jj]));
                end
                // Tiles 2 and 3 reuse same data
                for (kk = 0; kk < 64; kk = kk + 1) begin
                    sum = sum + ($signed(test_A[ii][kk]) * $signed(test_B[kk][jj]));
                end
                golden_C[ii][jj] = sum;
            end
        end
        $display("[INFO] Golden computation complete");
        $display("[INFO] Note: K=128 test reuses K=0-63 data for K=64-127 due to limited test storage");
        
        run_test("64x128x32 Multi-tile");
        print_corner("HW Result", 4, 0);
        print_corner("Golden", 4, 0);
        check_results(64, 32, 0);
        
        repeat(10) @(posedge clk);
        
        // =====================================================================
        // SUMMARY
        // =====================================================================
        $display("\n==================================================");
        $display("ALL TESTS COMPLETED");
        $display("Modular architecture with dual 64x32 capability");
        $display("==================================================\n");
        
        $finish;
    end
    
    initial begin
        #10000000;
        $display("[ERROR] Global timeout");
        $finish;
    end
    
    initial begin
        $dumpfile("mmu_modular.vcd");
        $dumpvars(0, tb_mmu_modular);
    end

endmodule
