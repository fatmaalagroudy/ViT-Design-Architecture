vlib work
vlog vit_top_integrated.sv simple_mmu.sv dyadic_params.sv softmax_pipelined.sv gelu_pipelined.sv layernorm_pipelined.sv systolic_array_mmu.sv vit_tb.sv
vsim -voptargs=+acc work.vit_tb
add wave *
run -all
#quit -sim