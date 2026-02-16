vlib work
vlog mmu_modular_complete.sv tb_mmu_modular.sv
vsim -voptargs=+acc work.tb_mmu_modular
add wave *
run -all
#quit -sim