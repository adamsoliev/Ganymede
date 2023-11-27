.PHONY: all
all: CPU run

VERILATOR=verilator
VINC := /usr/share/verilator/include

obj_dir/VCPU.cpp: CPU.sv
	@$(VERILATOR) --quiet-exit --trace -Wall -cc CPU.sv --top-module CPU

obj_dir/VCPU__ALL.a: obj_dir/VCPU.cpp
	@make --no-print-directory -C obj_dir -f VCPU.mk

CPU: CPU.cpp obj_dir/VCPU__ALL.a
	@g++ -I$(VINC) -I obj_dir            \
			$(VINC)/verilated.cpp       \
			$(VINC)/verilated_vcd_c.cpp \
			CPU.cpp obj_dir/VCPU__ALL.a \
			-o CPU 

TESTS := lb lbu lh lhu lw lwu ld addi slli slti sltiu xori ori andi auipc sb sh sw sd and sub sll slt sltu xor or and jal addiw slliw addw subw 
# failing: sllw lui

run:
	@$(foreach test,$(TESTS), \
        echo "Running test: $(test)"; \
        /usr/bin/time --quiet  -f "%e" timeout 5s ./CPU $(test) || echo "$(test) failed"; true;)
	# @ gtkwave CPUtrace.vcd

.PHONY: clean
clean:
	rm -rf obj_dir/ CPU CPUtrace.vcd

