.PHONY: all
all: CPU

VERILATOR=verilator
VINC := /usr/share/verilator/include

obj_dir/VCPU.cpp: CPU.sv
	$(VERILATOR) --trace -Wall -cc CPU.sv --top-module CPU

obj_dir/VCPU__ALL.a: obj_dir/VCPU.cpp
	make --no-print-directory -C obj_dir -f VCPU.mk

CPU: CPU.cpp obj_dir/VCPU__ALL.a
	g++ -I$(VINC) -I obj_dir            \
			$(VINC)/verilated.cpp       \
			$(VINC)/verilated_vcd_c.cpp \
			CPU.cpp obj_dir/VCPU__ALL.a \
			-o CPU 

.PHONY: clean
clean:
	rm -rf obj_dir/ CPU CPUtrace.vcd

