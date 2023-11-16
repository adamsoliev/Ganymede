
all:
	verilator -Wall -cc CPU.sv --top-module CPU; cd obj_dir; make -f VCPU.mk 
	g++ -I /usr/share/verilator/include -I obj_dir/ /usr/share/verilator/include/verilated.cpp CPU.cpp obj_dir/VCPU__ALL.a -o CPU 
	./CPU 

clean:
	rm -rf obj_dir; rm CPU

