# # Makefile for Verilog project

# # Define the target executable name
# TARGET = SOC

# # Source files (Verilog files)
# SOURCES = uart.v

# # Compiler and simulator
# IVERILOG = iverilog
# VVP = vvp

# # Build rule
# all: $(TARGET)

# $(TARGET): $(SOURCES)
# 	$(IVERILOG) -DBENCH -g2009 -o $(TARGET) $(SOURCES)

# # Run the simulation
# run: $(TARGET)
# 	$(VVP) $(TARGET)

# # Clean generated files
# clean:
# 	rm -f $(TARGET) *.vcd

# .PHONY: all run clean

# SOC
# all:
# 	verilator -Wall -cc SOC.v; cd obj_dir; make -f VSOC.mk
# 	g++ -I /usr/share/verilator/include -I obj_dir/ /usr/share/verilator/include/verilated.cpp SOC.cpp obj_dir/VSOC__ALL.a -o SOC 
# 	./SOC 

# clean:
# 	rm -rf obj_dir; rm SOC


all:
	verilator -Wall -cc CPU.sv; cd obj_dir; make -f VCPU.mk
	g++ -I /usr/share/verilator/include -I obj_dir/ /usr/share/verilator/include/verilated.cpp CPU.cpp obj_dir/VCPU__ALL.a -o CPU 
	./CPU 

clean:
	rm -rf obj_dir; rm CPU