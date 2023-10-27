# # Makefile for Verilog project

# # Define the target executable name
# TARGET = processor

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

all:
	verilator -Wall -cc uart.v; cd obj_dir; make -f Vuart.mk
	g++ -I /usr/share/verilator/include -I obj_dir/ /usr/share/verilator/include/verilated.cpp uart.cpp obj_dir/Vuart__ALL.a -o uart 
	./uart 

clean:
	rm -rf obj_dir; rm uart