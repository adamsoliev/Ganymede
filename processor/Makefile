# Makefile for Verilog project

# Define the target executable name
TARGET = processor

# Source files (Verilog files)
SOURCES = main.v bench.v 

# Compiler and simulator
IVERILOG = iverilog
VVP = vvp

# Build rule
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(IVERILOG) -DBENCH -g2009 -o $(TARGET) $(SOURCES)

# Run the simulation
run: $(TARGET)
	$(VVP) $(TARGET)

# Clean generated files
clean:
	rm -f $(TARGET) *.vcd

.PHONY: all run clean