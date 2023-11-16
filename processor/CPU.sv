/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSED */

module CPU(input    logic   clk,
           input    logic   rst);

    ////////////////////
    // IF
    ////////////////////
    logic [63:0] PC, PCNext, PCPlus4;
    logic [31:0] instr;
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            PC <= 0;
        end
        else begin
            PC <= PCNext;
        end
    end

    assign PCPlus4 = PC + 1;
    assign PCNext = PCPlus4;

    icache ic(
        .address(PC[31:0]),
        .rd(instr)
    );

    ////////////////////
    // DE
    ////////////////////
    logic [6:0] id_opcode = instr[6:0];
    logic [4:0] id_rd     = instr[11:7];
    logic [2:0] id_funct3 = instr[14:12];
    logic [4:0] id_rs1    = instr[19:15];
    logic [4:0] id_rs2    = instr[24:20];
    logic [6:0] id_funct7 = instr[31:25];

    logic [31:0] id_iimm = { {20{instr[31]}}, instr[31:20] };
    logic [31:0] id_simm = { {20{instr[31]}}, instr[31:25], instr[11:7] };
    logic [31:0] id_bimm = { {20{instr[31]}}, instr[7], instr[30:25], instr[11:8], 1'b0 };
    logic [31:0] id_uimm = { instr[31:12], 12'b0 };
    logic [31:0] id_jimm = { {11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0 };


    ////////////////////
    // DEBUG
    ////////////////////
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            $display("rst   PC:%h", PC);
        end
        else begin
            $display("clk   PC:%h   instr:%h", PC, instr);
        end
    end

endmodule


module icache(input     logic [31:0] address, 
              output    logic [31:0] rd
);
    logic [31:0] ICACHE[100:0];
    initial begin 
        $readmemh("./test/mem_instr", ICACHE);
    end

    assign rd = ICACHE[address];

endmodule

module dcache(input     logic        clk,
              input     logic [63:0] address, 
              input     logic [63:0] wd,
              input     logic        we,
              output    logic [63:0] rd
);
    logic [63:0] DCACHE[100:0];
    initial begin 
        $readmemh("./test/mem_data", DCACHE);
    end

    assign rd = DCACHE[address];

    always_ff @(posedge clk) begin
        if (we) DCACHE[address] <= wd;
    end

endmodule

module registerfile(input   logic           clk,
                    input   logic [5:0]     a1, a2, a3,
                    input   logic           we3,
                    input   logic [63:0]    wd3,
                    output  logic [63:0]    rd1,
                    output  logic [63:0]    rd2
);
    logic [63:0] REGS[31:0];

    assign rd1 = (a1 != 0) ? REGS[a1] : 0;
    assign rd2 = (a2 != 0) ? REGS[a2] : 0;

    always_ff @(posedge clk) begin
        if (we3) REGS[a3] <= wd3;
    end

endmodule

/* verilator lint_on DECLFILENAME */
/* verilator lint_off UNUSED */