`default_nettype none

module processor(
    input wire clk
);
    reg [31:0] MEM [0:4096];
    initial $readmemh("/home/adam/dev/computer-stuff/cpu/test_a/rv64ui-p-add", MEM);

    ////////////////////////////////////////////////////////////////////////////////
    // FETCH
    ////////////////////////////////////////////////////////////////////////////////
    reg [63:0] PC = 0;
    reg [31:0] instruction = 32'b0000000_00000_00000_000_00000_0110011; // NOP
    always @(posedge clk) begin
        instruction <= MEM[PC];
        PC <= PC + 1'b1;
    end

    ////////////////////////////////////////////////////////////////////////////////
    // DECODE
    ////////////////////////////////////////////////////////////////////////////////
    // 10 RISC-V instructions
    wire isLOAD         = (instruction[6:0] == 7'b0000011); // rd <- mem[rs1+Iimm]
    wire isLOAD_FP      = (instruction[6:0] == 7'b0000111);
    wire isMISC_MEM     = (instruction[6:0] == 7'b0001111);
    wire isOP_IMM       = (instruction[6:0] == 7'b0010011); // rd <- rs1 OP Iimm
    wire isAUIPC        = (instruction[6:0] == 7'b0010111); // rd <- PC + Uimm
    wire isOP_IMM_32    = (instruction[6:0] == 7'b0011011);
    wire isSTORE        = (instruction[6:0] == 7'b0100011); // mem[rs1+Simm] <- rs2
    wire isSTORE_FP     = (instruction[6:0] == 7'b0100111);
    wire isAMO          = (instruction[6:0] == 7'b0101111);
    wire isOP           = (instruction[6:0] == 7'b0110011); // rd <- rs1 OP rs2   
    wire isLUI          = (instruction[6:0] == 7'b0110111); // rd <- Uimm   
    wire isOP_32        = (instruction[6:0] == 7'b0111011);
    wire isMADD         = (instruction[6:0] == 7'b1000011);
    wire isMSUB         = (instruction[6:0] == 7'b1000111);
    wire isNMSUB        = (instruction[6:0] == 7'b1001011);
    wire isNMADD        = (instruction[6:0] == 7'b1001111);
    wire isOP_FP        = (instruction[6:0] == 7'b1010011);
    wire isBRANCH       = (instruction[6:0] == 7'b1100011); // if(rs1 OP rs2) PC<-PC+Bimm
    wire isJALR         = (instruction[6:0] == 7'b1100111); // rd <- PC+4; PC<-rs1+Iimm
    wire isJAL          = (instruction[6:0] == 7'b1101111); // rd <- PC+4; PC<-PC+Jimm
    wire isSYSTEM       = (instruction[6:0] == 7'b1110011); // special

    // 5 immediate formats
    wire [31:0] Uimm = { instruction[31:12], 12'b0 };
    wire [31:0] Iimm = { {20{instruction[31]}}, instruction[31:20] };
    wire [31:0] Simm = { {20{instruction[31]}}, instruction[31:25], instruction[11:7] };
    wire [31:0] Bimm = { {20{instruction[31]}}, instruction[7], instruction[30:25], instruction[11:8], 1'b0 };
    wire [31:0] Jimm = { {11{instruction[31]}}, instruction[31], instruction[19:12], instruction[20], instruction[30:21], 1'b0 };
    // wire [6:0] opcode = instruction[6:0];

    // source and destination registers
    wire [4:0] rs1Id = instruction[19:15];
    wire [4:0] rs2Id = instruction[24:20];
    wire [4:0] rdId  = instruction[11:7];

    // function codes
    wire [2:0] funct3 = instruction[14:12];
    wire [6:0] funct7 = instruction[31:25];

    ////////////////////////////////////////////////////////////////////////////////
    // EXECUTE
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // MEMORY ACCESS
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // WRITE BACK
    ////////////////////////////////////////////////////////////////////////////////


    always @(posedge clk) begin
        // $display("%h %b %b", instruction, instruction[31:24], instruction[6:0]);

        if (isLOAD     ) $display("PC:%3h %h  LOAD     ", PC, instruction);
        if (isLOAD_FP  ) $display("PC:%3h %h  LOAD_FP  ", PC, instruction);
        if (isMISC_MEM ) $display("PC:%3h %h  MISC_MEM ", PC, instruction);
        if (isOP_IMM   ) $display("PC:%3h %h  OP_IMM   ", PC, instruction);
        if (isAUIPC    ) $display("PC:%3h %h  AUIPC    ", PC, instruction);
        if (isOP_IMM_32) $display("PC:%3h %h  OP_IMM_32", PC, instruction);
        if (isSTORE    ) $display("PC:%3h %h  STORE    ", PC, instruction);
        if (isSTORE_FP ) $display("PC:%3h %h  STORE_FP ", PC, instruction);
        if (isAMO      ) $display("PC:%3h %h  AMO      ", PC, instruction);
        if (isOP       ) $display("PC:%3h %h  OP       ", PC, instruction);
        if (isLUI      ) $display("PC:%3h %h  LUI      ", PC, instruction);
        if (isOP_32    ) $display("PC:%3h %h  OP_32    ", PC, instruction);
        if (isMADD     ) $display("PC:%3h %h  MADD     ", PC, instruction);
        if (isMSUB     ) $display("PC:%3h %h  MSUB     ", PC, instruction);
        if (isNMSUB    ) $display("PC:%3h %h  NMSUB    ", PC, instruction);
        if (isNMADD    ) $display("PC:%3h %h  NMADD    ", PC, instruction);
        if (isOP_FP    ) $display("PC:%3h %h  OP_FP    ", PC, instruction);
        if (isBRANCH   ) $display("PC:%3h %h  BRANCH   ", PC, instruction);
        if (isJALR     ) $display("PC:%3h %h  JALR     ", PC, instruction);
        if (isJAL      ) $display("PC:%3h %h  JAL      ", PC, instruction);
        if (isSYSTEM   ) $display("PC:%3h %h  SYSTEM   ", PC, instruction);
   end

endmodule
