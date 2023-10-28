`default_nettype none

module uart_tx(
    input wire clk,
    input wire wr,
    input wire [7:0] data,
    output wire tx
);

    reg [8:0] l_data = 9'h1ff;

    always @(posedge clk) begin
        if (wr) begin
            l_data <= { data, 1'b0 };
        end
        else begin
            l_data <= {1'b1, l_data[8:1]};
        end
    end

    assign tx = l_data[0];
endmodule
