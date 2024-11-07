
module async_set_register #( n=8 ) (
    input  [n-1:0] D,
    output  [n-1:0] Q,
    input clock,
    input set
);
reg [n-1:0] _Q;
always @(posedge clock or posedge set) begin
    if(set == 1) 
    begin
      _Q <= {n{1'b1}};
    end 
    else 
    begin
      _Q <= D;
    end
end
assign Q = _Q;
endmodule
    