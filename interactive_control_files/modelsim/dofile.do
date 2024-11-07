add wave -position insertpoint  \
sim/:tb_spi_master:spi_master:* 
run -all
wave zoom full
