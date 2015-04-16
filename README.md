## DSGD Matrix Factorization with Spark

### Author
Siqi Wang (siqiw@andrew.cmu.edu)

### Introduction
This package implements DSGD matrix factorization algorithm using pySpark. The algorithm decomposes input matrix V into two lower ranked matrices W and H. 

The input file should be in sparse matrix format:

	i_0, j_0, value_0
	i_1, j_1, value_1
	...
	
The output files are in dense matrix format.

### Usage
	spark-submit dsgd_mf.py [--master $(master_url)] $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)



