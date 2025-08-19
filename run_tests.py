import subprocess

subprocess.run(['python', 'test.py', '--nt', '10', '--debug_test', '--org_type', 'summary', '--org_seed_per', '0.008', '--mut_rat', '0.01', '--imb', '1', '--nw', '32', '--max_bits_of_memory', '4', '--mem_cost', '0.0'])
subprocess.run(['python', 'test.py', '--nt', '10', '--debug_test', '--org_type', 'summary', '--org_seed_per', '0.008', '--mut_rat', '0.01', '--imb', '1', '--nw', '32', '--max_bits_of_memory', '4', '--mem_cost', '0.00001'])
subprocess.run(['python', 'test.py', '--nt', '10', '--debug_test', '--org_type', 'summary', '--org_seed_per', '0.008', '--mut_rat', '0.01', '--imb', '1', '--nw', '32', '--max_bits_of_memory', '4', '--mem_cost', '0.0', '--hard_defect', '32'])
subprocess.run(['python', 'test.py', '--nt', '10', '--debug_test', '--org_type', 'summary', '--org_seed_per', '0.008', '--mut_rat', '0.1', '--imb', '1', '--nw', '32', '--max_bits_of_memory', '4', '--mem_cost', '0.0'])
subprocess.run(['python', 'test.py', '--nt', '10', '--debug_test', '--org_type', 'summary', '--org_seed_per', '0.008', '--mut_rat', '0.01', '--imb', '3', '--nw', '32', '--max_bits_of_memory', '4', '--mem_cost', '0.0'])
