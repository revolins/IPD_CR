import subprocess
from tqdm import tqdm
import argparse
import datetime
import shutil
import os
import random
import time
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import generate_data

def det_output(args):
    folder_str = 'pd'
    if args.org_type == 'hybrid':
        folder_str = folder_str + "hybrid"
    if args.org_type == 'summary':
        folder_str = folder_str + "summary"
    if args.prob_org:
        folder_str = folder_str + "prob"
    if args.static:
        #args.mut_rat = 0.0
        folder_str = folder_str + "static"
    else:
        folder_str = folder_str + "coev"

    if args.noise > 0.0:
        folder_str = folder_str + "noise" + str(args.noise)
    if args.remove_mem_limit:
        folder_str = folder_str + "maxmem"
    if args.initial_memory_bits != 0:
        folder_str = folder_str + f'imb{str(args.initial_memory_bits)}'
    
    if args.mut_rat > 0.09:
        folder_str = folder_str + "highmut_"
    elif args.mut_rat <= 0.09 and args.mut_rat > 0.0:
        folder_str = folder_str + "lowmut_"
    else:
        folder_str = folder_str + "nomut_"

    if args.env_seed == 'coop':
        folder_str = folder_str + 'coop'
    if args.env_seed == 'hostile':
        folder_str = folder_str + 'hostile'
    if len(str(args.seed)) < 10:
        folder_str = folder_str + f'seed{args.seed}'

    now = datetime.datetime.now()
    current_time = now.strftime('%m%d%y%H%M%S')
    output_folder = folder_str + current_time

    return output_folder, folder_str

def format_cmd(args, arg_parser):
    temp_cmd = []
    arg_dict = vars(args).copy()
    print("ARG_DICT ", arg_dict, flush=True)
    arg_dict.pop('debug_test')
    arg_dict.pop('num_workers')
    arg_dict.pop('num_test')
    for i in arg_dict: 
        if arg_dict[i] != arg_parser.get_default(i):
            print(f'UPDATE -- {i} -- {arg_dict[i]}')
            if type(arg_dict[i]) != bool:
                temp_cmd.extend([f"--{i}", str(arg_dict[i])])
            else:
                temp_cmd.extend([f"--{i}"])

    if '--hard_defect' in temp_cmd:
        hf_test = temp_cmd.pop()
        for i in ast.literal_eval(hf_test):
            temp_cmd.append(i) 
    print(f"TEMPORARY COMMAND -- {temp_cmd}", flush=True)
    return temp_cmd

    if args.static:
        temp_cmd.append("--static")
    if args.mut_rat > 0.0 and not args.static:
        temp_cmd.extend(["--mr", str(args.mut_rat)])
    if args.noise > 0.0:
        temp_cmd.extend(["--noise", str(args.noise)])
    if args.hybrid and not args.prob_org:
        temp_cmd.extend(["--org_type", "hybrid_pd"])
    if args.env_seed == 'hostile':
        temp_cmd.extend(["--env_seed", "hostile", "--org_type", "hostile_pd"])
    if args.env_seed == 'coop':
        temp_cmd.extend(["--env_seed", "coop", "--org_type", "friend_pd"])
    if args.prob_org:
        temp_cmd.extend(["--org_type", "hybrid_pd", "--max_bits_of_memory", "0", "--max_bits_of_summary", "4"])
    if args.remove_mem_limit:
        temp_cmd.extend(["--max_bits_of_memory", "16"])
    if args.number_of_organisms != 500:
        temp_cmd.extend(["--no", str(args.number_of_organisms)])
    if args.tournament_size != 10:
        temp_cmd.extend(["--ts", str(args.tournament_size)])
    if args.initial_memory_bits != 0 and not args.static:
        #Temporary to reproduce results for Figure 6 in Co-Evolve Environment
        temp_cmd.extend(["--imb", str(args.initial_memory_bits)])
    if args.mutation_likelihood_of_bits_of_memory != 0.01:
        temp_cmd.extend(["--ml_mem", str(args.mutation_likelihood_of_bits_of_memory)])
    if args.mutation_likelihood_of_initial_memory_state != 0.01:
        temp_cmd.extend(["--ml_dec", str(args.mutation_likelihood_of_initial_memory_state)])

    return temp_cmd

def build_plt_cmd(args):
    temp_cmd = []
    if args.output_frequency != 10:
        temp_cmd.extend(["--output_frequency", str(args.output_frequency)])
    if args.number_of_generations != 500:
        temp_cmd.extend(["--number_of_generations", str(args.number_of_generations)])

    return temp_cmd

def run_subtest(command):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr} occurred while executing command: {command}"

def run_test(args, arg_parser):
    if os.path.exists("temp_test"):
        shutil.rmtree("temp_test")
    output_folder, test_type = det_output(args)
    temp_cmd = format_cmd(args, arg_parser)
    temp_str = build_plt_cmd(args)

    mem_cost = ['0.0', '0.01', '0.05', '0.075', '0.2']
    test_list = []
    for i in range(1, args.num_test + 1):
        for cost in mem_cost:
            #default_cmd = ["python", "main.py", "--m_c", cost, "-o", f"temp_test/temp_test{cost}"]
            default_cmd = ["python", "main.py", "--m_c", cost, "-o", f"{output_folder}/pd_{test_type}test{i}_{cost}_cost"]
            if not args.ignore_matching: default_cmd.extend(["--seed", f"{i}"])
            if len(temp_cmd) > 0: default_cmd.extend(temp_cmd)
            if len(temp_str) > 0: default_cmd.extend(temp_str)
            test_list.append(default_cmd)

    if args.ignore_matching: print("Initiating Unseeded Run")
    else: print("Initiating Seeded Run -- NOTE: large args.number_of_organisms values will take awhile")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(run_subtest, cmd): cmd for cmd in test_list}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                cmd = futures[future]
                
                f = open(f"{output_folder}/cmd_tests.txt", "a")
                cmd_str = ''
                for i in cmd:
                    cmd_str += str(i)
                    cmd_str += ' '
                f.write(cmd_str +'\n')
                f.close()
                if args.debug_test:
                    print(cmd)
                try:
                    if args.debug_test:
                        print(future.result())
                    future.result()
                except Exception as exc:
                    print(f"Subtest issued by this command: {cmd}. Generated the following exception: {exc}")
                pbar.update(1)

    subprocess.run(["python", "compile_csv.py", "-o", output_folder])
    plot_str = ["python", "plot_csv.py", "-o", output_folder]
    if len(temp_str) > 0: plot_str.extend(temp_str)
    subprocess.run(plot_str)

    print(f"Experiment Concluded, results stored in {output_folder}")

def main():
    # arg_parser = argparse.ArgumentParser(
    #     description='Testing Script for PD Experiments.')
    
    # arg_parser.add_argument("--num_test", "--nt", type=int, default=20, help="(int), (DEFAULT=20), Specify number of unique seeded runs")
    # arg_parser.add_argument("--static", "--s", action='store_true', default=False, help="(bool) (DEFAULT=False) Specify whether to test in static environment with no mutation rate")
    # arg_parser.add_argument("--mut_rat", "--mr", type=float, default=0.0, help="(float) (DEFAULT=0.0) Set the rate at which the system will mutate memory and decision list")
    # arg_parser.add_argument("--noise", "--n", type=float, default=0.0, help="(float) (DEFAULT = 0.0) Percent Likelihood the one of the opposing organisms moves is misread")
    # arg_parser.add_argument("--org_type", "--ot", type=str, default='pd', help="(str) (DEFAULT = pd) original memory model (pd), hybrid memory model (hybrid), summary memory model (summary)")
    # arg_parser.add_argument("--number_of_generations", "--ng", type=int, default=500, help="(int) (DEFAULT = 500) number of generations selected upon after a tournament")
    # arg_parser.add_argument("--number_of_organisms", "--no", type=int, default=500, help="(int) (DEFAULT=500) Number of organisms available in a population")
    # arg_parser.add_argument("--mutation_likelihood_of_bits_of_memory", "--ml_mem",  type=float, default=0.01)
    # arg_parser.add_argument("--mutation_likelihood_of_initial_memory_state", "--ml_dec", type=float, default=0.01)
    # arg_parser.add_argument("--tournament_size", "--ts", type=int, default=10, help="(int) (DEFAULT=10) Number organisms interacting during a round")
    # arg_parser.add_argument("--output_frequency", "--of", type=int, default=10, help="(int) (DEFAULT = 10) Determines the organisms output to the detail-*.csv, where * is the generation number")
    # arg_parser.add_argument("--ignore_matching", "--ms", action='store_true', default=False, help="(bool) (DEFAULT = False) If the experiment will match seeds to the runs")
    # arg_parser.add_argument("--remove_mem_limit", "--ml", action='store_true', default=False, help="(bool) (DEFAULT = False) If experiment will run without memory limit on number of bits of summary and memory")
    # arg_parser.add_argument("--env_seed", "--es", type=str, default='', help="(str) (DEFAULT = '') Specify 'hostile' or 'coop' to seed environment with all friendly or all hostile")
    # arg_parser.add_argument("--prob_org", "--po", action='store_true', default=False, help="(bool) (DEFAULT = False) Set to determine if organism is probabilistic")
    # arg_parser.add_argument("--num_workers", "--nw", type=int, default=4, help="(int) (DEFAULT=4) Specify number of cores used for threading")
    # arg_parser.add_argument("--debug_test", "--dbt", action='store_true', default=False, help="(bool) (DEFAULT=False) Specify whether each test run prints command used and thread result")
    # arg_parser.add_argument("--initial_memory_bits", "--imb", type=int, default=0, help="(int) (DEFAULT=0) Force PDOrg to have an initial number of memory bits")
    # arg_parser.add_argument("--seed", type=int, default=random.randint(0, int(time.time())))
    # arg_parser.add_argument("--hard_defect", "--hf", nargs='*', help='[16, 32, 48]', type=str)
    # args = arg_parser.parse_args()
    
    # run_test(args, arg_parser)
    
    generate_data()

if __name__ == "__main__":
    main()
