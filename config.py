import argparse
import time
import random
import numpy as np
import ast

def parse_everything():
    """Parse command line arguments and set global constants"""
    arg_parser = argparse.ArgumentParser(
        description='The changing environments program.')
    
    # Expects 1 argument: output folder
    arg_parser.add_argument("-o", "--output_folder", nargs=1)
    arg_parser.add_argument("--seed", type=int, default=random.randint(0, int(time.time())))
    arg_parser.add_argument("--number_of_generations", "--ng",  type=int, default=1000)
    arg_parser.add_argument("--number_of_organisms", "--no",  type=int, default=500)
    arg_parser.add_argument("--org_type", type=str, default="pd")
    arg_parser.add_argument("--org_seed_per", type=float, default=0.0)
    arg_parser.add_argument("--tournament_size", "--ts",  type=int, default=10)
    arg_parser.add_argument("--verbose", action='store_true', default=False)
    arg_parser.add_argument("--number_of_rounds", "--nr",  type=int, default=64)
    arg_parser.add_argument("--temptation", type=int, default=5)
    arg_parser.add_argument("--reward", type=int, default=3)
    arg_parser.add_argument("--punishment", type=int, default=1)
    arg_parser.add_argument("--sucker", type=int, default=0)
    arg_parser.add_argument("--proportion_cost_per_memory_bit", "--m_c", type=float, default=0.0)
    arg_parser.add_argument("--max_bits_of_memory", "--max_m", type=int, default=1)
    arg_parser.add_argument("--mutation_likelihood_of_bits_of_memory", "--ml_mem",  type=float, default=0.01)
    arg_parser.add_argument("--mutation_likelihood_of_initial_memory_state", "--ml_dec", type=float, default=0.01)
    arg_parser.add_argument("--toggle_self_memory_on", action='store_true', default=False)
    arg_parser.add_argument("--mutation_rate", "--mut_rat",  type=float, default=0.0)
    arg_parser.add_argument("--output_frequency", type=int, default=10)
    arg_parser.add_argument("--selection_by_static_competitor", "--static", action="store_true", default=False)
    arg_parser.add_argument("--randomized_rounds", action="store_true", default=False)
    arg_parser.add_argument("--noise", type=float, default=0.0)
    arg_parser.add_argument("--initial_memory_bits", "--imb", type=int, default=1)
    arg_parser.add_argument("--static_comps", "-stat_c", nargs='+', help='[ALL_DEFECT, TIT_FOR_TAT, COIN_FLIP] TODO: Fully-integrate', type=str)
    arg_parser.add_argument("--hard_defect", "--hf", nargs='*', help='[16, 32, 48]', type=str)
    #arg_parser.add_argument("--env_seed", type=str, default="")
    #arg_parser.add_argument("--prob_org", "--po", action='store_true', default=False)
    
    args = arg_parser.parse_args()    

    if args.hard_defect:
        if len(args.hard_defect) > 1:
            args.hard_defect = args.hard_defect[0]
            args.hard_defect = [int(i) for i in args.hard_defect]
        else:
            args.hard_defect = [int(args.hard_defect[0])]
    
    args.output_folder = str(args.output_folder[0])
    args.start_time = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args

class ConstConfig():
    """
    Single Config Class for handling experimental settings extracted from
    argparse as constants. Inspired by Thread-Safe Singleton Pattern: https://refactoring.guru/design-patterns/singleton/python/example#example-1
    """
    def __init__(self, args):
        self.output_folder = args.output_folder
        self.start_time = args.start_time
        self.seed = args.seed
        self.verbose = args.verbose
        self.number_of_organisms = args.number_of_organisms
        self.number_of_generations = args.number_of_generations
        self.org_type = args.org_type
        self.org_seed_per = args.org_seed_per
        self.mutation_rate = args.mutation_rate
        self.output_frequency = args.output_frequency
        self.max_bits_of_memory = args.max_bits_of_memory
        self.mutation_likelihood_of_bits_of_memory = args.mutation_likelihood_of_bits_of_memory
        self.mutation_likelihood_of_initial_memory_state = args.mutation_likelihood_of_initial_memory_state
        self.selection_by_static_competitor = args.selection_by_static_competitor
        self.number_of_rounds = args.number_of_rounds
        self.randomized_rounds = args.randomized_rounds
        self.noise = args.noise
        self.temptation = args.temptation
        self.reward = args.reward
        self.punishment = args.punishment
        self.sucker = args.sucker
        self.proportion_cost_per_memory_bit = args.proportion_cost_per_memory_bit
        self.toggle_self_memory_on = args.toggle_self_memory_on
        self.tournament_size = args.tournament_size
        self.initial_memory_bits = args.initial_memory_bits
        self.static_comps = args.static_comps
        self.hard_defect = args.hard_defect
    
    """
    Return the member variables from the argparse object 
    as properties so the member cannot be changed
    Since the CONST_CONF object in main.py
    is global then it shouldn't change but I'm paranoid about
    potential changes that emerge from accessing these member variables
    """ 
    @property
    def OUTPUT_FOLDER(self):
        return self.output_folder
    @property
    def START_TIME(self):
        return self.start_time
    @property
    def SEED(self):
        return self.seed
    @property
    def VERBOSE(self):
        return self.verbose
    @property
    def NUMBER_OF_ORGANISMS(self):
        return self.number_of_organisms
    @property
    def NUMBER_OF_GENERATIONS(self):
        return self.number_of_generations
    @property
    def ORG_TYPE(self):
        return self.org_type
    @property
    def ORG_SEED_PER(self):
        return self.org_seed_per
    @property
    def MUTATION_RATE(self):
        return self.mutation_rate
    @property
    def OUTPUT_FREQUENCY(self):
        return self.output_frequency
    @property
    def MAX_BITS_OF_MEMORY(self):
        return self.max_bits_of_memory
    @property
    def MUTATION_LIKELIHOOD_OF_BITS_OF_MEMORY(self):
        return self.mutation_likelihood_of_bits_of_memory
    @property
    def MUTATION_LIKELIHOOD_OF_INITIAL_MEMORY_STATE(self):
        return self.mutation_likelihood_of_initial_memory_state
    @property
    def SELECTION_BY_STATIC_COMPETITOR(self):
        return self.selection_by_static_competitor
    @property
    def NUMBER_OF_ROUNDS(self):
        return self.number_of_rounds
    @property
    def RANDOMIZED_ROUNDS(self):
        return self.randomized_rounds
    @property
    def NOISE(self):
        return self.noise
    @property
    def TEMPTATION(self):
        return self.temptation
    @property
    def REWARD(self):
        return self.reward
    @property
    def PUNISHMENT(self):
        return self.punishment
    @property
    def SUCKER(self):
        return self.sucker
    @property
    def PROPORTION_COST_PER_MEMORY_BIT(self):
        return self.proportion_cost_per_memory_bit
    @property
    def TOGGLE_SELF_MEMORY_ON(self):
        return self.toggle_self_memory_on
    @property
    def TOURNAMENT_SIZE(self):
        return self.tournament_size
    @property
    def INITIAL_MEMORY_BITS(self):
        return self.initial_memory_bits
    @property
    def STATIC_COMPETITORS(self):
        return self.static_comps
    
    @property
    def HARD_DEFECT(self):
        return self.hard_defect
    
CONST_CONF = ConstConfig(parse_everything()) #global Singleton object with experimental settings as constants. Placed here to avoid circular imports