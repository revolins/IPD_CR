"""
This module contains methods to evolve IPD strategies. 
"""

#!/usr/bin/python

from __future__ import division
from config import CONST_CONF
from collections import defaultdict
import random
import csv
import os
import datetime
import pd_selection
import template_org as to
import pd_make_detail_file
import shutil
import pandas as pd

def create_initial_population():
    """
    Create a starting population by forming a list of randomly generated organisms.
    """
    #TODO: Set rtft variant organism as a StaticOrg? May cause problems within environment?
    org_type_map = {"pd": to.PDOrg, "hybrid": to.HybridPDOrg, "summary": to.SummaryPDOrg} #, "hostile_pd": to.HybridPDOrg(gen_str='hostile')}

    if CONST_CONF.ORG_TYPE in org_type_map and CONST_CONF.ORG_SEED_PER == 0.0:

        return [org_type_map[CONST_CONF.ORG_TYPE]() for _ in range(CONST_CONF.NUMBER_OF_ORGANISMS)]
    else:
        num_variants = int(CONST_CONF.NUMBER_OF_ORGANISMS * CONST_CONF.ORG_SEED_PER)

        if CONST_CONF.INITIAL_MEMORY_BITS > 1:
            return [org_type_map[CONST_CONF.ORG_TYPE](gen_str='mr') for _ in range(num_variants)] + \
                                [org_type_map[CONST_CONF.ORG_TYPE](gen_str='lrg_nontft') for _ in range(CONST_CONF.NUMBER_OF_ORGANISMS - num_variants)]
        else:
            return [org_type_map[CONST_CONF.ORG_TYPE](gen_str='rtft') for _ in range(num_variants)] + \
                                [org_type_map[CONST_CONF.ORG_TYPE](gen_str='nontft') for _ in range(CONST_CONF.NUMBER_OF_ORGANISMS - num_variants)]
    
def get_mutated_population(population):
    """
    Return a new population with a percentage of organisms mutated based on the mutation rate.
    """
    new_population = []
    for org in population:
        if random.random() < CONST_CONF.MUTATION_RATE:
            new_org = org.get_mutant()
            new_population.append(new_org)
        else:
            new_population.append(org)
    return new_population

def print_status(generation, population, environment):
    """Outputs information to console. Only used if VERBOSE is true"""
    average_fitness = get_average_fitness(population, environment)
    print("Gen = {}  Pop = {}  Fit = {}".format(generation, population, average_fitness))

def get_tally_of_number_of_bits_of_memory(organisms):
    """
    Returns a list of length MAX_BITS_OF_MEMORY + 1 where each position represents the number
    of organisms with that many number of bits of memory
    """
     
    bits_of_memory = [org.genotype.number_of_bits_of_memory for org in organisms]
    
    tally = [0 for _ in range(CONST_CONF.MAX_BITS_OF_MEMORY + 1)]
    
    for bits in bits_of_memory:
        tally[bits] += 1
        
    return tally

def get_tally_of_number_of_bits_summary(organisms):
    """
    Returns a list of length MAX_BITS_OF_MEMORY + 1 where each
    position represents the number of organisms with that many number of bits of summary
    """

    bits_of_summary = [org.genotype.number_of_bits_of_summary for org in organisms]

    tally = [0 for _ in range(CONST_CONF.MAX_BITS_OF_MEMORY + 1)]

    for bits in bits_of_summary:
        tally[bits] += 1

    return tally

def convert_and_pivot_org_tracker(org_tracker):
    rows = []
    for strategy_key, move_dict in org_tracker.items():
        for move_type, gen_dict in move_dict.items():
            for generation, value in gen_dict.items():
                rows.append({
                    'Strategy': strategy_key,
                    'Generation': generation,
                    move_type: value  # Column named 'Coops' or 'Defects'
                })

    # Create the DataFrame from the flattened list of rows
    df = pd.DataFrame(rows)

    # Pivot so that Coops and Defects become columns
    pivoted_df = df.pivot_table(
        index=['Strategy', 'Generation'],
        values=['Coops', 'Defects'],
        aggfunc='sum',
        fill_value=0  # in case some combinations are missing
    ).reset_index()

    return pivoted_df

def update_tracker(org_tracker, i, org, move_type):
    if move_type == 'Coops': 
        move = org.coops
    if move_type == 'Defects': 
        move = org.defects
    if move_type == None:
        move = 0
    org_tracker[f"{org.genotype.decision_list}~{org.genotype.initial_memory}"][move_type][i] += move

    return org_tracker

def pd_evolve_population():
    """
    Evolution loop for PD org representation
    Returns data for "bits_of_memory_overtime.csv"
    """
    # A dictionary containing all past strategies evolved along the way
    past_organisms = {}

    # A dicitonary for tracking total cooperates and defects commited by population by organism

    # Create initial population
    organisms = create_initial_population()

    # Prepare header for output data
    mem_output = []
    sum_output = []
    declen_output = []
    fitlen_output = []
    coop_output = []
    defect_output = []
    mem_headers = []
    sum_headers = []
    dec_headers = []
    fit_headers = []
    for i in range(CONST_CONF.MAX_BITS_OF_MEMORY + 1):
        mem_headers.append("Organisms With " + str(i) + " Bits of Memory")
        sum_headers.append("Organisms With " + str(i) + " Bits of Summary")
    for i in range(CONST_CONF.NUMBER_OF_ORGANISMS):
        dec_headers.append(f"Organism #{i}")
        fit_headers.append(f"Organism #{i}")
    mem_output.append(mem_headers)
    sum_output.append(sum_headers)
    declen_output.append(dec_headers)
    fitlen_output.append(fit_headers)
    coop_output.append(dec_headers)
    defect_output.append(fit_headers)

    # Adding each organism's strategy as keys
    # Each key holds a list of occurrences for the same strategy
    # IDs and parents are not considered due to PD org's __hash__ method;
    # organisms appended to the same list will have same strategy, but not IDs or parents. 
    for org in organisms:
        # If the strategy is encountered multiple times, its occurrence is appended to a list
        if org in past_organisms:
            past_organisms[org].append(org)
        # Otherwise, a new list containing the strategy is created
        else:
            past_organisms[org] = [org]

    # Create detail file for first generation
    # There should be no parent data at this point
    pd_make_detail_file.make_file_detail(organisms, past_organisms, 0, CONST_CONF.OUTPUT_FOLDER)
    
    for i in range(CONST_CONF.NUMBER_OF_GENERATIONS):
        # Static Mode  
        for org in organisms: org.clean_rates() 
        if CONST_CONF.SELECTION_BY_STATIC_COMPETITOR: 
            organisms = pd_selection.get_next_generation_by_static_payout(organisms)
        # Coevolutionary Mode
        else: 
            organisms = pd_selection.get_next_generation_by_selection(organisms)
        # Mutate populataion
        organisms = get_mutated_population(organisms)

        # Calculates, for each generation, the count of organisms with memory lengths 
        # spanning from 0 to MAX_BITS_OF_MEMORY + 1. 
        mem_output.append(get_tally_of_number_of_bits_of_memory(organisms))
        if organisms[-1].genotype.type() == 'hybrid':
            sum_output.append(get_tally_of_number_of_bits_summary(organisms))
        declen_output.append([len(org.genotype.decision_list) for org in organisms])
        fitlen_output.append([org.fitness() for org in organisms])
        #coop_output.append([f"{org.genotype.decision_list}~{org.genotype.initial_memory}" for org in organisms])
        coop_output.append([org.coops / org.num_games if org.num_games > 0.0 else 0.0 for org in organisms])
        
        defect_output.append([f"{org.genotype.decision_list}~{org.genotype.initial_memory}" for org in organisms])
        defect_output.append([org.defects / org.num_games if org.num_games > 0.0 else 0.0 for org in organisms])
        

        # Adding more into existing dictionary
        # expect past_organisms to grow over time as newer strategies are discovered.
        for org in organisms:
            # org_tracker = update_tracker(org_tracker, i, org, 'Coops')
            # org_tracker = update_tracker(org_tracker, i, org, 'Defects')
            # if org in past_organisms:
            #     past_organisms[org].append(org) # This function call seems wildly redundant and inefficient
            # else:
            past_organisms[org] = [org]
        
        # Make detail file every OUTPUT_FREQUENCY generations
        if ( (i + 1) % CONST_CONF.OUTPUT_FREQUENCY == 0):
            pd_make_detail_file.make_file_detail(organisms, past_organisms, i + 1, CONST_CONF.OUTPUT_FOLDER)
            
    # org_tracker_df = convert_and_pivot_org_tracker(org_tracker)
    # org_tracker_df.to_csv("org_tracker_pivoted.csv", index=False)
    return mem_output, sum_output, declen_output, fitlen_output, coop_output, defect_output

def get_average_fitness(pop, environment):
    """Gets average fitness of a population"""
    total = 0
    for org in pop:
        total += org.fitness(environment)
    return total / len(pop)

def save_table_to_file(table, filename):
    """Write a table to a file"""
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(table)

def save_string_to_file(string, filename):
    """Write a string to a file"""
    with open(filename, "w") as f:
        f.write(string)

def join_path(filename):
        return os.path.join(CONST_CONF.OUTPUT_FOLDER, filename)

def generate_data():
    """The main function; generates all the data"""
    # Create output folder for storing every component of the experiment
    if os.path.exists(CONST_CONF.OUTPUT_FOLDER):
        shutil.rmtree(CONST_CONF.OUTPUT_FOLDER)
        #raise IOError("output_folder: {} already exists, please select a new folder name".format(CONST_CONF.OUTPUT_FOLDER))
    os.makedirs(CONST_CONF.OUTPUT_FOLDER)
    
    mem_output, sum_output, declen_output, fitlen_output, coop_output, defect_output = pd_evolve_population()
    save_table_to_file(mem_output, join_path("bits_of_memory_overtime.csv"))
    save_table_to_file(declen_output, join_path("decision_list_length_aggregate.csv"))
    save_table_to_file(fitlen_output, join_path("fitness_aggregate.csv"))
    save_table_to_file(coop_output, join_path("coop_aggregate.csv"))
    save_table_to_file(defect_output, join_path("defect_aggregate.csv"))
    if CONST_CONF.ORG_TYPE == "hybrid":
        save_table_to_file(sum_output, join_path("bits_of_summary_overtime.csv"))
        
    time_filename = join_path("time.dat")     
    start_time = datetime.datetime.fromtimestamp(CONST_CONF.START_TIME)
    end_time = datetime.datetime.now()
    time_str = "Start_time {}\nEnd_time {}\nDuration {}\n".format(start_time, end_time, end_time - start_time)
    save_string_to_file(time_str, time_filename)
