"""
This module creates detail files for each generation.
Detail files contain information on each PD org within a generation. 
"""

import csv
import numpy as np

def determine_strat(raw_strat):
    defs = raw_strat.count('F')
    coops = raw_strat.count('T')
    if defs == coops:
        return 'TFT-Variant'
    elif defs == 0:
        return 'AC'
    elif coops == 0:
        return 'AD'
    elif defs > coops:
        return 'D-Emerg'
    elif coops > defs:
        return 'C-Emerg'
    else:
        return 'GHOST'

# Master function to call all the helpers
def make_file_detail(organisms, past_organisms, current_generation, filepath):
    """
    Create detail file for the current generation. 
    For each organism, include:
    - Length or number of bits of initial (specific) memory 
    - Decision list
    - Initial (specific) memory list
    - Number of orgs alive with the same strategy in the current generation
    - ID
    - Parent ID
    - Decision List Length
    """
    # Print out header for detail file that tells whats in it
    # Print as a csv
    # To print data -- go through all dictionary keys, see strat, print output line: decision list, init memory, bits of mem
    # Contain number of orgs alive with that strat look at cur gen
    # Print out id of orgs
    # Print out id of parents


    # Create csv writer
    filename = filepath + '/detail-' + str(current_generation) + '.csv'
    if organisms[-1].genotype.type() == 'hybrid':
        header = ['MemBits' , 'SumBits' , 'Decisions' , 'Memory' , 'Summary' , 'LiveFitness', 'Alive' , 'Id' , 'ParentId', 'Cooperates', 'Defects', 'Cooperations', 'Defections', 'Reciprocate', 'Grudge', 'Forgiveness', 'Strategy', 'DecLength']
    else: header = ['Bits' , 'Decisions' , 'Memory' , 'LiveFitness', 'Alive' , 'Id' , 'ParentId', 'Cooperates', 'Defects', 'Cooperations', 'Defections', 'Reciprocate', 'Grudge', 'Forgiveness', 'Strategy', 'DecLength']

    # Put data where we want it
    data = []
    
    # Iterate through everything in dictionary
    for key in past_organisms:
        # Count number of organisms alive with the same strategy
        number_alive = 0
        temp_fitness = []

        for org in organisms:
            if hash(key) == hash(org):
                number_alive += 1
                temp_fitness.append(org.fitness())
        row = []
        row.append(key.genotype.number_of_bits_of_memory)

        if organisms[-1].genotype.type() == 'hybrid':
            row.append(key.genotype.number_of_bits_of_summary)

        row.append(key.genotype.decision_list)
        row.append(key.genotype.initial_memory)

        if organisms[-1].genotype.type() == 'hybrid':
            row.append(key.genotype.initial_summary)

        if number_alive == 0: 
            row.extend([0.0, 0, '', '', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '', ''])
            data.append(row)
            continue

        temp_coops, temp_defects = '', ''
        avg_coop_rate, avg_def_rate, temp_friend, temp_recip, temp_grudge, temp_forgive = 0.0,0.0,0.0,0.0,0.0,0.0
        
        if not temp_fitness: temp_fitness = [0.0]
        row.append(np.max(temp_fitness))
        row.append(number_alive)
        
        row.append([org.id for org in past_organisms[key]])
        row.append([org.parent for org in past_organisms[key]])
        for org in past_organisms[key]:
            temp_coops += str(org.coops / org.num_games if org.num_games > 0.0 else 0.0) + ' '
            temp_defects += str(org.defects / org.num_games if org.num_games > 0.0 else 0.0) + ' '
            avg_def_rate = np.nanmean(org.def_rate) if org.def_rate != [] else 0.0
            avg_coop_rate = np.nanmean(org.coop_rate)if org.coop_rate != [] else 0.0
            temp_recip = np.nanmean(org.recipro) if org.recipro != [] else 0.0
            temp_grudge = np.nanmean(org.grudge) if org.grudge != [] else 0.0
            temp_forgive = np.nanmean(org.forgive) if org.forgive != [] else 0.0
            if org.strategy != []:
                curr_strats = [determine_strat(org_strat[0])  for org_strat in org.strategy] 
            else: curr_strats = ['GHOST']
            
        row.append(temp_coops)
        row.append(temp_defects)
        row.append(avg_coop_rate) # lol here was the bug
        row.append(avg_def_rate)
        row.append(temp_recip)
        row.append(temp_grudge)
        row.append(temp_forgive)
        row.append(curr_strats)
        row.append([len(key.genotype.decision_list)])
        data.append(row)

    # Creates csv file
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

