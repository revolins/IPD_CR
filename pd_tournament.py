"""
This module takes two PD orgs and determines 
their score in an iterated Prisoner's Dilemma
"""

from config import CONST_CONF
from collections import defaultdict, Counter
import itertools
import numpy as np
import random
import time

def pd_payout(a_cooperates, b_cooperates):
    """
    Function my_reward determines reward given by the state of self and other
        
    Another way to implement below code:
    if self_is_cooperator:
        if other_is_cooperator:
            return reward
        return sucker
    if other_is_cooperator:
        return temptation
    return punishment
    """
    # a_cooperates and b_cooperates are determined by PDOrg's will_cooperate  
    if a_cooperates and b_cooperates:
        return CONST_CONF.REWARD, CONST_CONF.REWARD
    elif a_cooperates and not b_cooperates:
        return CONST_CONF.SUCKER, CONST_CONF.TEMPTATION
    elif not a_cooperates and b_cooperates:
        return CONST_CONF.TEMPTATION, CONST_CONF.SUCKER
    elif not a_cooperates and not b_cooperates:
        return CONST_CONF.PUNISHMENT, CONST_CONF.PUNISHMENT
    raise AssertionError("Impossible To Reach End of PD Payout")

# def longest_block(s):
#     substr_freq = defaultdict(int)
#     i = 0

#     while i < len(s):
#         start = i
#         first_char = s[i]

#         while i < len(s) and s[i] == first_char:
#             i += 1
#         mid = i

#         if mid < len(s):
#             second_char = s[mid]
#             while i < len(s) and s[i] == second_char:
#                 i += 1
#             end = i
#             block = s[start:end]
#             substr_freq[block] += 1
#         else:
#             break

#     if not substr_freq:
#         return '1' if s and s[-1] == 'T' else '0'

#     max_len = max(len(k) for k in substr_freq)

#     max_len_blocks = {k: v for k, v in substr_freq.items() if len(k) == max_len}

#     most_freq_block = max(max_len_blocks.items(), key=lambda x: x[1])[0]
#     return most_freq_block

def most_frequent_8block(s, min_len=8, max_len=8):
    freq = defaultdict(int)
    max_len_seen = 0

    for win_size in range(min_len, max_len + 1):
        for i in range(0, len(s) - win_size + 1, win_size):
            substr = s[i:i + win_size]
            freq[substr] += 1
            max_len_seen = max(max_len_seen, len(substr))

    longest_subs = {k: v for k, v in freq.items() if len(k) == max_len_seen}

    max_freq = max(longest_subs.values(), default=0)
    most_frequent = [k for k, v in longest_subs.items() if v == max_freq][0]

    return most_frequent, max_freq

def track_moves(move, phenotype):
    if move: phenotype += 'T'
    else: phenotype += 'F'

    return phenotype

def track_behavior(organism_a, organism_b, org_a_dec, org_b_dec, i):
    print(f"------------- {i} -------------")
    print("Organism A DL: ", organism_a.genotype.decision_list)
    print("Organism B DL: ", organism_b.genotype.decision_list)
    if organism_a.memory is not None:
        print("Organism A Memory: ", organism_a.memory)
    else: print("Organism A Memory: ", organism_a.genotype.initial_memory)
    if organism_b.memory is not None:
        print("Organism B Memory: ", organism_b.memory)
    else: print("Organism B Memory: ", organism_b.genotype.initial_memory)
    print("Organism A Decision: ", org_a_dec)
    print("Organism B Decision: ", org_b_dec, flush=True)

def cooperation_rate(org):
    return org.count('T') / len(org)

def defect_rate(org):
    return org.count('F') / len(org)

def retaliation_rate(me, opponent):
    defect_after_defect = 0
    count_defect = 0
    total_moves = len(me)

    for i in range(1, total_moves):
        if opponent[i - 1] == 'F':
            count_defect += 1
            if me[i] == 'F':
                defect_after_defect += 1

    return defect_after_defect / total_moves if count_defect else 0

def reciprocity_rate(me, opponent):
    return sum(1 for m, o in zip(me, opponent) if m == o) / len(me)

def grudge_rate(me, opponent):
    defect_after_cooperate = 0
    count_coop = 0
    total_moves = len(me)

    for i in range(1, total_moves):
        if opponent[i - 1] == 'T':
            count_coop += 1
            if me[i] == 'F':
                defect_after_cooperate += 1

    return defect_after_cooperate / total_moves if count_coop else 0

def forgiveness_rate(me, opponent):
    cooperate_after_defect = 0
    count_defect = 0
    total_moves = len(me)

    for i in range(1, total_moves):
        if opponent[i - 1] == 'F':
            count_defect += 1
            if me[i] == 'T':
                cooperate_after_defect += 1

    return cooperate_after_defect / total_moves if count_defect else 0


def run_game(organism_a, organism_b):
    """
    Run a game of NUMBER OF ROUNDS long
    Return payout for both organisms
    """
    organism_a.initialize_memory()
    organism_b.initialize_memory()
    organism_a.num_games += 1
    organism_b.num_games += 1
    
    total_payout_a = 0
    total_payout_b = 0
    if CONST_CONF.RANDOMIZED_ROUNDS:
        # Size 500 to match NUMBER_OF_GENERATIONS, scale=3 for full scope of std. dev., mean set to NUMBER_OF_ROUNDS
        curr_num_rounds = np.random.choice(list(np.random.default_rng(seed=CONST_CONF.SEED).normal(loc=CONST_CONF.NUMBER_OF_ROUNDS, scale=3, size=500).astype(int)))
        if curr_num_rounds <= 0: raise AssertionError(f"Number of Randomized Rounds - {curr_num_rounds} is too low, scale or loc requires adjustment")
    else:
        curr_num_rounds = CONST_CONF.NUMBER_OF_ROUNDS
    
    org_a_phen = ""
    org_b_phen = ""
    for i in range(curr_num_rounds):
        # Decisions from a and b
        # if CONST_CONF.PROB_ORG:
        #     print("PROBABILISTIC ORGANISM")
        #     a_cooperates = organism_a.will_cooperate_prob()
        #     b_cooperates = organism_b.will_cooperate_prob()
        # else:
        if CONST_CONF.HARD_DEFECT != None and i in CONST_CONF.HARD_DEFECT:
            a_cooperates = False
        else:
            a_cooperates = organism_a.will_cooperate()
        
        b_cooperates = organism_b.will_cooperate()
        #track_behavior(organism_a, organism_b, a_cooperates, b_cooperates, i)
        org_a_phen = track_moves(a_cooperates, org_a_phen)
        org_b_phen = track_moves(b_cooperates, org_b_phen)

        if random.random() < CONST_CONF.NOISE:
            noisy_decision = np.random.choice([0, 1])

            if noisy_decision == 0: a_cooperates = not a_cooperates
            else: b_cooperates = not b_cooperates

        # Resulting payout from these decisions
        payout_a, payout_b = pd_payout(a_cooperates, b_cooperates)
        
        # Organisms retain memory of their own moves and those of their opponents.
        if CONST_CONF.TOGGLE_SELF_MEMORY_ON:
            organism_a.store_bit_of_memory(a_cooperates) # Memory is not in genotype
            organism_a.store_bit_of_memory(b_cooperates)
            organism_b.store_bit_of_memory(b_cooperates)
            organism_b.store_bit_of_memory(a_cooperates)
        # Opponent moves only
        else:
            organism_a.store_bit_of_memory(b_cooperates)
            organism_b.store_bit_of_memory(a_cooperates)
    
        total_payout_a += payout_a
        total_payout_b += payout_b

    #org_a_strat = longest_block(org_a_phen)
    #org_b_strat = longest_block(org_b_phen)
    org_a8_strat = most_frequent_8block(org_a_phen)
    org_b8_strat = most_frequent_8block(org_b_phen)
    org_a_list = list(org_a_phen)
    org_b_list = list(org_b_phen)
    # print("Organism A: ", org_a_strat)
    # print("Organism B: ", org_b_strat)
    # print("Organism A: ", org_a8_strat)
    # print("Organism B: ", org_b8_strat)
    # print("Organism A Moves: ", org_a_phen)
    # print("Organism B Moves: ", org_b_phen)
    # print("------ Cooperates And Defects ------")
    # print("A - Cooperation Rate:", average_cooperation_rate(org_a_list))
    # print("A - Defect Rate:", average_defect_rate(org_a_list))
    # print("B - Cooperation Rate:", average_cooperation_rate(org_b_list))
    # print("B - Defect Rate:", average_defect_rate(org_b_list))
    # print("------------ A vs. B ------------")
    # print("Retaliation Rate:", retaliation_rate(org_a_list, org_b_list))
    # print("Reciprocity Rate:", reciprocity_rate(org_a_list, org_b_list))
    # print("Grudge Rate:", grudge_rate(org_a_list, org_b_list))
    # print("Forgiveness Rate:", forgiveness_rate(org_a_list, org_b_list))
    # print("------------ B vs. A ------------")
    # print("Retaliation Rate:", retaliation_rate(org_b_list, org_a_list))
    # print("Reciprocity Rate:", reciprocity_rate(org_b_list, org_a_list))
    # print("Grudge Rate:", grudge_rate(org_b_list, org_a_list))
    # print("Forgiveness Rate:", forgiveness_rate(org_b_list, org_a_list))
    organism_a.update_rate({'coop': cooperation_rate(org_a_list), 'defect': defect_rate(org_a_list), 'recipro': reciprocity_rate(org_a_list, org_b_list),
                   'grudge': grudge_rate(org_a_list, org_b_list), 'forgive': forgiveness_rate(org_a_list, org_b_list), '8cycle_strat' : org_a8_strat})
    organism_b.update_rate({'coop': cooperation_rate(org_b_list), 'defect': defect_rate(org_b_list), 'recipro': reciprocity_rate(org_b_list, org_a_list),
                   'grudge': grudge_rate(org_b_list, org_a_list), 'forgive': forgiveness_rate(org_b_list, org_a_list), '8cycle_strat' : org_b8_strat})
    # Stored moves are changed back to initial memory (taken from genotype)
    # Necessary to start new game, because organisms are shuffled
    organism_a.initialize_memory()
    organism_b.initialize_memory()
    
    return total_payout_a, total_payout_b

def adjusted_payout(organism_a, organism_b):
    """
    Returns adjusted payout reward (applied cost) for both organisms
    """
    def proportion_cost(org):
        if org.genotype.type() == 'hybrid':
            return CONST_CONF.PROPORTION_COST_PER_MEMORY_BIT * (org.genotype.number_of_bits_of_memory + org.genotype.number_of_bits_of_summary)
        else:
            return CONST_CONF.PROPORTION_COST_PER_MEMORY_BIT * org.genotype.number_of_bits_of_memory
    
    def get_adjusted_payout(payout, proportion_cost):
        
        """Apply cost to payout; fitness function"""
        return payout * (1 - proportion_cost)

    # Total payout for both organisms in a game (64 rounds) 
    payout_a, payout_b = run_game(organism_a, organism_b)

    a_proportion_cost = proportion_cost(organism_a)
    b_proportion_cost = proportion_cost(organism_b)
    
    adj_payout_a = get_adjusted_payout(payout_a, a_proportion_cost)
    adj_payout_b = get_adjusted_payout(payout_b, b_proportion_cost)
   
    return adj_payout_a, adj_payout_b

def vectorized_combinations(n):
        a, b = np.triu_indices(n, k=1)
        return np.stack([a, b], axis=1)

def get_average_payouts(organisms):
    """    
    COEVOLUTIONARY MODE
    Calculates the average payouts of all organisms in the list 
    (most likely contenders in a tournament).  
    Lists all possible pairs of organisms, calls adj_payout
    Averages all together
    Updates organisms.average_payout for every org in organisms list
    """
    total_payouts = [0.0 for _ in organisms] #Init payout

    # Generate all possible pairs of organisms
    # Ensures that each pair of organisms interact exactly once
    all_pairs = itertools.combinations(range(len(organisms)), 2)
    #print(list(all_pairs))
    
    #all_pairs = vectorized_combinations(10)
    
    # all_pairs = list(all_pairs)
    # if len(all_pairs) != 45: 
    #     print(len(list(all_pairs)))
    #     exit()
    # all_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
    # (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), 
    # (1, 7), (1, 8), (1, 9), (2, 3), (2, 4), (2, 5), (2, 6), 
    # (2, 7), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 7), 
    # (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), 
    # (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]
    # python main.py --m_c 0.0 -o temp_test --max_m 4 --imb 1 --org_type summary --org_seed_per 1.0

    # python main.py --m_c 0.0 -o temp_test --max_m 4 --imb 1 --org_type summary
    #[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
    # (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), 
    # (1, 7), (1, 8), (1, 9), (2, 3), (2, 4), (2, 5), (2, 6), 
    # (2, 7), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 7), 
    # (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), 
    # (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]

    # python main.py --m_c 0.0 -o temp_test --max_m 4 --imb 1 --org_type pd
    # [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
    # (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), 
    # (1, 7), (1, 8), (1, 9), (2, 3), (2, 4), (2, 5), (2, 6), 
    # (2, 7), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 7), 
    # (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), 
    # (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]

    # python main.py --m_c 0.0 -o temp_test --max_m 4 --imb 1 --org_type pd --ts 8
    # [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
    # (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), 
    # (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), 
    # (3, 7), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
    # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    # Store all orgs in an array and then every organism plays with the preceding 25 organisms and the subsequent 25 organisms in the array
    
    for i, j in all_pairs:
        org_a = organisms[i]
        org_b = organisms[j]
        payout_a, payout_b = adjusted_payout(org_a, org_b)
        total_payouts[i] += payout_a
        total_payouts[j] += payout_b

    # Number of opponents each organism competes with, or number of games it participates in     
    number_of_games_per_org = len(organisms) - 1
    # # Averaging payout for each organism
    # #print(f"TOTAL PAYOUTS {total_payouts}", flush=True)
    average_payouts = [payout / number_of_games_per_org for payout in total_payouts] 
    
    # # Update each organism's average_payout attribute
    for i in range(len(organisms)):
        organisms[i].average_payout = average_payouts[i]
    

def get_static_payouts(organisms, static_competitors):
    """
    STATIC MODE 
    Get average payouts for a list of organisms.
    """

    for org in organisms:
        # Update attribute directly
        org.average_payout = get_static_fitness(org, static_competitors)

def get_static_fitness(org, static_competitors):
    """
    STATIC MODE
    Gets fitness for a single organism against a group of fixed opponents
    """
    # Adjusted payouts for each game between the org and each opponent
    payouts = [adjusted_payout(org, comp)[0] for comp in static_competitors]

    return sum(payouts) / (float(len(payouts)))