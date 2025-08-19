"""
This module contains methods for selecting the next generation.
"""

import pd_tournament
import random
import template_org as to
from genotype_factory import PDGenotypeFactory as tg
from config import CONST_CONF
from time import time

# Hard-coded static competitors for baseline testing
MAX_BITS_OF_MEMORY = 1
ALL_DEFECT = to.PDOrg(tg.create_genotype("memory", number_of_bits_of_memory=0, decision_list=[False], initial_memory=[]))
TIT_FOR_TAT = to.PDOrg(tg.create_genotype("memory", number_of_bits_of_memory=1, decision_list=[False, True], initial_memory=[True]))
COIN_FLIP = to.PDStochasticOrg()
STATIC_COMPETITORS = [ALL_DEFECT, TIT_FOR_TAT, COIN_FLIP]

# Parameter is set through set_global_variables() in main.py
# This determines the number of organisms per tournament - TOURNAMENT_SIZE = None

def get_best_half(organisms):
    """Returns a list of the top half of the organisms in terms of payout"""
    # Sort organisms based on payout in descending order
    sorted_orgs = sorted(organisms, key=lambda org: org.average_payout, reverse=True)
    # Select best half of sorted organisms
    best_half_orgs = sorted_orgs[:len(organisms) // 2]
    # Return copy of selected best half
    return best_half_orgs[:]

def get_number_of_tournaments(organisms):
    """
    Calculate number of tournaments needed given a list of organisms 
    and predefined tournament size.
    """
    # Determine number of full tournaments
    number_of_tournaments = len(organisms) // CONST_CONF.TOURNAMENT_SIZE
    # Check for remaining organisms
    if len(organisms) % CONST_CONF.TOURNAMENT_SIZE:
        # One additional tournament for remaining organisms
        number_of_tournaments += 1
    return number_of_tournaments

def get_contender_generator(organisms):
    """
    Given a list of organisms, returns generator function that yields batches of
    contenders for tournaments.
    """
    number_of_tournaments = get_number_of_tournaments(organisms)  

    def generate_contenders(organisms):
        """Shuffle the organisms then group them into TOURNAMENT_SIZEd clumps"""
        while True:
            # Give organisms list a random order
            random.shuffle(organisms)
            # Yield contenders for each tournament
            for i in range(number_of_tournaments):
                # Slicing ensures each tournament has the right number of organisms
                # Yield "returns" each batch one at a time, not all at once
                yield organisms[CONST_CONF.TOURNAMENT_SIZE * i: CONST_CONF.TOURNAMENT_SIZE * (i + 1)]    
    return generate_contenders(organisms)

def get_next_generation_by_selection(organisms):
    """
    COEVOLUTIONARY MODE
    Conducts tournaments and calculate average payout.
    Calls _get_next_generation() to select organisms for the next generation.
    """
    # Determine number of tournaments to run
    number_of_tournaments = get_number_of_tournaments(organisms)

    # Define generator to yield contenders
    contender_generator = get_contender_generator(organisms)

    # In each tournament
    for _ in range(number_of_tournaments):
        # Obtain next batch of contenders
        contenders = next(contender_generator)
        # Get average payouts for this tournament, update organism attributes
        pd_tournament.get_average_payouts(contenders)

    # All organisms should have updated average payouts by this point
    # Select organisms for the next generation
    return _get_next_generation(organisms, contender_generator)


def _get_next_generation(organisms, contender_generator):
    """Selects organisms for the next generation"""
    # Next generation's population
    next_generation = []
    
    # Until the next generation reaches the population size
    while len(next_generation) < len(organisms):
        # Get batch of contenders
        contenders = next(contender_generator)
        # Get best half of contenders
        winners = get_best_half(contenders)
        # Add winners to the next generation
        next_generation += winners

    # Ensure next_generation has the same length as organisms
    return next_generation[:len(organisms)]
    
def get_next_generation_by_static_payout(organisms):
    """
    STATIC MODE
    Calculate average payouts against non-evolving opponents.
    Calls _get_next_generation() to select organisms for the next generation.
    """
    # Get payouts from playing against static competitors
    pd_tournament.get_static_payouts(organisms, STATIC_COMPETITORS)
    # Set up contender generator
    contender_generator = get_contender_generator(organisms)
    # Select organisms for the next generation
    return _get_next_generation(organisms, contender_generator)
