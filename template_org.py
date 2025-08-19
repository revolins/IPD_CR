import random
import numpy as np
from genotype_factory import PDGenotypeFactory as tg

from collections import deque, defaultdict
from config import CONST_CONF
from abc import ABC, abstractmethod

class AbstractOrg(ABC):
    """
    Abstract Organism class for inheritance of all other types of organisms
    """

    #Functions that do not change across organisms and aren't required in subcless
    def __eq__(self, other):
        """Overload equality operator based on genotype"""
        return self.genotype == other.genotype
    
    def __ne__(self, other):
        """Overload not equal operator"""
        return not self == other
    
    def __repr__(self):
        """In this case, the same as __str__"""
        return str(self)
    
    def __hash__(self):
        """Overload hash operator with that of genotype"""
        return hash(self.genotype)
    
    def dec_list(self):
        return self.genotype.decision_list

    # Abstract methods are required and change across organisms
    @abstractmethod
    def __init__(self): pass

    @abstractmethod
    def get_mutant(self): pass

    @abstractmethod
    def __str__(self): pass

    @abstractmethod
    def will_cooperate(self): pass

    @abstractmethod
    def initialize_memory(self): pass

    @abstractmethod
    def _create_random_genotype(self): pass

    @abstractmethod
    def _create_rtft_genotype(self): pass

    def update_rate(self, rate_dict):
        """
        Handle dictionary to update list of organism rate values
        """
        self.coop_rate.append(rate_dict['coop'])
        self.def_rate.append(rate_dict['defect'])
        self.recipro.append(rate_dict['recipro'])
        self.grudge.append(rate_dict['grudge'])
        self.forgive.append(rate_dict['forgive'])
        self.strategy.append(rate_dict['8cycle_strat'])

    # 'Hooks' that aren't required in subclasses but can change across organisms
    def store_bit_of_memory(self, did_cooperate):
        """
        Stores opponent's last move in memory at the right end of memory and
        deletes oldest move (on left)
        """
        self.memory.append(did_cooperate)
        self.memory.popleft()

    def fitness(self):
        if self.average_payout != None:
            return self.average_payout
        return 0.0

class PDOrg(AbstractOrg):
    """
    Concrete classes have to implement all abstract operations of the base
    class. They can also override some operations with a default implementation.
    """

    next_org_id = 0
    
    def __init__(self, genotype=None, parent=None, gen_str=None):
        if genotype is None and gen_str is None:
            genotype = self._create_random_genotype()
        if genotype is None and gen_str == 'rtft':
            genotype = self._create_rtft_genotype()
        self.genotype = genotype
        self.initialize_memory()
        self.id = PDOrg.next_org_id
        PDOrg.next_org_id += 1
        self.parent = parent
        self.average_payout = None
        self.payout = None
        self.coops = 0
        self.defects = 0
        self.num_games = 0
        self.def_rate = []
        self.coop_rate = []
        self.friend = []
        self.recipro = []
        self.grudge = []
        self.forgive = []
        self.strategy = []
        self.long_strat = []
        
    def get_mutant(self):
        """Get mutated version of self"""
        return PDOrg(self.genotype.get_mutant_of_self(), self.id)

    def __str__(self):
        """String representation"""
        return "PDOrg({})".format(self.genotype)
    
    def will_cooperate(self):
        """
        Returns True if organism will cooperate, else False for defection
        
        First convert self.memory to a binary string ("101")
        Then, convert binary string to integer (5)
        Return value of decision list at index
        """
        if not self.memory:
            decision_list_index = 0
        else:
            binary_string_index = "".join("1" if i else "0" for i in self.memory)
            decision_list_index = int(binary_string_index, 2)

        return self.genotype.decision_list[decision_list_index]
    
    def initialize_memory(self):
        self.memory = deque(self.genotype.initial_memory)
        #self.memory = np.array(self.genotype.initial_memory)

    def _create_random_genotype(self):
        """
        Creates random memory PD genotype
        
        Used by PDOrg as default returned genotype
        """
        number_of_bits_of_memory = random.randrange(CONST_CONF.MAX_BITS_OF_MEMORY + 1)
        if CONST_CONF.INITIAL_MEMORY_BITS != 0: number_of_bits_of_memory = CONST_CONF.INITIAL_MEMORY_BITS
        length = 2 ** number_of_bits_of_memory
        decision_list = [random.choice([True, False]) for _ in range(length)]
        initial_memory = [random.choice([True, False]) for _ in range(number_of_bits_of_memory)]
        return tg.create_genotype('memory', number_of_bits_of_memory, decision_list, initial_memory)
    
    def clean_rates(self):
        del self.def_rate
        del self.coop_rate
        del self.friend
        del self.recipro
        del self.grudge
        del self.forgive
        del self.strategy

        self.def_rate = []
        self.coop_rate = []
        self.friend = []
        self.recipro = []
        self.grudge = []
        self.forgive = []
        self.strategy = []
        #self.long_strat = []
    
    def _create_rtft_genotype(self):
        """
        Creates reactive-TFT variant memory PD genotype
        
        Used by PDOrg as TFT returned genotype
        """
        # Using "memory" for simplicity, there is only one memory "type"
        number_of_bits_of_memory = random.randrange(CONST_CONF.MAX_BITS_OF_MEMORY + 1)
        if CONST_CONF.INITIAL_MEMORY_BITS != 0: number_of_bits_of_memory = CONST_CONF.INITIAL_MEMORY_BITS
        length = number_of_bits_of_memory + 1
        decision_list = [False, True]
        initial_memory = [True]
        return tg.create_genotype('memory', number_of_bits_of_memory, decision_list, initial_memory)

class PDStochasticOrg(PDOrg):
    """
    This class creates a PDStochastic organism.
    A PD organism consists of a genotype, ID, parent, and average payout. 
    Reliant on PDOrg and will pass through most member functions
    """
    next_org_id = 0

    def __init__(self, genotype=None, parent=None, gen_str=None):
        if genotype is None and gen_str is None:
            genotype = tg.create_genotype('stochastic')
        self.genotype = genotype
        self.id = PDStochasticOrg.next_org_id
        PDStochasticOrg.next_org_id += 1
        self.parent = parent
        self.average_payout = None
        self.payout = None
    
    def get_mutant(self):
        new_genotype = random.random()
        return PDStochasticOrg(new_genotype, self.id)
    
    def _str_(self):
        return "PDStochasticOrg({})".format(self.genotype)

    def will_cooperate(self):
        return self.genotype.probability > random.random()

    def store_bit_of_memory(self, did_cooperate):
        pass
    
    def initialize_memory(self):
        pass

class HybridPDOrg(AbstractOrg):
    """
    Usually, concrete classes override only a fraction of base class'
    operations.
    """

    next_org_id = 0
    
    def __init__(self, genotype=None, parent=None, gen_str=None):
        if genotype is None and gen_str is None:
            genotype = self._create_random_genotype()
        # if genotype == 'coop':
        #     genotype = _create_friend_genotype()
        # if genotype == 'hostile':
        #     genotype = _create_hostile_genotype()
        self.genotype = genotype
        self.memory = None
        self.initialize_memory()
        self.id = HybridPDOrg.next_org_id
        HybridPDOrg.next_org_id += 1
        self.parent = parent
        self.average_payout = None
        self.payout = None
        self.coops = 0
        self.defects = 0
        self.num_games = 0
        self.strategy = []
        
    def get_mutant(self):
        """Get mutated version of self"""
        return HybridPDOrg(self.genotype.get_mutant_of_self(), self.id)

    def __str__(self):
        """String representation"""
        return "HybridPDOrg({})".format(self.genotype)
    
    def will_cooperate(self):
        """
        Returns True if organism will cooperate, else False for defection
        """
        # No specific or summary memory
        if not self.memory:
            decision_list_index = 0 # organism will have a decision list of size 1

        # At least specific or summary memory
        else:
            # Length of initial specific memory (k)
            len_memory = self.genotype.number_of_bits_of_memory
            assert len(self.genotype.initial_memory) == self.genotype.number_of_bits_of_memory

            # If specific memory exists
            if len_memory > 0:
                # Convert specific memory into binary string (True: 1, False: 0)
                binary_string = "".join("1" if i else "0" for i in list(self.memory)[:len_memory])
                # Convert binary string into integer
                binary_index = int(binary_string, 2)

                # Count number of cooperate (True) moves in summed memory
                # summary_index is 0 if summed memory is empty
                summary_index = sum(1 for i in list(self.memory)[len_memory:] if i==True)

                # Which "block" does binary_index belong to?
                # If summary memory doesn't exist, works like PDOrg
                decision_list_index = binary_index + (2 ** len_memory) * summary_index

            # If specified memory doesn't exist, summary memory has to at least exist
            else:
                assert len(self.genotype.initial_summary) > 0

                # In this case, decision list has size (j+1)
                decision_list_index = sum(1 for i in self.memory if i==True)

                assert decision_list_index <= (self.genotype.number_of_bits_of_summary)

        return self.genotype.decision_list[decision_list_index]
    
    def initialize_memory(self):
        """Get double-ended queue memory"""
        self.memory = deque(self.genotype.initial_memory + self.genotype.initial_summary)
    
    def _create_random_genotype(self):
        """
        Creates random memory PD genotype
        
        Used by HybridPDOrg as default returned genotype
        """
        # randrange generate number in range [0, MAX_BITS_OF_MEMORY] = total bits of mem
        # get random index to make a random split
        # one of the lists may or may not exist
        # This implies that organisms have the option to use specific, summary memory, or both
        total_bits_of_memory = random.randrange(CONST_CONF.MAX_BITS_OF_MEMORY + 1)
        number_of_bits_of_memory = random.randrange(total_bits_of_memory + 1)
        number_of_bits_of_summary = total_bits_of_memory - number_of_bits_of_memory

        assert number_of_bits_of_memory + number_of_bits_of_summary <= CONST_CONF.MAX_BITS_OF_MEMORY

        length = (2 ** number_of_bits_of_memory) * (number_of_bits_of_summary + 1)

        decision_list = [random.choice([True, False]) for _ in range(length)]
        initial_memory = [random.choice([True, False]) for _ in range(number_of_bits_of_memory)]
        initial_summary = [random.choice([True, False]) for _ in range(number_of_bits_of_summary)]
        return tg.create_genotype("hybrid", number_of_bits_of_memory, number_of_bits_of_summary, decision_list, initial_memory, initial_summary)

class SummaryPDOrg(AbstractOrg):
    next_org_id = 0

    def __init__(self, genotype=None, parent=None, gen_str=None):
        if genotype is None and gen_str is None:
            genotype = self._create_random_genotype()
        if genotype is None and 'rtft' in gen_str:
            genotype = self._create_rtft_genotype()
        if genotype is None and 'mr' in gen_str:
            genotype = self._create_mr_genotype()

        if genotype is None and 'lrg_nontft' in gen_str:
            genotype = self._create_large_nontft_genotype()
        if genotype is None and 'nontft' in gen_str:
            genotype = self._create_nontft_genotype()
        if genotype is None and 'rand' in gen_str:
            genotype = self._create_random_genotype()
        
        self.genotype = genotype
        self.initialize_memory()
        self.id = SummaryPDOrg.next_org_id
        SummaryPDOrg.next_org_id += 1
        self.parent = parent
        self.average_payout = None
        self.payout = None
        self.coops = 0
        self.defects = 0
        self.num_games = 0
        self.def_rate = []
        self.coop_rate = []
        self.friend = []
        self.recipro = []
        self.grudge = []
        self.forgive = []
        self.strategy = []
        self.long_strat = []
    
    def get_mutant(self):
        """Get mutated version of self"""
        return SummaryPDOrg(self.genotype.get_mutant_of_self(), self.id)

    def __str__(self): 
        """String representation"""
        return "SummaryPDOrg({})".format(self.genotype)
    
    def _track_action(self, decision):
        """
        Track Organism actions for storage as 'Cooperates' or 'Defects' in detail-*.csv
        """
        if decision: 
            self.coops += 1
        else: 
            self.defects += 1

    def will_cooperate(self):
        """
        Returns True if organism will cooperate, else False for defection
        
        Summary Memory Model makes decisions based on the number of previous cooperate moves in memory
        """
        if not self.memory:
            decision_list_index = 0
        else:
            #decision_list_index = np.sum(1 for i in np.array(self.memory) if i==True)
            decision_list_index = sum(1 for i in self.memory if i==True)

        self._track_action(self.genotype.decision_list[decision_list_index])

        return self.genotype.decision_list[decision_list_index]

    def clean_rates(self):
        del self.def_rate
        del self.coop_rate
        del self.friend
        del self.recipro
        del self.grudge
        del self.forgive
        del self.strategy

        self.def_rate = []
        self.coop_rate = []
        self.friend = []
        self.recipro = []
        self.grudge = []
        self.forgive = []
        self.strategy = []
        #self.long_strat = []

    def initialize_memory(self):
        self.memory = deque(self.genotype.initial_memory)

    def _create_random_genotype(self):
        """
        Creates random memory PD genotype
        
        Used by SummaryPDOrg as default returned genotype
        """
        # Using "memory" for simplicity, there is only one memory "type"
        number_of_bits_of_memory = random.randrange(CONST_CONF.MAX_BITS_OF_MEMORY + 1)
        if CONST_CONF.INITIAL_MEMORY_BITS != 0: number_of_bits_of_memory = CONST_CONF.INITIAL_MEMORY_BITS
        length = number_of_bits_of_memory + 1
        decision_list = [random.choice([True, False]) for _ in range(length)]
        initial_memory = [random.choice([True, False]) for _ in range(number_of_bits_of_memory)]
        return tg.create_genotype('summary', number_of_bits_of_memory, decision_list, initial_memory)
    
    def _create_rtft_genotype(self):
        """
        Creates reactive-TFT variant memory PD genotype
        
        Used by SummaryPDOrg as default returned genotype
        """
        # Using "memory" for simplicity, there is only one memory "type"
        number_of_bits_of_memory = 1
        decision_list = [False, True]
        initial_memory = [True]
        return tg.create_genotype('summary', number_of_bits_of_memory, decision_list, initial_memory)
    
    def _create_mr_genotype(self):
        """
        Creates majority-vote variant memory PD genotype
        
        Used by SummaryPDOrg as default returned genotype
        """
        # Using "memory" for simplicity, there is only one memory "type"
        number_of_bits_of_memory = 3
        decision_list = [False, False, True, True]
        initial_memory = [False, True, True]
        return tg.create_genotype('summary', number_of_bits_of_memory, decision_list, initial_memory)
    
    def _create_large_nontft_genotype(self):
        """
        Creates AD variant memory PD genotype
        
        Used by SummaryPDOrg as default returned genotype
        """
        # Using "memory" for simplicity, there is only one memory "type"
        number_of_bits_of_memory = 3
        decision_list = [False, False, False, False]
        initial_memory = [False, False, False]
        return tg.create_genotype('summary', number_of_bits_of_memory, decision_list, initial_memory)
    
    def _create_nontft_genotype(self):
        """
        Creates AD variant memory PD genotype
        
        Used by SummaryPDOrg as default returned genotype
        """
        # Using "memory" for simplicity, there is only one memory "type"
        number_of_bits_of_memory = 1
        decision_list = [False, False]
        initial_memory = [False]
        return tg.create_genotype('summary', number_of_bits_of_memory, decision_list, initial_memory)


