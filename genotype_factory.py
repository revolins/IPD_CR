
import random
from config import CONST_CONF
from abc import ABC, abstractmethod

class PDGenotypeFactory:
    @staticmethod
    def create_genotype(genotype_type, *args, **kwargs):
        if genotype_type == 'stochastic':
            return StochasticPDGenotype(*args, **kwargs)
        elif genotype_type == 'memory':
            return MemoryPDGenotype(*args, **kwargs)
        elif genotype_type == 'hybrid':
            return HybridPDGenotype(*args, **kwargs)
        elif genotype_type == 'summary':
            return SummaryPDGenotype(*args, **kwargs)
        else:
            raise ValueError(f"Unknown genotype type: {genotype_type}")

class PDGenotype(ABC):
    @abstractmethod
    def type(self):
        pass
    
    @abstractmethod
    def get_mutant_of_self(self):
        pass

class StochasticPDGenotype(object):
    """Genotype for static COIN FLIP/random type opponents in static environment"""
    def __init__(self, probability=.5, number_of_bits_of_memory=0):
        self.probability = probability
        self.number_of_bits_of_memory = number_of_bits_of_memory

    def type(self):
        return 'stochastic'


class MemoryPDGenotype(object):
    """
    Original Memory Model Genotype for inheriting in the PDOrg Class
    """

    def __init__(self, number_of_bits_of_memory, decision_list, initial_memory):
        assert 0 <= number_of_bits_of_memory <= CONST_CONF.MAX_BITS_OF_MEMORY
        assert len(decision_list) == 2 ** number_of_bits_of_memory
        assert len(initial_memory) == number_of_bits_of_memory
        self.number_of_bits_of_memory = number_of_bits_of_memory
        self.decision_list = decision_list
        self.initial_memory = initial_memory
    
    
    def __eq__(self, other):
        """Overload equality operator"""
        return (self.number_of_bits_of_memory == other.number_of_bits_of_memory and
                self.decision_list == other.decision_list and
                self.initial_memory == other.initial_memory)
    
    def __ne__(self, other):
        """Overload not equal operator"""
        return not self == other
    
    def __str__(self):
        """String representation"""
        return "MemoryPDGenotype({}, {}, {})".format(self.number_of_bits_of_memory,
                                                     self.decision_list,
                                                     self.initial_memory)
    
    def __repr__(self):
        """In this case, same as __str__"""
        return str(self)
        
    def __hash__(self):
        """Overload hash operator, necessary for dictionaries and such"""
        hashable_tuple = (self.number_of_bits_of_memory, 
            tuple(self.decision_list), 
            tuple(self.initial_memory))
        return hash(hashable_tuple)
    
    
    def type(self):
        """Return type as string for easy checking"""
        return 'memory'

    def get_mutant_of_self(self):
        """
        Determines when and how each type of mutation occurs.
        Returns new (mutated) genotype.
        """
        
        # Size mutation
        random_value = random.random()
        if random_value < CONST_CONF.MUTATION_LIKELIHOOD_OF_BITS_OF_MEMORY:
            return self._get_bits_of_memory_mutant()
        # Initial (specific) memory mutation
        if random_value < CONST_CONF.MUTATION_LIKELIHOOD_OF_BITS_OF_MEMORY + CONST_CONF.MUTATION_LIKELIHOOD_OF_INITIAL_MEMORY_STATE:
            return self._initial_memory_mutant()
        # Decision mutation
        return self._decision_list_mutant()


    def _get_bits_of_memory_mutant(self):
        """
        Increase or decrease length of initial memory by 1 bit.
        Affects length of decision list as well. 
        """
        should_increase_memory = random.choice([True, False])

         # If organism has no memory, don't decrease memory
        if self.number_of_bits_of_memory == 0 and not should_increase_memory:
            return self
        
        # If organism has maximum memory length, don't increase memory
        if self.number_of_bits_of_memory == CONST_CONF.MAX_BITS_OF_MEMORY and should_increase_memory:
            #Return full normal memory but hybrid relies on 2*k * (j+1)
            return self
        
        # If we increase memory length
        if should_increase_memory:
            new_number_of_bits_of_memory = self.number_of_bits_of_memory + 1 # (k)

            # Double list, duplicate decisions
            # Retain as much of the original pattern as possible, not sure if matters
            # Also try to mimic original mutation method as closely as possible
            new_decision_list = self.decision_list * 2

            # Add 1 extra bit to initial memory
            new_initial_memory = self.initial_memory[:]
            new_initial_memory.append(random.choice([True,False]))

            return MemoryPDGenotype(new_number_of_bits_of_memory, new_decision_list, new_initial_memory)
        
        # If we decrease memory length
        new_number_of_bits_of_memory = self.number_of_bits_of_memory - 1
        length_of_new_decision_list = len(self.decision_list) // 2

        # Update size of memory and decision lists, most distant past memory bits removed
        new_decision_list = self.decision_list[:length_of_new_decision_list]
        new_initial_memory = self.initial_memory[:-1]
        return MemoryPDGenotype(new_number_of_bits_of_memory, new_decision_list, new_initial_memory) 
        
    def _decision_list_mutant(self):
        """Randomly flip a single bit in decision list"""
        mutation_location = random.randrange(len(self.decision_list))
        new_decision_list = self.decision_list[:]
        new_decision_list[mutation_location] = not new_decision_list[mutation_location]
        return MemoryPDGenotype(self.number_of_bits_of_memory, new_decision_list, self.initial_memory)
        
    def _initial_memory_mutant(self):
        """
        Randomly flip a single bit in initial specified memory.
        This affects the state of memory the organism starts with.  
        """
        # If there is no memory, no change is made.
        if self.number_of_bits_of_memory == 0:
            return self
        
        # Mutate in specified memory
        mutation_location = random.randrange(len(self.initial_memory))
        new_initial_memory = self.initial_memory[:]
        new_initial_memory[mutation_location] = not new_initial_memory[mutation_location]
        return MemoryPDGenotype(self.number_of_bits_of_memory, self.decision_list, new_initial_memory)


class HybridPDGenotype(object):
    """
    Hybrid Memory Model Genotype for inheriting in the HybridPDOrg Class
    """

    def __init__(self, number_of_bits_of_memory, number_of_bits_of_summary, decision_list, initial_memory, initial_summary):
        assert 0 <= number_of_bits_of_memory + number_of_bits_of_summary <= CONST_CONF.MAX_BITS_OF_MEMORY
        assert len(decision_list) == (2 ** number_of_bits_of_memory) * (number_of_bits_of_summary + 1)
        assert len(initial_memory) == number_of_bits_of_memory
        assert len(initial_summary) == number_of_bits_of_summary

        self.number_of_bits_of_memory = number_of_bits_of_memory
        self.number_of_bits_of_summary = number_of_bits_of_summary
        self.decision_list = decision_list
        self.initial_memory = initial_memory
        self.initial_summary = initial_summary
    
    
    def __eq__(self, other):
        """Overload equality operator"""
        return (self.number_of_bits_of_memory == other.number_of_bits_of_memory and
                self.number_of_bits_of_summary == other.number_of_bits_of_summary and
                self.decision_list == other.decision_list and
                self.initial_memory == other.initial_memory and
                self.initial_summary == other.initial_summary)
    
    def __ne__(self, other):
        """Overload not equal operator"""
        return not self == other
    
    def __str__(self):
        """String representation"""
        return "HybridPDGenotype({}, {}, {}, {}, {})".format(self.number_of_bits_of_memory,
                                                     self.number_of_bits_of_summary,
                                                     self.decision_list,
                                                     self.initial_memory,
                                                     self.initial_summary)
    
    def __repr__(self):
        """In this case, same as __str__"""
        return str(self)
        
    def __hash__(self):
        """Overload hash operator, necessary for dictionaries and such"""
        hashable_tuple = (self.number_of_bits_of_memory, 
            self.number_of_bits_of_summary,
            tuple(self.decision_list), 
            tuple(self.initial_memory),
            tuple(self.initial_summary))
        return hash(hashable_tuple) # We don't consider IDs and parents, only strategy
    
    def type(self):
        """Return type as string for easy checking"""
        return 'hybrid'

    def get_mutant_of_self(self):
        """
        Determines when and how each type of mutation occurs.
        Returns new (mutated) genotype.
        """
        random_value = random.random()

        # Size mutation
        if random_value < CONST_CONF.MUTATION_LIKELIHOOD_OF_BITS_OF_MEMORY:
            #print("SIZE MUTATION")
            summary_or_memory = random.choice([True, False])
            if summary_or_memory:
                #print("MEMORY MUTATION")
                return self._get_bits_of_memory_mutant()
            else:
                #print("SUMMARY MUTATION")
                return self._get_bits_of_summary_mutant()
        # Initial (specific) memory mutation
        if random_value < CONST_CONF.MUTATION_LIKELIHOOD_OF_BITS_OF_MEMORY + CONST_CONF.MUTATION_LIKELIHOOD_OF_INITIAL_MEMORY_STATE:
            #print("INITIAL MUTATION")
            return self._initial_memory_mutant()
        # Decision mutation
        #print("DECISION MUTATION")
        return self._decision_list_mutant()
    
    def _print_debug_memory(self, new_number_of_bits_of_memory, new_number_of_bits_of_summary, new_decision_list):
        print("====================SUMMARY INCREASE====================")
        print("=====PREVIOUS=====")
        print("PREVIOUS memory bits", self.number_of_bits_of_memory)
        print("PREVIOUS summary bits", self.number_of_bits_of_summary)
        print("PREV DECISION LIST LEN", len(self.decision_list))
        print("PREV decision list", self.decision_list)
        print("=====NEW=====")
        print("NEW memory bits", new_number_of_bits_of_memory)
        print("NEW summary bits", new_number_of_bits_of_summary)
        print("NEW DECISION LIST LEN", len(new_decision_list))
        print("NEW decision list", new_decision_list)
    
    def _get_bits_of_summary_mutant(self):
        """
        Modify length of summary memory.
        Increases or decreases total memory count by 1 bit.
        Also impacts length of decision list. 
        """
        should_increase_summary = random.choice([True, False])

        # If organism has no summary, don't decrease anything
        if self.number_of_bits_of_summary == 0 and not should_increase_summary:
            return self
        
        # If organism has maximum total memory, don't increase anything
        if (self.number_of_bits_of_memory + self.number_of_bits_of_summary == CONST_CONF.MAX_BITS_OF_MEMORY) and should_increase_summary:
            return self

        new_number_of_bits_of_memory = self.number_of_bits_of_memory
        if should_increase_summary:
            # Increment length of summary
            new_number_of_bits_of_summary = self.number_of_bits_of_summary + 1
            new_decision_list = self.decision_list[:]
            # If summary memory is chosen, add 2^k random decision
            for i in range(2 ** self.number_of_bits_of_memory):
                new_decision_list.append(random.choice([True, False]))
            # Add 1 extra bit to summary memory
            new_initial_summary = self.initial_summary[:]
            new_initial_summary.append(random.choice([True, False]))

            # Length of new decision list 2^k(j+1)
            if len(new_decision_list) != (2 ** new_number_of_bits_of_memory) * (new_number_of_bits_of_summary + 1): 
                self._print_debug_memory(new_number_of_bits_of_memory, new_number_of_bits_of_summary, new_decision_list)
                
            assert len(new_decision_list) == (2 ** self.number_of_bits_of_memory) * (new_number_of_bits_of_summary + 1), "DECISION LIST LENGTHS DON'T MATCH (SUMMARY INCREASING)"
            return HybridPDGenotype(self.number_of_bits_of_memory, new_number_of_bits_of_summary, new_decision_list, self.initial_memory, new_initial_summary)
        
        # Decrease summed memory (j)
        if self.number_of_bits_of_summary > 0:
            new_number_of_bits_of_summary = self.number_of_bits_of_summary - 1
            # Remove most distant summary bit
            new_initial_summary = self.initial_summary[:-1]
        else: 
            new_number_of_bits_of_summary = self.number_of_bits_of_summary
            new_initial_summary = self.initial_summary

        # Decrease new decision list length (2^k(j+1))
        new_decision_list_length = (2 ** self.number_of_bits_of_memory) * (new_number_of_bits_of_summary + 1)
        new_decision_list = self.decision_list[:new_decision_list_length]
        if len(new_decision_list) != (2 ** new_number_of_bits_of_memory) * (new_number_of_bits_of_summary + 1): 
            self._print_debug_memory(new_number_of_bits_of_memory, new_number_of_bits_of_summary, new_decision_list)
            
        assert len(new_decision_list) == new_decision_list_length, "DECISION LIST LENGTHS DON'T MATCH (SUMMARY DECREASING)"
        return HybridPDGenotype(self.number_of_bits_of_memory, new_number_of_bits_of_summary, new_decision_list, self.initial_memory, new_initial_summary)

    def _get_bits_of_memory_mutant(self):
        """
        Modify length of specific memory.
        Increases or decreases total memory count by 1 bit.
        Also impacts length of decision list. 
        """
        should_increase_memory = random.choice([True, False])

        # If organism has no memory, don't decrease anything
        if self.number_of_bits_of_memory == 0 and not should_increase_memory:
            return self
        
        # If organism has maximum total memory, don't increase anything
        if (self.number_of_bits_of_memory + self.number_of_bits_of_summary == CONST_CONF.MAX_BITS_OF_MEMORY) and should_increase_memory:
            return self

        new_number_of_bits_of_summary = self.number_of_bits_of_summary
        if should_increase_memory:
            # Increase specific memory (k)
            new_number_of_bits_of_memory = self.number_of_bits_of_memory + 1 
            
            # If specific memory is chosen, double list
            new_decision_list = self.decision_list * 2
            new_initial_memory = self.initial_memory[:]
            # Add 1 extra bit to specific memory
            new_initial_memory.append(random.choice([True,False]))
        
            # Length of new decision list 2^k(j+1)
            if len(new_decision_list) != (2 ** new_number_of_bits_of_memory) * (new_number_of_bits_of_summary + 1):
                self._print_debug_memory(new_number_of_bits_of_memory, new_number_of_bits_of_summary, new_decision_list)

            assert len(new_decision_list) == (2 ** new_number_of_bits_of_memory) * (self.number_of_bits_of_summary + 1), "DECISION LIST LENGTHS DON'T MATCH (MEMORY INCREASING)"
            return HybridPDGenotype(new_number_of_bits_of_memory, self.number_of_bits_of_summary, new_decision_list, new_initial_memory, self.initial_summary)

        # Decrease specific memory (k)
        if self.number_of_bits_of_memory > 0:
            new_number_of_bits_of_memory = self.number_of_bits_of_memory - 1 
            # Remove most distant memory bit
            new_initial_memory = self.initial_memory[:-1]
        else: 
            new_number_of_bits_of_memory = self.number_of_bits_of_memory
            new_initial_memory = self.initial_memory

        #Decrease length of new decision list (2^k(j+1))
        length_of_new_decision_list = (2 ** new_number_of_bits_of_memory) * (self.number_of_bits_of_summary + 1) 
        new_decision_list = self.decision_list[:length_of_new_decision_list]
        # Length of new decision list 2^k(j+1)
        if len(new_decision_list) != (2 ** new_number_of_bits_of_memory) * (new_number_of_bits_of_summary + 1):
            self._print_debug_memory(new_number_of_bits_of_memory, new_number_of_bits_of_summary, new_decision_list)

        assert len(new_decision_list) == length_of_new_decision_list, "DECISION LIST LENGTHS DON'T MATCH (MEMORY DECREASING)"
        return HybridPDGenotype(new_number_of_bits_of_memory, self.number_of_bits_of_summary, new_decision_list, new_initial_memory, self.initial_summary) 
        
    def _decision_list_mutant(self):
        """Randomly flip a single bit in decision list"""
        mutation_location = random.randrange(len(self.decision_list))
        new_decision_list = self.decision_list[:]
        new_decision_list[mutation_location] = not new_decision_list[mutation_location]
        assert len(self.decision_list) == (2 ** self.number_of_bits_of_memory) * (self.number_of_bits_of_summary + 1), f"DECISION DOES NOT MATCH IN DECISION LIST MUTANT, Number of bits of memory: {self.number_of_bits_of_memory}, Number of bits of summary: {self.number_of_bits_of_summary}, New decision list: {new_decision_list}, Initial memory: {self.initial_memory}, Initial summary: {self.initial_summary}, Old Decision List Length: {len(self.decision_list)}, Old Decision List: {self.decision_list}, Assertion calculation: {(2 ** self.number_of_bits_of_memory) * (self.number_of_bits_of_summary + 1)}, Mutation location: {mutation_location}"
        return HybridPDGenotype(self.number_of_bits_of_memory, self.number_of_bits_of_summary, new_decision_list, self.initial_memory, self.initial_summary)

    def _initial_memory_mutant(self):
        """
        Randomly flip a single bit in initial specific and summed memory.
        This affects the state of memory the organism starts with.  
        """

        # No changes if there is no memory
        if self.number_of_bits_of_memory + self.number_of_bits_of_summary == 0:
            return self
        
        new_initial_memory = self.initial_memory[:]
        # If there is specific memory
        if self.number_of_bits_of_memory > 0:
            # Mutate in specific memory
            mutation_location = random.randrange(len(self.initial_memory))
            new_initial_memory[mutation_location] = not new_initial_memory[mutation_location]

        new_initial_summary = self.initial_summary[:]
        # If there is summary memory
        if self.number_of_bits_of_summary > 0:
            # Mutate in summary memory on initialization
            mutation_location = random.randrange(len(self.initial_summary))
            new_initial_summary[mutation_location] = not new_initial_summary[mutation_location]

        return HybridPDGenotype(self.number_of_bits_of_memory, self.number_of_bits_of_summary, self.decision_list, new_initial_memory, new_initial_summary)
    
class SummaryPDGenotype(object):
    """
    Summary Memory Model Genotype for inheriting in the SummaryPDOrg Class
    """

    def __init__(self, number_of_bits_of_memory, decision_list, initial_memory):
        # We use "memory" here for simplicity, since there is only one memory "type"
        assert 0 <= number_of_bits_of_memory <= CONST_CONF.MAX_BITS_OF_MEMORY
        # Summary Memory Model uses (k + 1) decisions, k being the memory size
        assert len(decision_list) == number_of_bits_of_memory + 1
        assert len(initial_memory) == number_of_bits_of_memory
        self.number_of_bits_of_memory = number_of_bits_of_memory
        self.decision_list = decision_list
        self.initial_memory = initial_memory
    
    def __eq__(self, other):
        """Overload equality operator"""
        return (self.number_of_bits_of_memory == other.number_of_bits_of_memory and
                self.decision_list == other.decision_list and
                self.initial_memory == other.initial_memory)
    
    def __ne__(self, other):
        """Overload not equal operator"""
        return not self == other
    
    def __str__(self):
        """String representation"""
        return "SummaryPDGenotype({}, {}, {})".format(self.number_of_bits_of_memory,
                                                     self.decision_list,
                                                     self.initial_memory)
    
    def __repr__(self):
        """In this case, same as __str__"""
        return str(self)
        
    def __hash__(self):
        """Overload hash operator, necessary for dictionaries and such"""
        hashable_tuple = (self.number_of_bits_of_memory, 
            tuple(self.decision_list), 
            tuple(self.initial_memory))
        return hash(hashable_tuple)
    
    def type(self):
        """Return type as string for easy checking"""
        return 'summary'

    def get_mutant_of_self(self):
        """
        Determines when and how each type of mutation occurs.
        Returns new (mutated) genotype.
        """
        
        # Size mutation
        random_value = random.random()
        if random_value < CONST_CONF.MUTATION_LIKELIHOOD_OF_BITS_OF_MEMORY:
            return self._get_bits_of_memory_mutant()
        # Initial memory mutation
        if random_value < CONST_CONF.MUTATION_LIKELIHOOD_OF_BITS_OF_MEMORY + CONST_CONF.MUTATION_LIKELIHOOD_OF_INITIAL_MEMORY_STATE:
            return self._initial_memory_mutant()
        # Decision mutation
        return self._decision_list_mutant()


    def _get_bits_of_memory_mutant(self):
        """
        Increase or decrease length of initial memory by 1 bit.
        Affects length of decision list as well. 
        """
        should_increase_memory = random.choice([True, False])

         # If organism has no memory, don't decrease memory
        if self.number_of_bits_of_memory == 0 and not should_increase_memory:
            return self
        
        # If organism has maximum memory length, don't increase memory
        if self.number_of_bits_of_memory == CONST_CONF.MAX_BITS_OF_MEMORY and should_increase_memory:
            return self
        
        # If we increase memory length
        if should_increase_memory:
            new_number_of_bits_of_memory = self.number_of_bits_of_memory + 1 # (k)

            # Add a random choice to the decision list
            new_decision_list = self.decision_list[:]
            new_decision_list.append(random.choice([True, False])) # k + 1 decisions

            # Add 1 extra bit to initial memory
            new_initial_memory = self.initial_memory[:]
            new_initial_memory.append(random.choice([True,False]))

            return SummaryPDGenotype(new_number_of_bits_of_memory, new_decision_list, new_initial_memory)
        
        # If we decrease memory length
        new_number_of_bits_of_memory = self.number_of_bits_of_memory - 1 # (k)
        length_of_new_decision_list = len(self.decision_list) - 1
        # Update size of memory and decision lists, most distant past memory bits removed
        new_decision_list = self.decision_list[:length_of_new_decision_list]
        new_initial_memory = self.initial_memory[:-1]
        return SummaryPDGenotype(new_number_of_bits_of_memory, new_decision_list, new_initial_memory) 
        
    def _decision_list_mutant(self):
        """Randomly flip a single bit in decision list"""
        mutation_location = random.randrange(len(self.decision_list))
        new_decision_list = self.decision_list[:]
        new_decision_list[mutation_location] = not new_decision_list[mutation_location]
        return SummaryPDGenotype(self.number_of_bits_of_memory, new_decision_list, self.initial_memory)
        
    def _initial_memory_mutant(self):
        """
        Randomly flip a single bit in initial specified memory.
        This affects the state of memory the organism starts with.  
        """
        # If there is no memory, no change is made.
        if self.number_of_bits_of_memory == 0:
            return self
        
        # Mutate in specified memory
        mutation_location = random.randrange(len(self.initial_memory))
        new_initial_memory = self.initial_memory[:]
        new_initial_memory[mutation_location] = not new_initial_memory[mutation_location]
        return SummaryPDGenotype(self.number_of_bits_of_memory, self.decision_list, new_initial_memory)