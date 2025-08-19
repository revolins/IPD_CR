# Cooperative Resilience within the Iterated Prisoner’s Dilemma

## Researchers:
Jay Revolinsky, Karen Suzue, Alec Said, Charles Ofria

# We assume that the user has anaconda/miniconda installed
https://docs.anaconda.com/free/miniconda/miniconda-install/

# Build Environment
```
conda env create -f environment.yml
```

# Startup Environmnet
```
conda activate memGA
```

# Run Experiments
```
python run_tests.py
```


## (Optional)
* --seed, --s   value to seed runs, for reproducibility
* --number_of_generations, --ng     number of generations for organisms (DEFAULT = 500)
* --number_of_organisms, --no   number of organisms involved in a given population (DEFAULT = 10)
* --org_type    Type of organism used in experiment, PD uses PDOrg, hybrid_pd uses HybridPDOrg (DEFAULT = pd)
* --tournament_size, --ts  Size of tournament for competing organisms (DEFAULT = 8)
* --verbose     True = full output to *-detail.csv, False = standard organism output (DEFAULT = True)
* --number_of_rounds, --nr  Number of rounds in a given tournament before next generation decided (DEFAULT = 64)
* --temptation, Value of defecting when other organism cooperates (DEFAULT = 5)
* --reward, Value of cooperating when other organism cooperates (DEFAULT = 3)
* --punishment, Value of defecting when other organism defects (DEFAULT = 1)
* --sucker, Value of cooperating when other organism defects (DEFAULT = 0)
* --proportion_cost_per_memory_bit, --m_c   Fitness cost imposed for each memory bit organism has (DEFAULT = 0.0)
* --max_bits_of_memory, --max_m     Limit on organism's memory list bits (DEFAULT = 4)
* --mutation_likelihood_of_bits_of_memory "--ml_mem",  Likelihood that memory list mutates after mutation decided (DEFAULT = 1.0)
* --mutation_likelihood_of_initial_memory_state, --ml_dec Likelihood that decision list mutates after mutation decided (DEFAULT = 1.0)
* --toggle_self_memory_on   True = Organism remembers their own moves, False = Organism ignores their past moves (DEFAULT = False)
* --mutation_rate, --mr  Mutation Rate to determine if organism will mutate during a given generation (DEFAULT = 0.0)
* --output_frequency    Rate at which organisms output their state (DEFAULT=10)
* --selection_by_static_competitor, --static    True = static, False = co-evolutionary (DEFAULT = False)
* --randomized_rounds  True = randomly-determine number of rounds following a normal distribution with the mean centered around --number_of_rounds
* --noise  (DEFAULT = 0.0) The probability with which an organism's move will be misread