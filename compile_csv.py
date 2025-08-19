import csv
import glob
import re
import pandas
import os
import argparse
import sys
import numpy as np
import ast

from tqdm import tqdm
from collections import defaultdict

maxInt = sys.maxsize

while True:
  try:
      csv.field_size_limit(maxInt)
      break
  except OverflowError:
      maxInt = int(maxInt/10)

def join_path(output_folder, filename):
        return os.path.join(output_folder, filename)

def build_experiment_csv(output_folder, csv):
  csv_type = csv.split('_')
  list_of_bits = glob.glob(join_path(output_folder, f'*/{csv}'))
  all_bits_df = pandas.DataFrame()

  print(f"Compiling Experimental {csv_type[2].upper()} Results")
  for individual_file in tqdm(list_of_bits):
      Condition = individual_file.split("/")[-1]
      Condition = Condition.split("_")[4]
      
      individual_file_df = pandas.read_csv(individual_file)
      individual_file_df['Condition'] = [Condition] * individual_file_df.shape[0]
      individual_file_df['Generation'] = range(individual_file_df.shape[0])
      all_bits_df = all_bits_df._append(individual_file_df)
  
  all_bits_df.to_csv(join_path(output_folder, f'all_bits_df_{csv_type[2]}_comp_more_values.csv'), header=True)

def dec_list_len_csv(output_folder, csv, new_csv):
  list_of_bits = glob.glob(join_path(output_folder, f'*/{csv}'))
  all_bits_df = pandas.DataFrame()
  
  for individual_file in tqdm(list_of_bits):
      Condition = individual_file.split("/")[-1]
      Condition = Condition.split("_")[4]
      
      individual_file_df = pandas.read_csv(individual_file)
      individual_file_df['Condition'] = [Condition] * individual_file_df.shape[0]
      individual_file_df['Generation'] = range(individual_file_df.shape[0])
      all_bits_df = all_bits_df._append(individual_file_df)
  
  all_bits_df.to_csv(join_path(output_folder, new_csv), header=True)

def combine_sum_mem_csv(output_folder):
  
  mem_df = pandas.read_csv(join_path(output_folder, 'all_bits_df_Memory_comp_more_values.csv'))
  sum_df = pandas.read_csv(join_path(output_folder, 'all_bits_df_Summary_comp_more_values.csv'))
  tracking_df = mem_df.drop(mem_df.columns[[0, -2, -1]], axis=1)
  new_headers = ['Index'] + [f'Organisms with {i} Bits Total' for i in range(len(list(tracking_df.columns)))] + ['Condition', 'Generation']

  mem_col_dict = {name:new_headers[i] for i, name in enumerate(list(mem_df.columns))}
  sum_col_dict = {name:new_headers[i] for i, name in enumerate(list(sum_df.columns))}

  sum_df.rename(columns=sum_col_dict, inplace=True)
  mem_df.rename(columns=mem_col_dict, inplace=True)
  
  assert mem_df.shape == sum_df.shape, f"Memory - {mem_df.shape} and Summary - {sum_df.shape} DataFrame mismatch shape"
  total_df = sum_df
  total_df[1:-2] = sum_df[1:-2] + mem_df[1:-2]
  total_df['Index'] = mem_df['Index']
  total_df['Condition'] = mem_df['Condition']
  total_df['Generation'] = mem_df['Generation']
  del total_df['Index']

  total_df.to_csv(join_path(output_folder, f'all_bits_df_Total_comp_more_values.csv'), header=True)

def make_strategy_dictionary(fileregx):
  def atoi(text):
      return int(text) if text.isdigit() else text

  def natural_keys(text):
      '''
      alist.sort(key=natural_keys) sorts in human order
      http://nedbatchelder.com/blog/200712/human_sorting.html
      (See Toothy's implementation in the comments)
      '''
      return [ atoi(c) for c in re.split('(\d+)', text) ]


  filelist = glob.glob(fileregx + "/detail-*.csv")

  number_of_strategies, num_coops, num_defs = {}, {}, {}

  filelist.sort(key=natural_keys)
  #initializing dictionary
  #puts all the strategies ever into dictionary
  last_file = filelist[-1]

  with open(last_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
      if row == []:
        continue
      key = row[1] + "~" + row[2]
      number_of_strategies[key] = []
      num_coops[key] = []
      num_defs[key] = []

  #makes keys of decisionlist+memory
  #adds number of alive to the keys location in dictionary

  for individual_file in filelist:
    with open(individual_file, 'r') as csvfile:
      reader = csv.reader(csvfile)
      next(reader)
      for row in reader:
        if row == []:
          continue
        key = row[1] + "~" + row[2]
        #Update index tracking for normal and hybrid functions
        number_of_strategies[key].append(int(row[4]))
        if int(row[4]) > 0:
            # Since we are mapping by genotype, need to correlate to genotype
            # From there determine statistical spread of phenotypes that are consistent with given organisms
            # For example, we track the rates for organisms binned based on whether they have more cooperative or defective phenotypes
            # We can do this by the proportion of phenotypes with 'D' for Defect and 'C' for Cooperate excedding the 51% threshold
            # TFT-Variant can be set directly within the 50% threshold 
            num_coops[key].append(int(np.average(list(map(float, row[7].split())))))
            num_defs[key].append(int(np.average(list(map(float, row[8].split())))))
        else: 
            num_coops[key].append(0)
            num_defs[key].append(0)
    max_list_length = max([ len(number_of_strategies[l]) for l in number_of_strategies])
    for key in number_of_strategies:
      if len(number_of_strategies[key]) < max_list_length:
        number_of_strategies[key].append(0)
        num_coops[key].append(0)
        num_defs[key].append(0)
   
  #picking out bad strats and makes a list of them
  failed_strategies = []
  for key in number_of_strategies:
    if sum(number_of_strategies[key]) <= 1:
      failed_strategies.append(key)

  #this is creating list of best common strategy
  common_one_strategy = [ (number_of_strategies[key][-1], key) for key in number_of_strategies]
  common_one_strategy = max(common_one_strategy, key= lambda x: x[0])[1]

  #delete bad strats from main dictionary
  for key in failed_strategies:
    del(number_of_strategies[key])

  strategies_df = pandas.DataFrame.from_dict(number_of_strategies, orient= 'index')
  coops_df = pandas.DataFrame.from_dict(num_coops, orient= 'index')
  defs_df = pandas.DataFrame.from_dict(num_defs, orient= 'index')
  #concatenating multiple data frames
  Condition = fileregx.split("/")[-1]
  Condition = Condition.split("_")[4]
  Condition = Condition[3:]
  strategies_df['Condition'] = [Condition] * len(number_of_strategies)
  strategies_df['Strategy'] = number_of_strategies.keys()
  
  return strategies_df, [common_one_strategy, Condition], coops_df, defs_df

# coop_rate[key].append(int(np.average(list(map(float, row[9].split())))))

def build_rate_values(row, rate, key, value):
  if row[value] == 'nan': row[value] = 0.0
  rate[key].append(row[value])
  return rate
   
   
def make_rate_dict(fileregx):
  def atoi(text):
      return int(text) if text.isdigit() else text

  def natural_keys(text):
      '''
      alist.sort(key=natural_keys) sorts in human order
      http://nedbatchelder.com/blog/200712/human_sorting.html
      (See Toothy's implementation in the comments)
      '''
      return [ atoi(c) for c in re.split('(\d+)', text) ]


  filelist = glob.glob(fileregx + "/detail-*.csv")

  number_of_strategies, coop_rate, def_rate = {}, {}, {}
  recip_rate, grudge_rate, forg_rate, strat_rate = {}, {}, {}, {}

  filelist.sort(key=natural_keys)
  #initializing dictionary
  #puts all the strategies ever into dictionary
  last_file = filelist[-1]

  with open(last_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
      if row == []:
        continue
      key = row[1] + "~" + row[2]
      number_of_strategies[key] = []
      coop_rate[key], def_rate[key] = [], []
      recip_rate[key], grudge_rate[key], forg_rate[key], strat_rate[key] = [], [], [], []

  #makes keys of decisionlist+memory
  #adds number of alive to the keys location in dictionary

  for individual_file in filelist:
    with open(individual_file, 'r') as csvfile:
      reader = csv.reader(csvfile)
      next(reader)
      for row in reader:
        if row == []:
          continue
        key = row[1] + "~" + row[2]
        #Update index tracking for normal and hybrid functions
        number_of_strategies[key].append(int(row[4]))
        # if key == '[False, False, False, False]~[True, True, False]': 
        #    print(row)
        if int(row[4]) > 0:
            # Since we are mapping by genotype, need to correlate to genotype
            # From there determine statistical spread of phenotypes that are consistent with given organisms
            # For example, we track the rates for organisms binned based on whether they have more cooperative or defective phenotypes
            # We can do this by the proportion of phenotypes with 'D' for Defect and 'C' for Cooperate excedding the 51% threshold
            # TFT-Variant can be set directly within the 50% threshold 
            #Bits,Decisions,Memory,LiveFitness,Alive,Id,ParentId,Cooperates,Defects,Cooperations,Defections,Reciprocate,Grudge,Forgiveness,Strategy,DecLength
            #    0,       1,      2,          3,   4, 5,        6,        7,       8,           9,        10,         11,    12,         13,     14,       15
            # So, read from strategy to determine whether the phenotype falls into a cooperative or defectionary bin
            # i.e. 100%, 75%, 62.5%, 50%, 37.5%, 25%, 0% from cooperative perspective
            # After which, consider the rate calculation from that generation to produce the average and variance on rate for a given generation.
            # Need to double-check if this works for multiple tests which can then simplify tests.
            # Finally, plotting should involve a relativelty simple line plot. 
            # Fitness plots could be interesting as well to determinen whether summary helps with more cooperative strategies##

            #TODO: Currently for [Fale, False, False, False]~[True, True, False] there will be use-cases that 
            # results in 1.0 coop rates which should be impossible. I believe that it may be a tracking error
            # The error aligns with the first instance of the code interpreting the strategy successfuly
            # Where strat_df.csv shows ['AD','AD','AD'] at generation 12 instead of dead or empty csvs. 
            # But at generation 14 where strat_df.csv is empty it goes back to zero! and then 1 again when strat_df.csv has ['AD','AD','AD']
            # Could it be that coop_rate is also empty and assigned the bool value of a list?
            # I think it's correlated to the data potentially being empty will want to test with a key = row[1] + "~" + row[2] that corresponds to sanity checking
            coop_rate = build_rate_values(row, coop_rate, key, 9)
            def_rate = build_rate_values(row, def_rate, key, 10)
            recip_rate = build_rate_values(row, recip_rate, key, 11)
            grudge_rate = build_rate_values(row, grudge_rate, key, 12)
            forg_rate = build_rate_values(row, forg_rate, key, 13)
            # TODO: Convert Strings for Strategies to Frequencies i.e. ['AD', 'AD', 'AD'] to {'AD': 3}
            # From there, may want to build a running total frequency of the five strategies over all subsequent plots
            # Current Idea, maybe just map to the five current archetypes and then determine the plotting from there: {'AD':3, 'AC':0, 'D-Emerg':0, 'C-Emerg':0, 'TFT':0}
            # Plot would basically dervive from a dict of dicts to establish global strategy frequency
            # Organize all plots by the initial frequency of the strategies once they are first derived.
            # So after three generations of being consider a 'GHOST' or 'DEAD' the organisms with 60% cooperative moves would then be considered a friendly organism.
            strat_rate[key].append(ast.literal_eval(row[14]))
        else: 
            coop_rate[key].append(0.0), def_rate[key].append(0.0), recip_rate[key].append(0.0)
            grudge_rate[key].append(0.0), forg_rate[key].append(0.0), strat_rate[key].append("['DEAD']")
    max_list_length = max([len(number_of_strategies[l]) for l in number_of_strategies])
    for key in number_of_strategies:
      if len(number_of_strategies[key]) < max_list_length:
        number_of_strategies[key].append(0)
        coop_rate[key].append(0.0), def_rate[key].append(0.0), recip_rate[key].append(0.0)
        grudge_rate[key].append(0.0), forg_rate[key].append(0.0), strat_rate[key].append("['GHOST']")

  coops_df = pandas.DataFrame.from_dict(coop_rate, orient= 'index')
  defs_df = pandas.DataFrame.from_dict(def_rate, orient= 'index')
  recip_df = pandas.DataFrame.from_dict(recip_rate, orient= 'index')
  grudge_df = pandas.DataFrame.from_dict(grudge_rate, orient= 'index')
  forg_df = pandas.DataFrame.from_dict(forg_rate, orient= 'index')
  strat_df = pandas.DataFrame.from_dict(strat_rate, orient= 'index')
  
  return {'coop': coops_df, 'defs': defs_df, 'recip': recip_df, 'grudge': grudge_df, 'forg': forg_df, 'strat': strat_df}


def format_common_strat(most_common):
    most_common[0] = most_common[0].split("~")
    most_common[0] = [eval(strat) for strat in most_common[0]]
    binary_most_common = ""
    for ls in most_common[0]:
      for bit in ls:
        if bit:
          binary_most_common += str(1)
        else:
          binary_most_common += str(0)
      binary_most_common += "~"
    most_common[0] = binary_most_common

    return most_common

def make_hybrid_dictionary(fileregx):
  def atoi(text):
      return int(text) if text.isdigit() else text

  def natural_keys(text):
      '''
      alist.sort(key=natural_keys) sorts in human order
      http://nedbatchelder.com/blog/200712/human_sorting.html
      (See Toothy's implementation in the comments)
      '''
      return [ atoi(c) for c in re.split('(\d+)', text) ]


  filelist = glob.glob(fileregx + "/detail-*.csv")

  number_of_strategies = {}

  filelist.sort(key=natural_keys)
  #initializing dictionary
  #puts all the strategies ever into dictionary
  last_file = filelist[-1]

  with open(last_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
      if row == []:
        continue
      key = row[2] + "~" + row[3]
      number_of_strategies[key] = []

  #makes keys of decisionlist+memory
  #adds number of alive to the keys location in dictionary

  for individual_file in filelist:
    with open(individual_file, 'r') as csvfile:
      reader = csv.reader(csvfile)
      next(reader)
      for row in reader:
        if row == []:
          continue
        key = row[2] + "~" + row[3]
        number_of_strategies[key].append(int(row[6]))
    max_list_length = max([ len(number_of_strategies[l]) for l in number_of_strategies])
    for key in number_of_strategies:
      if len(number_of_strategies[key]) < max_list_length:
        number_of_strategies[key].append(0)
   
  #picking out bad strats and makes a list of them
  failed_strategies = []
  for key in number_of_strategies:
    if sum(number_of_strategies[key]) <= 1:
      failed_strategies.append(key)

  #this is creating list of best common strategy
  common_one_strategy = [ (number_of_strategies[key][-1], key) for key in number_of_strategies]
  common_one_strategy = max(common_one_strategy, key= lambda x: x[0])[1]

  #delete bad strats from main dictionary
  for key in failed_strategies:
    del(number_of_strategies[key])

  strategies_df = pandas.DataFrame.from_dict(number_of_strategies, orient= 'index')
  #concatenating multiple data frames
  Condition = fileregx.split("/")[-1]
  Condition = Condition.split("_")[4]
  Condition = Condition[3:]
  strategies_df['Condition'] = [Condition] * len(number_of_strategies)
  strategies_df['Strategy'] = number_of_strategies.keys()

  return strategies_df, [common_one_strategy, Condition]

def format_hybrid_strat(most_common):
  most_common[0] = most_common[0].split("~")
  most_common[0] = [eval(strat) for strat in most_common[0]]
  binary_most_common = ""
  for ls in most_common[0]:
    for bit in ls:
      if bit:
        binary_most_common += str(1)
      else:
        binary_most_common += str(0)
    binary_most_common += "~"
  most_common[0] = binary_most_common
  return most_common


def build_hybrid_strat(list_most_common, paths, frames):
  strategies_df, most_common = make_hybrid_dictionary(paths)
  frames.append(strategies_df)
  most_common = format_hybrid_strat(most_common)
  list_most_common += ",".join(most_common) + "\n"
  return frames, list_most_common, strategies_df

def build_original_strat(list_most_common, paths, frames):
  strategies_df, most_common, coops_df, defs_df = make_strategy_dictionary(paths)
  frames.append(strategies_df)
  most_common = format_common_strat(most_common)
  list_most_common += ",".join(most_common) + "\n"
  return frames, list_most_common, strategies_df, coops_df, defs_df

def extract_cost_from_path(path):
    match = re.search(r'_(\d*\.?\d+)_cost', path)
    return match.group(1) if match else "unknown"

def compile_rate_csv(temp_df, rate_key, output_folder):
  #print(temp_df)
  temp_df.to_csv(join_path(output_folder, f"{rate_key}_df.csv"), header=False)

import pandas as pd
from collections import Counter

def average_numeric_by_label_streaming(
    dfs,
    numeric_cols=slice(1, 103),
    extra_meta_cols=None,
):
    extra_meta_cols = extra_meta_cols or []

    total_sum = None        # DataFrame (index = label), numeric cols only
    total_count = None      # same shape as total_sum; counts non-null cells
    rows_per_label = None   # Series counting rows per label across all dfs
    meta_tallies = {c: {} for c in extra_meta_cols}  # c -> {label: Counter()}

    for df in dfs:
        # numeric slice (by position), coerce to numeric
        num = df.iloc[:, numeric_cols].apply(pd.to_numeric, errors="coerce")
        labels = df.index  # label lives in the index

        # Per-df sums & counts by label
        sum_by = num.groupby(labels).sum(min_count=1)
        cnt_by = num.notna().groupby(labels).sum(min_count=1)

        # Accumulate
        if total_sum is None:
            total_sum = sum_by.copy()
            total_count = cnt_by.copy()
        else:
            total_sum = total_sum.add(sum_by, fill_value=0)
            total_count = total_count.add(cnt_by, fill_value=0)

        # Total rows per label (even if numeric cells are NaN)
        rpl = pd.Series(1, index=labels).groupby(level=0).sum()
        rows_per_label = rpl if rows_per_label is None else rows_per_label.add(rpl, fill_value=0)

        # Track representative metadata values (mode) if requested
        for c in extra_meta_cols:
            if c in df.columns:
                for lbl, val in zip(labels, df[c]):
                    if pd.notna(val):
                        meta_tallies[c].setdefault(lbl, Counter())[val] += 1

    # Final averages
    avg = total_sum.div(total_count).where(total_count > 0)

    # Build output
    out = avg.copy()
    out["n_rows"] = rows_per_label.reindex(out.index)

    # Attach meta reps (most frequent value per label)
    for c in extra_meta_cols:
        out[c] = [
            (meta_tallies[c].get(lbl).most_common(1)[0][0]
             if lbl in meta_tallies[c] and meta_tallies[c][lbl]
             else pd.NA)
            for lbl in out.index
        ]

    return out.reset_index(names="label")

def build_strat_csv(output_folder):
    frames = []
    list_most_common = "Common_Strategy, Condition\n"
    
    grp_coops = defaultdict(list)
    grp_defs = defaultdict(list)
    print("Compiling Strategy, Coops, Defects Results")
    for paths in tqdm(glob.glob(join_path(output_folder, "*"))):
        #print(paths)
        if (os.path.isdir(paths)):
          if 'hybrid' in output_folder:
            frames, list_most_common, strategies_df = build_hybrid_strat(list_most_common, paths, frames)
          else:
            group_key = extract_cost_from_path(paths)
           
            frames, list_most_common, strategies_df, coops_df, defects_df = build_original_strat(list_most_common, paths, frames)
            
            #{'coop': coops_df, 'defs': defs_df, 'recip': recip_df, 'grudge': grudge_df, 'forg': forg_df, 'strat': strat_df}

            grp_coops[group_key] = coops_df
            grp_defs[group_key] = defects_df
          rate_dict = make_rate_dict(paths)
    for paths in tqdm(glob.glob(join_path(output_folder, "*"))):
        if os.path.isdir(paths):
            group_key = extract_cost_from_path(paths)
           
            frames, list_most_common, strategies_df, coops_df, defects_df = build_original_strat(list_most_common, paths, frames) #need to confirm that this leads to all scenarios

            grp_coops[group_key] = coops_df
            grp_defs[group_key] = defects_df

    for key in list(rate_dict.keys()): rate_dict[key].to_csv(join_path(output_folder, f"{key}_df.csv"), header=False)

    with open(join_path(output_folder, "most_common.csv"), "w") as most_common_file:
        for item in list_most_common:
            most_common_file.write(item)

    print("Grouped Defect Titles: ", grp_defs.keys())
    for key in list(grp_defs.keys()):
       grp_defs[key].to_csv(join_path(output_folder, f"defects{key}_df.csv"), header=False)

    print("Grouped Co-Ops Titles: ", grp_coops.keys())  
    for key in list(grp_coops.keys()):
       grp_coops[key].to_csv(join_path(output_folder, f"coops{key}_df.csv"), header=False)
    strategies_df.to_csv(join_path(output_folder, "strategies_df.csv"), header=False)
    
    avg_running_all = average_numeric_by_label_streaming(frames,
                                    numeric_cols=slice(0, 103),
                                    extra_meta_cols=["Condition", "Strategy"])
    #print(avg_running_all)
    #print(avg_running_all[['label', 100]])
    avg_running_all.to_csv(join_path(output_folder, "strategies_df2.csv"), header=False)
    avg_running_all[['label', 100, 'n_rows']].to_csv(join_path(output_folder, "sanity_check.csv"), header=False)

def main():
    arg_parser = argparse.ArgumentParser(
        description='Function to convert csvs into DataFrames for plotting.')
    
    # Expects 1 argument: output folder
    arg_parser.add_argument("-o", "--output_folder", type=str, default="tests/pd_temp")
    args = arg_parser.parse_args()

    # for csv in ['bits_of_Memory_overtime.csv', 'bits_of_Summary_overtime.csv']:
    #  build_experiment_csv(args.output_folder, csv)
    # if 'hybrid' in args.output_folder: combine_sum_mem_csv(args.output_folder)
    # print(f"Compiling Decision List Length Results")
    # dec_list_len_csv(args.output_folder, 'decision_list_length_aggregate.csv', 'decision_list_length_overtime.csv')
    # print(f"Compiling Fitness Results")
    # dec_list_len_csv(args.output_folder, 'fitness_aggregate.csv', 'fitness_overtime.csv')
    build_strat_csv(args.output_folder)
    

if __name__ == "__main__":
    main()