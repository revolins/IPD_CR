import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import argparse
import scikit_posthocs as sp
import ast
import re
import csv

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #For those pesky deprecation warnings

def join_path(output_folder, filename):
        return os.path.join(output_folder, filename)

def retrieve_dll(args):
    print(f"Retrieving Decision List Length Plots")
    bits_of_memory_df = pd.read_csv(join_path(args.output_folder, f"decision_list_length_overtime.csv"))
    bits_of_memory_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    read_columns = bits_of_memory_df.columns[1:-2]
    bits_of_memory_df.drop(bits_of_memory_df.columns[0], axis=1, inplace=True)
    bits_of_memory_df['Generation'] = pd.to_numeric(bits_of_memory_df['Generation'], errors='coerce')
    bits_of_memory_df['Condition'] = pd.Categorical(pd.to_numeric(bits_of_memory_df['Condition'], errors='coerce'))
    for col in tqdm(read_columns):
        bits_of_memory_df[col] = pd.to_numeric(bits_of_memory_df[col], errors='coerce')

    weights = np.arange(1, len(read_columns) + 1)
    bits_of_memory_df = pd.concat([bits_of_memory_df, pd.DataFrame(read_columns)], axis=1)
    bits_of_memory_df = bits_of_memory_df.copy()
    bits_of_memory_df['Mean'] = bits_of_memory_df.apply(lambda row: np.average(row[:len(read_columns)], weights=weights), axis=1)

    summary_df = bits_of_memory_df.groupby(['Condition', 'Generation'], observed=True).agg(
        group_mean=('Mean', 'mean'),
        group_sd=('Mean', 'std')
    ).reset_index()

    return summary_df, bits_of_memory_df, read_columns

def retrieve_fit(args):
    print(f"Retrieving Fitness Plots")
    bits_of_memory_df = pd.read_csv(join_path(args.output_folder, f"fitness_overtime.csv"))
    bits_of_memory_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    read_columns = bits_of_memory_df.columns[1:-2]
    bits_of_memory_df.drop(bits_of_memory_df.columns[0], axis=1, inplace=True)
    bits_of_memory_df['Generation'] = pd.to_numeric(bits_of_memory_df['Generation'], errors='coerce')
    bits_of_memory_df['Condition'] = pd.Categorical(pd.to_numeric(bits_of_memory_df['Condition'], errors='coerce'))
    for col in tqdm(read_columns):
        bits_of_memory_df[col] = pd.to_numeric(bits_of_memory_df[col], errors='coerce')

    weights = np.arange(1, len(read_columns) + 1)
    bits_of_memory_df = pd.concat([bits_of_memory_df, pd.DataFrame(read_columns)], axis=1)
    bits_of_memory_df = bits_of_memory_df.copy()

    bits_of_memory_df['Mean'] = bits_of_memory_df.apply(lambda row: np.average(row[:len(read_columns)], weights=weights), axis=1)

    summary_df = bits_of_memory_df.groupby(['Condition', 'Generation'], observed=True).agg(
        group_mean=('Mean', 'mean'),
        group_sd=('Mean', 'std')
    ).reset_index()

    return summary_df, bits_of_memory_df, read_columns

def average_fitness(args):
    
    summary_df, bits_of_memory_df, read_columns = retrieve_fit(args)
    conditions = summary_df['Condition'].unique()

    palette = sns.color_palette("husl", len(conditions))
    sns.lmplot(data=summary_df, x='Generation', y='group_mean', hue='Condition', markers=False, scatter=False, palette=palette)
    plt.title(f'Average Fitness Over Time - Regression' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.ylabel(f'Average Fitness')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(join_path(args.output_folder, f'Average_Fitness_Regression.png'))

    palette = sns.color_palette("husl", len(conditions))
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, err_style="band", errorbar='ci', palette=palette)
    ax = plt.gca()
    for i, condition in enumerate(conditions):
        df_condition = summary_df[summary_df['Condition'] == condition]
        color = palette[i]
        
        ax.fill_between(x=df_condition['Generation'],
                        y1=df_condition['group_mean'] - df_condition['group_sd'],
                        y2=df_condition['group_mean'] + df_condition['group_sd'],
                        color=color, alpha=0.3)
    plt.title(f'Average Fitness Over Time' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.ylabel(f'Average Fitness')
    plt.grid(False)
    plt.savefig(join_path(args.output_folder, f'Average_Fitness_overtime.png'))

    palette = sns.color_palette("husl", len(conditions))
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, palette=palette)
    plt.title(f'Average Fitness Over Time' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.ylabel(f'Average Fitness')
    plt.grid(False)
    plt.savefig(join_path(args.output_folder, f'Average_Fitness_NoDashes.png'))

    plt.figure(figsize=(10, 6))
    plt.hist(bits_of_memory_df['Mean'], bins=30)
    plt.title(f'Distribution of Mean Fitness' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.xlabel(f'Mean Fitness')
    plt.ylabel('Frequency')
    plt.savefig(join_path(args.output_folder, f'Mean_Fitness.png'))

def average_dll(args):
    
    summary_df, bits_of_memory_df, read_columns = retrieve_dll(args)
    conditions = summary_df['Condition'].unique()

    palette = sns.color_palette("husl", len(conditions))
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, err_style="band", errorbar='ci', palette=palette)
    ax = plt.gca()
    for i, condition in enumerate(conditions):
        df_condition = summary_df[summary_df['Condition'] == condition]
        color = palette[i]
        
        ax.fill_between(x=df_condition['Generation'],
                        y1=df_condition['group_mean'] - df_condition['group_sd'],
                        y2=df_condition['group_mean'] + df_condition['group_sd'],
                        color=color, alpha=0.3)
    plt.title(f'Average Decision List Length Over Time' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.ylabel(f'Average Decision List Length')
    plt.grid(False)
    plt.savefig(join_path(args.output_folder, f'Average_Decision_List_Length.png'))

    palette = sns.color_palette("husl", len(conditions))
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, palette=palette)
    plt.title(f'Average Decision List Length Over Time' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.ylabel(f'Average Decision List Length')
    plt.grid(False)
    plt.savefig(join_path(args.output_folder, f'Average_Decision_List_Length_NoDashes.png'))

    plt.figure(figsize=(10, 6))
    plt.hist(bits_of_memory_df['Mean'], bins=30)
    plt.title(f'Distribution of Mean Decision List Length' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.xlabel(f'Mean Decision List Length')
    plt.ylabel('Frequency')
    plt.savefig(join_path(args.output_folder, f'Mean_Decision_List_Length.png'))

def average_mem(args, csv):
    csv_type = csv.split('_')
    print(f"Constructing {csv_type[3].upper()} Plots and Running Statistical Tests")
    bits_of_memory_df = pd.read_csv(join_path(args.output_folder, f"all_bits_df_{csv_type[3]}_comp_more_values.csv"))
    bits_of_memory_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    read_columns = bits_of_memory_df.columns[1:-2]
    bits_of_memory_df.drop(bits_of_memory_df.columns[0], axis=1, inplace=True)
    bits_of_memory_df['Generation'] = pd.to_numeric(bits_of_memory_df['Generation'], errors='coerce')
    bits_of_memory_df['Condition'] = pd.Categorical(pd.to_numeric(bits_of_memory_df['Condition'], errors='coerce'))
    for col in tqdm(read_columns):
        bits_of_memory_df[col] = pd.to_numeric(bits_of_memory_df[col], errors='coerce')

    weights = np.arange(0.0, float(len(read_columns)))
 
    bits_of_memory_df = pd.concat([bits_of_memory_df, pd.DataFrame(read_columns)], axis=1)
    bits_of_memory_df = bits_of_memory_df.copy()
    bits_of_memory_df['Mean'] = bits_of_memory_df.apply(lambda row: np.average(a=weights, weights=row[:len(weights)]), axis=1)

    summary_df = bits_of_memory_df.groupby(['Condition', 'Generation'], observed=True).agg(
        group_mean=('Mean', 'mean'),
        group_sd=('Mean', 'std')
    ).reset_index()

    if 'noise' in args.output_folder:
        summary_df.to_csv(f'noisy_data/{args.output_folder}_summary.csv')

    conditions = summary_df['Condition'].unique()

    palette = sns.color_palette("husl", len(conditions))
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, err_style="band", errorbar='ci', palette=palette)
    ax = plt.gca()
    for i, condition in enumerate(conditions):
        df_condition = summary_df[summary_df['Condition'] == condition]
        color = palette[i]
        
        ax.fill_between(x=df_condition['Generation'],
                        y1=df_condition['group_mean'] - df_condition['group_sd'],
                        y2=df_condition['group_mean'] + df_condition['group_sd'],
                        color=color, alpha=0.3)
        
    plt.title(f'Weighted Average Bits of {csv_type[3]} Over Time' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.ylabel(f'Weighted Average Bits of {csv_type[3]}')
    plt.grid(False)
    plt.savefig(join_path(args.output_folder, f'Weighted_Average_{csv_type[3]}.png'))

    tot_mem_bits_df = bits_of_memory_df.groupby(['Generation'], observed=True).agg(
        group_mean=('Mean', 'mean'),
        group_sd=('Mean', 'std')
    ).reset_index()

    #tot_mem_bits_df.loc[tot_mem_bits_df['group_mean'] == np.min(tot_mem_bits_df['group_mean']), 'group_mean'] = tot_mem_bits_df['group_mean'].mean()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=tot_mem_bits_df, x='Generation', y='group_mean', markers=False, dashes=False)
    plt.title(f'Total Memory Bits Over Time' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.ylabel(f'Total Memory Bits')
    plt.grid(False)
    plt.savefig(join_path(args.output_folder, f'Mem_Bits_Overtime_{csv_type[3]}.png'))

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, palette=palette)
    plt.title(f'Weighted Average Bits of {csv_type[3]} Over Time' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.ylabel(f'Weighted Average Bits of {csv_type[3]}')
    plt.grid(False)
    plt.savefig(join_path(args.output_folder, f'Weighted Average_{csv_type[3]}_NoDashes.png'))

    plt.figure(figsize=(10, 6))
    plt.hist(bits_of_memory_df['Mean'], bins=30)
    plt.title(f'Distribution of Mean Bits of {csv_type[3]}' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.xlabel(f'Mean Bits of {csv_type[3]}')
    plt.ylabel('Frequency')
    plt.savefig(join_path(args.output_folder, f'Mean_{csv_type[3]}.png'))

    stat_test(args, csv_type, summary_df, bits_of_memory_df)
    dll_plots(args, csv_type, conditions)

def dll_plots(args, csv_type, conditions):
  
    palette = sns.color_palette("husl", len(conditions)) #Matches average_mem
    dll_df, _, _ = retrieve_dll(args)
    plt.figure(figsize=(10, 6))
    #sns.lineplot(data=tot_mem_bits_df, x='Generation', y='group_mean', hue='group_mean', markers=False, dashes=True, palette=plt.colormaps["viridis"]) #Versus Memory?
    sns.lineplot(data=dll_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, err_style="band", errorbar='ci', palette=palette)
    ax = plt.gca()
    for i, condition in enumerate(conditions):
        df_condition = dll_df[dll_df['Condition'] == condition]
        color = palette[i]
        
        ax.fill_between(x=df_condition['Generation'],
                        y1=df_condition['group_mean'] - df_condition['group_sd'],
                        y2=df_condition['group_mean'] + df_condition['group_sd'],
                        color=color, alpha=0.3)
        
    plt.title(f'Average Decision List Length Over Time' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.ylabel(f'Average Decision List Length')
    plt.grid(False)
    plt.savefig(join_path(args.output_folder, f'Average_Decision_List_Length_{csv_type[3]}.png'))

    plt.figure(figsize=(10, 6))
    #sns.lineplot(data=summary_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, palette=palette) #Plot Versus Memory?
    sns.lineplot(data=dll_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, palette=palette)
    plt.title(f'Average Decision List Length Over Time' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.ylabel(f'Average Decision List Length')
    plt.grid(False)
    plt.savefig(join_path(args.output_folder, f'Average_Decision_List_Length_No_Dashes_{csv_type[3]}.png'))

def stat_test(args, csv_type, summary_df, bits_of_memory_df):
    dll_df, _, _ = retrieve_dll(args)
    forward_df = pd.concat([summary_df['group_mean'], dll_df['group_mean']], axis=1)
    inv_df = pd.concat([dll_df['group_mean'], summary_df['group_mean']], axis=1)
    with open(join_path(args.output_folder, "stat_output.txt"), "a") as f:
        f.write("########################################################################################################\n")
        f.write("#####################################        Correlation       #########################################\n")
        f.write("##############################       Memory versus Decision List Length         ########################\n")
        f.write("########################################################################################################\n")
        try:
            pearson_corr = summary_df['group_mean'].corr(dll_df['group_mean'], method='pearson')
            f.writelines(f'{csv_type[3].upper()} - Pearson Correlation test results: {pearson_corr}')
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - Pearson Correlation test failed, column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Pearson Correlation test failed, column mismatch likely", flush=True)
        try:
            pearson_inv = dll_df['group_mean'].corr(summary_df['group_mean'], method='pearson')
            f.writelines(f'{csv_type[3].upper()} - Pearson Inverse test results: {pearson_inv}\n')
        except:
            f.write(f"{csv_type[3].upper()} - Pearson Inverse test failed, column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Pearson Inverse test failed, column mismatch likely", flush=True)

        try:
            spearman_corr = summary_df['group_mean'].corr(dll_df['group_mean'], method='spearman')
            f.writelines(f'{csv_type[3].upper()} - Spearman Correlation test results: {spearman_corr}')
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - Spearman Correlation test failed, column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Spearman Correlation test failed, column mismatch likely", flush=True)
        
        try:
            spearman_inv = dll_df['group_mean'].corr(summary_df['group_mean'], method='spearman')
            f.writelines(f'{csv_type[3].upper()} - Spearman Inverse test results: {spearman_inv}')
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - Spearman Inverse test failed, column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Spearman Inverse test failed, column mismatch likely", flush=True)
        try:
            granger_caus = grangercausalitytests(forward_df, maxlag=2, verbose=True)
            f.writelines(f'{csv_type[3].upper()} - Granger Causality test results: {granger_caus}')
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - Granger Causality test failed, column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Granger Causality test failed, column mismatch likely", flush=True)
        try:
            granger_inv = grangercausalitytests(inv_df, maxlag=2, verbose=True)
            f.writelines(f'{csv_type[3].upper()} - Granger Inverse test results: {granger_inv}')
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - Granger Inverse test failed, column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Granger Inverse test failed, column mismatch likely", flush=True)
        f.write("########################################################################################################\n")
        f.write("#####################################     Multi-Comparison     #########################################\n")
        f.write("########################################################################################################\n")
        try:
            anova_results = stats.f_oneway(*[group['Mean'].values for name, group in bits_of_memory_df.groupby('Condition', observed=True)])
            f.writelines(f'{csv_type[3].upper()} - ANOVA test results: {anova_results}')
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - ANOVA test failed, column mismatch likely\n")
            print(f"{csv_type[3].upper()} - ANOVA test failed, column mismatch likely", flush=True)

        try:
            tukey_results = pairwise_tukeyhsd(bits_of_memory_df['Mean'], bits_of_memory_df['Condition'])
            f.writelines(f'{csv_type[3].upper()} - Tukey test results: {tukey_results}')
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - Tukey test failed, likely, column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Tukey test failed, likely, column mismatch likely", flush=True)

        try:
            kruskal_data = bits_of_memory_df[bits_of_memory_df['Generation'] == int(args.number_of_generations) - 1]
            kruskal_results = stats.kruskal(*[group['Mean'].values for name, group in kruskal_data.groupby('Condition', observed=True)])
            f.writelines(f"{csv_type[3].upper()} - Kruskal-Wallis test results: {kruskal_results}")
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - Kruskal-Wallis Test Failed, results likely identical or column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Kruskal-Wallis Test Failed, results likely identical or column mismatch likely", flush=True)

        try:
            comparison = MultiComparison(kruskal_data['Mean'], kruskal_data['Condition'])
            wilcox_results = comparison.allpairtest(stats.mannwhitneyu, method='bonferroni')
            f.writelines(f"{csv_type[3].upper()} - Bonferroni-Corrected Kruskal/Wilcox Test Results: {wilcox_results[0]}")
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - Bonferroni-Corrected Kruskal/Wilcox Test Failed, results likely identical or column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Bonferroni-Corrected Kruskal/Wilcox Test Failed, results likely identical or column mismatch likely", flush=True)

        try:
            posthoc_res = sp.posthoc_wilcoxon(a=bits_of_memory_df, val_col='Mean', group_col='Condition', p_adjust='bonferroni')
            f.write(f"{csv_type[3].upper()} - Post-hoc Wilcoxon Rank-sum test with Bonferonni correction Test Results:\n")
            f.writelines(str(posthoc_res))
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - Post-hoc Wilcoxon Rank-sum test with Bonferonni correction Test Failed, results likely identical or column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Post-hoc Wilcoxon Rank-sum test with Bonferonni correction Test Failed, results likely identical or column mismatch likely", flush=True)
        
        try:
            posthoc_res = sp.posthoc_wilcoxon(a=kruskal_data, val_col='Mean', group_col='Condition', p_adjust='bonferroni')
            f.write(f"{csv_type[3].upper()} - Post-hoc Wilcoxon Rank-sum test with Bonferonni correction Test Results on Kruskal-Data:\n")
            f.writelines(str(posthoc_res))
            f.write('\n')
        except:
            f.write(f"{csv_type[3].upper()} - Post-hoc Wilcoxon Rank-sum test with Bonferonni correction Test on Kruskal-Data Failed, results likely identical or column mismatch likely\n")
            print(f"{csv_type[3].upper()} - Post-hoc Wilcoxon Rank-sum test with Bonferonni correction Test on Kruskal-Data Failed, results likely identical or column mismatch likely", flush=True)
    f.close()

def format_coopdef_csv(args, type, n):
    coops_df = pd.read_csv(join_path(str(args.output_folder), f"{type}_df.csv"), header=None)
    column_names = ['Row_Label'] + list(range(len(coops_df.columns[1:-2]))) + ['Condition', 'Strategy']
    coops_df.columns = column_names
    coops_df.iloc[:, 1:] = coops_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    top_labels = coops_df.set_index('Row_Label').sum(axis=1).astype(float).nlargest(4).index
    coops_df = coops_df[coops_df['Row_Label'].isin(top_labels)]
    df_long = coops_df.melt(id_vars='Row_Label', var_name='Strategy', value_name='Frequency')
    df_long = df_long[df_long['Strategy'] != 'Condition']
    df_long = df_long[df_long['Strategy'] != 'Strategy']
    df_filt = df_long[df_long["Strategy"] < 10].copy()
    df_filt[f'Mean Average {type}'] = df_filt['Strategy']
    df_filt.drop(columns='Strategy', inplace=True)
    df_long = df_long[df_long['Strategy'] >= 10].copy()
    #cprint(df_filt)

    df_long[f"Mean Average {type}"] = df_long["Strategy"] // n
    df_long = df_long.groupby(["Row_Label", f"Mean Average {type}"], as_index=False)["Frequency"].mean()
    df_long[f'Mean Average {type}'] = df_long[f'Mean Average {type}'] + 9
    df_long = pd.concat([df_filt,df_long])
    print(df_long)

def coop_def_freq(args):
    # Plots top-k strategies in the first ten generations then the mean of every n subsequent generations
    # pdsummarymbits4coevimb1nomut_seed595704812032025230658
    top_k = 4
    n = 10
    print("Constructing Top 4 Strategy Coop-Defect Frequency Plot Over Time")
    
    # coops_df = format_coopdef_csv(args, 'coops', n)
    # defects_df = format_coopdef_csv(args, 'defects', n)

    # #plt.figure(figsize=(10, 6))
    # fig, axs = plt.subplots(ncols=2)
    # sns.barplot(data=coops_df, x='Mean Average coops', y='Frequency', hue='Row_Label', palette=sns.color_palette("tab10"), edgecolor='k', linewidth=0.5, ax=axs[0])
    # sns.barplot(data=defects_df, x='Mean Average defects', y='Frequency', hue='Row_Label', palette=sns.color_palette("tab10"), edgecolor='k', linewidth=0.5, ax=axs[1]) 
    # axs.tick_params(axis='x', labelrotation=45)

    df1 = pd.read_csv(f'{args.output_folder}/coops_df.csv', header=None)
    df2 = pd.read_csv(f'{args.output_folder}/defects_df.csv', header=None)

    df1.columns = ['key'] + [f'step_{i}' for i in range(df1.shape[1] - 1)]
    df2.columns = ['key'] + [f'step_{i}' for i in range(df2.shape[1] - 1)]
    df1['total'] = df1.iloc[:, 1:-1].sum(axis=1)
    df2['total'] = df2.iloc[:, 1:-1].sum(axis=1)
    combined_totals = df1[['key', 'total']].set_index('key') + df2[['key', 'total']].set_index('key')
    top_k_keys = combined_totals.nlargest(k := top_k, 'total').index  
    df1_top = df1[df1['key'].isin(top_k_keys)].set_index('key')
    df2_top = df2[df2['key'].isin(top_k_keys)].set_index('key')

    combined_df = df1_top.iloc[:, :-1].add(df2_top.iloc[:, :-1], fill_value=0)

    proportions_df = combined_df.div(combined_df.sum(axis=0), axis=1)
    df1_prop = df1_top.iloc[:, :-1].div(df1_top.iloc[:, :-1].sum(axis=0), axis=1)
    df2_prop = df2_top.iloc[:, :-1].div(df2_top.iloc[:, :-1].sum(axis=0), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for key in top_k_keys:
        axes[0].plot(df1_prop.columns, df1_prop.loc[key], label=key)
    axes[0].set_title('Cooperates')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Proportion')
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    for key in top_k_keys:
        axes[1].plot(df2_prop.columns, df2_prop.loc[key], label=key)
    axes[1].set_title('Defects')
    axes[1].set_xlabel('Generation')
    axes[1].legend()
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.suptitle(f'Top-{k} Organisms: Proportion of Cooperates and Defects Over Time', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.gca().set_facecolor('white') 
    plt.legend().set_visible(True)  
    plt.tight_layout()  
    plt.savefig(join_path(str(args.output_folder), "CoopDef_Frequency_Plot_Overtime.png"))

def coop_raw_freq(args, mem_cost):
    # Plots top-k strategies in the first ten generations then the mean of every n subsequent generations
    # pdsummarymbits4coevimb1nomut_seed595704812032025230658
    top_k = 4
    n = 10
    print("Constructing Top 4 Strategy Coop-Defect Frequency Raw Plot Over Time")

    # #plt.figure(figsize=(10, 6))
    # fig, axs = plt.subplots(ncols=2)
    # sns.barplot(data=coops_df, x='Mean Average coops', y='Frequency', hue='Row_Label', palette=sns.color_palette("tab10"), edgecolor='k', linewidth=0.5, ax=axs[0])
    # sns.barplot(data=defects_df, x='Mean Average defects', y='Frequency', hue='Row_Label', palette=sns.color_palette("tab10"), edgecolor='k', linewidth=0.5, ax=axs[1]) 
    # axs.tick_params(axis='x', labelrotation=45)

    df1 = pd.read_csv(f'{args.output_folder}/coops{mem_cost}_df.csv', header=None)
    df2 = pd.read_csv(f'{args.output_folder}/defects{mem_cost}_df.csv', header=None)

    df1.columns = ['key'] + [f'{i}' for i in range(df1.shape[1] - 1)]
    df2.columns = ['key'] + [f'{i}' for i in range(df2.shape[1] - 1)]
    df1['total'] = df1.iloc[:, 1:-1].sum(axis=1)
    df2['total'] = df2.iloc[:, 1:-1].sum(axis=1)
    combined_totals = df1[['key', 'total']].set_index('key') + df2[['key', 'total']].set_index('key')
    top_k_keys = combined_totals.nlargest(k := top_k, 'total').index  
    df1_top = df1[df1['key'].isin(top_k_keys)].set_index('key')
    df2_top = df2[df2['key'].isin(top_k_keys)].set_index('key')

    combined_df = df1_top.iloc[:, :-1].add(df2_top.iloc[:, :-1], fill_value=0)

    proportions_df = combined_df.div(combined_df.sum(axis=0), axis=1)
    df1_prop = df1_top.iloc[:, :-1]
    df2_prop = df2_top.iloc[:, :-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(df1_prop.columns))  # step indices
    bar_width = 0.8 / len(top_k_keys)

    for i, key in enumerate(top_k_keys):
        #axes[0].plot(df1_prop.columns, df1_prop.loc[key], label=key)
        axes[0].bar(x + i * bar_width, df1_prop.loc[key], width=bar_width, label=key)
    axes[0].set_xticks(x + bar_width * (len(top_k_keys) - 1) / 2)
    #axes[0].set_xticklabels(df1_prop.columns, rotation=45)
    axes[0].set_xticklabels([
        label if i % 4 == 0 else '' 
        for i, label in enumerate(df1_prop.columns)
    ], rotation=45)
    axes[0].set_title('Cooperates')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, which='minor', linestyle='--', linewidth=0.5)

    for i, key in enumerate(top_k_keys):
        #axes[1].plot(df2_prop.columns, df2_prop.loc[key], label=key)
        axes[1].bar(x + i * bar_width, df2_prop.loc[key], width=bar_width, label=key)
    axes[1].set_xticks(x + bar_width * (len(top_k_keys) - 1) / 2)
    #axes[1].set_xticklabels(df2_prop.columns, rotation=45)
    axes[1].set_xticklabels([
        label if i % 4 == 0 else '' 
        for i, label in enumerate(df2_prop.columns)
    ], rotation=45)
    axes[1].set_title('Defects')
    axes[1].set_xlabel('Generation')
    axes[1].legend()
    axes[1].grid(True, which='minor', linestyle='--', linewidth=0.5)

    if args.hard_defect != None:
        plt.suptitle(f'Top-{k} Organisms, Mem Cost {mem_cost}, HD: {str(args.hard_defect)}: Cooperates and Defects Over Time', fontsize=16)
    else:
        plt.suptitle(f'Top-{k} Organisms, Mem Cost {mem_cost}: Cooperates and Defects Over Time', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.gca().set_facecolor('white') 
    plt.legend().set_visible(True)  
    plt.tight_layout()
    if args.hard_defect != None:
        plt.savefig(join_path(str(args.output_folder), f"CoopDef_Raw_Frequency_Plot_{mem_cost}_hd{str(args.hard_defect)}_Overtime.png"))
    else:
        plt.savefig(join_path(str(args.output_folder), f"CoopDef_Raw_Frequency_Plot_{mem_cost}_Overtime.png"))

def strat_freq(args, strat_df):
    # Plots top-k strategies in the first ten generations then the mean of every n subsequent generations
    k = 4
    n = 10
    print("Constructing Top 4 Strategy Frequency Plot Over Time")
    strategies_df = pd.read_csv(join_path(str(args.output_folder), f"{strat_df}.csv"), header=None)
    # print(strategies_df)
    # print(strategies_df.dtypes.value_counts())
    # print("####")
    if '2' in strat_df:
        strategies_df.drop(strategies_df.columns[0], axis=1, inplace=True)
        strategies_df.drop(strategies_df.columns[-1], axis=1, inplace=True)
        strategies_df.columns = range(strategies_df.shape[1])
        for col in strategies_df.columns:
            if pd.api.types.is_float_dtype(strategies_df[col]) and not strategies_df[col].isna().any():
                strategies_df[col] = strategies_df[col].astype(int)
        # print(strategies_df.dtypes.value_counts())
        # print(strategies_df)
        # print("####")
        
    column_names = ['Row_Label'] + list(range(len(strategies_df.columns[1:-2]))) + ['Condition', 'Strategy']
    strategies_df.columns = column_names
    strategies_df.iloc[:, 1:] = strategies_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    top_labels = strategies_df.set_index('Row_Label').sum(axis=1).astype(float).nlargest(4).index
    strategies_df = strategies_df[strategies_df['Row_Label'].isin(top_labels)]
    df_long = strategies_df.melt(id_vars='Row_Label', var_name='Strategy', value_name='Frequency')
    df_long = df_long[df_long['Strategy'] != 'Condition']
    df_long = df_long[df_long['Strategy'] != 'Strategy']
    df_filt = df_long[df_long["Strategy"] < 10].copy()
    df_filt['Mean Strategy'] = df_filt['Strategy']
    df_filt.drop(columns='Strategy', inplace=True)
    df_long = df_long[df_long['Strategy'] >= 10].copy()
    #cprint(df_filt)

    df_long["Mean Strategy"] = df_long["Strategy"] // n
    df_long = df_long.groupby(["Row_Label", "Mean Strategy"], as_index=False)["Frequency"].mean()
    df_long['Mean Strategy'] = df_long['Mean Strategy'] + 9
    df_long = pd.concat([df_filt,df_long])
    #print(df_long)

    strategy_color_map = {
        "[False, True]~[True]": "green",
        "[True, False]~[False]": "red",
        "[True, True]~[True]": "blue",
        "[False, False]~[False]": "red",
        "[False, True]~[False]": "yellow",
        "[False]~[False]": "red",
        "[False]~[]": "red",
        "[True]~[]": "blue",
    }
    default_color = "black"
    df_long['Color'] = df_long['Row_Label'].map(strategy_color_map)#.fillna(default_color)

    pivot_df = df_long.pivot(index='Mean Strategy', columns='Row_Label', values='Frequency').fillna(0)
    n_groups = len(pivot_df)
    n_bars = len(pivot_df.columns)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, row_label in enumerate(pivot_df.columns):
        x_offset = x + i * bar_width
        y_vals = pivot_df[row_label].values
        color = strategy_color_map.get(row_label, default_color)

        ax.bar(
            x_offset, y_vals, bar_width, 
            label=row_label,
            edgecolor='k', linewidth=0.5
        )

    # X-axis ticks
    # ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
    # ax.set_xticklabels(pivot_df.index.astype(int), rotation=45)

    # plt.figure(figsize=(10, 6))
    # ax = sns.barplot(data=df_long, x='Mean Strategy', y='Frequency', hue='Row_Label', palette=sns.color_palette("tab10"), edgecolor='k', linewidth=0.5) 
    # ax.tick_params(axis='x', labelrotation=45)

    # plt.title(f"Top-{k} Strategy Frequency Over-Time" + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    # plt.gca().set_facecolor('white') 
    # plt.legend().set_visible(True)  
    # plt.tight_layout() 
    ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(pivot_df.index.astype(int), rotation=45)

    ax.set_xlabel("Mean Strategy")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Top-{k} Strategy Frequency Over-Time - Max Bits: {args.max_bits_of_memory} - H.D. = {args.hard_defect}")
    ax.legend(title="Strategy")
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout() 
    plt.savefig(join_path(str(args.output_folder), f"{strat_df}_Frequency_Plot_Overtime.png"))  

def strat_freq_nobound(args, strat_df):
    # Plots top-k strategies in the first ten generations then the mean of every n subsequent generations
    k = 4
    n = 10
    print("Constructing Unbounded Top 4 Strategy Frequency Plot Over Time")
    strategies_df = pd.read_csv(join_path(str(args.output_folder), f"{strat_df}.csv"), header=None)
    if '2' in strat_df:
        strategies_df.drop(strategies_df.columns[0], axis=1, inplace=True)
        strategies_df.drop(strategies_df.columns[-1], axis=1, inplace=True)
        strategies_df.columns = range(strategies_df.shape[1])
        for col in strategies_df.columns:
            if pd.api.types.is_float_dtype(strategies_df[col]) and not strategies_df[col].isna().any():
                strategies_df[col] = strategies_df[col].astype(int)
    column_names = ['Row_Label'] + list(range(len(strategies_df.columns[1:-2]))) + ['Condition', 'Strategy']
    strategies_df.columns = column_names
    strategies_df.iloc[:, 1:] = strategies_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    top_labels = strategies_df.set_index('Row_Label').sum(axis=1).astype(float).nlargest(4).index
    strategies_df = strategies_df[strategies_df['Row_Label'].isin(top_labels)]
    df_long = strategies_df.melt(id_vars='Row_Label', var_name='Strategy', value_name='Frequency')
    df_long = df_long[df_long['Strategy'] != 'Condition']
    df_long = df_long[df_long['Strategy'] != 'Strategy']
    df_long['Mean Strategy'] = df_long['Strategy']
    # df_filt = df_long[df_long["Strategy"] < 10].copy()
    # df_filt['Mean Strategy'] = df_filt['Strategy']
    # df_filt.drop(columns='Strategy', inplace=True)
    # df_long = df_long[df_long['Strategy'] >= 10].copy()

    # df_long["Mean Strategy"] = df_long["Strategy"] // n
    # df_long = df_long.groupby(["Row_Label", "Mean Strategy"], as_index=False)["Frequency"].mean()
    # df_long['Mean Strategy'] = df_long['Mean Strategy'] + 9
    # df_long = pd.concat([df_filt,df_long])

    # plt.figure(figsize=(10, 6))
    # ax = sns.barplot(data=df_long, x='Mean Strategy', y='Frequency', hue='Row_Label', palette=sns.color_palette("tab10"), edgecolor='k', linewidth=0.5) 
    # ax.tick_params(axis='x', labelrotation=45)

    # plt.title(f"Top-{k} Strategy Frequency Over-Time" + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    # plt.gca().set_facecolor('white') 
    # plt.legend().set_visible(True)  
    # plt.tight_layout()  
    # plt.savefig(join_path(str(args.output_folder), "Strategy_Frequency_Plot_Overtime_Unbounded.png")) 

    strategy_color_map = {
        "[False, True]~[True]": "green",
        "[True, False]~[False]": "red",
        "[True, True]~[True]": "blue",
        "[False, False]~[False]": "red",
        "[False, True]~[False]": "yellow",
        "[False]~[False]": "red",
        "[False]~[]": "red",
        "[True]~[]": "blue",
    }

    pivot_df = df_long.pivot(index='Mean Strategy', columns='Row_Label', values='Frequency').fillna(0)
    n_groups = len(pivot_df)
    n_bars = len(pivot_df.columns)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, row_label in enumerate(pivot_df.columns):
        x_offset = x + i * bar_width
        y_vals = pivot_df[row_label].values

        ax.bar(
            x_offset, y_vals, bar_width, 
            label=row_label,
            edgecolor='k', linewidth=0.5
        )

    ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(pivot_df.index.astype(int), rotation=45)

    ax.set_xlabel("Mean Strategy")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Top-{k} Strategy Frequency Over-Time - Max Bits: {args.max_bits_of_memory} - H.D. = {args.hard_defect}")
    ax.legend(title="Strategy")
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(join_path(str(args.output_folder), f"{strat_df}_Frequency_Plot_Overtime_Unbounded.png"))

def rate_freq_bound(args):
    # Plots top-k strategies in the first ten generations then the mean of every n subsequent generations
    k = 4
    n = 10
    print("filter h: ", args.filter_h)
    args.filter_h = 100
    print("Constructing Bounded Top 4 Coop Frequency Plot Over Time")
    strategies_df = pd.read_csv(join_path(str(args.output_folder), "coop_df.csv"), header=None)
    column_names = ['Row_Label'] + list(range(len(strategies_df.columns)-1))
    strategies_df.columns = column_names
    strategies_df.iloc[:, 1:] = strategies_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    top_labels = strategies_df.set_index('Row_Label').sum(axis=1).astype(float).nlargest(4).index
    strategies_df = strategies_df[strategies_df['Row_Label'].isin(top_labels)]
    df_long = strategies_df.melt(id_vars='Row_Label', var_name='Strategy', value_name='Frequency')
    df_long = df_long[df_long['Strategy'] != 'Strategy']
    df_long['Mean Strategy'] = df_long['Strategy']
    df_filt = df_long[df_long["Strategy"] < args.filter_h].copy()
    df_filt['Mean Strategy'] = df_filt['Strategy']
    df_filt.drop(columns='Strategy', inplace=True)
    df_long = df_long[df_long['Strategy'] >= args.filter_h].copy()

    df_long["Mean Strategy"] = df_long["Strategy"] // args.filter_h
    df_long = df_long.groupby(["Row_Label", "Mean Strategy"], as_index=False)["Frequency"].mean()
    df_long['Mean Strategy'] = df_long['Mean Strategy'] + (args.filter_h - 1)
    df_long = pd.concat([df_filt,df_long])

    strategy_color_map = {
        "[False, True]~[True]": "green",
        "[True, False]~[False]": "red",
        "[True, True]~[True]": "blue",
        "[False, False]~[False]": "red",
        "[True, False]~[True]": "red",
        "[False, False]~[True]": "orange",
        "[False, True]~[False]": "yellow",
        "[False]~[False]": "red",
        "[False]~[]": "red",
        "[True]~[]": "blue",
    }
    default_color = "black"
    df_long['Color'] = df_long['Row_Label'].map(strategy_color_map)#.fillna(default_color)

    pivot_df = df_long.pivot_table(index='Mean Strategy', columns='Row_Label', values='Frequency', aggfunc='mean').fillna(0)
    n_groups = len(pivot_df)
    n_bars = len(pivot_df.columns)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for i, row_label in enumerate(pivot_df.columns):
        x_offset = x + i * bar_width
        y_vals = pivot_df[row_label].values
        color = strategy_color_map.get(row_label, default_color)

        # ax.bar(
        #     x_offset, y_vals, bar_width, 
        #     label=row_label,
        #     edgecolor='k', linewidth=0.5
        # )

        ax.plot(x, y_vals, label=row_label, marker='o', color=color)

    ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(pivot_df.index.astype(int), rotation=45)

    ax.set_xlabel("Generation at $\it{k}$-timestep")
    ax.set_ylabel("Frequency")
    #ax.set_title(f"Top-{k} Strategy Frequency Over-Time - Max Bits: {args.max_bits_of_memory} - H.D. = {args.hard_defect}")
    ax.legend(title="Strategy")
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()  
    plt.savefig(join_path(str(args.output_folder), "Coop_Frequency_Plot_Overtime_Bounded.png")) 

def strat_freq_bound(args, strat_df):
    # Plots top-k strategies in the first ten generations then the mean of every n subsequent generations
    k = 4
    n = 10
    print("filter h: ", args.filter_h)
    print("Constructing Bounded Top 4 Strategy Frequency Plot Over Time")
    strategies_df = pd.read_csv(join_path(str(args.output_folder), f"{strat_df}.csv"), header=None)
    if '2' in strat_df:
        strategies_df.drop(strategies_df.columns[0], axis=1, inplace=True)
        strategies_df.drop(strategies_df.columns[-1], axis=1, inplace=True)
        strategies_df.columns = range(strategies_df.shape[1])
        for col in strategies_df.columns:
            if pd.api.types.is_float_dtype(strategies_df[col]) and not strategies_df[col].isna().any():
                strategies_df[col] = strategies_df[col].astype(int)
    column_names = ['Row_Label'] + list(range(len(strategies_df.columns[1:-2]))) + ['Condition', 'Strategy']
    strategies_df.columns = column_names
    strategies_df.iloc[:, 1:] = strategies_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    top_labels = strategies_df.set_index('Row_Label').sum(axis=1).astype(float).nlargest(4).index
    strategies_df = strategies_df[strategies_df['Row_Label'].isin(top_labels)]
    df_long = strategies_df.melt(id_vars='Row_Label', var_name='Strategy', value_name='Frequency')
    df_long = df_long[df_long['Strategy'] != 'Condition']
    df_long = df_long[df_long['Strategy'] != 'Strategy']
    df_long['Mean Strategy'] = df_long['Strategy']
    df_filt = df_long[df_long["Strategy"] < args.filter_h].copy()
    df_filt['Mean Strategy'] = df_filt['Strategy']
    df_filt.drop(columns='Strategy', inplace=True)
    df_long = df_long[df_long['Strategy'] >= args.filter_h].copy()

    print(len(df_long['Strategy']))
    df_long["Mean Strategy"] = df_long["Strategy"] // 10 #args.filter_h
   
    df_long = df_long.groupby(["Row_Label", "Mean Strategy"], as_index=False)["Frequency"].mean()
    df_long['Mean Strategy'] = df_long['Mean Strategy'] + (args.filter_h - 2)
    df_long = pd.concat([df_filt,df_long])

    strategy_color_map = {
        "[False]~[]": "red",
        "[False]~[False]": "red",
        "[False, False]~[False]": "red",
        "[True, False]~[True]": "red",
        "[False, False]~[True]": "orange",
        "[True, False]~[True]": "yellow",
        "[True, False]~[False]": "yellow",
        "[False, True]~[False]": "yellow",
        "[True, True]~[True]": "blue",
        "[True]~[]": "blue",
        "[False, True]~[True]": "green",
        "[False, False, False, False]~[False, False, False]": "red",
        "[False, False, True, False]~[False, False, False]": "red",
        "[False, True, False, False]~[False, False, False]": "red",
        "[False, False, False, True]~[False, False, False]": "red",
        "[False, True, True, False]~[False, False, False]": "red",
        "[False, False, True, False, False]~[False, True, False, False]": "red",
        "[False, True, False, True, True]~[False, False, False, True]":"green",
        "[False, True, False, True]~[True, True, True]":"green"
    }
    default_color = "black"
    df_long['Color'] = df_long['Row_Label'].map(strategy_color_map)#.fillna(default_color)

    pivot_df = df_long.pivot_table(index='Mean Strategy', columns='Row_Label', values='Frequency', aggfunc='mean').fillna(0)
    n_groups = len(pivot_df)
    n_bars = len(pivot_df.columns)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    for i, row_label in enumerate(pivot_df.columns):
        x_offset = x + i * bar_width
        y_vals = pivot_df[row_label].values
        color = strategy_color_map.get(row_label, default_color)

        ax.bar(
            x_offset, y_vals, bar_width, 
            label=row_label,
            edgecolor='k', linewidth=0.5, color=color
        )

    ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(pivot_df.index.astype(int), rotation=45, fontsize=12)
    ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=16)

    ax.set_xlabel("Generation at $\it{k}$-timestep", fontsize=24)
    ax.set_ylabel("Frequency", fontsize=24)
    #ax.set_title(f"Top-{k} Strategy Frequency Over-Time - Max Bits: {args.max_bits_of_memory} - H.D. = {args.hard_defect}")
    #x.legend(title="Strategy")

    plt.tight_layout()  
    plt.savefig(join_path(str(args.output_folder), f"{strat_df}_Frequency_Plot_Overtime_Bounded.png"), dpi=300) 
    plt.savefig(join_path(str(args.output_folder), f"{strat_df}_Frequency_Plot_Overtime_Bounded.eps"), dpi=300)


def strat_freq_split(args, strat_df):
    # Plots top-k strategies in the first ten generations then the mean of every n subsequent generations
    k = 4
    n = 10
    print("filter h: ", args.filter_h)
    print("Constructing Bounded Top 4 Strategy Frequency Plot Over Time")
    strategies_df = pd.read_csv(join_path(str(args.output_folder), f"{strat_df}.csv"), header=None)
    if '2' in strat_df:
        strategies_df.drop(strategies_df.columns[0], axis=1, inplace=True)
        strategies_df.drop(strategies_df.columns[-1], axis=1, inplace=True)
        strategies_df.columns = range(strategies_df.shape[1])
        for col in strategies_df.columns:
            if pd.api.types.is_float_dtype(strategies_df[col]) and not strategies_df[col].isna().any():
                strategies_df[col] = strategies_df[col].astype(int)
    column_names = ['Row_Label'] + list(range(len(strategies_df.columns[1:-2]))) + ['Condition', 'Strategy']
    strategies_df.columns = column_names
    strategies_df.iloc[:, 1:] = strategies_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    top_labels = strategies_df.set_index('Row_Label').sum(axis=1).astype(float).nlargest(4).index
    strategies_df = strategies_df[strategies_df['Row_Label'].isin(top_labels)]
    df_long = strategies_df.melt(id_vars='Row_Label', var_name='Strategy', value_name='Frequency')
    df_long = df_long[df_long['Strategy'] != 'Condition']
    df_long = df_long[df_long['Strategy'] != 'Strategy']
    df_long['Mean Strategy'] = df_long['Strategy']
    df_filt = df_long[df_long["Strategy"] < args.filter_h].copy()
    df_filt['Mean Strategy'] = df_filt['Strategy']
    df_filt.drop(columns='Strategy', inplace=True)
    df_long = df_long[df_long['Strategy'] >= args.filter_h].copy()

    strategy_color_map = {
        "[False]~[]": "red",
        "[False]~[False]": "red",
        "[False, False]~[False]": "red",
        "[True, False]~[True]": "red",
        "[False, False]~[True]": "orange",
        "[True, False]~[True]": "yellow",
        "[True, False]~[False]": "yellow",
        "[False, True]~[False]": "yellow",
        "[True, True]~[True]": "blue",
        "[True]~[]": "blue",
        "[False, True]~[True]": "green",
        "[False, False, False, False]~[False, False, False]": "red",
        "[False, False, True, False]~[False, False, False]": "red",
        "[False, True, False, False]~[False, False, False]": "red",
        "[False, False, False, True]~[False, False, False]": "red",
        "[False, True, True, False]~[False, False, False]": "red",
        "[False, False, False, False]~[True, True, True]": "red",
        "[False, True, True, False, True]~[True, True, True, False]":"blue",
        "[False, False, True, False, False]~[False, True, False, False]": "red",
        "[False, True, False, True, True]~[False, False, False, True]":"green",
        "[False, True, False, True]~[True, True, True]":"yellow",
        "[False, False, True, True]~[False, True, True]": "green"
    }
    default_color = "black"
    df_long['Color'] = df_long['Row_Label'].map(strategy_color_map)#.fillna(default_color)

    left_df  = df_filt.copy()
    right_df = df_long[df_long['Strategy'] >= args.filter_h].copy()

    right_df = (
        right_df.assign(**{"Mean Strategy": right_df["Strategy"] // 10})
                .groupby(["Row_Label", "Mean Strategy"], as_index=False)["Frequency"].mean()
    )
    right_df["Mean Strategy"] = right_df["Mean Strategy"] + (args.filter_h - 2)

    pivot_left  = left_df.pivot_table(index='Mean Strategy',  columns='Row_Label',
                                    values='Frequency', aggfunc='mean').fillna(0)
    pivot_right = right_df.pivot_table(index='Mean Strategy', columns='Row_Label',
                                    values='Frequency', aggfunc='mean').fillna(0)

    all_cols = sorted(set(pivot_left.columns) | set(pivot_right.columns))
    pivot_left  = pivot_left.reindex(columns=all_cols, fill_value=0)
    pivot_right = pivot_right.reindex(columns=all_cols, fill_value=0)

    n_bars    = len(all_cols)
    bar_width = 0.8 / max(n_bars, 1)

    n_left, n_right = len(pivot_left.index), len(pivot_right.index)
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12, 7), dpi=300,
        gridspec_kw={"width_ratios": [max(n_left, 1), max(n_right, 1)], "wspace": 0.08},
        sharey=True
    )

    for ax in (axL, axR):
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    xL = np.arange(n_left)
    for i, row_label in enumerate(all_cols):
        y_vals = pivot_left[row_label].values if n_left else []
        color  = strategy_color_map.get(row_label, default_color)
        axL.bar(xL + i*bar_width, y_vals, bar_width, label=row_label,
                edgecolor='k', linewidth=0.5, color=color)

    if n_left:
        axL.set_xticks(xL + bar_width*(n_bars - 1)/2)
        #axL.set_xticklabels(pivot_left.index.astype(int), rotation=45, fontsize=12)
        print(pivot_left.index.astype(int))
        axL.set_xticklabels(list(range(1, 26)), rotation=45, fontsize=12)
    else:
        axL.set_xticks([])

    xR = np.arange(n_right)
    for i, row_label in enumerate(all_cols):
        y_vals = pivot_right[row_label].values if n_right else []
        color  = strategy_color_map.get(row_label, default_color)
        axR.bar(xR + i*bar_width, y_vals, bar_width,
                edgecolor='k', linewidth=0.5, color=color)

    if n_right:
        axR.set_xticks(xR + bar_width*(n_bars - 1)/2)
        #axR.set_xticklabels(pivot_right.index.astype(int), rotation=45, fontsize=12)
        axR.set_xticklabels(list(range(200, 1001, 100)), rotation=45, fontsize=12)
    else:
        axR.set_xticks([])

    axL.legend(title="Mean Strategy")
    axL.set_ylabel("Frequency", fontsize=24)
    #for ax in (axL, axR):
    axL.tick_params(axis='y', labelsize=16)
    #axL.set_xlabel("Generation at $\\it{k}$-timestep", fontsize=24)
    fig.supxlabel("Generation at $\\it{k}$-timestep", fontsize=24)
    ymin = min(axL.get_ylim()[0], axR.get_ylim()[0])
    ymax = max(axL.get_ylim()[1], axR.get_ylim()[1])
    axL.set_ylim(ymin, ymax)
    axR.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)  
    plt.savefig(join_path(str(args.output_folder), f"{strat_df}_Frequency_Plot_Overtime_Bounded_Split.png"), dpi=300, bbox_inches="tight")
    plt.savefig(join_path(str(args.output_folder), f"{strat_df}_Frequency_Plot_Overtime_Bounded_Split.eps"), dpi=300, bbox_inches="tight")


def most_freq_k_strat(args):
    #TODO: Doesn't work right, just displays first gen strategies?
    k = 4
    print("Constructing Top 4 Strategy Frequency Plot")
    strategies_df = pd.read_csv(join_path(str(args.output_folder), "strategies_df.csv"), header=None)
    column_names = ['Row_Label'] + list(range(len(strategies_df.columns[1:-2]))) + ['Condition', 'Strategy']
    strategies_df.columns = column_names

    strategies_df.drop('Row_Label', axis=1, inplace=True)
    strategies_df = pd.melt(strategies_df, id_vars=['Condition', 'Strategy'], var_name='Generation', value_name='Frequency')
    strategies_df['Generation'] = pd.to_numeric(strategies_df['Generation'])
    strategies_df['Frequency'] = pd.to_numeric(strategies_df['Frequency'])

    plt.figure(figsize=(10, 6)) 
    top_strategies = strategies_df['Strategy'].value_counts().nlargest(k).index

    filtered_df = strategies_df[strategies_df['Strategy'].isin(top_strategies)]
    filtered_df.groupby('Strategy')['Frequency'].sum().plot(kind='bar')
    plt.title(f"Top-{k} Strategy Frequency" + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.gca().set_facecolor('white') 
    plt.legend().set_visible(True)  
    plt.tight_layout()  
    plt.savefig(join_path(str(args.output_folder), "Strategy_Frequency_Plot.png")) 

def common_strats(args):
    print("Constructing Common Strategy Plot")
    most_common_strategy = pd.read_csv(join_path(args.output_folder, "most_common.csv"))
    #File currently saves Condition with a space in front
    most_common_strategy[' Condition'] = pd.to_numeric(most_common_strategy[' Condition'], errors='coerce') 
    most_common_strategy['Common_Strategy'] = most_common_strategy['Common_Strategy'].astype(str)

    subset_data = most_common_strategy[(most_common_strategy[' Condition'] != -0.5) & (most_common_strategy[' Condition'] != 0.0)]
    dat = subset_data.pivot_table(index='Common_Strategy', columns=' Condition', aggfunc='size', fill_value=0)
    dat_melt = dat.reset_index().melt(id_vars='Common_Strategy', var_name=' Condition', value_name='Frequency')

    order = dat_melt['Common_Strategy'].value_counts().nlargest(4).index.tolist()
    dat_melt = dat_melt[dat_melt['Common_Strategy'].isin(order)]
    dat_melt['Common_Strategy'] = pd.Categorical(dat_melt['Common_Strategy'], categories=order, ordered=True)
    dat_melt.sort_values(by='Common_Strategy', inplace=True)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Common_Strategy', y='Frequency', data=dat_melt) #, hue=' Condition' , dodge=True , palette=["#8CF582", "#82F5EB", "#828BF5", "#82B5EF"]
    plt.title('Most Common Strategies' + f' - Max Bits: ' + str(args.max_bits_of_memory) + f' - H.D. =  {str(args.hard_defect)}')
    plt.xticks(rotation=10)
    plt.xlabel('Strategy')
    plt.ylabel('Frequency')
    plt.grid(axis='y')

    for container in plt.gca().containers:
        plt.bar_label(container, label_type='edge')
    plt.savefig(join_path(args.output_folder, "Most_Common_Strategies.png"))

def plot_heat_corr(output_folder, mem_cost):
    # Need to add index based on timesteps manually since current compile_csv doesnt.

    df = pd.read_csv(join_path(output_folder, f"coops{mem_cost}_df.csv"), header=None) # Current code didnt add timesteps so need to add manually
    df.reset_index(drop=True, inplace=True)
    # # df.reset_index(drop=True, inplace=True)
    # # df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
    # #df = pd.concat([df, orig_header], ignore_index=True)

    # # print(df.head())
    # # print(df.columns)
    # # print(df.index)

    # # #df.rename(columns={df.columns[0]: 'Genotype'}, inplace=True)
    # print(df.head())
    # exit()

    df.rename(columns={df.columns[0]: 'Genotype'}, inplace=True)

    df = df.set_index("Genotype")
    df.columns = df.columns.astype(float)
    df = df.apply(pd.to_numeric, errors='coerce')

    df = df[df.max(axis=1) > 32]
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.T, cmap='viridis', cbar_kws={'label': 'Move Frequency'})
    # for genotype in df.index:
    #     plt.plot(df.columns, df.loc[genotype], label=genotype)
    plt.xlabel("Genotype")
    plt.ylabel("Timestep")
    plt.title("Strategy Population Over Time")
    plt.tight_layout()
    #plt.show()
    plt.savefig(join_path(output_folder, f'Heat_Correlation.png'))

def main():
    arg_parser = argparse.ArgumentParser(
        description='Plotting function for handling DataFrames.')
    
    # Expects 1 argument: output folder
    arg_parser.add_argument("-o", "--output_folder", type=str, default="tests/pd_temp")
    arg_parser.add_argument("--output_frequency", type=int, default=10)
    arg_parser.add_argument("--number_of_generations", type=int, default=500)
    arg_parser.add_argument("-max_m", "--max_bits_of_memory", type=int, default=1)
    arg_parser.add_argument("-hf", "--hard_defect", nargs='*', help='16, 32, 48', type=str)
    arg_parser.add_argument("--mem_cost", nargs='*', help='0.0, 0.00075, 0.0075, 0.075, 0.75', type=str, default='0.0, 0.00075, 0.0075, 0.075, 0.75')
    arg_parser.add_argument("-fh", "--filter_h", help='20', type=int, default=25)
    
    args = arg_parser.parse_args()

    integers = re.findall(r'\b\d+\b', str(args.hard_defect))
    args.hard_defect = str(list(map(int, integers))) # yes this is a brute force but it ensures no breakage

    # if 'hybrid' in args.output_folder: 
    #     csv_list = ['all_bits_df_Memory_comp_more_values.csv', 'all_bits_df_Summary_comp_more_values.csv', 'all_bits_df_Total_comp_more_values.csv']
    # else: csv_list = ['all_bits_df_Memory_comp_more_values.csv']
    # if os.path.exists(join_path(args.output_folder, "stat_output.txt")):
    #     os.remove(join_path(args.output_folder, "stat_output.txt"))
    # for csv in csv_list:
    #     average_mem(args, csv)
    # average_dll(args)
    #average_fitness(args)
    # most_freq_k_strat(args)
    #strat_freq(args, "strategies_df")
    #strat_freq_nobound(args, "strategies_df")

    #strat_freq_bound(args, "strategies_df")
    strat_freq_split(args, "strategies_df")

    # strat_freq(args, "strategies_df2")
    # strat_freq_nobound(args, "strategies_df2")
    #strat_freq_bound(args, "strategies_df2")

    # plot_heat_corr(args.output_folder, args.mem_cost[0]) # currently only works across multiple memory conditions

    # rate_freq_bound(args)
    # #common_strats(args)
    # #coop_def_freq(args)

    # print("args.mem_cost: ", args.mem_cost)
    # print(type(args.mem_cost))
    # print([args.mem_cost])
    # if type(args.mem_cost) == list: args.mem_cost = args.mem_cost[0] #assuming ['0.0, 0.00075, 0.0075, 0.075, 0.75']
    # if type(args.mem_cost) == str:
    #     mem_list = [str(x.strip()) for x in args.mem_cost.split(',')]
    # else: mem_list = args.mem_cost
    # print(mem_list)

    # for mem_cost in mem_list:
    #     coop_raw_freq(args, mem_cost)
    

if __name__ == "__main__":
    main()

