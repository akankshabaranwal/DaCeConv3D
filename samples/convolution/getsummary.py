import pandas as pd
import argparse
import csv

parser = argparse.ArgumentParser(description='get summary stats')
parser.add_argument('-type','--type', type=str, required=True, help='select which summary stats')
parser.add_argument('-prefix','--prefix', type=str, required=True, help='select which function')

results = {}

args = parser.parse_args()
df = pd.read_csv (f'{args.prefix}_{args.type}.csv')
new_header = df.iloc[0]
x = df.columns[1]
results[args.prefix] = df[x].sum()
fields = [args.prefix, results[args.prefix]]
with open(f'summary_{args.type}.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
print(results)
