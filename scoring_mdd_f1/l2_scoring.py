# %%


import csv 
import argparse
import pandas as pd
import os 
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', help='path to input CSV file')
parser.add_argument('--out_dir', help='path to save')
args = parser.parse_args()

df = pd.read_csv(args.csv_file)
out_dir = args.out_dir
# Open the output files for writing
with open(out_dir+'/ref', 'w') as f_ref, open(out_dir+'/hyp', 'w') as f_hyp, \
     open(out_dir+'/decode', 'w') as f_pred:
    
    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        
        # Get the ID, ref, anno, and pred columns
        ID = row['path']
        ID = ID.replace(' ','')
        ref = row['ref']
        anno = row['anno']
        pred = row['predicted']
        
        # Write the ref, hyp, and pred to their respective files
        f_ref.write(f'{ID} {ref}\n')
        f_hyp.write(f'{ID} {anno}\n')
        f_pred.write(f'{ID} {pred}\n')


os.chdir('/alt/asr/yelkheir/kaldi/egs/gop_ls/s5')
# Define the command to be executed
command = ['/alt/asr/yelkheir/wav2vec_arabic/scoring/scoring.sh', 
           out_dir+'/ref',
           out_dir+'/hyp',
           out_dir+'/ref_human_detail'
           ]

# Use subprocess.run to execute the command
subprocess.run(' '.join(command), shell=True, cwd='/alt/asr/yelkheir/kaldi/egs/gop_ls/s5')

command = ['/alt/asr/yelkheir/wav2vec_arabic/scoring/scoring.sh', 
           out_dir+'/hyp',
           out_dir+'/decode',
           out_dir+'/human_our_detail'
           ]

# Use subprocess.run to execute the command
subprocess.run(' '.join(command), shell=True, cwd='/alt/asr/yelkheir/kaldi/egs/gop_ls/s5')


command = ['/alt/asr/yelkheir/wav2vec_arabic/scoring/scoring.sh', 
           out_dir+'/ref',
           out_dir+'/decode',
           out_dir+'/ref_our_detail'
           ]

# Use subprocess.run to execute the command
subprocess.run(' '.join(command), shell=True, cwd='/alt/asr/yelkheir/kaldi/egs/gop_ls/s5')


command = ['python3',
           '/alt/asr/yelkheir/wav2vec_arabic/scoring/analysis.py',
           '--INPUT_DIR',
           out_dir]

# Use subprocess.run to execute the command
subprocess.run(' '.join(command), shell=True)
# %%