import pandas as pd
from Bio import SeqIO
from itertools import product
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import argparse
from math import floor
from colorama import Fore, Back, Style, init

# Initialize colorama
init()

# Function to read sequences from fasta files
def ReadFasta(file):
    sequences = {}
    for record in SeqIO.parse(file, "fasta"):
        sequenceId = record.id.split('|')[1]  # Extract the sequence ID
        sequences[sequenceId] = str(record.seq)
    return sequences

# Function to generate all possible 2-mers of amino acids
def GenerateKMers(kMerLength=2): #ahah two-mers cause cancer
    aminoAcids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    return [''.join(p) for p in product(aminoAcids, repeat=kMerLength)]

# Function to calculate FFP for a given sequence
def CalculateFFP(sequence, kMers, kMerLength=2):
    ffp = dict.fromkeys(kMers, 0)
    for i in range(len(sequence) - 1):
        dinucleotide = sequence[i:i+kMerLength]
        if dinucleotide in ffp:
            ffp[dinucleotide] += 1
            
    total = sum(ffp.values())

    for p in ffp:
        ffp[p] /= total
    return list(ffp.values())

# Function to create a pandas dataframe with FFP values
def CreateDataframe(sequences, kMers, label, kMerLength=2):
    data = []
    for seq_id, seq in sequences.items():
        ffp_values = CalculateFFP(seq, kMers, kMerLength)
        ffp_values.append(label)
        data.append([seq_id] + ffp_values)
    columns = ['id'] + kMers + ['class']
    return pd.DataFrame(data, columns=columns)

# Function to evaluate models
def evaluate_model(featureMatrix, labelVector, model, crossValidator, scoring='accuracy'):
    scores = cross_val_score(model, featureMatrix, labelVector, cv=crossValidator, scoring=scoring)
    return scores.mean(), scores.std()


# TODO: Adicionar logging para ficar bonitao como eu
# TODO: Dividir em varios ficheiros tipo o antigo? 
# TODO: Confirmar os valores, parece muito OP, mas e capaz de ser mesmo assim porque é como eu
def main():
    parser = argparse.ArgumentParser(description='Protein sequence classification')
    parser.add_argument('-a', '--file_a', required=True, help='Path to the first fasta file', default='sequences/globin.fasta')
    parser.add_argument('-b', '--file_b', required=True, help='Path to the second fasta file', default='sequences/zincfinger.fasta')
    parser.add_argument('-k', '--kmer_length', type=int, default=2, help='Length of k-mer')
    
    args = parser.parse_args()
    
    firstSequence = ReadFasta(args.file_a)
    secondSequence = ReadFasta(args.file_b)
    kMersLength = args.kmer_length
    kMers = GenerateKMers(kMersLength)
    
    firstDataframe = CreateDataframe(firstSequence, kMers, 0, kMersLength)
    secondDataframe = CreateDataframe(secondSequence, kMers, 1, kMersLength)
    df = pd.concat([firstDataframe, secondDataframe])
    
    featureMatrix = df.iloc[:, 1:-1].values
    labelVector = df['class'].values
    
    crossValidator = StratifiedKFold(n_splits=10)
    
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'NaiveBayes': GaussianNB()
    }

    scoringTypes = ['accuracy', 'precision', 'recall', 'f1']

    results = {}
    for model_name, model in models.items():
        for scoring in scoringTypes:
            print(f"Evaluating {Fore.CYAN} {model_name} {Fore.RESET} with scoring {Fore.MAGENTA}{scoring}{Style.RESET_ALL}")
            mean, std = evaluate_model(featureMatrix, labelVector, model, crossValidator, scoring)
            if model_name not in results:
                results[model_name] = {}
            results[model_name][scoring] = {'mean': mean, 'std': std}
        print()

    for model_name, model in results.items():
        model_name_length = len(model_name)
        target = floor((50 - model_name_length - 2)/2)
        print("\n" + "="*target + " " + Fore.CYAN + model_name + Fore.RESET + " " + "="*target + ("=" if model_name_length % 2 != 0 else ""))
        for scoring, values in model.items():
            print(f"{Fore.MAGENTA}{scoring}{Style.RESET_ALL}: {Fore.GREEN} {round(values['mean'], 2)}  {Fore.BLUE}({round(values['std'], 2)}) {Fore.RESET}")
        print("="*50 + "\n")

if __name__ == '__main__':
    main()
