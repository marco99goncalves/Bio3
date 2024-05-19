import pandas as pd
from Bio import SeqIO
from itertools import product
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import argparse

from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

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
        # 'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'NaiveBayes': GaussianNB()
    }

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    results = {}
    with Progress() as progress:
        tasks = []
        for model_name, model in models.items():
            tasks.append(progress.add_task(f"Evaluating [cyan]{model_name}[/]", total=len(metrics)))

        task = 0
        for model_name, model in models.items():
            for scoring in metrics:
                mean, std = evaluate_model(featureMatrix, labelVector, model, crossValidator, scoring)
                if model_name not in results:
                    results[model_name] = {}
                results[model_name][scoring] = {'mean': mean, 'std': std}
                progress.update(tasks[task], advance=1)
            task += 1
    console.print()




    resultsTable = Table(show_lines=True)
    resultsTable.add_column("Model", style="cyan", justify="center")
    for scoring in metrics:
        resultsTable.add_column(scoring, style="magenta", justify="center")
        resultsTable.add_column(f"{scoring} std", style="blue", justify="center")
    
    for model_name, model in results.items():
        row = [model_name]
        for scoring in metrics:
            row.append(str(round(model[scoring]['mean'], 2)))
            row.append(str(round(model[scoring]['std'], 2)))
        resultsTable.add_row(*row)


    console.print(resultsTable)

if __name__ == '__main__':
    main()
