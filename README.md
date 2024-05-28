
# Protein Sequence Classification

## How to run
To run this project, first create a python enviroment using
```
python3 -m venv venv
```
and activate it
```
source venv/bin/activate
```

After this, install the requirements for the project using 
```
pip install -r requirements.txt
```

Finally, you can run the project using 
```
python3 classify_proteins.py -a [PATH_TO_FIRST_FILE] -b [PATH_TO_SECOND_FILE] -k [K-Mer number]
```
### Parameters

- [Optional | Default: "globin.fasta"] **-a**: The path to the first file 
- [Optional | Default: "zincfinger.fasta"] **-b**: The path to the second file 
- [Optional | Default: 2] **-k**: An integer that specifies how the size of the K-mer's to use.
- [Optional] **-h**: Prints a help message to the user


