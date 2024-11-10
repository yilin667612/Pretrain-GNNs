import pandas as pd 
from rdkit import DataStructs, Chem, MACCSkeys, AllChem 
import numpy as np
import torch
import random 

def structure_sim(smiles_list):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    fps = [Chem.RDKFingerprint(m) for m in mols]
    
    sm_list = []
    
    for i in range(len(smiles_list)):
        sm = torch.zeros(10, 2)
        n = 0
        for j in random.sample(range(len(smiles_list)), 10):
            sm[n][0] = j
            # Tanimoto
            sm[n][1] = DataStructs.FingerprintSimilarity(fps[i], fps[j])
            # Dice
            # sm[n][1] = DataStructs.FingerprintSimilarity(fps[i],fps[j],metric=DataStructs.DiceSimilarity)
            # sm[n][1] = DataStructs.DiceSimilarity(fps[i],fps[j])
            n += 1
        sm_list.append(sm)
        print(i)
    
    return torch.stack(sm_list)

def main():
    filename = 'zinc_standard_agent'
    zinc_data = pd.read_csv('/home/dyl/Graph-code/pretrain-gnns/chem/dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz', sep=',', compression='gzip', dtype='str')
    smiles_list = zinc_data['smiles']
    
    similarity_matrix = structure_sim(smiles_list)
    
    torch.save(similarity_matrix, '/home/dyl/Graph-code/pretrain-gnns/chem/dataset/Morgan/Morgan_tanimoto') 
    print("done")

if __name__ == "__main__":
    main()
