from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

def calculate_molecular_properties(smiles):
    """
    Calculate molecular properties from a SMILES string.
    
    Args:
        smiles (str): A SMILES representation of the molecule.
        
    Returns:
        dict: A dictionary containing molecular properties.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    
    properties = {
        'Molecular Weight': Descriptors.MolWt(molecule),
        'LogP': Descriptors.MolLogP(molecule),
        'Num Rotatable Bonds': Descriptors.NumRotatableBonds(molecule),
        'Num H Donors': Descriptors.NumHDonors(molecule),
        'Num H Acceptors': Descriptors.NumHAcceptors(molecule),
    }
    
    return properties

def smiles_to_dataframe(smiles_list):
    """
    Convert a list of SMILES strings to a DataFrame of molecular properties.
    
    Args:
        smiles_list (list): A list of SMILES strings.
        
    Returns:
        pd.DataFrame: A DataFrame containing molecular properties for each SMILES.
    """
    data = []
    for smiles in smiles_list:
        properties = calculate_molecular_properties(smiles)
        if properties:
            properties['SMILES'] = smiles
            data.append(properties)
    
    return pd.DataFrame(data)