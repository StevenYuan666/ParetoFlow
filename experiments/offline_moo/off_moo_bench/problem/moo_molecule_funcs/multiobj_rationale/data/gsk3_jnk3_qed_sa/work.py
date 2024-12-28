import sys

import rdkit.Chem.QED as QED
import sascorer as sascorer
from rdkit import Chem
from rdkit.Chem import Descriptors

for line in sys.stdin:
    rat, smiles = line.split()
    mol = Chem.MolFromSmiles(smiles)
    print(
        rat, smiles, QED.qed(mol), sascorer.calculateScore(mol), Descriptors.MolWt(mol)
    )
