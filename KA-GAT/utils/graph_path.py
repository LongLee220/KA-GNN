import dgl
import torch
import numpy as np
import networkx as nx
import math

from rdkit import Chem
from rdkit.Chem import AllChem
from jarvis.core.specie import chem_data, get_node_attributes
from rdkit.Chem import rdchem


def calculate_dis(A,B):
    AB = B - A
    dis = np.linalg.norm(AB)
    return dis








def bond_length_approximation(bond_type):
    bond_length_dict = {"SINGLE": 1.0, "DOUBLE": 1.4, "TRIPLE": 1.8, "AROMATIC": 1.5}
    return bond_length_dict.get(bond_type, 1.0)

def encode_bond_14(bond):
    #7+4+2+2+6 = 21
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1
    
    # bond_type = [0] * 4
    # bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1

    bond_type = [0, 0, 0, 0]
    bt = bond.GetBondType()
    if bt == rdchem.BondType.SINGLE:
        bond_type[0] = 1
    elif bt == rdchem.BondType.DOUBLE:
        bond_type[1] = 1
    elif bt == rdchem.BondType.TRIPLE:
        bond_type[2] = 1
    elif bt == rdchem.BondType.AROMATIC:
        bond_type[3] = 1
    
    bond_length = bond_length_approximation(bond.GetBondType())
    
    in_ring = [0, 0]
    in_ring[int(bond.IsInRing())] = 1
    
    non_bond_feature = [0]*6

    edge_encode = bond_dir + bond_type + [bond_length,bond_length**2] + in_ring + non_bond_feature

    return edge_encode



def non_bonded(charge_list,i,j,dis):
    charge_list = [float(charge) for charge in charge_list] 
    q_i = [charge_list[i]]
    q_j = [charge_list[j]]
    q_ij = [charge_list[i]*charge_list[j]]
    dis_1 = [1/dis]
    dis_2 = [1/(dis**6)]
    dis_3 = [1/(dis**12)]

    return q_i + q_j + q_ij + dis_1 + dis_2 +dis_3





def mmff_force_field(mol):
    try:
       
        AllChem.EmbedMolecule(mol)
        
        AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        return True
    except ValueError:
       
        return False


def uff_force_field(mol):
    try:
        
        AllChem.EmbedMolecule(mol)
        
        AllChem.UFFGetMoleculeForceField(mol)
        return True
    except ValueError:
        
        return False
    
def random_force_field(mol):
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
        return True
    except ValueError:
       
        return False



def check_common_elements(list1, list2, element1, element2):
    for i in range(len(list1)):
        if list1[i] == element1 and list2[i] == element2:
            return True  
    
    return False 

def tensor_nan_inf(per_bond_feat):
    nan_exists = any(math.isnan(x) if isinstance(x, float) else False for x in per_bond_feat)
    inf_exists = any(x == float('inf') if isinstance(x, float) else False for x in per_bond_feat)
    ninf_exists = any(x == float('-inf') if isinstance(x, float) else False for x in per_bond_feat)

    if nan_exists or inf_exists or ninf_exists:
        
        clean_list = [0 if isinstance(x, float) and math.isnan(x) else x for x in per_bond_feat]
        
        per_bond_feat = [1 if x == float('inf') else -1 if x == float('-inf') else x for x in clean_list]
        return per_bond_feat
    else:
        return per_bond_feat
    

    
def atom_to_graph(smiles,encoder_atom,encoder_bond):
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    else:
        mol = Chem.AddHs(mol) 
    sps_features = []
    coor = []
    edge_id = []
    atom_charges = []
    
    smiles_with_hydrogens = Chem.MolToSmiles(mol)

    tmp = []
    for num in smiles_with_hydrogens:
        if num not in ['[',']','(',')']:
            tmp.append(num)

    sm = {}
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        sm[atom_index] =  atom.GetSymbol()

    Num_toms = len(tmp)
    if Num_toms > 700:
        g = False

    else:
        if mmff_force_field(mol) == True:
            num_conformers = mol.GetNumConformers()
            
            if num_conformers > 0:
                AllChem.ComputeGasteigerCharges(mol)
                for ii, s in enumerate(mol.GetAtoms()):

                    per_atom_feat = []
                            
                    feat = list(get_node_attributes(s.GetSymbol(), atom_features=encoder_atom))
                    per_atom_feat.extend(feat)

                    sps_features.append(per_atom_feat )
                        
                    
                    
                    pos = mol.GetConformer().GetAtomPosition(ii)
                    coor.append([pos.x, pos.y, pos.z])

                    
                    charge = s.GetProp("_GasteigerCharge")
                    atom_charges.append(charge)

                edge_features = []
                src_list, dst_list = [], []
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondTypeAsDouble() 
                    src = bond.GetBeginAtomIdx()
                    dst = bond.GetEndAtomIdx()

                    src_list.append(src)
                    src_list.append(dst)
                    dst_list.append(dst)
                    dst_list.append(src)

                    src_coor = np.array(coor[src])
                    dst_coor = np.array(coor[dst])

                    s_d_dis = calculate_dis(src_coor,dst_coor)

                    per_bond_feat = []
                       
                    per_bond_feat.extend(encode_bond_14(bond))

                    edge_features.append(per_bond_feat)
                    edge_features.append(per_bond_feat)
                    edge_id.append([1])
                    edge_id.append([1])
                #print(edge_features)
                # cutoff <= 5
                for i in range(len(coor)):
                    coor_i =  np.array(coor[i])
                    
                    for j in range(i+1, len(coor)):
                        coor_j = np.array(coor[j])

                        s_d_dis = calculate_dis(coor_i,coor_j)

                        if 0< s_d_dis <= 5:
                            #if check_common_elements(src_list,dst_list,i,j):
                            if (not check_common_elements(src_list, dst_list, i, j) and not check_common_elements(src_list, dst_list, j, i)):
                            
                                src_list.extend([i,j])
                    
                                dst_list.extend([j,i])
                                per_bond_feat = [0]*15
                                per_bond_feat.extend(non_bonded(atom_charges,i,j,s_d_dis))
                                #nan_exists = any(math.isnan(x) if isinstance(x, float) else False for x in per_bond_feat)
                                clean_list = tensor_nan_inf(per_bond_feat)
                                
                                edge_features.append(clean_list)
                                edge_features.append(clean_list)

                                edge_id.append([0])
                                edge_id.append([0])

                coor_tensor = torch.tensor(coor, dtype=torch.float32)
                edge_feats = torch.tensor(edge_features, dtype=torch.float32)
                edge_id_feats = torch.tensor(edge_id, dtype=torch.float32)

                node_feats = torch.tensor(sps_features,dtype=torch.float32)

                
                # Number of atoms
                num_atoms = mol.GetNumAtoms()

                # Create a graph. undirected_graph
                g = dgl.DGLGraph()
                g.add_nodes(num_atoms)
                g.add_edges(src_list, dst_list)
                
                g.ndata['feat'] = node_feats
                g.ndata['coor'] = coor_tensor  
                g.edata['feat'] = edge_feats
                g.edata['id'] = edge_id_feats
                #print("Updata G: min =", edge_feats.min().item(), ", max =", edge_feats.max().item())
        
            else:
                g = False
        else:
            g = False
    return g





def path_complex_mol(Smile, encoder_atom,encoder_bond):
    
    g = atom_to_graph(Smile,encoder_atom,encoder_bond)
    
    if g != False:
        return g
    else:
        return False


if __name__ == '__main__':
    
    smiles = '[H]C([H])([H])C([H])([H])[H]'
    encoder_atom = "cgcnn"
    encoder_bond = "dim_14"
    encode_two_path = "dim_8"
    encode_tree_path = "dim_6"
    encoder_bond = encode_two_path,encode_tree_path
    Graph_list = path_complex_mol(smiles, encoder_atom,encoder_bond)
    print(Graph_list)
