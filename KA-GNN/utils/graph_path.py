import dgl
import torch
import numpy as np
import networkx as nx


from rdkit import Chem
from rdkit.Chem import AllChem
from jarvis.core.specie import chem_data, get_node_attributes
from rdkit.Chem import rdchem




def normalize_columns_01(input_tensor):
    # Assuming input_tensor is a 2D tensor (matrix)
    min_vals, _ = input_tensor.min(dim=0)
    max_vals, _ = input_tensor.max(dim=0)

    # Identify columns where max and min are not equal
    non_zero_mask = max_vals != min_vals

    # Avoid division by zero
    normalized_tensor = input_tensor.clone()
    normalized_tensor[:, non_zero_mask] = (input_tensor[:, non_zero_mask] - min_vals[non_zero_mask]) / (max_vals[non_zero_mask] - min_vals[non_zero_mask] + 1e-10)

    return normalized_tensor



def calculate_dis(A,B):
    AB = B - A
    dis = np.linalg.norm(AB)
    return dis


def calculate_angle(A, B, C):
    # vector(AB), vector(BC),# angle = arccos(AB*BC/|AB||BC|)
    AB = B - A
    BC = C - B

    dot_product = np.dot(AB, BC)
    norm_AB = np.linalg.norm(AB)
    norm_BC = np.linalg.norm(BC)
    if norm_AB * norm_BC != 0:
        cos_theta = dot_product / (norm_AB * norm_BC)
    else:
        cos_theta = 0.0  
    #cos_theta = dot_product / (norm_AB * norm_BC)
    if -1 <= cos_theta <= 1:
        angle_rad = np.arccos(cos_theta)
    else:
        angle_rad = 0 
    return angle_rad



def encode_chirality(atom):
    chirality_tags = [0] * 4  # Assuming 4 possible chirality tags
    
    if atom.HasProp("_CIPCode"):
        chirality = atom.GetProp("_CIPCode")
        if chirality == "R":
            chirality_tags[0] = 1
            #chirality_tags[1] = 1
        elif chirality == "S":
            chirality_tags[1] = 1
            #chirality_tags[2] = 1
        elif chirality == "E":
            chirality_tags[2] = 1
            #chirality_tags[3] = 1
        elif chirality == "Z":
            chirality_tags[3] = 1
            #chirality_tags[0] = 1
    
    return chirality_tags


def encode_atom(atom):
    #atom_type = [0] * 119
    #atom_type[atom.GetAtomicNum() - 1] = 1
    
    aromaticity = [0, 0]
    aromaticity[int(atom.GetIsAromatic())] = 1

    formal_charge = [0] * 16
    formal_charge[atom.GetFormalCharge() + 8] = 1  # Assuming formal charges range from -8 to +8

    chirality_tags = encode_chirality(atom)

    degree = [0] * 11  # Assuming degrees up to 10
    degree[atom.GetDegree()] = 1

    num_hydrogens = [0] * 9  # Assuming up to 8 hydrogens
    num_hydrogens[atom.GetTotalNumHs()] = 1

    hybridization = [0] * 5  # Assuming 5 possible hybridization types
    hybridization_type = atom.GetHybridization()
    valid_hybridization_types = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
    if  hybridization_type > 5:
        print(hybridization_type)
    for i, hybridization_type in enumerate(valid_hybridization_types):
        if  hybridization_type in valid_hybridization_types:
            hybridization[i] = 1  
    #atom_type +
    return aromaticity + formal_charge + chirality_tags + degree + num_hydrogens + hybridization



def get_bond_formal_charge(bond):
    # 获取连接到键两端的原子
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    
    chirality_tags_atom = encode_chirality(atom1) 
    chirality_tags_atom.extend(encode_chirality(atom2))

    return chirality_tags_atom


def bond_length_onehot(bond_length):
    if 1.2 <= bond_length <= 1.6:
        #return [1, 1, 0, 0, 0]  # Single Bond
        return [1, 0, 0, 0, 0]  # Single Bond
    elif 1.3 <= bond_length <= 1.5:
        #return [0, 1, 1, 0, 0]  # Double Bond
        return [0, 1, 0, 0, 0]  
    elif 1.1 <= bond_length <= 1.3:
        #return [0, 0, 1, 1, 0]  # Triple Bond
        return [0, 0, 1, 0, 0]
    elif 1.4 <= bond_length <= 1.5:
        #return [0, 0, 0, 1, 1]  # Aromatic Bond
        return [0, 0, 0, 1, 0]
    else:
        #return [1, 0, 0, 0, 1]  # None of the above
        return [0, 0, 0, 0, 1]
    
def bond_type_map(bond_type):
    bond_length_dict = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
    return bond_length_dict[bond_type]


def bond_stereo_onehot(bond):
    stereo = bond.GetStereo()
    
    if stereo == Chem.BondStereo.STEREOANY:
        return [1, 0, 0, 0, 0]  # Any Stereo
    elif stereo == Chem.BondStereo.STEREOCIS:
        return [0, 1, 0, 0, 0]  # CIS Stereo
    elif stereo == Chem.BondStereo.STEREOTRANS:
        return [0, 0, 1, 0, 0]  # TRANS Stereo
    elif stereo == Chem.BondStereo.STEREONONE:
        return [0, 0, 0, 1, 0]  # No defined Stereo
    else:
        return [0, 0, 0, 0, 1]  # None of the above


def encode_bond_26(bond,mol):
    #26 dim
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1
    
    bond_type = [0] * 4
    bond_type[bond_type_map(str(bond.GetBondType()))] = 1

    bond_lg = AllChem.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    bond_length = bond_length_onehot(bond_lg)#5
    
    in_ring = [0]*2 #2
    in_ring[int(bond.IsInRing())] = 1

    #stereo = bond_stereo_onehot(bond) #5
    stereo = get_bond_formal_charge(bond)#8

    return bond_dir + bond_type + bond_length + in_ring + stereo



def bond_length_approximation(bond_type):
    bond_length_dict = {"SINGLE": 1.0, "DOUBLE": 1.4, "TRIPLE": 1.8, "AROMATIC": 1.5}
    return bond_length_dict.get(bond_type, 1.0)

def encode_bond_14(bond):
    #7+4+2+2+6 = 21
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1
    
    #bond_type = [0] * 4
    #bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1
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
    if len(list1) != len(list2):
        return False  
    
    for i in range(len(list1)):
        if list1[i] == element1 and list2[i] == element2:
            return True  
    
    return False  

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

                        if s_d_dis <= 5:
                            #if check_common_elements(src_list,dst_list,i,j):
                            if (not check_common_elements(src_list, dst_list, i, j) and not check_common_elements(src_list, dst_list, j, i)):
                            
                                src_list.append(i)
                                src_list.append(j)
                                dst_list.append(j)
                                dst_list.append(i)
                                per_bond_feat = [0]*15
                                per_bond_feat.extend(non_bonded(atom_charges,i,j,s_d_dis))

                                edge_features.append(per_bond_feat)
                                edge_features.append(per_bond_feat)
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
