import dgl
import torch
import numpy as np
import networkx as nx


from rdkit import Chem
from rdkit.Chem import AllChem
from jarvis.core.specie import chem_data, get_node_attributes




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
        # 处理分母为零的情况，例如给一个默认值或进行其他操作
        cos_theta = 0.0  # 你可以根据具体情况调整这里的值
    #cos_theta = dot_product / (norm_AB * norm_BC)
    # 假设 cos_theta 是你的输入
    if -1 <= cos_theta <= 1:
        angle_rad = np.arccos(cos_theta)
    else:
    # 在这里处理无效的输入，例如给出一个默认值或者引发异常
        angle_rad = 0  # 或者 raise ValueError("Invalid cos_theta value")

    #angle_rad = np.arccos(cos_theta)

    #angle_deg = np.degrees(angle_rad)

    return angle_rad



def calculate_triangle_properties(point1, point2, point3,encode_two_path):
    point1, point2, point3 = point1.cpu().numpy(), point2.cpu().numpy(), point3.cpu().numpy()
    pps = []
    # 计算质心坐标
    centroid = np.mean([point1, point2, point3], axis=0)
    #pps.append(centroid)
    # 计算坐标到质心的距离
    distances = [np.linalg.norm(point - centroid) for point in [point1, point2, point3]]
    pps.extend(distances)
    # 计算三角形的边长
    a = np.linalg.norm(point2 - point1)
    b = np.linalg.norm(point3 - point2)
    c = np.linalg.norm(point1 - point3)
    pps.extend([a,b,c])

    if encode_two_path == "dim_8":
        # 计算三角形的角度（弧度）
        A = calculate_angle(point1, point2, point3)
        #B = calculate_angle(point2, point1, point3)
        #C = calculate_angle(point2, point3, point1)
        #pps.extend([A,B,C])
        pps.extend([A,A**2])

    elif encode_two_path == "dim_10":
        # 计算三角形的角度（弧度）
        A = calculate_angle(point1, point2, point3)
        B = calculate_angle(point2, point1, point3)
        C = calculate_angle(point2, point3, point1)
        pps.extend([A,B,C])
        #pps.extend([A])

    
    if a > 0 and b > 0 and c > 0 and (a + b > c) and (a + c > b) and (b + c > a):
        s = 0.5 * (a + b + c)  # 半周长
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        pps.append(area)
    else:
        pps.append(0)

    return(pps)




def cross_product(a, b):
    return np.cross(a, b)

# 计算向量长度
def vector_length(a):
    return np.linalg.norm(a)

# 计算四个顶点形成的四面体的四个面积
def tetrahedron_areas(a, b, c, d):
    ab = b - a
    ac = c - a
    ad = d - a
    bc = c - b
    bd = d - b
    cd = d - c
    
    area_abcd = 0.5 * vector_length(cross_product(ab, ac))
    area_abdc = 0.5 * vector_length(cross_product(ab, ad))
    area_acbd = 0.5 * vector_length(cross_product(ac, ad))
    area_bcad = 0.5 * vector_length(cross_product(bc, bd))
    
    return [area_abcd, area_abdc, area_acbd, area_bcad]



def calculate_triangle_area(point1, point2, point3):
    # 计算两个向量
    # 计算三角形的边长
    a = np.linalg.norm(point2 - point1)
    b = np.linalg.norm(point3 - point2)
    c = np.linalg.norm(point1 - point3)
    
    if a > 0 and b > 0 and c > 0 and (a + b > c) and (a + c > b) and (b + c > a):
        s = 0.5 * (a + b + c)  # 半周长
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        return area
    else:
        return 0
    

def area_of_quadrilateral(p1, p2, p3, p4):
    """
    计算四边形的面积
    参数:
        p1, p2, p3, p4: 四个点的坐标，每个点为一个三维向量，如 [x, y, z]
    返回值:
        四边形的面积
    """
    # 计算两个对角线向量
    d1 = np.array(p2) - np.array(p4)
    d2 = np.array(p3) - np.array(p1)
    
    # 使用向量叉乘计算面积
    area = 0.5 * np.linalg.norm(np.cross(d1, d2))
    
    return area


def calculate_dihedral_angle(point1, point2, point3,point4,encode_tree_path):
    point1, point2, point3,point4 = point1.cpu().numpy(), point2.cpu().numpy(), point3.cpu().numpy(),point4.cpu().numpy()

    pps = []
    volume = np.abs(np.dot(np.cross(point2 - point1, point3 - point1), point4 - point1)) / 6.0
    pps.extend([volume])

    if encode_tree_path == "dim_6":
    # 计算四个面的法向量
        normal_vectors = [
            np.cross(point2 - point1, point3 - point1),
            np.cross(point3 - point2, point4 - point2),
            #np.cross(point4 - point3, point1 - point3),
            #np.cross(point1 - point4, point2 - point4)
        ]

        
    # 计算四个面的二面角
    angles = []
    for i in range(len(normal_vectors)):
        norm_i = np.linalg.norm(normal_vectors[i])
        for j in range(i + 1, len(normal_vectors)):
            norm_j = np.linalg.norm(normal_vectors[j])
            if norm_i != 0 and norm_j != 0:
                cos_angle = np.dot(normal_vectors[i], normal_vectors[j]) / (np.linalg.norm(normal_vectors[i]) * np.linalg.norm(normal_vectors[j]))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                
            else:
                cos_angle = 0
            sin_angle = np.sqrt(1 - cos_angle**2)

            angles.extend([cos_angle,sin_angle])

    pps.extend(angles)
     
    #关键边
    #dis_1 = calculate_dis(point2, point3)
    dis_2 = calculate_dis(point1, point4)
    
    pps.extend([dis_2])

    #面积
    #face_1 = calculate_triangle_area(point1, point2, point3)
    #face_2 = calculate_triangle_area(point2, point3,point4)
    #pps.extend([face_1,face_2])

    #area = area_of_quadrilateral(point1, point2, point3,point4)
    #pps.extend([area])

    #四面体表面积
    area = area_of_quadrilateral(point1, point2, point3,point4)
    pps.extend([area])

    dis_3 = calculate_dis(point1, point2) + calculate_dis(point2, point3)
    dis_4 = calculate_dis(point2, point3) + calculate_dis(point3, point4)
    pps.extend([dis_3,dis_4])

    return pps


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
    
    bond_type = [0] * 4
    bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1
    
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
        # 尝试嵌入分子
        AllChem.EmbedMolecule(mol)
        # 创建 MMFF 力场
        AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        return True
    except ValueError:
        # 如果捕获到ValueError异常，则无法嵌入分子，返回False
        return False


def uff_force_field(mol):
    try:
        # 尝试嵌入分子
        AllChem.EmbedMolecule(mol)
        # 创建 MMFF 力场
        AllChem.UFFGetMoleculeForceField(mol)
        return True
    except ValueError:
        # 如果捕获到ValueError异常，则无法嵌入分子，返回False
        return False
    
def random_force_field(mol):
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
        return True
    except ValueError:
        # 如果捕获到ValueError异常，则无法嵌入分子，返回False
        return False


def check_common_elements(list1, list2, element1, element2):
    if len(list1) != len(list2):
        return False  # 如果列表长度不相同，直接返回 False
    
    for i in range(len(list1)):
        if list1[i] == element1 and list2[i] == element2:
            return True  # 如果找到一对匹配的元素，返回 True
    
    return False  # 如果没有找到匹配的元素，返回 False

def atom_to_graph(smiles,encoder_atom,encoder_bond):
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # 加氢
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
                        
                    
                    # 获取原子坐标信息
                    pos = mol.GetConformer().GetAtomPosition(ii)
                    coor.append([pos.x, pos.y, pos.z])

                    # 获取电荷并存储
                    charge = s.GetProp("_GasteigerCharge")
                    atom_charges.append(charge)

                edge_features = []
                src_list, dst_list = [], []
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondTypeAsDouble() # 存储键信息
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
                            if check_common_elements(src_list,dst_list,i,j):
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
                g.ndata['coor'] = coor_tensor  # 添加原子坐标信息
                g.edata['feat'] = edge_feats
                g.edata['id'] = edge_id_feats
            
            else:
                g = False
        else:
            g = False
    return g





def create_bond_angle(g,encode_two_path):
    # Create an undirected NetworkX graph from the DGL grap
    edge_feats = g.edata['feat']
    coor_feats = g.ndata['coor']
    edge_id_feats = g.edata['id']
    
    edge_src, edge_dst = g.edges()
    num_nodes = g.number_of_edges()
    
    lg = dgl.DGLGraph()
    lg.add_nodes(num_nodes)
    lg.ndata['feat'] = edge_feats

    lg_src = []
    lg_dst = []
    lg_edge_feats = []
    for i in range(num_nodes):
        #print(edge_id_feats[i]== torch.tensor([1.]))
        if edge_id_feats[i] == torch.tensor([1.]):
            edge_i = [edge_src[i],edge_dst[i]]

            for j in range(i+1,num_nodes):
                if edge_id_feats[j] == torch.tensor([1]):
                    edge_j = [edge_src[j],edge_dst[j]]

                    if edge_i[1] == edge_j[0] and edge_i[0] != edge_j[1]:

                        A,B,C = edge_i[0], edge_j[0], edge_j[1]
                        A_coor = g.ndata['coor'][A]
                        B_coor = g.ndata['coor'][B]
                        C_coor = g.ndata['coor'][C]

                        pps = calculate_triangle_properties(A_coor, B_coor, C_coor,encode_two_path)
                        lg_edge_feats.append(pps)
                        lg_edge_feats.append(pps)
                        
                        lg_src.append(i)
                        lg_dst.append(j)

                        lg_src.append(j)
                        lg_dst.append(i)

                    elif edge_i[0] == edge_j[1] and edge_i[1] != edge_j[0]:  
                        #A,B,C = list(set([int(edge_i[0]),(int(edge_i[1]))]).union(set([int(edge_j[0]),int(edge_j[1])])))
                        A,B,C = edge_i[1],edge_i[0],edge_j[0]

                        A_coor = g.ndata['coor'][A]
                        B_coor = g.ndata['coor'][B]
                        C_coor = g.ndata['coor'][C]

                        pps = calculate_triangle_properties(A_coor, B_coor, C_coor,encode_two_path)
                        lg_edge_feats.append(pps)
                        lg_edge_feats.append(pps)
                        
                        lg_src.append(i)
                        lg_dst.append(j)

                        lg_src.append(j)
                        lg_dst.append(i)
                else:
                    continue
        else:
            continue

    lg.add_edges(lg_src, lg_dst)
    
    edge_feats = torch.tensor(lg_edge_feats, dtype=torch.float32)
    lg.edata['feat'] = edge_feats
    return lg



def create_dihedral_angle(g, lg, encode_tree_path):
    # Create an undirected NetworkX graph from the DGL grap
    edge_feats = lg.edata['feat']
    coor_feats = g.ndata['coor']
    lg_edge_src, lg_edge_dst = lg.edges()
    g_edge_src, g_edge_dst = g.edges()

    num_nodes = lg.number_of_edges()
    
    fg = dgl.DGLGraph()
    fg.add_nodes(num_nodes)
    fg.ndata['feat'] = edge_feats

    fg_src = []
    fg_dst = []
    fg_edge_feats = []
    for i in range(num_nodes):
        edge_i = [lg_edge_src[i],lg_edge_dst[i]]
        
        for j in range(i+1,num_nodes):
            edge_j = [lg_edge_src[j],lg_edge_dst[j]]
            
            if edge_i[1] == edge_j[0] and edge_i[0] != edge_j[1]:
                m = edge_i[0]
                h = edge_i[1]
                n =  edge_j[1]

                Union_nodes = [g_edge_src[m].item(),g_edge_dst[m].item(),g_edge_dst[h].item(),g_edge_dst[n].item()]
      
                if len(set(Union_nodes)) == 4:
                  
                    A, B, C, D = Union_nodes

                    A_coor = coor_feats[A]
                    B_coor = coor_feats[B]
                    C_coor = coor_feats[C]
                    D_coor = coor_feats[D]

                    pps = calculate_dihedral_angle(A_coor, B_coor, C_coor,D_coor,encode_tree_path)

                    fg_edge_feats.append(pps)
                    fg_edge_feats.append(pps)
                    
                    fg_src.append(i)
                    fg_dst.append(j)

                    fg_src.append(j)
                    fg_dst.append(i)
                else:
                    continue


            elif edge_i[0] == edge_j[1] and edge_i[1] != edge_j[0]:  
                m = edge_i[0]
                h = edge_i[1]
                l = edge_j[1]
                n =  edge_j[0]
                
                Union_nodes = [g_edge_src[n].item(),g_edge_dst[n].item(),g_edge_dst[m].item(),g_edge_dst[h].item()]


                if len(set(Union_nodes)) == 4:
                    A, B, C, D = Union_nodes
                    A_coor = coor_feats[A]
                    B_coor = coor_feats[B]
                    C_coor = coor_feats[C]
                    D_coor = coor_feats[D]


                    pps = calculate_dihedral_angle(A_coor, B_coor, C_coor,D_coor,encode_tree_path)
                    
                    fg_edge_feats.append(pps)
                    fg_edge_feats.append(pps)
                    
                    fg_src.append(i)
                    fg_dst.append(j)

                    fg_src.append(j)
                    fg_dst.append(i)
                else:
                    continue

    fg.add_edges(fg_src, fg_dst)
    if len(fg_src) > 0:
 
        edge_feats = torch.tensor(fg_edge_feats, dtype=torch.float32)
        fg.edata['feat'] = edge_feats
        return fg
    else:
        fg = False
        return fg



def path_complex_mol(Smile, encoder_atom,encoder_bond):
    #generate graph
    # 创建一个空的DGL图
    g = atom_to_graph(Smile,encoder_atom,encoder_bond)
    
    if g != False:
        return g
    else:
        return False


if __name__ == '__main__':
    # 创建一个简单的图
    smiles = '[H]C([H])([H])C([H])([H])[H]'
    encoder_atom = "cgcnn"
    encoder_bond = "dim_14"
    encode_two_path = "dim_8"
    encode_tree_path = "dim_6"
    encoder_bond = encode_two_path,encode_tree_path
    Graph_list = path_complex_mol(smiles, encoder_atom,encoder_bond)
    print(Graph_list)