import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import obonet
import random
import torch
import math
from Bio import SeqIO
import Bio.PDB
import pickle as pickle
import os
from Bio import PDB
from rdkit import Chem
import blosum as bl
import pymol
from pymol import cmd
from torch_geometric.data import Data

class CFG:
    pdbfiles: str = "/home/paul/Desktop/BioHack-Project-Walkthrough/pdbind-refined-set/"
    AA_mol2_files: str = "/home/paul/Desktop/BioHack-Project-Walkthrough/AA_mol2/"
    home: str = '/home/paul/Desktop/BioHack-Project-Walkthrough/BioHack/'

with open('atom2emb.pkl', 'rb') as f:
    atom2emb = pickle.load(f)
    
with open('AA_embeddings.pkl', 'rb') as f:
    AA_embeddings = pickle.load(f)
    
with open('bond_type_dict.pkl', 'rb') as f:
    bond_type_dict = pickle.load(f)
    
with open('AA_OHE.pkl', 'rb') as f:
    OHE_dict = pickle.load(f)

def get_atom_symbol(atomic_number):
    return Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), atomic_number)

def remove_hetatm(input_pdb_file, output_pdb_file):
    # Open the input PDB file for reading and the output PDB file for writing
    with open(input_pdb_file, 'r') as infile, open(output_pdb_file, 'w') as outfile:
        for line in infile:
            # Check if the line starts with 'HETATM' (non-protein atoms)
            if line.startswith('HETATM'):
                continue  # Skip this line (HETATM record)
            # Write all other lines to the output file
            outfile.write(line)
            
def get_atom_types_from_sdf(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file)
    atom_types = set()

    for mol in supplier:
        if mol is not None:
            atoms = mol.GetAtoms()
            atom_types.update([atom.GetSymbol() for atom in atoms])

    return sorted(list(atom_types))

def get_atom_types_from_mol2_split(mol2_file):
    atom_types = set()

    with open(mol2_file, 'r') as mol2:
        reading_atoms = False
        for line in mol2:
            if line.strip() == '@<TRIPOS>ATOM':
                reading_atoms = True
                continue
            elif line.strip() == '@<TRIPOS>BOND':
                break

            if reading_atoms:
                parts = line.split()
                if len(parts) >= 5:
                    atom_type = parts[5]
                    atom_types.add(atom_type)
    
    atom_types_split = set()
    for atom in atom_types:
        atom_types_split.add(str(atom).split('.')[0])
        

    return sorted(list(atom_types_split))

def get_atom_types_from_mol2(mol2_file):
    atom_types = set()

    with open(mol2_file, 'r') as mol2:
        reading_atoms = False
        for line in mol2:
            if line.strip() == '@<TRIPOS>ATOM':
                reading_atoms = True
                continue
            elif line.strip() == '@<TRIPOS>BOND':
                break

            if reading_atoms:
                parts = line.split()
                if len(parts) >= 5:
                    atom_type = parts[5]
                    atom_types.add(atom_type)

    return sorted(list(atom_types))

def get_atom_list_from_mol2_split(mol2_file):
    atoms = []
    with open(mol2_file, 'r') as mol2:
        reading_atoms = False
        for line in mol2:
            if line.strip() == '@<TRIPOS>ATOM':
                reading_atoms = True
                continue
            elif line.strip() == '@<TRIPOS>BOND':
                break

            if reading_atoms:
                parts = line.split()
                if len(parts) >= 5:
                    atom_type = parts[5]
                    atoms.append(atom_type)
    
    atom_list = []
    for atom in atoms:
        atom_list.append(str(atom).split('.')[0])
        

    return atom_list

def get_atom_list_from_mol2(mol2_file):
    atoms = []
    with open(mol2_file, 'r') as mol2:
        reading_atoms = False
        for line in mol2:
            if line.strip() == '@<TRIPOS>ATOM':
                reading_atoms = True
                continue
            elif line.strip() == '@<TRIPOS>BOND':
                break

            if reading_atoms:
                parts = line.split()
                if len(parts) >= 5:
                    atom_type = parts[5]
                    atoms.append(atom_type)

    return atoms

def get_bond_types_from_mol2(mol2_file):
    bond_types = set()

    with open(mol2_file, 'r') as mol2:
        reading_bonds = False
        for line in mol2:
            if line.strip() == '@<TRIPOS>BOND':
                reading_bonds = True
                continue
            elif reading_bonds and line.strip().startswith('@<TRIPOS>'):
                break

            if reading_bonds:
                parts = line.split()
                if len(parts) >= 4:
                    bond_type = parts[3]
                    bond_types.add(bond_type)

    return sorted(list(bond_types))

def read_mol2_bonds(mol2_file):
    bonds = []
    bond_types = []

    with open(mol2_file, 'r') as mol2:
        reading_bonds = False
        for line in mol2:
            if line.strip() == '@<TRIPOS>BOND':
                reading_bonds = True
                continue
            elif reading_bonds and line.strip().startswith('@<TRIPOS>'):
                break

            if reading_bonds:
                parts = line.split()
                if len(parts) >= 4:
                    atom1_index = int(parts[1])
                    atom2_index = int(parts[2])
                    bond_type = parts[3]
                    bonds.append((atom1_index, atom2_index))
                    bond_types.append(bond_type)

    return bonds, bond_types

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

def calc_contact_map(uniID,map_distance):
    pdb_code = uniID
    pdb_filename = uniID+"_pocket_clean.pdb"
    structure = Bio.PDB.PDBParser(QUIET = True).get_structure(pdb_code, (CFG.pdbfiles +'/'+pdb_code+'/'+pdb_filename))
    model = structure[0]
    flag1 = 0
    flag2 = 0
    idx = 0
    index = []
    chain_info = []
    
    for chain1 in model:
        for resi in chain1:
            index.append(idx)
            idx += 1
            chain_info.append([chain1.id,resi.id])
        for chain2 in model:
            if flag1 == 0:
                dist_matrix = calc_dist_matrix(model[chain1.id], model[chain2.id])
            else:
                new_matrix = calc_dist_matrix(model[chain1.id], model[chain2.id])
                dist_matrix = np.hstack((dist_matrix,new_matrix))
            flag1 += 1
        flag1 = 0
        if flag2 == 0:
            top_matrix = dist_matrix
        else:
            top_matrix = np.vstack((top_matrix,dist_matrix))
        flag2 += 1
    
    contact_map = top_matrix < map_distance
    return contact_map, index, chain_info

one_letter_to_three_letter_dict = {'G':'gly',
                                   'A':'ala',
                                   'V':'val',
                                   'C':'cys',
                                   'P':'pro',
                                   'L':'leu',
                                   'I':'ile',
                                   'M':'met',
                                   'W':'trp',
                                   'F':'phe',
                                   'K':'lys',
                                   'R':'arg',
                                   'H':'his',
                                   'S':'ser',
                                   'T':'thr',
                                   'Y':'tyr',
                                   'N':'asn',
                                   'Q':'gln',
                                   'D':'asp',
                                   'E':'glu'
    
}

def BLOSUM_encode_single(seq,AA_dict):
    allowed = set("gavcplimwfkrhstynqdeuogavcplimwfkrhstynqde")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence has broken AA: {invalid}")
    vec = AA_dict[seq]
    return vec

matrix = bl.BLOSUM(62)
allowed_AA = "GAVCPLIMWFKRHSTYNQDE"
BLOSUM_dict_three_letter = {}
for i in allowed_AA:
    vec = []
    for j in allowed_AA:
        vec.append(matrix[i][j])
    BLOSUM_dict_three_letter.update({one_letter_to_three_letter_dict[i]:torch.Tensor(vec)})


def read_mol2_bonds_and_atoms(mol2_file):
    bonds = []
    bond_types = []
    atom_types = {}
    atom_coordinates = {}

    with open(mol2_file, 'r') as mol2:
        reading_bonds = False
        reading_atoms = False
        for line in mol2:
            if line.strip() == '@<TRIPOS>BOND':
                reading_bonds = True
                continue
            elif line.strip() == '@<TRIPOS>ATOM':
                reading_atoms = True
                continue
            elif line.strip().startswith('@<TRIPOS>SUBSTRUCTURE'):
                break
            elif reading_bonds and line.strip().startswith('@<TRIPOS>'):
                reading_bonds = False
            elif reading_atoms and line.strip().startswith('@<TRIPOS>'):
                reading_atoms = False


            if reading_bonds:
                parts = line.split()
                if len(parts) >= 4:
                    atom1_index = int(parts[1])
                    atom2_index = int(parts[2])
                    bond_type = parts[3]
                    bonds.append((atom1_index, atom2_index))
                    bond_types.append(bond_type)

            if reading_atoms:
                parts = line.split()
                if len(parts) >= 6:
                    atom_index = int(parts[0])
                    atom_type = parts[5]
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    atom_types[atom_index] = atom_type.split('.')[0]
                    atom_coordinates[atom_index] = (x, y, z)

    return bonds, bond_types, atom_types, atom_coordinates    
    
def uniID2graph(uniID,map_distance, norm_map_distance = 12.0):
    atom_name = 'CA'
    atom_names = ['N','CA','C']
    node_feature = []
    edge_index = []
    edge_attr = []
    coord = []
    y = []
    contact_map, index, chain_info = calc_contact_map(uniID,map_distance)
    pdb_code = uniID
    pdb_filename = uniID+"_pocket_clean.pdb"
    structure = Bio.PDB.PDBParser(QUIET = True).get_structure(pdb_code, (CFG.pdbfiles +'/'+pdb_code+'/'+pdb_filename))
    model = structure[0]
    
    for i in index:
        node_feature.append(AA_embeddings[model[chain_info[i][0]][chain_info[i][1]].get_resname()])
        coord.append([model[chain_info[i][0]][chain_info[i][1]][name].coord for name in atom_names])
        y.append(OHE_dict[model[chain_info[i][0]][chain_info[i][1]].get_resname()])
        for j in index:
            if contact_map[i,j] == 1 and j > i:
                edge_index.append([i,j])
                d = []
                for name1 in atom_names:
                    for name2 in atom_names:
                        diff_vector = model[chain_info[i][0]][chain_info[i][1]][name1].coord - model[chain_info[j][0]][chain_info[j][1]][name2].coord
                        dist = (np.sqrt(np.sum(diff_vector * diff_vector)))
                        for l in range(12):
                            d.append(np.exp((-1.0*(dist - 2.0*(l + 0.5))**2.0)/norm_map_distance)) 
                bond_type = bond_type_dict['nc']
                edge_attr.append(np.hstack((d,bond_type)))
    
    
    prot_coord = [torch.Tensor(i[1]) for i in coord]
    new_prot_coord = torch.stack(prot_coord)
    
    edge_index = np.array(edge_index)
    edge_index = edge_index.transpose()
    edge_index = torch.Tensor(edge_index)
    edge_index = edge_index.to(torch.int64)
    edge_attr = torch.Tensor(np.array(edge_attr))
    node_feature = torch.stack(node_feature)
    y = torch.stack(y)
    graph = Data(x = node_feature, edge_index = edge_index,edge_attr = edge_attr, pos = new_prot_coord)
    graph.y = y
    return graph, coord

def molecule2graph(filename,map_distance, norm_map_distance = 12.0):
    node_feature = []
    edge_index = []
    edge_attr = []
    y = []
    mol2_file = CFG.pdbfiles+filename+'/'+filename+'_ligand.mol2'
    bonds, bond_types, atom_types, atom_coordinates = read_mol2_bonds_and_atoms(mol2_file)
    for atom in atom_types:
        node_feature.append(torch.Tensor(atom2emb[atom_types[atom]]))
        y.append(torch.zeros(20))
    

    for atom1 in range(1, len(atom_types)+1):
        for atom2 in range(atom1 + 1, len(atom_types)+1):
            bonded_flag = 0
            for i, bond in enumerate(bonds):
                if (atom1 in bond) and (atom2 in bond):
                    edge_index.append([bond[0] - 1,bond[1] - 1])
                    coord1 = np.array(atom_coordinates[bond[0]])
                    coord2 = np.array(atom_coordinates[bond[1]])
                    dist = math.dist(coord1, coord2)
                    #dist = np.sqrt(np.sum((coord1 - coord2)*(coord1 - coord2)))
                    d = []
                    for l in range(12):
                        d.append(np.exp((-1.0*(dist - 2.0*(l + 0.5))**2.0)/norm_map_distance))
                    bond_type = bond_type_dict[bond_types[i]]
                    edge_attr.append(np.hstack((d,d,d,d,d,d,d,d,d,bond_type)))
                    bonded_flag = 1
                
            if bonded_flag == 0:
                coord1 = np.array(atom_coordinates[atom1])
                coord2 = np.array(atom_coordinates[atom2])
                dist = math.dist(coord1, coord2)
                #dist = np.sqrt(np.sum((coord1 - coord2)*(coord1 - coord2)))
                if dist < map_distance:
                    edge_index.append([atom1 - 1,atom2 - 1])
                    d = []
                    for l in range(12):
                        d.append(np.exp((-1.0*(dist - 2.0*(l + 0.5))**2.0)/norm_map_distance))
                    bond_type = bond_type_dict['nc']
                    edge_attr.append(np.hstack((d,d,d,d,d,d,d,d,d,bond_type)))

    
    edge_index = np.array(edge_index)
    edge_index = edge_index.transpose()
    edge_index = torch.Tensor(edge_index)
    edge_index = edge_index.to(torch.int64)
    edge_attr = torch.Tensor(np.array(edge_attr))
    node_feature = torch.stack(node_feature)
    y = torch.stack(y)
    graph = Data(x = node_feature, edge_index = edge_index,edge_attr = edge_attr)#, pos = new_mol_coords)
    graph.y = y
    return graph, atom_coordinates

def id2fullgraph(filename, map_distance, norm_map_distance = 12.0):
    prot_graph, prot_coord = uniID2graph(filename,map_distance)
    prot_graph = prot_graph.to('cpu')
    mol_graph, mol_coord = molecule2graph(filename,map_distance)
    mol_graph = mol_graph.to('cpu')
    mol_coord = [mol_coord[i] for i in mol_coord]
    node_features = torch.cat((prot_graph.x,mol_graph.x),dim = 0)
    y = torch.cat((prot_graph.y,mol_graph.y),dim = 0)
    update_edge_index = mol_graph.edge_index + prot_graph.x.size()[0]
    edge_index = torch.cat((prot_graph.edge_index,update_edge_index), dim = 1)
    edge_attr = torch.cat((prot_graph.edge_attr,mol_graph.edge_attr), dim = 0)
    
    new_edge_index = []
    new_edge_attr = []
    for i in range(len(mol_coord)):
        for j in range(len(prot_coord)):
            d = []
            for k in range(len(prot_coord[j])):
                dist_vec = mol_coord[i] - prot_coord[j][k]
                dist = np.sqrt(np.sum(dist_vec*dist_vec))
                if k == 1:
                    d_check = dist
                for l in range(12):
                    d.append(np.exp((-1.0*(dist - 2.0*(l + 0.5))**2.0)/norm_map_distance))
            if (d_check/map_distance) < 1.0:
                new_edge_index.append([j,i + len(prot_coord)])
                new_edge_attr.append((np.hstack((d,d,d,bond_type_dict['nc']))))
    
    #Master_node
    node_features = torch.cat((node_features,torch.zeros(len(atom2emb['N'])).unsqueeze(0)),dim = 0)
    y = torch.cat((y,torch.zeros(20).unsqueeze(0)),dim = 0)
    
    for i in range(len(node_features) - 1):
        new_edge_index.append([i,int(len(node_features)-1)])
        bond_type = bond_type_dict['nc']
        new_edge_attr.append(np.hstack((np.zeros(3*len(d)),bond_type)))
    
    new_edge_index = np.array(new_edge_index)
    new_edge_index = new_edge_index.transpose()
    new_edge_index = torch.Tensor(new_edge_index)
    new_edge_index = new_edge_index.to(torch.int64)
    new_edge_attr = torch.Tensor(np.array(new_edge_attr))
    
    #Include Ca or atom coordinates
    prot_coord = [torch.Tensor(i[1]) for i in prot_coord]
    new_prot_coord = torch.stack(prot_coord)
    mol_coord = [torch.Tensor(i) for i in mol_coord]
    new_mol_coord = torch.stack(mol_coord)
    coords = torch.vstack((new_prot_coord, new_mol_coord))
    
    edge_index = torch.cat((edge_index,new_edge_index), dim = 1)
    edge_attr = torch.cat((edge_attr,new_edge_attr), dim = 0)
    
    graph = Data(x = node_features, edge_index = edge_index,edge_attr = edge_attr, pos = coords)
    graph.y = y
    
    return graph

def molecule2graph_AA(filename,map_distance, norm_map_distance = 12.0):
    node_feature = []
    edge_index = []
    edge_attr = []
    mol2_file = CFG.AA_mol2_files+filename
    bonds, bond_types, atom_types, atom_coordinates = read_mol2_bonds_and_atoms(mol2_file)
    for atom in atom_types:
        node_feature.append(torch.Tensor(atom2emb[atom_types[atom]]))
    

    for atom1 in range(1, len(atom_types)+1):
        for atom2 in range(atom1 + 1, len(atom_types)+1):
            bonded_flag = 0
            for i, bond in enumerate(bonds):
                if (atom1 in bond) and (atom2 in bond):
                    edge_index.append([bond[0] - 1,bond[1] - 1])
                    coord1 = np.array(atom_coordinates[bond[0]])
                    coord2 = np.array(atom_coordinates[bond[1]])
                    dist = math.dist(coord1, coord2)
                    d = []
                    for l in range(12):
                        d.append(np.exp((-1.0*(dist - 2.0*(l + 0.5))**2.0)/norm_map_distance))
                    bond_type = bond_type_dict[bond_types[i]]
                    edge_attr.append(np.hstack((d,d,d,d,d,d,d,d,d,bond_type)))
                    bonded_flag = 1
                
            if bonded_flag == 0:
                coord1 = np.array(atom_coordinates[atom1])
                coord2 = np.array(atom_coordinates[atom2])
                dist = math.dist(coord1, coord2)
                if dist < map_distance:
                    edge_index.append([atom1 - 1,atom2 - 1])
                    d = []
                    for l in range(12):
                        d.append(np.exp((-1.0*(dist - 2.0*(l + 0.5))**2.0)/norm_map_distance))
                    bond_type = bond_type_dict['nc']
                    edge_attr.append(np.hstack((d,d,d,d,d,d,d,d,d,bond_type)))

    
    edge_index = np.array(edge_index)
    edge_index = edge_index.transpose()
    edge_index = torch.Tensor(edge_index)
    edge_index = edge_index.to(torch.int64)
    edge_attr = torch.Tensor(np.array(edge_attr))
    node_feature = torch.stack(node_feature)
    
    #Master_node
    new_edge_index = []
    new_edge_attr = []
    node_feature = torch.cat((node_feature,torch.zeros(len(atom2emb['N'])).unsqueeze(0)),dim = 0)
    
    for i in range(len(node_feature) - 1):
        new_edge_index.append([i,int(len(node_feature)-1)])
        bond_type = bond_type_dict['nc']
        new_edge_attr.append(np.hstack((np.zeros(9*len(d)),bond_type)))
    
    new_edge_index = np.array(new_edge_index)
    new_edge_index = new_edge_index.transpose()
    new_edge_index = torch.Tensor(new_edge_index)
    new_edge_index = new_edge_index.to(torch.int64)
    new_edge_attr = torch.Tensor(np.array(new_edge_attr))    
    
    edge_index = torch.cat((edge_index,new_edge_index), dim = 1)
    edge_attr = torch.cat((edge_attr,new_edge_attr), dim = 0)
    
    graph = Data(x = node_feature, edge_index = edge_index,edge_attr = edge_attr)#, pos = new_mol_coords)
    graph.label = filename.split('.')[0]
    softmax = nn.Softmax(dim = 0)
    graph.y = softmax(BLOSUM_encode_single(graph.label,BLOSUM_dict_three_letter))
    return graph

upper2lower = {
    "ala": "ALA",
    "arg": "ARG",
    "asn": "ASN",
    "asp": "ASP",
    "cys": "CYS",
    "gln": "GLN",
    "glu": "GLU",
    "gly": "GLY",
    "his": "HIS",
    "ile": "ILE",
    "leu": "LEU",
    "lys": "LYS",
    "met": "MET",
    "phe": "PHE",
    "pro": "PRO",
    "ser": "SER",
    "thr": "THR",
    "trp": "TRP",
    "tyr": "TYR",
    "val": "VAL",
}

AA_3_letters = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
