import numpy as np
import os
import pickle
from os import walk
from numpy import linalg as LA
from collections import defaultdict

all_amino = []

max_residues = 3000

amino_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'PYL', 'SER', 'SEC', 'THR', 'TRP', 'TYR', 'VAL', 'ASX', 'GLX', 'XAA', 'XLE']

amino_short = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'PYL': 'O', 'SER': 'S', 'SEC': 'U',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'ASX': 'B', 'GLX': 'Z',
    'XAA': 'X', 'XLE': 'J'
}

# 新增函数：选择中心原子
def get_center_atom(residue):
    if 'CA' in residue['type_atm']:
        return residue['coords'][residue['type_atm'].index('CA')]
    elif 'N' in residue['type_atm']:
        return residue['coords'][residue['type_atm'].index('N')]
    elif 'C' in residue['type_atm']:
        return residue['coords'][residue['type_atm'].index('C')]
    elif 'O' in residue['type_atm']:
        return residue['coords'][residue['type_atm'].index('O')]
    elif 'CB' in residue['type_atm']:
        return residue['coords'][residue['type_atm'].index('CB')]
    elif 'CD' in residue['type_atm']:
        return residue['coords'][residue['type_atm'].index('CD')]
    elif 'CG' in residue['type_atm']:
        return residue['coords'][residue['type_atm'].index('CG')]
    else:
        return np.mean(residue['coords'], axis=0)

def create_fingerprints(atoms, adjacency, radius):
    """Extract r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using WeisfeilerLehman-like algorithm."""

    fingerprints = []
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        for i in range(len(atoms)):
            vertex = atoms[i]
            neighbors = tuple(set(tuple(sorted(atoms[np.where(adjacency[i] > 0.0001)[0]]))))
            fingerprint = (vertex, neighbors)
            fingerprints.append(fingerprint_dict[fingerprint])
    return np.array(fingerprints)

def create_amino_acids(acids):
    retval = [acid_dict[acid_name] if acid_name in amino_list else acid_dict['MET'] if acid_name == 'FME' else acid_dict['TMP'] for acid_name in acids]
    return np.array(retval)

def dump_dictionary(dictionary, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dict(dictionary), f)

def is_empty_pdb(pdb_id):
    empty = False
    with open(pdb_id + '.pdb', 'r') as f:
        for ln in f:
            if ln.startswith('<html>'):
                empty = True
                break
    return empty

def replace_pdb(pdb_id):
    with open(pdb_id + '.pdb', 'r') as f:
        filedata = f.read()
        filedata = filedata.replace('DBREF1', 'DBREF')
        filedata = filedata.replace('DBREF2', 'DBREF')
        filedata = filedata.replace('DBREF3', 'DBREF')
        filedata = filedata.replace('DBREF4', 'DBREF')
        filedata = filedata.replace('DBREF5', 'DBREF')
        filedata = filedata.replace('\\\'', 'P')
    with open(pdb_id + '.pdb', 'w') as f:
        f.write(filedata)

def parse_protein(pdb_name):
    protein_info = []

    with open(pdb_name + ".pdb", "r") as fi:
        mdl = False
        for ln in fi:
            if ln.startswith("NUMMDL"):
                mdl = True
                break

    with open(pdb_name + ".pdb", "r") as fi:
        id = []

        if mdl:
            for ln in fi:
                if ln.startswith("ATOM") or ln.startswith("HETATM"):
                    id.append(ln)
                elif ln.startswith("ENDMDL"):
                    break
        else:
            for ln in fi:
                if ln.startswith("ATOM") or ln.startswith("HETATM"):
                    id.append(ln)

    count = 0
    seq = {'type_atm': [], 'ind': [], 'amino': [], 'group': [], 'coords': []}

    for element in id:
        type_atm = element[0:6].strip().split()[0]
        ind = int(element[6:12].strip().split()[0])
        atom = element[12:17].strip().split()[0]
        amino = element[17:21].strip().split()[0]
        group_id = int(element[22:26].strip().split()[0])
        x_coord = float(element[30:38].strip().split()[0])
        y_coord = float(element[38:46].strip().split()[0])
        z_coord = float(element[46:54].strip().split()[0])
        coords = np.array([x_coord, y_coord, z_coord])

        seq['type_atm'].append(atom)
        seq['ind'].append(int(ind))
        seq['amino'].append(amino)
        seq['group'].append(int(group_id))
        seq['coords'].append(coords)

        count += 1

    protein_info.append(seq['amino'])
    protein_info.append(seq['group'])
    protein_info.append(seq['coords'])
    protein_info.append(seq['type_atm'])

    return protein_info

def unique(list1):
    x = np.array(list1)
    return np.unique(x)

def group_by_coords(group, amino, coords, type_atm):
    uniq_group = np.unique(group)
    group_coords = np.zeros((uniq_group.shape[0], 3))
    group_amino = []

    np_group = np.array(group)

    for i, e in enumerate(uniq_group):
        inds = np.where(np_group == e)[0]
        residue = {'type_atm': [type_atm[j] for j in inds], 'coords': [coords[j] for j in inds]}
        group_coords[i, :] = get_center_atom(residue)
        group_amino.append(amino[inds[0]])

    return group_coords, group_amino

def get_graph_from_struct(group_coords, group_amino):
    num_residues = group_coords.shape[0]

    if num_residues > max_residues:
        num_residues = max_residues

    residues = group_amino[:num_residues]

    retval = [[0 for i in range(0, num_residues)] for j in range(0, num_residues)]

    residue_type = []
    for i in range(0, num_residues):
        if residues[i] == 'FME':
            residues[i] = 'MET'
        elif residues[i] not in amino_list:
            residues[i] = 'TMP'

        residue_type.append(residues[i])

        for j in range(i + 1, num_residues):
            x, y = group_coords[i], group_coords[j]
            retval[i][j] = LA.norm(x - y)
            retval[j][i] = retval[i][j]

    retval = np.array(retval)

    threshold = 8

    for i in range(0, num_residues):
        for j in range(0, num_residues):
            if retval[i, j] <= threshold:
                retval[i, j] = 1
            else:
                retval[i, j] = 0

    n = retval.shape[0]
    adjacency = retval + np.eye(n)
    degree = sum(adjacency)
    d_half = np.sqrt(np.diag(degree))
    d_half_inv = np.linalg.inv(d_half)
    adjacency = np.matmul(d_half_inv, np.matmul(adjacency, d_half_inv))
    return residue_type, np.array(adjacency)

radius = 1

atom_dict = defaultdict(lambda: len(atom_dict))
acid_dict = defaultdict(lambda: len(acid_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

filepath = "pdb_files/"
os.chdir(filepath)

dir_input = ('input' + str(radius) + '/')
os.makedirs(dir_input, exist_ok=True)

os.chdir('../')

f = []
for (dirpath, dirnames, filenames) in walk(filepath):
    f.extend(filenames)
    break

os.chdir(filepath)

pdb_ids = []
for data in f:
    pdb_ids.append(data.strip().split('.')[0])

num_prots = len(pdb_ids)

count = 0
q_count = 0

adjacencies, proteins, pnames, pseqs = [], [], [], []

for n in range(num_prots):
    if not is_empty_pdb(pdb_ids[n]):
        replace_pdb(pdb_ids[n])
        pdb_name = pdb_ids[n]

        print('/'.join(map(str, [n + 1, num_prots])))
        print(pdb_name)

        try:
            protein_info = parse_protein(pdb_name)
            amino, group, coords, type_atm = protein_info
            group_coords, group_amino = group_by_coords(group, amino, coords, type_atm)
            residue_type, adjacency = get_graph_from_struct(group_coords, group_amino)
            atoms = create_amino_acids(residue_type)

            fingerprints = create_fingerprints(atoms, adjacency, radius)

            adjacencies.append(adjacency)
            proteins.append(fingerprints)
            pnames.append(pdb_name)

            d_seq = {}
            for no, g in enumerate(group):
                if g not in d_seq.keys():
                    d_seq[g] = amino[no]

            seq_pr = ''
            for k in d_seq:
                if d_seq[k] in amino_list:
                    seq_pr += amino_short[d_seq[k]]

            pseqs.append(seq_pr)

            count += 1

            if count % 10 == 0 or n == num_prots - 1:
                count = 0

                np.save(dir_input + 'proteins_' + str(10 * q_count + 1) + '_' + str(10 * (q_count + 1)), proteins)
                np.save(dir_input + 'adjacencies_' + str(10 * q_count + 1) + '_' + str(10 * (q_count + 1)), adjacencies)
                np.save(dir_input + 'names_' + str(10 * q_count + 1) + '_' + str(10 * (q_count + 1)), pnames)
                np.save(dir_input + 'seqs_' + str(10 * q_count + 1) + '_' + str(10 * (q_count + 1)), pseqs)

                adjacencies, proteins, pnames, pseqs = [], [], [], []
                q_count += 1

        except:
            print(pdb_name)

dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
print('Length of fingerprint dictionary: ' + str(len(fingerprint_dict)))
