import numpy as np
import joblib
import pandas as pd
from os import walk
import os

def SaveChainPDB(query_path, query_dir, filename, query_id):

    pdb_file = "{}\{}".format(query_dir,filename)
    save_file = '{}\{}.pdb'.format(query_path,query_id)

    with open(pdb_file, "r") as fi:
        mdl = False
        for ln in fi:
            if ln.startswith("NUMMDL"):
                mdl = True
                break

    with open(pdb_file, "r") as fi:
        id = []

        if mdl:
            for ln in fi:
                if ln.startswith("ATOM"):# or ln.startswith("HETATM"):
                    id.append(ln)
                elif ln.startswith("ENDMDL"):
                    break
        else:
            for ln in fi:
                if ln.startswith("ATOM"):
                    id.append(ln)
        id.append('ENDMDL')

    with open(save_file, 'w') as f:
        f.writelines(id)

    return

def def_atom_features():
    A = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,3,0]}
    V = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,1,0], 'CG1':[0,3,0], 'CG2':[0,3,0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,1,1] }
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,1], 'CG':[0,2,1], 'CD':[0,2,1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,1,0], 'CD1':[0,3,0], 'CD2':[0,3,0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'CG1':[0,2,0], 'CG2':[0,3,0], 'CD1':[0,3,0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,2,0], 'CD':[0,2,0], 'NE':[0,1,0], 'CZ':[1,0,0], 'NH1':[0,2,0], 'NH2':[0,2,0] }
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[-1,0,0], 'OD1':[-1,0,0], 'OD2':[-1,0,0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[-1,0,0], 'OE1':[-1,0,0], 'OE2':[-1,0,0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'OG':[0,1,0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'OG1':[0,1,0], 'CG2':[0,3,0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'SG':[-1,1,0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,0,0], 'OD1':[0,0,0], 'ND2':[0,2,0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,0,0], 'OE1':[0,0,0], 'NE2':[0,2,0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'ND1':[-1,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'NE2':[-1,1,1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,2,0], 'CE':[0,2,0], 'NZ':[0,3,1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,0,1], 'OH':[-1,1,0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'SD':[0,0,0], 'CE':[0,3,0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,0,1], 'NE1':[0,1,1], 'CE2':[0,0,1], 'CE3':[0,1,1], 'CZ2':[0,1,1], 'CZ3':[0,1,1], 'CH2':[0,1,1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                   'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0]/2+0.5,i_fea[1]/3,i_fea[2]]

    return atom_features

def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict ={'GLY':'G','ALA':'A','VAL':'V','ILE':'I','LEU':'L','PHE':'F','PRO':'P','MET':'M','TRP':'W','CYS':'C',
               'SER':'S','THR':'T','ASN':'N','GLN':'Q','TYR':'Y','HIS':'H','ASP':'D','GLU':'E','LYS':'K','ARG':'R'}
    atom_count = -1
    res_count = -1
    pdb_file = open(file_path,'r')
    pdb_res = pd.DataFrame(columns=['ID','atom','res','res_id','xyz','B_factor'])
    res_id_list = []
    before_res_pdb_id = None
    Relative_atomic_mass = {'H':1,'C':12,'O':16,'N':14,'S':32,'FE':56,'P':31,'BR':80,'F':19,'CO':59,'V':51,
                            'I':127,'CL':35.5,'CA':40,'B':10.8,'ZN':65.5,'MG':24.3,'NA':23,'HG':200.6,'MN':55,
                            'K':39.1,'AP':31,'AC':227,'AL':27,'W':183.9,'SE':79,'NI':58.7}


    while True:
        line = pdb_file.readline()
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count+=1
            res_pdb_id = int(line[22:26])
            if res_pdb_id != before_res_pdb_id:
                res_count +=1
            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ['N','CA','C','O','H']:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = res_dict[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                atom_fea = [0.5,0.5,0.5]

            try:
                bfactor = float(line[60:66])
            except ValueError:
                bfactor = 0.5

            tmps = pd.Series(
                {'ID': atom_count, 'atom':line[12:16].strip(),'atom_type':atom_type, 'res': res, 'res_id': int(line[22:26]),
                 'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                 'B_factor': bfactor,'mass':Relative_atomic_mass[atom_type],'is_sidechain':is_sidechain,
                 'charge':atom_fea[0],'num_H':atom_fea[1],'ring':atom_fea[2]})
            if len(res_id_list) == 0:
                res_id_list.append(int(line[22:26]))
            elif res_id_list[-1] != int(line[22:26]):
                res_id_list.append(int(line[22:26]))
            pdb_res = pdb_res.append(tmps,ignore_index=True)
        if line.startswith('ENDMDL'):
            break

    return pdb_res,res_id_list

def PDBFeature(query_id, PDB_chain_dir, results_dir):

    print('PDB_chain -> PDB_DF')
    pdb_path = PDB_chain_dir+'\{}.pdb'.format(query_id)
    pdb_DF, res_id_list = get_pdb_DF(pdb_path)

    with open(results_dir+'\{}.df'.format(query_id),'wb') as f:
        joblib.dump({'pdb_DF':pdb_DF,'res_id_list':res_id_list},f)

    print('Extract PDB_feature')

    res_sidechain_centroid = []
    res_types = []
    for res_id in res_id_list:
        res_type = pdb_DF[pdb_DF['res_id'] == res_id]['res'].values[0]
        res_types.append(res_type)

        res_atom_df = pdb_DF[pdb_DF['res_id'] == res_id]
        xyz = np.array(res_atom_df['xyz'].tolist())
        masses = np.array(res_atom_df['mass'].tolist()).reshape(-1,1)
        centroid = np.sum(masses*xyz,axis=0)/np.sum(masses)
        res_sidechain_atom_df = pdb_DF[(pdb_DF['res_id'] == res_id) & (pdb_DF['is_sidechain'] == 1)]
        if len(res_sidechain_atom_df) == 0:
            res_sidechain_centroid.append(centroid)
        else:
            xyz = np.array(res_sidechain_atom_df['xyz'].tolist())
            masses = np.array(res_sidechain_atom_df['mass'].tolist()).reshape(-1, 1)
            sidechain_centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
            res_sidechain_centroid.append(sidechain_centroid)

    res_sidechain_centroid = np.array(res_sidechain_centroid)

    return

def PDBResidueFeature(query_path,query_id):

    atom_vander_dict = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.85,'H':1.2,'D':1.2,'SE':1.9,'P':1.8,'FE':2.23,'BR':1.95,
                        'F':1.47,'CO':2.23,'V':2.29,'I':1.98,'CL':1.75,'CA':2.81,'B':2.13,'ZN':2.29,'MG':1.73,'NA':2.27,
                        'HG':1.7,'MN':2.24,'K':2.75,'AC':3.08,'AL':2.51,'W':2.39,'NI':2.22}
    for key in atom_vander_dict.keys():
        atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)

    with open('{}/{}.df'.format(query_path,query_id), 'rb') as f:
        tmp = joblib.load(f)
    pdb_DF, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
    pdb_DF = pdb_DF[pdb_DF['atom_type']!='H']

    # atom features
    mass = np.array(pdb_DF['mass'].tolist()).reshape(-1, 1)
    mass = mass / 32
    B_factor = np.array(pdb_DF['B_factor'].tolist()).reshape(-1, 1)
    if (max(B_factor) - min(B_factor)) == 0:
        B_factor = np.zeros(B_factor.shape) + 0.5
    else:
        B_factor = (B_factor - min(B_factor)) / (max(B_factor) - min(B_factor))
    is_sidechain = np.array(pdb_DF['is_sidechain'].tolist()).reshape(-1, 1)
    charge = np.array(pdb_DF['charge'].tolist()).reshape(-1, 1)
    num_H = np.array(pdb_DF['num_H'].tolist()).reshape(-1, 1)
    ring = np.array(pdb_DF['ring'].tolist()).reshape(-1, 1)
    atom_type = pdb_DF['atom_type'].tolist()
    atom_vander = np.zeros((len(atom_type), 1))
    for i, type in enumerate(atom_type):
        try:
            atom_vander[i] = atom_vander_dict[type]
        except:
            atom_vander[i] = atom_vander_dict['C']

    atom_feas = [mass, B_factor, is_sidechain, charge, num_H, ring, atom_vander]
    atom_feas = np.concatenate(atom_feas,axis=1)

    res_atom_feas = []
    atom_begin = 0
    for i, res_id in enumerate(res_id_list):
        res_atom_df = pdb_DF[pdb_DF['res_id'] == res_id]
        atom_num = len(res_atom_df)
        res_atom_feas_i = atom_feas[atom_begin:atom_begin + atom_num]
        res_atom_feas_i = np.average(res_atom_feas_i, axis=0).reshape(1, -1)
        res_atom_feas.append(res_atom_feas_i)
        atom_begin += atom_num
    res_atom_feas = np.concatenate(res_atom_feas, axis=0)
    print(res_atom_feas.shape)

    save = query_path + '\\' + query_id
    np.save(save, res_atom_feas)

    return

def Feature(query_path, query_dir, query_list):
    #
    for idx, filename in enumerate(query_list, start=1):
        query_id = 'save_' + filename.rsplit('.pdb', 1)[0]
        print(query_id)
        # feature extract
        print('1.extracting query chain...')
        SaveChainPDB(query_path, query_dir, filename, query_id)
        with open('{}\{}.pdb'.format(query_path, query_id),'r') as f:
            text = f.readlines()

        residue_num = 0
        for line in text:
            if line.startswith('ATOM'):
                residue_type = line[17:20]
                if residue_type not in ["GLY","ALA","VAL","ILE","LEU","PHE","PRO","MET","TRP","CYS",
                                        "SER","THR","ASN","GLN","TYR","HIS","ASP","GLU","LYS","ARG"]:
                    print("ERROR: There are mutant residues in your structure!")
                    raise ValueError

                residue_num += 1

        print('2.extracting features...')

        PDBFeature(query_id, query_path, query_path)

        PDBResidueFeature(query_path, query_id)

    return


def save_residue_features(query_path, query_list):
    num_proteins = len(query_list)
    batch_size = 10
    num_batches = num_proteins // batch_size
    if num_proteins % batch_size != 0:
        num_batches += 1
    print(num_batches)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_proteins)

        batch_features = []
        for idx in range(start_idx, end_idx):
            filename = query_list[idx]
            query_id = 'save_' + filename.rsplit('.pdb', 1)[0]
            feature_file = os.path.join(query_path, f'{query_id}.npy')
            if os.path.exists(feature_file):
                features = np.load(feature_file)
                batch_features.append(features)

        folder_name = f'batch'
        folder_path = os.path.join(query_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        if batch_features:
            save_path = os.path.join(folder_path, f'batch_residue_features_{start_idx + 1}_{end_idx}.npy')
            np.save(save_path, batch_features)

    return


query_path = "your_path"

if __name__ == '__main__':
    query_dir = "your_protein_structural_pathway"
    query_list = []
    for (dirpath, dirnames, filenames) in walk(query_dir):
        query_list.extend(filenames)
        break
    print(query_list)

    Feature(query_path, query_dir, query_list)
    save_residue_features(query_path, query_list)



