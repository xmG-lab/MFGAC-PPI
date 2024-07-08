import multiprocessing
import os
from os import walk
from multiprocessing import Pool, Process
import subprocess
from Bio.PDB import PDBParser, MMCIFParser
import warnings
import math
import torch
import numpy as np
from esm import FastaBatchedDataset, pretrained


warnings.filterwarnings("ignore")
dist_threshold = 10
N_REF_POINTS = 31
longToShort = {'GLY': 'G',
               'ALA': 'A',
               'VAL': 'V',
               'LEU': 'L',
               'ILE': 'I',
               'PHE': 'F',
               'TRP': 'W',
               'TYR': 'Y',
               'ASP': 'D',
               'HIS': 'H',
               'ASN': 'N',
               'GLU': 'E',
               'MET': 'M',
               'MSE': 'M',
               'ARG': 'R',
               'SER': 'S',
               'THR': 'T',
               'CYS': 'C',
               'PRO': 'P',
               'SEC': 'U',
               'PYL': 'O',
               'LYS': 'K',
               'GLN': 'Q',
               'UNK': 'X'}


class GetFasta:
    def __init__(self, query_dir, query_list, tmp_dir, n_process):
        self.query_dir = query_dir
        self.output_dir = tmp_dir
        self.query_list = query_list
        self.n = n_process
        with open(os.path.join(self.output_dir, "proteins.fasta"), "w"):
            pass

    def get_single_fasta(self, pdb_path: str) -> str:
        try:
            struc = PDBParser().get_structure('protein', pdb_path)
            model = struc[0]
        except:
            struc = MMCIFParser().get_structure('protein', pdb_path)
            model = next(iter(struc), None)
        atom_ids = ('CA',)
        fasta = ""
        for chain in model:
            for res in chain:
                for atom in atom_ids:
                    if atom in res:
                        resname = res.get_resname()
                        if resname not in list(longToShort.keys()):
                            fasta += 'X'
                        else:
                            fasta += longToShort[res.get_resname()]
        return fasta

    def split_query_list(self):
        sub_query_lists = []
        m = len(self.query_list) // self.n
        for i in range(self.n-1):
            sub_query_lists.append(self.query_list[m*i:m*(i+1)])
        sub_query_lists.append(self.query_list[m*(self.n-1):])
        return sub_query_lists

    def write_fasta(self, fasta_list):
        with open(os.path.join(self.output_dir, "proteins.fasta"), "a") as f:
            f.writelines(fasta_list)

    def run_get_single_fasta(self, sub_query_list):
        fasta_list = []
        for query in sub_query_list:
            query_path = os.path.join(self.query_dir, query)
            fasta = self.get_single_fasta(query_path)
            fasta_list.append(">"+query+"\n"+fasta+"\n")
        return fasta_list

    def get_fasta_multiproc(self):
        pool_runfasta = Pool(self.n)
        sub_query_lists = self.split_query_list()
        for sub_query_list in sub_query_lists:
            pool_runfasta.apply_async(self.run_get_single_fasta, (sub_query_list,), callback=self.write_fasta)
        pool_runfasta.close()
        pool_runfasta.join()


class ESM:
    def __init__(self, fasta_file, tmp_dir, device, n_process=1):
        self.fasta = fasta_file
        self.tmp_dir = tmp_dir
        self.model_location = 'esm2_t33_650M_UR50D'
        self.n = n_process
        self.device = device


    def split_fasta_file(self):
        with open(self.fasta, 'r') as f:
            lines = f.readlines()
        lines_count = len(lines)
        lines_per_file = (lines_count // (self.n * 2) + 1) * 2
        for i in range(self.n):
            file_path = os.path.join(self.tmp_dir, f"fasta_{i}")
            with open(file_path, 'w') as f_out:
                f_out.writelines(lines[i * lines_per_file: (i + 1) * lines_per_file])

    def run_esm2(self, fasta_file):
        model, alphabet = pretrained.load_model_and_alphabet(self.model_location)
        model.eval()
        model.to(self.device)

        dataset = FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches, pin_memory=True
        )

        result_dict = {}
        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                toks = toks.to(device=self.device, non_blocking=True)

                out = model(toks, repr_layers=[33], return_contacts=False)

                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }
                for i, label in enumerate(labels):
                    truncate_len = min(1022, len(strs[i]))
                    result = {
                        layer: t[i, 1: truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                    result_dict[f"{label}"] = np.array(result[33])
        return result_dict

    def run_esm2_multiproc(self, result_dict):
        pool_runesm = Pool(self.n)
        results = []
        for i in range(self.n):
            fasta_file = os.path.join(self.tmp_dir, f"fasta_{i}")
            results.append(pool_runesm.apply_async(self.run_esm2, (fasta_file,)))
        pool_runesm.close()
        pool_runesm.join()
        esm_feature_dict = {}
        for result in results:
            esm_feature_dict.update(result.get())
        result_dict['esm_feature_dict'] = esm_feature_dict


def get_data(query_dir, query_list, tmp_dir, n_process=4, n_gpu_process=1, device=torch.device('cpu')):
    getfasta = GetFasta(query_dir, query_list, tmp_dir, n_process=n_process)
    getfasta.get_fasta_multiproc()
    esm_runner = ESM(f"{tmp_dir}/proteins.fasta", tmp_dir, n_process=n_gpu_process, device=device)
    esm_runner.split_fasta_file()
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    p1 = Process(target=esm_runner.run_esm2_multiproc, args=(result_dict,))

    p1.start()
    p1.join()

    return result_dict

import os
import numpy as np


def save_features_batch(esm_features, output_dir, batch_size):
    batch_count = math.ceil(len(esm_features) / batch_size)
    for i in range(batch_count):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, len(esm_features))
        batch_values = list(esm_features.values())[batch_start:batch_end]
        filename = os.path.join(output_dir, f"esm_features_batch_{batch_start + 1}_{batch_end}.npy")
        np.save(filename, batch_values)



# 使用示例
output_dir = "./esm_features_batches"
os.makedirs(output_dir, exist_ok=True)



if __name__ == "__main__":
    query_dir = "your_protein_structural_pathway"
    query_list = []
    for (dirpath, dirnames, filenames) in walk(query_dir):
        query_list.extend(filenames)
        break
    print(query_list)
    # query_list = os.listdir(query_dir)
    tmp_dir = "./tmp_dir"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    n_process = 4  
    n_gpu_process = 1 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    result_dict = get_data(query_dir, query_list, tmp_dir, n_process=n_process, n_gpu_process=n_gpu_process, device=device)

    esm_features = result_dict['esm_feature_dict']
    for query_id, features in esm_features.items():
        print(f"ESM features for {query_id}: {features}")
        print(len(features))
    save_features_batch(esm_features, output_dir, batch_size=10)