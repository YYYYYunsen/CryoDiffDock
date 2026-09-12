import os
import pickle
from argparse import ArgumentParser
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from Bio import SeqIO

from datasets.constants import three_to_one

parser = ArgumentParser()
parser.add_argument('--out_file', type=str, default="data/prepared_for_esm.fasta")
parser.add_argument('--dataset', type=str, default="pdbbind")
parser.add_argument('--data_dir', type=str, default='../data/BindingMOAD_2020_ab_processed_biounit/pdb_protein/', help='')
args = parser.parse_args()

biopython_parser = PDBParser()


def get_structure_from_file(file_path):
    structure = biopython_parser.get_structure('random_id', file_path)
    structure = structure[0]
    l = []
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    # 已经替换为 A，顺便修正了 print 打印的文案
                    seq += 'A'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex ', file_path, '. Replacing it with Alanine (A).')
        
        # 保护机制：如果某条链解析出来是空的（比如全是杂原子），不加入列表，防止生成非法的空 FASTA 序列
        if seq:
            l.append(seq)
    return l

data_dir = args.data_dir
names = os.listdir(data_dir)

if args.dataset == 'pdbbind':
    sequences = []
    ids = []

    for name in tqdm(names):
        dir_path = os.path.join(data_dir, name)
        
        # 1. 拦截器一：必须是文件夹（直接过滤掉 .DS_Store, log.txt, .csv 等一切幽灵文件）
        if not os.path.isdir(dir_path):
            continue
            
        # 2. 拦截器二：文件夹中必须至少包含一个 .npz 文件
        has_npz = any(fname.endswith('.npz') for fname in os.listdir(dir_path))
        if not has_npz:
            continue

        # 确定蛋白质文件路径
        if os.path.exists(os.path.join(dir_path, f'{name}_protein_processed.pdb')):
            rec_path = os.path.join(dir_path, f'{name}_protein_processed.pdb')
        else:
            rec_path = os.path.join(dir_path, f'{name}_protein.pdb')
            
        # 3. 拦截器三：必须存在对应的 PDB 文件
        if not os.path.exists(rec_path):
            continue
            
        l = get_structure_from_file(rec_path)
        for i, seq in enumerate(l):
            sequences.append(seq)
            ids.append(f'{name}_chain_{i}')
            
    records = []
    for (index, seq) in zip(ids, sequences):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ''
        records.append(record)
    SeqIO.write(records, args.out_file, "fasta")

elif args.dataset == 'moad':
    names = [n[:6] for n in names]
    name_to_sequence = {}

    for name in tqdm(names):
        if name == '.DS_Store': continue
        if os.path.exists(os.path.join(data_dir, f'{name}_protein_processed.pdb')):
            rec_path = os.path.join(data_dir, f'{name}_protein_processed.pdb')
        elif os.path.exists(os.path.join(data_dir, f'{name}_protein.pdb')):
            rec_path = os.path.join(data_dir, f'{name}_protein.pdb')
        else:
            print(f"We are skipping {name} because there was no {name}_protein.pdb or {name}_protein_processed.pdb")
            continue
        l = get_structure_from_file(rec_path)
        for i, seq in enumerate(l):
            name_to_sequence[name + '_chain_' + str(i)] = seq

    # save to file
    with open(args.out_file, 'wb') as f:
        pickle.dump(name_to_sequence, f)