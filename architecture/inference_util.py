import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from architecture.data.rxn_graph import BondDissociate
from architecture.data.initial_containers import DGLwBDEMappings
from architecture.data.single_run_tools import prod_to_reac_atom_map, prod_to_reac_bond_map
from architecture.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize
from rdkit import Chem
import torch
import dgl
import pickle
import os

# Rutas candidatas para transforms
paths = ['model_data/transforms', 'deepbde/model_data/transforms']

# Buscar y cargar transforms
for path in paths:
    try:
        with open(path, 'rb') as f:
            transforms = pickle.load(f)
        # Obtener el directorio base para usarlo como raíz del modelo
        base_dir = os.path.dirname(path)
        break
    except FileNotFoundError:
        continue
else:
    raise FileNotFoundError(f"No se encontró el archivo 'transforms' en ninguna de las rutas: {paths}")

# Extraer datos del diccionario
out_mean = transforms['val_mean']
out_stdev = transforms['val_stdev']
stders = transforms['transform']

# Definir ruta del modelo relativo al transforms cargado
model_path = os.path.join(base_dir, 'model')

model = torch.load(model_path, map_location='cpu')
model.eval()

# atomic_num_list = [1, 6, 7, 8]
atomic_num_list = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17]
aprop = ['atomic_num', 'total_degree', 'total_num_hs', 'ring_of_size', 'is_in_ring']
bprop = ['is_in_ring', 'ring_of_size', 'dative']
gprop = ['num_atoms', 'num_bonds', 'total_weight']
featurizers = {'atom' : AtomFeaturize(aprop, atomic_num_list), 'bond' : BondFeaturize(bprop), 'global' : GlobalFeaturize(gprop)}

def sm_to_mol(cann):
    return Chem.AddHs(Chem.MolFromSmiles(cann))

def single_predict(reac_sm, broken_bond_idx, prod_sms=None):
    with torch.no_grad():
        r_m = sm_to_mol(reac_sm)

        # take from dataset bond splitting method
        Chem.Kekulize(r_m, clearAromaticFlags=True)
        bond = r_m.GetBondWithIdx(broken_bond_idx)
        mh = Chem.RWMol(r_m)
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        mh.RemoveBond(a1, a2)

        mh.GetAtomWithIdx(a1).SetNoImplicit(True)
        mh.GetAtomWithIdx(a2).SetNoImplicit(True)

        # Call SanitizeMol to update radicals
        Chem.SanitizeMol(mh)
        if prod_sms == None:
            p_ms = list(Chem.GetMolFrags(mh, asMols=True))
        else:
            p_ms = [sm_to_mol(sm) for sm in prod_sms]

        r_g = DGLwBDEMappings.dgl_from_mol(r_m)
        p_gs = [DGLwBDEMappings.dgl_from_mol(m) for m in p_ms]

        feats = {nt : [featurizers[nt](m) for m in [r_m, *p_ms]] for nt in featurizers}
        # feats = {nt : stders[nt].transform(torch.cat(feats[nt])) for nt in ['bond', 'atom', 'global']}
        feats = {nt : stders[nt](torch.cat(feats[nt])) for nt in ['bond', 'atom', 'global']}
        feats = {nt : torch.tensor(feats[nt], dtype=torch.float) for nt in ['bond', 'atom', 'global']}

        atom_map_for_rxn = prod_to_reac_atom_map([r_m, p_ms], [broken_bond_idx])
        bond_map_for_rxn = prod_to_reac_bond_map(p_ms, atom_map_for_rxn, r_m, broken_bond_idx)
        rxn_atom_mappings = DGLwBDEMappings.to_concat_map(atom_map_for_rxn)
        rxn_bond_mappings = DGLwBDEMappings.to_concat_map(bond_map_for_rxn)
        prods_has_bonds = [len(bm) > 0 for bm in bond_map_for_rxn[:-1]]

        rxn_feat_gen = BondDissociate(rxn_atom_mappings, rxn_bond_mappings, prods_has_bonds, None)
        rxn_feat_gen.reacs, rxn_feat_gen.prods = [0], [1, 2]

        pred = out_mean + (out_stdev * model(dgl.batch([r_g, *p_gs]), feats, [rxn_feat_gen]))

        return pred

def multi_predict(reac_sm, broken_bond_idxs):
    with torch.no_grad():
        r_m = sm_to_mol(reac_sm)

        # take from dataset bond splitting method
        Chem.Kekulize(r_m, clearAromaticFlags=True)
        r_g = DGLwBDEMappings.dgl_from_mol(r_m)

        def get_products_from_broken_idx(broken_bond_idx):
            bond = r_m.GetBondWithIdx(broken_bond_idx)
            mh = Chem.RWMol(r_m)
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            mh.RemoveBond(a1, a2)

            mh.GetAtomWithIdx(a1).SetNoImplicit(True)
            mh.GetAtomWithIdx(a2).SetNoImplicit(True)

            # Call SanitizeMol to update radicals
            Chem.SanitizeMol(mh)
            p_ms = list(Chem.GetMolFrags(mh, asMols=True))

            p_gs = [DGLwBDEMappings.dgl_from_mol(m) for m in p_ms]

            return p_ms, p_gs
        
        p_ms_gs = [get_products_from_broken_idx(broken_bond_idx) for broken_bond_idx in broken_bond_idxs]

        p_mss = [p_ms for p_ms, _p_gs in p_ms_gs]

        # BondDissociate object for each reaction
        def get_bond_dissociate(p_ms, broken_bond_idx, prod_pair_idx):
            atom_map_for_rxn = prod_to_reac_atom_map([r_m, p_ms], [broken_bond_idx])
            # print(atom_map_for_rxn)
            if len(atom_map_for_rxn) == 0:
                return None
            bond_map_for_rxn = prod_to_reac_bond_map(p_ms, atom_map_for_rxn, r_m, broken_bond_idx)
            rxn_atom_mappings = DGLwBDEMappings.to_concat_map(atom_map_for_rxn)
            rxn_bond_mappings = DGLwBDEMappings.to_concat_map(bond_map_for_rxn)
            prods_has_bonds = [len(bm) > 0 for bm in bond_map_for_rxn[:-1]]

            rxn_feat_gen = BondDissociate(rxn_atom_mappings, rxn_bond_mappings, prods_has_bonds, None)
            # rxn_feat_gen.reacs, rxn_feat_gen.prods = [0], [prod_pair_idx * 2 + 1, prod_pair_idx * 2 + 2]

            return rxn_feat_gen

        # filtering step for matches that were not found
        bond_dissocs = [get_bond_dissociate(p_ms, broken_bond_idx, prod_pair_idx) for prod_pair_idx, (p_ms, broken_bond_idx) in enumerate(zip(p_mss, broken_bond_idxs))]
        bond_dissocs_gs = [(bd, p_m_g) for bd, p_m_g in zip(bond_dissocs, p_ms_gs) if bd != None]
        bond_dissocs = [bd for bd, p_m_g in bond_dissocs_gs]
        p_ms_gs = [p_m_g for _bd, p_m_g in bond_dissocs_gs]
        p_mss = [p_ms for p_ms, _p_gs in p_ms_gs]
        p_gss = [p_gs for _p_ms, p_gs in p_ms_gs]

        for idx, bd in enumerate(bond_dissocs):
            bd.reacs = [0]
            bd.prods = [idx * 2 + 1, idx * 2 + 2]

        feats = {nt : [featurizers[nt](m) for m in [r_m, *[p_m for p_ms in p_mss for p_m in p_ms]]] for nt in featurizers}
        # feats = {nt : stders[nt].transform(torch.cat(feats[nt])) for nt in ['bond', 'atom', 'global']}
        feats = {nt : stders[nt](torch.cat(feats[nt])) for nt in ['bond', 'atom', 'global']}
        feats = {nt : torch.tensor(feats[nt], dtype=torch.float) for nt in ['bond', 'atom', 'global']}

        pred = out_mean + (out_stdev * model(dgl.batch([r_g, *[p_g for p_gs in p_gss for p_g in p_gs]]), feats, bond_dissocs))

        return pred

# return all valid bonds for deepbde and their predictions
def predict_all(reac_sm):    
    def valid_bond(bond):
        # if skip_rings and bond.IsInRing():
        #     return False

        if bond.GetBondTypeAsDouble() > 1.9999:
            return False
        
        return True
    
    bond_idxs = [i for i, bond in enumerate(sm_to_mol(reac_sm).GetBonds()) if valid_bond(bond)]

    preds = multi_predict(reac_sm, bond_idxs)

    return bond_idxs, preds