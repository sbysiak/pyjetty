#!/usr/bin/env python


from __future__ import print_function

import fastjet as fj
import fjcontrib
import fjext
import fjtools

import tqdm
import argparse
import os
import numpy as np

from pyjetty.mputils import MPBase, pwarning, pinfo, perror, treewriter, jet_analysis
from pyjetty.alice_analysis.process.base import common_base
from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext

import warnings
warnings.simplefilter('ignore',lineno=755)
from functools import partial

# see:
# https://github.com/matplo/pyjetty/blob/5a515bf7714d7c040ae90ebf5d4343dbee1a4874/pyjetty/alice_analysis/process/user/james/process_groomers.py#L824-L1033



class RealBckgGenerator(common_base.CommonBase):
    def __init__(self,
                 path_to_data,
                 rm_n_leading_jets=0,
                 jet_def=fj.JetDefinition(fj.kt_algorithm, 0.4),
                 centrality=(0,10),
				 mult_min=0,
                 seed=None,
                 verbose=True):
        self.list_of_files = process_files_list(path_to_data)
        self.rm_n_leading_jets = rm_n_leading_jets
        self.jet_def = jet_def
        self.centrality = centrality
        self.mult_min = mult_min
        self.last_event_prop = dict()
        self.verbose = verbose
        self.rng = np.random.default_rng(seed=seed)
        self.cache = []


    def __str__(self):
        s = []
        variables = self.__dict__.keys()
        for v in variables:
          if v == 'list_of_files':
              l = self.__dict__[v]
              s.append('{} = list of {} files, from {} to {}'.format(v, len(l), l[0], l[-1]))
          else:
              s.append('{} = {}'.format(v, self.__dict__[v]))
        return "[i] {} with \n .  {}".format(self.__class__.__name__, '\n .  '.join(s))

    def _hash(self, s):
        return hash(s) % int(1e7)

    def _rm_leading_jets(self, tracks):
        from functools import reduce
        cs = fj.ClusterSequence(tracks, self.jet_def)
        jets_wo_n_leading = fj.sorted_by_pt(cs.inclusive_jets())[self.rm_n_leading_jets:]
        # merge all constituents of all jets except N leading
        tracks_wo_leading_jets = reduce(lambda a, b: a+b, [j.constituents() for j in jets_wo_n_leading])

        self.last_event_prop['mult_wo_lead'] = len(tracks_wo_leading_jets)
        self.last_event_prop['pTmean_wo_lead'] = np.mean([p.pt() for p in tracks_wo_leading_jets])
        # return tracks_wo_leading_jets
        # or force vectorized output:
        return fjext.vectorize_pt_eta_phi([p.pt() for p in tracks_wo_leading_jets],
                                   [p.eta() for p in tracks_wo_leading_jets],
                                   [p.phi() for p in tracks_wo_leading_jets],
                                   int(-1e6))

    def _read_random_event(self):
        import uproot
        # TODO: set random seed

        # TODO: for now each file has equal weight
        # not correct as this way an event from low-Nev file has higher chance to be selected
        # better: set probs ~ to size or #events in each file
        file_sel = self.rng.choice(self.list_of_files)
        froot = uproot.open(file_sel)
        tree_ev = froot['PWGHF_TreeCreator/tree_event_char']
        tree_p = froot['PWGHF_TreeCreator/tree_Particle']

        # TODO: add cut on z_vtx?
        # events_avail = tree_ev.pandas.df(branches=['ev_id', 'centrality']).query('centrality > @self.centrality[0] and centrality < @self.centrality[1]')
        # event_sel = np.random.choice(events_avail['ev_id'])
        events_avail = tree_ev.arrays("ev_id", library="numpy", cut=f"(centrality > {self.centrality[0]}) & (centrality < {self.centrality[1]})")['ev_id']
        event_sel = self.rng.choice(events_avail)

        # pt_eta_phi_arr = tree_p.pandas.df().query('ev_id == @event_sel')[['ParticlePt', 'ParticleEta', 'ParticlePhi']].to_numpy().T
        pt_eta_phi_arr = tree_p.arrays(["ParticlePt", "ParticleEta", "ParticlePhi"], cut=(f"ev_id == {event_sel}"), library="pandas").to_numpy().T

        # Use swig'd function to create a vector of fastjet::PseudoJets from numpy arrays of pt,eta,phi
        user_index_offset = int(-1e6)
        fj_particles = fjext.vectorize_pt_eta_phi(*pt_eta_phi_arr, user_index_offset)

        self.last_event_prop = dict(file=file_sel, ev_id=event_sel, file_hash=self._hash(file_sel),
                                    # centrality=events_avail.query('ev_id == @event_sel').iloc[0]['centrality'],
                                    centrality=tree_ev.arrays('centrality', cut=f'ev_id == {event_sel}', library='numpy')['centrality'][0],
                                    mult=len(fj_particles), pTmean=np.mean([p.pt() for p in fj_particles]))
        return fj_particles

    def _read_all_events_from_file(self, file_sel=None):
        import uproot

        if file_sel is None:
            file_sel = self.rng.choice(self.list_of_files)
        print(f'BCKG: reading file {file_sel}')
        froot = uproot.open(file_sel)
        tree_ev = froot['PWGHF_TreeCreator/tree_event_char']
        tree_p = froot['PWGHF_TreeCreator/tree_Particle']

        # TODO: add cut on z_vtx?
        # events_avail = tree_ev.pandas.df(branches=['ev_id', 'centrality']).query('centrality > @self.centrality[0] and centrality < @self.centrality[1]')
        # event_sel = np.random.choice(events_avail['ev_id'])
        particles_all = tree_p.arrays(["ParticlePt", "ParticleEta", "ParticlePhi", "ev_id"], library="pandas")
        events_avail = tree_ev.arrays("ev_id", library="numpy", cut=f"(centrality > {self.centrality[0]}) & (centrality < {self.centrality[1]})")['ev_id']
        for event_sel in events_avail:
            # event_sel = np.random.choice(events_avail)

            # pt_eta_phi_arr = tree_p.pandas.df().query('ev_id == @event_sel')[['ParticlePt', 'ParticleEta', 'ParticlePhi']].to_numpy().T
            # pt_eta_phi_arr = tree_p.arrays(["ParticlePt", "ParticleEta", "ParticlePhi"], cut=(f"ev_id == {event_sel}"), library="pandas").to_numpy().T
            pt_eta_phi_arr = particles_all.query(f"ev_id == {event_sel}")[["ParticlePt", "ParticleEta", "ParticlePhi"]].to_numpy().T
            if (pt_eta_phi_arr.shape[1] < self.mult_min):
                continue
            # Use swig'd function to create a vector of fastjet::PseudoJets from numpy arrays of pt,eta,phi
            user_index_offset = int(-1e6)
            fj_particles = fjext.vectorize_pt_eta_phi(*pt_eta_phi_arr, user_index_offset)
            event_prop = dict(file=file_sel, ev_id=event_sel, file_hash=self._hash(file_sel),
                                        # centrality=events_avail.query('ev_id == @event_sel').iloc[0]['centrality'],
                                        centrality=tree_ev.arrays('centrality', cut=f'ev_id == {event_sel}', library='numpy')['centrality'][0],
                                        mult=len(fj_particles), pTmean=np.mean([p.pt() for p in fj_particles]))
            self.cache.append((fj_particles, event_prop))
            # return fj_particles

    def load_event(self, cache=True):
        # alternative, faster implementation:
        # read N or all events from given file and cache it
        if cache:
            return self._load_event_cached()
        i = 1
        while True:
            try:
                tracks = self._read_random_event()
                if len(tracks) < self.mult_min:
                    continue
                n_before = len(tracks)
                if self.rm_n_leading_jets > 0:
                    tracks = self._rm_leading_jets(tracks)
            except TypeError as e:
                i+=1
                print('FAIL FOR ', self.last_event_prop, '\n', e)
                continue
            break
        n_after = len(tracks)
        if self.verbose:
            print(f'load_event():: success after {i} trials')
            # print(f'#tracks before removal of {self.rm_n_leading_jets} leading jets: {n_before}, after: {n_after}')
            print(self.last_event_prop)
        return tracks

    def _load_event_cached(self):
        i = 1
        while True:
            try:
                if not len(self.cache):
                    print('BCKG: cache empty')
                    self._read_all_events_from_file()
                print(f'BCKG: cache N={len(self.cache)} evts')
                tracks, self.last_event_prop = self.cache.pop()
                n_before = len(tracks)
                if self.rm_n_leading_jets > 0:
                    tracks = self._rm_leading_jets(tracks)
            except TypeError as e:
                i+=1
                print('FAIL FOR ', self.last_event_prop, '\n', e)
                continue
            break
        n_after = len(tracks)
        if self.verbose:
            print(f'load_event():: success after {i} trials')
            # print(f'#tracks before removal of {self.rm_n_leading_jets} leading jets: {n_before}, after: {n_after}')
            print(self.last_event_prop)
        return tracks

def flag_prong_matching(j_comb, j_true, jet_def, grooming_algo='', grooming_params=[], store_values=None):
    #  Return flag based on where >50% of subleading matched pt resides:
    #    1: subleading
    #    2: leading, swap (>10% of leading in subleading)
    #    3: leading, mis-tag (<10% of leading in subleading)
    #    4: ungroomed
    #    5: outside
    #    6: other (i.e. 50% is not in any of the above)
    #    7: pp-truth passed grooming, but combined jet failed grooming
    #    8: combined jet passed grooming, but pp-truth failed grooming
    #    9: both pp-truth and combined jet failed SoftDrop
    gshop_comb = fjcontrib.GroomerShop(j_comb, jet_def)
    gshop_true = fjcontrib.GroomerShop(j_true, jet_def)

    ls_comb = getattr(gshop_comb, grooming_algo)(*grooming_params)
    ls_true = getattr(gshop_true, grooming_algo)(*grooming_params)

    has_parents_comb = ls_comb.pair().has_constituents()
    has_parents_true = ls_true.pair().has_constituents()

    if has_parents_true and has_parents_comb:
        matched_pt_subleading_subleading = fjtools.matched_pt(ls_comb.softer(), ls_true.softer())
        matched_pt_subleading_leading = fjtools.matched_pt(ls_comb.harder(), ls_true.softer())
        matched_pt_leading_subleading = fjtools.matched_pt(ls_comb.softer(), ls_true.harder())
        matched_pt_subleading_groomed = fjtools.matched_pt(ls_comb.pair(), ls_true.softer())
        matched_pt_subleading_ungroomed = fjtools.matched_pt(j_comb, ls_true.softer())
        matched_pt_subleading_ungroomed_notgroomed = matched_pt_subleading_ungroomed - matched_pt_subleading_groomed
        matched_pt_subleading_outside = 1 - matched_pt_subleading_ungroomed
        matched_pt_leading_leading = fjtools.matched_pt(ls_comb.harder(), ls_true.harder())

        if store_values is not None:
            store_values['subleading_subleading'] = matched_pt_subleading_subleading
            store_values['subleading_leading'] = matched_pt_subleading_leading
            # store_values['leading_subleading'] = matched_pt_leading_subleading
            # store_values['subleading_groomed'] = matched_pt_subleading_groomed
            # store_values['subleading_ungroomed'] = matched_pt_subleading_ungroomed
            # store_values['subleading_ungroomed_notgroomed'] = matched_pt_subleading_ungroomed_notgroomed
            # store_values['subleading_outside'] = matched_pt_subleading_outside
            # store_values['leading_leading'] = matched_pt_leading_leading

        prong_matching_threshold = 0.5
        if matched_pt_subleading_subleading > prong_matching_threshold:
          return 1
        elif matched_pt_subleading_leading > prong_matching_threshold:
          if matched_pt_leading_subleading > 0.1:
            return 2
          else:
            return 3
        elif matched_pt_subleading_ungroomed_notgroomed > prong_matching_threshold:
          return 4
        elif matched_pt_subleading_outside > prong_matching_threshold:
          return 5
        elif matched_pt_leading_leading >= 0.:
          return 6
        else:
          pwarning('Warning -- flag not specified!')
          return -1
    elif has_parents_true:
          return 7
    elif  has_parents_comb:
          return 8
    else:
          return 9

def match_dR(j, partons, drmatch = 0.1):
    mps = [p for p in partons if j.delta_R(p) < drmatch]
    # for p in fj.sorted_by_pt(mps)[0]:
    if len(mps) < 1:
        return None, False, False
    p = fj.sorted_by_pt(mps)[0]
    pyp = pythiafjext.getPythia8Particle(p)
    # print(p, pyp.id(), pyp.isQuark(), pyp.isGluon())
    return pyp.id(), pyp.isQuark(), pyp.isGluon()

def process_files_list(input_files_param):
    """
    if `input_files_param` is file ending with txt extension then returns list of files from that file
        lines starting with hashtag `#` are skipped
    if `input_files_param` is command starting from `supported_commands` then it executes it and returns list of files
    otherwise assume `input_files_param` is directory containing the files on which `ls` is run
    """
    if input_files_param is None:
        return None
    list_of_files = []
    if input_files_param.endswith(".txt"):
        print(f"\nReading files' names from the file: {input_files_param}")
        with open(input_files_param) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                list_of_files.append(line.replace("\n", ""))
    else:
        import subprocess
        supported_commands = ("ls", "find")  # must be tuple not list
        if input_files_param.startswith(supported_commands):
            cmd = input_files_param
        else:
            cmd = f"ls {input_files_param}"
        print(f"\nRunning command: {cmd}")
        cmd_output = subprocess.check_output(cmd, shell=True, text=True)
        for line in cmd_output.split("\n"):
            if line.startswith("#"):
                continue
            if line:
                list_of_files.append(line.replace("\n", ""))
    if len(list_of_files) < 1:
        raise ValueError("List of files empty!")
    from_to = (
        f"from\n{list_of_files[0]}\nto\n{list_of_files[-1]}"
        if len(list_of_files) > 2
        else f"{list_of_files}"
    )
    print(f"List of N={len(list_of_files)} files created, {from_to}")
    return list_of_files

def print_pyp(pyp, prefix='some psj '):
    # print(type(pyp))
    if type(pyp) == pythia8.Particle:
        # print(prefix+f" pt,eta,phi = {pyp.pT():.1f}, {pyp.eta():.3f}, {pyp.phi():.3f}, id={pyp.name() if pyp else ''}, index={pyp.index() if pyp else ''}, mothers={' <-'.join([pythia.event[idx].name() for idx in pyp.motherList()]) if pyp else ''} {'      <-----' if pyp and any([abs(pythia.event[idx].id()) in B_hadrons_MC_codes for idx in pyp.motherList()]) else ''}")
        print(prefix+f" pt,eta,phi = {pyp.pT():.1f}, {pyp.eta():.3f}, {pyp.phi():.3f}")
        # print('^^ CASE 1', prefix)
    else:
        if type(pyp) == fj.PseudoJet and pythiafjext.getPythia8Particle(pyp):
            print_pyp(pythiafjext.getPythia8Particle(pyp), prefix)
            # print('^^ CASE 2', prefix)
        else:
            print(prefix+f' pt,eta,phi = {pyp.pt():.1f}, {pyp.eta():.3f}, {pyp.phi():.3f}')
            # print('^^ CASE 3', type(pyp), prefix)

def print_jet(j, constit=True, substr=dict(), add_info=[], prefix=''):
    # print(f'pt,eta,phi = {j.pt():.2f}, {j.eta():.3f}, {j.phi():.3f} \t(px,py,pz = {j.px():.2f}, {j.py():.2f}, {j.pz():.2f})')
    # print(f'px,py,pz = ')
    print_pyp(j, prefix+f' jet (Nc={len(j.constituents())})')
    # print(f'Nconst={len(j.constituents())}')
    if add_info:
        for f in add_info:
            print(prefix+'\t'+f(j))
    if constit:
        for c in fj.sorted_by_pt(j.constituents()):
            print_pyp(c, prefix+'\tconstit')
        print()
    # if substr:
    #     # print(substr)
    #     for k,elems in substr.items():
    #         for i,e in enumerate(elems):
    #             print_pyp(e, prefix+f'\t{i}. {k}')
    #         print()
    pass

def print_aver_pt(j):
    return f'<pt>={np.mean([c.pt() for c in j.constituents()]):.2f}'

def print_true_frac(j):
    n_true = len(([p.pt() for p in j.constituents() if p.user_index() > 0]))
    n_bckg = len(([p.pt() for p in j.constituents() if p.user_index() < 0]))
    ptsum_true = np.sum([p.pt() for p in j.constituents() if p.user_index() > 0])
    ptsum_bckg = np.sum([p.pt() for p in j.constituents() if p.user_index() < 0])
    return f'true pt frac = {ptsum_true/(ptsum_true+ptsum_bckg):.2f}, true n frac = {n_true/(n_true+n_bckg):.2f}'

def print_match(j_match, j):
    return f'dR = {j.delta_R(j_match):.2f}, shared_pt_frac = {fjtools.matched_pt(j_match, j):.2f} ,  {fjtools.matched_pt(j, j_match):.2f}'

def groomer2str(g_algo, g_params):
    if g_algo == 'soft_drop' and len(g_params) == 3:
        # drop last param which is just jet radius
        g_params = g_params[:2]
    return g_algo + ("" if not g_params else "_".join([str(p).replace(".", "") for p in ["",]+ g_params]))

def ls2str(ls):
    # print(dir(ls))
    return f'z={ls.z():.3f} Delta={ls.Delta():.3f} kt={ls.kt():.2f} Erad={ls.kt()/(ls.z()*ls.Delta()):.2f}'

def is_hb(pid):
    pid = abs(pid)
    return (pid > 500 and pid < 600) or (pid > 5000 and pid < 6000)

def is_hc(pid):
    pid = abs(pid)
    return (pid > 400 and pid < 500) or (pid > 4000 and pid < 5000)



# def main():
parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly', prog=os.path.basename(__file__))
pyconf.add_standard_pythia_args(parser)
### steering
parser.add_argument('--nw', help="no warn", default=False, action='store_true')
parser.add_argument('--ignore-mycfg', help="ignore some settings hardcoded here", default=False, action='store_true')
parser.add_argument('--enable-thermal-background', help="enable thermal background calc", default=True, action='store_true')
parser.add_argument('--enable-real-background', help="enable real background (from data), supersedes thermal", default=False, action='store_true')
parser.add_argument('--output', help="output file name", default='leadsj_vs_x_output.root', type=str)
parser.add_argument('--dont-write-true-jets', default=False, action='store_true')
parser.add_argument('--dont-write-comb-jets', default=False, action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')

### jet and phase-space params
parser.add_argument('--min-jet-pt', help="min jet pT for selector", default=5.0, type=float)
parser.add_argument('--eta', help="set eta range must be uniform (e.g. abs(eta) < 0.9, which is ALICE TPC fiducial acceptance)",
                    type=float, default=0.9)
parser.add_argument('--jet-R', help="jet radius", type=float, default=0.4)
parser.add_argument('--recluster-algo', help="reclustering algo: KT or AKT or CA (=default)", default='CA', type=str)

### background
parser.add_argument('--bckg-seed', help="pr gen seed", type=int, default=1111)

### real background
parser.add_argument('--path-real-background', help="text file with list of files or path to directory containing data", default='./data/', type=str)

### thermal bckg, params from arxiv2006.01812
parser.add_argument('--thermal-Navg', help="mean dN/deta particles per event in thermal background generator", type=int, default=1800)
parser.add_argument('--thermal-Nsigma', help="sigma of N particles per event in thermal background generator", type=int, default=400)
parser.add_argument('--thermal-beta', help="beta in thermal background generator", type=float, default=0.5)
parser.add_argument('--thermal-alpha', help="alpha in thermal background generator", type=float, default=2)

### ConstituentSubtractor
parser.add_argument('--dRmax', help="dRmax in CS", default=0.25, type=float)
parser.add_argument('--alpha', help="alpha in CS", default=0, type=float)



args = parser.parse_args()

# print the banner first
fj.ClusterSequence.print_banner()
print()
# set up our jet definition and a jet selector
jet_R0 = args.jet_R
jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0)
jet_selector = fj.SelectorPtMin(args.min_jet_pt) & fj.SelectorPtMax(2000.0) & fj.SelectorAbsEtaMax(args.eta - jet_R0) # & fj.SelectorPtMin(args.py_pthatmin)
print(jet_def)

all_jets = []


mycfg = ['']
if args.ignore_mycfg:
    mycfg = []
pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)
if not pythia:
    perror("pythia initialization failed.")

if args.recluster_algo == 'CA':
    jet_def_lund = fj.JetDefinition(fj.cambridge_algorithm, 1.0)
elif args.recluster_algo == 'AKT':
    jet_def_lund = fj.JetDefinition(fj.antikt_algorithm, 1.0)
elif args.recluster_algo == 'KT':
    jet_def_lund = fj.JetDefinition(fj.kt_algorithm, 1.0)
else:
    raise ValueError(args.recluster_algo)
lund_gen = fjcontrib.LundGenerator(jet_def_lund)
print (lund_gen.description())

jet_matching_distance = 0.6
print(f'jet matching distance: {jet_matching_distance} (match jets with dR < jet_matching_distance * jetR)')

# jet_def_rc005 = fj.JetDefinition(fj.antikt_algorithm, 0.05)
# jet_def_rc01 = fj.JetDefinition(fj.antikt_algorithm, 0.1)
# jet_def_rc02 = fj.JetDefinition(fj.antikt_algorithm, 0.2)
# jet_def_rc03 = fj.JetDefinition(fj.antikt_algorithm, 0.3)
# print(jet_def_rc005)
# print(jet_def_rc01)
# print (jet_def_rc02)
# print (jet_def_rc03)

if not args.dont_write_true_jets: tw_true = treewriter.RTreeWriter(name = 'jets_true', file_name = args.output.replace('.root', '_true.root'))
if not args.dont_write_comb_jets: tw_comb = treewriter.RTreeWriter(name = 'jets_comb', file_name = args.output.replace('.root', '_comb.root'))

tgbkg = None
bckg_gen = None
if args.enable_thermal_background or args.enable_real_background:
    from pyjetty.mputils import CEventSubtractor, CSubtractorJetByJet
    cs = CEventSubtractor(alpha=args.alpha,
                         max_distance=args.dRmax,
                         max_eta=args.eta,
                         bge_rho_grid_size=0.25, # or 1.0  as in https://github.com/matplo/pyjetty/blob/5a515bf7714d7c040ae90ebf5d4343dbee1a4874/pyjetty/alice_analysis/config/theta_g/PbPb/james_groomers_thermal.yaml
                         max_pt_correct=100)
    print(cs)
    if args.enable_real_background:
        # raise NotImplementedError
        bckg_gen = RealBckgGenerator(args.path_real_background, verbose=args.verbose, seed=args.bckg_seed, mult_min=1800)

    elif args.enable_thermal_background:
        from pyjetty.alice_analysis.process.base import thermal_generator
        ### hopefully this sets seed for thermal generator
        np.random.seed(args.bckg_seed)
        bckg_gen = thermal_generator.ThermalGenerator(N_avg=args.thermal_Navg * 2 * args.eta,
                                                 sigma_N=args.thermal_Nsigma, # not specified in ref
                                                 beta=args.thermal_beta,
                                                 alpha=args.thermal_alpha,
                                                 eta_max=args.eta)
    print(bckg_gen)



res_arr = []
no_match_counter_true = 0
match_counter_true = 0
no_match_counter_comb = 0
match_counter_comb = 0

exec_id = np.random.randint(1e6)
print(f'excution ID = {exec_id}')
for i_event in tqdm.tqdm(range(args.nev)):
    if args.verbose:
        print('\n---\n')
    if not pythia.next():
        continue
    # parts = pythiafjext.vectorize(pythia, True, -1, 1, False)
    partons = pythiafjext.vectorize_select(pythia, [pythiafjext.kParton], 0, True)
    parts = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, False)
    # parts = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, False)
    jets = jet_selector(jet_def(parts))

    fj_particles_truth = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, False)

    ### skip events without true jet
    # if not len(jet_selector(fj.ClusterSequence(fj_particles_truth, jet_def).inclusive_jets())): continue


    if bckg_gen:
        fj_particles_combined_beforeCS = bckg_gen.load_event()

        # if args.verbose:
            # print('BEFORE: ',len(fj_particles_combined_beforeCS))
        [fj_particles_combined_beforeCS.push_back(p) for p in fj_particles_truth]
        # if args.verbose:
            # print('AFTER: ', len(fj_particles_combined_beforeCS), len([p for p in fj_particles_combined_beforeCS if p.user_index() < 0]))
    else:
        raise NotImplementedError
        fj_particles_combined = fj_particles_truth

    # Perform constituent subtraction for each R_max
    # fj_particles_combined = [self.constituent_subtractor[i].process_event(fj_particles_combined_beforeCS) for i, R_max in enumerate(self.max_distance)]
    fj_particles_combined = cs.process_event(fj_particles_combined_beforeCS)
    # fj_particles_combined = fj_particles_combined_beforeCS

    bckg_prop = dict()
    for part_set, prefix in [(fj_particles_combined_beforeCS, 'comb_beforeCS'),
                             (fj_particles_combined, 'comb_afterCS'),
                             ([p for p in fj_particles_combined_beforeCS if p.user_index() < 0], 'pureBckg_beforeCS'),
                             ([p for p in fj_particles_combined if p.user_index() < 0], 'pureBckg_afterCS')
                            ]:
        bckg_prop[prefix+'_mult'] = len(part_set)
        bckg_prop[prefix+'_pTmean'] = np.mean([p.pt() for p in part_set])

    cs_truth = fj.ClusterSequence(fj_particles_truth, jet_def)
    jets_truth = fj.sorted_by_pt(cs_truth.inclusive_jets())
    jets_truth_selected = jet_selector(jets_truth)
    # jets_truth_selected_matched = jet_selector_truth_matched(jets_truth)

    # cs_combined = fj.ClusterSequence(fj_particles_combined[i], jet_def)
    cs_combined = fj.ClusterSequence(fj_particles_combined, jet_def)
    jets_combined = fj.sorted_by_pt(cs_combined.inclusive_jets())
    jets_combined_selected = jet_selector(jets_combined)

    cs_combined_noCS = fj.ClusterSequence(fj_particles_combined_beforeCS, jet_def)
    jets_combined_noCS = fj.sorted_by_pt(cs_combined_noCS.inclusive_jets())
    jets_combined_noCS_selected = jet_selector(jets_combined_noCS)

    cs_bckg = fj.ClusterSequence([p for p in fj_particles_combined if p.user_index() < 0], jet_def)
    jets_bckg = fj.sorted_by_pt(cs_bckg.inclusive_jets())
    jets_bckg_selected = jet_selector(jets_bckg)


    if args.verbose:
        for j in jets_truth_selected:
            print_jet(j, prefix='true', constit=False, add_info=[print_aver_pt,])
        for j in jets_combined_selected:
            print_jet(j, prefix='combined', constit=False, add_info=[print_aver_pt, print_true_frac])
        print('combined jets only having true pt above threshold:')
        for j in jets_combined_noCS_selected:
            ptsum_true = np.sum([p.pt() for p in j.constituents() if p.user_index() > 0])
            if ptsum_true < 5: continue
            print_jet(j, prefix='combined_noCS', constit=False, add_info=[print_aver_pt, print_true_frac])

        for j in jets_bckg_selected:
            print_jet(j, prefix='bckg', constit=False, add_info=[print_aver_pt,])


    if not args.dont_write_true_jets:
        # match combined jets to the pythia jets
        for j_true in jets_truth_selected:
            if not hasattr(j_true, 'matches'):
                j_true.matches = []
            else:
                perror(f'matched jets array already there - it should not happen {j_true.matches}')
            for j_comb in jets_combined_selected:
                dR = j_true.delta_R(j_comb)
                if dR < jet_matching_distance * jet_R0:
                    j_true.matches.append(j_comb)

        for j_true in jets_truth_selected:

            j_type = match_dR(j_true, partons, jet_R0 / 2.)
            # if j_type[0] is None:
            #     if args.nw:
            #         continue
            #     pwarning('Jet with no parton label')
            #     continue

            if args.verbose:
                print_jet(j_true, prefix='-- true', constit=False)
            if len(j_true.matches) > 1:
                pwarning('More than 1 match for true -- take closest')
                dR_min = 100.
                for j_comb in j_true.matches:
                    dR = j_true.delta_R(j_comb)
                    if dR < dR_min:
                        j_comb_matched = j_comb
                        dR_min = dR
                    if args.verbose:
                        print_jet(j_comb, prefix='-- match candid', constit=False, add_info=[partial(print_match, j=j_true),])
                # shared_pt_frac_max = -1
                # for j_comb in j_true.matches:
                #     shared_pt_frac = fjtools.matched_pt(j_true)
                #     if shared_pt_frac > shared_pt_frac_max:
                #         j_comb_matched = j_comb
                #         shared_pt_frac_max = shared_pt_frac
                match_counter_true += 1
            elif len(j_true.matches) < 1:
                pwarning('No matches for true')
                j_comb_matched = None
                no_match_counter_true += 1
            else:
                if args.verbose:
                    print('Exactly 1 match for true')
                j_comb_matched = j_true.matches[0]
                match_counter_true += 1


            if args.verbose and j_comb_matched:
                print_jet(j_comb_matched, prefix='-- matched', constit=False, add_info=[partial(print_match, j=j_true),])


            list_of_groomers = [    ('soft_drop', [0, 0.001]),
                                    ('soft_drop', [0, 0.01]),
                                    ('soft_drop', [0, 0.03]),
                                     ('soft_drop', [0, 0.1]),
                                     ('soft_drop', [0, 0.2]),
                                     ('soft_drop', [0, 0.3]),

                                     ('soft_drop', [1, 0.1, jet_R0]),
                                     ('soft_drop', [1, 0.3, jet_R0]),
                                     ('soft_drop', [1, 1, jet_R0]),

                                     ('soft_drop', [2, 0.1, jet_R0]),
                                     ('soft_drop', [2, 0.3, jet_R0]),
                                     ('soft_drop', [2, 1, jet_R0]),
                                     ('soft_drop', [2, 5, jet_R0]),

                                     ('dynamical', [0.01]),
                                     ('dynamical', [0.1]),
                                     ('dynamical', [0.3]),
                                     ('dynamical', [0.5]),
                                     ('dynamical', [1.0]),
                                     ('dynamical', [2.0]),

                                     ('max_pt_softer', []),
                                     ('max_z', []),
                                     ('max_kt', []),
                                     ('max_kappa', []),
                                     ('max_tf', []),
                                     ('min_tf', [])]
            groomers_flags = dict()
            groomed_true_ls = dict()
            groomed_comb_ls = dict()
            gshop_true = fjcontrib.GroomerShop(j_true, jet_def_lund)
            if j_comb_matched:
                gshop_comb = fjcontrib.GroomerShop(j_comb_matched, jet_def_lund)
                matched_pt_res = {}
                for g_algo, g_params in list_of_groomers:
                    matched_pt_res_cur = {}
                    flag = flag_prong_matching(j_comb_matched, j_true, jet_def_lund, g_algo, g_params, matched_pt_res_cur)
                    name = groomer2str(g_algo, g_params)
                    matched_pt_res[name] = matched_pt_res_cur
                    groomers_flags[name] = flag
                    groomed_true_ls[name] = getattr(gshop_true, g_algo)(*g_params)
                    groomed_comb_ls[name] = getattr(gshop_comb, g_algo)(*g_params)
            else:
                for g_algo, g_params in list_of_groomers:
                    name = groomer2str(g_algo, g_params)
                    groomers_flags[name] = 10
                    groomed_true_ls[name] = getattr(gshop_true, g_algo)(*g_params)

            if args.verbose:
                print(groomers_flags)
                print('true:')
                print('\n'.join([k +': '+ ls2str(ls) for k,ls in groomed_true_ls.items()]))
                if j_comb_matched:
                    print('combined:')
                    print('\n'.join([k +': '+ ls2str(ls) for k,ls in groomed_comb_ls.items()]))
                # print(groomed_comb_ls)

                print('all splits, true:')
                for ls in lund_gen.result(j_true):
                    print(ls2str(ls))
                print('all splits, combined:')
                if j_comb_matched:
                    for ls in lund_gen.result(j_comb_matched):
                        print(ls2str(ls))

            constits_true = fj.sorted_by_pt(j_true.constituents())
            constits_comb = fj.sorted_by_pt(j_comb_matched.constituents()) if j_comb_matched else None
            tw_true.fill_branches(
                              evt_id = i_event + int(1e6) * exec_id,
                              j = j_true,
                              j_comb = j_comb_matched,
                              dR_comb = j_true.delta_R(j_comb_matched) if j_comb_matched else None,
                              flag = groomers_flags,

                              pt_matched = fjtools.matched_pt(j_comb_matched, j_true)  if j_comb_matched else None,
                              pt_matched2 = fjtools.matched_pt(j_true, j_comb_matched) if j_comb_matched else None,

                              n_matches = len(j_true.matches),
                              matches = j_true.matches,
                              dR_matches = [j_true.delta_R(j_match) for j_match in j_true.matches],

                              lund = [ls for ls in lund_gen.result(j_true)],
                              lund_comb = [ls for ls in lund_gen.result(j_comb_matched)] if j_comb_matched else None,
                              groomed = groomed_true_ls,
                              groomed_comb = groomed_comb_ls if j_comb_matched else None,

                              matched_pt = matched_pt_res if j_comb_matched else None,

                              const = constits_true,
                              const_pT = [pythia.event[c.user_index()].pT() if c.user_index() > 0 else 0 for c in constits_true],
                              const_pid = [pythia.event[c.user_index()].id() if c.user_index() > 0 else 0 for c in constits_true],
                              hasBeauty = any([is_hb(pythia.event[c.user_index()].id()) for c in constits_true if c.user_index() > 0]),
                              hasCharm = any([is_hc(pythia.event[c.user_index()].id()) for c in constits_true if c.user_index() > 0]),

                              const_comb = constits_comb,
                              const_comb_pT = [pythia.event[c.user_index()].pT() if c.user_index() > 0 else 0 for c in constits_comb] if constits_comb else None,
                              const_comb_pid = [pythia.event[c.user_index()].id() if c.user_index() > 0 else 0 for c in constits_comb] if constits_comb else None,
                              hasBeauty_comb = any([is_hb(pythia.event[c.user_index()].id()) for c in constits_comb if c.user_index() > 0]) if constits_comb else None,
                              hasCharm_comb = any([is_hc(pythia.event[c.user_index()].id()) for c in constits_comb if c.user_index() > 0]) if constits_comb else None,


                              bckg2 = bckg_gen.last_event_prop if hasattr(bckg_gen, 'last_event_prop') else None,
                              bckg = bckg_prop,

                                ppid           = j_type[0] if j_type[0] else 99,
                                pquark         = j_type[1] if j_type[1] else 99,
                                pglue          = j_type[2] if j_type[2] else 99,

                                pycode         = pythia.info.code(),
                                pysigmagen  = pythia.info.sigmaGen(),
                                pysigmaerr  = pythia.info.sigmaErr(),
                                pyid1       = pythia.info.id1pdf(),
                                pyid2       = pythia.info.id1pdf(),
                                pyx1         = pythia.info.x1pdf(),
                                pyx2           = pythia.info.x2pdf(),
                                pypdf1      = pythia.info.pdf1(),
                                pyQfac         = pythia.info.QFac(),
                                pyalphaS     = pythia.info.alphaS(),

                                pypthat     = pythia.info.pTHat(),
                                pymhat         = pythia.info.mHat(),
                                )
            tw_true.fill_tree()
        # don't pollute further part by mistake
        # del j_true, j_comb_matched


    if not args.dont_write_comb_jets:
        # match pythia jets to the combined jets
        for j_comb in jets_combined_selected:
            if not hasattr(j_comb, 'matches'):
                j_comb.matches = []
            else:
                perror(f'matched jets array already there - it should not happen {j_comb.matches}')
            for j_true in jets_truth_selected:
                dR = j_comb.delta_R(j_true)
                if dR < jet_matching_distance * jet_R0:
                    j_comb.matches.append(j_true)

        for j_comb in jets_combined_selected:

            j_type = match_dR(j_comb, partons, jet_R0 / 2.)
            # if j_type[0] is None:
            #     if args.nw:
            #         continue
            #     pwarning('Jet with no parton label')
            #     continue

            if args.verbose:
                print_jet(j_comb, prefix='-- comb', constit=False)
            if len(j_comb.matches) > 1:
                pwarning('More than 1 match for comb -- take closest')
                dR_min = 100.
                for j_true in j_comb.matches:
                    dR = j_comb.delta_R(j_true)
                    if dR < dR_min:
                        j_true_matched = j_true
                        dR_min = dR
                    if args.verbose:
                        print_jet(j_true, prefix='-- match candid', constit=False, add_info=[partial(print_match, j=j_comb),])
                # shared_pt_frac_max = -1
                # for j_comb in j_true.matches:
                #     shared_pt_frac = fjtools.matched_pt(j_true)
                #     if shared_pt_frac > shared_pt_frac_max:
                #         j_comb_matched = j_comb
                #         shared_pt_frac_max = shared_pt_frac
                match_counter_comb += 1
            elif len(j_comb.matches) < 1:
                if args.verbose: print('No matches for comb')
                j_true_matched = None
                no_match_counter_comb += 1
            else:
                if args.verbose:
                    print('Exactly 1 match for comb')
                j_true_matched = j_comb.matches[0]
                match_counter_comb += 1


            if args.verbose and j_true_matched:
                print_jet(j_true_matched, prefix='-- matched', constit=False, add_info=[partial(print_match, j=j_comb),])


            list_of_groomers = [    ('soft_drop', [0, 0.001]),
                                    ('soft_drop', [0, 0.01]),
                                    ('soft_drop', [0, 0.03]),
                                     ('soft_drop', [0, 0.1]),
                                     ('soft_drop', [0, 0.2]),
                                     ('soft_drop', [0, 0.3]),

                                     ('soft_drop', [1, 0.1, jet_R0]),
                                     ('soft_drop', [1, 0.3, jet_R0]),
                                     ('soft_drop', [1, 1, jet_R0]),

                                     ('soft_drop', [2, 0.1, jet_R0]),
                                     ('soft_drop', [2, 0.3, jet_R0]),
                                     ('soft_drop', [2, 1, jet_R0]),
                                     ('soft_drop', [2, 5, jet_R0]),

                                     ('dynamical', [0.01]),
                                     ('dynamical', [0.1]),
                                     ('dynamical', [0.3]),
                                     ('dynamical', [0.5]),
                                     ('dynamical', [1.0]),
                                     ('dynamical', [2.0]),

                                     ('max_pt_softer', []),
                                     ('max_z', []),
                                     ('max_kt', []),
                                     ('max_kappa', []),
                                     ('max_tf', []),
                                     ('min_tf', [])]
            groomers_flags = dict()
            groomed_true_ls = dict()
            groomed_comb_ls = dict()
            gshop_comb = fjcontrib.GroomerShop(j_comb, jet_def_lund)
            if j_true_matched:
                gshop_true = fjcontrib.GroomerShop(j_true_matched, jet_def_lund)
                matched_pt_res = {}
                for g_algo, g_params in list_of_groomers:
                    matched_pt_res_cur = {}
                    flag = flag_prong_matching(j_true_matched, j_comb, jet_def_lund, g_algo, g_params, matched_pt_res_cur)
                    name = groomer2str(g_algo, g_params)
                    matched_pt_res[name] = matched_pt_res_cur
                    groomers_flags[name] = flag
                    groomed_comb_ls[name] = getattr(gshop_comb, g_algo)(*g_params)
                    groomed_true_ls[name] = getattr(gshop_true, g_algo)(*g_params)
            else:
                for g_algo, g_params in list_of_groomers:
                    name = groomer2str(g_algo, g_params)
                    groomers_flags[name] = 10
                    groomed_comb_ls[name] = getattr(gshop_comb, g_algo)(*g_params)

            if args.verbose:
                print(groomers_flags)
                print('combined:')
                print('\n'.join([k +': '+ ls2str(ls) for k,ls in groomed_comb_ls.items()]))
                if j_true_matched:
                    print('true:')
                    print('\n'.join([k +': '+ ls2str(ls) for k,ls in groomed_true_ls.items()]))
                # print(groomed_comb_ls)

                print('all splits, comb:')
                for ls in lund_gen.result(j_comb):
                    print(ls2str(ls))
                print('all splits, true:')
                if j_true_matched:
                    for ls in lund_gen.result(j_true_matched):
                        print(ls2str(ls))

            constits_comb = fj.sorted_by_pt(j_comb.constituents())
            constits_true = fj.sorted_by_pt(j_true_matched.constituents()) if j_true_matched else None
            tw_comb.fill_branches(
                              evt_id = i_event + int(1e6) * exec_id,
                              j = j_comb,
                              j_true = j_true_matched,
                              dR_true = j_comb.delta_R(j_true_matched) if j_true_matched else None,
                              flag = groomers_flags,

                              pt_matched = fjtools.matched_pt(j_true_matched, j_comb)  if j_true_matched else None,
                              pt_matched2 = fjtools.matched_pt(j_comb, j_true_matched) if j_true_matched else None,

                              n_matches = len(j_comb.matches),
                              matches = j_comb.matches,
                              dR_matches = [j_comb.delta_R(j_match) for j_match in j_comb.matches],

                              lund = [ls for ls in lund_gen.result(j_comb)],
                              lund_true = [ls for ls in lund_gen.result(j_true_matched)] if j_true_matched else None,
                              groomed = groomed_comb_ls,
                              groomed_true = groomed_true_ls if j_true_matched else None,

                              matched_pt = matched_pt_res if j_true_matched else None,


                              const = constits_comb,
                              const_pT = [pythia.event[c.user_index()].pT() if c.user_index() > 0 else 0 for c in constits_comb],
                              const_pid = [pythia.event[c.user_index()].id() if c.user_index() > 0 else 0 for c in constits_comb],
                              hasBeauty = any([is_hb(pythia.event[c.user_index()].id()) for c in constits_comb if c.user_index() > 0]),
                              hasCharm = any([is_hc(pythia.event[c.user_index()].id()) for c in constits_comb if c.user_index() > 0]),

                              const_true = constits_true,
                              const_true_pT = [pythia.event[c.user_index()].pT() if c.user_index() > 0 else 0 for c in constits_true] if constits_true else None,
                              const_true_pid = [pythia.event[c.user_index()].id() if c.user_index() > 0 else 0 for c in constits_true] if constits_true else None,
                              hasBeauty_true = any([is_hb(pythia.event[c.user_index()].id()) for c in constits_true if c.user_index() > 0]) if constits_true else None,
                              hasCharm_true = any([is_hc(pythia.event[c.user_index()].id()) for c in constits_true if c.user_index() > 0]) if constits_true else None,

                              bckg2 = bckg_gen.last_event_prop if hasattr(bckg_gen, 'last_event_prop') else None,
                              bckg = bckg_prop,

                                ppid           = j_type[0] if j_type[0] else 999,
                                pquark         = j_type[1] if j_type[1] else 999,
                                pglue          = j_type[2] if j_type[2] else 999,

                                pycode      = pythia.info.code(),
                                pysigmagen  = pythia.info.sigmaGen(),
                                pysigmaerr  = pythia.info.sigmaErr(),
                                pyid1       = pythia.info.id1pdf(),
                                pyid2       = pythia.info.id1pdf(),
                                pyx1         = pythia.info.x1pdf(),
                                pyx2           = pythia.info.x2pdf(),
                                pypdf1      = pythia.info.pdf1(),
                                pyQfac         = pythia.info.QFac(),
                                pyalphaS     = pythia.info.alphaS(),

                                pypthat     = pythia.info.pTHat(),
                                pymhat         = pythia.info.mHat(),
                                )
            tw_comb.fill_tree()
        # del j_comb, j_true_matched


print(f'#true jets w/  matches: {match_counter_true}\n#true jets w/o matches: {no_match_counter_true}')
print(f'#comb jets w/  matches: {match_counter_comb}\n#comb jets w/o matches: {no_match_counter_comb}')

pythia.stat()

if not args.dont_write_true_jets: tw_true.write_and_close()
if not args.dont_write_comb_jets: tw_comb.write_and_close()

# import pandas as pd
# df = pd.DataFrame(res_arr)
# df["ptsum_frac"] = df.eval("ptsum_true/(ptsum_true+ptsum_bckg)")
# # np.corrcoef(df.ptsum_frac.to_numpy(), df.pt.to_numpy())
# df["pt_aver"] = df.eval("pt/(n_true+n_bckg)")



# if __name__ == '__main__':
#     main()
