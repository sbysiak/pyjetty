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

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext

from pyjetty.mputils import MPBase, pwarning, pinfo, perror, treewriter, jet_analysis

# see:
# https://github.com/matplo/pyjetty/blob/5a515bf7714d7c040ae90ebf5d4343dbee1a4874/pyjetty/alice_analysis/process/user/james/process_groomers.py#L824-L1033



def print_pyp(pyp, prefix='some psj '):
    # print(type(pyp))
    if type(pyp) == pythia8.Particle:
        # print(prefix+f" pt,eta,phi = {pyp.pT():.1f}, {pyp.eta():.3f}, {pyp.phi():.3f}, id={pyp.name() if pyp else ''}, index={pyp.index() if pyp else ''}, mothers={' <-'.join([pythia.event[idx].name() for idx in pyp.motherList()]) if pyp else ''} {'      <-----' if pyp and any([abs(pythia.event[idx].id()) in B_hadrons_MC_codes for idx in pyp.motherList()]) else ''}")
        print(prefix+f" pt,eta,phi = {pyp.pT():.1f}, {pyp.eta():.3f}, {pyp.phi():.3f}")
        print('^^ CASE 1', prefix)
    else:
        if type(pyp) == fj.PseudoJet and pythiafjext.getPythia8Particle(pyp):
            print_pyp(pythiafjext.getPythia8Particle(pyp), prefix)
            print('^^ CASE 2', prefix)
        else:
            print(prefix+f' pt,eta,phi = {pyp.pt():.1f}, {pyp.eta():.3f}, {pyp.phi():.3f}')
            # print('^^ CASE 3', prefix)


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


def match_dR(j, partons, drmatch = 0.1):
    mps = [p for p in partons if j.delta_R(p) < drmatch]
    # for p in fj.sorted_by_pt(mps)[0]:
    if len(mps) < 1:
        return None, False, False
    p = fj.sorted_by_pt(mps)[0]
    pyp = pythiafjext.getPythia8Particle(p)
    # print(p, pyp.id(), pyp.isQuark(), pyp.isGluon())
    return pyp.id(), pyp.isQuark(), pyp.isGluon()



# def main():
parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly', prog=os.path.basename(__file__))
pyconf.add_standard_pythia_args(parser)
parser.add_argument('--nw', help="no warn", default=False, action='store_true')
parser.add_argument('--ignore-mycfg', help="ignore some settings hardcoded here", default=False, action='store_true')
parser.add_argument('--enable-background', help="enable background calc", default=True, action='store_true')
parser.add_argument('--output', help="output file name", default='leadsj_vs_x_output.root', type=str)

# for background
parser.add_argument('--cent-bin', help="centraility bin 0 is the  0-5 percent most central bin", type=int, default=0)
parser.add_argument('--seed', help="pr gen seed", type=int, default=1111)
parser.add_argument('--harmonics', help="set harmonics flag (0 : v1 - v5) , (1 : v2 - v5) , (2: v3 - v5) , (3: v1 - v4) , (4: v1 - v3) , (5: uniform dN/dphi no harmonics) , (6 : v1 - v2 , v4 - v5) , (7 : v1 - v3 , v5) , (8 : v1 , v3 - v5) , (9 : v1 only) , (10 : v2 only) , (11 : v3 only) , (12 : v4 only) , (13 : v5 only)",
                    type=int, default=5)
parser.add_argument('--eta', help="set eta range must be uniform (e.g. abs(eta) < 0.9, which is ALICE TPC fiducial acceptance)",
                    type=float, default=0.9)
parser.add_argument('--qa', help="PrintOutQAHistos", default=False, action='store_true')

parser.add_argument('--dRmax', default=0.25, type=float)
parser.add_argument('--alpha', default=0, type=float)


args = parser.parse_args()


# print the banner first
fj.ClusterSequence.print_banner()
print()
# set up our jet definition and a jet selector
jet_R0 = 0.4
jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0)
jet_selector = fj.SelectorPtMin(20.0) & fj.SelectorPtMax(2000.0) & fj.SelectorAbsEtaMax(args.eta - jet_R0) # & fj.SelectorPtMin(args.py_pthatmin)
print(jet_def)

all_jets = []


mycfg = ['']
if args.ignore_mycfg:
    mycfg = []
pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)
if not pythia:
    perror("pythia initialization failed.")

jet_def_lund = fj.JetDefinition(fj.cambridge_algorithm, 1.0)
lund_gen = fjcontrib.LundGenerator(jet_def_lund)
print (lund_gen.description())
dy_groomer = fjcontrib.DynamicalGroomer(jet_def_lund)
print (dy_groomer.description())


jet_def_rc005 = fj.JetDefinition(fj.antikt_algorithm, 0.05)
jet_def_rc01 = fj.JetDefinition(fj.antikt_algorithm, 0.1)
jet_def_rc02 = fj.JetDefinition(fj.antikt_algorithm, 0.2)
jet_def_rc03 = fj.JetDefinition(fj.antikt_algorithm, 0.3)
print(jet_def_rc005)
print(jet_def_rc01)
print (jet_def_rc02)
print (jet_def_rc03)

tw = treewriter.RTreeWriter(name = 'jets', file_name = args.output)
tgbkg = None
thg = None
if args.enable_background:
    from pyjetty.alice_analysis.process.base import thermal_generator
    ### params from arxiv2006.01812
    thg = thermal_generator.ThermalGenerator(N_avg=int(1800*1.8),
                                             sigma_N=500, # not specified in ref
                                             beta=0.5,
                                             eta_max=args.eta)
    print(thg)

    from pyjetty.mputils import CEventSubtractor, CSubtractorJetByJet
    cs = CEventSubtractor(alpha=args.alpha,
                         max_distance=args.dRmax,
                         max_eta=args.eta,
                         bge_rho_grid_size=0.25, # or 1.0  as in https://github.com/matplo/pyjetty/blob/5a515bf7714d7c040ae90ebf5d4343dbee1a4874/pyjetty/alice_analysis/config/theta_g/PbPb/james_groomers_thermal.yaml
                         max_pt_correct=100)
    print(cs)



res_arr = []
no_match_counter = 0
match_counter = 0
for i in tqdm.tqdm(range(args.nev)):
    print('\n---\n')
    if not pythia.next():
        continue
    # parts = pythiafjext.vectorize(pythia, True, -1, 1, False)
    partons = pythiafjext.vectorize_select(pythia, [pythiafjext.kParton], 0, True)
    parts = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, False)
    # parts = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, False)
    jets = jet_selector(jet_def(parts))


    fj_particles_truth = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, False)
    if not len(jet_selector(fj.ClusterSequence(fj_particles_truth, jet_def).inclusive_jets())): continue
    if thg:
        fj_particles_combined_beforeCS = thg.load_event()
        print('BEFORE: ',len(fj_particles_combined_beforeCS))
        [fj_particles_combined_beforeCS.push_back(p) for p in fj_particles_truth]
        print('AFTER: ', len(fj_particles_combined_beforeCS), len([p for p in fj_particles_combined_beforeCS if p.user_index() < 0]))
    else:
        raise NotImplementedError
        fj_particles_combined = fj_particles_truth

    # Perform constituent subtraction for each R_max
    # fj_particles_combined = [self.constituent_subtractor[i].process_event(fj_particles_combined_beforeCS) for i, R_max in enumerate(self.max_distance)]
    fj_particles_combined = cs.process_event(fj_particles_combined_beforeCS)
    # fj_particles_combined = fj_particles_combined_beforeCS



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


    def print_aver_pt(j):
        return f'<pt>={np.mean([c.pt() for c in j.constituents()]):.2f}'

    def print_true_frac(j):
        n_true = len(([p.pt() for p in j.constituents() if p.user_index() > 0]))
        n_bckg = len(([p.pt() for p in j.constituents() if p.user_index() < 0]))
        ptsum_true = np.sum([p.pt() for p in j.constituents() if p.user_index() > 0])
        ptsum_bckg = np.sum([p.pt() for p in j.constituents() if p.user_index() < 0])
        return f'true pt frac = {ptsum_true/(ptsum_true+ptsum_bckg):.2f}, true n frac = {n_true/(n_true+n_bckg):.2f}'


    for j in jets_truth_selected:
        print_jet(j, prefix='true', constit=False, add_info=[print_aver_pt,])
    for j in jets_combined_selected:
        print_jet(j, prefix='combined', constit=False, add_info=[print_aver_pt, print_true_frac])

        res = dict(
                pt = j.pt(),
                n_true = len(([p.pt() for p in j.constituents() if p.user_index() > 0])),
                n_bckg = len(([p.pt() for p in j.constituents() if p.user_index() < 0])),
                ptsum_true = np.sum([p.pt() for p in j.constituents() if p.user_index() > 0]),
                ptsum_bckg = np.sum([p.pt() for p in j.constituents() if p.user_index() < 0]),
                )
        res_arr.append(res)

    print('combined jets only having true pt above threshold:')
    for j in jets_combined_noCS_selected:
        ptsum_true = np.sum([p.pt() for p in j.constituents() if p.user_index() > 0])
        if ptsum_true < 5: continue
        print_jet(j, prefix='combined_noCS', constit=False, add_info=[print_aver_pt, print_true_frac])

    for j in jets_bckg_selected:
        print_jet(j, prefix='bckg', constit=False, add_info=[print_aver_pt,])


    # match combined jets to the pythia jets
    matching_R_multiplier = 1.5
    for j_true in jets_truth_selected:
        if not hasattr(j_true, 'matches'):
            j_true.matches = []
        for j_comb in jets_combined_selected:
            dR = j_true.delta_R(j_comb)
            if dR < matching_R_multiplier * jet_R0:
                j_true.matches.append(j_comb)




    def flag_prong_matching(j_comb, j_true, grooming_algo='', grooming_params=[]):
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
        gshop_comb = fjcontrib.GroomerShop(j_comb)
        gshop_true = fjcontrib.GroomerShop(j_true)

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

    for j_true in jets_truth_selected:

        j_type = match_dR(j_true, partons, jet_R0 / 2.)
        if j_type[0] is None:
            if args.nw:
                continue
            pwarning('Jet with no parton label')
            continue


        from functools import partial
        def print_match(j_match,j_true):
            return f'dR = {j_true.delta_R(j_match):.2f}, shared_pt_frac = {fjtools.matched_pt(j_match, j_true):.2f} ,  {fjtools.matched_pt(j_true, j_match):.2f}'

        print_jet(j_true, prefix='-- true', constit=False)
        if len(j_true.matches) > 1:
            pwarning('More then 1 match -- take closest')
            dR_min = 100.
            for j_comb in j_true.matches:
                dR = j_true.delta_R(j_comb)
                if dR < dR_min:
                    j_comb_matched = j_comb
                    dR_min = dR
                print_jet(j_comb, prefix='-- match candid', constit=False, add_info=[partial(print_match, j_true=j_true),])
            # shared_pt_frac_max = -1
            # for j_comb in j_true.matches:
            #     shared_pt_frac = fjtools.matched_pt(j_true)
            #     if shared_pt_frac > shared_pt_frac_max:
            #         j_comb_matched = j_comb
            #         shared_pt_frac_max = shared_pt_frac
        elif len(j_true.matches) < 1:
            pwarning('No matches')
            j_comb_matched = None
        else:
            pwarning('Exactly 1 match')
            j_comb_matched = j_true.matches[0]

        if not j_comb_matched:
            pwarning('Skip jet due to invalid match')
            no_match_counter += 1
            continue
        else:
            match_counter += 1

        print_jet(j_comb_matched, prefix='-- matched', constit=False, add_info=[partial(print_match, j_true=j_true),])


        def groomer2str(g_algo, g_params):
            return g_algo + ("" if not g_params else "_".join([str(p).replace(".", "") for p in ["",]+ g_params]))


        def ls2str(ls):
            # print(dir(ls))
            return f'z={ls.z():.3f} Delta={ls.Delta():.3f} kt={ls.kt():.2f} Erad={ls.kt()/(ls.z()*ls.Delta()):.2f}'

        list_of_groomers = [    ('soft_drop', [0, 0.001]),
                                ('soft_drop', [0, 0.01]),
                                 ('soft_drop', [0, 0.1]),
                                 ('soft_drop', [0, 0.2]),
                                 ('soft_drop', [0, 0.3]),
                                 ('dynamical', [0.1]),
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
        gshop_true = fjcontrib.GroomerShop(j_true)
        gshop_comb = fjcontrib.GroomerShop(j_comb_matched)
        for g_algo, g_params in list_of_groomers:
            flag = flag_prong_matching(j_comb_matched, j_true, g_algo, g_params)
            name = groomer2str(g_algo, g_params)
            groomers_flags[name] = flag
            groomed_true_ls[name] = getattr(gshop_true, g_algo)(*g_params)
            groomed_comb_ls[name] = getattr(gshop_comb, g_algo)(*g_params)
        print(groomers_flags)
        print('true:')
        print('\n'.join([k +': '+ ls2str(ls) for k,ls in groomed_true_ls.items()]))
        print('combined:')
        print('\n'.join([k +': '+ ls2str(ls) for k,ls in groomed_comb_ls.items()]))
        # print(groomed_comb_ls)

        print('all splits, true:')
        for ls in lund_gen.result(j_true):
            print(ls2str(ls))
        print('all splits, combined:')
        for ls in lund_gen.result(j_comb_matched):
            print(ls2str(ls))

        tw.fill_branches( j = j_true,
                          j_comb = j_comb_matched,
                          dR_comb = j_true.delta_R(j_comb_matched),
                          flag = groomers_flags,

                          pt_matched = fjtools.matched_pt(j_comb_matched, j_true),
                          pt_matched2 = fjtools.matched_pt(j_true, j_comb_matched),

                          n_matches = len(j_true.matches),
                          matches = j_true.matches,
                          dR_matches = [j_true.delta_R(j_match) for j_match in j_true.matches],

                          lund = [ls for ls in lund_gen.result(j_true)],
                          lund_comb = [ls for ls in lund_gen.result(j_comb_matched)],
                          groomed = groomed_true_ls,
                          groomed_comb = groomed_comb_ls,

                          #const = j_true.constituents(),
                          #const_comb = j_comb_matched.constituents(),

                            ppid           = j_type[0],
                            pquark         = j_type[1],
                            pglue          = j_type[2],

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
        tw.fill_tree()


print(f'#jets w/  matches: {match_counter}\n #jets w/o matches: {no_match_counter}')

pythia.stat()

tw.write_and_close()

import pandas as pd
df = pd.DataFrame(res_arr)
df["ptsum_frac"] = df.eval("ptsum_true/(ptsum_true+ptsum_bckg)")
# np.corrcoef(df.ptsum_frac.to_numpy(), df.pt.to_numpy())
df["pt_aver"] = df.eval("pt/(n_true+n_bckg)")



# if __name__ == '__main__':
#     main()
