#!/usr/bin/env python3
"""Extract features from undirected graph
"""

import argparse
import time, datetime
import os, sys, random
from os.path import join as pjoin
from os.path import isfile
import inspect

import numpy as np
import scipy; import scipy.optimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import igraph
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import product, combinations
from myutils import parallelize

CID = 'compid'
##########################################################
def interiority(dataorig):
    """Calculate the interiority index of the two rows. @vs has 2rows and n-columns, where
    n is the number of features"""
    # info(inspect.stack()[0][3] + '()')
    data = np.abs(dataorig)
    abssum = np.sum(data, axis=1)
    den = np.min(abssum)
    num = np.sum(np.min(data, axis=0))
    return num / den

##########################################################
def jaccard(dataorig, a):
    """Calculate the interiority index of the two rows. @vs has 2rows and n-columns, where
    n is the number of features"""
    data = np.abs(dataorig)
    den = np.sum(np.max(data, axis=0))
    datasign = np.sign(dataorig)
    plus_ = np.abs(datasign[0, :] + datasign[1, :])
    minus_ = np.abs(datasign[0, :] - datasign[1, :])
    splus = np.sum(plus_ * np.min(data, axis=0))
    sminus = np.sum(minus_ * np.min(data, axis=0))
    num = a * splus - (1 - a) * sminus
    return num / den

##########################################################
def coincidence(data, a):
    inter = interiority(data)
    jac = jaccard(data, a)
    return inter * jac

##########################################################
def get_coincidx_values(dataorig, alpha, standardize):
    """Get coincidence value between each combination in @dataorig"""
    # info(inspect.stack()[0][3] + '()')
    n, m = dataorig.shape
    if standardize:
        data = StandardScaler().fit_transform(dataorig)
    else:
        data = dataorig

    adj = np.zeros((n, n), dtype=float)
    for comb in list(combinations(range(n), 2)):
        data2 = data[list(comb)]
        c = coincidence(data2, alpha)
        adj[comb[0], comb[1]] = adj[comb[1], comb[0]] = c

    return adj

##########################################################
def get_reachable_vertices_exact(adj, vs0, h):
    """Get the vertices reachable in *exactly* h steps. This
    implies that, for instance, self may be included in the result."""
    if h == 0: return vs0, adj

    adjh = adj
    for i in range(h-1):
        adjh = np.dot(adjh, adj)

    rows, cols = adjh.nonzero()
    reachable = []
    for v in vs0:
        z = cols[np.where(rows == v)]
        reachable.extend(z)

    return reachable, adjh

##########################################################
def get_neighbourhood(adj, vs0, h, itself=False):
    """Get the entire neighbourhood centered on vs0, including self"""
    if h == 0: return vs0 if itself else []
    neighsprev, _ = get_reachable_vertices_exact(adj, vs0, h - 1)
    neighs, _ =  get_reachable_vertices_exact(adj, vs0, h)
    diff = set(neighsprev).union(set(neighs))
    if itself: return diff
    else: return diff.difference(set(vs0))

##########################################################
def get_ring(adj, vs0, h):
    """Get the hth rings"""
    if h == 0: return []
    neigh1 = get_neighbourhood(adj, vs0, h-1)
    neigh2 = get_neighbourhood(adj, vs0, h)
    return list(set(neigh2).difference(set(neigh1)))

##########################################################
def calculate_hiennodes(neighvs): # OK
    return len(neighvs)

##########################################################
def calculate_hienedges(adj, ringcur):
    return adj[ringcur, :][:, ringcur].sum() / 2

##########################################################
def calculate_hierdegree(adj, ringcur, ringnxt):
    return adj[ringcur, :][:, ringnxt].sum()

##########################################################
def calculate_hierclucoeff(he, hn):
    if hn == 1: return 0
    return 2 * (he / (hn * (hn - 1)))

##########################################################
def calculate_hieconvratio(hd, hnnxt):
    #TODO: define what to do when hn = 0
    if hnnxt == 0: return 0
    return hd / hnnxt

##########################################################
def extract_hirarchical_feats(adj, v, h):
    """Extract hierarchical features"""
    ringcur = get_ring(adj, [v], h)
    ringnxt = get_ring(adj, [v], h+1)
    hn = calculate_hiennodes(ringcur)
    he = calculate_hienedges(adj, ringcur)
    hd = calculate_hierdegree(adj, ringcur, ringnxt)
    hc = calculate_hierclucoeff(he, hn)
    cr = calculate_hieconvratio(hd, calculate_hiennodes(ringnxt))
    return [hn, he, hd, hc, cr]

##########################################################
def extract_hierarchical_feats_all(adj,  h):
    # info(inspect.stack()[0][3] + '()')
    labels = 'hn he hd hc cr'.split(' ')
    feats = []
    for v in range(adj.shape[0]):
        feats.append(extract_hirarchical_feats(adj, v, h))
    return feats, labels

##########################################################
def vattributes2edges(g, attribs, aggreg='sum'):
    info(inspect.stack()[0][3] + '()')
    m = g.ecount()
    for attrib in attribs:
        values = g.vs[attrib]
        for j in range(m):
            src, tgt = g.es[j].source, g.es[j].target
            if aggreg == 'sum':
                g.es[j][attrib] = g.vs[src][attrib] + g.vs[tgt][attrib]
    return g

##########################################################
def generate_graph(modelstr, outdir):
    """Generate an undirected graph according to @modelstr. It should be MODEL,N,PARAM"""
    # info(inspect.stack()[0][3] + '()')
    parsed = modelstr.split(',')
    model = parsed[0];
    n = int(parsed[1])
    k = float(parsed[2])

    if model == 'er':
        erdosprob = k / n
        if erdosprob > 1: erdosprob = 1
        g = igraph.Graph.Erdos_Renyi(n, erdosprob)
    elif model == 'ba':
        m = round(k / 2)
        if m == 0: m = 1
        g = igraph.Graph.Barabasi(n, m)
    elif model == 'gr':
        r = get_rgg_params(n, k)
        g = igraph.Graph.GRG(n, radius=r, torus=False)

    g.to_undirected()
    g = g.connected_components().giant()
    g.simplify()

    gpath = pjoin(outdir, 'graph.png')
    coords = g.layout(layout='fr')
    return g, g.get_adjacency_sparse()

##########################################################
def extract_features(adj, h):
    vfeats, labels = extract_hierarchical_feats_all(adj,  h)
    return np.array(vfeats), labels

#############################################################
def get_rgg_params(nvertices, avgdegree):
    rggcatalog = {
        '20000,6': 0.056865545,
    }

    if '{},{}'.format(nvertices, avgdegree) in rggcatalog.keys():
        return rggcatalog['{},{}'.format(nvertices, avgdegree)]

    def f(r):
        g = igraph.Graph.GRG(nvertices, r)
        return np.mean(g.degree()) - avgdegree

    r = scipy.optimize.brentq(f, 0.0001, 10000)
    return r

##########################################################
def plot_graph(g, coordsin, labels, vsizes, outpath):
    coords = np.array(g.layout(layout='fr')) if coordsin is None else coordsin
    visual_style = {}
    visual_style["layout"] = coords
    visual_style["bbox"] = (960, 960)
    visual_style["margin"] = 10
    visual_style['vertex_label'] = labels
    visual_style['vertex_color'] = 'blue'
    visual_style['vertex_size'] = vsizes
    visual_style['vertex_frame_width'] = 0
    igraph.plot(g, outpath, **visual_style)
    return coords

##########################################################
def plot_graph_adj(adj, coords, labels, vsizes, outpath):
    g = igraph.Graph.Weighted_Adjacency(adj, mode='undirected', attr='weight',
                                        loops=False)
    coords = plot_graph(g, coords, labels, vsizes, outpath)
    return coords

##########################################################
def threshold_values(coinc, thresh, newval=0):
    """Values less than or equal to @thresh are set to zero"""
    coinc[coinc <= thresh] = newval
    return coinc

##########################################################
def label_communities(g, attrib, vszs, plotpath):
    vclust = g.components(mode='weak')
    ncomms = vclust.__len__()
    g.vs[CID] = vclust.membership

    membstr = [str(x) for x in g.vs[CID]]
    _ = plot_graph(g, None, None, vszs, plotpath)
    # info(vclust.summary())

    return g

##########################################################
def get_num_adjacent_groups(g, compid):
    membs = g.vs[CID]
    unvisited = set(np.where(np.array(membs) == compid)[0])
    n = len(unvisited)
    visited = set() # Visited but not explored
    explored = set()

    v = unvisited.pop(); visited.add(v)
    adjgrps = []
    group = []

    for i in range(1000000):
        if len(explored) == n: # All vertices from compid was explored
            adjgrps.append(group)
            return n, adjgrps
        elif len(visited) == 0: # No more vertices to explore in this group
            adjgrps.append(group)
            group = []
            v = unvisited.pop(); visited.add(v)
        else:
            v = visited.pop()
            neighs = set(g.neighborhood(v))
            neighs = neighs.intersection(unvisited)
            unvisited = unvisited.difference(neighs)
            visited = visited.union(neighs)
            explored.add(v)
            group.append(v)
    raise Exception('Something wrong')

##########################################################
def get_num_adjacent_groups_all(g):
    membership = g.vs[CID]
    nmembs = len(np.unique(membership))

    nadjgrps = np.zeros((nmembs, 2), dtype=int)
    for compid in range(nmembs):
        compsz, adjgrps = get_num_adjacent_groups(g, compid)
        nadjgrps[compid] = [compsz, len(adjgrps)]
    return nadjgrps

##########################################################
def get_feats_from_components(g, mincompsz):
    membs = np.array(g.vs[CID])
    comps, compszs = np.unique(membs, return_counts=True)
    ncomps = len(comps)
    feats = []

    aux = []
    for compid in comps:
        vs = g.vs.select(compid_eq=compid)
        if len(vs) <= mincompsz: continue
        sz = len(vs)
        degs = vs.degree()
        # aux.append([sz, np.mean(degs), np.std(degs)])
        aux.append([sz, np.mean(degs)])

    aux = np.array(aux)
    ws = aux[:, 0] / np.sum(aux[:, 0])
    # data = np.column_stack((aux, aux[:, 0] * aux[:, 1]))
    aux2 = aux[:, 1] * ws
    data = np.column_stack((aux, aux2))

    feats = [len(data)]

    means = data.mean(axis=0)
    stds = data.std(axis=0)
    feats.extend([means[0], stds[0], means[1], stds[1], means[2], stds[2]])
    return feats

##########################################################
def run_experiment(modelstr, h, runid, outdir):
    expidstr = '{}_{}'.format(modelstr, runid)
    info(expidstr)
    random.seed(runid); np.random.seed(runid) # Random seed

    t = 0.65
    # t = 0.4
    coincexp = 3
    mincompsz = 4

    op = {
        'graphorig': pjoin(outdir, '{}_0graphorig.png'.format(expidstr)),
        'graphcoinc': pjoin(outdir, '{}_1graphcoinc.png'.format(expidstr)),
    }

    g, adj = generate_graph(modelstr, outdir)
    n = g.vcount()
    vszs = np.array(g.degree()) + 1 # In case it is zero

    # vlbls = [str(i) for i in range(g.vcount())]
    vlbls = None

    coords1 = plot_graph(g, None, vlbls, vszs, op['graphorig']) # It is going to be overwriten
    vfeats, featlbls = extract_features(adj, h)
    coinc = get_coincidx_values(vfeats, .5, True)
    coinc = np.power(coinc, coincexp)

    coinc = threshold_values(coinc, t)

    coords2 = plot_graph_adj(coinc, None, vlbls, vszs, op['graphcoinc'])
    gcoinc = igraph.Graph.Weighted_Adjacency(coinc, mode='undirected')
    gcoinc = label_communities(gcoinc, CID, vszs, op['graphcoinc'])

    g.vs[CID] = gcoinc.vs[CID]
    membstr = [str(x) for x in g.vs[CID]]

    _ = plot_graph(g, coords1, vlbls, vszs, op['graphorig'])
    # feats = get_num_adjacent_groups_all(g)
    feats = [n]
    feats.extend(get_feats_from_components(gcoinc, mincompsz))
    return feats

##########################################################
def run_experiments_all(modelstr, hs, runids, nprocs, outdir):
    info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, 'res.csv')
    if os.path.isfile(outpath): return pd.read_csv(outpath)
    argsconcat = [x for x in product(modelstr, hs, runids, [outdir])]
    # argsconcat = reversed(argsconcat)

    featsall = parallelize(run_experiment, nprocs, argsconcat)

    params1 = np.array([x[0].split(',') for x in argsconcat], dtype=object)
    params2 = np.array([x[1] for x in argsconcat], dtype=object).reshape(-1, 1)
    featsall = np.array(featsall, dtype=object)
    featsall = np.column_stack((params1, params2, featsall))
    cols = ['model', 'nreq', 'k', 'x', 'runid', 'nreal', 'ncomps',
            'szmean', 'szstd', 'degmeanmean', 'degmeanstd', 'degwmeanmean', 'degwmeanstd']
    df = pd.DataFrame(featsall, columns=cols)
    df.to_csv(outpath, index=False)
    return df

##########################################################
def plot_results(df, outdir):
    info(inspect.stack()[0][3] + '()')
    plotdir = pjoin(outdir, 'plots')
    os.makedirs(plotdir, exist_ok=True)
    feats = ['ncomps', 'szmean', 'szstd', 'degmeanmean', 'degmeanstd',
             'degwmeanmean', 'degwmeanstd']
    models = ['er', 'gr', 'ba']
    for feat in feats:
        plotpath = pjoin(plotdir, feat + '.png')
        fig, ax = plt.subplots()
        for model in models:
            z = df.loc[df.model == model][feat]
            df.loc[df.model == model][feat].plot.kde(ax=ax, label=model, legend=True)
        ax.set_ylabel('')
        ax.set_xlabel(feat[0].upper() + feat[1:])
        plt.savefig(plotpath)
        plt.close()

##########################################################
def main(nruns, nprocs, outdir):
    info(inspect.stack()[0][3] + '()')

    runids = range(nruns)
    hs = [2]

    n = 400
    k = 6
    # n = 50
    # k = 6
    modelstr = [
            'er,N,K,0',
            'gr,N,K,0',
            'ba,N,K,0',
            ]
    modelstr = [m.replace('N', str(n)).replace('K', str(k)) for m in modelstr]

    df = run_experiments_all(modelstr, hs, runids, nprocs, outdir)
    plot_results(df, outdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nruns', default=1, type=int,
                        help='Number of runs for each experiment')
    parser.add_argument('--nprocs', default=1, type=int, help='Number of procs')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)
    main(args.nruns, args.nprocs, args.outdir)
    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
