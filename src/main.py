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
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import igraph
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations

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
def get_coincidx_graph(dataorig, alpha, standardize, outdir):
    """Get graph of individual elements"""
    info(inspect.stack()[0][3] + '()')
    coincpath = pjoin(outdir, 'coinc.csv')
    impath = pjoin(outdir, 'coinc.png')

    if isfile(coincpath):
        return np.loadtxt(coincpath, delimiter=','), coincpath

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

    fig, ax = plt.subplots()
    im = ax.imshow(adj, cmap='hot', interpolation='nearest')
    fig.colorbar(im)
    plt.savefig(impath)
    np.savetxt(coincpath, adj, delimiter=',')
    return adj, coincpath

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
    info(inspect.stack()[0][3] + '()')
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
def plot_motifs(gorig, coincpath, outdir):
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'coinc.pdf')
    coinc = np.loadtxt(coincpath, delimiter=',', dtype=float)
    coinc = np.power(coinc, 3)
    gcoinc = igraph.Graph.Weighted_Adjacency(coinc, mode='undirected')
    gcoinc.delete_edges(gcoinc.es.select(weight_lt=.3))

    # comm = gcoinc.community_multilevel(weights='weight')
    ncomms = 5
    comm = gcoinc.components(mode='weak')
    szs = comm.sizes()
    largestcommids = np.flip(np.argsort(szs))[:ncomms]
    membs = np.array(comm.membership)
    m = gcoinc.vcount()

    palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
    vcolours = np.array(['#d9d9d9'] * m)

    ##########################################################
    # Making the colormap
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list(
        'new',
        ['#d9d9d9'] + palette[:ncomms] ,
        N = 8)

    fig, ax = plt.subplots()
    im = ax.imshow(np.random.random((3,3)), interpolation ='nearest',
                    origin ='lower', cmap = cm)
    # ax.set_title("bin: % s" % all_bin)
    fig.colorbar(im, ax=ax)
    plt.savefig(pjoin(outdir, 'colorbar.pdf'))

    ##########################################################
    for i in range(ncomms):
        vcolours[np.where(membs == largestcommids[i])[0]] = palette[i]

    outpath = pjoin(outdir, 'coinc.pdf')
    igraph.plot(gcoinc, outpath, edge_width=np.abs(gcoinc.es['weight']),
                bbox=(1200, 1200),
                vertex_size=20, vertex_color=vcolours.tolist())

    ##########################################################
    g = gorig.copy()
    # eids = g.get_vids(edges)
    # todel = set(range(g.ecount())).difference(eids)

    n = g.vcount()
    vcolours = np.array(['#d9d9d9'] * n)

    g.vs['comm'] = -1
    for i in range(ncomms):
        aux = np.where(membs == largestcommids[i])[0]
        g.vs[aux]['comm'] = i
        vcolours[aux] = palette[i]
        
    # aux = np.array(edges)[np.where(membs == largestcommids[0])[0]]
    # g.vs['comm'] = 0
    # aux = np.array(edges)[np.where(membs == largestcommids[0])[0]]
    # g.es[g.get_eids(aux)]['comm'] = 0

    # g.delete_edges(todel)
    # degs = g.degree()
    # g.delete_vertices(np.where(np.array(degs) == 0)[0])

    # membs = np.array(g.es['comm'])
    # m = g.ecount()
    # n = g.vcount()
    # vcolours = np.array(['#d9d9d9'] * n)
    # breakpoint()
    

    # fh = open('/tmp/titles.lst', 'w')
    # membslbls = []
    # for i in range(ncomms):
        # vcolours[np.where(membs == i)[0]] = palette[i]
        # vv = np.unique(np.array([ [x.source, x.target] for x in np.array(g.es)[np.where(membs == i)[0]] ]).flatten())
        # vv = np.unique(np.array([ [x.source, x.target] for x in np.array(g.es)[np.where(membs == i)[0]] ]).flatten())
        # membslbls.append(np.array(g.vs['title'])[vv])
        # fh.write(str(i) + '\n')
        # fh.write('\n'.join(membslbls[i]))
        # fh.write('\n\n')
    # fh.close()

    outpath = pjoin(outdir, 'orig.png')
    igraph.plot(g, outpath, bbox=(1200, 1200),
                # vertex_label=g.vs['title'],
                # vertex_color='black',
                vertex_color=vcolours,
                vertex_size=20,
                )
                # edge_color=ecolours.tolist())
                # edge_color=ecolours.tolist())

    outpath = pjoin(outdir, 'orig.pdf')
    igraph.plot(g, outpath, bbox=(1200, 1200),
                # vertex_label=g.vs['title'],
                # vertex_color='black',
                vertex_color=vcolours,
                vertex_size=20,
                )
                # edge_color=ecolours.tolist())
                # edge_color=ecolours.tolist())

##########################################################
def generate_graph(modelstr, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    # g = igraph.Graph.Erdos_Renyi(n, p, directed=False, loops=False)
    # breakpoint()
    # n = 101
    p = 0.05
    h = 2
    g = igraph.Graph.GRG(200, radius=0.2, torus=False)
    g.to_undirected()

    g = g.connected_components().giant()
    adj = g.get_adjacency_sparse()
    return g, adj

##########################################################
def extract_features(adj):
    vfeats, labels = extract_hierarchical_feats_all(adj,  h)
    return np.array(vfeats), labels

##########################################################
def run_experiment(modelstr, h, runid, outdir):
    random.seed(runid); np.random.seed(runid) # Random seed
    # info(inspect.stack()[0][3] + '()')

    generate_graph(modelstr, outdir)
    vfeats, lbls = extract_features(adj)
    coinc, coincpath = get_coincidx_graph(vfeats, .5, True, outdir)
    plot_motifs(g, coincpath, outdir)

    # Extract features
    # vfeats, labels = extract_hierarchical_feats_all(adj,  h)
    # vfeats = np.array(vfeats)
    # for i, l in enumerate(labels):
        # g.vs[l] = vfeats[:, i]

    # g = vattributes2edges(g, labels, aggreg='sum')
    # efeats = np.array([g.es[l] for l in labels]).T

    # Determine components
    # Calculate statistics in each group
    # Plot distributions for each

##########################################################
def main(nprocs, outdir):
    info(inspect.stack()[0][3] + '()')

    nruns = 1
    runids = range(nruns)
    hs = [2]
    outdirs = [outdir]

    n = 100
    k = 6
    modelstr = [
            'er,N,K,0',
            'ba,N,K,0',
            'gr,N,K,0',
            ]
    modelstr = [m.replace('N', str(n)).replace('K', str(k)) for m in modelstr]


    from itertools import product
    argsconcat = [x for x in product(modelstr, hs, runids, outdirs)]
    # argsconcat = reversed(argsconcat)
    parallelize(run_experiment, nprocs, argsconcat)

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
    main(args.nprocs, args.outdir)
    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
