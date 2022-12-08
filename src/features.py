#!/usr/bin/env python3
"""Extract features from undirected graph
"""

import argparse
import time, datetime
import os
from os.path import join as pjoin
from os.path import isfile
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import igraph
import pandas as pd

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
def graph_from_dfs(dfes, dfvs):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    g = igraph.Graph(n=len(dfvs))
    g.vs['wid'] = dfvs.wid.values
    g.vs['title'] = dfvs.title.values
    g.add_edges(dfes[['srcvid', 'tgtvid']].values)
    g.simplify(multiple=True, loops=True)
    return g

##########################################################
def get_induced_subgraph(snppath, widspath, outdir):
    """Get the subgraph induced by the vertices with @wids wiki ids"""
    info(inspect.stack()[0][3] + '()')

    snpfiltpath = pjoin(outdir, 'snap0_es.tsv')
    widsfiltpath = pjoin(outdir, 'snap0_vs.tsv')

    if isfile(snpfiltpath):
        dfsnpfilt = pd.read_csv(snpfiltpath, sep='\t')
        dfwidsfilt = pd.read_csv(widsfiltpath, sep='\t')
        return dfsnpfilt, dfwidsfilt

    dfsnp = pd.read_csv(snppath, sep='\t')
    dfwids = pd.read_csv(widspath, sep='\t')
    dfwids = dfwids.loc[dfwids.ns == 0] # Filter pages from the main ns

    # Filter snapshot by the reference vertices
    dfsnpfilt = dfsnp.merge(dfwids, how='inner', left_on='src', right_on='id')
    dfsnpfilt = dfsnpfilt.merge(dfwids, how='inner', left_on='tgt', right_on='id')

    # Get titles of the vertices
    uwids = sorted(np.unique(dfsnpfilt[['src', 'tgt']].values.flatten()))
    dfwidsfilt = pd.DataFrame(uwids, columns=['wid']).merge(
        dfwids, how='inner', left_on='wid', right_on='id')
    dfwidsfilt = dfwidsfilt.drop_duplicates(keep='first')
    dfwidsfilt.to_csv(widsfiltpath, sep='\t', index=False,
                    columns=['wid', 'title'],)

    n = len(uwids)
    wid2vid = {wid:vid for vid, wid in enumerate(uwids)}

    # Convert wiki-id to vertex id
    dfsnpfilt = dfsnpfilt[['src', 'tgt']].copy()
    dfsnpfilt['srcvid'] = dfsnpfilt.src.map(wid2vid)
    dfsnpfilt['tgtvid'] = dfsnpfilt.tgt.map(wid2vid)
    dfsnpfilt.to_csv(snpfiltpath, sep='\t', index=False,
                    columns=['src', 'tgt', 'srcvid', 'tgtvid'])
    return dfsnpfilt, dfwidsfilt

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
    labels = 'hn he hd hc cr'.split(' ')
    feats = []
    for v in range(adj.shape[0]):
        feats.append(extract_hirarchical_feats(adj, v, h))
    return feats, labels

##########################################################
def main(outdir):
    info(inspect.stack()[0][3] + '()')

    n = 20
    p = 0.1
    h = 2

    g = igraph.Graph.Erdos_Renyi(n, p, directed=False, loops=False)

    g = g.connected_components().giant()
    adj = g.get_adjacency_sparse()
    feats, labels = extract_hierarchical_feats_all(adj,  h)
    breakpoint()

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
