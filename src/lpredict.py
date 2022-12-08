#!/usr/bin/env python3
"""Link prediction
"""

import argparse
import time, datetime
import os, sys, itertools, inspect
from os.path import join as pjoin
from os.path import isfile
import random

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme, append_to_file, parallelize
from scipy.ndimage import convolve, gaussian_filter
import igraph
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from myutils import transform
from itertools import combinations

##########################################################
def get_vertex_partitions_of_the_union(g1, g2):
    wids1 = set(np.array(g1.vs['wid']))
    wids2 = set(np.array(g2.vs['wid']))
    inter = wids1.intersection(wids2)
    left = wids1.difference(wids2)
    right = wids2.difference(wids1)
    return left, inter, right

##########################################################
def get_edge_wids(g):
    es = np.array(g.get_edgelist())
    vwids = g.vs['wid']

    ewids = []
    for edge in es:
        src, tgt = edge
        ewids.append([vwids[src], vwids[tgt]])
    return ewids

##########################################################
def get_link_partitions_of_the_union(g1, g2):
    info(inspect.stack()[0][3] + '()')
    wids1, wids2 = get_edge_wids(g1), get_edge_wids(g2)

    left, inter, right = [], [], []

    for ewid in wids1 + wids2:
        if ewid in wids1:
            if ewid in wids2 :
                if not (ewid in inter): # Avoid double insertions
                    inter.append(ewid)
            else:
                left.append(ewid)
        else:
            right.append(ewid)

    return left, inter, right

##########################################################
def vattributes2edges(g, attribs, aggreg='sum'):
    m = g.ecount()
    for attrib in attribs:
        values = g.vs[attrib]
        for j in range(m):
            src, tgt = g.es[j].source, g.es[j].target
            if aggreg == 'sum':
                g.es[j][attrib] = g.vs[src][attrib] + g.vs[tgt][attrib]
    return g

##########################################################
def feats_from_edgewids(edgewids, g, eattribs):
    wids = {wid: i for i, wid in enumerate(g.vs['wid'])}
    edgevids = []
    for srcwid, tgtwid in edgewids:
        edgevids.append([wids[srcwid], wids[tgtwid]])

    edgevids = np.array(edgevids)

    feats = []
    for attrib in eattribs:
        srcs, tgts = edgevids[:, 0], edgevids[:, 1]
        vals = (np.array(g.vs[srcs][attrib]) + np.array(g.vs[srcs][attrib])) / 2
        feats.append(vals)

    return np.array(feats).T

##########################################################
def normalize_with_margins(x, nbins):
    """The binwidth is defined based on nbins and the extra spacing for the
    smallest and largest value (half binwith on each side)."""
    margin = (np.max(x) - np.min(x)) / nbins / 2
    a = np.min(x) - margin
    b = np.max(x) + margin
    xnorm = (x - a) / (b - a)
    dx = (b - a) / nbins
    return xnorm, a, b, dx

##########################################################
def scale(x, a, b):
    return (x - a) / (b - a)

##########################################################
def descale(xnorm, a, b):
    return xnorm * (b -a) + a

##########################################################
def gaussian_2d(n, sigma):
  """Return N-Dimensional Gaussian Kernel
  @param integer  size  size of kernel / will be round to a nearest odd number
  @param float    sigma standard deviation of gaussian
  """
  s = int(n / 2)
  x, y = np.mgrid[-s:s+1, -s:s+1]
  k = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * (sigma ** 2)))
  return k / k.sum()

##########################################################
def get_sorted_sublists(setA, dtype=int):
    newset = np.ndarray((len(setA), len(setA[0])), dtype=dtype)
    for i, subset in enumerate(setA):
        newset[i, :] = sorted(subset)
    return newset

##########################################################
def load_graph(gpath):
    info(inspect.stack()[0][3] + '()')
    # Load graphs
    g = igraph.Graph.Read_GraphML(gpath)
    del g.vs['id']
    g = g.as_undirected()
    info('*Disregarding* directions')
    g.simplify()
    g.vs['wid'] = np.array(g.vs['wid']).astype(int).tolist() # Convert to int
    return g

##########################################################
def remove_disjoint_vertices(g1, g2):
    info(inspect.stack()[0][3] + '()')
    left, inter, right = get_vertex_partitions_of_the_union(g1, g2)
    g1.delete_vertices(np.where(np.isin(np.array(g1.vs['wid']), list(left)))[0])
    g2.delete_vertices(np.where(np.isin(np.array(g2.vs['wid']), list(right)))[0])
    return g1, g2

##########################################################
def get_complement_edges(vs, es):
    info(inspect.stack()[0][3] + '()')
    existed = get_sorted_sublists(es)
    combsall = np.array(list(combinations(vs, 2)))
    combsall = get_sorted_sublists(combsall)

    pair2str = lambda pair: '{:010d}_{:010d}'.format(pair[0], pair[1])
    str2pair = lambda txt: [int(x) for x in txt.split('_')]

    never = set()
    for c in combsall: never.add(pair2str(c))

    for c in existed:
        query = pair2str(c)
        if query in never:
            never.remove(query)

    return list(map(str2pair, never))

##########################################################
def plot_features_individually(feats, vattribs, outdir):
    info(inspect.stack()[0][3] + '()')
    # Plot features individually
    histdir = pjoin(outdir, 'hists')
    os.makedirs(histdir, exist_ok=True)
    for _set in feats.keys():
        for j, attrib in enumerate(vattribs):
            fig, ax = plt.subplots()
            vals = feats[_set][:, j]

            nbins = 10

            # In case we don't know the range
            dx = (np.max(vals) - np.min(vals)) / nbins / 2
            aa = np.min(vals) - dx
            bb = np.max(vals) + dx

            binwidth = (bb - aa) / (nbins - 1)
            xs = np.linspace(aa, bb, nbins)
            ys, _ = np.histogram(vals, bins=xs)
            ax.bar(xs[:-1], ys, width=binwidth)

            plt.savefig(pjoin(histdir, '{}_{}.png'.format(attrib, _set)))
            plt.close()

##########################################################
def split_train_test(feats):
    info(inspect.stack()[0][3] + '()')
    partitions = feats.keys()
    partinds = {}
    X_train = np.ndarray((0, 5), dtype=float)
    Y_train = []
    X_test = np.ndarray((0, 5), dtype=float)
    Y_test = []

    for i, p in enumerate(partitions):
        X = feats[p]
        shuffled = list(range(len(X))); np.random.shuffle(shuffled)
        splitid = int(.8 * len(X))
        traininds, testinds = shuffled[:splitid], shuffled[splitid:]
        partinds[p] = [traininds, testinds]
        X_train = np.vstack((X_train, feats[p][traininds]))
        Y_train.extend([i] * len(traininds))
        X_test = np.vstack((X_test, feats[p][testinds]))
        Y_test.extend([i] * len(testinds))
    Y_train = np.array(Y_train); Y_test = np.array(Y_test);
    return X_train, Y_train, X_test, Y_test

##########################################################
def ml_approach1(X_train, Y_train, X_test, Y_test):
    info(inspect.stack()[0][3] + '()')
    scaler = StandardScaler()
    lr = sklearn.svm.SVC()
    model = Pipeline([('standardize', scaler), ('log_reg', lr)])
    model.fit(X_train, Y_train)
    Y_test_hat = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_hat) * 100
    print(test_accuracy)

##########################################################
def ml_approach2(X_train, Y_train, X_test, Y_test):
    info(inspect.stack()[0][3] + '()')
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    Y_test_hat = gnb.fit(X_train, Y_train).predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_hat) * 100
    print(test_accuracy)

##########################################################
def plot_pca_from_feats(X_train, Y_train, labels, outdir):
    info(inspect.stack()[0][3] + '()')
    a, evecs, evals = transform.pca(X_train)
    fig, ax = plt.subplots()
    for i, subset in enumerate(labels):
        inds = np.where(Y_train == i)[0]
        ax.scatter(a[inds, 0], a[inds, 1], label=subset, alpha=0.5)
    fig.legend()
    plt.savefig(pjoin(outdir, 'feats_pca.png'))

##########################################################
def ml_approach_luc(X_train, Y_train, X_test, Y_test, partitions,
                    vattribs, infopath, outdir):
    info(inspect.stack()[0][3] + '()')
    # Luc's approach
    histnbins = 20
    eps = .0001
    convsigma = 2.0
    interpdir = pjoin(outdir, 'interp')
    os.makedirs(interpdir, exist_ok=True)

    probclass = []
    for i, p in enumerate(partitions):
        probclass.append(len(np.where(Y_test == i)[0]))
    probclass = np.array(probclass) / np.sum(probclass)

    for attribs in list(combinations(vattribs, 2)):
        suff = ','.join(attribs)
        probs = np.zeros((len(Y_test), len(partitions)), dtype=float)
        for i, subset in enumerate(partitions):
            inds = np.where(Y_train == i)[0]
            vals = X_train[inds, :]
            attrids = [vattribs.index(attr) for attr in attribs]
            vals = vals[:, attrids]
            vals0, vals1 = vals[:, 0], vals[:, 1]

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(vals[:, 0], vals[:, 1])
            plt.savefig(pjoin(interpdir, '{}_{}_scatter.png'.format(
                suff, subset)))
            plt.close()

            xnorm0, a0, b0, dx0 = normalize_with_margins(vals0, histnbins)
            xnorm1, a1, b1, dx1 = normalize_with_margins(vals1, histnbins)

            x0 = np.linspace(a0, b0, histnbins+1)
            x1 = np.linspace(a1, b1, histnbins+1)
            hist, _ = np.histogramdd(vals, (x0, x1))
            hist = hist.T # X is horizontal now

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(hist, origin='lower')
            plt.savefig(pjoin(interpdir, '{}_{}_hist.png'.format(
                suff, subset)))
            plt.close()

            s = [convsigma] * hist.ndim
            hist2 = gaussian_filter(hist, s, mode='constant')

            outpath = pjoin(interpdir, '{}_{}_hist_conv.png'.format(
                suff, subset))

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(hist2, origin='lower')
            plt.savefig(outpath)
            plt.close()

            dens = hist2 / np.sum(hist2)

            # Get the hist bin the values fall in
            bininds = X_test[:, attrids] - np.array([a0, a1])
            bininds = (bininds / np.array([dx0, dx1])).astype(int)

            # For points outside our ad-hoc range we assign 0
            issues = np.where((bininds >= len(dens)) | (bininds < 0))[0]
            for l in range(len(Y_test)):
                if l in issues: continue
                probs[l, i] = dens[bininds[l, 0], bininds[l, 1]]

        maxinds = np.argmax(probs, axis=1)
        conf = confusion_matrix(Y_test, maxinds)
        outpath = pjoin(interpdir, '{}_conf.png'.format(
            suff, subset))

        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf,
                              display_labels=partitions).plot(ax=ax)
        plt.savefig(outpath)
        plt.close()

        normalized = probs * probclass
        Y_test_hat = np.argmax(normalized, axis=1)
        test_accuracy = accuracy_score(Y_test, Y_test_hat)
        txt = '{} {}: {:.02f}'.format(attribs[0], attribs[1], test_accuracy)
        append_to_file(infopath, txt)

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
def calculate_shortest_paths(g1, deletedvids, keptvids, addedvids, outdir):
    info(inspect.stack()[0][3] + '()')
    infopath = pjoin(outdir, 'shtestpaths.txt')
    if isfile(infopath): return
    n = g1.vcount()
    nshuffles = 100
    spl = np.zeros((n ,n), dtype=int)
    for v in range(n):
        t0 = time.time()
        # If undirected, this can be optimized
        aux = g1.get_shortest_paths(v, to=None, weights=None, mode='out')
        spl[v, :] = [len(a) for a in aux]
    spl = spl - 1

    def avg_path_length_btw_edges(vids):
        ds = []
        for e1, e2 in combinations(vids, 2):
            ds.append(np.min([spl[e1[0], e2[0]], spl[e1[0], e2[1]],
                              spl[e1[1], e2[0]], spl[e1[1], e2[1]]]))
        return np.mean(ds)

    avg1 = avg_path_length_btw_edges(deletedvids)
    # never = get_complement_edges(list(range(g1.vcount())),
                         # deletedvids+keptvids+addedvids)

    avg2 = avg_path_length_btw_edges(keptvids)
    avg3 = avg_path_length_btw_edges(addedvids)
    avg4 = avg_path_length_btw_edges(deletedvids + keptvids + addedvids)

    szs = [len(deletedvids), len(keptvids), len(addedvids)]
    esall = deletedvids + keptvids + addedvids

    def func(esall):
        aux = esall.copy()
        np.random.shuffle(aux)
        row = []
        for sz in szs:
            row.append(avg_path_length_btw_edges(aux[:sz]))
            aux = aux[sz:]
        return row

    argsconcat = [[esall]] * nshuffles
    avgs = parallelize(func, 1, argsconcat)

    zmean = np.mean(avgs, axis=0)
    zstd = np.std(avgs, axis=0)

    txt = ''
    txt += '1-0:\t{:.02f}\tz:\t{:.02f}+-{:.02f}\n'. \
        format(avg1, zmean[0], zstd[0])
    txt += '1-1:\t{:.02f}\tz:\t{:.02f}+-{:.02f}\n'. \
        format(avg2, zmean[1], zstd[1])
    txt += '0-1:\t{:.02f}\tz:\t{:.02f}+-{:.02f}\n'. \
        format(avg3, zmean[2], zstd[2])
    txt += 'All\t{:.02f}\n'.format(avg4)
    append_to_file(infopath, txt)

##########################################################
def calculate_weighted_shortest_paths(adjcoinc, deletedvids, keptvids, addedvids, outdir):
    info(inspect.stack()[0][3] + '()')
    infopath = pjoin(outdir, 'coincshtestpaths.txt')
    # infopath = pjoin(outdir, 'del.txt')
    if isfile(infopath): return

    # adjcoinc = adjcoinc[:60, :60] # TODO: remove this
    # deletedvidsorig = deletedvids.copy()
    # deletedvids = []
    # for e in deletedvidsorig.copy():
        # if e[0] < 60 and e[1] < 60:
            # deletedvids.append(e)

    d = 3
    adj = np.power(adjcoinc, d)
    wthresh = .3

    g1 = igraph.Graph.Weighted_Adjacency(adj, mode='undirected')
    g1.delete_edges(g1.es.select(weight_lt=wthresh))
    nshuffles = 100

    weights = (1 - np.array(g1.es['weight'])).tolist()

    n = g1.vcount()
    spl = np.zeros((n, n), dtype=float)
    for v in range(n - 1):
        t0 = time.time()
        remain = list(range(v+1, n))
        aux = g1.get_shortest_paths(v, to=remain, weights=weights, mode='out')

        wlens = []
        for path in aux:
            wlen = 0
            for j in range(len(path) -1):
                wlen += adj[path[j], path[j+1]]
            wlens.append(wlen)

        spl[v, v+1:] = wlens
    spl += spl.T

    def avg_path_length_btw_edges(vids):
        ds = []
        for e1, e2 in combinations(vids, 2):
            ds.append(np.min([spl[e1[0], e2[0]], spl[e1[0], e2[1]],
                              spl[e1[1], e2[0]], spl[e1[1], e2[1]]]))
        return np.mean(ds)

    avg1 = avg_path_length_btw_edges(deletedvids)
    # never = get_complement_edges(list(range(g1.vcount())),
                         # deletedvids+keptvids+addedvids)

    avg2 = avg_path_length_btw_edges(keptvids)
    avg3 = avg_path_length_btw_edges(addedvids)
    avg4 = avg_path_length_btw_edges(deletedvids + keptvids + addedvids)

    szs = [len(deletedvids), len(keptvids), len(addedvids)]
    esall = deletedvids + keptvids + addedvids

    def func(esall):
        aux = esall.copy()
        np.random.shuffle(aux)
        row = []
        for sz in szs:
            row.append(avg_path_length_btw_edges(aux[:sz]))
            aux = aux[sz:]
        return row

    argsconcat = [[esall]] * nshuffles
    avgs = parallelize(func, 1, argsconcat)

    zmean = np.mean(avgs, axis=0)
    zstd = np.std(avgs, axis=0)

    txt = ''
    txt += '1-0:\t{:.02f}\tz:\t{:.02f}+-{:.02f}\n'. \
        format(avg1, zmean[0], zstd[0])
    txt += '1-1:\t{:.02f}\tz:\t{:.02f}+-{:.02f}\n'. \
        format(avg2, zmean[1], zstd[1])
    txt += '0-1:\t{:.02f}\tz:\t{:.02f}+-{:.02f}\n'. \
        format(avg3, zmean[2], zstd[2])
    txt += 'All\t{:.02f}\n'.format(avg4)
    append_to_file(infopath, txt)

##########################################################
def plot_motifs(gorig, edges, coincpath, outdir):
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'coinc.pdf')
    x = np.loadtxt(coincpath, delimiter=',')
    z = np.power(x, 3)
    g1 = igraph.Graph.Weighted_Adjacency(z, mode='undirected')
    g1.delete_edges(g1.es.select(weight_lt=.6))

    # comm = g1.community_multilevel(weights='weight')
    ncomms = 7
    comm = g1.components(mode='weak')
    szs = comm.sizes()
    largestcommids = np.flip(np.argsort(szs))[:ncomms]
    membs = np.array(comm.membership)
    m = g1.vcount()

    palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
    vcolours = np.array(['#d9d9d9'] * m)



    ##########################################################
    # Making the colormap
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list(
        'new',
        # palette[:ncomms] + ['#d9d9d9'],
        ['#d9d9d9'] + palette[:ncomms] ,
        N = 8)

    fig, ax = plt.subplots()
    im = ax.imshow(np.random.random((3,3)), interpolation ='nearest',
                    origin ='lower', cmap = cm)
    # ax.set_title("bin: % s" % all_bin)
    fig.colorbar(im, ax=ax)
    plt.savefig('/tmp/colorbar.pdf')

    ##########################################################
    for i in range(ncomms):
        vcolours[np.where(membs == largestcommids[i])[0]] = palette[i]

    outpath = pjoin(outdir, 'coinc.pdf')
    igraph.plot(g1, outpath, edge_width=np.abs(g1.es['weight']), bbox=(1200, 1200),
                vertex_size=5, vertex_color=vcolours.tolist())

    ##########################################################
    g = gorig.copy()
    eids = g.get_eids(edges)
    todel = set(range(g.ecount())).difference(eids)

    g.es['comm'] = -1
    for i in range(ncomms):
        aux = np.array(edges)[np.where(membs == largestcommids[i])[0]]
        g.es[g.get_eids(aux)]['comm'] = i
    aux = np.array(edges)[np.where(membs == largestcommids[0])[0]]
    g.es[g.get_eids(aux)]['comm'] = 0

    g.delete_edges(todel)
    degs = g.degree()
    g.delete_vertices(np.where(np.array(degs) == 0)[0])

    membs = np.array(g.es['comm'])
    m = g.ecount()
    ecolours = np.array(['#d9d9d9'] * m)

    fh = open('/tmp/titles.lst', 'w')
    membslbls = []
    for i in range(ncomms):
        ecolours[np.where(membs == i)[0]] = palette[i]
        vv = np.unique(np.array([ [x.source, x.target] for x in np.array(g.es)[np.where(membs == i)[0]] ]).flatten())
        membslbls.append(np.array(g.vs['title'])[vv])
        fh.write(str(i) + '\n')
        fh.write('\n'.join(membslbls[i]))
        fh.write('\n\n')
    fh.close()

    outpath = pjoin(outdir, 'orig.png')
    igraph.plot(g, outpath, bbox=(1200, 1200),
                # vertex_label=g.vs['title'],
                vertex_color='black',
                vertex_size=5,
                edge_color=ecolours.tolist())

    

    outpath = pjoin(outdir, 'orig.pdf')
    igraph.plot(g, outpath, bbox=(1200, 1200),
                vertex_label=g.vs['title'],
                vertex_color='black',
                vertex_size=5,
                edge_color=ecolours.tolist())

##########################################################
def plot_graph(gorig, deletedvids, keptvids, addedvids, outdir):
    info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, 'graphfull.pdf')
    if isfile(outpath): return
    g1 = gorig.copy()
    g1.add_edges(addedvids)
    m = g1.ecount()

    ecolours = np.ones((m, 4), dtype=float) * .5
    ecolours[g1.get_eids(deletedvids), :] = [1, 0, 0, 0.5]
    ecolours[g1.get_eids(addedvids), :] = [0, 0, 1, 0.5]

    igraph.plot(g1, outpath, bbox=(1200, 1200),
                vertex_label=g1.vs['title'],
                vertex_size=5,
                edge_color=ecolours.tolist())

##########################################################
def main(g1path, g2path, outdir):
    random.seed(0); np.random.seed(0)
    vattribs = ['hn', 'he', 'hd', 'hc', 'cr']
    partitions = ['deleted', 'kept', 'added']

    infopath = pjoin(outdir, 'info.txt')
    g1, g2 = load_graph(g1path), load_graph(g2path)
    g1, g2 = remove_disjoint_vertices(g1, g2) # Keep just common vertices
    wids = {wid: i for i, wid in enumerate(g1.vs['wid'])}

    # g1dual = g1.linegraph()
    # g2dual = g2.linegraph()

    n = g1.vcount()
    deleted, kept, added = get_link_partitions_of_the_union(g1, g2)


    deletedvids = [[wids[i], wids[j]] for i, j in deleted]
    keptvids = [[wids[i], wids[j]] for i, j in kept]
    addedvids = [[wids[i], wids[j]] for i, j in added]
    # calculate_shortest_paths(g1, deletedvids, keptvids, addedvids, outdir)

    plot_graph(g1, deletedvids, keptvids, addedvids, outdir)

    feats = {}
    feats['deleted'] = feats_from_edgewids(deleted, g1, vattribs)
    feats['kept'] = feats_from_edgewids(kept, g1, vattribs)
    feats['added'] = feats_from_edgewids(added, g1, vattribs)
    # txt = 'deleted:{}, kept:{}, added:{}'.format(len(deleted), len(kept), len(added))
    # append_to_file(infopath, txt)

    featsall = np.vstack((feats['deleted'], feats['kept'], feats['added']))
    a = .5

    coinc, coincpath = get_coincidx_graph(featsall, a, True, outdir)
    # plot_motifs(g1, deletedvids, '/home/tokuda/temp/coinc_498.csv', outdir)

    calculate_weighted_shortest_paths(coinc, deletedvids, keptvids, addedvids, outdir)

    # plot_features_individually(feats, vattribs, outdir)
    # X_train, Y_train, X_test, Y_test = split_train_test(feats)
    # plot_pca_from_feats(X_train, Y_train, partitions, outdir)

    # ml_approach1(X_train, Y_train, X_test, Y_test)
    # ml_approach2(X_train, Y_train, X_test, Y_test)
    # ml_approach_luc(X_train, Y_train, X_test, Y_test, partitions,
                    # vattribs, infopath, outdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph1', required=True, help='Graph 1 path (graphml)')
    parser.add_argument('--graph2', required=True, help='Graph 2 path (graphml)')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.graph1, args.graph2, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
