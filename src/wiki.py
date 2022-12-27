#!/usr/bin/env python3
"""Extract graph from wikipedia snapshots
"""

import argparse
import time, datetime
import os, sys, inspect, pickle
from os.path import join as pjoin
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme, append_to_file
import pandas as pd
import igraph

##########################################################
def load_dataframe(dfpath):
    info(inspect.stack()[0][3] + '()')

    sep = '\t'
    pklpath = dfpath.replace('.csv', '.pkl').replace('.tsv', '.pkl')
    if os.path.isfile(pklpath):
        return pickle.load(open(pklpath, 'rb'))
    elif os.path.isfile(dfpath):
        df = pd.read_csv(dfpath, sep=sep, low_memory=False)
        pickle.dump(df, open(pklpath, 'wb'))
        return df

##########################################################
def get_out_page_ids(queryid, df):
    return df.loc[df.page_id_from == queryid]['page_id_to'].unique()

##########################################################
def filter_df_by_ids(df, ids, removeid=-1):
    """Keep just rows containing src and tgt in ids"""
    # info(inspect.stack()[0][3] + '()')
    dffilt = df.loc[df.isin(ids).page_id_from & df.isin(ids).page_id_to]
    dffilt = dffilt[dffilt.page_id_from != removeid]
    dffilt = dffilt[dffilt.page_id_to != removeid]
    return dffilt

##########################################################
def get_id_all_pages(df):
    return df[['page_id_from', 'page_id_to']].values.flatten()

##########################################################
def convert_wids_to_vids(df, uid2vid):
    info(inspect.stack()[0][3] + '()')
    c1 = df.page_id_from.map(uid2vid)
    c2 = df.page_id_to.map(uid2vid)
    return pd.concat((c1, c2), axis=1)

##########################################################
def export_graphml(dfedges, dftitles, outpath):
    info(inspect.stack()[0][3] + '()')
    g = igraph.Graph(len(dftitles))
    g.add_edges(dfedges.values)
    g.vs['title'] = dftitles.title.values
    g.vs['wid'] = dftitles.id.values
    g.write_graphml(outpath)

##########################################################
def main(query, snap1path, snap2path, logpath, outdir):
    info(inspect.stack()[0][3] + '()')

    if not os.path.isfile(snap1path) or not os.path.isfile(snap2path):
        info('Check if snap1 and snap2 exist')
        return

    logfh = open(logpath, 'a')
    def log(txt):
        info(txt)
        logfh.write(txt + '\n')

    df1 = load_dataframe(snap1path)
    df2 = load_dataframe(snap2path)

    # f = os.path.basename(os.path.realpath(snap1path))
    f1 = os.path.basename(snap1path)
    outsuff1 = pjoin(outdir, f1.replace('.csv', ''))
    f2 = os.path.basename(snap2path)
    outsuff2 = pjoin(outdir, f2.replace('.csv', ''))

    queryid = df1.loc[df1.page_title_from == query].page_id_from.iloc[0]
    ids1 = get_out_page_ids(queryid, df1)
    ids2 = get_out_page_ids(queryid, df2)
    ids = np.concatenate((ids1, ids2))

    # log('Full graph (vs from either snapshots) n:{}'.format(len(ids)))

    df1aux = filter_df_by_ids(df1, ids, removeid=queryid)
    df2aux = filter_df_by_ids(df2, ids, removeid=queryid)
    ids1 = set(get_id_all_pages(df1aux))
    ids2 = set(get_id_all_pages(df2aux))
    ids = ids1.intersection(ids2) # Keeping just vs from both snapshots

    df1filt = filter_df_by_ids(df1aux, ids, removeid=queryid)
    df2filt = filter_df_by_ids(df2aux, ids, removeid=queryid)

    # uid2vid = {uid:i for i, uid in enumerate(ids)}

    df3 = pd.concat((df1filt, df2filt))
    aux = np.row_stack((df3[['page_id_from', 'page_title_from']],
                      df3[['page_id_to', 'page_title_to']]))
    df4 = pd.DataFrame(aux, columns=['id', 'title']).drop_duplicates()
    df4.to_csv(pjoin(outdir, 'titles.lst'), sep='\t', index=False)
    uid2vid = {uid:vid for vid, uid in enumerate(df4.id)}
    n = len(df4)

    n1 = len(get_id_all_pages(df1filt))
    n2 = len(get_id_all_pages(df2filt))
    log('orign1:{}, orign2:{}, n:{}, m1:{}, m2:{}'. \
         format(n1, n2, len(df4), len(df1filt), len(df2filt)))

    # Reindex
    df1filt = convert_wids_to_vids(df1filt, uid2vid)
    df2filt = convert_wids_to_vids(df2filt, uid2vid)

    # Export list of edges
    df1filt.to_csv(outsuff1 + '_edges.tsv', index=False, header=None, sep='\t')
    df2filt.to_csv(outsuff2 + '_edges.tsv', index=False, header=None, sep='\t')

    # Export graphml
    export_graphml(df1filt, df4, outsuff1 + '.graphml')
    export_graphml(df2filt, df4, outsuff2 + '.graphml')

    logfh.close()

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snap1', default='./data/snap1.csv',
                        help='Path to the prev snapshot')
    parser.add_argument('--snap2', default='./data/snap2.csv',
                        help='Path to the following snapshot')
    parser.add_argument('--query', required=True, help='Query')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.query, args.snap1, args.snap2, readmepath, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
