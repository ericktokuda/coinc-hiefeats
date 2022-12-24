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
from myutils import info, create_readme
import pandas as pd

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
    """Short description """
    info(inspect.stack()[0][3] + '()')
    dffilt = df.loc[df.isin(ids).page_id_from & df.isin(ids).page_id_to]
    dffilt = dffilt[dffilt.page_id_from != removeid]
    dffilt = dffilt[dffilt.page_id_to != removeid]
    return dffilt

##########################################################
def get_id_all_pages(df):
    return df[['page_id_from', 'page_id_to']].values.flatten()

##########################################################
def main(query, snap1path, snap2path, outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    f = os.path.basename(snap1path)
    df1 = load_dataframe(snap1path)
    df2 = load_dataframe(snap2path)

    queryid = df1.loc[df1.page_title_from == query].page_id_from.iloc[0]
    ids1 = get_out_page_ids(queryid, df1)
    ids2 = get_out_page_ids(queryid, df2)
    ids = np.concatenate((ids1, ids2))

    info('Full graph n:'.format(len(ids)))

    df1aux = filter_df_by_ids(df1, ids, removeid=queryid)
    df2aux = filter_df_by_ids(df2, ids, removeid=queryid)
    ids1 = set(get_id_all_pages(df1aux))
    ids2 = set(get_id_all_pages(df2aux))
    ids = ids1.intersection(ids2)

    df1filt = filter_df_by_ids(df1aux, ids, removeid=queryid)
    df2filt = filter_df_by_ids(df2aux, ids, removeid=queryid)

    uid2vid = {uid:i for i, uid in enumerate(ids)}

    df3 = pd.concat((df1filt, df2filt))
    aux = np.row_stack((df3[['page_id_from', 'page_title_from']],
                      df3[['page_id_to', 'page_title_to']]))
    df4 = pd.DataFrame(aux, columns=['id', 'title']).drop_duplicates()

    n1 = len(get_id_all_pages(df1filt))
    n2 = len(get_id_all_pages(df2filt))

    info('n:{}, n1:{}, n2:{}, m1:{}, m2:{}'. \
         format(len(ids), n1, n2, len(df1filt), len(df2filt)))

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('--snap1', required=True, help='Path to the prev snapshot')
    # parser.add_argument('--snap2', required=True, help='Path to the later snapshot')
    parser.add_argument('--query', required=True, help='Query')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    # snap1path = args.snap1
    # snap2path = args.snap2
    # snap1path = './enwiki.wikilink_graph.2002-03-01.csv'
    # snap2path = './enwiki.wikilink_graph.2003-03-01.csv'
    snap1path = './enwiki.wikilink_graph.2015-03-01.csv'
    snap2path = './enwiki.wikilink_graph.2018-03-01.csv'

    main(args.query, snap1path, snap2path, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
