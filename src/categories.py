#!/usr/bin/env python3
"""Analyze categories of current snapshot
"""

import argparse
import time, datetime
import os, sys, random
from os.path import join as pjoin
import inspect

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import subprocess
from subprocess import Popen, DEVNULL, PIPE
import pandas as pd

MYSQL2CSVPATH = 'snapshot/mysqlgz2csv.sh'
NSPAGE = 0
NSCATEG = 14

##########################################################
def shell_call(cmd):
    info(cmd)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    info('output:{}'.format(output))
    # info('err:{}'.format(err))
    return output, err

##########################################################
def get_tmpfile_path(ndigits, outdir):
    """Get pseudo-random temporary file name"""
    return pjoin(outdir, 'tmp{}'.format(int(random.random()*10**(ndigits +1))))

##########################################################
def parse_categlinks_dump(catlinksdump, outpath, debug, tmpdir):
    """Parse category links sql.gz dump and get the fields srcid and tgttitle"""
    info(inspect.stack()[0][3] + '()')

    tmp1 = get_tmpfile_path(5, tmpdir)
    tmp2 = get_tmpfile_path(5, tmpdir)

    # if os.path.isfile(outpath):
        # info('Output path {} already exists'.format(outpath))
        # return

    cmd = '''./{} {} > {}'''.format(MYSQL2CSVPATH, catlinksdump, tmp1)
    output, err = shell_call(cmd)

    cmd = '''cat {} | awk -F "','" '{{print $1}}' > {}'''.format(tmp1, tmp2)
    output, err = shell_call(cmd)

    txt = open(tmp2, 'rb').read()
    lines = txt.split(b'\n')

    fh = open(outpath, 'wb')

    if debug: lines = lines[:100]

    for l in lines:
        if not b",'" in l: continue
        l = l.replace(b",'", b'\t').replace(b"\\'", b"'")
        fh.write(l + b'\n')
    fh.close()

    if debug: return

    os.remove(tmp1)
    os.remove(tmp2)

##########################################################
def parse_page_dump(pagedump, outpath, debug, tmpdir):
    """Parse compressed category sql.gz dump and get the three
    first fields: pageid, ns, title"""
    info(inspect.stack()[0][3] + '()')

    tmp1 = get_tmpfile_path(5, tmpdir)
    tmp2 = get_tmpfile_path(5, tmpdir)

    if os.path.isfile(outpath):
        info('Output path {} already exists'.format(outpath))
        return

    cmd = '''./{} {} > {}'''.format(MYSQL2CSVPATH, pagedump, tmp1)
    output, err = shell_call(cmd)

    cmd = '''sed "s/',\(.\).*/',\\1/g" {} > {}'''.format(tmp1, tmp2)
    output, err = shell_call(cmd)

    txt = open(tmp2, 'rb').read()
    lines = txt.split(b'\n')
    if debug: lines = lines[:100]

    fh = open(outpath, 'wb')
    for l in lines:
        if not b"'," in l: continue
        txt, isredir = l.split(b"',")
        if not b",'" in txt: continue
        txt, title = txt.split(b",'")
        title = title.replace(b"\\'", b"'")
        if not b"," in txt: continue
        pageid, ns = txt.split(b",")

        fh.write(pageid + b'\t' + ns + b'\t' +  title + b'\t' + isredir + b'\n')
    fh.close()

    if debug: return

    os.remove(tmp1)
    os.remove(tmp2)

##########################################################
def filter_categ_from_pages(pagetsv, outpath):
    """Filter categories from pagetsv using the namespace"""
    info(inspect.stack()[0][3] + '()')

    if os.path.isfile(outpath):
        info('Output path {} already exists'.format(outpath))
        return

    cmd = '''grep -P '\t{}\t' {} | cut -d$'\t' -f1,3 > {}'''. \
        format(NSCATEG, pagetsv, outpath)
    output, err = shell_call(cmd)

##########################################################
def analyze_categs(catlinkspath, outdir):
    df = pd.read_csv(catspath)

##########################################################
def get_descendants_from_categ(querycat, dfpages, dfcatlinks, debug, curdepth,
                               maxdepth=-1):
    '''Recursive function for getting the descendants from a desired categ'''
    if curdepth == maxdepth: return []

    children = dfcatlinks.loc[dfcatlinks.tgttitle == querycat]
    matched = children.merge(dfpages, how='left', left_on='srcid', right_on='id')

    leaves = matched[['id', 'ns', 'title']].values
    n = len(leaves)
    leaves = np.c_[leaves, np.ones(n, dtype=int) * (curdepth + 1)]

    children = matched.loc[matched.ns == NSCATEG].title.values
    if debug: children = children[:3]
    for query in children:
        desc = get_descendants_from_categ(query, dfpages, dfcatlinks,
                                          debug, curdepth + 1, maxdepth)
        if len(desc): leaves = np.vstack((leaves, desc))

    return leaves

##########################################################
def get_descendants_from_categ_all(querycat, pagetsv, catlinkstsv,
                                   maxdepth, debug, outdir):
    info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, '{}_max{}.tsv'.format(querycat, maxdepth))

    if os.path.exists(outpath):
        info('Output path {} already exists'.format(outpath))
        return
    dfpage = pd.read_csv(pagetsv, sep='\t',
                         names=['id', 'ns', 'title', 'isredir'],
                         low_memory=False)
    dfcatlinks = pd.read_csv(catlinkstsv, sep='\t', names=['srcid', 'tgttitle'])
    leaves = get_descendants_from_categ(querycat, dfpage, dfcatlinks,
                                        debug, 0, maxdepth)
    df = pd.DataFrame(leaves).drop_duplicates(keep='first')
    df.to_csv(outpath, sep='\t', index=False, header=['id', 'ns', 'title', 'hops'])
    return outpath

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    info(' '.join(sys.argv))
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--querycat', required=True, help='Category to be searched')
    parser.add_argument('--dumpsdir', required=True, help='Folder containing the dumps')
    parser.add_argument('--maxdepth', default=1, type=int, help='Dump folder')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    fs = sorted(os.listdir(args.dumpsdir))

    for f in fs:
        if f.endswith('-categorylinks.sql.gz'):
            catlinksgz = pjoin(args.dumpsdir, f)
        elif f.endswith('-page.sql.gz'):
            pagegz = pjoin(args.dumpsdir, f)

    catlinkstsv = pjoin(args.outdir, 'catlinks.tsv')
    if args.debug: catlinkstsv = catlinkstsv + '.debug'
    pagetsv = pjoin(args.outdir, 'page.tsv')
    if args.debug: pagetsv = pagetsv + '.debug'

    parse_page_dump(pagegz, pagetsv, args.debug, args.outdir)
    parse_categlinks_dump(catlinksgz, catlinkstsv, args.debug, args.outdir)
    get_descendants_from_categ_all(args.querycat, pagetsv, catlinkstsv,
                                   args.maxdepth, args.debug, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
