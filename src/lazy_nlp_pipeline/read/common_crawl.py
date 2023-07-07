from collections.abc import Generator, Iterable
import gzip
import json
import logging
import os

from tqdm.auto import tqdm
import urllib.request
from warcio.archiveiterator import ArchiveIterator


def cc_get_crawl_ids(data_folder='./', download_collinfo_json: bool = True,
                     ids: str = 'latest') -> list[str]:
    collinfo_fpath = os.path.join(data_folder, 'collinfo.json')
    if not os.path.isfile(collinfo_fpath):
        if not download_collinfo_json:
            raise ValueError(f'File {collinfo_fpath!r} should be present'
                             ' or `download_crawls_index` set to True')
        collinfo_url = 'https://index.commoncrawl.org/collinfo.json'
        urllib.request.urlretrieve(collinfo_url, filename=collinfo_fpath)
        logging.info(f'Loaded file {collinfo_fpath}')
    with open(collinfo_fpath) as f:
        collinfo = json.load(f)
    if ids == 'latest':
        return [collinfo[0]['id']]
    if ids == 'all':
        return [i['id'] for i in collinfo]
    raise ValueError(f'ids should be "latest" or "all", got {ids!r}')


def cc_get_cluster_idx_records(crawl_ids: Iterable[str], surt_prefix: str = '', data_folder='./',
                               download_cluster_idxs: bool = False,
                               ) -> list[dict]:
    crawlid_fpath_list = []
    for crawl_id in tqdm(crawl_ids, desc='Loading cluster.idx files', leave=False):
        cluster_index_folder = os.path.join(data_folder, crawl_id)
        os.makedirs(cluster_index_folder, exist_ok=True)
        cluster_index_fpath = os.path.join(cluster_index_folder, 'cluster.idx')
        if not os.path.isfile(cluster_index_fpath):
            if not download_cluster_idxs:
                raise ValueError(f'File {cluster_index_fpath!r} should be present'
                                 ' or `download_cluster_indexes` set to True')
            cluster_index_url = f'https://data.commoncrawl.org/cc-index/collections/{crawl_id}/indexes/cluster.idx'
            urllib.request.urlretrieve(cluster_index_url, filename=cluster_index_fpath)
            logging.info(f'Loaded file {cluster_index_fpath}')
        crawlid_fpath_list.append((crawl_id, cluster_index_fpath))

    index_records = []
    for crawl_id, cluster_index_fpath in tqdm(crawlid_fpath_list, desc='Parsing cluster.idx files', leave=False):
        index_block_lines: list[str] = []
        with open(cluster_index_fpath) as f:
            prev_line = ''
            while True:
                line = f.readline()
                if line > surt_prefix:
                    index_block_lines.append(prev_line)
                    if not line.startswith(surt_prefix):
                        break
                prev_line = line

        for L in index_block_lines:
            _, index_file, offset, length, _ = L.split('\t')
            index_records.append(dict(surt_prefix=surt_prefix,
                                      crawl_id=crawl_id, fname=index_file,
                                      offset=int(offset), length=int(length)))
    return index_records


def cc_get_cdx_records(cluster_idx_records, data_folder='./',
                       download_cdx: bool = False,
                       ) -> list[dict]:
    cdx_to_download = []
    for cluster_idx_record in cluster_idx_records:
        crawl_id = cluster_idx_record['crawl_id']
        cdx_folder = os.path.join(data_folder, crawl_id, 'cdx')
        os.makedirs(cdx_folder, exist_ok=True)
        cdx_fname = cluster_idx_record['fname']
        cdx_fpath = os.path.join(cdx_folder, cdx_fname)
        if not os.path.isfile(cdx_fpath):
            cdx_url = f'https://data.commoncrawl.org/cc-index/collections/{crawl_id}/indexes/{cdx_fname}'
            cdx_to_download.append((cdx_url, cdx_fpath))
    cdx_to_download = list(set(cdx_to_download))

    for cdx_url, cdx_fpath in tqdm(cdx_to_download, desc='Downloading cdx-XX.gz files', leave=False):
        if not download_cdx:
            raise ValueError(f'File {cdx_fpath!r} should be present'
                             ' or `download_cdx` set to True')
        urllib.request.urlretrieve(cdx_url, filename=cdx_fpath)
        logging.info(f'Loaded file {cdx_fpath}')

    cdx_records = []
    for cluster_idx_record in tqdm(cluster_idx_records, desc='Parsing cluster_idx_records', leave=False):
        crawl_id = cluster_idx_record['crawl_id']
        cdx_folder = os.path.join(data_folder, crawl_id, 'cdx')
        cdx_fname = cluster_idx_record['fname']
        cdx_fpath = os.path.join(cdx_folder, cdx_fname)

        cdx_lines = []
        with open(cdx_fpath, 'rb') as f:
            f.seek(cluster_idx_record['offset'])
            data = f.read(cluster_idx_record['length'])
            lines = gzip.decompress(data).decode().split('\n')
            lines = [L for L in lines if L and L.startswith(cluster_idx_record['surt_prefix'])]
            cdx_lines.extend(lines)

        for L in cdx_lines:
            _, _, json_data = L.split(maxsplit=2)
            record = json.loads(json_data)
            cdx_records.append(record)
    return cdx_records


def cc_get_warc_records(cdx_records) -> Generator[tuple[list, list, str], None, None]:
    # TODO: implement caching
    for cdx_record in tqdm(cdx_records, desc='Downloading WARC record', leave=False):
        filename = cdx_record['filename']
        offset = int(cdx_record['offset'])
        length = int(cdx_record['length'])
        end = offset + length - 1
        url = f'https://data.commoncrawl.org/{filename}'
        req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-{end}'})
        with urllib.request.urlopen(req) as resp:
            # data = resp.read()
            # warc_str = gzip.decompress(data).decode()
            for record in ArchiveIterator(resp):
                warc_headers = record.rec_headers.headers
                http_headers = record.http_headers.headers
                html_str = record.content_stream().read().decode()
                yield warc_headers, http_headers, html_str
