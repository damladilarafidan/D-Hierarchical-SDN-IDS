# %% [markdown]
# # D-Hierarchical: Inductive Multi-Seed Evaluation, Calibration, and Tabular Baselines for SDN/NIDS
#
# Official evaluation script and notebook reproducing all results reported in the paper:
# * **Datasets:** InSDN (5 stratified splits, seeds 0–4), CICIDS2017 (`TrafficLabelling` with IP endpoints, 3 splits), UNSW-NB15 (3 splits).
# * **Inductive Protocol (Split-Before-Graph):** Feature scalers (`StandardScaler` after signed `log1p`) and weighted node feature aggregations are fitted strictly on training flows. Unseen test endpoints map to the `UNK` (zero) node vector.
# * **Models Evaluated:**
#   - `D-Hierarchical` (GCTLayerResNorm × 2 + `/24` SubnetAggregation + EdgeAwareLSTMResNorm) with composite synthetic keys (`scope='paper'`) and true IP + `/24` subnet hierarchies (`scope='ip'`).
#   - `Random Forest` and `LightGBM` (`edge` and `edge+node` views).
#   - `Hybrid (GNN OOF -> LightGBM)` via 2-fold out-of-fold cross-fitting inside the training split.
# * **Calibration:** Validation-only coordinate-ascent logit bias calibration (`raw` vs `cal`), evaluated without touching test labels.
# * **Outputs:** `v2_runs.csv`, `v2_perclass.csv`, `v2_summary.csv`, `v2_tests.csv`.

# %%
import subprocess, sys, importlib, os

def _has(m):
    try:
        importlib.import_module(m)
        return True
    except Exception:
        return False

if not _has('torch_geometric'):
    print('Installing torch_geometric...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch_geometric'])
if not _has('lightgbm') and os.path.isdir('/kaggle'):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'lightgbm'])
print('Packages ready.')

# %%
import os, gc, time, math, glob, ipaddress, traceback, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LSTM
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score
from scipy import stats as st

QUICK = os.environ.get('LEAK_QUICK') == '1'
SPLITS      = [0, 1, 2]
GNN_SPLIT   = 0
E4_SEEDS    = [0, 1, 2]
RF_TREES    = 200
HIDDEN      = 128
EPOCHS      = 30
PATIENCE    = 8
LR          = 1e-3
BATCH       = 512          # random 512-flow mini-batch subgraphs
MAX_ROWS    = None         # optional subsampling (QUICK mode only)
TIME_BUDGET_H = 11.0

if QUICK:
    SPLITS, E4_SEEDS, RF_TREES, HIDDEN, EPOCHS, PATIENCE, MAX_ROWS = [0], [0, 1], 20, 16, 1, 1, 30_000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = os.environ.get('LEAK_OUT') or ('/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
T_START = time.time()
print(f'device={device} QUICK={QUICK} splits={SPLITS} HIDDEN={HIDDEN} EPOCHS={EPOCHS}')

def time_left():
    return TIME_BUDGET_H * 3600 - (time.time() - T_START)

def safe(fn, *a, **k):
    if time_left() < 0:
        print('  !! time budget exceeded, skipping'); return None
    try:
        return fn(*a, **k)
    except Exception:
        traceback.print_exc(); return None
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

ROOTS = [os.environ.get('DATA_DIR', './data'), '/kaggle/input']

def find_files(pattern):
    out = []
    for r in ROOTS:
        if os.path.isdir(r):
            out += glob.glob(os.path.join(r, '**', pattern), recursive=True)
    return sorted(set(out))

# %%
# ═══ DATASET LOADERS ═════════════════════════════════════════════
# Each loader returns a dict with:
#   name, y (int), classes, NF (raw node feature cols), w (node weight col), EF (raw edge feature cols),
#   key_fn(scope) -> (src_key, dst_key) arrays, subnet_fn(key) -> str, stats (aggregation functions)
_RFC1918 = [ipaddress.ip_network(c) for c in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16')]

def _internal(ip):
    try:
        a = ipaddress.ip_address(ip)
        return any(a in n for n in _RFC1918)
    except Exception:
        return False

def _num(s, default=0.0):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(default)

def _subsample(df, ycol):
    if MAX_ROWS and len(df) > MAX_ROWS:
        r, keep = np.random.RandomState(0), []
        for _, idx in df.groupby(ycol).indices.items():
            n = min(len(idx), max(30, int(MAX_ROWS * len(idx) / len(df))))
            keep.append(r.choice(idx, n, replace=False))
        df = df.iloc[np.sort(np.concatenate(keep))]
    return df.reset_index(drop=True)

def _finish(name, df, y_raw, NF, w, EF, key_fn, subnet_fn, stats):
    keep = (pd.Series(y_raw).map(pd.Series(y_raw).value_counts()) >= 10).values  # filter classes with <10 total samples
    df, y_raw = df[keep].reset_index(drop=True), np.asarray(y_raw)[keep]
    classes, y = np.unique(y_raw, return_inverse=True)
    kf = lambda scope: tuple(np.asarray(a)[keep] for a in key_fn(scope))
    D = dict(name=name, df=df, y=y.astype(np.int64), classes=classes, NF=NF, w=w, EF=EF,
             key_fn=kf, subnet_fn=subnet_fn, stats=stats)
    print(f'[{name}] {len(df):,} flows | {len(classes)} classes: ' +
          ', '.join(f'{c}={n}' for c, n in zip(classes, np.bincount(y))))
    return D


def load_insdn():
    fs = [f for f in find_files('*.csv') if os.path.basename(f).lower() in
          ('normal_data.csv', 'ovs.csv', 'metasploitable-2.csv')]
    fs = list({os.path.basename(f).lower(): f for f in fs}.values())
    if len(fs) < 3:
        print('InSDN not found, skipping.'); return None
    raw = pd.concat((pd.read_csv(f, on_bad_lines='skip', low_memory=False) for f in fs), ignore_index=True)
    raw.columns = raw.columns.str.strip()
    raw['Label'] = raw['Label'].astype(str).str.strip().str.replace(r'\s+', '', regex=True)
    raw = _subsample(raw, 'Label')
    proto = raw['Protocol'].astype(str).str.strip().str.upper().map(
        {'6': 'TCP', 'TCP': 'TCP', '17': 'UDP', 'UDP': 'UDP', '1': 'ICMP', 'ICMP': 'ICMP'}).fillna('OTHER')
    fwd, bwd = _num(raw['Tot Fwd Pkts']), _num(raw['Tot Bwd Pkts'])
    tf, tb = _num(raw['TotLen Fwd Pkts']), _num(raw['TotLen Bwd Pkts'])
    dur = _num(raw['Flow Duration'])
    flows = np.maximum(1, _num(raw.get('Subflow Fwd Pkts', 1), 1))
    pk, by = fwd + bwd, tf + tb
    df = pd.DataFrame({
        'dt': np.maximum(1, dur), 'switch': 1.0, 'pktcount': pk, 'bytecount': by, 'dur': dur, 'tot_dur': dur,
        'flows': flows, 'packetins': _num(raw.get('Fwd PSH Flags', 0)) + _num(raw.get('Bwd PSH Flags', 0)),
        'pktperflow': pk / flows, 'byteperflow': by / flows, 'pktrate': _num(raw.get('Flow Pkts/s', 0)),
        'Pairflow': 0.0, 'port_no': _num(raw['Dst Port']), 'tx_bytes': tf, 'rx_bytes': tb,
        'tx_kbps': _num(raw.get('Fwd Pkts/s', 0)), 'rx_kbps': _num(raw.get('Bwd Pkts/s', 0)),
        'tot_kbps': _num(raw.get('Flow Byts/s', 0)) / 1000.0})
    for c, a, b in [('tx_rx_ratio', 'tx_bytes', 'rx_bytes'), ('bytes_per_pkt', 'bytecount', 'pktcount'),
                    ('pkt_flow_ratio', 'pktcount', 'flows'), ('byte_flow_ratio', 'bytecount', 'flows'),
                    ('kbps_ratio', 'tx_kbps', 'rx_kbps'), ('pkt_rate_norm', 'pktrate', 'dur'),
                    ('byte_rate', 'bytecount', 'dur'), ('tx_per_flow', 'tx_bytes', 'flows'),
                    ('rx_per_flow', 'rx_bytes', 'flows'), ('psh_ratio', 'packetins', 'pktcount')]:
        df[c] = df[a] / (df[b].abs() + 1.0)
    for p in ('TCP', 'UDP', 'ICMP'):
        df[f'proto_{p}'] = (proto == p).astype(float)
    NF = ['dt', 'switch', 'pktcount', 'bytecount', 'flows', 'packetins', 'pktperflow', 'byteperflow', 'pktrate',
          'Pairflow', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps', 'port_no']
    EF = ['dt', 'pktcount', 'bytecount', 'dur', 'tot_dur', 'flows', 'packetins', 'pktperflow', 'byteperflow',
          'pktrate', 'Pairflow', 'port_no', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps',
          'tx_rx_ratio', 'bytes_per_pkt', 'pkt_flow_ratio', 'byte_flow_ratio', 'kbps_ratio', 'pkt_rate_norm',
          'byte_rate', 'tx_per_flow', 'rx_per_flow', 'psh_ratio', 'proto_TCP', 'proto_UDP', 'proto_ICMP']
    src, dst = raw['Src IP'].astype(str).values, raw['Dst IP'].astype(str).values
    tail = ('_' + proto.values + '_' + df['port_no'].astype(np.int64).astype(str).values + '_1')

    def key_fn(scope):
        if scope == 'paper':                              # composite synthetic key: IP_PROTO_PORT_SWITCH
            return src + tail, dst + tail
        m = {ip: (ip if _internal(ip) else 'ext_' + '.'.join(ip.split('.')[:2])) for ip in np.unique(np.r_[src, dst])}
        return np.array([m[i] for i in src]), np.array([m[i] for i in dst])     # 'ip': internal host IP / external /16 prefix

    def subnet_fn(k):
        ip = k.split('_')[0] if not k.startswith('ext_') else k
        return ('extsub_' + k[4:].split('.')[0]) if k.startswith('ext_') else '.'.join(ip.split('.')[:3])
    return _finish('InSDN', df, raw['Label'].values, NF, 'dt', EF, key_fn, subnet_fn,
                   ['mean', 'max', 'min', 'std', 'sum'])


def load_unsw():
    fs = [f for f in find_files('*.csv') if ('training-set' in os.path.basename(f).lower()
                                             or 'testing-set' in os.path.basename(f).lower())]
    fs = list({os.path.basename(f).lower(): f for f in fs}.values())
    if not fs:
        print('UNSW-NB15 not found, skipping.'); return None
    raw = pd.concat((pd.read_csv(f, low_memory=False) for f in fs), ignore_index=True)
    raw.columns = raw.columns.str.strip().str.lower()
    raw['attack_cat'] = raw['attack_cat'].astype(str).str.strip().replace({'nan': 'Normal', '': 'Normal'})
    raw = _subsample(raw, 'attack_cat')
    g = lambda c: _num(raw[c]) if c in raw else pd.Series(0.0, index=raw.index)
    df = pd.DataFrame({'pktcount': g('spkts') + g('dpkts'), 'bytecount': g('sbytes') + g('dbytes'),
                       'dur': g('dur'), 'tx_bytes': g('sbytes'), 'rx_bytes': g('dbytes'),
                       'tx_pkts': g('spkts'), 'rx_pkts': g('dpkts')})
    for c in ['sload', 'dload', 'sjit', 'djit', 'sttl', 'dttl', 'tcprtt', 'synack', 'sinpkt', 'dinpkt',
              'smean', 'dmean', 'ct_srv_src', 'ct_srv_dst', 'rate', 'is_sm_ips_ports']:
        df[c] = g(c)
    pr = raw['proto'].astype(str).str.lower()
    for p in ('tcp', 'udp', 'icmp'):
        df[f'proto_{p}'] = (pr == p).astype(float)
    for c, a, b in [('byte_ratio', 'tx_bytes', 'rx_bytes'), ('pkt_ratio', 'tx_pkts', 'rx_pkts'),
                    ('jit_ratio', 'sjit', 'djit'), ('load_ratio', 'sload', 'dload'), ('byte_rate', 'bytecount', 'dur')]:
        df[c] = df[a] / (df[b].abs() + 1.0)
    NF = ['pktcount', 'bytecount', 'dur', 'tx_bytes', 'rx_bytes', 'tx_pkts', 'rx_pkts', 'sload', 'dload',
          'sttl', 'dttl', 'smean', 'dmean', 'rate', 'ct_srv_src', 'ct_srv_dst']
    EF = [c for c in df.columns]
    st_ = raw['state'].astype(str).str.upper().values
    sv = raw['service'].astype(str).str.lower().values
    ttl = pd.cut(df['dttl'], bins=[-1e9, 0, 64, 128, 1e9], labels=['none', 'low', 'mid', 'high']).astype(str).values

    def key_fn(scope):                                    # composite key: proto_state / svc_service_ttl
        return pr.values + '_' + st_, 'svc_' + sv + '_ttl' + ttl
    return _finish('UNSW-NB15', df, raw['attack_cat'].values, NF, 'pktcount', EF, key_fn,
                   lambda k: k.split('_')[0], ['mean', 'max', 'min', 'sum'])


def load_cicids():
    fs = []
    fs_ip = []
    for f in find_files('*.csv'):
        b = os.path.basename(f).lower()
        if b in ('normal_data.csv', 'ovs.csv', 'metasploitable-2.csv') or 'unsw' in f.lower() or 'nf-' in b:
            continue
        try:
            h = [c.strip() for c in pd.read_csv(f, nrows=0, encoding='latin1').columns]
        except Exception:
            continue
        if 'Label' in h and ('Destination Port' in h or 'Dst Port' in h) and ('Flow Duration' in h):
            (fs_ip if ('Source IP' in h or 'Src IP' in h) else fs).append(f)
    # If both TrafficLabelling (with IPs) and MachineLearningCVE are present,
    # prefer TrafficLabelling to avoid duplicating flows and to retain IP endpoints.
    fs = fs_ip or fs
    if not fs:
        print('CICIDS2017 not found, skipping.'); return None
    print(f'  CICIDS2017: {len(fs)} files ({"with IP columns" if fs_ip else "no IP columns"})')
    raw = pd.concat((pd.read_csv(f, low_memory=False, encoding='latin1') for f in fs), ignore_index=True)
    raw.columns = raw.columns.str.strip()
    raw['Label'] = raw['Label'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    raw = raw[~raw['Label'].isin(['', 'nan', 'Label'])]
    ben = raw['Label'].str.upper() == 'BENIGN'
    if ben.sum() > 300_000:                                  # cap benign class at 300k flows
        raw = pd.concat([raw[ben].sample(300_000, random_state=42), raw[~ben]])
    raw = _subsample(raw.reset_index(drop=True), 'Label')
    pick = lambda *cs: next((c for c in cs if c in raw), None)
    cols = {'fwd': pick('Total Fwd Packets', 'Tot Fwd Pkts'), 'bwd': pick('Total Backward Packets', 'Tot Bwd Pkts'),
            'lf': pick('Total Length of Fwd Packets', 'TotLen Fwd Pkts'),
            'lb': pick('Total Length of Bwd Packets', 'TotLen Bwd Pkts'), 'dur': 'Flow Duration',
            'bps': pick('Flow Bytes/s', 'Flow Byts/s'), 'pps': pick('Flow Packets/s', 'Flow Pkts/s'),
            'fps': pick('Fwd Packets/s', 'Fwd Pkts/s'), 'bwps': pick('Bwd Packets/s', 'Bwd Pkts/s'),
            'fh': pick('Fwd Header Length'), 'bh': pick('Bwd Header Length'),
            'pm': pick('Packet Length Mean', 'Pkt Len Mean'), 'ps': pick('Packet Length Std', 'Pkt Len Std'),
            'im': pick('Flow IAT Mean'), 'is': pick('Flow IAT Std'), 'port': pick('Destination Port', 'Dst Port')}
    g = lambda k: _num(raw[cols[k]]) if cols[k] else pd.Series(0.0, index=raw.index)
    df = pd.DataFrame({k: g(k) for k in cols})
    for fl in ('FIN', 'SYN', 'RST', 'PSH', 'ACK'):
        c = pick(f'{fl} Flag Count', f'{fl} Flag Cnt')
        df[fl.lower()] = _num(raw[c]) if c else 0.0
    df['pktcount'], df['bytecount'] = df['fwd'] + df['bwd'], df['lf'] + df['lb']
    pc = pick('Protocol')
    proto = _num(raw[pc]).astype(int).astype(str).values if pc else np.full(len(raw), '0')
    NF = ['pktcount', 'bytecount', 'dur', 'pps', 'lf', 'lb', 'fps', 'bwps', 'bps', 'fh', 'bh', 'pm', 'ps', 'port']
    EF = list(df.columns)
    sip, dip = pick('Source IP', 'Src IP'), pick('Destination IP', 'Dst IP')
    if sip and dip:
        s, d = raw[sip].astype(str).values, raw[dip].astype(str).values
        key_fn = lambda scope: (s + '_' + proto, d + '_' + proto)
        subnet_fn = lambda k: '.'.join(k.split('_')[0].split('.')[:3])
        print('  CICIDS2017: IP columns detected -> IP_PROTO nodes')
    else:
        def _pg(p):
            p = int(p)
            return {80: 'HTTP', 443: 'HTTPS', 22: 'SSH', 21: 'FTP', 25: 'SMTP', 53: 'DNS'}.get(
                p, f'SYS_{p}' if p < 1024 else (f'REG_{(p // 100) * 100}' if p < 5000 else f'DYN_{(p // 1000) * 1000}'))
        pb = pd.cut(df['pps'], bins=5, labels=['vlow', 'low', 'mid', 'high', 'vhigh']).astype(str).values
        pg = np.array([_pg(p) for p in df['port'].values])
        key_fn = lambda scope: (pb + '_proto' + proto, pg + '_proto' + proto)
        subnet_fn = lambda k: k.split('_')[0]
        print('  CICIDS2017: No IP columns -> fallback composite key (pktrate bin / port group)')
    return _finish('CICIDS2017', df, raw['Label'].values, NF, 'dur', EF, key_fn, subnet_fn,
                   ['mean', 'max', 'min', 'sum'])

# Datasets to evaluate:
DATASETS_TO_RUN = ['InSDN', 'UNSW-NB15', 'CICIDS2017']
_LOADERS = {'InSDN': load_insdn, 'UNSW-NB15': load_unsw, 'CICIDS2017': load_cicids}
DATA = [d for d in (safe(_LOADERS[n]) for n in DATASETS_TO_RUN) if d is not None]
print('Loaded datasets:', [d['name'] for d in DATA])

# %%
# ═══ INDUCTIVE GRAPH & FEATURE CONSTRUCTION ══════════════════════
def slog(a):
    a = np.asarray(a, np.float64)
    return np.sign(a) * np.log1p(np.abs(a))

def splits_for(D, seed):
    idx = np.arange(len(D['y']))
    tr, tmp = train_test_split(idx, test_size=0.36, random_state=seed, stratify=D['y'])
    va, te = train_test_split(tmp, test_size=0.556, random_state=seed, stratify=D['y'][tmp])
    return tr, va, te

def build_cell(D, sp, label_in_key, transductive, y=None, scope='paper', shuffle_subnet=False):
    y = D['y'] if y is None else y
    tr, va, te = sp
    fit = np.arange(len(y)) if transductive else tr
    ks, kd = D['key_fn'](scope)
    if label_in_key:
        lab = y.astype(str)
        ks, kd = np.char.add(np.char.add(ks.astype(str), '|L'), lab), np.char.add(np.char.add(kd.astype(str), '|L'), lab)
    # flow features: fit StandardScaler strictly on 'fit' (training) rows
    fn = slog(D['df'][D['NF']].values); fe = slog(D['df'][D['EF']].values)
    fn = StandardScaler().fit(fn[fit]).transform(fn)
    E = np.nan_to_num(StandardScaler().fit(fe[fit]).transform(fe)).astype(np.float32)
    w = D['df'][D['w']].values.astype(np.float64).clip(min=0) + 1e-8
    # node table: weighted aggregation strictly over 'fit' (training) flows
    K = np.r_[ks[fit], kd[fit]]
    codes, uniq = pd.factorize(K)
    W = np.r_[w[fit], w[fit]]
    Wn = W / np.bincount(codes, weights=W)[codes]
    V = pd.DataFrame(np.r_[fn[fit], fn[fit]] * Wn[:, None])
    agg = V.groupby(codes).agg(D['stats']).fillna(0.0).values
    agg = np.nan_to_num(StandardScaler().fit_transform(agg)).astype(np.float32)
    X = np.zeros((len(uniq) + 1, agg.shape[1]), np.float32); X[1:] = agg      # id 0 = UNK
    k2i = pd.Series(np.arange(1, len(uniq) + 1), index=uniq)
    sid = k2i.reindex(ks).fillna(0).astype(np.int64).values
    did = k2i.reindex(kd).fillna(0).astype(np.int64).values
    sub_keys = ['unknown'] + [D['subnet_fn'](str(k).split('|L')[0]) for k in uniq]
    sub = pd.factorize(pd.Series(sub_keys))[0].astype(np.int64)
    if shuffle_subnet:
        r = np.random.RandomState(123); sub[1:] = sub[1:][r.permutation(len(sub) - 1)]
    return dict(X=X, E=E, sid=sid, did=did, sub=sub, y=y, tr=tr, va=va, te=te, n_nodes=len(uniq),
                n_sub=len(set(sub_keys)))

def diag(C):
    te, y = C['te'], C['y']
    return dict(n_nodes=C['n_nodes'], node_flow_ratio=C['n_nodes'] / len(y), n_subnets=C['n_sub'],
                nodes_per_subnet=C['n_nodes'] / max(C['n_sub'] - 1, 1),
                nmi_src=normalized_mutual_info_score(C['sid'][te], y[te]),
                nmi_dst=normalized_mutual_info_score(C['did'][te], y[te]),
                unk_rate_test=float(((C['sid'][te] == 0) | (C['did'][te] == 0)).mean()))

def rf_eval(C, view, seed):
    X = {'edge': lambda i: C['E'][i],
         'node': lambda i: np.hstack([C['X'][C['sid'][i]], C['X'][C['did'][i]]]),
         'edge+node': lambda i: np.hstack([C['E'][i], C['X'][C['sid'][i]], C['X'][C['did'][i]]])}[view]
    t0 = time.time()
    rf = RandomForestClassifier(RF_TREES, n_jobs=-1, random_state=seed)
    rf.fit(X(C['tr']), C['y'][C['tr']])
    p = rf.predict(X(C['te'])); yt = C['y'][C['te']]
    return dict(macro_f1=f1_score(yt, p, average='macro', zero_division=0), acc=accuracy_score(yt, p),
                per_class=f1_score(yt, p, average=None, labels=np.arange(C['y'].max() + 1), zero_division=0),
                time_s=round(time.time() - t0, 1))

# %%
# ═══ D-HIERARCHICAL ARCHITECTURE ═════════════════════════════════
class GCTLayerResNorm(nn.Module):
    def __init__(self, i_n, i_e, o):
        super().__init__()
        self.node_proj, self.edge_proj = Linear(i_n, o), Linear(i_e, o)
        self._attn = nn.Sequential(Linear(3 * o, 1, bias=False), nn.LeakyReLU(0.2))
        self._edge_update = nn.Sequential(Linear(3 * o, o), nn.ReLU())
        self._res_node = Linear(i_n, o) if i_n != o else nn.Identity()
        self._res_edge = Linear(i_e, o) if i_e != o else nn.Identity()
        self._norm_node, self._norm_edge = nn.LayerNorm(o), nn.LayerNorm(o)

    def forward(self, x, ei, ea):
        xp, ep = self.node_proj(x), self.edge_proj(ea)
        r, c = ei; xi, xj = xp[r], xp[c]
        a = pyg_softmax(self._attn(torch.cat([xi, xj, ep], -1)), r, num_nodes=xp.size(0))
        xo = torch.zeros_like(xp).index_add_(0, r, a * (xj + ep))
        return (self._norm_node(F.relu(xo) + self._res_node(x)),
                self._norm_edge(self._edge_update(torch.cat([xi, xj, ep], -1)) + self._res_edge(ea)))


class EdgeAwareLSTMResNorm(MessagePassing):
    def __init__(self, n_in, e_in, H):
        super().__init__(aggr='add')
        self.node_proj, self.edge_proj = Linear(n_in, H), Linear(e_in, H)
        self.rnn, self._norm = LSTM(H, H), nn.LayerNorm(H)

    def forward(self, x, ei, ea):
        return self.propagate(ei, x=self.node_proj(x), edge_attr=self.edge_proj(ea))

    def message(self, x_j, edge_attr):
        return torch.tanh(x_j + edge_attr)

    def update(self, aggr, x):
        o, _ = self.rnn(aggr.unsqueeze(0))
        return self._norm(o.squeeze(0) + x)


class SubnetAggregation(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.subnet_gnn = nn.Sequential(Linear(H, H), nn.GELU(), nn.LayerNorm(H), Linear(H, H), nn.LayerNorm(H))
        self.broadcast = nn.Sequential(Linear(2 * H, H), nn.GELU(), nn.LayerNorm(H))

    def forward(self, x, n2s):
        ns = int(n2s.max()) + 1
        se = x.new_zeros(ns, x.size(1)).index_add_(0, n2s, x)
        cnt = x.new_zeros(ns, 1).index_add_(0, n2s, x.new_ones(x.size(0), 1))
        return self.broadcast(torch.cat([x, self.subnet_gnn(se / (cnt + 1e-8))[n2s]], -1))


class DHier(nn.Module):
    """GCTLayerResNorm x2 (+SubnetAgg) + EdgeAwareLSTMResNorm + MLP classifier head on [x_src, e]."""
    def __init__(self, nd, ed, H, C, subnet=True):
        super().__init__()
        self.l1, self.l2 = GCTLayerResNorm(nd, ed, H), GCTLayerResNorm(H, H, H)
        self.sa = SubnetAggregation(H) if subnet else None
        self.lstm = EdgeAwareLSTMResNorm(H, H, H)
        self.clf = nn.Sequential(Linear(2 * H, H), nn.ReLU(), nn.Dropout(0.3), Linear(H, H // 2), nn.ReLU(),
                                 Linear(H // 2, C))

    def forward(self, x, ei, ea, n2s):
        x, ea = self.l1(x, ei, ea)
        if self.sa is not None:
            x = self.sa(x, n2s)
        x, ea = self.l2(x, ei, ea)
        x = self.lstm(x, ei, ea)
        return self.clf(torch.cat([x[ei[0]], ea], -1))


# %%
# ═══ EVALUATION CONFIGURATION ════════════════════════════════════
# Inductive protocol: node tables and scalers fitted strictly on training flows.
SPLITS_BY_DS = {'InSDN': [0, 1, 2, 3, 4], 'UNSW-NB15': [0, 1, 2], 'CICIDS2017': [0, 1, 2]}
OOF_FOLDS   = 2                # hybrid: 2-fold out-of-fold cross-fitting within training split
MIN_SUPPORT = 10               # macro_f1_ge10: classes with test support >= 10
CAL_GRID    = np.arange(-4.0, 4.0001, 0.25)
CAL_ROUNDS  = 3
LGB_TREES   = 600
if QUICK:
    SPLITS_BY_DS = {k: [0] for k in SPLITS_BY_DS}
    LGB_TREES = 30

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception as e:                                   # fallback if lightgbm is unavailable locally
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGB = False
    print('!! lightgbm not found -> using HistGradientBoosting fallback:', type(e).__name__)

def lgb_fit_proba(Xtr, ytr, Xva, yva, Xs, seed, nC):
    """LightGBM with early stopping strictly on the validation set."""
    if HAS_LGB:
        m = lgb.LGBMClassifier(n_estimators=LGB_TREES, learning_rate=0.05, num_leaves=63, min_child_samples=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                               objective='multiclass', num_class=nC, random_state=seed, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
        P = [m.predict_proba(X, num_iteration=m.best_iteration_) for X in Xs]
    else:
        m = HistGradientBoostingClassifier(max_iter=LGB_TREES, random_state=seed).fit(Xtr, ytr)
        P = [m.predict_proba(X) for X in Xs]
    return [_full_proba(p, m.classes_, nC) for p in P]

def _full_proba(p, cls, nC):
    out = np.zeros((len(p), nC), np.float64); out[:, np.asarray(cls, int)] = p
    return out

def rf_fit_proba(Xtr, ytr, Xs, seed, nC):
    m = RandomForestClassifier(RF_TREES, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    return [_full_proba(m.predict_proba(X), m.classes_, nC) for X in Xs]

def feats(C, view, idx):
    if view == 'edge':
        return C['E'][idx]
    return np.hstack([C['E'][idx], C['X'][C['sid'][idx]], C['X'][C['did'][idx]]])

# %%
# ═══ METRICS & VALIDATION LOGIT CALIBRATION ══════════════════════
def _f1_vec(yt, p, nC):
    cm = np.bincount(yt * nC + p, minlength=nC * nC).reshape(nC, nC)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros(nC), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(nC), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(nC), where=(prec + rec) > 0)
    return f1, prec, rec, cm.sum(1)

def metrics(yt, p, nC):
    f1, pr, rc, sup = _f1_vec(yt, p, nC)
    present = sup > 0                                        # classes with positive support
    big = sup >= MIN_SUPPORT
    return dict(macro_f1=f1[present].mean(), macro_f1_ge10=f1[big].mean(), macro_p=pr[present].mean(),
                macro_r=rc[present].mean(), acc=float((yt == p).mean()), per_class=f1)

def calibrate(Pva, yva, nC):
    """Per-class additive logit offset tuned via coordinate ascent on validation Macro-F1."""
    L = np.log(Pva + 1e-9); b = np.zeros(nC)
    best = _f1_vec(yva, (L + b).argmax(1), nC)[0].mean()
    for _ in range(CAL_ROUNDS):
        improved = False
        for c in range(nC):
            for v in CAL_GRID:
                bb = b.copy(); bb[c] = v
                f = _f1_vec(yva, (L + bb).argmax(1), nC)[0].mean()
                if f > best + 1e-9:
                    best, b, improved = f, bb, True
        if not improved:
            break
    return b, best

def score_both(name, Pva, Pte, C, nC, extra):
    """Evaluate both raw and validation-calibrated predictions on the test split."""
    yva, yte = C['y'][C['va']], C['y'][C['te']]
    b, vcal = calibrate(Pva, yva, nC)
    vraw = _f1_vec(yva, Pva.argmax(1), nC)[0].mean()
    rows = []
    for tag, P, vf in (('raw', Pte, vraw), ('cal', np.log(Pte + 1e-9) + b, vcal)):
        m = metrics(yte, P.argmax(1), nC)
        rows.append(dict(model=name, calib=tag, val_macro_f1=vf, bias=' '.join(f'{x:+.2f}' for x in b)
                         if tag == 'cal' else '', **extra, **m))
    return rows

# %%
# ═══ D-HIERARCHICAL TRAINING & PROBABILITY PREDICTION ════════════
def gnn_fit(C, seed, subnet=True, pred_sets=None):
    """Train on C['tr'] with early stopping on C['va'] Macro-F1; return softmax probabilities for pred_sets."""
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.tensor(C['X'], device=device); E = torch.tensor(C['E'], device=device)
    S = torch.tensor(C['sid'], device=device); Dd = torch.tensor(C['did'], device=device)
    SUB = torch.tensor(C['sub'], device=device); Y = torch.tensor(C['y'], device=device)
    nC = int(C['y'].max()) + 1
    model = DHier(X.shape[1], E.shape[1], HIDDEN, nC, subnet).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR / 100)
    crit = nn.CrossEntropyLoss()

    def batch(idx):
        s, d = S[idx], Dd[idx]
        u, inv = torch.unique(torch.cat([s, d]), return_inverse=True)
        ei = torch.stack([inv[:len(idx)], inv[len(idx):]])
        return model(X[u], ei, E[idx], SUB[u])

    @torch.no_grad()
    def proba(idx_np):
        model.eval(); out = []
        idx = torch.tensor(np.asarray(idx_np), device=device)
        for i in range(0, len(idx), BATCH):
            out.append(torch.softmax(batch(idx[i:i + BATCH]).float(), 1).cpu())
        return torch.cat(out).numpy().astype(np.float64)

    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    tr = torch.tensor(C['tr'])
    yva = C['y'][C['va']]
    best, best_ep, best_st, bad = -1, 0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g)].to(device)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            crit(batch(b), Y[b]).backward(); opt.step()
        sched.step()
        vf = _f1_vec(yva, proba(C['va']).argmax(1), nC)[0].mean()
        if vf > best:
            best, best_ep, bad = vf, ep, 0
            best_st = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_st)
    out = {k: proba(v) for k, v in (pred_sets or {}).items()}
    info = dict(best_ep=best_ep, gnn_val_f1=best, time_s=round(time.time() - t0, 1))
    del model, X, E
    return out, info

# %% [markdown]
# ## Multi-Seed Inductive Evaluation, Validation Calibration, and Hybrid GNN->LightGBM
#
# For each stratified split (inductive `split-before-graph` protocol):
# * **Tabular Baselines:** Random Forest and LightGBM on `edge` and `edge+node` feature views.
# * **D-Hierarchical:** Evaluated with composite synthetic keys (`D-Hier`) and, on InSDN, with true IP endpoints and `/24` SubnetAggregation (`D-Hier IP+SubnetAgg`).
# * **Hybrid (`Hybrid GNN->LGBM`):** 2-fold cross-fitting inside the training split (each fold constructs its node table strictly from its own training half). LightGBM is trained on `[edge + node + OOF GNN probabilities]`.
# * All models are reported both **raw** and **validation-calibrated** (`cal`).

# %%
RUN_ROWS = []

def v2_save():
    if not RUN_ROWS:
        return
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'per_class'} for r in RUN_ROWS])
    df.to_csv(os.path.join(OUT_DIR, 'v2_runs.csv'), index=False)
    pc = []
    for r in RUN_ROWS:
        for c, f in zip(r['classes'], r['per_class']):
            pc.append(dict(dataset=r['dataset'], split=r['split'], model=r['model'], calib=r['calib'],
                           cls=c, f1=f))
    pd.DataFrame(pc).to_csv(os.path.join(OUT_DIR, 'v2_perclass.csv'), index=False)

def add(rows, D, s):
    for r in rows:
        r.update(dataset=D['name'], split=s, classes=list(D['classes']))
        RUN_ROWS.append(r)
        print(f'  s{s} {r["model"]:<22} {r["calib"]:<3} MacroF1={r["macro_f1"]:.4f} '
              f'(>=10: {r["macro_f1_ge10"]:.4f}) P={r["macro_p"]:.3f} R={r["macro_r"]:.3f} '
              f'| val={r["val_macro_f1"]:.4f}')
    v2_save()

def run_split(D, s):
    nC = len(D['classes'])
    sp = splits_for(D, s)
    tr, va, te = sp
    C = build_cell(D, sp, 0, 0, scope='paper')
    print(f'\n--- {D["name"]} split {s}: train={len(tr):,} val={len(va):,} test={len(te):,} '
          f'nodes={C["n_nodes"]:,} UNK(test)={diag(C)["unk_rate_test"]:.3f}')

    # (1) Tabular baselines (RF & LightGBM)
    for view in ('edge', 'edge+node'):
        Xtr, Xva, Xte = feats(C, view, tr), feats(C, view, va), feats(C, view, te)
        ytr, yva = C['y'][tr], C['y'][va]
        t0 = time.time()
        r = safe(rf_fit_proba, Xtr, ytr, [Xva, Xte], s, nC)
        if r:
            add(score_both(f'RF {view}', r[0], r[1], C, nC, dict(time_s=round(time.time() - t0, 1))), D, s)
        t0 = time.time()
        r = safe(lgb_fit_proba, Xtr, ytr, Xva, yva, [Xva, Xte], s, nC)
        if r:
            add(score_both(f'LGBM {view}', r[0], r[1], C, nC, dict(time_s=round(time.time() - t0, 1))), D, s)
        del Xtr, Xva, Xte; gc.collect()

    # (2) D-Hierarchical (synthetic composite key, inductive)
    r = safe(gnn_fit, C, s, True, {'va': va, 'te': te})
    if r:
        P, info = r
        add(score_both('D-Hier', P['va'], P['te'], C, nC, info), D, s)

    # (3) InSDN: true IP endpoints + /24 SubnetAggregation
    if D['name'] == 'InSDN':
        Cip = safe(build_cell, D, sp, 0, 0, scope='ip')
        if Cip is not None:
            r = safe(gnn_fit, Cip, s, True, {'va': va, 'te': te})
            if r:
                P, info = r
                add(score_both('D-Hier IP+SubnetAgg', P['va'], P['te'], Cip, nC, info), D, s)
            del Cip; gc.collect()

    # (4) Hybrid: 2-fold out-of-fold GNN probabilities -> LightGBM
    t0 = time.time()
    skf = StratifiedKFold(OOF_FOLDS, shuffle=True, random_state=s)
    oof = np.zeros((len(tr), nC)); Pva = np.zeros((len(va), nC)); Pte = np.zeros((len(te), nC))
    ok = True
    for k, (a, b) in enumerate(skf.split(tr, C['y'][tr])):
        A, B = tr[a], tr[b]
        Ck = safe(build_cell, D, (A, va, B), 0, 0, scope='paper')     # node table constructed strictly from fold A
        r = safe(gnn_fit, Ck, s * 10 + k, True, {'B': B, 'va': va, 'te': te}) if Ck is not None else None
        if r is None:
            ok = False; break
        P, _ = r
        oof[b] = P['B']; Pva += P['va'] / OOF_FOLDS; Pte += P['te'] / OOF_FOLDS
        del Ck; gc.collect()
    if ok:
        # report OOF-averaged GNN standalone alongside Hybrid GNN->LGBM
        add(score_both('D-Hier OOF-avg', Pva, Pte, C, nC, dict(time_s=round(time.time() - t0, 1))), D, s)
        Xtr = np.hstack([feats(C, 'edge+node', tr), oof])
        Xva = np.hstack([feats(C, 'edge+node', va), Pva])
        Xte = np.hstack([feats(C, 'edge+node', te), Pte])
        r = safe(lgb_fit_proba, Xtr, C['y'][tr], Xva, C['y'][va], [Xva, Xte], s, nC)
        if r:
            add(score_both('Hybrid GNN->LGBM', r[0], r[1], C, nC, dict(time_s=round(time.time() - t0, 1))), D, s)
        del Xtr, Xva, Xte
    del C; gc.collect()

for D in DATA:
    print(f'\n{"#" * 90}\n# {D["name"]}\n{"#" * 90}')
    for s in SPLITS_BY_DS.get(D['name'], [0, 1, 2]):
        if time_left() < 0:
            print('!! time budget reached'); break
        safe(run_split, D, s)

# %% [markdown]
# ## Summary & Paired Statistical Comparisons
# * **H5a:** Validation calibration improves D-Hierarchical (`cal − raw > 0`).
# * **H5b:** Calibrated Hybrid vs. best validation-selected calibrated tree model.
# * **H5c:** LightGBM `edge+node` vs. `edge` (`raw`) — contribution of graph node features under inductive evaluation.

# %%
def paired(a, b):
    """Paired difference (b - a) across identical splits with 95% t-CI and paired t-test p-value."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if len(d) < 2:
        return d.mean() if len(d) else np.nan, np.nan, np.nan, np.nan
    se = d.std(ddof=1) / math.sqrt(len(d))
    if se == 0:
        return d.mean(), d.mean(), d.mean(), (1.0 if d.mean() == 0 else 0.0)
    h = st.t.ppf(0.975, len(d) - 1) * se
    return d.mean(), d.mean() - h, d.mean() + h, st.ttest_rel(b, a).pvalue

def verdict(m, lo):
    if not np.isfinite(m) or m <= 0:
        return 'not_supported'
    return 'supported' if (np.isfinite(lo) and lo > 0) else 'inconclusive'

runs = pd.DataFrame([{k: v for k, v in r.items() if k not in ('per_class', 'classes')} for r in RUN_ROWS])
if len(runs):
    summ = runs.groupby(['dataset', 'model', 'calib']).agg(
        macro_f1=('macro_f1', 'mean'), std=('macro_f1', 'std'), macro_f1_ge10=('macro_f1_ge10', 'mean'),
        macro_p=('macro_p', 'mean'), macro_r=('macro_r', 'mean'), val_macro_f1=('val_macro_f1', 'mean'),
        n=('macro_f1', 'count')).reset_index()
    summ.to_csv(os.path.join(OUT_DIR, 'v2_summary.csv'), index=False)
    print('\nSUMMARY (mean across splits)\n')
    print(summ.pivot_table(index=['dataset', 'model'], columns='calib',
                           values=['macro_f1', 'macro_f1_ge10']).round(4).to_string())

    TREES = ['RF edge', 'RF edge+node', 'LGBM edge', 'LGBM edge+node']
    tests = []
    def get(ds, model, calib, col='macro_f1'):
        g = runs[(runs.dataset == ds) & (runs.model == model) & (runs.calib == calib)]
        return g.set_index('split')[col]
    for ds in runs.dataset.unique():
        for col in ('macro_f1', 'macro_f1_ge10'):
            # H5a
            a, b = get(ds, 'D-Hier', 'raw', col), get(ds, 'D-Hier', 'cal', col)
            ix = a.index.intersection(b.index)
            m, lo, hi, p = paired(a[ix], b[ix])
            tests.append(dict(dataset=ds, metric=col, hyp='H5a', compare='D-Hier cal − raw', delta=m, lo=lo,
                              hi=hi, p=p, n=len(ix), verdict=verdict(m, lo)))
            # H5b: best calibrated tree selected strictly via validation mean
            vals = {t: runs[(runs.dataset == ds) & (runs.model == t) & (runs.calib == 'cal')].val_macro_f1.mean()
                    for t in TREES}
            vals = {t: v for t, v in vals.items() if np.isfinite(v)}
            if vals:
                bt = max(vals, key=vals.get)
                a, b = get(ds, bt, 'cal', col), get(ds, 'Hybrid GNN->LGBM', 'cal', col)
                ix = a.index.intersection(b.index)
                m, lo, hi, p = paired(a[ix], b[ix])
                tests.append(dict(dataset=ds, metric=col, hyp='H5b', compare=f'Hybrid cal − {bt} cal (val-selected)',
                                  delta=m, lo=lo, hi=hi, p=p, n=len(ix), verdict=verdict(m, lo)))
            # H5c
            a, b = get(ds, 'LGBM edge', 'raw', col), get(ds, 'LGBM edge+node', 'raw', col)
            ix = a.index.intersection(b.index)
            m, lo, hi, p = paired(a[ix], b[ix])
            tests.append(dict(dataset=ds, metric=col, hyp='H5c', compare='LGBM edge+node − edge (raw)', delta=m,
                              lo=lo, hi=hi, p=p, n=len(ix), verdict=verdict(m, lo)))
            # descriptive comparison: D-Hier cal vs best calibrated tree
            if vals:
                a, b = get(ds, bt, 'cal', col), get(ds, 'D-Hier', 'cal', col)
                ix = a.index.intersection(b.index)
                m, lo, hi, p = paired(a[ix], b[ix])
                tests.append(dict(dataset=ds, metric=col, hyp='descriptive', compare=f'D-Hier cal − {bt} cal',
                                  delta=m, lo=lo, hi=hi, p=p, n=len(ix), verdict=''))
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(OUT_DIR, 'v2_tests.csv'), index=False)
    print('\nPAIRED STATISTICAL TESTS (across splits)\n')
    print(tests.round(4).to_string(index=False))

print(f'\nTotal runtime: {(time.time() - T_START) / 3600:.2f} hours')
