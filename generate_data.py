import numpy as np
import pandas as pd
import os

print("=" * 55)
print("  Generating NSL-KDD style dataset")
print("=" * 55)

np.random.seed(42)
os.makedirs('data', exist_ok=True)

COLUMNS = [
    'duration','protocol_type','service','flag',
    'src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count',
    'serror_rate','srv_serror_rate','rerror_rate',
    'srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label'
]

def make_normal(n):
    d = {
        'duration':          np.random.exponential(0.5, n),
        'protocol_type':     np.random.choice([0,1,2], n, p=[0.6,0.3,0.1]),
        'service':           np.random.randint(0, 65, n),
        'flag':              np.random.choice([0,1,2,3], n, p=[0.7,0.15,0.1,0.05]),
        'src_bytes':         np.random.exponential(500, n),
        'dst_bytes':         np.random.exponential(1000, n),
        'land':              np.zeros(n),
        'wrong_fragment':    np.random.choice([0,1], n, p=[0.97,0.03]),
        'urgent':            np.zeros(n),
        'hot':               np.random.poisson(0.5, n),
        'num_failed_logins': np.random.choice([0,1], n, p=[0.95,0.05]),
        'logged_in':         np.random.choice([0,1], n, p=[0.3,0.7]),
        'num_compromised':   np.zeros(n),
        'root_shell':        np.zeros(n),
        'su_attempted':      np.zeros(n),
        'num_root':          np.zeros(n),
        'num_file_creations':np.random.poisson(0.2, n),
        'num_shells':        np.zeros(n),
        'num_access_files':  np.random.poisson(0.1, n),
        'num_outbound_cmds': np.zeros(n),
        'is_host_login':     np.zeros(n),
        'is_guest_login':    np.random.choice([0,1], n, p=[0.95,0.05]),
        'count':             np.random.randint(1, 100, n),
        'srv_count':         np.random.randint(1, 100, n),
        'serror_rate':       np.random.beta(1, 20, n),
        'srv_serror_rate':   np.random.beta(1, 20, n),
        'rerror_rate':       np.random.beta(1, 20, n),
        'srv_rerror_rate':   np.random.beta(1, 20, n),
        'same_srv_rate':     np.random.beta(8, 2, n),
        'diff_srv_rate':     np.random.beta(2, 8, n),
        'srv_diff_host_rate':np.random.beta(1, 10, n),
        'dst_host_count':    np.random.randint(100, 255, n),
        'dst_host_srv_count':np.random.randint(50, 255, n),
        'dst_host_same_srv_rate':     np.random.beta(7, 3, n),
        'dst_host_diff_srv_rate':     np.random.beta(2, 8, n),
        'dst_host_same_src_port_rate':np.random.beta(5, 5, n),
        'dst_host_srv_diff_host_rate':np.random.beta(1, 10, n),
        'dst_host_serror_rate':       np.random.beta(1, 20, n),
        'dst_host_srv_serror_rate':   np.random.beta(1, 20, n),
        'dst_host_rerror_rate':       np.random.beta(1, 20, n),
        'dst_host_srv_rerror_rate':   np.random.beta(1, 20, n),
        'label': np.full(n, 'Normal')
    }
    return pd.DataFrame(d)

def make_dos(n):
    df = make_normal(n)
    df['count']        = np.random.randint(200, 511, n)
    df['srv_count']    = np.random.randint(200, 511, n)
    df['serror_rate']  = np.random.beta(8, 2, n)
    df['src_bytes']    = np.random.exponential(50, n)
    df['dst_bytes']    = np.zeros(n)
    df['label']        = 'DoS'
    return df

def make_probe(n):
    df = make_normal(n)
    df['diff_srv_rate']          = np.random.beta(8, 2, n)
    df['dst_host_diff_srv_rate'] = np.random.beta(7, 3, n)
    df['count']                  = np.random.randint(100, 511, n)
    df['label']                  = 'Probe'
    return df

def make_r2l(n):
    df = make_normal(n)
    df['num_failed_logins'] = np.random.randint(3, 10, n)
    df['is_guest_login']    = np.ones(n)
    df['rerror_rate']       = np.random.beta(6, 4, n)
    df['label']             = 'R2L'
    return df

def make_u2r(n):
    df = make_normal(n)
    df['root_shell']          = np.ones(n)
    df['su_attempted']        = np.ones(n)
    df['num_root']            = np.random.randint(1, 5, n)
    df['num_file_creations']  = np.random.randint(3, 15, n)
    df['label']               = 'U2R'
    return df

# Generate 50,000 training samples
print("\n[1/3] Generating training samples...")
train = pd.concat([
    make_normal(27500),
    make_dos(9000),
    make_probe(6000),
    make_r2l(4500),
    make_u2r(3000),
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Generate 10,000 test samples
print("[2/3] Generating test samples...")
test = pd.concat([
    make_normal(5500),
    make_dos(2000),
    make_probe(1200),
    make_r2l(800),
    make_u2r(500),
], ignore_index=True).sample(frac=1, random_state=99).reset_index(drop=True)

# Save to CSV
print("[3/3] Saving to data folder...")
train.to_csv('data/KDDTrain+.csv', index=False)
test.to_csv('data/KDDTest+.csv',   index=False)

print("\n" + "=" * 55)
print("  Dataset Generated!")
print(f"  Train : {len(train):,} samples, {len(train.columns)-1} features")
print(f"  Test  : {len(test):,} samples")
print("\n  Label distribution (train):")
for label, count in train['label'].value_counts().items():
    pct = count / len(train) * 100
    bar = '█' * int(pct / 3)
    print(f"    {label:8s} {count:6,}  {bar} {pct:.1f}%")
print("\n  Files saved:")
print("    data/KDDTrain+.csv")
print("    data/KDDTest+.csv")
print("\n  Day 2 COMPLETE — ready to train models!")
print("=" * 55)