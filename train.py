import os
import datetime
import pickle
from pathlib import Path
import numpy as np
from socket import gethostname
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, f1_score, precision_score
import argparse
import logging

from hitf_model import HITF

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Path of the input data.')
parser.add_argument('--name', '-n', default='', type=str, help='Name of the experiment.')
parser.add_argument('--rank', default=50, help='Number of the latent factors (phenotypes).')
args = parser.parse_args()

R = args.rank

# exp_id
exp_id = args.name
if exp_id != '':
    exp_id = exp_id + '-'
out_path = './results/{exp_id}R{R}{date:%m%d}-{host}-{pid}'.format(exp_id=exp_id, R=R, date=datetime.date.today(), host=gethostname(), pid=os.getpid())
dump_file = os.path.join(out_path, 'dump')
if not os.path.isdir(out_path):
    os.makedirs(out_path)
# out_path.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(exp_id)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(os.path.join(out_path, 'train_projection.log'))
fh.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

DATA_PATH = args.data_path
train_ratio = 0.7

# load data
D = np.loadtxt(os.path.join(DATA_PATH, 'D.csv'), delimiter=',')
M = np.loadtxt(os.path.join(DATA_PATH, 'M.csv'), delimiter=',')
labels = np.loadtxt(os.path.join(DATA_PATH, 'labels.csv'), delimiter=',')

idx_train, idx_test, *_ = train_test_split(np.arange(labels.shape[0]), labels, train_size=train_ratio, random_state=75)
np.savez(os.path.join(out_path, 'idx'), idx_train=idx_train, idx_test=idx_test)

Dtrain, Mtrain = D[idx_train], M[idx_train]
Dtest, Mtest = D[idx_test], M[idx_test]


hitf = HITF(R=R, use_cuda=True, logger=logger)
hitf.decompose(M=Mtrain, Dprime=Dtrain)
test_proj = hitf.project(M=Mtest, Dprime=Dtest)

with open(os.path.join(out_path, 'training_iters.info'.format(R)), 'wb') as f:
    pickle.dump(hitf.iters_info, f)

np.savez(os.path.join(out_path, 'factors_and_projections'.format(R)),
         U1=hitf.U[0].cpu().numpy(),
         U2=hitf.U[1].cpu().numpy(),
         U3=hitf.U[2].cpu().numpy(),
         test_proj=test_proj.cpu().numpy())

LR = LogisticRegression(max_iter=1000)
LR.fit(hitf.U[0].cpu().numpy(), labels[idx_train])

pred = LR.predict(test_proj.cpu().numpy())
pred_prob = LR.predict_proba(test_proj.cpu().numpy())
print('f1: ', f1_score(labels[idx_test], pred))
print('recall: ', recall_score(labels[idx_test], pred))
print('precision: ', precision_score(labels[idx_test], pred))
fpr, tpr, thresholds = roc_curve(labels[idx_test], pred_prob[:, 1])
auc = roc_auc_score(labels[idx_test], pred_prob[:, 1])
print('AUC: ', auc)



