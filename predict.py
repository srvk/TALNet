import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import sys, os, os.path
import csv
import numpy
import cPickle
import librosa
import torch
import torch.nn as nn
from torch.autograd import Variable
from Net import Net
from scipy.io import savemat

def extract(wav):
    spec = librosa.core.stft(wav, n_fft = 4096,
                             hop_length = 400, win_length = 1024,
                             window = 'hann', center = True, pad_mode = 'constant')
    mel = librosa.feature.melspectrogram(S = numpy.abs(spec), sr = 16000, n_mels = 64, fmax = 8000)
    logmel = librosa.core.power_to_db(mel)
    return logmel.T.astype('float32')

def save_rttm(filename, frame_pred):
    nClasses = frame_pred.shape[1]
    z = numpy.zeros((nClasses, 1), dtype = 'bool')
    output = numpy.hstack([z, frame_pred.T, z])
    cls_ids, starts = (~output[:, :-1] & output[:, 1:]).nonzero()
    _, ends = (output[:, :-1] & ~output[:, 1:]).nonzero()

    FRAME_LEN = 0.1
    with open(path_prefix + '.rttm', 'w') as f:
        for cls, start, end in zip(cls_ids, starts, ends):
            f.write('SPEAKER\t%s\t1\t%.1f\t%.1f\t<NA>\t<NA>\t"%s"\t<NA>\t<NA>\n' % \
                (file_prefix, start * FRAME_LEN, (end - start) * FRAME_LEN, class_names[cls]))

# MAIN PROGRAM STARTS HERE
# Load class names
class_names = []
with open('class_labels_indices.csv', 'r') as f:
    f.readline()
    reader = csv.reader(f, delimiter = ',', quotechar = '"')
    for row in reader:
        class_id = int(row[0])
        class_names.append(row[2])

# Load model
class Object(object):
    pass
args = Object()
args.embedding_size = 1024
args.n_conv_layers = 10
args.n_pool_layers = 5
args.kernel_size_time = 3
args.kernel_size_freq = 3
args.gru = True
args.batch_norm = True
args.dropout = 0.0
args.pooling = 'lin'
model = Net(args)
if torch.cuda.is_available():
    model = model.cuda()
    state_dict = torch.load('model.pt')
else:
    state_dict = torch.load('model.pt', map_location = 'cpu')
model.load_state_dict(state_dict['model'])
model.eval()

# Load audio file and extract features
INPUT_AUDIO = sys.argv[1]
wav, _ = librosa.load(INPUT_AUDIO, sr = 16000, mono = True)
feat = extract(wav)
with open('normalizer.pkl', 'rb') as f:
    mu, sigma = cPickle.load(f)
feat = (feat - mu) / sigma

# Make predictions
input = Variable(torch.from_numpy(numpy.expand_dims(feat, 0).astype('float32')))
if torch.cuda.is_available():
    input = input.cuda()
frame_prob = model(input)[1].data.cpu().numpy()[0]
with open('thresholds.pkl', 'rb') as f:
    thres = cPickle.load(f)
frame_pred = frame_prob >= thres

# Save predictions
path_prefix = os.path.splitext(INPUT_AUDIO)[0]
file_prefix = os.path.basename(path_prefix)
savemat(path_prefix + '.frame_prob.mat', {'frame_prob': frame_prob})
save_rttm(path_prefix + '.rttm', frame_pred)
