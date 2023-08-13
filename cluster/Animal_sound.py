import pandas as pd
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import  pyplot as plt
from glob import glob
import warnings
warnings.filterwarnings("ignore")
audio_files = glob('audio/*.wav')
mfccs =[]
spectral_centroid =[]
l = len(audio_files)

for i in range (l):
    animal, sr = librosa.load(audio_files[i])
    mfccs_anim = librosa.feature.mfcc(y=animal, sr=sr, n_mfcc=20)
    spectral_centroids_anims = librosa.feature.spectral_centroid(y=animal, sr=sr)
    mfccs.append(max(mfccs_anim[9]))
    spectral_centroid.append(max(max(spectral_centroids_anims)))
df = pd.DataFrame()
df['mfcc'] = mfccs
df['spectral'] = spectral_centroid
scaler = StandardScaler()
df[['mfcc_t','spectral_t']] =  scaler.fit_transform(df[['mfcc','spectral']])
KM = KMeans(n_clusters=2)
y_predict = KM.fit_predict(df[['mfcc','spectral']])
df['cluster'] = y_predict
print(df)
plt.scatter(df['mfcc_t'],df['spectral_t'],c=df['cluster'])
plt.grid()
plt.xlabel('MFCC')
plt.ylabel('SPECTRAL CENTROID')
plt.show()

# print(mfccs)
# print(spectral_centroid)