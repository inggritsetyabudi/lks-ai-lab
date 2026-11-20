"""Dipakai SETELAH membuat submission; lihat PETUNJUK_DAN_SOAL.md."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

base=Path(__file__).resolve().parent
submission=pd.read_csv(base/'submission.csv',dtype={'id_laporan':str})
key_path=base/'KUNCI_TEST_LKS_BUKA_SETELAH_SELESAI.csv'
if not key_path.exists():
    raise FileNotFoundError('Salin file kunci yang diunduh terpisah ke folder ini SETELAH membuat submission.csv')
truth=pd.read_csv(key_path,dtype={'id_laporan':str})
assert submission.columns.tolist()==['id_laporan','kategori'], 'Kolom harus id_laporan,kategori'
assert submission.id_laporan.is_unique and truth.id_laporan.is_unique, 'ID duplikat'
assert set(submission.id_laporan)==set(truth.id_laporan), 'ID submission dan test tidak cocok'
df=truth.merge(submission,on='id_laporan',validate='one_to_one',suffixes=('_true','_pred'))
y=df.kategori_true; p=df.kategori_pred
print('Accuracy :',round(accuracy_score(y,p),4))
print('Macro-F1 :',round(f1_score(y,p,average='macro'),4))
print('Weighted-F1 :',round(f1_score(y,p,average='weighted'),4))
print(classification_report(y,p,zero_division=0))
print(pd.DataFrame(confusion_matrix(y,p,labels=sorted(y.unique())),index=sorted(y.unique()),columns=sorted(y.unique())))
