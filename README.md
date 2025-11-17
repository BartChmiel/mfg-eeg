1. 
uv venv --python 3.12

2.
.venv/Scripts/activate

3.
pip install -r .\requirements.txt

4.
Uruchamianie analizy


Pojedyncza para kanałów (CSV lub MAT):
python -m scripts.mfg_pair --file <path> --chan-a <A> --chan-b <B> --mode gc --out <dir>

Przykład CSV:
python -m scripts.mfg_pair --file data/grasp-and-lift-eeg-detection/train/subj10_series1_data.csv --chan-a Fp1 --chan-b Fp2 --mode gc --out out/csv_run

Przykład MAT (ROI):
python -m scripts.mfg_pair --file data/mat/A_01_47_FLA_CONG_mfgin.mat --chan-a ROI_01 --chan-b ROI_02 --mode gc --out out/mat_run

Batch processing (wszystkie pliki w folderze):
python -m scripts.mfg_batch --root <root-folder> --chan-a <A> --chan-b <B> --mode gc --out-root <dir>

Przykład (wszystkie CSV):
python -m scripts.mfg_batch --root data/grasp-and-lift-eeg-detection/train --chan-a Fp1 --chan-b Fp2 --mode gc --out-root out/csv_run

Przykład (tylko 5 pierwszych plików):
python -m scripts.mfg_batch --root data/grasp-and-lift-eeg-detection/train --chan-a Fp1 --chan-b Fp2 --mode gc --max-files 5 --out-root out/csv_run

