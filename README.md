

SETUP ŚRODOWISKA (Windows / PowerShell)

Utworzenie środowiska venv (Python 3.12)

uv venv --python 3.12

Aktywacja środowiska

.venv\Scripts\activate

Instalacja zależności

pip install -r .\requirements.txt

2. STRUKTURA DANYCH

Projekt zakłada dane Kaggle w formacie:

subjXX_seriesYY_data.csv
subjXX_seriesYY_events.csv

Przykładowa struktura katalogów:

data/
grasp-and-lift-eeg-detection/
train/
subj01_series01_data.csv
subj01_series01_events.csv
...

3. GŁÓWNA ANALIZA FAZOWA (MACIERZE POŁĄCZEŃ)

Batch dla całego zbioru train/

zapis wyników osobno dla każdego subjecta:

python -m scripts.mfg_kaggle_phase_matrix_batch ^
--root data/grasp-and-lift-eeg-detection/train ^
--out out/test ^
--mode gc ^
--metric energy ^
--lags-ms 0 50 100 150 200 300 400 500 ^
--m 4 ^
--group-by subject ^
--epoch-len-s 2.0

Wyniki:

out/test/<GROUP>/

*.png -> heatmapy macierzy połączeń

phase_top_edges_*.txt -> ranking najsilniejszych krawędzi

<GROUP>:

subjXX -> gdy --group-by subject

ALL -> gdy --group-by all

4. TRYB PCA DLA WSZYSTKICH PAR KANAŁÓW 

4.1 Budowa globalnej bazy PCA (all-pairs)

python -m scripts.build_kaggle_basis_allpairs_batch ^
--root data/grasp-and-lift-eeg-detection/train ^
--out out/basis_allpairs.npz ^
--mode gc ^
--epoch-len-s 2.0 ^
--m 4 ^
--lags-ms 0 50 100 150 200 ^
--pca-r 3

Powstaje:

out/basis_allpairs.npz

4.2 Analiza fazowa w trybie PCA

python -m scripts.mfg_kaggle_phase_matrix_batch ^
--root data/grasp-and-lift-eeg-detection/train ^
--out out/test_pca ^
--mode gc ^
--basis-allpairs out/basis_allpairs.npz ^
--pca-r 3 ^
--epoch-len-s 2.0 ^
--m 4 ^
--lags-ms 0 50 100 150 200 ^
--group-by subject

Wyniki PCA:

out/test_pca/<GROUP>/

phasegrid_*.png -> siatka PC x lag

phase_top_edges__pc_lag*ms.txt -> top-k krawędzie

5. META-ANALIZA MIĘDZY SUBJECTAMI

Liczy:

replikowalność między subjectami

test dwumianowy

opcjonalnie korekcję BH-FDR

Podstawowe uruchomienie:

python -m scripts.meta_analysis ^
--dir out/test_pca ^
--topk 10 ^
--min-subjects 4 ^
--mode gc

Z FDR:

python -m scripts.meta_analysis ^
--dir out/test_pca ^
--topk 10 ^
--min-subjects 4 ^
--mode gc ^
--use-fdr

6. NAJWAŻNIEJSZE ARGUMENTY CLI

mfg_kaggle_phase_matrix_batch:

--root -> katalog z danymi train/
--out -> katalog wynikowy
--mode -> gc lub corr
--epoch-len-s -> długość okna fazy (sekundy)
--m -> stopień baz Legendre
--lags-ms -> lista opóźnień w ms
--group-by -> all lub subject
--metric -> energy / chi2 / neglog10p (bez PCA)
--basis-allpairs -> ścieżka do .npz (włącza PCA)
--pca-r -> liczba komponentów PCA

7. SZYBKI TEST DZIAŁANIA (SMOKE TEST)

Uruchomienie na 2 plikach:

python -m scripts.mfg_kaggle_phase_matrix_batch ^
--root data/grasp-and-lift-eeg-detection/train ^
--out out/smoke ^
--mode gc ^
--metric energy ^
--lags-ms 0 50 100 ^
--m 4 ^
--epoch-len-s 2.0 ^
--group-by all ^
--max-files 2

Powinno utworzyć:

out/smoke/ALL/

z kilkoma plikami .png i .txt.
