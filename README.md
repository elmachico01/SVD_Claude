# SVD & YOLO per la Steganografia di Immagini

**Progetto d'esame — *Statistical and Mathematical Methods for AI*** (Prof. M. Popolizio)

Steganografia di immagini basata sulla **Decomposizione ai Valori Singolari (SVD)**,
con embedding **guidato da YOLO**: l'informazione segreta viene nascosta nelle
regioni di *sfondo* dell'immagine (lontano dagli oggetti rilevati), così da
restare impercettibile all'occhio **e** invisibile a un rilevatore di oggetti a
valle.

La SVD compare in **due punti**, entrambi presi direttamente dalle slide del corso:

1. **Compressione del segreto — SVD troncata (Eckart–Young).** L'immagine segreta
   è sostituita dalla sua migliore approssimazione di rango `k` e memorizzata nei
   fattori compatti `(U_k, σ_k, V_kᵀ)`. È esattamente la *riduzione di
   dimensionalità* / compressione vista a lezione: il payload passa da `m·n` a
   `k(m+n+1)` numeri.
2. **Embedding nel cover — SVD a blocchi + QIM.** Il cover è diviso in blocchi
   `B×B`; per ogni blocco `A = UΣVᵀ` si quantizza il valore singolare massimo
   `σ₁` (Quantisation Index Modulation) per nascondere un bit. `σ₁` concentra
   quasi tutta l'energia del blocco (Eckart–Young), quindi è il posto più stabile
   dove nascondere informazione.

YOLOv8 ha **due ruoli**: (a) costruisce una *mappa di priorità* che spinge i bit
nello sfondo proteggendo gli oggetti; (b) serve a **valutare** quanto il
contenuto semantico sopravvive all'embedding (preservazione delle detection,
IoU, mAP@0.5).

---

## 1. Struttura del progetto

```
SVD_Claude/
├── src/                       # libreria
│   ├── svd_core.py            # matematica SVD (teoria del corso)
│   ├── steganography.py       # compressione segreto + embedding/estrazione
│   ├── yolo_guidance.py       # YOLOv8: guida embedding + metriche detection
│   ├── metrics.py             # PSNR, SSIM, MSE, NC, BER, capacità
│   ├── attacks.py             # JPEG, rumore, blur, rescale, …
│   ├── dataset.py             # caricamento COCO-128 + immagine segreta
│   └── pipeline.py            # orchestrazione end-to-end + config HC/RB
├── experiments/
│   ├── run_experiments.py     # batch su tutte le 128 immagini → CSV + JSON
│   └── make_figures.py        # genera tutte le figure
├── scripts/
│   ├── demo.py                # demo su singola immagine
│   ├── generate_report.py     # report (Markdown + PDF) dai dati
│   └── generate_slides.py     # presentazione PowerPoint dai dati
├── docs/
│   └── THEORY.md              # background matematico (mappato sulle slide)
├── results/                   # output: CSV, JSON, figure, esempi
├── data/                      # COCO-128 (scaricato in automatico)
└── requirements.txt
```

---

## 2. Installazione (locale)

Serve **Python 3.10+**. Si consiglia un ambiente virtuale.

```bash
# 1. clona ed entra nella cartella
git clone <repo-url> && cd SVD_Claude

# 2. ambiente virtuale
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. dipendenze
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota su PyTorch / YOLO.** `ultralytics` installa PyTorch automaticamente.
> Su CPU va benissimo (il progetto è pensato per girare senza GPU). I pesi
> `yolov8n.pt` (~6 MB) vengono scaricati al primo utilizzo.
>
> **Nota sul dataset.** COCO-128 (128 immagini reali di COCO) viene scaricato
> automaticamente dal mirror GitHub ufficiale. Se la rete è dietro proxy e il
> download fallisce, scarica a mano
> `https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip`
> e scompattalo in `data/` (deve risultare `data/coco128/images/train2017/*.jpg`).
>
> Se YOLO non fosse disponibile nel tuo ambiente, il codice **ripiega**
> automaticamente su una saliency classica (OpenCV) così la pipeline gira
> comunque (le metriche di detection vengono semplicemente disattivate).

---

## 3. Come eseguirlo

### 3.1 Demo su una singola immagine

```bash
python scripts/demo.py --index 2 --mode hc                 # alta capacità
python scripts/demo.py --index 2 --mode rb --attack jpeg_q50   # modalità robusta
```

Stampa le metriche e salva una figura riassuntiva in `results/examples/`.

### 3.2 Esperimento completo su ≥100 immagini (COCO-128)

```bash
python experiments/run_experiments.py            # tutte e 128 le immagini
# oppure un sottoinsieme:
python experiments/run_experiments.py --limit 100
```

Tempo indicativo: ~10–15 minuti su CPU per 128 immagini. Produce:

| File | Contenuto |
|------|-----------|
| `results/metrics_per_image.csv` | una riga per immagine, **tutte** le metriche (HC + RB) |
| `results/summary.json` | statistiche aggregate (medie, mediane, per-attacco) |
| `results/secret_compression.csv` | qualità della SVD troncata del segreto al variare di `k` |

A fine run viene stampata a schermo una tabella riassuntiva.

### 3.3 Genera le figure

```bash
python experiments/make_figures.py     # legge results/ e scrive results/figures/
```

---

## 4. Le due modalità operative (operating points)

| | **High-Capacity (HC)** | **Robust (RB)** |
|---|---|---|
| blocco SVD | 8×8 | 16×16 |
| passo QIM `Δ` | 24 | 128 |
| ripetizione | nessuna | ×3 (voto di maggioranza) |
| segreto | 64×64, rango k=10 | 16×16, rango k=3 |
| obiettivo | massima capacità + impercettibilità | resistenza agli attacchi |

Le due modalità illustrano il **compromesso capacità ↔ robustezza** governato
dalla concentrazione di energia nei valori singolari (Eckart–Young): blocchi più
grandi concentrano più energia in `σ₁` ⇒ più robustezza, ma meno capacità; la
compressione SVD massimizza la capacità ma elimina la ridondanza ⇒ più fragilità.

---

## 5. Report e PowerPoint (dai TUOI dati)

Dopo aver eseguito gli esperimenti in locale, i deliverable si rigenerano con i
**tuoi** numeri:

```bash
python scripts/generate_report.py      # → docs/REPORT.md  e  docs/REPORT.pdf
python scripts/generate_slides.py      # → docs/presentazione.pptx
```

Entrambi leggono `results/summary.json`, `results/metrics_per_image.csv` e le
figure in `results/figures/`. **Per la consegna**: esegui §3.2 e §3.3, poi §5.

---

## 6. Mappatura con il programma del corso

| Concetto della lezione | Dove nel progetto |
|---|---|
| SVD `A = UΣVᵀ` | `svd_core.svd_decompose` |
| SVD troncata / Eckart–Young | `svd_core.truncated_svd`, `eckart_young_errors`; compressione segreto |
| Fattore di compressione `k(m+n+1)/mn` | `svd_core.compression_factor` |
| Criterio dell'energia `E(k)` per la scelta di `k` | `svd_core.energy_ratio`, `rank_for_energy`; `THEORY.md` §8.3 |
| SVD ↔ autovalori `σ_i=√λ_i` | `svd_core.verify_svd_eigendecomposition` |
| Numero di condizionamento `κ=σ₁/σₙ` | `svd_core.condition_number` |
| Piccoli σ amplificano il rumore → embedding in σ₁ | `THEORY.md` §5, §6.1; `steganography._qim_embed_sigma1` |
| Perturbazione dei valori singolari (teorema di Weyl) | `THEORY.md` §7; esperimenti di robustezza |
| Pseudoinversa / minimi quadrati | `svd_core.pseudoinverse`, `lstsq_via_svd` |

Vedi `docs/THEORY.md` per il dettaglio matematico. In particolare: §6 (embedding
e decodifica passo-passo, ruolo dei valori singolari), §7 (effetto delle
perturbazioni su decodifica e ricostruzione, teorema di Weyl), §8 (motivazione
teorica di tutti i parametri `Δ`, `B`, `k`, `R`, con PSNR previsto vs misurato).
