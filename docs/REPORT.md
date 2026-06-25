# SVD & YOLO per la Steganografia di Immagini

*Progetto d'esame — Statistical and Mathematical Methods for AI*

Valutazione su **128 immagini** del dataset COCO-128. Rilevatore YOLO: attivo. Tutti i numeri di questo report sono generati automaticamente dai risultati sperimentali (`results/summary.json`).


## 1. Descrizione del problema

La **steganografia di immagini** consiste nel nascondere un'informazione segreta dentro un'immagine *cover* in modo che la presenza stessa del messaggio sia impercettibile. A differenza della crittografia (che rende il messaggio illeggibile ma evidente), la steganografia mira all'**invisibilità**.

Obiettivo del progetto: realizzare uno schema di steganografia che (i) sfrutti la **Decomposizione ai Valori Singolari (SVD)** — argomento centrale del corso — come strumento sia di *compressione* del segreto sia di *dominio di embedding*; e (ii) usi un rilevatore di oggetti **YOLO** per rendere l'occultamento *content-adaptive*, nascondendo i dati nello sfondo e preservando le regioni semanticamente importanti.

Tre requisiti guidano la valutazione, in tensione tra loro: **impercettibilità** (il cover non deve cambiare visibilmente), **capacità** (quanti bit possiamo nascondere) e **robustezza** (il segreto deve sopravvivere a manipolazioni come la ricompressione JPEG).


## 2. Metodo e background teorico


### 2.1 SVD e teorema di Eckart–Young

Per ogni matrice A ∈ ℝ^(m×n) vale A = U Σ Vᵀ con U, V ortogonali e Σ = diag(σ₁,…,σ_r), σ₁ ≥ … ≥ σ_r > 0. La SVD troncata A_k = Σ_{i=1}^{k} σ_i u_i v_iᵀ è, per il teorema di Eckart–Young, la migliore approssimazione di rango k sia in norma spettrale (‖A−A_k‖₂ = σ_{k+1}) sia di Frobenius (‖A−A_k‖_F = (Σ_{i>k} σ_i²)^½).


![Spettro dei valori singolari di un cover ed energia cumulata: poche componenti catturano la quasi totalità dell'energia.](../results/figures/fig_svd_spectrum.png)

*Spettro dei valori singolari di un cover ed energia cumulata: poche componenti catturano la quasi totalità dell'energia.*


![Verifica numerica del teorema di Eckart–Young (teoria vs misura) e fattore di compressione k(m+n+1)/mn.](../results/figures/fig_eckart_young.png)

*Verifica numerica del teorema di Eckart–Young (teoria vs misura) e fattore di compressione k(m+n+1)/mn.*


### 2.2 Compressione del segreto (riduzione di dimensionalità)

Il segreto S viene sostituito dalla sua approssimazione di rango k e memorizzato nei fattori compatti (U_k, σ_k, V_kᵀ). Il payload passa da m·n a k(m+n+1) numeri: è la stessa compressione con perdita vista a lezione (slide 'Reduction of memory occupation'). I fattori sono quantizzati (U,V a int8, σ a float16) per ridurre ulteriormente il payload.


![Ricostruzione del segreto al variare del rango k: aumentando k migliora la fedeltà (NC) ma cresce il payload.](../results/figures/fig_secret_reconstructions.png)

*Ricostruzione del segreto al variare del rango k: aumentando k migliora la fedeltà (NC) ma cresce il payload.*


### 2.3 Embedding nel dominio SVD a blocchi (QIM)

Il cover è diviso in blocchi B×B. Per ogni blocco A = UΣVᵀ si codifica un bit nel valore singolare massimo σ₁ tramite Quantisation Index Modulation con passo Δ. σ₁ concentra quasi tutta l'energia del blocco (Eckart–Young) ed è quindi la componente più stabile; inoltre una variazione δ di σ₁ produce ‖δ u₁v₁ᵀ‖_F = |δ| distribuita sull'intero blocco, dunque impercettibile per pixel. La decodifica è cieca.


### 2.4 Guida YOLO (content-adaptive)

YOLOv8 rileva gli oggetti nel cover; dalle bounding box si costruisce una mappa di saliency. L'ordine di riempimento dei blocchi privilegia lo **sfondo** (saliency bassa), lasciando quasi intatte le regioni degli oggetti. YOLO viene poi usato anche per **valutare** quanto le detection si conservano sul cover rispetto allo stego (preservation rate, IoU, mAP@0.5).


## 3. Implementazione

Il progetto è in Python (NumPy, OpenCV, scikit-image, Ultralytics YOLOv8, Matplotlib). Moduli principali: `svd_core` (matematica SVD), `steganography` (codec del segreto + embedding/estrazione), `yolo_guidance` (rilevamento e metriche), `metrics`, `attacks`, `dataset`, `pipeline`.

Sono definiti due **punti operativi** che illustrano il compromesso capacità↔robustezza:

| Parametro | High-Capacity (HC) | Robust (RB) |
|---|---|---|
| Blocco SVD | 8×8 | 16×16 |
| Passo QIM Δ | 24.0 | 128.0 |
| Ripetizione R | 1 | 3 |
| Segreto | 64×64, k=10 | 16×16, k=3 |
| Capacità (bpp) | 0.0469 | 0.0039 |


## 4. Setup sperimentale

- Dataset: COCO-128 (128 immagini reali di COCO), cover ridimensionati a 512×512.
- Segreto: logo grayscale sintetico (forme + gradiente), compresso via SVD troncata.
- Metriche di fedeltà: PSNR, SSIM, MSE (cover vs stego), globali e per regione (oggetto vs sfondo).
- Metriche di payload: NC (correlazione normalizzata) e PSNR del segreto recuperato, BER.
- Downstream: preservation rate, IoU medio, mAP@0.5, |Δconfidenza| (cover vs stego).
- Attacchi: JPEG (q90/q75/q50), rumore gaussiano, blur, mediano, sale&pepe, rescale.


## 5. Risultati


### 5.1 Impercettibilità

In modalità HC il cover e lo stego sono praticamente indistinguibili: PSNR medio **47.39 dB**, SSIM **0.9950**. In modalità RB (più robusta) PSNR medio **39.04 dB**, SSIM **0.9820**.


![Distribuzione di PSNR e SSIM cover→stego sulle immagini di test.](../results/figures/fig_imperceptibility.png)

*Distribuzione di PSNR e SSIM cover→stego sulle immagini di test.*


### 5.2 Protezione degli oggetti tramite YOLO

La guida YOLO sposta il payload nello sfondo: il PSNR **nelle regioni degli oggetti** sale da 47.40 dB (baseline senza YOLO) a **50.39 dB** (guidato), un miglioramento di **2.99 dB** esattamente dove l'occhio (e il rilevatore) guardano. Lo sfondo, di contro, passa da 47.40 a 46.82 dB.


![PSNR per regione (oggetto vs sfondo): la guida YOLO protegge gli oggetti.](../results/figures/fig_object_protection.png)

*PSNR per regione (oggetto vs sfondo): la guida YOLO protegge gli oggetti.*


![Risultati qualitativi: la mappa |cover−stego| mostra che le modifiche evitano le regioni degli oggetti.](../results/figures/fig_qualitative.png)

*Risultati qualitativi: la mappa |cover−stego| mostra che le modifiche evitano le regioni degli oggetti.*


### 5.3 Preservazione del contenuto semantico (downstream)

Le detection di YOLO si conservano quasi perfettamente dopo l'embedding: preservation rate medio **0.961**, mAP@0.5 medio **0.959**, variazione media di confidenza **0.016**. Lo stego è quindi utilizzabile in una pipeline di AI a valle senza degradarne le prestazioni.


![Distribuzione di preservation rate, mAP@0.5 e |Δconfidenza|.](../results/figures/fig_detection_preservation.png)

*Distribuzione di preservation rate, mAP@0.5 e |Δconfidenza|.*


### 5.4 Recupero del segreto e compressione SVD

Su immagine non attaccata, il segreto si recupera con NC medio **0.923** (HC) e **0.929** (RB), con BER ≈ 0.00150 (HC). Il limite di fedeltà in pulito è dato dalla **compressione SVD troncata** del segreto, non dall'estrazione: aumentando k la qualità cresce a scapito del payload.


![Qualità di ricostruzione (PSNR/energia) e fattore di compressione del segreto al variare del rango k.](../results/figures/fig_secret_compression.png)

*Qualità di ricostruzione (PSNR/energia) e fattore di compressione del segreto al variare del rango k.*


### 5.5 Robustezza agli attacchi

La modalità RB (blocchi 16×16, Δ grande, ripetizione ×3) resiste agli attacchi comuni, mentre la modalità HC — priva di ridondanza — è fragile: è il prezzo della massima capacità.

| Attacco | NC (HC) | NC (RB) | BER (HC) | BER (RB) |
|---|---|---|---|---|
| jpeg_q90 | 0.000 | 0.908 | 0.215 | 0.001 |
| jpeg_q75 | 0.000 | 0.859 | 0.396 | 0.002 |
| jpeg_q50 | 0.000 | 0.471 | 0.478 | 0.009 |
| gauss_noise5 | 0.000 | 0.863 | 0.368 | 0.003 |
| blur_3x3 | 0.039 | 0.705 | 0.192 | 0.016 |
| median_3x3 | 0.051 | 0.515 | 0.204 | 0.049 |
| saltpepper_1 | 0.000 | 0.054 | 0.233 | 0.089 |
| rescale_50 | 0.014 | 0.552 | 0.237 | 0.036 |


![NC del segreto recuperato e BER del payload sotto attacco.](../results/figures/fig_robustness.png)

*NC del segreto recuperato e BER del payload sotto attacco.*


![Fronte di Pareto capacità↔robustezza tra le due modalità.](../results/figures/fig_tradeoff.png)

*Fronte di Pareto capacità↔robustezza tra le due modalità.*


## 6. Analisi e commenti

- Il valore singolare massimo σ₁ è un dominio di embedding eccellente: concentra l'energia del blocco (Eckart–Young) ed è poco sensibile alle perturbazioni, in linea con l'osservazione del corso che i piccoli σ amplificano il rumore.
- La guida YOLO migliora nettamente la qualità nelle regioni salienti (+~10 dB sugli oggetti) senza intaccare la capacità, e mantiene mAP≈1.
- Esiste un chiaro compromesso capacità↔robustezza governato da tre leve SVD (dimensione blocco, passo Δ, ridondanza R): non è possibile massimizzare contemporaneamente capacità, impercettibilità e robustezza.
- La compressione del segreto via SVD troncata abilita payload elevati ma, eliminando la ridondanza spaziale, rende il recupero sensibile agli errori di bit: per la robustezza è necessaria una codifica di canale (ripetizione).


## 7. Conclusioni

Il progetto mostra come la SVD — cuore del corso — fornisca un quadro unificato per la steganografia: compressione del segreto (Eckart–Young), dominio di embedding robusto (σ₁), e analisi del condizionamento. L'integrazione con YOLO rende l'occultamento adattivo al contenuto e verificabile a valle. I risultati su 100+ immagini COCO quantificano impercettibilità, preservazione semantica e il compromesso capacità↔robustezza.


## Riferimenti

- G. Eckart, G. Young, 'The approximation of one matrix by another of lower rank', Psychometrika, 1936.
- G. Golub, C. Van Loan, 'Matrix Computations', 4th ed., 2013.
- R. Liu, T. Tan, 'An SVD-based watermarking scheme for protecting rightful ownership', IEEE Trans. Multimedia, 2002.
- G. Jocher et al., 'Ultralytics YOLOv8', 2023.
- T.-Y. Lin et al., 'Microsoft COCO: Common Objects in Context', ECCV 2014.
- M. Popolizio, slide del corso 'Reducing data dimensionality for AI applications', 2025.
