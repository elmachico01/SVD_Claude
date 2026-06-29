# Steganografia di Immagini basata su SVD con Guida YOLO

**Compressione e occultamento di informazione mediante Decomposizione ai Valori Singolari, con embedding adattivo al contenuto**

*Progetto d'esame — Statistical and Mathematical Methods for AI (Prof. M. Popolizio)*  
*Valutazione sperimentale su 128 immagini reali del dataset COCO-128 — rilevatore YOLOv8 attivo*  

## Sommario

Questo progetto realizza e valuta uno schema di **steganografia di immagini** in cui la **Decomposizione ai Valori Singolari (SVD)** — argomento centrale del corso — compare in due ruoli distinti e complementari. In primo luogo la SVD troncata, ottimale per il teorema di Eckart–Young, comprime l'immagine segreta riducendone l'occupazione di memoria; in secondo luogo l'SVD a blocchi fornisce il dominio in cui nascondere il messaggio, modulando il valore singolare massimo di ogni blocco (Quantisation Index Modulation). Un rilevatore di oggetti **YOLOv8** rende l'occultamento **adattivo al contenuto**, spingendo l'informazione nello sfondo e proteggendo le regioni semanticamente rilevanti; lo stesso rilevatore quantifica poi quanto il contenuto sopravvive all'embedding.

Sull'intero COCO-128 (128 immagini) lo schema raggiunge un'impercettibilità elevata (PSNR medio **47.39 dB**, SSIM **0.9950** in modalità ad alta capacità), preserva quasi perfettamente le rilevazioni a valle (mAP@0.5 medio **0.959**) e recupera il segreto in assenza di attacchi con correlazione normalizzata media **0.923**. Una seconda modalità, robusta, sopravvive ad attacchi comuni (JPEG, rumore, filtraggi) grazie a blocchi più grandi e a un codice a ripetizione. I risultati quantificano in modo esplicito il compromesso fondamentale tra **capacità**, **impercettibilità** e **robustezza**, governato interamente da grandezze legate alla SVD.


## 1. Introduzione e obiettivi

La **steganografia** nasconde un'informazione dentro un mezzo apparente (qui un'immagine *cover*) in modo che la *presenza stessa* del messaggio resti impercettibile. Si differenzia dalla crittografia, che rende il messaggio illeggibile ma non ne nasconde l'esistenza: la steganografia punta all'**invisibilita'**, non solo alla segretezza.

Un buon schema steganografico deve bilanciare tre requisiti in tensione tra loro:

- **Impercettibilita'** — il cover modificato (*stego*) non deve risultare visibilmente alterato (PSNR/SSIM elevati);
- **Capacita'** — quanti bit di informazione si riescono a nascondere (bit per pixel, bpp);
- **Robustezza** — il messaggio deve sopravvivere a manipolazioni involontarie o ostili (ricompressione JPEG, rumore, ridimensionamento).

L'obiettivo del progetto è costruire uno schema che (i) usi la **SVD** come strumento matematico unificante — sia per comprimere il segreto sia come dominio di embedding — e (ii) sfrutti un rilevatore di oggetti per rendere l'occultamento **content-adaptive**: nascondere i dati dove l'occhio e gli algoritmi a valle guardano meno, cioe' nello sfondo, preservando gli oggetti. Oltre a misurare le metriche classiche, si verifica una proprietà nuova e utile: che l'immagine stego resti **usabile da una pipeline di intelligenza artificiale a valle**, mantenendo inalterate le rilevazioni di YOLO.


## 2. Background matematico

Tutta la notazione segue le slide del corso. La SVD è lo strumento centrale: se ne richiamano le proprietà usate nel progetto.


### 2.1 Decomposizione ai Valori Singolari

Per ogni matrice reale A di dimensione m×n esistono due matrici ortogonali U (m×m) e V (n×n) e una matrice diagonale Σ tali che A = UΣVᵀ, con σ₁ ≥ σ₂ ≥ … ≥ σ_r > 0 valori singolari e r = rango(A). Equivalentemente A si scrive come somma di matrici di rango 1 ordinate per importanza decrescente:


> A = Σᵢ σᵢ · uᵢ vᵢᵀ   (i = 1 … r)

I vettori singolari uᵢ, vᵢ hanno norma unitaria: questa proprietà sarà cruciale per stimare la distorsione dell'embedding (Sez. 3.3). Implementazione: `svd_core.svd_decompose`.


### 2.2 SVD troncata e teorema di Eckart–Young

Trattenendo i primi k termini si ottiene la SVD troncata A_k = Σᵢ≤ₖ σᵢ uᵢ vᵢᵀ. Il teorema di Eckart–Young afferma che A_k è la **migliore approssimazione di rango k** di A, sia in norma spettrale sia di Frobenius, e che l'errore è governato esattamente dai valori singolari scartati:


> ‖A − A_k‖₂ = σₖ₊₁      ‖A − A_k‖_F = ( Σᵢ>ₖ σᵢ² )^{1/2}

La **frazione di energia** trattenuta da A_k, E(k) = (Σᵢ≤ₖ σᵢ²)/(Σᵢ σᵢ²), fornisce un criterio pratico per scegliere k. La Fig. 1 mostra, su un cover reale, che lo spettro dei valori singolari decade rapidamente e che **poche componenti catturano la quasi totalità dell'energia** (90% con k=3, 95% con k=8, 99% con k=41 su una matrice 512×512): è la ragione per cui le immagini naturali sono comprimibili. La Fig. 2 verifica numericamente Eckart–Young, mostrando che l'errore di approssimazione misurato coincide con la previsione teorica.


![Fig. 1 — Spettro dei valori singolari di un cover (scala log) ed energia cumulata: il decadimento rapido giustifica la compressione a basso rango.](../results/figures/fig_svd_spectrum.png)

*Fig. 1 — Spettro dei valori singolari di un cover (scala log) ed energia cumulata: il decadimento rapido giustifica la compressione a basso rango.*


![Fig. 2 — Verifica del teorema di Eckart–Young: errore teorico vs misurato (norma 2 e Frobenius) e fattore di compressione k(m+n+1)/mn al crescere di k.](../results/figures/fig_eckart_young.png)

*Fig. 2 — Verifica del teorema di Eckart–Young: errore teorico vs misurato (norma 2 e Frobenius) e fattore di compressione k(m+n+1)/mn al crescere di k.*


### 2.3 Occupazione di memoria e fattore di compressione

Memorizzare A_k richiede le k colonne di U_k, i k valori singolari e le k righe di V_kᵀ, cioe' k(m+n+1) numeri contro gli mn della matrice piena. Il **fattore di compressione** è ρ(k) = k(m+n+1)/mn. E' esattamente il conto della slide *Reduction of memory occupation*, qui applicato al payload da nascondere. Implementazione: `svd_core.compression_factor`.


### 2.4 Legame con la decomposizione spettrale e condizionamento

Le colonne di V e U sono gli autovettori di AᵀA e AAᵀ, e σᵢ = √λᵢ: il modulo `svd_core.verify_svd_eigendecomposition` verifica questa identita' a meno della precisione di macchina. Il **numero di condizionamento** κ(A) = σ₁/σ_r misura quanto i piccoli valori singolari amplifichino il rumore: come osservato nelle slide sui sistemi rettangolari, modificare un σ piccolo è fragile. Questo motiva la scelta progettuale di nascondere l'informazione nel valore singolare **massimo** σ₁, la componente più stabile (Sez. 3.3). La pseudoinversa di Moore–Penrose A⁺ = VΣ⁺Uᵀ e la sua versione troncata, anch'esse trattate a lezione, sono implementate in `svd_core.pseudoinverse` e verificate nei test.


## 3. Metodo proposto

La pipeline elabora una coppia (cover, segreto) in cinque passi:

1. il segreto viene **compresso** con SVD troncata in fattori compatti;
2. YOLO produce una **mappa di priorità** dal contenuto del cover;
3. il payload viene **nascosto** nei blocchi del cover modulando σ₁ (QIM), riempiendo prima lo sfondo;
4. il segreto viene **estratto in modo cieco** e ricostruito via SVD inversa;
5. YOLO viene riapplicato a cover e stego per misurare la **preservazione semantica**.


### 3.1 Compressione del segreto (SVD troncata)

L'immagine segreta S (grayscale) è sostituita dalla sua approssimazione di rango k e serializzata in un payload auto-descrittivo: un'intestazione di 9 byte (magic, versione, altezza, larghezza, k) seguita dai fattori σ_k (float16), U_k e V_kᵀ (quantizzati a int8). La quantizzazione dei fattori riduce ulteriormente il payload. Il decoder legge prima l'intestazione e poi esattamente il numero di bit atteso. Implementazione: `steganography.compress_secret / decompress_secret`.


### 3.2 Embedding nel dominio SVD a blocchi (QIM su σ₁)

Il cover è diviso in blocchi B×B. Per ogni blocco portante A = UΣVᵀ si codifica un bit nel valore singolare massimo σ₁ tramite **Quantisation Index Modulation** con passo Δ:


> σ₁' = ⌊σ₁/Δ⌋·Δ + (0.25 o 0.75)·Δ   a seconda del bit

Il blocco viene ricostruito come A' = UΣ'Vᵀ e la decodifica è **cieca**: il bit si legge dalla parte frazionaria di σ₁/Δ (soglia 0.5), con margine di robustezza Δ/4. Si codifica in σ₁ per due ragioni complementari, entrambe radicate nella SVD: (a) σ₁ concentra quasi tutta l'energia del blocco (Eckart–Young), quindi è la componente più stabile sotto perturbazioni; (b) la distorsione introdotta è ‖δ·u₁v₁ᵀ‖_F = |δ|, **distribuita su tutti i pixel del blocco** perché u₁, v₁ hanno norma unitaria — quindi impercettibile pixel per pixel.


### 3.3 Guida YOLO (occultamento adattivo al contenuto)

YOLOv8 rileva gli oggetti nel cover; dalle bounding box si costruisce una mappa di *saliency* che assegna priorità alta agli oggetti e bassa allo sfondo. L'ordine di riempimento dei blocchi è deciso da questa mappa (sfondo prima, oggetti per ultimi), con un permutazione pseudo-casuale basata su chiave a rompere i pareggi. Lo stesso ordinamento è ricostruibile in fase di estrazione. In assenza di rilevatore lo schema ripiega su una permutazione cieca basata solo sulla chiave, senza bisogno del detector. Implementazione: `yolo_guidance.priority_map`, `steganography.embedding_order`.


### 3.4 Robustezza: codice a ripetizione

Per la robustezza ogni bit logico viene scritto in R siti consecutivi (codice a ripetizione) e decodificato a **voto di maggioranza**. Questo aumenta la resistenza agli errori al prezzo di una capacità ridotta di un fattore R.


## 4. Implementazione e punti operativi

Il progetto è in Python (NumPy, OpenCV, scikit-image, Ultralytics YOLOv8, Matplotlib). I moduli principali sono: `svd_core` (matematica SVD), `steganography` (codec del segreto + embedding/estrazione), `yolo_guidance` (rilevamento e metriche), `metrics`, `attacks`, `dataset` e `pipeline` (orchestrazione).

Per illustrare il compromesso capacità↔robustezza sono definiti due **punti operativi** di riferimento, usati in tutti gli esperimenti:

| Parametro | High-Capacity (HC) | Robust (RB) |
|---|---|---|
| Lato blocco SVD B | 8×8 | 16×16 |
| Passo QIM Δ | 24.0 | 128.0 |
| Ripetizione R | 1 | 3 |
| Segreto | 64×64, k=10 | 16×16, k=3 |
| Payload (bit) | 10472 | 888 |
| Capacita' (bpp) | 0.0469 | 0.0039 |
| Obiettivo | max capacità + impercettibilità | resistenza agli attacchi |

Le tre leve sono tutte di natura SVD: blocchi più grandi (B) concentrano più energia in σ₁ (più robusto, meno capacità); un passo Δ maggiore allarga il margine Δ/4 (più robusto, più distorsione); la ripetizione R aggiunge ridondanza di canale a scapito della capacità.


## 5. Protocollo sperimentale

- **Dataset**: COCO-128, 128 immagini reali di COCO, cover ridimensionati a 512×512; in totale 630 oggetti rilevati da YOLO, con detection presenti nel 97.7% delle immagini.
- **Segreto**: logo grayscale 64×64 (HC) / 16×16 (RB), compresso via SVD troncata.
- **Fedelta'**: PSNR, SSIM, MSE tra cover e stego, sia globali sia per regione (oggetto vs sfondo).
- **Payload**: correlazione normalizzata (NC) e PSNR del segreto recuperato, bit error rate (BER).
- **Downstream**: detection preservation rate, IoU medio, mAP@0.5 e |Δconfidenza| tra le rilevazioni su cover e su stego.
- **Attacchi**: JPEG (q90/q75/q50), rumore gaussiano, blur 3×3, filtro mediano 3×3, sale e pepe, ridimensionamento al 50%.
- **Costo**: l'intera batteria su 128 immagini (HC+RB, baseline, attacchi, downstream) gira in ~10.8 minuti su CPU.


## 6. Risultati


### 6.1 Compressione del segreto via SVD troncata

La Tab. 2 e la Fig. 3 quantificano il compromesso qualità↔payload della compressione del segreto al variare del rango k (segreto 64×64). Gia' con k=10 si trattiene il 99.1% dell'energia con un payload di soli 1309 byte (fattore di compressione 0.31); aumentando k la fedeltà cresce ma il payload anche, fino a superare la matrice piena. La Fig. 4 mostra visivamente come la ricostruzione migliori con k.

| k | Payload (byte) | Compressione ρ(k) | Energia trattenuta | PSNR ricostr. (dB) |
|---|---|---|---|---|
| 2 | 269 | 0.063 | 0.9515 | 16.74 |
| 4 | 529 | 0.126 | 0.9737 | 19.42 |
| 6 | 789 | 0.189 | 0.9820 | 21.00 |
| 8 | 1049 | 0.252 | 0.9874 | 22.52 |
| 10 | 1309 | 0.315 | 0.9911 | 23.97 |
| 12 | 1569 | 0.378 | 0.9936 | 25.31 |
| 16 | 2089 | 0.504 | 0.9969 | 28.10 |
| 20 | 2609 | 0.630 | 0.9986 | 30.75 |
| 24 | 3129 | 0.756 | 0.9993 | 32.67 |
| 32 | 4169 | 1.008 | 0.9999 | 35.90 |

*Tab. 2 — Qualita' di ricostruzione, payload ed energia della SVD troncata del segreto al variare di k (da results/secret_compression.csv).*


![Fig. 3 — PSNR ed energia trattenuta (sinistra) e fattore di compressione (destra) del segreto al variare del rango k.](../results/figures/fig_secret_compression.png)

*Fig. 3 — PSNR ed energia trattenuta (sinistra) e fattore di compressione (destra) del segreto al variare del rango k.*


![Fig. 4 — Ricostruzione del segreto al crescere di k: la correlazione normalizzata (NC) con l'originale aumenta con il rango.](../results/figures/fig_secret_reconstructions.png)

*Fig. 4 — Ricostruzione del segreto al crescere di k: la correlazione normalizzata (NC) con l'originale aumenta con il rango.*


### 6.2 Impercettibilita'

In modalità HC cover e stego sono praticamente indistinguibili: PSNR medio **47.39 dB** (deviazione standard appena 0.13 dB su 128 immagini) e SSIM **0.9950**. La modalità RB, più robusta, paga in qualità (PSNR **39.04 dB**, SSIM **0.9820**) a causa del passo Δ molto maggiore. La Fig. 5 mostra le distribuzioni: estremamente concentrate, segno di un comportamento stabile su tutto il dataset.


![Fig. 5 — Distribuzione di PSNR (sinistra) e SSIM (destra) cover→stego per le due modalità.](../results/figures/fig_imperceptibility.png)

*Fig. 5 — Distribuzione di PSNR (sinistra) e SSIM (destra) cover→stego per le due modalità.*


### 6.3 Protezione degli oggetti tramite YOLO

La guida YOLO sposta il payload nello sfondo: il PSNR **nelle regioni degli oggetti** sale da 47.40 dB (baseline senza guida) a **50.40 dB** con la guida, mentre lo sfondo scende simmetricamente da 47.40 a 46.82 dB. Il miglioramento sugli oggetti è positivo nel **96.8% delle immagini** (guadagno mediano +1.47 dB; in molte immagini la regione degli oggetti resta del tutto intatta, con PSNR che tende all'infinito). E' esattamente il comportamento desiderato: spostare il rumore di embedding là dove conta meno. La Fig. 7 lo mostra qualitativamente — la mappa |cover−stego| evita nettamente le regioni degli oggetti, che coincidono con le zone a saliency alta.


![Fig. 6 — PSNR per regione (oggetto vs sfondo): la guida YOLO protegge gli oggetti. A destra la distribuzione del PSNR sugli oggetti.](../results/figures/fig_object_protection.png)

*Fig. 6 — PSNR per regione (oggetto vs sfondo): la guida YOLO protegge gli oggetti. A destra la distribuzione del PSNR sugli oggetti.*


![Fig. 7 — Esempi qualitativi: cover, stego, differenza amplificata |cover−stego|, saliency YOLO e segreto recuperato. Le modifiche evitano gli oggetti.](../results/figures/fig_qualitative.png)

*Fig. 7 — Esempi qualitativi: cover, stego, differenza amplificata |cover−stego|, saliency YOLO e segreto recuperato. Le modifiche evitano gli oggetti.*


### 6.4 Preservazione del contenuto semantico (downstream)

Le rilevazioni di YOLO si conservano quasi perfettamente dopo l'embedding: detection preservation rate medio **0.961**, mAP@0.5 medio **0.959**, IoU medio **0.986** e variazione media di confidenza appena **0.016**. Le rilevazioni restano perfettamente invariate nell'**79.2% delle immagini**. Da notare che la guida YOLO migliora anche questo aspetto rispetto alla baseline (preservation 0.961 vs 0.942). Lo stego è quindi utilizzabile in una pipeline di AI a valle senza degradarne le prestazioni — una proprietà non ovvia e di interesse pratico.


![Fig. 8 — Distribuzione di detection preservation rate, mAP@0.5 (stego vs cover) e |Δconfidenza| sulle 128 immagini.](../results/figures/fig_detection_preservation.png)

*Fig. 8 — Distribuzione di detection preservation rate, mAP@0.5 (stego vs cover) e |Δconfidenza| sulle 128 immagini.*


### 6.5 Recupero del segreto in assenza di attacchi

Su immagine non attaccata il segreto si recupera con NC medio **0.923** (mediana 0.977; NC ≥ 0.95 nell'83.6% delle immagini) e BER medio **0.00150**, con il 57.8% delle immagini a BER nullo. In modalità RB il payload è ancora più affidabile (BER nullo nell'86.7% delle immagini). Il limite di fedeltà in pulito non è dovuto all'estrazione (che è quasi perfetta) ma alla **compressione SVD** del segreto: è il prezzo del basso rango scelto per contenere il payload, come già visto in Sez. 6.1.


### 6.6 Robustezza agli attacchi

La Tab. 3 e la Fig. 9 riassumono la robustezza. La modalità RB (blocchi 16×16, Δ grande, ripetizione ×3) resiste bene agli attacchi comuni — ad esempio JPEG-q90 con NC 0.91 e BER 0.001, rumore gaussiano con NC 0.86 — mentre la modalità HC, priva di ridondanza e con blocchi piccoli, crolla già al primo attacco: è il prezzo della massima capacità. L'attacco più severo per entrambe è il sale e pepe, che introduce outlier impulsivi che perturbano fortemente σ₁.

| Attacco | NC (HC) | NC (RB) | BER (HC) | BER (RB) |
|---|---|---|---|---|
| jpeg_q90 | 0.000 | 0.908 | 0.215 | 0.001 |
| jpeg_q75 | 0.000 | 0.859 | 0.396 | 0.002 |
| jpeg_q50 | 0.000 | 0.480 | 0.478 | 0.009 |
| gauss_noise5 | 0.000 | 0.863 | 0.368 | 0.003 |
| blur_3x3 | 0.039 | 0.705 | 0.193 | 0.016 |
| median_3x3 | 0.053 | 0.515 | 0.204 | 0.049 |
| saltpepper_1 | 0.000 | 0.054 | 0.233 | 0.089 |
| rescale_50 | 0.014 | 0.553 | 0.237 | 0.036 |

*Tab. 3 — Correlazione normalizzata del segreto recuperato e BER del payload sotto attacco, per le due modalità.*


![Fig. 9 — NC del segreto recuperato (sinistra) e BER del payload (destra) sotto ciascun attacco.](../results/figures/fig_robustness.png)

*Fig. 9 — NC del segreto recuperato (sinistra) e BER del payload (destra) sotto ciascun attacco.*


### 6.7 Il compromesso capacità↔robustezza

La Fig. 10 colloca le due modalità su un piano capacità (bpp) vs robustezza media (NC sotto attacco). I due punti sono agli estremi opposti di un fronte di Pareto: HC massimizza la capacità (0.0469 bpp) ma è fragile, RB è robusta (NC medio 0.61) ma con capacità ~12× inferiore (0.0039 bpp). Non esiste un punto che massimizzi simultaneamente capacità, impercettibilità e robustezza: la scelta dipende dall'applicazione.


![Fig. 10 — Fronte di Pareto capacità↔robustezza tra le due modalità operative.](../results/figures/fig_tradeoff.png)

*Fig. 10 — Fronte di Pareto capacità↔robustezza tra le due modalità operative.*


## 7. Discussione

- **σ₁ è un dominio di embedding eccellente.** Concentra l'energia del blocco (Eckart–Young) ed è poco sensibile alle perturbazioni, coerentemente con l'osservazione del corso che i piccoli valori singolari amplificano il rumore. La distorsione si distribuisce sull'intero blocco, rendendola impercettibile per pixel.
- **La guida YOLO funziona e non costa capacità.** Migliora la qualità nelle regioni salienti e, soprattutto, mantiene quasi inalterate le rilevazioni a valle, rendendo lo stego utilizzabile in una pipeline di AI.
- **Il compromesso è intrinseco e quantificabile.** Le tre leve (B, Δ, R) sono tutte SVD-based: spostano il sistema lungo il fronte capacità↔robustezza ma non lo superano.
- **Compressione e robustezza sono in tensione.** La SVD troncata massimizza la capacità eliminando la ridondanza spaziale del segreto, ma proprio questo rende il recupero sensibile agli errori di bit: per la robustezza serve reintrodurre ridondanza a livello di canale (ripetizione).


## 8. Limiti e sviluppi futuri

- L'estrazione guidata richiede la stessa mappa di priorità all'estrazione; una variante completamente cieca usa la permutazione a chiave, rinunciando pero' alla protezione adattiva degli oggetti.
- Il sale e pepe resta l'attacco più critico: un filtraggio di pre-processing o un codice correttore più forte (es. BCH al posto della semplice ripetizione) migliorerebbero la robustezza.
- La SVD a blocchi non è invariante a rotazioni/crop geometrici; domini sincronizzati (es. template di riallineamento) sarebbero un'estensione naturale.
- Il segreto è un logo sintetico: si potrebbe estendere a payload arbitrari (testo, chiavi) mantenendo la stessa pipeline.


## 9. Conclusioni

Il progetto mostra come la SVD — cuore del corso — fornisca un quadro unificato per la steganografia di immagini: compressione del segreto via troncamento ottimale (Eckart–Young), dominio di embedding robusto (il valore singolare massimo) e analisi del condizionamento a guidare le scelte progettuali. L'integrazione con YOLO rende l'occultamento adattivo al contenuto e, soprattutto, verificabile a valle. La valutazione su 128 immagini reali quantifica un'impercettibilità elevata, una preservazione semantica quasi perfetta e un compromesso capacità↔robustezza chiaro e interpretabile in termini puramente algebrici.


## Riferimenti

1. G. Eckart, G. Young, *The approximation of one matrix by another of lower rank*, Psychometrika, 1936.
2. G. Golub, C. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013.
3. R. Liu, T. Tan, *An SVD-based watermarking scheme for protecting rightful ownership*, IEEE Trans. Multimedia, 2002.
4. B. Chen, G. Wornell, *Quantization index modulation*, IEEE Trans. Information Theory, 2001.
5. G. Jocher et al., *Ultralytics YOLOv8*, 2023.
6. T.-Y. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
7. M. Popolizio, slide del corso *Reducing data dimensionality for AI applications*, 2025.


## Appendice A — Mappatura concetti del corso ↔ codice

| Concetto della lezione | Dove nel progetto |
|---|---|
| SVD A = UΣVᵀ | svd_core.svd_decompose |
| SVD troncata / Eckart–Young | svd_core.truncated_svd; compressione segreto |
| Fattore di compressione k(m+n+1)/mn | svd_core.compression_factor |
| SVD ↔ autovalori (σ=√λ) | svd_core.verify_svd_eigendecomposition |
| Numero di condizionamento κ=σ₁/σ_r | svd_core.condition_number |
| Pseudoinversa / minimi quadrati | svd_core.pseudoinverse, lstsq_via_svd |


## Appendice B — Sintesi delle metriche principali

| Metrica (HC, media) | Valore |
|---|---|
| PSNR cover→stego | 47.39 dB |
| SSIM cover→stego | 0.9950 |
| PSNR oggetti (guidato) | 50.40 dB |
| Capacita' | 0.0469 bpp (10472 bit) |
| NC segreto (pulito) | 0.923 |
| BER segreto (pulito) | 0.00150 |
| Detection preservation rate | 0.961 |
| mAP@0.5 (stego vs cover) | 0.959 |
| IoU medio | 0.986 |
