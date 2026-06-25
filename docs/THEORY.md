# Background matematico

Documento di supporto al progetto *SVD & YOLO per la Steganografia di Immagini*.
Tutta la notazione segue le slide del corso **Statistical and Mathematical
Methods for AI** (M. Popolizio).

---

## 1. Decomposizione ai Valori Singolari (SVD)

Per ogni matrice reale $A \in \mathbb{R}^{m\times n}$ esistono matrici ortogonali
$U \in \mathbb{R}^{m\times m}$, $V \in \mathbb{R}^{n\times n}$ e una matrice
diagonale $\Sigma \in \mathbb{R}^{m\times n}$ tali che

$$A = U\Sigma V^{T}, \qquad \Sigma = \operatorname{diag}(\sigma_1,\dots,\sigma_r),
\quad \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0,$$

dove $r = \operatorname{rank}(A)$. Le colonne $u_i$ di $U$ e $v_i$ di $V$ sono i
**vettori singolari** sinistri e destri. Equivalentemente

$$A = \sum_{i=1}^{r} \sigma_i\, u_i v_i^{T},$$

cioè $A$ è somma di $r$ matrici di rango 1 ordinate per importanza decrescente.

Nel progetto: `src/svd_core.py::svd_decompose`.

---

## 2. SVD troncata e teorema di Eckart–Young

Trattenendo solo i primi $k$ termini si ottiene la **SVD troncata**

$$A_k = \sum_{i=1}^{k} \sigma_i\, u_i v_i^{T}.$$

**Teorema (Eckart–Young).** Per ogni $k < \operatorname{rank}(A)$, $A_k$ è la
*migliore* approssimazione di rango $k$ sia in norma spettrale che di Frobenius:

$$\lVert A - A_k\rVert_2 = \min_{\operatorname{rank}(B)=k} \lVert A-B\rVert_2 = \sigma_{k+1},$$
$$\lVert A - A_k\rVert_F = \min_{\operatorname{rank}(B)=k} \lVert A-B\rVert_F = \Big(\sum_{i=k+1}^{\min(m,n)} \sigma_i^2\Big)^{1/2}.$$

L'errore di troncamento è quindi controllato esattamente dai valori singolari
scartati. La **frazione di energia** trattenuta da $A_k$ è

$$E(k) = \frac{\sum_{i\le k}\sigma_i^2}{\sum_i \sigma_i^2},$$

e fornisce un criterio pratico per scegliere $k$ (es. $E(k)\ge 0.95$).

Nel progetto: `eckart_young_errors`, `energy_ratio`, `rank_for_energy`.
La verifica numerica (teoria vs misura) è in `make_figures.py::fig_eckart_young`.

**Uso 1 — compressione del segreto.** L'immagine segreta $S$ viene sostituita da
$S_k$ e memorizzata come $(U_k,\sigma_k,V_k^{T})$. È la *riduzione di
dimensionalità* del corso applicata al payload da nascondere.

---

## 3. Footprint di memoria e fattore di compressione

Memorizzare $A_k$ richiede le $k$ colonne di $U_k$ ($mk$ numeri), i $k$ valori
singolari e le $k$ righe di $V_k^{T}$ ($nk$ numeri):

$$\underbrace{k(m+n+1)}_{\text{rango }k} \quad\text{contro}\quad \underbrace{mn}_{\text{matrice piena}}.$$

Il **fattore di compressione** è

$$\rho(k) = \frac{k(m+n+1)}{mn}.$$

Per il segreto $64\times64$ con $k=10$ si ha $\rho \approx 0.32$: il payload è
circa un terzo della memoria della matrice piena. È esattamente il conto delle
slide *"Reduction of memory occupation"*.

Nel progetto: `compression_factor`, `storage_floats`,
`steganography.payload_size_bytes`.

---

## 4. SVD e decomposizione spettrale

Date le matrici simmetriche $A^{T}A$ e $AA^{T}$:

$$A^{T}A = V\Lambda V^{T}, \qquad AA^{T} = U\Lambda U^{T}, \qquad
\sigma_i = \sqrt{\lambda_i}.$$

Le colonne di $V$ sono autovettori di $A^{T}A$, quelle di $U$ di $AA^{T}$. Il
legame $\sigma_i=\sqrt{\lambda_i}$ è verificato numericamente in
`verify_svd_eigendecomposition` (lo scarto è dell'ordine della precisione di
macchina).

---

## 5. Condizionamento e robustezza dell'estrazione

Il **numero di condizionamento** sui valori singolari non nulli è

$$\kappa(A) = \frac{\sigma_1}{\sigma_r}.$$

Le slide sui sistemi rettangolari osservano che i **piccoli valori singolari
amplificano il rumore**. Questo spiega la scelta progettuale di nascondere il bit
nel valore singolare **massimo** $\sigma_1$: è la componente più stabile e meno
sensibile alle perturbazioni (rumore, compressione JPEG), mentre modificare un
$\sigma_i$ piccolo sarebbe fragile.

La **pseudoinversa di Moore–Penrose** $A^{+}=V\Sigma^{+}U^{T}$ e la sua versione
troncata (TSVD) forniscono la soluzione ai minimi quadrati / a norma minima dei
sistemi rettangolari, con regolarizzazione dei $\sigma_i$ piccoli. Nel progetto:
`pseudoinverse`, `lstsq_via_svd` (con verifica in `tests/`).

---

## 6. Embedding: Quantisation Index Modulation su $\sigma_1$

Per un blocco cover $A$ di lato $B$ si calcola $A=U\Sigma V^{T}$ e si codifica un
bit $b$ nel valore singolare massimo con passo $\Delta$:

$$\sigma_1' = \Big\lfloor \tfrac{\sigma_1}{\Delta} \Big\rfloor \Delta +
\begin{cases} 0.25\,\Delta & b=0\\[2pt] 0.75\,\Delta & b=1 \end{cases}$$

e si ricostruisce il blocco $A' = U\Sigma' V^{T}$. La decodifica è cieca:

$$b = \begin{cases} 0 & \text{se } \big(\sigma_1'/\Delta \bmod 1\big) < 0.5\\ 1 & \text{altrimenti.}\end{cases}$$

Poiché i vettori singolari hanno norma unitaria, una variazione $\delta$ di
$\sigma_1$ produce $\lVert \delta\, u_1 v_1^{T}\rVert_F = \lvert\delta\rvert$,
**distribuita su tutto il blocco** ⇒ impercettibile per pixel. Il margine di
robustezza è $\Delta/4$.

Nel progetto: `steganography._qim_embed_sigma1`, `_qim_read_sigma1`.

---

## 7. Compromesso capacità ↔ robustezza

Tre leve, tutte di natura SVD:

- **Dimensione del blocco $B$.** Blocchi più grandi ⇒ $\sigma_1$ più grande e più
  "low-frequency" ⇒ più robusto agli attacchi, ma meno blocchi ⇒ meno capacità.
- **Passo $\Delta$.** $\Delta$ grande ⇒ margine $\Delta/4$ grande ⇒ robusto, ma
  maggiore distorsione (PSNR più basso).
- **Codice a ripetizione $R$.** Ogni bit logico è scritto in $R$ siti e decodificato
  a maggioranza ⇒ robustezza a costo di capacità $/R$.

Le due modalità del progetto (High-Capacity e Robust) sono due punti su questo
fronte di Pareto, quantificati in `results/figures/fig_tradeoff.png`.
