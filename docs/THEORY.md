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
$$\lVert A - A_k\rVert_F = \min_{\operatorname{rank}(B)=k} \lVert A-B\rVert_F = \Big(\sum_{i=k+1}^{r} \sigma_i^2\Big)^{1/2}.$$

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

Per il segreto $64\times64$ con $k=10$ si ha $\rho = 10\cdot129/4096 \approx 0.31$:
il payload è circa un terzo della memoria della matrice piena. Dopo la
quantizzazione dei fattori ($U,V \to$ int8, $\sigma \to$ float16, §6.2) il
payload effettivo è di 1309 byte contro i 4096 byte dell'immagine piena
($\approx 0.32$). È esattamente il conto delle slide *"Reduction of memory
occupation"*.

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

## 5. Condizionamento e stabilità

Il **numero di condizionamento** sui valori singolari non nulli è

$$\kappa(A) = \frac{\sigma_1}{\sigma_r}.$$

Le slide sui sistemi rettangolari osservano che i **piccoli valori singolari
amplificano il rumore**: nella soluzione ai minimi quadrati compaiono i
reciproci $1/\sigma_i$, quindi le componenti associate ai $\sigma_i$ piccoli
sono le più instabili. La stessa osservazione, letta al contrario, dice che
$\sigma_1$ è la componente **più stabile** della matrice — ed è la motivazione
per usarla come portante dell'embedding (§6.1).

La **pseudoinversa di Moore–Penrose** $A^{+}=V\Sigma^{+}U^{T}$ e la sua versione
troncata (TSVD) forniscono la soluzione ai minimi quadrati / a norma minima dei
sistemi rettangolari, con regolarizzazione dei $\sigma_i$ piccoli. Nel progetto:
`pseudoinverse`, `lstsq_via_svd` (con verifica in `tests/`).

---

## 6. Embedding e decoding nel dominio SVD

### 6.1 Ruolo dei valori singolari nell'embedding

La scelta di codificare l'informazione **nei valori singolari**, e in
particolare in $\sigma_1$, non è arbitraria ma discende da quattro proprietà:

1. **Concentrazione di energia (Eckart–Young).** Per i blocchi $B\times B$ di
   immagini naturali (fortemente correlati spazialmente) $\sigma_1^2$ raccoglie
   tipicamente il 95–99 % dell'energia $\lVert A\rVert_F^2 = \sum_i\sigma_i^2$:
   $\sigma_1$ è la coordinata "più significativa" del blocco nella base di
   matrici di rango 1 $\{u_iv_i^T\}$.
2. **Stabilità alle perturbazioni (teorema di Weyl, §7.1).** Una perturbazione
   $E$ del blocco sposta *tutti* i valori singolari al più di $\lVert E\rVert_2$
   in assoluto; la perturbazione **relativa** è quindi minima su $\sigma_1$ e
   massima sui $\sigma_i$ piccoli, dove è comparabile con il valore stesso: un
   bit scritto in un $\sigma_i$ piccolo sarebbe illeggibile già dopo un attacco
   lieve. È l'analogo, in embedding, dell'osservazione del corso per cui i
   piccoli $\sigma$ amplificano il rumore (§5).
3. **Invarianza geometrica.** I valori singolari sono invarianti per
   trasformazioni ortogonali ($PAQ$ con $P,Q$ ortogonali ha gli stessi
   $\sigma_i$ di $A$): sono un descrittore *intrinseco* dell'energia del blocco,
   non legato a una particolare disposizione dei pixel.
4. **Distorsione distribuita e a bassa frequenza.** Variare $\sigma_1$ di
   $\delta$ cambia il blocco di $\delta\,u_1v_1^{T}$; poiché $u_1,v_1$ hanno
   norma unitaria, $\lVert\delta\,u_1v_1^{T}\rVert_F = \lvert\delta\rvert$ e la
   modifica si spalma su tutti i $B^2$ pixel **lungo la componente più liscia**
   del blocco ($u_1v_1^T$ è la struttura dominante, di tipo "bassa frequenza").
   Modificare un $\sigma_i$ piccolo inietterebbe invece componenti oscillanti,
   simili a rumore: più visibili e più facili da rilevare per la steganalisi.

### 6.2 Embedding: Quantisation Index Modulation su $\sigma_1$

**Stadio 1 — serializzazione del segreto.** Il segreto $S$ ($h\times w$) è
compresso con SVD troncata di rango $k$ e serializzato nel bitstream

$$\underbrace{\texttt{magic}|\texttt{ver}|h|w|k}_{\text{header, 9 byte}}\;\big|\;
\underbrace{\sigma_1,\dots,\sigma_k}_{\text{float16}}\;\big|\;
\underbrace{U_k \cdot 127}_{\text{int8}}\;\big|\;
\underbrace{V_k^{T}\cdot 127}_{\text{int8}}$$

(le colonne di $U_k$ e le righe di $V_k^T$ hanno elementi in $[-1,1]$ perché di
norma unitaria, quindi la scala fissa 127 è ben posta). La lunghezza totale è
$9 + 2k + k(h+w)$ byte, funzione **solo** di $(h,w,k)$: l'header rende il
payload auto-descrittivo.

**Stadio 2 — scrittura di un bit per blocco.** Il cover è diviso in blocchi
$B\times B$ (per canale). L'ordine dei blocchi è deciso una volta per tutte da
una chiave PRNG e dalla mappa di priorità YOLO (sfondo prima, oggetti per
ultimi). Per ogni blocco portante $A=U\Sigma V^T$ il bit $b$ è codificato nel
valore singolare massimo con passo di quantizzazione $\Delta$:

$$\sigma_1' = \Big\lfloor \tfrac{\sigma_1}{\Delta} \Big\rfloor \Delta +
\begin{cases} 0.25\,\Delta & b=0\\[2pt] 0.75\,\Delta & b=1 \end{cases}$$

e il blocco è ricostruito come $A' = U\Sigma' V^{T}$ (i vettori singolari
restano invariati: cambia solo $\sigma_1$). I due offset $0.25\Delta$ e
$0.75\Delta$ sono i **centri delle due semicelle** dell'intervallo di
quantizzazione: la scelta massimizza (e rende simmetrico) il margine di errore,
pari a $\Delta/4$ in entrambe le direzioni. Con ridondanza $R>1$ ogni bit logico
è scritto in $R$ siti consecutivi dell'ordinamento (codice a ripetizione).

Nel progetto: `steganography.compress_secret`, `_qim_embed_sigma1`,
`embed_bits`, `embedding_order`.

### 6.3 Decoding: estrazione cieca e ricostruzione

L'estrazione è **cieca**: non richiede né il cover originale né i valori
singolari originali (a differenza dello schema di Liu–Tan, che conserva $U,V$
del watermark e per questo è vulnerabile ad attacchi di falsa estrazione).
Servono solo la chiave, i parametri $(B,\Delta,R)$ e — se usata in embedding —
la stessa mappa di priorità.

1. **Ordine dei siti.** Si ricostruisce lo stesso ordinamento deterministico dei
   blocchi usato in embedding (stessa chiave, stessa mappa di priorità).
2. **Lettura di un bit.** Per ogni blocco si calcola la SVD e si decide dalla
   parte frazionaria di $\sigma_1/\Delta$:
   $$b = \begin{cases} 0 & \text{se } \big(\sigma_1/\Delta \bmod 1\big) < 0.5\\
   1 & \text{altrimenti.}\end{cases}$$
   La cella di decisione è larga $\Delta/2$ e il valore embedded siede al suo
   centro: il bit resta corretto finché $\sigma_1$ si sposta meno di $\Delta/4$
   (§7.1). Con $R>1$ il bit logico è il **voto di maggioranza** degli $R$ siti.
3. **Header prima, payload poi.** Si leggono i primi $72$ bit (9 byte), si
   validano *magic* e plausibilità di $(h,w,k)$, e da questi si determina la
   lunghezza esatta del payload: si leggono quindi esattamente i bit necessari.
4. **De-serializzazione e ricostruzione.** Dai byte estratti si recuperano
   $\tilde\sigma$ (float16), $\tilde U_k, \tilde V_k^T$ (int8/127) e si
   ricostruisce
   $$\hat S = \tilde U_k\,\operatorname{diag}(\tilde\sigma)\,\tilde V_k^{T},$$
   con arrotondamento e clip in $[0,255]$. Il parser è difensivo: $\sigma$
   corrotti (NaN/inf da float16 danneggiati) vengono azzerati, cioè la
   componente di rango 1 corrispondente viene semplicemente eliminata — la
   ricostruzione degrada al rango inferiore invece di esplodere.

Nel progetto: `_qim_read_sigma1`, `extract_bits`, `reveal_secret`,
`decompress_secret`.

---

## 7. Effetto delle perturbazioni su decodifica e ricostruzione

### 7.1 Perturbazione dei valori singolari: teorema di Weyl

Un attacco (rumore, compressione JPEG, filtraggio…) trasforma il blocco stego
$A'$ in $A'+E$. Il **teorema di Weyl** limita lo spostamento di *ogni* valore
singolare:

$$\big|\sigma_i(A'+E) - \sigma_i(A')\big| \;\le\; \lVert E\rVert_2
\;\le\; \lVert E\rVert_F \qquad \forall i.$$

Due conseguenze dirette:

- **Condizione di decodifica corretta.** Il bit QIM resta corretto se
  $\lVert E\rVert_2 < \Delta/4$: il margine di robustezza dello schema è
  esattamente il raggio della semicella (§6.2–6.3). La perturbazione assoluta è
  la stessa per tutti i $\sigma_i$, ma solo $\sigma_1$ è abbastanza grande da
  ospitare un passo $\Delta$ ampio senza distruggere il blocco — di nuovo il
  punto 2 di §6.1.
- **Stima quantitativa per rumore gaussiano.** Per $E$ con elementi i.i.d.
  $\mathcal N(0,\sigma_{\text{noise}}^2)$ la teoria delle matrici casuali dà
  $\mathbb E\lVert E\rVert_2 \approx 2\sigma_{\text{noise}}\sqrt B$. Con
  $\sigma_{\text{noise}}=5$ (attacco `gauss_noise5`):
  - **HC** ($B=8$, $\Delta=24$): margine $\Delta/4 = 6$ contro
    $\lVert E\rVert_2 \approx 28$ ⇒ molti bit si ribaltano (BER misurato 0.37);
  - **RB** ($B=16$, $\Delta=128$): margine $32$ contro $\approx 40$ ⇒ errori
    sporadici, corretti dal voto di maggioranza (BER misurato 0.003).

  I numeri sperimentali della tabella di robustezza sono quindi **previsti**
  dalla teoria delle perturbazioni, non solo osservati.
- **Perché JPEG moderato non rompe RB.** L'errore di quantizzazione JPEG è
  concentrato sulle alte frequenze, cioè (nei blocchi naturali) sulle componenti
  associate ai $\sigma_i$ piccoli; la proiezione di $E$ su $u_1v_1^T$ è piccola
  e $\sigma_1$ si muove molto meno del bound $\lVert E\rVert_2$. Per qualità
  aggressive (q50) l'errore cresce e serve l'intero margine $\Delta/4=32$.

### 7.2 Dagli errori di bit alla ricostruzione: bilancio dell'errore

Detta $\hat S$ la ricostruzione finale, l'errore totale si decompone (per
disuguaglianza triangolare) in tre contributi:

$$\lVert S - \hat S\rVert_F \;\le\;
\underbrace{\lVert S - S_k\rVert_F}_{\text{troncamento}} +
\underbrace{\lVert S_k - \tilde S_k\rVert_F}_{\text{quantizzazione}} +
\underbrace{\lVert \tilde S_k - \hat S\rVert_F}_{\text{errori di bit}}.$$

1. **Troncamento** — deterministico e noto in anticipo: per Eckart–Young vale
   $\lVert S-S_k\rVert_F = (\sum_{i>k}\sigma_i^2)^{1/2}$, cioè $1-E(k)$ in
   energia. Con $k=10$ sul segreto $64\times64$, $E(k)=99.1\%$: è questo il
   *tetto* di fedeltà in assenza di attacchi (NC $\approx 0.92$, PSNR
   $\approx 24$ dB), che infatti coincide col valore misurato su immagini non
   attaccate.
2. **Quantizzazione dei fattori** — limitata e uniforme: l'errore per elemento
   di $U,V$ è $\le 1/254$, quello su $\sigma_i$ è l'errore relativo del
   float16 ($\approx 10^{-3}$); il contributo complessivo è trascurabile
   rispetto al troncamento.
3. **Errori di bit** — l'unico contributo *casuale*, e fortemente **non
   uniforme** rispetto alla posizione del bit:
   - un bit errato in $U_k$ o $V_k^T$ altera un solo elemento int8 ⇒ errore
     locale di norma $\le \sigma_i\cdot(2\cdot127)/127^2$: degrado graduale;
   - un bit errato negli **esponenti** di un $\sigma_i$ float16 può moltiplicare
     la componente per ordini di grandezza ⇒ potenzialmente catastrofico; la
     mitigazione è l'azzeramento dei $\sigma$ implausibili (la componente viene
     persa, non amplificata);
   - un bit errato nell'**header** invalida $(h,w,k)$ ⇒ fallimento totale della
     decodifica. Per questo l'header occupa i primi siti dell'ordinamento (i
     blocchi di sfondo più "sicuri") ed è protetto, in modalità RB, dalla
     ripetizione $R=3$.

   Questa gerarchia spiega la forma delle curve di robustezza: al crescere del
   BER la NC non degrada linearmente ma crolla appena vengono colpiti header o
   valori singolari — ed è la ragione teorica per cui la modalità robusta
   **richiede** il codice a ripetizione (§8.4).

---

## 8. Scelta dei parametri: motivazioni teoriche

### 8.1 Passo $\Delta$: distorsione prevista vs margine di robustezza

Il passo $\Delta$ governa *entrambi* i lati del compromesso, e il suo effetto è
interamente prevedibile. In embedding $\sigma_1$ si sposta di
$\delta = \sigma_1'-\sigma_1$; assumendo la fase $\sigma_1 \bmod \Delta$
uniforme (vero per $\sigma_1 \gg \Delta$) e bit equiprobabili,

$$\mathbb E[\delta^2] = \frac{1}{2\Delta}\int_0^\Delta\!\!
\big[(0.25\Delta-\varphi)^2 + (0.75\Delta-\varphi)^2\big]\,d\varphi
= \frac{7}{48}\,\Delta^2 .$$

Poiché la modifica di norma $|\delta|$ si spalma su $B^2$ pixel (§6.1) e solo
una frazione $\alpha$ dei blocchi porta bit, l'MSE per pixel atteso è
$\alpha\,\tfrac{7}{48}\Delta^2/B^2$ e quindi

$$\mathrm{PSNR}_{\text{atteso}} = 10\log_{10}
\frac{255^2\,B^2}{\alpha\,\tfrac{7}{48}\,\Delta^2}.$$

Verifica sui due punti operativi (cover $512\times512$, 3 canali):

| | $\alpha$ (siti usati) | PSNR atteso | PSNR misurato |
|---|---|---|---|
| HC ($B=8,\ \Delta=24$) | $10472/12288 = 0.85$ | 47.6 dB | 47.4 dB |
| RB ($B=16,\ \Delta=128$) | $2664/3072 = 0.87$ | 39.1 dB | 39.0 dB |

L'accordo (scarto < 0.3 dB, dovuto ad arrotondamento e clipping a uint8)
conferma che la distorsione è governata dalla sola teoria. Sul lato robustezza,
la condizione di Weyl $\Delta/4 > \mathbb E\lVert E\rVert_2 \approx
2\sigma_{\text{noise}}\sqrt B$ dà la regola di progetto
$\Delta > 8\,\sigma_{\text{noise}}\sqrt B$: RB ($\Delta=128$, $B=16$) tollera
per costruzione rumore con $\sigma_{\text{noise}} \approx 4$, HC
($\Delta=24$) solo $\approx 0.75$ — la fragilità di HC è una scelta, il prezzo
del massimo PSNR/capacità.

### 8.2 Lato del blocco $B$

- **Capacità:** i siti sono $3(N/B)^2$ — dimezzare $B$ quadruplica la capacità
  ($B=8$: 12288 siti; $B=16$: 3072).
- **Robustezza:** per un blocco quasi costante di livello $c$, $\sigma_1
  \approx cB$ cresce **linearmente** in $B$, mentre la perturbazione attesa
  $2\sigma_{\text{noise}}\sqrt B$ cresce solo come $\sqrt B$: blocchi più
  grandi permettono passi $\Delta$ più ampi (in rapporto a $\sigma_1$) e sono
  intrinsecamente più stabili. Inoltre $u_1v_1^T$ di un blocco grande è una
  componente a frequenza ancora più bassa, che sopravvive meglio a blur e JPEG.

Da qui $B=8$ per la modalità capacità e $B=16$ per quella robusta.

### 8.3 Rango $k$ (e dimensione) del segreto

Due vincoli determinano $k$:

- **Criterio dell'energia (dal corso):** il più piccolo $k$ con
  $E(k)\ge$ soglia. Per il segreto $64\times64$ la tabella
  `results/secret_compression.csv` dà $E(8)=98.7\%$, $E(10)=99.1\%$,
  $E(12)=99.4\%$: **$k=10$ è il più piccolo rango con energia $\ge 99\%$**.
- **Vincolo di capacità:** il payload deve stare nel cover,
  $8\,[9+2k+k(h+w)]\cdot R \le 3(N/B)^2$. In HC ($R=1$, $B=8$): $k=10$ richiede
  10472 bit su 12288 (utilizzo 85 %), mentre $k=12$ ne richiederebbe 12552 —
  non entra. In RB il vincolo è ancora più stringente per via di $R=3$: con
  segreto $16\times16$, $k=3$ richiede $888\times3 = 2664$ bit su 3072, mentre
  $k=4$ ne richiederebbe $1160\times3 = 3480$ — non entra. **In HC $k$ è scelto
  dall'energia, in RB è il massimo rango compatibile con la ridondanza.**

### 8.4 Fattore di ripetizione $R$

$R$ è **dispari** perché il voto di maggioranza non ammetta pareggi. Se $p$ è la
probabilità di errore del singolo sito, l'errore sul bit logico con $R=3$ è

$$P_{\text{err}} = 3p^2(1-p) + p^3 \approx 3p^2 \quad (p\ \text{piccolo}):$$

un canale con $p=5\%$ scende allo 0.7 %. È il salto qualitativo visibile nella
tabella di robustezza (BER HC vs RB), ottenuto al costo di $1/3$ della
capacità. $R=3$ è il minimo dispari $>1$: valori maggiori renderebbero il
payload RB incompatibile con la capacità (§8.3).

---

## 9. Compromesso capacità ↔ robustezza

Tre leve, tutte di natura SVD e tutte quantificate in §8:

- **Dimensione del blocco $B$** (§8.2). Blocchi più grandi ⇒ $\sigma_1$ più
  grande e più "low-frequency" ⇒ più robusto agli attacchi, ma meno blocchi ⇒
  meno capacità.
- **Passo $\Delta$** (§8.1). $\Delta$ grande ⇒ margine $\Delta/4$ grande ⇒
  robusto, ma maggiore distorsione (PSNR più basso, in modo esattamente
  prevedibile).
- **Codice a ripetizione $R$** (§8.4). Ogni bit logico è scritto in $R$ siti e
  decodificato a maggioranza ⇒ robustezza a costo di capacità $/R$.

Le due modalità del progetto (High-Capacity e Robust) sono due punti su questo
fronte di Pareto, quantificati in `results/figures/fig_tradeoff.png`.
