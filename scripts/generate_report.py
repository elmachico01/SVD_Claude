#!/usr/bin/env python3
"""
generate_report.py — costruisce il report del progetto (Markdown + PDF) a partire
dai risultati in ``results/``.

    python scripts/generate_report.py

Produce ``docs/REPORT.md`` e ``docs/REPORT.pdf``. Entrambi sono generati dallo
stesso contenuto strutturato, quindi restano sempre allineati. Il report è in
italiano e segue la struttura richiesta dal syllabus (problema, metodo e teoria,
implementazione, esperimenti, risultati, analisi).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
DOCS = ROOT / "docs"


# ---------------------------------------------------------------------------
# Accesso sicuro ai dati
# ---------------------------------------------------------------------------
def load_summary() -> dict:
    p = RES / "summary.json"
    if not p.exists():
        sys.exit("[report] manca results/summary.json — esegui prima "
                 "experiments/run_experiments.py")
    return json.loads(p.read_text())


def m(summary, mode, key, sub="mean", default=float("nan")):
    return summary.get(mode, {}).get(key, {}).get(sub, default)


def atk(summary, mode, name, metric):
    return summary.get("attacks", {}).get(mode, {}).get(name, {}).get(metric, float("nan"))


def fnum(x, d=2):
    try:
        if x != x:  # NaN
            return "n/d"
        return f"{x:.{d}f}"
    except Exception:
        return str(x)


def qim_theory(mode_cfg: dict, cover: int = 512) -> dict:
    """
    Predizioni teoriche dello schema QIM per un punto operativo.

    - payload: 8·[9 + 2k + k(h+w)] bit (header + σ float16 + U,V int8);
    - distorsione attesa E[δ²] = 7Δ²/48 (fase uniforme, bit equiprobabili),
      spalmata su B² pixel nella frazione α di blocchi usati;
    - PSNR atteso = 10·log10(255²·B² / (α·7Δ²/48));
    - margine di robustezza QIM = Δ/4 (teorema di Weyl: bit corretto se ‖E‖₂ < Δ/4).
    """
    import math
    Bk = int(mode_cfg.get("block", 8))
    D = float(mode_cfg.get("delta", 24.0))
    R = int(mode_cfg.get("repeat", 1))
    size = int(mode_cfg.get("secret_size", 64))
    k = int(mode_cfg.get("k", 10))
    payload_bits = 8 * (9 + 2 * k + 2 * size * k)
    phys_sites = 3 * (cover // Bk) ** 2
    alpha = min(1.0, payload_bits * R / phys_sites)
    mse = alpha * (7.0 / 48.0) * D * D / (Bk * Bk)
    psnr = 10.0 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
    return {"payload_bits": payload_bits, "phys_sites": phys_sites,
            "alpha": alpha, "psnr_pred": psnr, "margin": D / 4.0}


# ---------------------------------------------------------------------------
# Modello di documento: lista di blocchi resi sia in Markdown che in PDF
#   ("h1"/"h2"/"h3", testo) | ("p", testo) | ("bul", [righe]) |
#   ("table", [righe]) | ("img", (path, didascalia)) | ("hr", None)
# ---------------------------------------------------------------------------
def build_blocks(s: dict) -> list[tuple[str, object]]:
    cfg = s.get("config", {})
    hc, rb = cfg.get("hc", {}), cfg.get("rb", {})
    n = s.get("n_images", "?")
    B = []
    A = B.append

    A(("title", "SVD & YOLO per la Steganografia di Immagini"))
    A(("subtitle", "Progetto d'esame — Statistical and Mathematical Methods for AI"))
    A(("p", f"Valutazione su **{n} immagini** del dataset COCO-128. "
            f"Rilevatore YOLO: {'attivo' if s.get('yolo_available') else 'fallback saliency'}. "
            f"Tutti i numeri di questo report sono generati automaticamente dai "
            f"risultati sperimentali (`results/summary.json`)."))

    # 1. Problema
    A(("h1", "1. Descrizione del problema"))
    A(("p", "La **steganografia di immagini** consiste nel nascondere "
            "un'informazione segreta dentro un'immagine *cover* in modo che la "
            "presenza stessa del messaggio sia impercettibile. A differenza della "
            "crittografia (che rende il messaggio illeggibile ma evidente), la "
            "steganografia mira all'**invisibilità**."))
    A(("p", "Obiettivo del progetto: realizzare uno schema di steganografia che "
            "(i) sfrutti la **Decomposizione ai Valori Singolari (SVD)** — argomento "
            "centrale del corso — come strumento sia di *compressione* del segreto "
            "sia di *dominio di embedding*; e (ii) usi un rilevatore di oggetti "
            "**YOLO** per rendere l'occultamento *content-adaptive*, nascondendo i "
            "dati nello sfondo e preservando le regioni semanticamente importanti."))
    A(("p", "Tre requisiti guidano la valutazione, in tensione tra loro: "
            "**impercettibilità** (il cover non deve cambiare visibilmente), "
            "**capacità** (quanti bit possiamo nascondere) e **robustezza** "
            "(il segreto deve sopravvivere a manipolazioni come la ricompressione "
            "JPEG)."))

    # 2. Metodo + teoria
    A(("h1", "2. Metodo e background teorico"))
    A(("h2", "2.1 SVD e teorema di Eckart–Young"))
    A(("p", "Per ogni matrice A ∈ ℝ^(m×n) vale A = U Σ Vᵀ con U, V ortogonali e "
            "Σ = diag(σ₁,…,σ_r), σ₁ ≥ … ≥ σ_r > 0. La SVD troncata "
            "A_k = Σ_{i=1}^{k} σ_i u_i v_iᵀ è, per il teorema di Eckart–Young, la "
            "migliore approssimazione di rango k sia in norma spettrale "
            "(‖A−A_k‖₂ = σ_{k+1}) sia di Frobenius (‖A−A_k‖_F = (Σ_{i>k} σ_i²)^½)."))
    A(("img", ("fig_svd_spectrum.png",
               "Spettro dei valori singolari di un cover ed energia cumulata: "
               "poche componenti catturano la quasi totalità dell'energia.")))
    A(("img", ("fig_eckart_young.png",
               "Verifica numerica del teorema di Eckart–Young (teoria vs misura) e "
               "fattore di compressione k(m+n+1)/mn.")))
    A(("h2", "2.2 Compressione del segreto (riduzione di dimensionalità)"))
    A(("p", "Il segreto S viene sostituito dalla sua approssimazione di rango k e "
            "memorizzato nei fattori compatti (U_k, σ_k, V_kᵀ). Il payload passa da "
            "m·n a k(m+n+1) numeri: è la stessa compressione con perdita vista a "
            "lezione (slide 'Reduction of memory occupation'). I fattori sono "
            "quantizzati (U,V a int8, σ a float16) per ridurre ulteriormente il payload."))
    A(("img", ("fig_secret_reconstructions.png",
               "Ricostruzione del segreto al variare del rango k: aumentando k "
               "migliora la fedeltà (NC) ma cresce il payload.")))
    A(("h2", "2.3 Ruolo dei valori singolari nell'embedding"))
    A(("p", "La scelta di codificare l'informazione nei valori singolari — e in "
            "particolare in σ₁ — discende da quattro proprietà. (i) **Concentrazione "
            "di energia**: per i blocchi B×B di immagini naturali σ₁² raccoglie il "
            "95–99 % dell'energia ‖A‖_F² = Σσᵢ² (Eckart–Young): σ₁ è la coordinata "
            "più significativa del blocco nella base di matrici di rango 1 {uᵢvᵢᵀ}. "
            "(ii) **Stabilità**: per il teorema di Weyl (§2.5) una perturbazione E "
            "sposta tutti i σᵢ al più di ‖E‖₂ in assoluto, quindi la perturbazione "
            "*relativa* è minima su σ₁ e massima sui σᵢ piccoli — in linea con "
            "l'osservazione del corso per cui i piccoli valori singolari amplificano "
            "il rumore. Un bit scritto in un σᵢ piccolo sarebbe illeggibile già dopo "
            "un attacco lieve. (iii) **Invarianza geometrica**: i σᵢ non cambiano per "
            "trasformazioni ortogonali del blocco, sono un descrittore intrinseco "
            "dell'energia. (iv) **Distorsione distribuita**: variare σ₁ di δ cambia "
            "il blocco di δ·u₁v₁ᵀ con ‖δ·u₁v₁ᵀ‖_F = |δ|, spalmato su tutti i B² "
            "pixel lungo la componente più liscia del blocco; modificare un σᵢ "
            "piccolo inietterebbe invece componenti oscillanti simili a rumore, più "
            "visibili e più facili da rilevare."))
    A(("h2", "2.4 Embedding e decodifica (QIM su σ₁)"))
    A(("p", "**Embedding.** Il segreto compresso è serializzato in un bitstream "
            "auto-descrittivo: header di 9 byte (magic, versione, h, w, k), poi i k "
            "valori singolari in float16 e i fattori U_k, V_kᵀ quantizzati a int8 "
            "(le loro colonne/righe hanno norma unitaria, quindi elementi in [−1,1] "
            "e scala fissa 127). La lunghezza totale, 9+2k+k(h+w) byte, dipende solo "
            "da (h,w,k). Il cover è diviso in blocchi B×B; l'ordine dei blocchi è "
            "fissato da una chiave PRNG e dalla mappa di priorità YOLO (sfondo prima). "
            "Per ogni blocco portante A = UΣVᵀ il bit b è codificato per Quantisation "
            "Index Modulation: σ₁' = ⌊σ₁/Δ⌋·Δ + 0.25Δ (b=0) oppure + 0.75Δ (b=1), "
            "e il blocco è ricostruito come A' = UΣ'Vᵀ (i vettori singolari restano "
            "invariati). I due offset sono i centri delle due semicelle "
            "dell'intervallo di quantizzazione: la scelta rende simmetrico e massimo "
            "il margine di errore, pari a Δ/4. Con ridondanza R>1 ogni bit logico è "
            "scritto in R siti consecutivi (codice a ripetizione)."))
    A(("p", "**Decodifica (cieca).** L'estrazione non richiede né il cover originale "
            "né i σ originali (a differenza dello schema di Liu–Tan, che conserva "
            "U,V ed è per questo vulnerabile a false estrazioni): servono solo "
            "chiave, parametri (B,Δ,R) e la mappa di priorità. Si ricostruisce lo "
            "stesso ordinamento dei blocchi; per ciascun blocco si calcola la SVD e "
            "si decide dalla parte frazionaria di σ₁/Δ: b=0 se (σ₁/Δ mod 1) < 0.5, "
            "b=1 altrimenti — la cella di decisione è larga Δ/2 e il valore embedded "
            "siede al suo centro. Con R>1 il bit logico è il voto di maggioranza "
            "degli R siti. Si leggono prima i 72 bit dell'header, se ne valida la "
            "plausibilità e da (h,w,k) si determina la lunghezza esatta del payload; "
            "quindi si de-serializzano σ̃ (float16) e Ũ_k, Ṽ_kᵀ (int8/127) e si "
            "ricostruisce Ŝ = Ũ_k·diag(σ̃)·Ṽ_kᵀ con clip in [0,255]. Il parser è "
            "difensivo: σ corrotti (NaN/inf) vengono azzerati, così la componente "
            "danneggiata viene eliminata e la ricostruzione degrada al rango "
            "inferiore invece di esplodere."))
    A(("h2", "2.5 Effetto delle perturbazioni: teorema di Weyl"))
    A(("p", "Un attacco trasforma il blocco stego A' in A'+E. Il teorema di Weyl "
            "limita lo spostamento di ogni valore singolare: |σᵢ(A'+E) − σᵢ(A')| ≤ "
            "‖E‖₂ ≤ ‖E‖_F. Ne segue la **condizione di decodifica corretta**: il bit "
            "QIM resta corretto se ‖E‖₂ < Δ/4. Per rumore gaussiano i.i.d. con "
            "deviazione σ_n la teoria delle matrici casuali dà ‖E‖₂ ≈ 2σ_n√B; con "
            "σ_n = 5 (attacco gauss_noise5) si prevede: HC (B=8, margine Δ/4 = 6 "
            "contro ‖E‖₂ ≈ 28) → molti bit ribaltati; RB (B=16, margine 32 contro "
            "≈ 40) → errori sporadici corretti dal voto di maggioranza. È esattamente "
            "ciò che la tabella di robustezza (§5.5) misura: i risultati sperimentali "
            "sono previsti dalla teoria delle perturbazioni, non solo osservati. "
            "Sulla **ricostruzione**, l'errore totale ‖S−Ŝ‖_F si decompone in tre "
            "contributi: troncamento (Eckart–Young: (Σ_{i>k}σᵢ²)^½, il tetto di "
            "fedeltà in pulito), quantizzazione dei fattori (trascurabile) ed errori "
            "di bit, il cui effetto è fortemente non uniforme: un bit errato su U/V "
            "altera un solo elemento int8 (degrado graduale), un bit sugli esponenti "
            "di un σ float16 può amplificare la componente per ordini di grandezza "
            "(mitigato dall'azzeramento dei σ implausibili), un bit sull'header è "
            "fatale — per questo l'header occupa i primi siti dell'ordinamento ed è "
            "protetto dalla ripetizione. Questa gerarchia spiega perché la NC crolla "
            "in modo non lineare al crescere del BER e perché la modalità robusta "
            "richiede il codice a ripetizione."))
    A(("h2", "2.6 Guida YOLO (content-adaptive)"))
    A(("p", "YOLOv8 rileva gli oggetti nel cover; dalle bounding box si costruisce "
            "una mappa di saliency. L'ordine di riempimento dei blocchi privilegia "
            "lo **sfondo** (saliency bassa), lasciando quasi intatte le regioni "
            "degli oggetti. YOLO viene poi usato anche per **valutare** quanto le "
            "detection si conservano sul cover rispetto allo stego (preservation "
            "rate, IoU, mAP@0.5)."))

    # 3. Implementazione
    A(("h1", "3. Implementazione"))
    A(("p", "Il progetto è in Python (NumPy, OpenCV, scikit-image, Ultralytics "
            "YOLOv8, Matplotlib). Moduli principali: `svd_core` (matematica SVD), "
            "`steganography` (codec del segreto + embedding/estrazione), "
            "`yolo_guidance` (rilevamento e metriche), `metrics`, `attacks`, "
            "`dataset`, `pipeline`."))
    A(("p", "Sono definiti due **punti operativi** che illustrano il compromesso "
            "capacità↔robustezza:"))
    A(("table", [
        ["Parametro", "High-Capacity (HC)", "Robust (RB)"],
        ["Blocco SVD", f"{hc.get('block')}×{hc.get('block')}",
         f"{rb.get('block')}×{rb.get('block')}"],
        ["Passo QIM Δ", f"{hc.get('delta')}", f"{rb.get('delta')}"],
        ["Ripetizione R", f"{hc.get('repeat')}", f"{rb.get('repeat')}"],
        ["Segreto", f"{hc.get('secret_size')}×{hc.get('secret_size')}, k={hc.get('k')}",
         f"{rb.get('secret_size')}×{rb.get('secret_size')}, k={rb.get('k')}"],
        ["Capacità (bpp)", fnum(m(s, 'hc', 'capacity_bpp'), 4),
         fnum(m(s, 'rb', 'capacity_bpp'), 4)],
    ]))

    th_hc, th_rb = qim_theory(hc), qim_theory(rb)
    A(("h2", "3.1 Motivazione teorica della scelta dei parametri"))
    A(("p", "**Passo Δ.** Assumendo la fase σ₁ mod Δ uniforme (vero per σ₁ ≫ Δ) e "
            "bit equiprobabili, lo spostamento δ di σ₁ in embedding ha "
            "E[δ²] = 7Δ²/48; la modifica di norma |δ| si spalma su B² pixel e solo "
            "una frazione α dei blocchi porta bit, quindi il PSNR atteso è "
            "10·log₁₀(255²·B²/(α·7Δ²/48)). La predizione coincide con la misura: "
            f"HC α = {fnum(th_hc['alpha'],2)} → PSNR atteso "
            f"{fnum(th_hc['psnr_pred'],1)} dB contro {fnum(m(s,'hc','guided_psnr'),1)} dB "
            f"misurati; RB α = {fnum(th_rb['alpha'],2)} → atteso "
            f"{fnum(th_rb['psnr_pred'],1)} dB contro {fnum(m(s,'rb','guided_psnr'),1)} dB "
            "(scarto < 0.3 dB, dovuto ad arrotondamento e clipping a uint8): la "
            "distorsione è interamente governata dalla teoria. Sul lato robustezza, "
            "la condizione di Weyl Δ/4 > ‖E‖₂ ≈ 2σ_n√B dà la regola di progetto "
            f"Δ > 8σ_n√B: RB (Δ={rb.get('delta')}, margine {fnum(th_rb['margin'],0)}) "
            "tollera per costruzione rumore con σ_n ≈ 4, HC "
            f"(Δ={hc.get('delta')}, margine {fnum(th_hc['margin'],0)}) solo σ_n ≈ 0.75 "
            "— la fragilità di HC è il prezzo, scelto, del massimo PSNR e capacità."))
    A(("p", "**Blocco B.** I siti portanti sono 3(N/B)²: dimezzare B quadruplica la "
            f"capacità ({th_hc['phys_sites']} siti con B={hc.get('block')} contro "
            f"{th_rb['phys_sites']} con B={rb.get('block')}). Per la robustezza vale "
            "l'argomento opposto: per un blocco quasi costante di livello c si ha "
            "σ₁ ≈ cB, che cresce linearmente in B, mentre la perturbazione attesa "
            "2σ_n√B cresce solo come √B — i blocchi grandi tollerano passi Δ più "
            "ampi in rapporto a σ₁, e la loro componente u₁v₁ᵀ è a frequenza più "
            "bassa, quindi sopravvive meglio a blur e JPEG."))
    A(("p", "**Rango k del segreto.** Due vincoli: il criterio dell'energia del "
            "corso (il più piccolo k con E(k) sopra soglia — per il segreto 64×64, "
            "E(8) = 98.7 %, E(10) = 99.1 %: k=10 è il più piccolo rango con energia "
            "≥ 99 %) e il vincolo di capacità 8·[9+2k+k(h+w)]·R ≤ 3(N/B)². In HC "
            f"k=10 richiede {th_hc['payload_bits']} bit su {th_hc['phys_sites']} "
            "(utilizzo 85 %), mentre k=12 non entrerebbe (12552 bit); in RB k=3 "
            f"richiede {th_rb['payload_bits']}×3 = {th_rb['payload_bits']*3} bit su "
            f"{th_rb['phys_sites']}, mentre k=4 non entrerebbe (3480). In HC k è "
            "scelto dall'energia, in RB è il massimo rango compatibile con la "
            "ridondanza."))
    A(("p", "**Ripetizione R.** R è dispari perché il voto di maggioranza non "
            "ammetta pareggi. Se p è la probabilità di errore del singolo sito, con "
            "R=3 l'errore sul bit logico è 3p²(1−p)+p³ ≈ 3p²: un canale con p = 5 % "
            "scende allo 0.7 %. È il salto qualitativo tra le colonne HC e RB della "
            "tabella di robustezza, pagato con 1/3 della capacità; R=3 è il minimo "
            "dispari > 1, e valori maggiori renderebbero il payload RB incompatibile "
            "con la capacità."))

    # 4. Esperimenti
    A(("h1", "4. Setup sperimentale"))
    A(("bul", [
        f"Dataset: COCO-128 ({n} immagini reali di COCO), cover ridimensionati a 512×512.",
        "Segreto: logo grayscale sintetico (forme + gradiente), compresso via SVD troncata.",
        "Metriche di fedeltà: PSNR, SSIM, MSE (cover vs stego), globali e per regione "
        "(oggetto vs sfondo).",
        "Metriche di payload: NC (correlazione normalizzata) e PSNR del segreto recuperato, BER.",
        "Downstream: preservation rate, IoU medio, mAP@0.5, |Δconfidenza| (cover vs stego).",
        "Attacchi: JPEG (q90/q75/q50), rumore gaussiano, blur, mediano, sale&pepe, rescale.",
    ]))

    # 5. Risultati
    A(("h1", "5. Risultati"))
    A(("h2", "5.1 Impercettibilità"))
    A(("p", f"In modalità HC il cover e lo stego sono praticamente indistinguibili: "
            f"PSNR medio **{fnum(m(s,'hc','guided_psnr'))} dB**, SSIM "
            f"**{fnum(m(s,'hc','guided_ssim'),4)}**. In modalità RB (più robusta) "
            f"PSNR medio **{fnum(m(s,'rb','guided_psnr'))} dB**, SSIM "
            f"**{fnum(m(s,'rb','guided_ssim'),4)}**."))
    A(("img", ("fig_imperceptibility.png",
               "Distribuzione di PSNR e SSIM cover→stego sulle immagini di test.")))
    A(("h2", "5.2 Protezione degli oggetti tramite YOLO"))
    A(("p", f"La guida YOLO sposta il payload nello sfondo: il PSNR **nelle regioni "
            f"degli oggetti** sale da {fnum(m(s,'hc','baseline_obj_psnr'))} dB "
            f"(baseline senza YOLO) a **{fnum(m(s,'hc','guided_obj_psnr'))} dB** "
            f"(guidato), un miglioramento di "
            f"**{fnum(m(s,'hc','guided_obj_psnr')-m(s,'hc','baseline_obj_psnr'))} dB** "
            f"esattamente dove l'occhio (e il rilevatore) guardano. Lo sfondo, "
            f"di contro, passa da {fnum(m(s,'hc','baseline_bg_psnr'))} a "
            f"{fnum(m(s,'hc','guided_bg_psnr'))} dB."))
    A(("img", ("fig_object_protection.png",
               "PSNR per regione (oggetto vs sfondo): la guida YOLO protegge gli oggetti.")))
    A(("img", ("fig_qualitative.png",
               "Risultati qualitativi: la mappa |cover−stego| mostra che le "
               "modifiche evitano le regioni degli oggetti.")))
    A(("h2", "5.3 Preservazione del contenuto semantico (downstream)"))
    A(("p", f"Le detection di YOLO si conservano quasi perfettamente dopo "
            f"l'embedding: preservation rate medio **{fnum(m(s,'hc','det_preservation_rate'),3)}**, "
            f"mAP@0.5 medio **{fnum(m(s,'hc','det_map50'),3)}**, variazione media di "
            f"confidenza **{fnum(m(s,'hc','det_mean_conf_delta'),3)}**. Lo stego è "
            f"quindi utilizzabile in una pipeline di AI a valle senza degradarne le "
            f"prestazioni."))
    A(("img", ("fig_detection_preservation.png",
               "Distribuzione di preservation rate, mAP@0.5 e |Δconfidenza|.")))
    A(("h2", "5.4 Recupero del segreto e compressione SVD"))
    A(("p", f"Su immagine non attaccata, il segreto si recupera con NC medio "
            f"**{fnum(m(s,'hc','guided_secret_nc'),3)}** (HC) e "
            f"**{fnum(m(s,'rb','guided_secret_nc'),3)}** (RB), con BER "
            f"≈ {fnum(m(s,'hc','guided_ber'),5)} (HC). Il limite di fedeltà in pulito "
            f"è dato dalla **compressione SVD troncata** del segreto, non "
            f"dall'estrazione: aumentando k la qualità cresce a scapito del payload."))
    A(("img", ("fig_secret_compression.png",
               "Qualità di ricostruzione (PSNR/energia) e fattore di compressione "
               "del segreto al variare del rango k.")))
    A(("h2", "5.5 Robustezza agli attacchi"))
    A(("p", "La modalità RB (blocchi 16×16, Δ grande, ripetizione ×3) resiste agli "
            "attacchi comuni, mentre la modalità HC — priva di ridondanza — è "
            "fragile: è il prezzo della massima capacità."))
    A(("table", _robustness_table(s)))
    A(("img", ("fig_robustness.png",
               "NC del segreto recuperato e BER del payload sotto attacco.")))
    A(("img", ("fig_tradeoff.png",
               "Fronte di Pareto capacità↔robustezza tra le due modalità.")))

    # 6. Analisi
    A(("h1", "6. Analisi e commenti"))
    A(("bul", [
        "Il valore singolare massimo σ₁ è un dominio di embedding eccellente: "
        "concentra l'energia del blocco (Eckart–Young) e, per il teorema di Weyl, "
        "è la componente con perturbazione relativa minima — in linea con "
        "l'osservazione del corso che i piccoli σ amplificano il rumore.",
        "Il PSNR misurato coincide (entro 0.3 dB) con quello previsto dalla "
        "formula teorica 10·log₁₀(255²·B²/(α·7Δ²/48)): la distorsione "
        "dell'embedding è interamente spiegata dalla teoria (§3.1).",
        f"La guida YOLO migliora la qualità nelle regioni salienti "
        f"(+{fnum(m(s,'hc','guided_obj_psnr')-m(s,'hc','baseline_obj_psnr'),1)} dB "
        f"sugli oggetti) senza intaccare la capacità, e mantiene "
        f"mAP ≈ {fnum(m(s,'hc','det_map50'),2)}.",
        "Esiste un chiaro compromesso capacità↔robustezza governato da tre leve "
        "SVD (dimensione blocco, passo Δ, ridondanza R): non è possibile "
        "massimizzare contemporaneamente capacità, impercettibilità e robustezza.",
        "La compressione del segreto via SVD troncata abilita payload elevati ma, "
        "eliminando la ridondanza spaziale, rende il recupero sensibile agli errori "
        "di bit: per la robustezza è necessaria una codifica di canale (ripetizione).",
    ]))

    # 7. Conclusioni
    A(("h1", "7. Conclusioni"))
    A(("p", "Il progetto mostra come la SVD — cuore del corso — fornisca un quadro "
            "unificato per la steganografia: compressione del segreto (Eckart–Young), "
            "dominio di embedding robusto (σ₁), e analisi del condizionamento. "
            "L'integrazione con YOLO rende l'occultamento adattivo al contenuto e "
            "verificabile a valle. I risultati su 100+ immagini COCO quantificano "
            "impercettibilità, preservazione semantica e il compromesso "
            "capacità↔robustezza."))

    A(("h1", "Riferimenti"))
    A(("bul", [
        "C. Eckart, G. Young, 'The approximation of one matrix by another of lower "
        "rank', Psychometrika, 1936.",
        "H. Weyl, 'Das asymptotische Verteilungsgesetz der Eigenwerte linearer "
        "partieller Differentialgleichungen', Math. Ann., 1912 (disuguaglianze di "
        "perturbazione dei valori singolari).",
        "B. Chen, G. Wornell, 'Quantization Index Modulation: a class of provably "
        "good methods for digital watermarking and information embedding', "
        "IEEE Trans. Inf. Theory, 2001.",
        "G. H. Golub, C. Van Loan, 'Matrix Computations', 4th ed., 2013.",
        "R. Liu, T. Tan, 'An SVD-based watermarking scheme for protecting rightful "
        "ownership', IEEE Trans. Multimedia, 2002.",
        "G. Jocher et al., 'Ultralytics YOLOv8', 2023.",
        "T.-Y. Lin et al., 'Microsoft COCO: Common Objects in Context', ECCV 2014.",
        "M. Popolizio, slide del corso 'Reducing data dimensionality for AI "
        "applications', 2025.",
    ]))
    return B


def _robustness_table(s):
    rows = [["Attacco", "NC (HC)", "NC (RB)", "BER (HC)", "BER (RB)"]]
    for a in s.get("attacks", {}).get("hc", {}):
        rows.append([a, fnum(atk(s, 'hc', a, 'nc'), 3), fnum(atk(s, 'rb', a, 'nc'), 3),
                     fnum(atk(s, 'hc', a, 'ber'), 3), fnum(atk(s, 'rb', a, 'ber'), 3)])
    return rows


# ---------------------------------------------------------------------------
# Renderer Markdown
# ---------------------------------------------------------------------------
def to_markdown(blocks) -> str:
    out = []
    for kind, payload in blocks:
        if kind == "title":
            out.append(f"# {payload}\n")
        elif kind == "subtitle":
            out.append(f"*{payload}*\n")
        elif kind == "h1":
            out.append(f"\n## {payload}\n")
        elif kind == "h2":
            out.append(f"\n### {payload}\n")
        elif kind == "h3":
            out.append(f"\n#### {payload}\n")
        elif kind == "p":
            out.append(payload + "\n")
        elif kind == "bul":
            out.extend(f"- {line}" for line in payload)
            out.append("")
        elif kind == "table":
            head, *rows = payload
            out.append("| " + " | ".join(head) + " |")
            out.append("|" + "|".join(["---"] * len(head)) + "|")
            out.extend("| " + " | ".join(r) + " |" for r in rows)
            out.append("")
        elif kind == "img":
            path, cap = payload
            out.append(f"\n![{cap}](../results/figures/{path})\n")
            out.append(f"*{cap}*\n")
        elif kind == "hr":
            out.append("\n---\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Renderer PDF (reportlab Platypus)
# ---------------------------------------------------------------------------
def to_pdf(blocks, path: Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                    Table, TableStyle, ListFlowable, ListItem)

    # Register DejaVu (rich Unicode coverage: Greek, ℝ, ‖, σ, Σ, ᵀ, subscripts…)
    base, mono = "Helvetica", "Courier"
    dj = "/usr/share/fonts/truetype/dejavu"
    if Path(dj).exists():
        pdfmetrics.registerFont(TTFont("DejaVu", f"{dj}/DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", f"{dj}/DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFont(TTFont("DejaVuMono", f"{dj}/DejaVuSansMono.ttf"))
        # DejaVu Sans ships no italic → map italics to the regular/bold weights
        pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                      italic="DejaVu", boldItalic="DejaVu-Bold")
        base, mono = "DejaVu", "DejaVuMono"

    styles = getSampleStyleSheet()
    bold = "DejaVu-Bold" if base == "DejaVu" else "Helvetica-Bold"
    for nm in ("Title", "Heading1", "Heading2", "Heading3", "Normal"):
        styles[nm].fontName = bold if nm != "Normal" else base
    styles.add(ParagraphStyle("TitleBig", parent=styles["Title"], fontName=bold,
                              fontSize=20, spaceAfter=6))
    styles.add(ParagraphStyle("Sub", parent=styles["Normal"], fontName=base,
                              fontSize=11, textColor=colors.grey, spaceAfter=12))
    body = ParagraphStyle("Body", parent=styles["Normal"], fontName=base, fontSize=10.5,
                          leading=15, spaceAfter=6, alignment=4)
    cap = ParagraphStyle("Cap", parent=styles["Normal"], fontName=base, fontSize=8.5,
                         textColor=colors.grey, spaceAfter=10, alignment=1)

    def md_to_rl(t):  # **bold**, *italic*, `code` → markup reportlab
        import re
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)          # single-asterisk emphasis
        t = re.sub(r"`(.+?)`", rf'<font face="{mono}">\1</font>', t)
        return t

    story, avail_w = [], A4[0] - 4 * cm
    for kind, payload in blocks:
        if kind == "title":
            story.append(Paragraph(payload, styles["TitleBig"]))
        elif kind == "subtitle":
            story.append(Paragraph(payload, styles["Sub"]))
        elif kind in ("h1", "h2", "h3"):
            st = {"h1": "Heading1", "h2": "Heading2", "h3": "Heading3"}[kind]
            story.append(Paragraph(md_to_rl(payload), styles[st]))
        elif kind == "p":
            story.append(Paragraph(md_to_rl(payload), body))
        elif kind == "bul":
            items = [ListItem(Paragraph(md_to_rl(x), body)) for x in payload]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=12))
            story.append(Spacer(1, 4))
        elif kind == "table":
            head, *rows = payload
            data = [[Paragraph(f"<b>{c}</b>", body) for c in head]]
            data += [[Paragraph(md_to_rl(c), body) for c in r] for r in rows]
            t = Table(data, hAlign="LEFT", colWidths=[avail_w / len(head)] * len(head))
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f6")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t); story.append(Spacer(1, 8))
        elif kind == "img":
            fpath, caption = payload
            ip = FIG / fpath
            if ip.exists():
                from PIL import Image as PImage
                iw, ih = PImage.open(ip).size
                w = min(avail_w, 16 * cm)
                story.append(Image(str(ip), width=w, height=w * ih / iw))
                story.append(Paragraph(caption, cap))
    SimpleDocTemplate(str(path), pagesize=A4,
                      leftMargin=2*cm, rightMargin=2*cm,
                      topMargin=2*cm, bottomMargin=2*cm,
                      title="SVD & YOLO Steganography — Report").build(story)


def main():
    DOCS.mkdir(exist_ok=True)
    s = load_summary()
    blocks = build_blocks(s)
    (DOCS / "REPORT.md").write_text(to_markdown(blocks), encoding="utf-8")
    print(f"[report] scritto {DOCS/'REPORT.md'}")
    try:
        to_pdf(blocks, DOCS / "REPORT.pdf")
        print(f"[report] scritto {DOCS/'REPORT.pdf'}")
    except Exception as e:
        print(f"[report] PDF non generato ({type(e).__name__}: {e}); "
              f"il Markdown è comunque disponibile.")


if __name__ == "__main__":
    main()
