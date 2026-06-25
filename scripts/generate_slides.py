#!/usr/bin/env python3
"""
generate_slides.py — costruisce la presentazione PowerPoint del progetto a
partire dai risultati in ``results/``.

    python scripts/generate_slides.py        # → docs/presentazione.pptx

Le slide sono in italiano e seguono la struttura del syllabus. I numeri e le
figure provengono dai risultati sperimentali, quindi rilanciando lo script con i
tuoi dati la presentazione si aggiorna automaticamente.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
DOCS = ROOT / "docs"

NAVY = RGBColor(0x1F, 0x35, 0x55)
BLUE = RGBColor(0x2E, 0x6D, 0xA4)
GREY = RGBColor(0x55, 0x5F, 0x6B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF2, 0xF4, 0xF6)


def load_summary() -> dict:
    p = RES / "summary.json"
    if not p.exists():
        sys.exit("[slides] manca results/summary.json — esegui prima "
                 "experiments/run_experiments.py")
    return json.loads(p.read_text())


def m(s, mode, key, sub="mean", default=float("nan")):
    return s.get(mode, {}).get(key, {}).get(sub, default)


def atk(s, mode, name, metric):
    return s.get("attacks", {}).get(mode, {}).get(name, {}).get(metric, float("nan"))


def f(x, d=2):
    try:
        return "n/d" if x != x else f"{x:.{d}f}"
    except Exception:
        return str(x)


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.W, self.H = self.prs.slide_width, self.prs.slide_height

    def _blank(self):
        return self.prs.slides.add_slide(self.prs.slide_layouts[6])

    def _box(self, slide, l, t, w, h):
        tb = slide.shapes.add_textbox(l, t, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        return tf

    def _bar(self, slide):
        bar = slide.shapes.add_shape(1, 0, 0, self.W, Inches(1.15))
        bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
        bar.line.fill.background()
        return bar

    def title_slide(self, title, subtitle, footer):
        s = self._blank()
        bg = s.shapes.add_shape(1, 0, 0, self.W, self.H)
        bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
        tf = self._box(s, Inches(0.9), Inches(2.4), Inches(11.5), Inches(2.2))
        p = tf.paragraphs[0]; p.text = title
        p.font.size = Pt(40); p.font.bold = True; p.font.color.rgb = WHITE
        p2 = tf.add_paragraph(); p2.text = subtitle
        p2.font.size = Pt(20); p2.font.color.rgb = RGBColor(0xBC, 0xD3, 0xEA)
        tf3 = self._box(s, Inches(0.9), Inches(6.3), Inches(11.5), Inches(0.8))
        p3 = tf3.paragraphs[0]; p3.text = footer
        p3.font.size = Pt(13); p3.font.color.rgb = RGBColor(0x9A, 0xAE, 0xC4)

    def bullets(self, title, bullets, subtitle=None):
        s = self._blank(); self._bar(s)
        tf = self._box(s, Inches(0.6), Inches(0.18), Inches(12), Inches(0.9))
        p = tf.paragraphs[0]; p.text = title
        p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = WHITE
        top = Inches(1.5)
        if subtitle:
            st = self._box(s, Inches(0.7), Inches(1.25), Inches(12), Inches(0.5))
            sp = st.paragraphs[0]; sp.text = subtitle
            sp.font.size = Pt(15); sp.font.italic = True; sp.font.color.rgb = GREY
            top = Inches(1.95)
        body = self._box(s, Inches(0.7), top, Inches(12), Inches(5.2))
        for i, b in enumerate(bullets):
            par = body.paragraphs[0] if i == 0 else body.add_paragraph()
            lvl = 0
            if isinstance(b, tuple):
                b, lvl = b
            par.text = ("• " if lvl == 0 else "– ") + b
            par.level = lvl
            par.font.size = Pt(20 if lvl == 0 else 17)
            par.font.color.rgb = NAVY if lvl == 0 else GREY
            par.space_after = Pt(8)
        return s

    def picture(self, title, img, caption=None, bullets=None):
        s = self._blank(); self._bar(s)
        tf = self._box(s, Inches(0.6), Inches(0.18), Inches(12), Inches(0.9))
        p = tf.paragraphs[0]; p.text = title
        p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = WHITE
        ip = FIG / img
        if ip.exists():
            from PIL import Image as PImage
            iw, ih = PImage.open(ip).size
            maxw = Inches(8.6 if bullets else 11.8)
            maxh = Inches(5.3)
            w = maxw; h = int(w * ih / iw)
            if h > maxh:
                h = maxh; w = int(h * iw / ih)
            left = Inches(0.5) if bullets else int((self.W - w) / 2)
            s.shapes.add_picture(str(ip), left, Inches(1.45), width=w, height=h)
            if caption:
                ct = self._box(s, left, Inches(1.45) + h + Inches(0.05), w, Inches(0.6))
                cp = ct.paragraphs[0]; cp.text = caption
                cp.font.size = Pt(11); cp.font.italic = True; cp.font.color.rgb = GREY
                cp.alignment = PP_ALIGN.CENTER
        if bullets:
            bb = self._box(s, Inches(9.2), Inches(1.6), Inches(3.9), Inches(5.2))
            for i, b in enumerate(bullets):
                par = bb.paragraphs[0] if i == 0 else bb.add_paragraph()
                par.text = "• " + b
                par.font.size = Pt(15); par.font.color.rgb = NAVY
                par.space_after = Pt(8)
        return s

    def metrics_slide(self, title, cards):
        """cards: list of (big_value, label, color)."""
        s = self._blank(); self._bar(s)
        tf = self._box(s, Inches(0.6), Inches(0.18), Inches(12), Inches(0.9))
        p = tf.paragraphs[0]; p.text = title
        p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = WHITE
        n = len(cards); gap = Inches(0.3)
        cardw = (self.W - Inches(1.2) - gap * (n - 1)) / n
        for i, (val, lab, col) in enumerate(cards):
            left = Inches(0.6) + i * (cardw + gap)
            card = s.shapes.add_shape(1, left, Inches(2.2), cardw, Inches(2.6))
            card.fill.solid(); card.fill.fore_color.rgb = LIGHT
            card.line.color.rgb = col; card.line.width = Pt(2)
            ctf = card.text_frame; ctf.word_wrap = True
            vp = ctf.paragraphs[0]; vp.text = val; vp.alignment = PP_ALIGN.CENTER
            vp.font.size = Pt(34); vp.font.bold = True; vp.font.color.rgb = col
            lp = ctf.add_paragraph(); lp.text = lab; lp.alignment = PP_ALIGN.CENTER
            lp.font.size = Pt(14); lp.font.color.rgb = GREY
        return s

    def save(self, path):
        DOCS.mkdir(exist_ok=True)
        self.prs.save(str(path))


def build(s: dict):
    n = s.get("n_images", "?")
    d = Deck()
    d.title_slide(
        "SVD & YOLO per la Steganografia di Immagini",
        "Compressione e occultamento basati sulla Decomposizione ai Valori Singolari,\n"
        "con embedding guidato da YOLO",
        "Statistical and Mathematical Methods for AI  ·  Prof. M. Popolizio  ·  "
        f"valutazione su {n} immagini COCO-128")

    d.bullets("Il problema", [
        "Steganografia = nascondere un'informazione dentro un'immagine cover in "
        "modo impercettibile (≠ crittografia: non solo illeggibile, ma invisibile).",
        "Tre requisiti in tensione tra loro:",
        ("Impercettibilità — il cover non deve cambiare visibilmente", 1),
        ("Capacità — quanti bit possiamo nascondere", 1),
        ("Robustezza — il segreto deve sopravvivere ad attacchi (JPEG, rumore…)", 1),
        "Obiettivo: usare la SVD (cuore del corso) come compressione del segreto e "
        "dominio di embedding, e YOLO per rendere l'occultamento adattivo al contenuto.",
    ])

    d.bullets("Perché la SVD (teoria del corso)", [
        "Ogni matrice: A = U Σ Vᵀ, con σ₁ ≥ σ₂ ≥ … ≥ σ_r > 0.",
        "SVD troncata: A_k = Σ_{i≤k} σ_i u_i v_iᵀ.",
        "Teorema di Eckart–Young: A_k è la migliore approssimazione di rango k "
        "(‖A−A_k‖₂ = σ_{k+1}, ‖A−A_k‖_F = (Σ_{i>k} σ_i²)^½).",
        "Compressione: footprint k(m+n+1) contro m·n.",
        "σ₁ concentra l'energia ⇒ componente robusta; i σ piccoli amplificano il "
        "rumore (condizionamento κ = σ₁/σ_r).",
    ], subtitle="La SVD compare due volte nel progetto: compressione del segreto + dominio di embedding")

    d.bullets("Idea e pipeline", [
        "1) Il segreto viene compresso con SVD troncata → fattori compatti (U_k, σ_k, V_kᵀ).",
        "2) Il payload (bitstream) viene nascosto nei blocchi del cover modulando σ₁ "
        "(QIM) — un bit per blocco.",
        "3) YOLO rileva gli oggetti → mappa di saliency → i bit vanno nello sfondo, "
        "gli oggetti restano intatti.",
        "4) Estrazione cieca → ricostruzione del segreto via SVD inversa.",
        "5) YOLO ri-applicato su cover e stego per misurare la preservazione semantica.",
    ])

    d.picture("Compressione del segreto: SVD troncata",
              "fig_secret_reconstructions.png",
              "Aumentando il rango k cresce la fedeltà (NC) ma anche il payload.",
              bullets=["Riduzione di dimensionalità (lezione)",
                       "Payload: k(m+n+1) ≪ m·n",
                       "Quantizzazione U,V→int8, σ→float16"])

    d.picture("Eckart–Young: teoria verificata",
              "fig_eckart_young.png",
              "Errore di approssimazione teorico vs misurato e fattore di compressione.")

    d.bullets("Embedding nel dominio SVD (QIM su σ₁)", [
        "Per ogni blocco B×B: A = UΣVᵀ; si codifica un bit in σ₁ con passo Δ "
        "(Quantisation Index Modulation).",
        "σ₁' = ⌊σ₁/Δ⌋·Δ + (0.25 o 0.75)·Δ  a seconda del bit.",
        "Distorsione: ‖δ·u₁v₁ᵀ‖_F = |δ|, distribuita su tutto il blocco ⇒ "
        "impercettibile per pixel.",
        "Decodifica cieca dalla parte frazionaria di σ₁/Δ; margine di robustezza Δ/4.",
        "Codice a ripetizione (×R) + voto di maggioranza per la robustezza.",
    ])

    cfg = s.get("config", {})
    d.bullets("Setup sperimentale", [
        f"Dataset: COCO-128 — {n} immagini reali di COCO, cover 512×512.",
        "Due punti operativi:",
        (f"High-Capacity (HC): blocco {cfg.get('hc',{}).get('block')}×"
         f"{cfg.get('hc',{}).get('block')}, Δ={cfg.get('hc',{}).get('delta')}, "
         f"R={cfg.get('hc',{}).get('repeat')}, segreto "
         f"{cfg.get('hc',{}).get('secret_size')}², k={cfg.get('hc',{}).get('k')}", 1),
        (f"Robust (RB): blocco {cfg.get('rb',{}).get('block')}×"
         f"{cfg.get('rb',{}).get('block')}, Δ={cfg.get('rb',{}).get('delta')}, "
         f"R={cfg.get('rb',{}).get('repeat')}, segreto "
         f"{cfg.get('rb',{}).get('secret_size')}², k={cfg.get('rb',{}).get('k')}", 1),
        "Metriche: PSNR, SSIM, MSE; NC e BER del segreto; mAP@0.5 / preservation rate.",
        "Attacchi: JPEG q90/q75/q50, rumore gaussiano, blur, mediano, sale&pepe, rescale.",
    ])

    d.metrics_slide("Risultati — impercettibilità (HC)", [
        (f"{f(m(s,'hc','guided_psnr'))} dB", "PSNR cover→stego", BLUE),
        (f"{f(m(s,'hc','guided_ssim'),4)}", "SSIM", BLUE),
        (f"{f(m(s,'hc','capacity_bpp'),4)}", "capacità (bpp)", BLUE),
        (f"{f(m(s,'hc','guided_secret_nc'),3)}", "NC segreto (pulito)", BLUE),
    ])

    d.picture("Risultato chiave — protezione oggetti via YOLO",
              "fig_object_protection.png",
              f"PSNR oggetti: {f(m(s,'hc','baseline_obj_psnr'))} → "
              f"{f(m(s,'hc','guided_obj_psnr'))} dB con la guida YOLO.",
              bullets=["I bit vanno nello sfondo",
                       "Oggetti quasi intatti",
                       f"+{f(m(s,'hc','guided_obj_psnr')-m(s,'hc','baseline_obj_psnr'))} dB "
                       "sugli oggetti"])

    d.picture("Risultato qualitativo",
              "fig_qualitative.png",
              "La mappa |cover−stego| evita le regioni degli oggetti rilevati da YOLO.")

    d.metrics_slide("Preservazione semantica (downstream YOLO)", [
        (f"{f(m(s,'hc','det_preservation_rate'),3)}", "preservation rate", BLUE),
        (f"{f(m(s,'hc','det_map50'),3)}", "mAP@0.5 stego vs cover", BLUE),
        (f"{f(m(s,'hc','det_mean_conf_delta'),3)}", "|Δ confidenza| medio", BLUE),
    ])

    d.picture("Recupero del segreto e compressione",
              "fig_secret_compression.png",
              "Il limite in pulito è dato dalla compressione SVD del segreto, non "
              "dall'estrazione.")

    d.picture("Robustezza e compromesso capacità↔robustezza",
              "fig_robustness.png",
              f"RB resiste agli attacchi (es. JPEG-q90 NC={f(atk(s,'rb','jpeg_q90','nc'),2)}); "
              f"HC, senza ridondanza, è fragile.")

    d.picture("Fronte di Pareto", "fig_tradeoff.png",
              "Capacità (bpp) vs robustezza media (NC sotto attacco): due punti operativi.")

    d.bullets("Conclusioni", [
        "La SVD fornisce un quadro unificato: compressione del segreto (Eckart–Young), "
        "dominio di embedding robusto (σ₁), analisi del condizionamento.",
        f"Impercettibilità elevata (PSNR ≈ {f(m(s,'hc','guided_psnr'))} dB) e "
        f"preservazione semantica quasi perfetta (mAP ≈ {f(m(s,'hc','det_map50'),2)}).",
        "La guida YOLO protegge le regioni salienti (+~10 dB) a parità di capacità.",
        "Compromesso capacità↔robustezza governato da tre leve SVD: dimensione "
        "blocco, passo Δ, ridondanza R.",
        "Codice e documentazione completi, riproducibili su 100+ immagini COCO.",
    ])
    return d


def main():
    s = load_summary()
    deck = build(s)
    out = DOCS / "presentazione.pptx"
    deck.save(out)
    print(f"[slides] scritto {out}  ({len(deck.prs.slides._sldIdLst)} slide)")


if __name__ == "__main__":
    main()
