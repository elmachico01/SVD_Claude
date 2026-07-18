# Protocollo di riesecuzione e validazione

Questo documento descrive la procedura da seguire dopo ogni modifica al metodo. I risultati presenti nel repository prima della revisione **non devono essere riutilizzati**, perché il QIM, il formato del payload e le metriche di decodifica sono cambiati.

## 1. Ambiente pulito

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Annotare nel verbale di esecuzione:

```bash
python --version
pip freeze > results/environment.txt
```

## 2. Verifiche preliminari

```bash
python tests/test_core.py
```

Tutti i test devono risultare `PASS`. In caso contrario non eseguire il batch completo.

Eseguire poi due smoke test:

```bash
python scripts/demo.py --index 0 --mode hc
python scripts/demo.py --index 0 --mode rb --attack jpeg_q75
```

Controllare visivamente le figure in `results/examples/`.

## 3. Eliminazione dei risultati precedenti

Conservare eventualmente una copia separata dei risultati storici, quindi rimuovere almeno:

```text
results/metrics_per_image.csv
results/summary.json
results/secret_compression.csv
results/figures/*.png
```

Non mescolare output generati da versioni diverse del codice.

## 4. Esperimento completo

```bash
python experiments/run_experiments.py --limit 128 --seed 0
```

Durante l'esecuzione verificare che:

- vengano elaborate 128 immagini;
- non compaiano eccezioni sistematiche;
- `guided_decode_success` e `baseline_decode_success` siano registrati;
- la sorgente della guida sia distinta tra `yolo`, `spectral_residual` e `gradient_fallback`;
- il file finale contenga una riga per ogni immagine elaborata.

## 5. Controlli sui dati

Prima di produrre report e slide controllare:

1. numero di righe del CSV;
2. percentuale di successo della decodifica pulita;
3. media e massimo BER pulito;
4. numero di immagini con zero detection YOLO;
5. distribuzione delle sorgenti di guidance;
6. presenza di valori NaN/inf nelle metriche aggregate;
7. coerenza tra configurazioni dichiarate e configurazioni salvate nel JSON.

Il risultato pulito ideale è:

```text
clean decode success = 100%
clean BER = 0 per tutte le immagini
```

Se non viene raggiunto, i casi falliti devono essere elencati e analizzati, non nascosti dalla sola media.

## 6. Figure e deliverable

Solo dopo aver validato CSV e JSON:

```bash
python experiments/make_figures.py
python scripts/generate_report.py
python scripts/generate_slides.py
```

Il report e la presentazione devono riportare esplicitamente:

- commit Git usato;
- seed;
- versione Python e principali dipendenze;
- numero di immagini effettivamente elaborate;
- distinzione tra estrazione cieca keyed e estrazione guidata con side information;
- `consistency AP@0.5` come confronto cover/stego, non come vero mAP COCO;
- risultati negativi e limiti sperimentali.

## 7. File da inviare per la revisione finale

Dopo l'esecuzione fornire:

```text
results/metrics_per_image.csv
results/summary.json
results/secret_compression.csv
results/environment.txt
results/figures/
```

Questi file saranno la sola fonte numerica per la versione finale di relazione e PowerPoint.
