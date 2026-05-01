# PhytoMyco-Pred

**PhytoMyco-Pred** is an AI-assisted web server for predicting antifungal compounds against major plant-pathogenic fungi. The platform first predicts whether a query compound is likely to possess antifungal potential and, for compounds predicted as antifungal, estimates species-wise targeting likelihood across supported fungal pathogens.

**Avalaible at:** https://phytomycopred.streamlit.app/

---

## Scope

PhytoMyco-Pred is designed for early-stage prioritization of candidate antifungal compounds for plant disease management and crop protection research. 

---

## Supported Fungal Species

The current prediction panel includes the following plant-pathogenic fungi:

1. *Alternaria solani*
2. *Aspergillus flavus*
3. *Botrytis cinerea*
4. *Colletotrichum gloeosporioides*
5. *Fusarium oxysporum*
6. *Magnaporthe oryzae*
7. *Penicillium expansum*
8. *Puccinia graminis* f. sp. *tritici*
9. *Rhizoctonia solani*
10. *Sclerotinia sclerotiorum*

---

## Prediction Strategy

PhytoMyco-Pred uses a two-stage prediction framework:

### Stage 1: Generic Antifungal Prediction

The first model predicts whether a compound is likely to possess antifungal potential against plant-pathogenic fungi in general.

### Stage 2: Species-wise Targeting Prediction

If a compound is predicted as antifungal, species-wise models estimate the fungal species against which the compound is most likely to show antifungal relevance.

---

## Input Format

### Single-compound mode

The user provides:

- Compound ID
- Canonical SMILES

Example:

```text
Compound ID: CMPD_001
Canonical SMILES: O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12
```

### Batch mode

Upload a CSV file with the following required column:

```text
canonical_smiles
```

An optional compound identifier column may also be included:

```text
compound_id
```

Example CSV:

```csv
compound_id,canonical_smiles
CMPD_001,O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12
CMPD_002,COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O
```

---

## Output

The server provides:

### Generic Prediction Output

- Compound ID
- Input SMILES
- Canonical SMILES
- Generic antifungal score
- Generic prediction: Antifungal / Non-antifungal
- Confidence category

### Species-wise Prediction Output

For compounds predicted as antifungal, the server provides:

- Fungal species
- Species-wise score
- Targeting likelihood: Likely target / Unlikely target
- Confidence category

The results can be downloaded as CSV files for further analysis.

---

