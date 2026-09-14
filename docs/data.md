# Expected data layout and schemas

No patient data is included in this repository. The scripts expect the following
layout relative to the repository root:

```
data/
├── matched_data/
│   ├── PA/                      standing posteroanterior radiographs, one DICOM per patient
│   │   ├── <patient_id>_....dcm
│   │   └── ...
│   └── LAT/                     standing lateral radiographs, one DICOM per patient
│       ├── <patient_id>_....dcm
│       └── ...
└── AIS_Surgery_clean.csv        labels + clinical variables
```

## Patient identifier

`patient_id` is the join key between the label table and the DICOM files, matched
on the **first 8 characters of the filename**, zero-padded. A file named
`12345678_151208_5_.Seq1.Ser1.Img1.dcm` is therefore matched to `patient_id`
`12345678`. If the CSV carries an `ID` column instead, the digits are extracted and
zero-padded to 8 characters.

A patient is included only if **both** a PA and a lateral DICOM are present. Patients
without both views are dropped at dataset construction, which is how the initial
cohort of 174 reduces to the 155 analysed (Figure 1).

## `AIS_Surgery_clean.csv`

| Column | Type | Description |
|---|---|---|
| `ID` / `patient_id` | str | 8-digit zero-padded identifier |
| `Structural_L` | 0/1 | **Prediction target** — 1 if the lumbar curve is structural |
| `Sex` | str (F/M) | mapped to 0/1 at load time |
| `Age` | float | years |
| `Lumbar_Cobb` | float | standing lumbar Cobb angle, degrees |
| `L1_S1_Lordosis` | float | L1–S1 lordosis, degrees |
| `Cobb` | float | major Cobb angle, degrees |
| `Thoracic_Cobb` | float | thoracic Cobb angle, degrees |
| `T1_12_Kyphosis` | float | T1–T12 kyphosis, degrees |
| `T4_12_Kyphosis` | float | T4–T12 kyphosis, degrees |

Missing numeric entries are filled with `0.0`; missing `Sex` defaults to `0`.

## Clinical feature sets

The model input is selected by the `clinical_set` argument, defined in
[`src/multimodal/dataset.py`](../src/multimodal/dataset.py):

| Set | Columns | Dim |
|---|---|---|
| **`lumbar_only`** (published) | `Sex`, `Age`, `Lumbar_Cobb`, `L1_S1_Lordosis` | 4 |
| `global_lumbar` | adds `Cobb` | 5 |
| `full_spine` | adds `Thoracic_Cobb`, `T1_12_Kyphosis`, `T4_12_Kyphosis` | 8 |

## Variables that must never become features

Side-bending measurements (`Lumbar_Cobb_Bending`, `Thoracic_Cobb_Bending`) **define
the label** — structurality is adjudicated on the side-bending radiograph. They are
excluded from every clinical feature set, and are used only for label definition and
for the post hoc error characterization in
[`src/common/error_analysis.py`](../src/common/error_analysis.py). Feeding them to
the model would leak the outcome.

## Cohort

155 patients with adolescent idiopathic scoliosis who underwent posterior spinal
fusion at a single tertiary referral center between December 2005 and December 2024,
each with a standing PA radiograph, a standing lateral radiograph, and a
contemporaneous side-bending radiograph.

| Label | n |
|---|---|
| Structural lumbar curve (`Structural_L` = 1) | 82 |
| Non-structural (`Structural_L` = 0) | 73 |

The class balance is mild and is handled with a `pos_weight` in the loss computed
from each fold's training split rather than by resampling.
