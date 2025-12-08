# Mechanical Properties of LPBF-Built Super Duplex Stainless Steel 2507 (UNS S32750)

This repository hosts the experimental data, images, and analysis notebooks used to study how **laser powder bed fusion (LPBF)** and **subsequent post-processing/heat-treatment steps** affect the mechanical response of **super duplex stainless steel 2507**.

The files here belong to a larger research workflow carried out and are organized so they can be reused in Google Colab or local Jupyter environments.

---

## 1. Scope

- LPBF-built SDSS 2507 (UNS S32750), standard chemistry.
- As-built and several stress-relieved / solution-annealed conditions:
  - `AS` (as-built)
  - `SR400_1h`
  - `SR450_1h`
  - `SR500_1h`
  - `SR550_1h`
  - `SA1100_15min`
- Focus on **hardness, tensile behavior, and microstructural evidence** (EBSD/XRD images).
- Goal: **map post-processing → microstructure → mechanical properties** for LPBF SDSS.

---

## 2. Repository Structure

The repo currently contains (representative list from the root):

- `Hardndess_test_visualization_3_ipynb.ipynb` – Colab/Jupyter notebook for plotting hardness data.
- `Tensile_test_visualization_3_Ni.ipynb` – notebook for reading tensile test outputs and generating stress–strain figures.
- `hardndess_ script` / `Notebook script_tensile test` – helper scripts/notebooks for figure generation.
- `AS.ang.zip`, `SR300.ang.zip`, `SR550.ang.zip`, `SA1100.zip` – EBSD/ANG inputs for crystallographic/microstructural analysis (zipped).
- `XRD datasets/` – XRD-related data for phase/microstructure verification.
- `L-PBF process conditions matrix` – processing parameters used for the printed batches.
- Image assets: `AS.png`, `SR400_1h.png`, `SR450_1h.png`, `SR500_1h.png`, `SR550_1h .png`, `SA1100_15min.png`, `VED_porosity.png`, `micrographs_printing_params.png`, `charpy_fractography.png`, etc.  
  These are mainly for documentation, figure building, or manuscript graphics.

> **Note**: file names reflect the processing condition. Keep this naming when you add new datasets — it makes automated plotting simpler.

---

## 3. How to Use (Google Colab workflow)

1. Open the notebook you want, e.g. **`Hardndess_test_visualization_3_ipynb.ipynb`**.
2. Run it in **Google Colab**.
3. When prompted, **upload your CSV/XLSX** that contains the mechanical data for the selected condition (`AS`, `SR400_1h`, …).
4. The notebook will:
   - clean/standardize column names,
   - generate publication-style plots (thick axes, larger marker size),
   - export `.png` at high resolution for manuscripts.

You can also mount Google Drive in Colab if your data sits there.

_Add this badge at the top if you like:_

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengedagnaw/Mech_Props-L-PBF-SDSS)
