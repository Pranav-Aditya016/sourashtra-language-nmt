# IEEE Paper — Build Instructions

## Files in this folder

| File | Description |
|------|-------------|
| `sourashtra_nmt_paper.tex` | Main LaTeX source |
| `fig1_model_comparison_em.png` | EM comparison bar chart |
| `fig2_multi_metric_comparison.png` | Multi-metric (EM/BLEU/chrF) comparison |
| `fig3_training_loss_curves.png` | Training loss curves (V1–V5) |
| `fig4_v3_v4_v5_em_progression.png` | Validation EM over epochs |
| `fig5_hybrid_analysis.png` | Hybrid neural vs retrieval breakdown |
| `fig6_category_performance.png` | Per-category accuracy |
| `fig8_error_analysis.png` | Error type distribution |
| `fig9_improvement_progression.png` | V1→V5 improvement timeline |
| `webapp_translate.png` | Website screenshot (translate tab) — **capture first** |
| `webapp_dictionary.png` | Website screenshot (dictionary tab) — **capture first** |

## Step 1: Capture Website Screenshots

Make sure the Flask server is running:
```bash
cd "c:\Pranav Aditya\MY_Project"
.\.venv\Scripts\python.exe app.py
```

Then in another terminal:
```bash
pip install selenium
python capture_screenshots.py
```

This will save `webapp_translate.png` and `webapp_dictionary.png` into this folder.

## Step 2: Compile the Paper

### Option A: Overleaf (Recommended)
1. Go to [overleaf.com](https://www.overleaf.com)
2. Create a new project → Upload Project
3. Upload all files from this `ieee_paper/` folder
4. Click Compile

### Option B: Local LaTeX
```bash
cd ieee_paper
pdflatex sourashtra_nmt_paper.tex
bibtex sourashtra_nmt_paper
pdflatex sourashtra_nmt_paper.tex
pdflatex sourashtra_nmt_paper.tex
```

## Output
The compiled PDF will be `sourashtra_nmt_paper.pdf`
