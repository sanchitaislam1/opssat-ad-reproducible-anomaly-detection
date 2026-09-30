from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures"; OUT.mkdir(exist_ok=True)

m=pd.read_csv(ROOT/"results/authoritative_model_results.csv")
plt.figure(figsize=(9,5))
plt.bar(m["Model"],m["F1"])
plt.ylabel("F1-score"); plt.xlabel("Classifier")
plt.title("OPSSAT-AD Test F1-score by Classifier")
plt.xticks(rotation=30,ha="right"); plt.tight_layout()
plt.savefig(OUT/"fig_model_comparison.png",dpi=300); plt.close()

a=pd.read_csv(ROOT/"results/authoritative_ablation_results.csv")
plt.figure(figsize=(8,5))
plt.bar(a["FeatureGroup"],a["F1"])
plt.ylabel("F1-score"); plt.xlabel("Feature group")
plt.title("Feature-group Ablation on OPSSAT-AD")
plt.xticks(rotation=25,ha="right"); plt.tight_layout()
plt.savefig(OUT/"fig_feature_ablation.png",dpi=300); plt.close()

rep=pd.DataFrame({
"Representation":["Official 18","Robust +18","Temporal +18","Spectral +18","All 32"],
"F1":[.935185,.904762,.925926,.914286,.918660],
"PR-AUC":[.967187,.963320,.971596,.973358,.973222]})
x=range(len(rep))
plt.figure(figsize=(9,5))
plt.plot(x,rep.F1,marker="o",label="F1")
plt.plot(x,rep["PR-AUC"],marker="o",label="PR-AUC")
plt.xticks(list(x),rep.Representation,rotation=20,ha="right")
plt.ylabel("Score"); plt.ylim(.85,1.0)
plt.title("Official vs. Expanded Raw-derived Feature Representations")
plt.legend(); plt.tight_layout()
plt.savefig(OUT/"fig_representation_comparison.png",dpi=300); plt.close()

c=pd.read_csv(ROOT/"results/authoritative_channel_results.csv")
plt.figure(figsize=(9,5))
plt.bar(c["Channel"],c["F1"])
plt.ylabel("F1-score"); plt.xlabel("Held-out channel")
plt.title("Leave-one-channel-out Generalization")
plt.xticks(rotation=30,ha="right"); plt.tight_layout()
plt.savefig(OUT/"fig_channel_robustness.png",dpi=300); plt.close()

print("Figures written to",OUT)
