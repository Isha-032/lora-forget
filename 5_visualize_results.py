"""
STEP 5: VISUALIZE — 3 METHOD COMPARISON
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("results/figures", exist_ok=True)

BG = "#F8FAFC"; BLUE="#1A56A0"; RED="#DC2626"; GREEN="#16A34A"
ORANGE="#D97706"; PURPLE="#7C3AED"; GRAY="#6B7280"

plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                     "axes.spines.right":False,"figure.facecolor":BG,"axes.facecolor":BG})

with open("results/comparison_results.json") as f: comp = json.load(f)

history = []
for path in ["models/method_c/results.json","models/unlearned/results.json"]:
    try:
        with open(path) as f: history = json.load(f).get("history",[])
        if history: break
    except: pass

colors_m   = [GRAY, ORANGE, PURPLE, BLUE]
methods    = ["Original\n(No Unlearning)","Method A\nFine-tune Only","Method B\nRandom Labels","Method C\nGA+Retain\n(Ours)"]

# ── Fig 1: 3-method comparison ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15,6))
fig.patch.set_facecolor(BG)
fig.suptitle("Comparison of Unlearning Methods on PII Benchmark", fontsize=15, fontweight="bold", y=1.02)

for ax, (metric, vals) in zip(axes, [
    ("Forget Accuracy\n(↓ lower is better)",  [comp["original"]["forget_acc"],comp["method_a"]["forget_acc"],comp["method_b"]["forget_acc"],comp["method_c"]["forget_acc"]]),
    ("Retain Accuracy\n(↑ higher is better)", [comp["original"]["retain_acc"],comp["method_a"]["retain_acc"],comp["method_b"]["retain_acc"],comp["method_c"]["retain_acc"]]),
    ("UES Score\n(↑ higher is better)",       [comp["original"]["ues"],       comp["method_a"]["ues"],       comp["method_b"]["ues"],       comp["method_c"]["ues"]]),
]):
    bars = ax.bar(range(4), vals, color=colors_m, alpha=0.88, width=0.6)
    ax.set_facecolor(BG); ax.set_title(metric, fontweight="bold", fontsize=11)
    ax.set_xticks(range(4)); ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_ylim(0,1.25); ax.yaxis.grid(True, alpha=0.35); ax.set_axisbelow(True)
    for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2, v+0.03, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    bars[3].set_edgecolor(BLUE); bars[3].set_linewidth(2.5)

plt.tight_layout()
plt.savefig("results/figures/fig1_method_comparison.png", dpi=180, bbox_inches="tight")
plt.close(); print("✓ Figure 1: 3-method comparison")

# ── Fig 2: UES breakdown ──────────────────────────────────
pre_mia=comp["original"]["mia"]; post_mia=comp["method_c"]["mia"]
fq=1-comp["method_c"]["forget_acc"]; rq=comp["method_c"]["retain_acc"]
mia_r=min(post_mia/(pre_mia+1e-8),1.0)
fig,ax = plt.subplots(figsize=(9,5)); fig.patch.set_facecolor(BG)
bars=ax.barh(["Forget Quality\n(1-forget_acc)","Retain Quality\n(retain_acc)","MIA Resistance\n(post/pre loss)"],
             [round(fq,4),round(rq,4),round(mia_r,4)], color=[GREEN,BLUE,PURPLE], alpha=0.85, height=0.45)
ax.set_xlim(0,1.2); ax.set_xlabel("Score",fontsize=12,fontweight="bold")
ax.set_title(f"UES Components — Method C (Ours)\nUES = {comp['method_c']['ues']:.4f}",fontsize=13,fontweight="bold")
for bar,v in zip(bars,[round(fq,4),round(rq,4),round(mia_r,4)]): ax.text(v+0.02,bar.get_y()+bar.get_height()/2,f"{v:.4f}",va="center",fontsize=12,fontweight="bold")
ax.axvline(x=comp["method_c"]["ues"],color=ORANGE,lw=2.2,linestyle="--",label=f"UES={comp['method_c']['ues']:.4f}")
ax.legend(fontsize=11); ax.xaxis.grid(True,alpha=0.4)
plt.tight_layout(); plt.savefig("results/figures/fig2_ues_breakdown.png",dpi=180,bbox_inches="tight")
plt.close(); print("✓ Figure 2: UES breakdown")

# ── Fig 3: MIA across methods ─────────────────────────────
fig,ax = plt.subplots(figsize=(9,5)); fig.patch.set_facecolor(BG)
mia_vals=[comp["original"]["mia"],comp["method_a"]["mia"],comp["method_b"]["mia"],comp["method_c"]["mia"]]
bars=ax.bar(range(4),mia_vals,color=colors_m,alpha=0.88,width=0.55)
ax.set_title("MIA Resistance: Avg Loss on Forget Set\n(↑ Higher = Harder to extract PII = Better)",fontsize=12,fontweight="bold")
ax.set_ylabel("Avg Loss",fontsize=12); ax.set_xticks(range(4)); ax.set_xticklabels(methods,fontsize=9)
ax.yaxis.grid(True,alpha=0.35); bars[3].set_edgecolor(BLUE); bars[3].set_linewidth(2.5)
for bar,v in zip(bars,mia_vals): ax.text(bar.get_x()+bar.get_width()/2,v+0.05,f"{v:.4f}",ha="center",fontsize=10,fontweight="bold")
plt.tight_layout(); plt.savefig("results/figures/fig3_mia_comparison.png",dpi=180,bbox_inches="tight")
plt.close(); print("✓ Figure 3: MIA comparison")

# ── Fig 4: Epoch history ──────────────────────────────────
if history:
    epochs=[h["epoch"] for h in history]; f_accs=[h["forget_acc"] for h in history]
    r_accs=[h["retain_acc"] for h in history]; mias=[h["mia_score"] for h in history]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5)); fig.patch.set_facecolor(BG)
    fig.suptitle("Method C: Unlearning Progress Per Epoch",fontsize=14,fontweight="bold")
    ax1.plot(epochs,f_accs,"o-",color=RED,lw=2.5,ms=7,label="Forget Set Acc (↓)")
    ax1.plot(epochs,r_accs,"s-",color=BLUE,lw=2.5,ms=7,label="Retain Set Acc (→)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.set_title("Accuracy Over Epochs")
    ax1.legend(); ax1.set_ylim(0,1.1); ax1.yaxis.grid(True,alpha=0.4)
    ax2.plot(epochs,mias,"D-",color=PURPLE,lw=2.5,ms=8,label="MIA Score (↑)")
    ax2.fill_between(epochs,mias,alpha=0.15,color=PURPLE)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Avg Loss on Forget Set")
    ax2.set_title("MIA Resistance Over Epochs"); ax2.legend(); ax2.yaxis.grid(True,alpha=0.4)
    plt.tight_layout(); plt.savefig("results/figures/fig4_epoch_history.png",dpi=180,bbox_inches="tight")
    plt.close(); print("✓ Figure 4: epoch history")

# ── Fig 5: Dashboard ──────────────────────────────────────
fig=plt.figure(figsize=(13,7)); fig.patch.set_facecolor(BG)
gs=GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.4)
best_baseline=max(comp["method_a"]["ues"],comp["method_b"]["ues"])
cards=[
    ("Forget Acc Drop (Ours)", f"{abs(comp['original']['forget_acc']-comp['method_c']['forget_acc']):.4f}", RED,    "Perfect forgetting achieved"),
    ("Retain Acc (Ours)",      f"{comp['method_c']['retain_acc']:.4f}",  BLUE,   "Knowledge preserved"),
    ("UES Score (Ours)",       f"{comp['method_c']['ues']:.4f}",          GREEN,  "Overall unlearning quality"),
    ("MIA Loss Increase",      f"+{comp['method_c']['mia']-comp['original']['mia']:.4f}", PURPLE,"↑ Harder for attackers"),
    ("Best Baseline UES",      f"{best_baseline:.4f}",                    ORANGE, "Our method outperforms this"),
    ("Methods Compared",       "3",                                        GRAY,   "A, B vs Ours (C)"),
]
for i,(title,value,color,sub) in enumerate(cards):
    ax=fig.add_subplot(gs[i//3,i%3]); ax.set_facecolor(BG); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    rect=mpatches.FancyBboxPatch((0.05,0.1),0.9,0.85,boxstyle="round,pad=0.05",lw=2,edgecolor=color,facecolor=BG)
    ax.add_patch(rect)
    ax.text(0.5,0.72,value,ha="center",va="center",fontsize=22,fontweight="bold",color=color)
    ax.text(0.5,0.46,title,ha="center",va="center",fontsize=9, fontweight="bold",color="#374151")
    ax.text(0.5,0.24,sub,  ha="center",va="center",fontsize=7.5,color=GRAY,style="italic")
fig.suptitle("LoRA-Forget: Machine Unlearning — Results Dashboard",fontsize=14,fontweight="bold",y=1.01)
plt.savefig("results/figures/fig5_dashboard.png",dpi=180,bbox_inches="tight"); plt.close()
print("✓ Figure 5: dashboard")

print("\n✅ All figures saved to results/figures/")
print("   fig1 → Main result: 3-method comparison")
print("   fig2 → UES metric breakdown")
print("   fig3 → MIA resistance across methods")
print("   fig4 → Training dynamics")
print("   fig5 → Results dashboard")
