#!/usr/bin/env python3
"""Post-completion Top-N analysis; refuses to create scientific outputs before 36/36."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, values: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--top-n",type=int,default=1500,choices=(1500,))
    args=parser.parse_args(); registry_path=ROOT/"configs/experiment_registry/u1000_top1500_training_scaling_matrix.json"
    registry=json.loads(registry_path.read_text(encoding="utf-8")); entries=registry["entries"]
    if len(entries)!=36 or not all(item["status"]=="completed" and item["validation_result"]=="PASS" for item in entries):
        raise SystemExit("REFUSED: scientific analysis requires 36/36 completed, validated Top1500 runs")
    subprocess.run([str(ROOT/"envs/camels-gnn/bin/python"),
                    "scripts/validation/manage_u1000_top1500_training_scaling_matrix.py","--aggregate"],cwd=ROOT,check=True)
    report=ROOT/"reports/experiment_registry"; per_run=rows(report/"u1000_top1500_final_per_run_metrics.csv")
    paired=[]
    for count in (20,50,100,200,450,700):
        for seed in (42,123,2025):
            group=[row for row in per_run if int(row["training_universe_count"])==count and int(row["seed"])==seed]
            by={row["model"]:row for row in group}; evolve=by["EvolveGCNHRegressor"]; static=by["StaticGCNRegressor"]
            paired.append({"training_count":count,"seed":seed,"evolve_mae":evolve["test_mae"],"static_mae":static["test_mae"],
                           "static_minus_evolve_mae":float(static["test_mae"])-float(evolve["test_mae"])})
    write(report/"u1000_top1500_static_vs_evolve_paired_comparison.csv",paired,list(paired[0]))
    anchor=rows(report/"u1000_top1000_final_per_run_metrics.csv"); comparison=[]
    for row in per_run:
        model="EvolveGCN-H" if row["model"]=="EvolveGCNHRegressor" else "Static GCN"
        match=next(item for item in anchor if item["model"]==model and item["training_count"]==row["training_universe_count"] and item["seed"]==row["seed"])
        comparison.append({"model":model,"training_count":row["training_universe_count"],"seed":row["seed"],
                           "top1000_mae":match["mae"],"top1500_mae":row["test_mae"],
                           "top1500_minus_top1000_mae":float(row["test_mae"])-float(match["mae"])})
    write(report/"u1000_top1000_vs_top1500_comparison.csv",comparison,list(comparison[0]))
    (report/"u1000_top1500_training_scaling_scientific_interpretation.md").write_text(
        "# U1000 Top1500 training-scaling scientific interpretation\n\n"
        "Generated only after 36/36 artifact-validated runs. Interpret the validated paired tables and figures; "
        "weak performance and prediction compression are scientific outcomes, not infrastructure failures.\n",encoding="utf-8")
    import matplotlib.pyplot as plt
    output=ROOT/"presentation_assets/u1000_top1500_training_scaling"; output.mkdir(parents=True,exist_ok=True)
    labels={"EvolveGCNHRegressor":"EvolveGCN-H","StaticGCNRegressor":"Static GCN"}
    for metric,filename in (("test_mae","01_mae_learning_curve.png"),("test_rmse","02_rmse_learning_curve.png"),
                            ("test_r2","03_r2_learning_curve.png"),("prediction_sd_ratio","04_prediction_sd_ratio.png")):
        fig,ax=plt.subplots(figsize=(7,4.5))
        for model,label in labels.items():
            x=[]; y=[]
            for count in (20,50,100,200,450,700):
                vals=[float(row[metric]) for row in per_run if row["model"]==model and int(row["training_universe_count"])==count]
                x.append(count); y.append(sum(vals)/len(vals))
            ax.plot(x,y,marker="o",label=label)
        ax.set(xlabel="Training universes",ylabel=metric); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output/filename,dpi=180); plt.close(fig)
    train700=[item for item in entries if item["training_universe_count"]==700]
    prediction_rows=[]
    for item in train700:
        values=rows(ROOT/item["experiment_directory"]/"predictions/test_predictions.csv")
        for value in values:
            prediction_rows.append({"model":labels[item["model"]],"seed":item["seed"],
                                    "true":float(value["true_omega_m"]),"pred":float(value["pred_omega_m"])})
    for kind,filename in (("scatter","05_train700_true_vs_predicted.png"),
                          ("sorted","06_train700_sorted_low_to_high.png"),
                          ("residual","07_train700_residual_vs_true.png")):
        fig,ax=plt.subplots(figsize=(7,5))
        for label in labels.values():
            group=[value for value in prediction_rows if value["model"]==label]
            if kind=="scatter": ax.scatter([v["true"] for v in group],[v["pred"] for v in group],s=8,alpha=.3,label=label)
            elif kind=="sorted":
                group=sorted(group,key=lambda value:value["true"]); ax.plot(range(len(group)),[v["pred"] for v in group],alpha=.65,label=label)
            else: ax.scatter([v["true"] for v in group],[v["pred"]-v["true"] for v in group],s=8,alpha=.3,label=label)
        if kind=="scatter":
            limits=[min(v["true"] for v in prediction_rows),max(v["true"] for v in prediction_rows)]; ax.plot(limits,limits,"k--",lw=1)
            ax.set(xlabel="True Omega_m",ylabel="Predicted Omega_m")
        elif kind=="sorted": ax.set(xlabel="Test samples sorted by true Omega_m",ylabel="Predicted Omega_m")
        else: ax.axhline(0,color="black",ls="--",lw=1); ax.set(xlabel="True Omega_m",ylabel="Prediction residual")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(output/filename,dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for model in ("EvolveGCN-H","Static GCN"):
        group=[value for value in comparison if value["model"]==model]
        x=sorted({int(value["training_count"]) for value in group})
        y=[sum(float(value["top1500_minus_top1000_mae"]) for value in group if int(value["training_count"])==count)/3 for count in x]
        ax.plot(x,y,marker="o",label=model)
    ax.axhline(0,color="black",ls="--",lw=1); ax.set(xlabel="Training universes",ylabel="Top1500 MAE − Top1000 MAE")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output/"08_top1000_vs_top1500_paired.png",dpi=180); plt.close(fig)
    print("PASS: final Top1500 tables/comparisons and eight presentation figures generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
