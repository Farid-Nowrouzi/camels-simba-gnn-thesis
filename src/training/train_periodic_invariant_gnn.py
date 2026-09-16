"""Train the controlled invariant GNN on train/validation partitions only."""
from __future__ import annotations
import argparse, csv, hashlib, json, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from src.data.periodic_invariant_features import periodic_invariant_features
from src.models.periodic_invariant_gnn import PeriodicInvariantGNNRegressor, count_parameters
from src.training.sparse_batch import collate_sparse_static, sparse_batch_to
from src.training.split_manifest import load_dataset_provenance, load_split_manifest

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def _load(path: Path) -> dict[str, Any]:
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")

def _final_samples(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for uid, item in data.items():
        if "Nodes_list" not in item or "edge_index_list" not in item or "mask_list" not in item:
            raise ValueError("controlled trainer requires sparse temporal dataset")
        result[uid] = {"x": item["Nodes_list"][-1], "edge_index": item["edge_index_list"][-1],
                       "mask": item["mask_list"][-1], "target": item["target"]}
    return result

class _Dataset(Dataset):
    def __init__(self, samples: dict[str, dict[str, Any]], ids: list[str]): self.samples, self.ids = samples, ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, index):
        uid = self.ids[index]; s = self.samples[uid]
        return uid, {"graph_storage":"sparse_edge_index", "x":s["x"].float(), "edge_index":s["edge_index"].long(), "mask":s["mask"].float()}, torch.as_tensor(s["target"], dtype=torch.float32).view(1)

def _collate(rows): return [r[0] for r in rows], collate_sparse_static([r[1] for r in rows]), torch.stack([r[2] for r in rows])
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def train(*, dataset_path: str | Path, split_manifest_path: str | Path, seed: int, output_dir: str | Path,
          protocol_freeze_path: str | Path, epochs: int = 300, batch_size: int = 8, patience: int = 40,
          max_train_universes: int | None = None, max_val_universes: int | None = None, device: str = "cpu") -> dict[str, Any]:
    """Fit only train/validation samples. This function never creates a test loader."""
    set_seed(seed); dataset_path, split_path, out = Path(dataset_path), Path(split_manifest_path), Path(output_dir)
    protocol = Path(protocol_freeze_path)
    if not protocol.exists(): raise FileNotFoundError("protocol freeze is required")
    samples = _final_samples(_load(dataset_path)); identity = load_dataset_provenance(dataset_path).get("dataset_sha256", _sha(dataset_path))
    manifest = load_split_manifest(split_path, sorted(samples), identity, expected_seed=seed)
    train_ids, val_ids = list(manifest["train_ids"]), list(manifest["val_ids"])
    if max_train_universes is not None: train_ids = train_ids[:max_train_universes]
    if max_val_universes is not None: val_ids = val_ids[:max_val_universes]
    if not train_ids or not val_ids: raise ValueError("train and validation partitions must be nonempty")
    train_loader = DataLoader(_Dataset(samples, train_ids), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), collate_fn=_collate)
    val_loader = DataLoader(_Dataset(samples, val_ids), batch_size=batch_size, shuffle=False, collate_fn=_collate)
    dev = torch.device(device); model = PeriodicInvariantGNNRegressor().to(dev)
    if count_parameters(model) != 5270: raise RuntimeError("parameter-count guard failed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    out.mkdir(parents=True, exist_ok=True); best, stale, best_epoch = float("inf"), 0, 0; history=[]; ckpt = out / "best_validation_checkpoint.pt"
    def evaluate(loader):
        model.eval(); total=0.; n=0; preds=[]
        with torch.no_grad():
            for ids, graph, target in loader:
                graph=sparse_batch_to(graph, dev); target=target.to(dev); node, edge=periodic_invariant_features(graph["x"], graph["edge_index"])
                prediction=model(node, graph["edge_index"], edge, graph["batch"], graph["num_graphs"]); total += torch.nn.functional.mse_loss(prediction,target,reduction="sum").item(); n += len(ids); preds += list(zip(ids,prediction.cpu().view(-1).tolist(),target.cpu().view(-1).tolist()))
        return total/n, preds
    for epoch in range(1, epochs+1):
        model.train(); total=0.; n=0
        for ids, graph, target in train_loader:
            graph=sparse_batch_to(graph, dev); target=target.to(dev); node, edge=periodic_invariant_features(graph["x"], graph["edge_index"]); optimizer.zero_grad()
            loss=torch.nn.functional.mse_loss(model(node,graph["edge_index"],edge,graph["batch"],graph["num_graphs"]),target); loss.backward(); clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total += loss.item()*len(ids); n += len(ids)
        val_mse, _ = evaluate(val_loader); train_mse=total/n; scheduler.step(val_mse); history.append({"epoch":epoch,"train_mse":train_mse,"validation_mse":val_mse})
        if val_mse < best:
            best, stale, best_epoch = val_mse, 0, epoch; torch.save({"model_state_dict":model.state_dict(),"seed":seed,"epoch":epoch},ckpt)
        else: stale += 1
        if stale >= patience: break
    payload={"seed":seed,"dataset_path":str(dataset_path),"dataset_sha256":_sha(dataset_path),"split_manifest_path":str(split_path),"split_manifest_sha256":_sha(split_path),"protocol_freeze_sha256":_sha(protocol),"test_status":"pending / forbidden","scientific_run":not(max_train_universes or max_val_universes or epochs < 300),"best_epoch":best_epoch,"best_validation_mse":best,"checkpoint":"best_validation_checkpoint.pt","checkpoint_sha256":_sha(ckpt),"history":history,"parameter_count":5270}
    (out/"metadata.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");
    with (out/"epoch_metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=history[0]); w.writeheader(); w.writerows(history)
    return payload

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--dataset-path",required=True); p.add_argument("--split-manifest-path",required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--output-dir",required=True); p.add_argument("--protocol-freeze-path",required=True); p.add_argument("--epochs",type=int,default=300); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--patience",type=int,default=40); p.add_argument("--max-train-universes",type=int); p.add_argument("--max-val-universes",type=int); p.add_argument("--device",default="cpu"); a=p.parse_args(); train(dataset_path=a.dataset_path,split_manifest_path=a.split_manifest_path,seed=a.seed,output_dir=a.output_dir,protocol_freeze_path=a.protocol_freeze_path,epochs=a.epochs,batch_size=a.batch_size,patience=a.patience,max_train_universes=a.max_train_universes,max_val_universes=a.max_val_universes,device=a.device)
