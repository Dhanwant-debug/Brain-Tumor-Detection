"""Small helpers to store and load training information (history, time, parameters)."""
import json
import os
import config as cfg


def _info_path(key):
    return os.path.join(cfg.RESULTS_DIR, f"{key}_run_info.json")


def save_run_info(key, histories, seconds, model):
    """histories: list of Keras History.history dicts (MobileNetV2 has two phases)."""
    merged = {}
    for h in histories:
        for k, v in h.items():
            merged.setdefault(k, []).extend(float(x) for x in v)
    info = {"train_time_sec": round(seconds, 1), "epochs_run": len(merged.get("loss", [])),
            "params": int(model.count_params()), "history": merged}
    with open(_info_path(key), "w") as f:
        json.dump(info, f)
    print(f"Training time: {seconds:.0f}s | epochs run: {info['epochs_run']} | params: {info['params']:,}")


def load_run_info(key):
    p = _info_path(key)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)
