"""kltn_model.py — dùng model cuối của đồ án (không cần notebook).

    from kltn_model import load
    m = load("final_model.pt")
    P = m.predict([text_vec, vision_vec])     # (n, 20) xác suất; mỗi vec shape (n, dim) theo m.inputs
    labels = m.decide(P)                       # list nhãn mỗi video (ngưỡng đã tune)
    prof = m.profile(P)                        # hồ sơ creator: TB xác suất các video (bỏ Others), cần >= 5 video
"""
import numpy as np, torch, torch.nn as nn


class TwoTower(nn.Module):
    def __init__(self, dims, n_labels, d=256, hidden=256, dropout=0.2, fuse="concat", proj_layers=2, aux=False):
        super().__init__()
        def mk(k):
            layers = [nn.Linear(k, d)]
            for _ in range(proj_layers - 1):
                layers += [nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d)]
            return nn.Sequential(*layers)
        self.pt, self.pv = mk(dims[0]), mk(dims[1])
        self.nt, self.nv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fuse = fuse
        if fuse == "gate":
            self.gate = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Linear(64, 2))
        elif fuse == "gmu":
            self.gate = nn.Linear(2 * d, d)
        din = d if fuse in ("gate", "gmu") else 2 * d
        self.head = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_labels))
        self.aux = aux
        if aux:
            self.aux_t, self.aux_v = nn.Linear(d, n_labels), nn.Linear(d, n_labels)

    def forward(self, xt, xv):
        ht, hv = self.nt(self.pt(xt)), self.nv(self.pv(xv))
        if self.fuse == "gate":
            a = torch.softmax(self.gate(torch.cat([ht, hv], 1)), 1)
            h = a[:, :1] * ht + a[:, 1:] * hv
        elif self.fuse == "gmu":
            z = torch.sigmoid(self.gate(torch.cat([ht, hv], 1)))
            h = z * torch.tanh(ht) + (1 - z) * torch.tanh(hv)
        else:
            h = torch.cat([ht, hv], 1)
        return self.head(h)


class MLP(nn.Module):
    def __init__(self, dims, n_labels, hiddens=(256,), dropout=0.2):
        super().__init__()
        layers, d = [], sum(dims)
        for h in hiddens:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_labels)]
        self.net = nn.Sequential(*layers)

    def forward(self, *xs):
        return self.net(torch.cat(xs, dim=1))


class Predictor:
    def __init__(self, ck, model, device):
        self.ck, self.model, self.device = ck, model, device
        self.labels, self.threshold, self.inputs = ck["labels"], ck["threshold"], ck["inputs"]

    def _norm(self, X, st):
        X = np.asarray(X, dtype=np.float32)
        if self.ck["norm"] == "zscore":
            return (X - st["mean"]) / st["std"]
        if self.ck["norm"] == "scale":
            return X / st["div"]
        return X

    @torch.no_grad()
    def predict(self, xs):
        xs = [torch.tensor(self._norm(X, st), device=self.device) for X, st in zip(xs, self.ck["stats"])]
        out = self.model(*xs)
        out = out[0] if isinstance(out, tuple) else out
        return torch.sigmoid(out).cpu().numpy()

    def decide(self, P):
        P = np.atleast_2d(P)
        D = P >= self.threshold
        D[np.arange(len(P)), P.argmax(1)] = True        # luôn có ít nhất 1 nhãn
        return [[self.labels[j] for j in np.where(r)[0]] for r in D]

    def profile(self, P, exclude=None):
        exclude = self.ck.get("profile_exclude", ["Others"]) if exclude is None else exclude
        keep = [j for j, l in enumerate(self.labels) if l not in exclude]
        p = np.atleast_2d(P)[:, keep].mean(0)
        return dict(sorted(((self.labels[j], float(v)) for j, v in zip(keep, p)), key=lambda kv: -kv[1]))


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    a, L = ck["arch"], len(ck["labels"])
    if a["type"] == "TwoTower":
        model = TwoTower(a["dims"], L, d=a["d"], hidden=a["hidden"], fuse=a["fuse"], proj_layers=a["proj_layers"], aux=a["aux"])
    else:
        model = MLP(a["dims"], L, hiddens=a["hiddens"])
    model.load_state_dict(ck["state_dict"], strict=True)
    return Predictor(ck, model.to(device).eval(), device)
