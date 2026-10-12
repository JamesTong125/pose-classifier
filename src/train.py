import numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src import config as C
from src.model import PoseMLP


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on", device)

    X = np.load(C.PROC_DIR / "X.npy")
    y = np.load(C.PROC_DIR / "y.npy")
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)
    scaler = StandardScaler().fit(X_tr)

    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    Xtr, Xte = t(scaler.transform(X_tr), torch.float32), t(scaler.transform(X_te), torch.float32)
    ytr, yte = t(y_tr, torch.long), t(y_te, torch.long)

    model = PoseMLP(X.shape[1], len(le.classes_)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(C.EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), C.BATCH_SIZE):
            idx = perm[i:i + C.BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(f"epoch {epoch:3d}  test acc {acc:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1).cpu().numpy()
    print(f"\nBest test accuracy: {best_acc:.3f}")
    print(classification_report(y_te, preds, target_names=le.classes_))

    C.MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "state_dict": best_state, "in_dim": X.shape[1],
        "classes": list(le.classes_),
        "mean": scaler.mean_, "scale": scaler.scale_,
    }, C.MODEL_PATH)
    print("Saved", C.MODEL_PATH)


if __name__ == "__main__":
    main()
