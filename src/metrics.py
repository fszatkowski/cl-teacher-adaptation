from collections import Counter

import numpy as np
import torch


def cka(model1, model2, dataloader, device) -> float:
    model1.eval()
    model2.eval()
    with torch.no_grad():
        cka_list = []
        for images, _ in dataloader:
            images = images.to(device)
            _, features1 = model1(images, return_features=True)
            _, features2 = model2(images, return_features=True)
            _cka = _CKA(features1, features2)
            cka_list.append(_cka)

    return float(sum(cka_list) / len(cka_list))


def _CKA(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return _HSIC(X, Y) / torch.sqrt(_HSIC(X, X) * _HSIC(Y, Y))


def _HSIC(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    GX = X @ X.T
    GY = Y @ Y.T

    n = GX.shape[0]
    H = torch.eye(n, device=X.device) - (1 / n)

    return torch.trace(GX @ H @ GY @ H)


def cm(model, dataloaders, n_tasks, device):
    confusion_matrix = np.zeros((n_tasks, n_tasks))

    model.eval()
    with torch.no_grad():
        for i, dl in enumerate(dataloaders):
            task_ids = []
            for images, _ in dl:
                images = images.to(device)
                outputs = model(images)
                outputs = torch.stack(outputs, dim=1)
                outputs = torch.max(outputs, dim=-1)[0]
                task_ids.extend(outputs.argmax(dim=-1).tolist())
            counts = Counter(task_ids)
            for j, val in counts.items():
                confusion_matrix[i, j] = val / len(dl.dataset)

    return confusion_matrix
