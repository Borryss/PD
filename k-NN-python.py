import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# === MAŁY ZBIÓR DANYCH ===
X_small = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7],
    [8, 6]
])
y_small = np.array([0, 0, 0, 1, 1, 1])

# === DUŻY ZBIÓR DANYCH ===
X_large = np.array([
    [1, 2], [2, 1], [2, 3], [3, 3], [1, 3], [3, 2],
    [6, 5], [7, 6], [8, 6], [7, 8], [6, 7], [8, 7]
])
y_large = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Punkt testowy
nowy_punkt = np.array([[5, 5]])

# Lista konfiguracji
konfiguracje = [
    ('Mały zbiór, k=3, metryka: euklidesowa', X_small, y_small, 3, 'euclidean'),
    ('Mały zbiór, k=3, metryka: manhattan',   X_small, y_small, 3, 'manhattan'),
    ('Duży zbiór, k=3, metryka: euklidesowa', X_large, y_large, 3, 'euclidean'),
    ('Duży zbiór, k=7, metryka: euklidesowa', X_large, y_large, 7, 'euclidean'),
    ('Duży zbiór, k=7, metryka: manhattan',   X_large, y_large, 7, 'manhattan'),
]

# Tworzymy subploty
fig, axs = plt.subplots(3, 2, figsize=(16, 20))
axs = axs.flatten()

for i, (tytul, X, y, k, metryka) in enumerate(konfiguracje):
    model = KNeighborsClassifier(n_neighbors=k, metric=metryka)
    model.fit(X, y)
    pred = model.predict(nowy_punkt)

    axs[i].scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', label='Klasa 0')
    axs[i].scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', label='Klasa 1')
    axs[i].scatter(nowy_punkt[:, 0], nowy_punkt[:, 1],
                   color='green' if pred[0]==0 else 'orange',
                   edgecolors='black', s=120,
                   label=f'Nowy punkt (klasa {pred[0]})')

    # Linie do sąsiadów
    odleglosci, indeksy = model.kneighbors(nowy_punkt)
    for idx in indeksy[0]:
        axs[i].plot([X[idx, 0], nowy_punkt[0, 0]], [X[idx, 1], nowy_punkt[0, 1]], 'k--', alpha=0.5)

    axs[i].set_title(tytul, fontsize=13)
    axs[i].legend()
    axs[i].grid(True)

# Usuwamy ostatni pusty wykres (jeśli jest)
if len(konfiguracje) % 2 != 0:
    fig.delaxes(axs[-1])

# Większe odstępy między wykresami
plt.subplots_adjust(
    left=0.08,
    right=0.95,
    top=0.93,
    bottom=0.05,
    wspace=0.3,
    hspace=0.4
)

plt.suptitle('Porównanie działania metody k Najbliższych Sąsiadów (k-NN)', fontsize=18)
plt.show()
