import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import diags
from scipy.sparse.linalg import cg

n=100
o=1.5
Grid_V=np.zeros((n, n))

taille_grille = n * n
diagonale = np.full(taille_grille, 4.0)
indices = np.arange(n)
diagonale[indices * n] = 3.0
diagonale[(indices + 1) * n - 1] = 3.0
diagonale_horizontale = np.full(taille_grille - 1, -1.0)
diagonale_horizontale[indices[1:] * n - 1] = 0.0
diagonale_verticale = np.full(taille_grille - n, -1.0)
Laplacien = diags(
    [
        diagonale,
        diagonale_horizontale,
        diagonale_horizontale,
        diagonale_verticale,
        diagonale_verticale,
    ],
    [0, -1, 1, -n, n],
    format="csr",
)


def Voisins(i, j):
    voisins=[]
    if i>0:
        voisins.append((i-1, j))
    if i<n-1:
        voisins.append((i+1, j))
    if j>0:
        voisins.append((i, j-1))
    if j<n-1:
        voisins.append((i, j+1))
    return voisins


def Calcul_Potentiel(eclair, potentiel_initial=None):
    if potentiel_initial is None:
        potentiel = np.tile(np.linspace(0, 1, n)[:, None], (1, n))
    else:
        potentiel = potentiel_initial.copy()
    potentiel[0, :] = 0
    potentiel[-1, :] = 1

    fixe = np.zeros((n, n), dtype=bool)
    fixe[0, :] = True
    fixe[-1, :] = True
    for i, j in eclair:
        if 0 < i < n - 1:
            fixe[i, j] = True
            potentiel[i, j] = 0

    libres = ~fixe
    valeurs_fixes = np.where(fixe, potentiel, 0)
    second_membre = np.zeros((n, n))
    second_membre[1:, :] += valeurs_fixes[:-1, :]
    second_membre[:-1, :] += valeurs_fixes[1:, :]
    second_membre[:, 1:] += valeurs_fixes[:, :-1]
    second_membre[:, :-1] += valeurs_fixes[:, 1:]

    indices_libres = np.flatnonzero(libres)
    operateur = Laplacien[indices_libres][:, indices_libres]
    solution, info = cg(
        operateur,
        second_membre[libres],
        x0=potentiel[libres],
        rtol=1e-4,
        atol=0,
        maxiter=3000,
    )
    if info != 0:
        raise RuntimeError(f"Le calcul du potentiel n'a pas converge (info={info}).")

    potentiel[libres] = np.clip(solution, 0, 1)
    return potentiel


def Voisins_Disponibles(eclair):
    eclair_set = set(eclair)
    return sorted({
        voisin
        for i, j in eclair
        for voisin in Voisins(i, j)
        if voisin[0] != 0 and voisin not in eclair_set
    })


def proba_voisins(voisins, potentiel):
    poids = np.maximum([potentiel[x, y] for x, y in voisins], 0) ** o
    total = np.sum(poids)
    if total == 0:
        return np.full(len(voisins), 1 / len(voisins))
    return poids / total


def Choice_Voisin(eclair, potentiel):
    voisins = Voisins_Disponibles(eclair)
    if not voisins:
        return None
    proba = proba_voisins(voisins, potentiel)
    return voisins[np.random.choice(len(voisins), p=proba)]


Strike = [(0, n // 2)]
Grid_V = Calcul_Potentiel(Strike, Grid_V)
Historique_Potentiel = [Grid_V.astype(np.float32)]

while not any(i == n - 1 for i, _ in Strike):
    suivant = Choice_Voisin(Strike, Grid_V)
    if suivant is None:
        break
    Strike.append(suivant)
    Grid_V = Calcul_Potentiel(Strike, Grid_V)
    Historique_Potentiel.append(Grid_V.astype(np.float32))

if not any(i == n - 1 for i, _ in Strike):
    print("L'eclair est bloque avant d'atteindre le bas.")

eclair_cmap = LinearSegmentedColormap.from_list(
    "eclair_electrique",
    ["#12002f", "#183bc9", "#7627de", "#e33aaf", "#ff846f", "#ffe66b"],
)

fig, ax = plt.subplots(figsize=(8, 8), facecolor="#100820")
ax.set_facecolor("#100820")
champ = ax.imshow(
    Grid_V,
    cmap=eclair_cmap,
    interpolation="bilinear",
    vmin=0,
    vmax=1,
)
barre = fig.colorbar(champ, ax=ax, label="Potentiel")
barre.ax.yaxis.set_tick_params(color="#e8dcf5")
plt.setp(barre.ax.get_yticklabels(), color="#e8dcf5")
barre.set_label("Potentiel", color="#e8dcf5")

ligne = [col for _, col in Strike]
colonne = [row for row, _ in Strike]
halo = ax.scatter([], [], s=110, color="#ffb629", alpha=0.22, zorder=3)
coeur = ax.scatter([], [], s=28, color="#ffd84d", alpha=0.9, zorder=4)
noyau = ax.scatter([], [], s=8, color="#fffdf4", zorder=5)
pointe = ax.scatter([], [], s=42, color="#fffdf4", edgecolors="#ffd84d",
                    linewidths=1.2, zorder=6)
titre = ax.set_title("", color="#fff4d6", fontsize=14, pad=14)
ax.set_xlabel("Position horizontale", color="#e8dcf5")
ax.set_ylabel("Position verticale", color="#e8dcf5")
ax.tick_params(colors="#e8dcf5")


def initialiser():
    for trace in (halo, coeur, noyau):
        trace.set_offsets(np.empty((0, 2)))
    champ.set_data(Historique_Potentiel[0])
    pointe.set_offsets(np.empty((0, 2)))
    titre.set_text("")
    return champ, halo, coeur, noyau, pointe, titre


def animer(etape):
    fin = etape + 1
    champ.set_data(Historique_Potentiel[etape])
    positions = np.column_stack((ligne[:fin], colonne[:fin]))
    for trace in (halo, coeur, noyau):
        trace.set_offsets(positions)
    pointe.set_offsets([[ligne[etape], colonne[etape]]])
    titre.set_text(
        f"Développement de l'éclair — étape {fin}/{len(Strike)}",
    )
    return champ, halo, coeur, noyau, pointe, titre


animation = FuncAnimation(
    fig,
    animer,
    frames=len(Strike),
    init_func=initialiser,
    interval=10,
    blit=True,
    repeat=False,
)
plt.show()