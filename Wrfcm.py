import numpy as np
import cv2
import matplotlib.pyplot as plt



# L'algorithme WRFCM
def WRFCM(X, I, nn,mm,c, m,n, eps, xi, beta,max_iter):
    K, L = X.shape
    U = np.random.rand(c, K)
    U = U / U.sum(axis=1)[:, np.newaxis]
    V = np.ones((c,L))
    W = np.ones((K,L))
    R = np.zeros((K, L))
    Conv = []

    window_size = n
    d = np.zeros(K)
    # Appliquer la fenêtre pour chaque pixel
    for j in range(K):
        # Convertir l'indice j en coordonnées (i, k)
        i, k = divmod(j, nn)

        # Calculer les indices de la fenêtre pour l'image originale
        i_min, i_max = max(0, i - window_size // 2), min(mm, i + window_size // 2 + 1)
        k_min, k_max = max(0, k - window_size // 2), min(L, k + window_size // 2 + 1)

        # Extraire la fenêtre de l'image originale
        window = I[i_min:i_max, k_min:k_max, :]

        # Ajuster la dimension pour pouvoir calculer la norme
        window_flat = window.reshape(-1, L)

        # Calculer la distance euclidienne et la somme
        d[j] = np.linalg.norm(I[i, k, :] - window_flat, axis=1).sum()


    for it in range(max_iter):

        # Calcul des centres des clusters V
        for i in range(c):
            for l in range(L):
                numerator_sum = 0
                denominator_sum = 0

                for j in range(K):
                    numerator_sum += (U[i, j] ** m) * np.sum((X[j, l] - R[j, l]) / (1 + d[j]))
                    denominator_sum += (U[i, j] ** m) * np.sum(1 / (1 + d[j]))

                V[i, l] = numerator_sum / denominator_sum

        # Calcul de la matrice résiduelle R
        for i in range(K):
            for j in range(L):
                numerator = 0
                denominator = 0
                for q in range(c):
                    numerator += (U[q, i] ** m) * (X[i, j] - V[q, j]) / (1 + d[i])
                    denominator += (U[q, i] ** m) / (1 + d[i])

                R[i, j] = numerator / (denominator + (2 * beta[j] * (W[i,j] ** 2)) / (1 + d[i]))

        # Mise à jour des poids W
        W = np.exp(-xi * (R ** 2))

        # Mise à jour des degrés d'appartenance
        new_U = np.zeros_like(U)
        for j in range(K):
            for i in range(c):
                numerator = np.sum((np.linalg.norm(X[j, :] - R[j, :] - V[i, :]) ** 2 / (1 + d[j])) ** (-1 / (m - 1)))
                denominator = np.sum([(np.sum((np.linalg.norm(X[j, :] - R[j, :] - V[q, :]) ** 2 / (1 + d[j])) ** (-1 / (m - 1)))) for q in range(c)])
                new_U[i, j] = numerator / denominator

        new_U = new_U / new_U.sum(axis=1)[:, np.newaxis]



        print("U(t+1) - U(t) = ", np.linalg.norm(new_U - U))
        Conv.append(np.linalg.norm(new_U - U))
        # Vérification de la convergence
        if np.linalg.norm(new_U - U) < eps:
            break

        U = new_U


    return new_U, V, R, W, (it+1) , Conv


# You must change the path of the image and the initial parameters 
'''

# Paramètres
c = 3
m = 2
n = 3
eps = 10e-6
xi = 0.0001
max_iter = 100

#Chargement de l'image
I = cv2.imread('Your Path.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
#I = cv2.resize(I, (200, 200))
mm, nn, dd = I.shape

#Reshape
X = I.reshape(mm * nn, dd)

#Calcul du parametre Beta
beta = np.zeros(dd)
for l in range(dd):
    beta[l] = (10 * np.std(X[:, l])) / 100

#Appel de WRFCM
U, V, R, W, it, Conv = WRFCM(X, I, nn,mm,c, m,n, eps, xi, beta,max_iter)

print("Algorithme terminé après ", it, " itérations")

# Choisir la classe majoritaire pour chaque pixel
labels = np.argmax(U, axis=0)

#Remodeler les étiquettes pour l'image originale
label_update = labels.reshape(mm, nn)

#Création d'une carte de couleurs pour les clusters
cluster_colors = np.random.randint(0, 256, size=(c, 3), dtype=np.uint8)

#Génération de l'image segmentée en utilisant les labels
segmented_image = np.zeros((mm, nn, 3), dtype=np.uint8)

for i in range(c):
    #Attribution d'une couleur différente à chaque cluster
    segmented_image[label_update == i] = cluster_colors[i]



# Affichage du résultat du clustering avec Matplotlib
plt.figure(figsize=(12, 6))

# Affichage de l'image d'origine
plt.subplot(1, 2, 1)
plt.imshow(I)
plt.title('Observed Image')
plt.axis('off')

# Affichage de l'image segmentée
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('WRFCM Result')
plt.axis('off')

plt.show()

'''