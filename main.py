import re
import numpy as np

# zmiana na małe litery
def tokenizacja(text):
    return re.findall(r'\b\w+\b', text.lower())

# input
n = int(input().strip())
dokumenty = [input().strip() for i in range(n)]
zapytanie = input().strip()
k = int(input().strip())

# Przygotuj macierz term-dokument 𝐶
termy = sorted(set(word for doc in dokumenty for word in tokenizacja(doc)))
term_document = np.array([[1 if term in tokenizacja(doc) else 0 for doc in dokumenty] for term in termy])

# Przeprowadź dekompozycję macierzy 𝐶wg SVD,
U, S, Vt = np.linalg.svd(term_document, full_matrices=False)

# Przeprowadź aproksymację rzędu 𝑘, otrzymując 𝐶𝑘
Uk = U[:, :k]
Sk = np.diag(S[:k])
Vk = Vt[:k, :]

# Oblicz macierz 𝛴𝑘𝑉𝑘𝑇, której kolumny są wektorami dokumentów w zredukowanej przestrzeni
Ck = Uk @ Sk @ Vk

# Oblicz wektor zapytania w zredukowanej przestrzeni 𝑞𝑘
zapytanie_wektor = np.array([1 if term in tokenizacja(zapytanie) else 0 for term in termy])
zapytanie_zredukowane = np.linalg.inv(Sk) @ Uk.T @ zapytanie_wektor

# Oblicz podobieństwo zapytania do każdego z dokumentów wg miary cosinusa
miara_cosinus = []
for dok_wektor in (Sk @ Vk).T:
    numerator = np.dot(zapytanie_zredukowane, dok_wektor)
    denominator = np.linalg.norm(zapytanie_zredukowane) * np.linalg.norm(dok_wektor)
    similarity = numerator / denominator if denominator != 0 else 0
    miara_cosinus.append(round(float(similarity), 2))

# output
print(miara_cosinus)
