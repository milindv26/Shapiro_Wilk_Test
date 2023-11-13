import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import pandas as pd

# Generuj losowe dane1 i zdefiniuj alpha
# Generate random data1 and define alpha
#np.random.seed(123)
data = np.random.normal(0, 1, 30)
alpha = 0.05

# Przechowuj statystyki testowe
# Store test statistics

iterations = range(1, 1001)
p_values = np.zeros(len(iterations)) # Macierz zer na średnie p-values | Matrix of zeros to mean p-values
powers = np.zeros(len(iterations)) # Macierz zer na średnia moc testu | Matrix of zeros on average test power
blad_II_r = np.zeros(len(iterations))

# Moc testu liczy sie dla danych ktore pochodza z rozkladu innego niz normalny
# The power of the test counts for data that comes from a distribution other than normal

# Żeby zrobić wykres z punktami w krzywej Gaussa trzeba dla kazdego x policzyc gestosc funkcji w tym punkcie i wtedy zrobic wykres

# Przeprowadź test Shapiro-Wilka wielokrotnie
for index, n_iter in enumerate(iterations):
    pv_test = np.zeros(n_iter) # Macierz zer na p-value z każdego powtórzenia testu
    po_test = np.zeros(n_iter) # Macierz zer na p-value < alpha z każdego powtórzenia testu

    # Wielokrotne powtarzanie testu shapiro-wilka i zapisywanie p-value do macierzy
    for i in range(0,n_iter):
        random_data = np.random.normal(0, 1, 30)
        other_data = stats.expon.rvs(size = 30)
        p1 = stats.shapiro(random_data).pvalue
        p2 = stats.shapiro(other_data).pvalue
        
        # Dodawanie p-value do macierzy
        pv_test[i] = p1

        # Wstawianie 1, gdy p-value < alpha
        po_test[i] = int(p2 <= alpha)

    # Przypisywanie danemu indeksowi odpowiedającemu iteracji w liście p_values średnie wszystkich p-value z MC
    p_values[index] = np.mean(pv_test)
    powers[index] = np.mean(po_test)
    blad_II_r[index] = 1 - np.mean(po_test)

dane1 = np.vstack((iterations, p_values))
dane2 = np.vstack((iterations, powers))
dane3 = np.vstack((iterations, blad_II_r))

sns.set_theme()

# Wykres p-value w zależności do iteracji
fig, ax = plt.subplots()
ax.scatter(dane1[0][dane1[1] >= alpha], dane1[1][dane1[1] >= alpha], color = 'green', label = f'p-value > {alpha}')
ax.scatter(dane1[0][dane1[1] <= alpha], dane1[1][dane1[1] <= alpha], color = 'red', label = f'p-value < {alpha}')
ax.plot(dane1[0], dane1[1], color = "black", alpha = 0.5)
plt.title('p-value testu w zależności od ilości iteracji')
plt.ylim(-0.05, 1.05)
plt.xlabel('iteracja')
plt.ylabel('p-value')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend()
plt.show()

# Wykres błędu II rodzaju i mocy testu w zależności od iteracji
fig, ax = plt.subplots()
ax.plot(dane3[0], dane3[1], color = 'orange', label = 'Błąd II rodzaju')
ax.plot(dane2[0], dane2[1], color = '#af0b1e', label = 'Moc testu')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim(-0.05, 1.05)
plt.xlabel('iteracja')
plt.ylabel('')
plt.title('Błąd II rodzaju i moc testu w zależności od ilości iteracji')
plt.legend()
plt.show()

# Wykres błędu II rodzaju testu w zależności od iteracji
fig, ax = plt.subplots()
ax.plot(dane3[0], dane3[1], color = 'orange')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim(-0.05, 1.05)
plt.xlabel('iteracja')
plt.ylabel('błąd II rodzaju')
plt.title('Błąd II rodzaju w zależności od ilości iteracji')
plt.show()

# Wykres mocy testu w zależności od iteracji
fig, ax = plt.subplots()
ax.plot(dane2[0], dane2[1], color = '#af0b1e')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim(-0.05, 1.05)
plt.xlabel('iteracja')
plt.ylabel('moc testu')
plt.title('Moc testu w zależności od ilości iteracji')
plt.show()

# Wykres błędu absolutnego
def gen_mc(n_iter):
    # Przechowuj statystyki testowe
    pv = np.zeros(n_iter)
    # Przeprowadź test Shapiro-Wilka wielokrotnie
    for i in range(n_iter):
        random_data = np.random.normal(0, 1, 30)
        p = stats.shapiro(random_data).pvalue
        pv[i] = p

    result = abs(stats.shapiro(data).pvalue - np.mean(pv))
    return result

it = range(0, 501)
bbez_lst=[]

for n in it:
    bezwzgledny = gen_mc(n)
    bbez_lst.append(bezwzgledny)

data = {'iteracje': it,
        'bbez': bbez_lst}
df = pd.DataFrame(data)
#print(df.head(10))

sns.set_theme()
fig, ax = plt.subplots(figsize=(6, 4))
p1 = sns.scatterplot(x='iteracje', y='bbez', data=df, ax=ax, color='red')
p2 = sns.lineplot(x='iteracje', y='bbez', data=df, ax=ax)
plt.ylim(-0.01, 1)
plt.ylabel('błąd bezwzględny')
plt.title('Błąd bezwględny pomiędzy testem teoretycznym a metodą Monte Carlo')
plt.show() 

# Wykres czasu na potrzebnego na iteracje
def gen_mc(n_iter):
    # Przechowuj statystyki testowe
    pv = np.zeros(n_iter)
    # Przeprowadź test Shapiro-Wilka wielokrotnie
    for i in range(n_iter):
        random_data = np.random.normal(0, 1, 30)
        p = stats.shapiro(random_data).pvalue
        pv[i] = p

sns.set_theme()

it = [10, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
time_lst = []
for n in it:
    execution_time = timeit.timeit(stmt='gen_mc(n)', setup='from __main__ import gen_mc, n', number=1)
    time_lst.append(execution_time)

data = {'iteracje': it,
        'time': time_lst}
df = pd.DataFrame(data)
#print(df.head(10))

fig, ax = plt.subplots(figsize=(6, 4))
p1 = sns.scatterplot(x='iteracje', y='time', data=df, ax=ax, color='red')
p2 = sns.lineplot(x='iteracje', y='time', data=df, ax=ax)
plt.ylabel('czas [s]')
plt.title('Złożoność obliczeniowa metody Monte Carlo zobrazowana w czasie')
plt.show()
