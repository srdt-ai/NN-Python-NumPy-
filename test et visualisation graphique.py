import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
plt.style.use("seaborn")

n_iter = # m = depend de la taille du corpus (test sur 300 spour CorpusThomisticum)
learning_rate = 0.05

history = [backward(model, X, y, learning_rate) for _ in range(n_iter)]

plt.plot(range(len(history)), history, color="skyblue")
plt.show()

learning = one_hot_encode(word_to_id["learning"], len(word_to_id))
result = forward(model, [learning], return_cache=False)[0]

for word in (id_to_word[id] for id in np.argsort(result)[::-1]):
    print(word)

### vectorisation generale (de l'ensemble du texte)

model["w1"]

### vectorisation particuliere (recherche par mot

get_embedding(model, "exemple_de_mot (test: actus)")



