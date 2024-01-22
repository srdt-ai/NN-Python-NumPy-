def init_network(vocab_size, n_embedding):
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }
    return model

model = init_network(len(word_to_id), # p = nombre de dimensions de l'espace vectorise)

def forward(model, X, return_cache=True):
    cache = {}
    
    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    
    if not return_cache:
        return cache["z"]
    return cache

def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

(X @ model["w1"]).shape

(# w = nombre d'exemples, # p = nombre de dimensions)
    
(X @ model["w1"] @ model["w2"]).shape

(# w = nombre d'exemples, # z = nombre de tokens / segments dans le texte)
    







