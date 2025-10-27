import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec

#number of models
mnum = 100
#size of embedding (divisible by 4) 
vecsize = 8
#size of window
pwindow = 2

models = []

ipa = {
	'EH2':"ɛ̀",			'K':"k",
	'S':"s",			'L':"l",
	'AH0':"ə",			'M':"m",
	'EY1':"é",			'SH':"š",
	'N':"n",			'P':"p",
	'OY2':"ɔj",			'T':"t",
	'OW1':"ó",			'Z':"z",
	'W':"w",			'D':"d",
	'AH1':"ʌ́",			'B':"b",
	'EH1':"ɛ́",			'V':"v",
	'IH1':r"ɪ́",			'AA1':"á",	
	'R':"r",			'AY1':"áj",
	'ER0':"r̩",			'AE1':"ǽ",
	'AE2':"æ̀",			'AO1':"ɔ́",
	'NG':"ŋ",			'G':"g",
	'IH0':"ɪ",			'TH':"θ",
	'IY2':"ì",			'F':"f",
	'DH':"ð",			'IY1':"í",
	'HH':"h",			'UH1':"ʊ́",
	'IY0':"i",			'OY1':"ɔ́j",
	'OW2':"ò",			'CH':"č",
	'UW1':"ú",			'IH2':"ɪ̀",
	'EH0':"ɛ",			'AO2':"ɔ̀",
	'AA0':"a",			'AA2':"à",
	'OW0':"o",			'EY0':"e",
	'AE0':"æ",			'AW2':"àw",
	'AW1':"áw",			'EY2':"è",
	'UW0':"u",			'AH2':"ʌ̀",
	'UW2':"ù",			'AO0':r"ɔ",
	'JH':r"ǰ",			'Y':"j",
	'ZH':"ž",			'AY2':"àj",
	'ER1':"ŕ̩",			'UH2':"ʊ̀",
	'AY0':"aj",			'ER2':"r̩̀",
	'OY0':"ɔj",			'UH0':"ʊ",
	'AW0':"aw"
}

from gensim.models import KeyedVectors
for m in range(mnum):
	models.append(
		KeyedVectors.load(
			'../paper/models/testsav' + str(m) + '.kvmodel'
		)
	)

keys = []
vals = []
for key in models[0].index_to_key:
	keys.append(ipa[key])
	vals.append(models[0][key])

embedding_vectors = np.array(vals)

tsne = TSNE(n_components=2,random_state=42,perplexity=3)

reduced_embeddings = tsne.fit_transform(embedding_vectors)

plt.figure(figsize=(10,8))
plt.scatter(reduced_embeddings[:,0],reduced_embeddings[:,1])

for i,word in enumerate(keys):
	plt.annotate(
		word,
		(reduced_embeddings[i,0],reduced_embeddings[i,1]),
		textcoords="offset points",
		xytext=(5,5),
		ha='center'
	)

plt.title('2D Visualization of Embeddings (t-SNE)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()

