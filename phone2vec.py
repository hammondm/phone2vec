import warnings
warnings.simplefilter(
	action='ignore',
	category=FutureWarning
)
import gensim
from gensim.models import Word2Vec
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#etexts = 'etexts'
etexts = 'etextsSMALL'

#number of models
mnum = 100
#size of embedding (divisible by 4)
vecsize = 8
#size of window
pwindow = 2

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

training = 'abcc bbccba aaaa bacb ccbbaa bbbbccbaa'
training = training.split()
training = [list(t) for t in training]

tmod = Word2Vec(
	training,
	vector_size=2,
	window=1,
	sg=1
)

#read cmudict
f = open('/Users/hammond/' + etexts + '/cmu06/cmudict.0.6','r')
t = f.read()
f.close()

#fix booboo: E21 -> EH2
t = re.sub('E21','EH2',t)

#break into lines
t = t.split('\n')
#strip header
t = t[49:]

#create list of lists of sounds
words = []
for line in t:
	m = re.search('  *',line)
	if m:
		word = line[m.start():].split()
		words.append(word)

#calculate total number of sounds
sounds = []
for word in words:
	for letter in word:
		sounds.append(letter)
sounds = set(sounds)

#make skipgram models
models = []

#only do this ONCE!
#for m in range(mnum):
#	model = Word2Vec(
#		words,
#		min_count=100,
#		size=vecsize,
#		window=pwindow,
#		negative=10,
#		workers=4,
#		#iter=10,
#		sg=1
#	)
#	models.append(model.wv)
#
#import os
#os.mkdir('models')
#for i in range(len(models)):
#	models[i].save('models/testsav' + str(i) + '.kvmodel')

from gensim.models import KeyedVectors
for m in range(mnum):
	models.append(
		KeyedVectors.load(
			'models/testsav' + str(m) + '.kvmodel'
		)
	)

#function to get model avg
def getsim(mods,n,s,**kwargs):
	if 'topn' in kwargs:
		n = kwargs['topn']
		del kwargs['topn']
	all = {}
	for snd in s:
		all[snd] = 0
	for mod in mods:
		res = mod.most_similar(
			topn=len(s),
			**kwargs
		)
		for r in res:
			all[r[0]] += r[1]
	for a in all:
		all[a] /= len(mods)
	return sorted(
		all.items(),
		key=lambda x: x[1],
		reverse=True
	)[:n]

def printclusters(n):
	numclusters = n
	kmclusters = kms[numclusters-2].predict(mzerovals)
	kmpairs = zip(mzeronames,kmclusters)
	clusters = []
	for i in range(numclusters):
		clusters.append('')
	for kmp in sorted(kmpairs,key=lambda x: x[1]):
		clusters[kmp[1]] = clusters[kmp[1]] + ipa[kmp[0]] + ', '
	print(r'\begin{enumerate}')
	for c in clusters:
		print(r'\item',c[:-2],'\n\n')
	print(r'\end{enumerate}')

def simtab(sim):
	print(r'\begin{tabular}[t]{cr}')
	for p in sim:
		s = r'{} & {:.3f} \\'
		s = s.format(ipa[p[0]],p[1])
		print(s)
	print(r'\end{tabular}')

#does doesnt_match() for all models
def doesntset(l):
	res = {}
	for m in models:
		ptkb = m.doesnt_match(l)
		if ptkb in res:
			res[ptkb] += 1
		else:
			res[ptkb] = 1
	print(r'\begin{tabular}[t]{ll}')
	for x in res:
		print(ipa[x],'&',res[x],r'\\')
	print(r'\end{tabular}')
