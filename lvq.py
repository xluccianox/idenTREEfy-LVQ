import numpy as np
import neurolab as nl
import csv

#Defino las especies posibles
especies = {}
especies['Eucalyptus_Melliodora']  = [1,0,0,0,0,0,0,0]
especies['Ficus_Benjamina'] = [0,1,0,0,0,0,0,0]
especies['Ginkgo_Biloba'] = [0,0,1,0,0,0,0,0]
especies['Melia_Azedarach'] = [0,0,0,1,0,0,0,0]
especies['Platanus_Hispanica'] = [0,0,0,0,1,0,0,0]
especies['Populus_Nigra'] = [0,0,0,0,0,1,0,0]
especies['Quercus_Palustris'] = [0,0,0,0,0,0,1,0]
especies['Tilia_Cordata'] = [0,0,0,0,0,0,0,1]

#Leo el archivo del dataset CSV
input = []
target = []
with open('hojas-dataset.txt', 'rb') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
    for row in reader:
    	input.append(map(float,row[:5]))
        target.append(especies[row[5]])

# Creo el conjunto de entrenamiento
input = np.array(input)
target = np.array(target)

# Creo una red tipo Kohonen-LVQ con 50 neuronas en la capa competitiva, funcion de activacion equiprobable y 1000 iteraciones
net = nl.net.newlvq(nl.tool.minmax(input), 50, [.125,.125,.125,.125,.125,.125,.125,.125], epoch = 1000)

# Entreno la red
print 'Entrenando la red...'
error = net.train(input, target)

i = [[5.388059701492537,0.31503342325288,1.5585913587009053,1.8382633010917735,0.9979837101840756], # Eucalyptus_Melliodora
	 [2.139130434782609,0.5808038009859917,1.6809768561157492,1.6715470747274044,0.9966067089124588], # Ficus_Benjamina
	 [0.6124567474048442,0.6152522920151243,1.4249341894508127,1.8375135330171544,1.6309003448270327], # Ginkgo_Biloba
	 [0.9976190476190476,0.0668923642903687,2.70395267545039,4.167607506495124,1.0738159505234968], # Melia_Azedarach
	 [1.0182648401826484,0.6050438809231855,1.8147597636654156,1.6914304481372575,0.9972210981314964], # Platanus_Hispanica
	 [0.9900662251655629,0.6089311730083087,1.7975653697233918,1.6941157323548481,1.0096236072980627], # Populus_Nigra
	 [1.002247191011236,0.06594019800726614,2.844245086307583,4.092757670566289,1.0370482352372872], # Quercus_Palustris
	 [1.2478632478632479,0.5468422377075948,1.6229414745376236,1.8699762337712431,1.0078221415641342] # Tilia_Cordata
	 ]

 # Clasifico especimenes
output = net.sim(i)

# Imprimo los resultados
print 'Resultados:'
for o in output:
	if(o[0] == 1):
		print 'Eucalyptus_Melliodora'
	elif(o[1] == 1):
		print 'Ficus_Benjamina'
	elif(o[2] == 1):
		print 'Ginkgo_Biloba'
	elif(o[3] == 1):
		print 'Melia_Azedarach'
	elif(o[4] == 1):
		print 'Platanus_Hispanica'
	elif(o[5] == 1):
		print 'Populus_Nigra'
	elif(o[6] == 1):
		print 'Quercus_Palustris'
	elif(o[7] == 1):
		print 'Tilia_Cordata'