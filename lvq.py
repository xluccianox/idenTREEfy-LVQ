"""
Example of use LVQ network
==========================

"""
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

# Create train samples
input = np.array(input)
target = np.array(target)


# 
net = nl.net.newlvq(nl.tool.minmax(input), 50, [.125,.125,.125,.125,.125,.125,.125,.125])
# Train network
error = net.train(input, target)

i = [[5.388059701492537,0.31503342325288,1.5585913587009053,1.8382633010917735,0.9979837101840756],
	 [2.139130434782609,0.5808038009859917,1.6809768561157492,1.6715470747274044,0.9966067089124588],
	 [0.6124567474048442,0.6152522920151243,1.4249341894508127,1.8375135330171544,1.6309003448270327],
	 [0.9976190476190476,0.0668923642903687,2.70395267545039,4.167607506495124,1.0738159505234968],
	 [1.0182648401826484,0.6050438809231855,1.8147597636654156,1.6914304481372575,0.9972210981314964],
	 [0.9900662251655629,0.6089311730083087,1.7975653697233918,1.6941157323548481,1.0096236072980627],
	 [1.002247191011236,0.06594019800726614,2.844245086307583,4.092757670566289,1.0370482352372872],
	 [1.2478632478632479,0.5468422377075948,1.6229414745376236,1.8699762337712431,1.0078221415641342]
	 ]
o = net.sim(i)
print(o)