# Premier projet 

Implémentation d'indexation d'images en utilisant un processus RMAC

## Prerequisites
- [Python][1] (3.6)

### Principe
Dans cette partie du projet nous cherchons à déterminer les images les plus similaire à une image en entrée. 

Pour cela nous utilions comme descripteur l'output de l'algorithme de RMAC. Pour chaque image nous sauvegardons cette output dans le dossier data/descriptors. Ensuite lorsque nous voulons comparer la similarité entre deux images nous utilisons la fonction cosine_similarity de la librairie sklearn. 
Nous effectuons cette étape entre la query et chaque image. 

#### R-MAC (Regional Maximum Activations of Convolutions)
R-MAC regroupe plusieurs régions d'image en un vecteur de caractéristiques compactes de longueur fixe et est donc robuste à l'échelle et à la translation. Cette représentation peut traiter des images haute résolution de différents formats et obtenir une précision concurrentielle.

Premièrement, nous devons charger un réseau pré-formé, nous chargerons un modèle VGG sans les couches supérieures (constituées de couches entièrement connectées).

Nous gelons chaque layer de notre réseau. 

Nous créons les regions comme décrit dans le sujet de tp, nous appliquons ensuite maxpooling dans toutes les régions correspondantes de chacune des 512 cartes de caractéristiques.

## Execution
Pour executer le programme on lance : 
python main.py

Si nous souhaitons modifier la query nous modifions l'indice utilisé dans la ligne 76 : 
query = descriptors.pop(4)

## References
- Frederic, P from Shallow to Deep Representation for multimedia database. Lectures \
2019.
- Adrian, R from pyimagesearch.com. Website 2014.
- Noa, G from Github. 2018