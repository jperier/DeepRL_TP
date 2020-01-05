_Thiboud Pierre-Elliott_  
_Perier-Camby Julien_

# DeepRL_TP

## Installation de l'environnement

Ce projet a été réalisé avec un environnement virtuel d'Anaconda 3, 
nous allons donc utiliser tant le gestionnaire de paquet `conda` 
que `pip`.

Etant donné que `gym` n'est pas totalement compatible avec Windows, 
la procédure d'installation pour faire fonctionner les 
environnements Atari diffère quelque peu avec cet OS.

### Linux

1. Créer le *virtual env* avec Anaconda : 
`conda create -n your-env-name python=3.7 pip`
2. "Rentrer" dans le *virtual env* : `activate your-env-name`
3. Ajouter le package `git` à Anaconda (si vous ne l'avez pas déjà) : `conda install git`
4. Cloner ce projet : `git clone https://github.com/jperier/DeepRL_TP.git deeprl_tp`
5. Se déplacer dans le dossier : `cd deeprl_tp`
6. Installer les dépendances : `pip install -r requirements.txt`
7. Installer PyTorch :
   - Si vous souhaitez utiliser cuda : `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
   - Sinon : `conda install pytorch torchvision cpuonly -c pytorch`

### Windows

La procédure d'installation pour Windows est identique à celle de Linux 
jusqu'à l'étape 5.

1. Installer les dépendances : `pip install -r requirements-windows.txt`
2. Vous aurez besoin des outils de build de Visual Studio (Microsoft Visual
 C++ Build Tools for Visual Studio 2019), [disponibles ici](https://visualstudio.microsoft.com/downloads/).
3. Pour installer les environnements Atari de `gym` : `pip install git+https://github.com/Kojoley/atari-py.git`
4. Installer PyTorch :
   - Si vous souhaitez utiliser cuda : `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
   - Sinon : `conda install pytorch torchvision cpuonly -c pytorch`
   
### Remarques

Même si vous n'installez pas les packages nécessaires pour faire fonctionner 
cuda, le code disponible dans ce répertoire fonctionnera quand même (il sera 
juste beaucoup plus long à s'exécuter, d'où l'intérêt d'avoir une bonne carte 
graphique :wink: )
