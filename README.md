🍷 Wine Quality Prediction App
Ce projet est une application web Flask permettant de prédire la qualité d’un vin rouge à partir de ses propriétés physico-chimiques. Un modèle de régression entraîné sur le dataset winequality-red.csv est utilisé pour effectuer la prédiction.

📊 Aperçu du projet
L'utilisateur saisit les données suivantes via un formulaire :

Fixed Acidity

Volatile Acidity

Citric Acid

Residual Sugar

Chlorides

Free Sulfur Dioxide

Total Sulfur Dioxide

Density

pH

Sulphates

Alcohol

Le modèle prédit ensuite une qualité de vin (note sur 10) et affiche un graphe radar des caractéristiques saisies.

🛠️ Technologies utilisées
Python 3.x

Flask

Scikit-learn

Pandas / Numpy

Chart.js (pour les graphiques interactifs)

HTML/CSS (Jinja2 templating)
