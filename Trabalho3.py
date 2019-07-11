import collections

from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

trainData = pd.read_csv('train.csv')
trainData.describe()

trainPriceRange = trainData["price_range"]
trainData = trainData.drop("price_range", axis=1)

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(style="whitegrid")

for columns in [['battery_power', 'px_height', 'px_width', 'ram'], ['clock_speed', 'm_dep'], ['fc', 'n_cores', 'pc', 'sc_h', 'sc_w', 'talk_time'], ['int_memory', 'mobile_wt']]:
  name_data = pd.DataFrame(data=[[column, value] for column in columns for value in trainData[column]], columns=['column', 'value'])
  sns.violinplot(x=name_data['column'], y=name_data['value'])
  plt.show()

counterTable = list()
for column in ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']:
  counter = collections.Counter(trainData[column])
  counterTable.extend([[x, y, column] for x, y in counter.items()])
count_dp = pd.DataFrame(data=counterTable, columns=['value', 'count', 'column'])
ax = sns.barplot(x="value", y="count", hue="column", data=count_dp)
plt.show()

clf = tree.DecisionTreeClassifier(random_state=0)
cross_return_tree = cross_validate(clf, trainData, trainPriceRange, cv=10, return_estimator=True)

print(cross_return_tree["test_score"])
print(cross_return_tree["test_score"].mean())

min_test = min(cross_return_tree["test_score"])
max_test = max(cross_return_tree["test_score"])
max_index = [i for i, j in enumerate(cross_return_tree["test_score"]) if j == max_test]
best_tree = cross_return_tree["estimator"][max_index[0]]
print(best_tree.feature_importances_)

height = best_tree.feature_importances_
bars = trainData.columns
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars, rotation = 90)
plt.show()

scaler = MinMaxScaler()
scaler.fit(trainData)
trainDataScaled = scaler.transform(trainData)
trainDataScaledDF = pd.DataFrame(data = trainDataScaled, columns = trainData.columns)

neigh = KNeighborsClassifier(n_neighbors=5)
cross_return_knn = cross_validate(neigh, trainData, trainPriceRange, cv=10, return_estimator=True)
cross_return_knn_scaled = cross_validate(neigh, trainDataScaledDF, trainPriceRange, cv=10, return_estimator=True)

print(cross_return_knn["test_score"])
print(cross_return_knn["test_score"].mean())
print(cross_return_knn_scaled["test_score"])
print(cross_return_knn_scaled["test_score"].mean())

clf_mlp = MLPClassifier(hidden_layer_sizes=(10,5), max_iter = 300, learning_rate_init=0.01, random_state=0)
cross_return_mlp_scaled = cross_validate(clf_mlp, trainDataScaledDF, trainPriceRange, cv=10, return_estimator=True)
min_test = min(cross_return_mlp_scaled["test_score"])
max_test = max(cross_return_mlp_scaled["test_score"])
max_index = [i for i, j in enumerate(cross_return_mlp_scaled["test_score"]) if j == max_test]
best_mlp_scaled = cross_return_mlp_scaled["estimator"][max_index[0]]

print(sorted(cross_return_mlp_scaled["test_score"]))
print(cross_return_mlp_scaled["test_score"].mean())

classifier_list = 10 * ["dt"] + 10 * ["knn"] + 10 * ["knn_s"] + 10 * ["mlp"]
score_list = cross_return_tree["test_score"].tolist() + cross_return_knn["test_score"].tolist() +\
  cross_return_knn_scaled["test_score"].tolist() + cross_return_mlp_scaled["test_score"].tolist()
scores = pd.DataFrame([pd.Series(classifier_list), pd.Series(score_list)], index = ["classifier", "score"]).transpose()

ax = sns.violinplot(x=classifier_list, y=score_list)
plt.show()

