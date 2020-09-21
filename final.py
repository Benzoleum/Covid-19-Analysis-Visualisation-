import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('covid_all.csv')
total_cases = df[(df.location == 'Russia')].total_cases.astype(int).reset_index(name='Total Cases').to_numpy()
labels = df[(df.location == 'Russia')].label.to_numpy()
labels_uk = df[(df.location == 'United Kingdom')].label.to_numpy()
labels_mexico = df[(df.location == 'Mexico')].label.to_numpy()
total_cases_uk = df[(df.location == 'United Kingdom')].total_cases.astype(int).reset_index(
    name='Total Cases').to_numpy()
total_cases_peru = df[(df.location == 'Peru')].total_cases.astype(int).reset_index(name='Total Cases').to_numpy()
total_cases_mexico = df[(df.location == 'Mexico')].total_cases.astype(int).reset_index(name='Total Cases').to_numpy()
new_cases = df[(df.location == 'Russia')].new_cases.astype(int).to_numpy()
new_cases_uk = df[(df.location == 'United Kingdom')].new_cases.astype(int).to_numpy()
new_cases_peru = df[(df.location == 'Peru')].new_cases.astype(int).to_numpy()
new_cases_mexico = df[(df.location == 'Mexico')].new_cases.astype(int).to_numpy()
deaths = df[(df.location == 'Russia')].total_deaths.astype(int).to_numpy()
new_deaths = df[(df.location == 'Russia')].new_deaths.astype(int).to_numpy()
new_deaths_uk = df[(df.location == 'United Kingdom')].new_deaths.astype(int).to_numpy()
new_deaths_mexico = df[(df.location == 'Mexico')].new_deaths.astype(int).to_numpy()
deaths_uk = df[(df.location == 'United Kingdom')].total_deaths.astype(int).to_numpy()
deaths_peru = df[(df.location == 'Peru')].total_deaths.astype(int).to_numpy()
deaths_mexico = df[(df.location == 'Mexico')].total_deaths.astype(int).to_numpy()
gg = df.groupby(['location', 'week']).new_deaths.apply(np.array)
gz = df.groupby(['location', 'week']).label.apply(np.array)

average_new_deaths = [np.mean(x) for x in gg]
std_new_deaths = [np.std(x) for x in gg]

average_mex = average_new_deaths[:32]
average = average_new_deaths[32:64]
average_uk = average_new_deaths[64:]

std_mex = std_new_deaths[:32]
std = std_new_deaths[32:64]
std_uk = std_new_deaths[64:]

colors = []
for i in np.array(gz):
    colors.append(i[0].strip())

colors_mex = colors[:32]
colors_ru = colors[32:64]
colors_uk = colors[64:]

print('\nThere are ' + str(total_cases.max()) + ' cases of COVID-19 in Russia as of now.')
print('\nThere have been an average of ' + str(round(new_cases.mean(), 2)) + ' new daily cases.')
print('\nWith just ' + str(
    deaths.max()) + ' deaths, the goal of my project is to analyze if these statistics can be trusted.')
print(
    '\nThe reason I am very skeptical about the statistics that the Russian government is releasing is not just '
    'the fact that Russia was always very secretive and secluded, but also if we take a look at the official '
    'numbers of other countries, the difference between total cases and deaths is just stunning.')
print('\nFor instance, with almost 3 times less total cases than in Russia (' + str(
    total_cases_uk.max()) + '), there are ' + str(deaths_uk.max()) + ' deaths in the UK.')
print('\nWith ' + str(total_cases_peru.max()) + ' in Peru, there are ' + str(deaths_peru.max()) + ' deaths.')
print('\nAnd with ' + str(total_cases_mexico.max()) + ' in Mexico, there are ' + str(
    deaths_mexico.max()) + ' deaths cases.')
print(
    '\nI do realize that the infectious situations are different in these countries, with different healthcare and climates,'
    'however it still looks very suspicious that Russia has only ' + str(
        round(deaths.max() / total_cases.max(), 2)) + ' death rate.')

print('\nA pictorial representation might help truly see the immense difference in the figures.')

print(
    '\nAs the graph (Fig. 1) clearly shows, UK seems to be flattening the curve and stabilizing the situation despite the explosion of deaths in the beginning, Mexico unfortunately is experiencing an extreme increase in deaths, '
    'it\'s a little hard to read the Peru graph, probably due to the lack of information, but it seems there\'s a constant increase in deaths daily, and Russia is the only country on the graph which seems to handle the situation very well, with a very gradual and slow increase.')

print(
    '\nBut that does not prove anything, one might argue that the Russians were very careful and handled '
    'the situation better than any other country. Let\'s investigate further.')

X = list(zip(average_mex, std_mex))
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
le = LabelEncoder()
Y = colors_mex
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=3)

error_rate = []
for k in range(1, 17, 2):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, Y_train)
    pred_k = knn_classifier.predict(X_test)
    error_rate.append(np.mean(pred_k != Y_test))

knn_classifier = KNeighborsClassifier(n_neighbors=1)  # using the best k computed above
knn_classifier.fit(X, Y)

NB_classifier = GaussianNB().fit(X, Y)

new_instance = np.asmatrix(list(zip(average, std)))
new_instance_scaled = scaler.transform(new_instance)

prediction_mex = knn_classifier.predict(new_instance_scaled)
prediction_nb_mexico = NB_classifier.predict(new_instance)

score_knn_mex = accuracy_score(colors_ru, prediction_mex)
score_nb_mex = accuracy_score(colors_ru, prediction_nb_mexico)

unique_mex, counts_mex = np.unique(prediction_mex, return_counts=True)  # count the number of red and green labels
unique_mex_nb, counts_mex_nb = np.unique(prediction_nb_mexico,
                                         return_counts=True)  # count the number of red and green labels
unique, counts = np.unique(colors_ru, return_counts=True)

ru_num = dict(zip(unique, counts))
mex_num_knn = dict(zip(unique_mex, counts_mex))
mex_num_nb = dict(zip(unique_mex_nb, counts_mex_nb))

y_true = colors_ru
y_pred_knn_mex = prediction_mex
y_pred_nb_mex = prediction_nb_mexico
cf_matrix_knn_mex = confusion_matrix(y_true, y_pred_knn_mex)
cf_matrix_nb_mex = confusion_matrix(y_true, y_pred_nb_mex)

tpr_y_true = [1 if i == 'green' else 0 for i in y_true]
tpr_y_pred_knn_mex = [1 if i == 'green' else 0 for i in y_pred_knn_mex]

mcm_knn_mex = multilabel_confusion_matrix(y_true, y_pred_knn_mex)
tn_knn_mex = mcm_knn_mex[:, 0, 0]
tp_knn_mex = mcm_knn_mex[:, 1, 1]
fn_knn_mex = mcm_knn_mex[:, 1, 0]
fp_knn_mex = mcm_knn_mex[:, 0, 1]

tpr_knn_mex = recall_score(tpr_y_true, tpr_y_pred_knn_mex)
tnr_knn_mex = tn_knn_mex / (tn_knn_mex + fp_knn_mex)

tpr_y_pred_nb_mex = [1 if i == 'green' else 0 for i in y_pred_nb_mex]

mcm_nb_mex = multilabel_confusion_matrix(y_true, y_pred_nb_mex)
tn_nb_mex = mcm_nb_mex[:, 0, 0]
tp_nb_mex = mcm_nb_mex[:, 1, 1]
fn_nb_mex = mcm_nb_mex[:, 1, 0]
fp_nb_mex = mcm_nb_mex[:, 0, 1]

tpr_nb_mex = recall_score(tpr_y_true, tpr_y_pred_nb_mex)
tnr_nb_mex = tn_nb_mex / (tn_nb_mex + fp_nb_mex)


print(
    '\nIn order to get a better understanding, I decided to implement the KNN and Gaussian Naive Bayesian. '
    'I assigned labels to the weeks just as we did with the HW assignment, green being the week '
    'with less new deaths than the previous week, and red being more new deaths than the previous week. '
    'I decided to train the algorithm on Mexico and compare how would Russian statistic look like '
    'under similar circumstances. Using 1 neighbor, I got ' + str(
        round(score_knn_mex,
              2)) + ' accuracy for KNN, and ' + str(round(score_nb_mex, 2)) + ' score for naive bayesian.')

print('\nPrediction of Red and Green weeks in Russia using KNN Mexican model: ' + str(mex_num_knn))
print('\nPrediction of Red and Green weeks in Russia using Naive Bayesian Mexican model: ' + str(mex_num_nb))
print('\nActual Red and Green weeks in Russia: ' + str(ru_num))

print('\nAs you can see from the above, there should be at least ' + str(
    mex_num_knn['red'] - ru_num['red']) + ' more weeks in KNN model, and at least ' + str(
    mex_num_nb['red'] - ru_num['red']) + ' more weeks, using the Naive Bayesian model.')

print('\nThe confusion matrix for the kNN Mexico:')
print(cf_matrix_knn_mex)
print('\nThe TPR for the kNN Mexico:')
print(tpr_knn_mex)
print('\nThe TNR for the kNN Mexico:')
print(tnr_knn_mex)

print('\nThe confusion matrix for the Naive Bayesian Mexico:')
print(cf_matrix_nb_mex)
print('\nThe TPR for the Naive Bayesian Mexico:')
print(tpr_nb_mex)
print('\nThe TNR for the Naive Bayesian Mexico:')
print(tnr_nb_mex)


X = list(zip(average_uk, std_uk))
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
le = LabelEncoder()
Y = colors_uk

knn_classifier = KNeighborsClassifier(n_neighbors=1)  # using the best k computed above
knn_classifier.fit(X, Y)

NB_classifier = GaussianNB().fit(X, Y)

new_instance = np.asmatrix(list(zip(average, std)))
new_instance_scaled = scaler.transform(new_instance)

prediction_uk = knn_classifier.predict(new_instance_scaled)
prediction_nb_uk = NB_classifier.predict(new_instance)

score_knn_uk = accuracy_score(colors_ru, prediction_uk)
score_nb_uk = accuracy_score(colors_ru, prediction_nb_uk)

unique_uk, counts_uk = np.unique(prediction_uk, return_counts=True)  # count the number of red and green labels
unique_uk_nb, counts_uk_nb = np.unique(prediction_nb_uk,
                                       return_counts=True)  # count the number of red and green labels
unique, counts = np.unique(colors_ru, return_counts=True)

ru_num = dict(zip(unique, counts))
uk_num_knn = dict(zip(unique_uk, counts_uk))
uk_num_nb = dict(zip(unique_uk_nb, counts_uk_nb))

y_pred_knn_uk = prediction_uk
y_pred_nb_uk = prediction_nb_uk
cf_matrix_knn_uk = confusion_matrix(y_true, y_pred_knn_uk)
cf_matrix_nb_uk = confusion_matrix(y_true, y_pred_nb_uk)

tpr_y_true = [1 if i == 'green' else 0 for i in y_true]
tpr_y_pred_knn_uk = [1 if i == 'green' else 0 for i in y_pred_knn_uk]

mcm_knn_uk = multilabel_confusion_matrix(y_true, y_pred_knn_mex)
tn_knn_uk = mcm_knn_uk[:, 0, 0]
tp_knn_uk = mcm_knn_uk[:, 1, 1]
fn_knn_uk = mcm_knn_uk[:, 1, 0]
fp_knn_uk = mcm_knn_uk[:, 0, 1]

tpr_knn_uk = recall_score(tpr_y_true, tpr_y_pred_knn_uk)
tnr_knn_uk = tn_knn_uk / (tn_knn_uk + fp_knn_uk)

tpr_y_pred_nb_uk = [1 if i == 'green' else 0 for i in y_pred_nb_uk]

mcm_nb_uk = multilabel_confusion_matrix(y_true, y_pred_nb_uk)
tn_nb_uk = mcm_nb_uk[:, 0, 0]
tp_nb_uk = mcm_nb_uk[:, 1, 1]
fn_nb_uk = mcm_nb_uk[:, 1, 0]
fp_nb_uk = mcm_nb_uk[:, 0, 1]

tpr_nb_uk = recall_score(tpr_y_true, tpr_y_pred_nb_uk)
tnr_nb_uk = tn_nb_uk / (tn_nb_uk + fp_nb_uk)


print('\nIf we follow the same logic using the UK model, the accuracy for KNN is a little worse'
      ', with a score of ' + str(round(score_knn_uk, 2)) +
      ', and ' + str(round(score_nb_uk, 2)) + ' for the Naive Bayesian.')

print('\nPrediction of Red and Green weeks in Russia using KNN UK model: ' + str(uk_num_knn))
print('\nPrediction of Red and Green weeks in Russia using Naive Bayesian UK model: ' + str(uk_num_nb))
print('\nActual Red and Green weeks in Russia: ' + str(ru_num))

print('\nInterestingly, there are more green weeks using the KNN UK model, with ' + str(
    ru_num['red'] - uk_num_knn['red']) + ' more green weeks, however, there are still at least ' + str(
    uk_num_nb['red'] - ru_num['red']) + ' more red weeks, using the Naive Bayesian model.')

print(
    '\nAs we can see from the prediction, following the scenarios of Mexico and the UK, there should be about 5 weeks of '
    'an increase in new deaths, than it is now in Russia. ')

print('\nThe confusion matrix for the kNN UK:')
print(cf_matrix_knn_uk)
print('\nThe TPR for the kNN UK:')
print(tpr_knn_uk)
print('\nThe TNR for the kNN UK:')
print(tnr_knn_uk)

print('\nThe confusion matrix for the Naive Bayesian UK:')
print(cf_matrix_nb_uk)
print('\nThe TPR for the Naive Bayesian UK:')
print(tpr_nb_uk)
print('\nThe TNR for the Naive Bayesian UK:')
print(tnr_nb_uk)

print(
    '\nI am not an expert in virology and data science, unfortunately, but even to me '
    'the Figure 3 looks very suspicious with the Russian average number of deaths line being too gradual '
    'to be true, while the UK and Mexico both look very steep.')

fig, axs = plt.subplots(2, 2)
fig.suptitle('Deaths in Russia vs. Others')
axs[0, 0].set_title('# of deaths overtime (Fig. 1)')
axs[0, 0].plot(deaths, color='black', label='Russia')
axs[0, 0].plot(deaths_uk, color='red', label='UK')
axs[0, 0].plot(deaths_mexico, color='orange', label='Mexico')
axs[0, 0].plot(deaths_peru, color='magenta', label='Peru')
axs[0, 0].legend()
axs[0, 1].set_title('Error Rate vs. k for New_Deaths Labels (Fig. 2)')
axs[0, 1].plot(range(1, 17, 2), error_rate, color='red', linestyle='dashed', marker='o', markerfacecolor='black',
               markersize=10)
axs[0, 1].set_ylabel('Error rate')
axs[0, 1].set_xlabel('number of neighbors: k')
axs[1, 0].set_title('Average # of deaths per week (Fig. 3)')
axs[1, 0].set_xlabel('Days')
axs[1, 0].set_ylabel('Average # of deaths per week')
axs[1, 0].plot(average, color='black', label='Russia')
axs[1, 0].plot(average_uk, color='red', label='UK')
axs[1, 0].plot(average_mex, color='orange', label='Mexico')
axs[1, 0].legend()
# plt.show()

print(
    '\nIn the conclusion I wish to say that, I did not choose this project because I just want to accuse Russia of anything, '
    'i\'s been bothering me from the very beginning of that entire situation, when almost every single country on the planet is'
    'suffering the virus and doing their best to fight it and find the solution by providing accurate data and trying to cooperate '
    'with other countries, Russia is trying to show everyone else that they have it under control, with a minuscule amount of deaths.'
    ' If it continues to issue false statistics, not only it will hurt the country in the long term, other countries will be'
    'affected ultimately because of the lack of true data.')
