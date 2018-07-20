from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

classifier_one = RandomForestClassifier()
classifier_two = GaussianNB()
classifier_three = LogisticRegression()

dataset = load_iris()
train_features, test_features, train_labels, test_labels = train_test_split(dataset.data,
                                                                            dataset.target,
                                                                            test_size=0.3,
                                                                            random_state=42)

voting_classifier = VotingClassifier(estimators=[('RandomForestClassifier', classifier_one),
                                                 ('GaussianNB', classifier_two),
                                                 ('LogisticRegression', classifier_three)],
                                     voting='soft', # voting can be soft also
                                     weights=[1, 1, 1]) #  weightage to be given to each of the algorithm for voting

voting_classifier.fit(train_features, train_labels)
score = voting_classifier.score(test_features, test_labels)
print(voting_classifier.predict(test_features))
print("Score of the voting algorithm formed is", score*100)
