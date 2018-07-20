from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

dataset = load_iris()
train_features, test_features, train_labels, test_labels = train_test_split(dataset.data,
                                                                            dataset.target,
                                                                            test_size=0.5,
                                                                            random_state=42)
def classifier(classifier_type, classifier_name):
    classifier_type.fit(train_features, train_labels)
    score = classifier_type.score(test_features, test_labels)
    prediction = classifier_type.predict(test_features)
    print("Classifier used", classifier_name, "with accuracy", score*100)

classifier(DecisionTreeClassifier(), "DecisionTreeClassifier")
classifier(RandomForestClassifier(), "RandomForestClassifier")
classifier(AdaBoostClassifier(), "AdaBoostClassifier")
classifier(BaggingClassifier(), "BaggingClassifier")
classifier(ExtraTreesClassifier(), "ExtraTreesClassifier")
classifier(GradientBoostingClassifier(), "GradientBoostingClassifier")
classifier(ExtraTreesClassifier(), "ExtraTreesClassifier")
# classifier(VotingClassifier())