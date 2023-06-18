# memuat dataset
from sklearn.datasets import load_iris
iris = load_iris()

# pemisahan variabel target dan fitur
X = iris.data
y = iris.target

# pembagian data menjadi training set dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# pengkodean variabel kategorikal (jika ada)
# tidak diperlukan pada dataset iris karena semua variabel bersifat numerik

# membuat model Naïve Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# melatih model dengan data training
gnb.fit(X_train, y_train)

# melakukan prediksi terhadap data test set
y_pred = gnb.predict(X_test)

# evaluasi performa model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
