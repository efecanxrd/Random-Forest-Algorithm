from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# iris veri setini yükle
iris = load_iris()

# özellikleri ve hedef sınıfı ayır
X = iris.data
y = iris.target

# verileri eğitim ve test kümeleri olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Random Forest modelini oluştur
random_forest = RandomForestClassifier(n_estimators=100)

# Modeli eğit
random_forest.fit(X_train, y_train)

# Test seti üzerinde modelin doğruluğunu ölç
y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
