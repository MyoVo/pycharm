import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Average
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV
warnings.filterwarnings("ignore")
import numpy as np

class DeepNeuralDecisionForest(BaseEstimator, ClassifierMixin):
    def __init__(self, num_trees=3, tree_depth=3, num_classes=2, learning_rate=0.001):
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        outputs = [self.build_tree(x) for _ in range(self.num_trees)]

        if len(outputs) > 1:
            output = Average()(outputs)
        else:
            output = outputs[0]

        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    def build_tree(self, inputs):
        x = inputs
        for _ in range(self.tree_depth):
            x = Dense(self.num_classes, activation='softmax', use_bias=False)(x)
        return x

    def fit(self, X_train, y_train, batch_size=32, epochs=10):
        self.build_model(X_train.shape[1:])
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        self.model.fit(X_train, y_train_cat, batch_size=batch_size, epochs=epochs, verbose=0)

    def predict(self, X):
        pred = self.model.predict(X)
        return pred.argmax(axis=1)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        if sample_weight is not None:
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            return accuracy_score(y, y_pred)

    def get_params(self, deep=False):
        return {'num_trees': self.num_trees, 'tree_depth': self.tree_depth,
                'num_classes': self.num_classes, 'learning_rate': self.learning_rate}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

def tune_dndf_model(X_train, y_train, X_test, y_test):
    param_grid = {
        'num_trees': Integer(1, 10),
        'tree_depth': Integer(1, 7),
        'learning_rate': Real(0.0001, 0.01, prior='log-uniform'),
        'batch_size': Categorical([16, 32, 64])
    }
    bayes_search = BayesSearchCV(estimator=DeepNeuralDecisionForest(num_classes=len(set(y_train))),
                                 search_spaces=param_grid, n_iter=30, cv=5, n_jobs=1, scoring='accuracy')
    bayes_search.fit(X_train, y_train)
    best_params = bayes_search.best_params_
    best_dndf_model = DeepNeuralDecisionForest(num_trees=best_params['num_trees'],
                                               tree_depth=best_params['tree_depth'],
                                               learning_rate=best_params['learning_rate'],
                                               num_classes=len(set(y_train)))
    best_dndf_model.build_model(X_train.shape[1:])
    best_dndf_model.fit(X_train, y_train, batch_size=int(best_params['batch_size']), epochs=1000)
    y_pred = best_dndf_model.predict(X_test)
    dndf_accuracy = accuracy_score(y_test, y_pred)
    print("Best Parameters:", best_params)
    #print("DNDT Accuracy:", dndf_accuracy)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # 绘制百分比混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    # 绘制热图并指定自定义标签
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Supine', 'Side', 'Foetus', 'Prone'],
                yticklabels=['Supine', 'Side', 'Foetus', 'Prone'])
    plt.title('Confusion Matrix (Bo-DNDT)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # 保存图形为SVG文件
    plt.savefig('DNDT.svg', format='svg', bbox_inches='tight')
    plt.show()
    return best_dndf_model, dndf_accuracy
