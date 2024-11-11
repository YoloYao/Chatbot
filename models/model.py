from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class Model:
    def init(self, vectorizer, tf_transformer):
        self.vectorizer = vectorizer if vectorizer is not None else CountVectorizer()
        self.tf_transformer = tf_transformer if tf_transformer is not None else TfidfTransformer()
        self.tf_data = None
        self.data = None
