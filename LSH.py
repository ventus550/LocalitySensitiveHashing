import numpy as np
from dataclasses import dataclass
import json

@dataclass
class LSH:
	r: int
	H: np.ndarray | None = None
	buckets: dict | None = None

	def _transform(self, X):
		return (np.sign(X @ self.H.T) + 1) / 2
	
	def _group(self, Y, HX):
		unique_hashes = np.unique(HX, axis=0)
		groups = {}

		for hash in unique_hashes:
			mask = (HX == hash).all(axis=1)
			group = Y[mask]
			groups[tuple(hash)] = group
		return groups

	def fit(self, X, Y):
		features = X.shape[1]
		self.H = np.random.normal(size=(self.r, features))
		self.buckets = self._group(Y, self._transform(X))
		return self
	
	def save_json(self, path):
		with open(path, 'w') as file:
			json.dump({
				"buckets": {str(list(k)): list(v) for k, v in self.buckets.items()},
				"H": self.H.tolist(),
				"r": self.r
			}, file)

	def __getitem__(self, x):
		x = self._transform(np.expand_dims(x, 0))
		return self.buckets[tuple(x.squeeze())]