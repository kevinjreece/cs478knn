import numpy as np
import kjr_tools
import math

class NearestNeighbor:
	def __init__(self, k, regression=False):
		self.k = k
		self.regression = regression

	def learn(self, all_data):
		self.all_data = all_data
		self.possible_labels = list(set(self.all_data[:, -1]))
		print 'possible_labels: {}\n'.format(self.possible_labels)

	def predict(self, single_instance):
		k_nearest = []
		for data in self.all_data:
			if len(k_nearest) < self.k: # if we don't already have k neighbors
				k_nearest.append(data)
			else:
				dist = self.calcDist(single_instance, data)
				for nearest_point in k_nearest:
					if dist < self.calcDist(single_instance, nearest_point):
						kjr_tools.removeNPArrayFromList(k_nearest, nearest_point)
						k_nearest.append(data)
						break
		k_nearest = np.array(k_nearest)
		# print '\ninstance: {}\nk_nearest: {}\n'.format(single_instance, k_nearest)
		if self.regression:
			raise NotImplementedError()
		else:
			return max(self.possible_labels, key=list(k_nearest[:, -1]).count)

	def measureAccuracy(self, all_data):
		num_elements = len(all_data)
		count = 0

		if self.regression:
			return NotImplementedError()
		else:
			num_correct = 0
			for data in all_data:
				count += 1
				print "\r{0}/{1}\t{2:.2f}%".format(count, num_elements, count / float(num_elements) * 100),
				label = data[-1]
				prediction = self.predict(data)
				
				if prediction == label:
					num_correct += 1
			
			accuracy = float(num_correct) / num_elements # classification accuracy
			return accuracy

	def calcDist(self, pointA, pointB):
		return math.sqrt(sum([self.calcSquaredAttributeDist(a, b) for a, b in zip(pointA[:-1], pointB[:-1])]))

	def calcSquaredAttributeDist(self, a, b):
		try:
			float_a = float(a)
			float_b = float(b)
			return (float_a - float_b) ** 2
		except Exception:
			return 0 if a == b else 1

	def calcDistMatrix(self, train_data, test_data):
		dist_mat = []
		for test in test_data:
			row = []
			for train in train_data:
				row.append((self.calcDist(train, test), train[-1]))
			dist_mat.append(sorted(row))
		return dist_mat
