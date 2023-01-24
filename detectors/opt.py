import numpy as np
from scipy.spatial import distance

class OPT():
	''' The base class of OPT families '''
	def __init__(self, default_pool_size=50):
		self._default_pool_size = default_pool_size

	# Virtual methods
	# type <- get_opt_type(self)
	# display_hash_func_parameters(self)
	# x' <- format_for_opt(self, x)
	# key <- get_hash_value(self, x, hash_index)


class E2LSH(OPT):
	''' Class to build E2 locality sensitive hashing family '''

	def __init__(self, bin_width=4, norm=2, default_pool_size=50):
		OPT.__init__(self, default_pool_size)
		self._dimensions = -1
		self._bin_width = bin_width
		self._norm = norm
		self.A_array = None
		self.B_array = None 


	def get_lsh_type(self):
		return 'L'+str(self._norm)+'LSH'


	def display_hash_func_parameters(self):
		for i in range(len(self.A_array)):
			print (self.A_array[i], self.B_array[i])

	def fit(self, data):
		if (data == None).all():
			return
		self._dimensions = len(data[0])-1

		self.A_array = []
		self.B_array = [] 
		if self._norm == 1:
			self.A_array.append(np.random.standard_cauchy(self._dimensions))
		elif self._norm == 2:
			self.A_array.append(np.random.normal(0.0, 1.0, self._dimensions))
		self.B_array.append(np.random.uniform(0.0, self._bin_width))
		for i in range(1, self._default_pool_size):
			repeated = True
			while repeated == True:
				repeated = False
				a=[]
				if self._norm == 1:
					a=np.random.standard_cauchy(self._dimensions)
				elif self._norm == 2:
					a=np.random.normal(0.0, 1.0, self._dimensions)
				b = np.random.uniform(0, self._bin_width)
				for j in range(0, len(self.A_array)):
					if np.array_equal(a, self.A_array[j]) and b == self.B_array[j]:
						repeated = True
						break
				if repeated == False:	
					self.A_array.append(a)
					self.B_array.append(b)	


	def format_for_lsh(self, x):
		return x


	def get_hash_value(self, x, hash_index):
		cur_len = len(self.A_array)
		while hash_index >= cur_len:
			repeated = True
			while repeated == True:
				repeated = False
				a=[]
				if self._norm == 1:
					a=np.random.standard_cauchy(self._dimensions)
				elif self._norm == 2:
					a=np.random.normal(0.0, 1.0, self._dimensions)
				b = np.random.uniform(0, self._bin_width)
				for j in range(0, cur_len):
					if np.array_equal(a, self.A_array[j]) and b == self.B_array[j]:
						repeated = True
						break
				if repeated == False:
					self.A_array.append(a)
					self.B_array.append(b)
					cur_len += 1
		return int(np.floor((np.dot(x, self.A_array[hash_index])+self.B_array[hash_index])/self._bin_width))
		

class AngleLSH(OPT):
	def __init__(self, default_pool_size=50):
		OPT.__init__(self, default_pool_size)
		self._weights = None

	def get_lsh_type(self):
		return 'AngleLSH'

	def display_hash_func_parameters(self):
		for i in range(len(self._weights)):
			print(self._weights[i])

	def fit(self, data):
		if data is None:
			return
		self._dimensions = len(data[0])-1

		self._weights=[]

		self._weights.append(np.random.normal(0.0, 1.0, self._dimensions))
		for i in range(1, self._default_pool_size):
			repeated = True
			while repeated == True:
				repeated = False
				weight=np.random.normal(0.0, 1.0, self._dimensions)
				for j in range(0, len(self._weights)):
					if np.array_equal(weight, self._weights[j]):
						repeated = True
						break
				if repeated == False:	
					self._weights.append(weight)


	def format_for_lsh(self, x):
		return x

	def get_hash_value(self, x, hash_index):
		cur_len = len(self._weights)
		while hash_index >= cur_len:
			repeated = True
			while repeated == True:
				repeated = False
				weight=np.random.normal(0.0, 1.0, self._dimensions)
				for j in range(0, cur_len):
					if np.array_equal(weight, self._weights[j]):
						repeated = True
						break
				if repeated == False:	
					self._weights.append(weight)
					cur_len += 1

		return -1 if np.dot(x, self._weights[hash_index]) <0 else 1


class HierHash():
	def __init__(self, number_of_bin=2, cal_distance=distance.euclidean):
		super().__init__()
		self._number_of_bin = number_of_bin
		self._distance = cal_distance

	def get_hash_type(self):
		return "Hier_Hash"

	def fit(self, centers, keys, sizes):
		self.centers = centers
		self.keys = keys
		self.sizes = sizes

	def get_hash_value(self, x):
		mindis = np.inf
		for i in range(len(self.centers)):
			newcenter = (self.centers[i]*self.sizes[i] + x) / (self.sizes[i]+1)
			dis = (self._distance(x, newcenter) + self._distance(self.centers[i], newcenter) * self.sizes[i]) ##/ (self.sizes[i]+1)

			if dis < mindis:
				mindis = dis
				key = self.keys[i]

		return key

	def __str__(self):
		return "("+str(self._number_of_bin)+", "+str(self.centers)+", "+str(self.keys)+")"