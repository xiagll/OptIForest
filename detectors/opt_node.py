class LSHNode:
	def __init__(self,center=[],data_size=0,children={},children_count={},hash_func_index=-1,lof=0):
		self._center = center
		self._data_size = data_size
		self._children = children
		self._children_count = children_count
		self._hash_func_index = hash_func_index
		self._lof = lof

	def display(self):
		print(self._hash_func_index)
		print(self._data_size)
		print(self._children)
		print(self._children_count)
		print(self._lof)

	def get_children(self):
		return self._children

	def get_data_size(self):
		return self._data_size

	def get_hash_func_index(self):
		return self._hash_func_index

	def get_children_count(self):
		return self._children_count

	def get_lof(self):
		return self._lof

	def get_center(self):
		return self._center


class OptNode:
	def __init__(self,center=[],data_size=0,hash_function=None,children={},children_count={},lof=1):
		self._data_size = data_size
		self._hash_function = hash_function
		self._children = children
		self._children_count = children_count
		self._center = center
		self._lof = lof

	def display(self):
		print(self)

	def __str__(self):
		return "("+str(self._data_size)+", "+str(self._hash_function)+", "+str(self._children_count)+")"


	def get_data_size(self):
		return self._data_size


	def get_hash_function(self):
		return self._hash_function


	def get_children(self):
		return self._children


	def get_children_count(self):
		return self._children_count

	def get_center(self):
		return self._center

	def get_lof(self):
		return self._lof
