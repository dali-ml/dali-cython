default:
	python3 setup.py clean
	python3 setup.py build_ext --inplace

cuda:
	python3 setup.py clean
	python3 setup.py build_ext --inplace cuda

clean:
	python3 setup.py clean
