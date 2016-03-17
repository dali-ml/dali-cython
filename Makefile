default:
	python3 setup.py clean
	python3 setup.py build_ext --inplace

cuda:
	python3 setup.py clean cuda
	python3 setup.py build_ext --inplace cuda

test:
	nose2

clean:
	python3 setup.py clean
