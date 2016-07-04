#include "python_tape.h"

#include <memory>
#include <stdexcept>
#include "dali/tensor/Tape.h"

class PyObjectHolder {
    public:
        PyObject * pyobject;
        PyObjectHolder(PyObject* _pyobject) : pyobject(_pyobject) {
            Py_XINCREF(_pyobject);
        }
        ~PyObjectHolder() {
            Py_XDECREF(pyobject);
        }
};

void emplace_back(PyObject * callback) {
    auto callback_holder = std::make_shared<PyObjectHolder>(callback);

    graph::emplace_back([callback_holder]() {
        PyObject *arglist;
        PyObject *result;
        arglist = Py_BuildValue("()");
        result = PyEval_CallObject(callback_holder->pyobject, arglist);
        Py_DECREF(arglist);
        if (result == NULL) {
            throw std::runtime_error("Error in callback");
        }
        Py_DECREF(result);
    });
}
