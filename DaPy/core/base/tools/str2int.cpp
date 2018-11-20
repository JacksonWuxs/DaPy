// This code is reference from: 
// https: //stackoverflow.com/questions/1309123/fast-string-to-integer-conversion-in-python
// written by earl in Aug 21, 2009.

# include "Python.h"

static PyObject *str2int(PyObject *self, PyObject *args) {
    char *s; unsigned r = 0;
    if (!PyArg_ParseTuple(args, "s", &s)) return NULL;
    for (r = 0; *s; r = r * 10 + *s++ - '0');
    return Py_BuildValue("i", r);
}