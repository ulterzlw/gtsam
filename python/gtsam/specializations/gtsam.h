/* Please refer to:
 * https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
 * These are required to save one copy operation on Python calls.
 *
 * NOTES
 * =================
 *
 * `py::bind_vector` and similar machinery gives the std container a Python-like
 * interface, but without the `<pybind11/stl.h>` copying mechanism. Combined
 * with `PYBIND11_MAKE_OPAQUE` this allows the types to be modified with Python,
 * and saves one copy operation.
 */

#ifdef GTSAM_ALLOCATOR_TBB
py::bind_vector<std::vector<gtsam::Key, tbb::tbb_allocator<gtsam::Key> > >(m_, "KeyVector");
py::implicitly_convertible<py::list, std::vector<gtsam::Key, tbb::tbb_allocator<gtsam::Key> > >();
#else
py::bind_vector<std::vector<gtsam::Key> >(m_, "KeyVector");
py::implicitly_convertible<py::list, std::vector<gtsam::Key> >();
#endif