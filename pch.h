#pragma once

#define _ENABLE_EXTENDED_ALIGNED_STORAGE

#pragma inline_depth(255)
#pragma managed(push, off)

#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_invoke.h>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <boost/align/aligned_allocator.hpp>
#include <boost/mpi.hpp>