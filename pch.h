#pragma once

#define _ENABLE_EXTENDED_ALIGNED_STORAGE

#pragma inline_depth(255)
#pragma managed(push, off)

#include <deque>

#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_invoke.h>

#include <boost/align/aligned_allocator.hpp>
#include <boost/mpi.hpp>
