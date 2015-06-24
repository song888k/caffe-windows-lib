#ifndef _COMPAT_HPP_
#define _COMPAT_HPP_

#if defined (_MSC_VER)

#pragma warning(disable:4005) // Macro redefinition like GOOGLE_PREDICT_TRUE
#pragma warning(disable:4018) // Expression signed/unsigned mismatch
#pragma warning(disable:4244) // Conversion from '__int64' to 'int'
#pragma warning(disable:4251) // Class 'type1' needs to have dll-interface to be used by clients of class 'type2'
#pragma warning(disable:4267) // Conversion from 'size_t' to 'int'
#pragma warning(disable:4275) // Non-DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'
#pragma warning(disable:4661) // No suitable definition provided for explicit template instantiation request
#pragma warning(disable:4715) // Not all control paths return a value
#pragma warning(disable:4716) // For functions without a return value
#pragma warning(disable:4812) // Obsolete declaration style
#pragma warning(disable:4996) // For deprecated functions like open(), close()

#include <io.h>
#include <process.h>
#include <stdio.h>
#include <intrin.h>
#include <float.h>
#include <direct.h>

#define getpid _getpid
#define snprintf _snprintf
#define __builtin_popcount __popcnt
#define __builtin_popcountl __popcnt64

inline float round(float x) {return (int) (x + 0.5f);}
inline double round(double x) {return (int) (x + 0.5);}
inline int isnan(double x) {return _isnan(x);}
inline int isinf(float x) {return !_finite(x);}
inline int mkdir(const char *pathname, unsigned int mode) {return _mkdir(pathname);} 

#   if defined (CAFFE_DLL_EXPORTS)
#       define CAFFE_CLASS_EXPORTS __declspec(dllexport)
#       define CAFFE_FUNCTION_EXPORTS __declspec(dllexport)
#   else
#       define CAFFE_CLASS_EXPORTS __declspec(dllimport)
#       define CAFFE_FUNCTION_EXPORTS
#   endif

#else
#   define CAFFE_CLASS_EXPORTS
#   define CAFFE_FUNCTION_EXPORTS
#endif

#endif /* _COMPAT_HPP_ */