#ifndef __FA_API__
#define __FA_API__

#if defined(_WIN32)
#ifdef CXX_BUILD
#define EXPORT __declspec(dllexport) 
#else
#define EXPORT __declspec(dllimport) 
#endif
#else
#define EXPORT
#endif

#if defined(_WIN32)
EXPORT const char* test();
#else
EXPORT const char* test();
#endif
#endif