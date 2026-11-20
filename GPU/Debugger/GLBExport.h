#pragma once

#include <string>

class Path;

namespace GPURecord {
bool ExportFrameDumpToGLB(const Path &dumpPath, Path *outputPath, std::string *error);
}
