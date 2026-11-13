// Copyright (c) 2026 PPSSPP GLB Manager contributors.
// This file is part of PPSSPP and is distributed under the GPL-2.0-or-later.

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <zstd.h>

#include "Common/File/DirListing.h"
#include "Common/File/FileUtil.h"
#include "Common/File/Path.h"
#include "Core/ELF/ParamSFO.h"
#include "Core/Util/PathUtil.h"
#include "GPU/Debugger/GLBExport.h"
#include "GPU/Debugger/RecordFormat.h"

namespace GPURecord {
namespace {

constexpr uint32_t PPDMP_HEADER_SIZE = 24;
constexpr uint32_t MAX_DUMP_BYTES = 512 * 1024 * 1024;

struct ParsedCommand {
	uint8_t type = 0;
	uint32_t size = 0;
	uint32_t ptr = 0;
};

struct VertexLayout {
	int stride = 0;
	int weightOffset = -1;
	int uvOffset = -1;
	int colorOffset = -1;
	int normalOffset = -1;
	int positionOffset = -1;
	int weightFormat = 0;
	int weightCount = 0;
	int uvFormat = 0;
	int colorFormat = 0;
	int normalFormat = 0;
	int positionFormat = 0;
	int morphCount = 1;
};

static std::string JsonEscape(const std::string &value);

struct GPUStateSnapshot {
	uint32_t vertexType = 0;
	uint32_t texAddress[8]{};
	uint32_t texBufferWidth[8]{};
	float boneMatrices[96]{};
	bool boneValid[8]{};
	float worldMatrix[12]{};
	bool worldValid = false;
};

struct PendingDraw {
	std::vector<uint8_t> vertices;
	std::vector<uint8_t> indices;
	GPUStateSnapshot state;
	bool haveVertices = false;
};

struct AccessorInfo {
	uint32_t view = 0;
	uint32_t componentType = 5126;
	uint32_t count = 0;
	const char *type = "SCALAR";
	std::array<float, 3> min{};
	std::array<float, 3> max{};
	bool hasBounds = false;
};

struct BufferViewInfo {
	uint32_t offset = 0;
	uint32_t length = 0;
	uint32_t target = 0;
};

struct PrimitiveInfo {
	uint32_t positionAccessor = 0;
	uint32_t normalAccessor = 0;
	uint32_t uvAccessor = 0;
	uint32_t indexAccessor = 0;
	uint32_t material = 0;
};

struct ModelOutput {
	std::vector<uint8_t> binary;
	std::vector<BufferViewInfo> views;
	std::vector<AccessorInfo> accessors;
	std::vector<PrimitiveInfo> primitives;
	std::vector<std::string> imagesJson;
	std::vector<std::string> texturesJson;
	std::vector<std::string> materialsJson;
	std::map<std::string, uint32_t> materialByPath;
};

static uint32_t ReadU32(const uint8_t *p) {
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t ReadU16(const uint8_t *p) {
	return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int16_t ReadS16(const uint8_t *p) {
	return (int16_t)ReadU16(p);
}

static float ReadFloat(const uint8_t *p) {
	float value = 0.0f;
	std::memcpy(&value, p, sizeof(value));
	return value;
}

static float DecodeFloat24(uint32_t value) {
	uint32_t bits = (value & 0x00FFFFFF) << 8;
	float result = 0.0f;
	std::memcpy(&result, &bits, sizeof(result));
	return result;
}

static bool ReadFileBytes(const Path &path, std::vector<uint8_t> *bytes) {
	FILE *fp = File::OpenCFile(path, "rb");
	if (!fp)
		return false;

	if (std::fseek(fp, 0, SEEK_END) != 0) {
		std::fclose(fp);
		return false;
	}
	long length = std::ftell(fp);
	if (length < 0 || (uint64_t)length > MAX_DUMP_BYTES) {
		std::fclose(fp);
		return false;
	}
	if (std::fseek(fp, 0, SEEK_SET) != 0) {
		std::fclose(fp);
		return false;
	}

	bytes->resize((size_t)length);
	bool ok = length == 0 || std::fread(bytes->data(), 1, (size_t)length, fp) == (size_t)length;
	std::fclose(fp);
	return ok;
}

static bool ReadDump(const Path &path, std::vector<ParsedCommand> *commands, std::vector<uint8_t> *pushBuffer, std::string *error) {
	std::vector<uint8_t> file;
	if (!ReadFileBytes(path, &file)) {
		*error = "Could not read the PPDMP file.";
		return false;
	}
	if (file.size() < PPDMP_HEADER_SIZE + 8 || std::memcmp(file.data(), HEADER_MAGIC, 8) != 0) {
		*error = "This is not a valid PPSSPP frame dump.";
		return false;
	}

	uint32_t version = ReadU32(file.data() + 8);
	if (version < 5 || version > VERSION) {
		*error = "Only PPDMP versions 5 and 6 are supported by GLB Manager.";
		return false;
	}

	uint32_t commandCount = ReadU32(file.data() + PPDMP_HEADER_SIZE);
	uint32_t pushSize = ReadU32(file.data() + PPDMP_HEADER_SIZE + 4);
	if (commandCount == 0 || commandCount > 2000000 || pushSize == 0 || pushSize > MAX_DUMP_BYTES) {
		*error = "The frame dump has invalid command or buffer sizes.";
		return false;
	}

	size_t offset = PPDMP_HEADER_SIZE + 8;
	if (offset + 4 > file.size()) {
		*error = "The frame dump is truncated.";
		return false;
	}
	uint32_t compressedCommandsSize = ReadU32(file.data() + offset);
	offset += 4;
	if (compressedCommandsSize > file.size() - offset) {
		*error = "The frame dump command block is truncated.";
		return false;
	}

	std::vector<uint8_t> commandBytes((size_t)commandCount * 9);
	size_t decodedCommandsSize = ZSTD_decompress(commandBytes.data(), commandBytes.size(),
		file.data() + offset, compressedCommandsSize);
	if (ZSTD_isError(decodedCommandsSize) || decodedCommandsSize != commandBytes.size()) {
		*error = "Could not decompress the PPDMP command block.";
		return false;
	}
	offset += compressedCommandsSize;

	if (offset + 4 > file.size()) {
		*error = "The frame dump buffer block is missing.";
		return false;
	}
	uint32_t compressedBufferSize = ReadU32(file.data() + offset);
	offset += 4;
	if (compressedBufferSize > file.size() - offset) {
		*error = "The frame dump buffer block is truncated.";
		return false;
	}

	pushBuffer->resize(pushSize);
	size_t decodedBufferSize = ZSTD_decompress(pushBuffer->data(), pushBuffer->size(),
		file.data() + offset, compressedBufferSize);
	if (ZSTD_isError(decodedBufferSize) || decodedBufferSize != pushBuffer->size()) {
		*error = "Could not decompress the PPDMP graphics buffer.";
		return false;
	}

	commands->resize(commandCount);
	for (uint32_t i = 0; i < commandCount; ++i) {
		const uint8_t *p = commandBytes.data() + (size_t)i * 9;
		(*commands)[i].type = p[0];
		(*commands)[i].size = ReadU32(p + 1);
		(*commands)[i].ptr = ReadU32(p + 5);
		const ParsedCommand &cmd = (*commands)[i];
		if (cmd.ptr > pushSize || cmd.size > pushSize - cmd.ptr) {
			*error = "The PPDMP contains an invalid graphics-data pointer.";
			return false;
		}
	}
	return true;
}

static std::string Lowercase(std::string value) {
	for (char &c : value)
		c = (char)std::tolower((unsigned char)c);
	return value;
}

static std::map<std::string, Path> FindSavedTextures(const std::string &gameID) {
	std::map<std::string, Path> result;
	const Path textureDirectory = GetSysDirectory(DIRECTORY_TEXTURES) / gameID / "new";
	std::vector<File::FileInfo> files;
	if (!File::GetFilesInDir(textureDirectory, &files, ".png"))
		return result;

	for (const File::FileInfo &file : files) {
		if (file.isDirectory || file.name.size() < 8)
			continue;
		const std::string prefix = Lowercase(file.name.substr(0, 8));
		if (prefix.find_first_not_of("0123456789abcdef") == std::string::npos && !result.count(prefix))
			result[prefix] = file.fullName;
	}
	return result;
}

static int AlignOffset(int value, int alignment) {
	if (alignment <= 1)
		return value;
	return (value + alignment - 1) & ~(alignment - 1);
}

static bool MakeVertexLayout(uint32_t vertexType, VertexLayout *layout) {
	layout->weightFormat = (vertexType >> 9) & 3;
	layout->weightCount = layout->weightFormat ? (((vertexType >> 14) & 7) + 1) : 0;
	layout->uvFormat = vertexType & 3;
	layout->colorFormat = (vertexType >> 2) & 7;
	layout->normalFormat = (vertexType >> 5) & 3;
	layout->positionFormat = (vertexType >> 7) & 3;
	layout->morphCount = ((vertexType >> 18) & 7) + 1;

	if (layout->morphCount != 1 || layout->positionFormat == 0 || layout->uvFormat == 0 || layout->normalFormat == 0)
		return false;

	int offset = 0;
	int maxAlignment = 1;
	auto addField = [&](int size, int alignment, int *fieldOffset) {
		if (size == 0) {
			*fieldOffset = -1;
			return;
		}
		offset = AlignOffset(offset, alignment);
		*fieldOffset = offset;
		offset += size;
		maxAlignment = std::max(maxAlignment, alignment);
	};

	const int weightSize = layout->weightFormat == 1 ? 1 : layout->weightFormat == 2 ? 2 : layout->weightFormat == 3 ? 4 : 0;
	const int weightAlignment = weightSize ? weightSize : 1;
	addField(weightSize * layout->weightCount, weightAlignment, &layout->weightOffset);

	const int uvSize = layout->uvFormat == 1 ? 2 : layout->uvFormat == 2 ? 4 : 8;
	const int uvAlignment = layout->uvFormat == 1 ? 1 : layout->uvFormat == 2 ? 2 : 4;
	addField(uvSize, uvAlignment, &layout->uvOffset);

	const int colorSizeTable[8] = { 0, 0, 0, 0, 2, 2, 2, 4 };
	const int colorAlignment = layout->colorFormat == 7 ? 4 : layout->colorFormat >= 4 ? 2 : 1;
	addField(colorSizeTable[layout->colorFormat], colorAlignment, &layout->colorOffset);

	const int normalSizeTable[4] = { 0, 3, 6, 12 };
	const int normalAlignment = layout->normalFormat == 1 ? 1 : layout->normalFormat == 2 ? 2 : 4;
	addField(normalSizeTable[layout->normalFormat], normalAlignment, &layout->normalOffset);

	const int positionSizeTable[4] = { 0, 3, 6, 12 };
	const int positionAlignment = layout->positionFormat == 1 ? 1 : layout->positionFormat == 2 ? 2 : 4;
	addField(positionSizeTable[layout->positionFormat], positionAlignment, &layout->positionOffset);

	layout->stride = AlignOffset(offset, maxAlignment);
	return layout->stride > 0;
}

static float DecodeUnsignedComponent(const uint8_t *p, int format) {
	switch (format) {
	case 1: return p[0] * (1.0f / 128.0f);
	case 2: return ReadU16(p) * (1.0f / 32768.0f);
	case 3: return ReadFloat(p);
	default: return 0.0f;
	}
}

static float DecodeSignedComponent(const uint8_t *p, int format) {
	switch (format) {
	case 1: return (int8_t)p[0] * (1.0f / 128.0f);
	case 2: return ReadS16(p) * (1.0f / 32768.0f);
	case 3: return ReadFloat(p);
	default: return 0.0f;
	}
}

static void TransformPoint(const float *matrix, const float in[3], float out[3]) {
	out[0] = matrix[0] * in[0] + matrix[1] * in[1] + matrix[2] * in[2] + matrix[3];
	out[1] = matrix[4] * in[0] + matrix[5] * in[1] + matrix[6] * in[2] + matrix[7];
	out[2] = matrix[8] * in[0] + matrix[9] * in[1] + matrix[10] * in[2] + matrix[11];
}

static void TransformDirection(const float *matrix, const float in[3], float out[3]) {
	out[0] = matrix[0] * in[0] + matrix[1] * in[1] + matrix[2] * in[2];
	out[1] = matrix[4] * in[0] + matrix[5] * in[1] + matrix[6] * in[2];
	out[2] = matrix[8] * in[0] + matrix[9] * in[1] + matrix[10] * in[2];
}

static void Normalize(float v[3]) {
	const float length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (length > 0.000001f) {
		v[0] /= length;
		v[1] /= length;
		v[2] /= length;
	}
}

static float IdentityValue(int index) {
	static const float identity[12] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	};
	return identity[index];
}

static uint32_t TextureAddress(const GPUStateSnapshot &state, int level) {
	return (state.texAddress[level] & 0x00FFFFF0) | ((state.texBufferWidth[level] << 8) & 0x0F000000);
}

static bool ReadIndex(const std::vector<uint8_t> &data, int format, uint32_t index, uint32_t *value) {
	if (format == 0) {
		*value = index;
		return true;
	}
	const int size = format == 1 ? 1 : format == 2 ? 2 : 4;
	if ((uint64_t)index * size + size > data.size())
		return false;
	const uint8_t *p = data.data() + (size_t)index * size;
	if (format == 1)
		*value = p[0];
	else if (format == 2)
		*value = ReadU16(p);
	else
		*value = ReadU32(p) & 0xFFFF;
	return true;
}

static uint32_t AppendBinary(std::vector<uint8_t> *binary, const void *data, size_t size) {
	while (binary->size() & 3)
		binary->push_back(0);
	const uint32_t offset = (uint32_t)binary->size();
	const uint8_t *bytes = (const uint8_t *)data;
	binary->insert(binary->end(), bytes, bytes + size);
	return offset;
}

static uint32_t AddView(ModelOutput *model, uint32_t offset, uint32_t length, uint32_t target) {
	BufferViewInfo view;
	view.offset = offset;
	view.length = length;
	view.target = target;
	model->views.push_back(view);
	return (uint32_t)model->views.size() - 1;
}

static uint32_t AddFloatAccessor(ModelOutput *model, const std::vector<float> &values, uint32_t components, const char *type, bool positionBounds) {
	const uint32_t offset = AppendBinary(&model->binary, values.data(), values.size() * sizeof(float));
	const uint32_t view = AddView(model, offset, (uint32_t)(values.size() * sizeof(float)), 34962);
	AccessorInfo accessor;
	accessor.view = view;
	accessor.componentType = 5126;
	accessor.count = (uint32_t)(values.size() / components);
	accessor.type = type;

	if (positionBounds && components == 3 && !values.empty()) {
		accessor.hasBounds = true;
		for (int j = 0; j < 3; ++j) {
			accessor.min[j] = values[j];
			accessor.max[j] = values[j];
		}
		for (size_t i = 0; i < values.size(); i += 3) {
			for (int j = 0; j < 3; ++j) {
				accessor.min[j] = std::min(accessor.min[j], values[i + j]);
				accessor.max[j] = std::max(accessor.max[j], values[i + j]);
			}
		}
	}
	model->accessors.push_back(accessor);
	return (uint32_t)model->accessors.size() - 1;
}

static uint32_t AddIndexAccessor(ModelOutput *model, const std::vector<uint32_t> &indices) {
	const uint32_t offset = AppendBinary(&model->binary, indices.data(), indices.size() * sizeof(uint32_t));
	const uint32_t view = AddView(model, offset, (uint32_t)(indices.size() * sizeof(uint32_t)), 34963);
	AccessorInfo accessor;
	accessor.view = view;
	accessor.componentType = 5125;
	accessor.count = (uint32_t)indices.size();
	accessor.type = "SCALAR";
	model->accessors.push_back(accessor);
	return (uint32_t)model->accessors.size() - 1;
}

static bool AddTextureMaterial(ModelOutput *model, const Path &texturePath, uint32_t *materialIndex) {
	const std::string key = texturePath.ToString();
	auto existing = model->materialByPath.find(key);
	if (existing != model->materialByPath.end()) {
		*materialIndex = existing->second;
		return true;
	}

	std::vector<uint8_t> png;
	if (!ReadFileBytes(texturePath, &png) || png.size() < 8 ||
		std::memcmp(png.data(), "\x89PNG\r\n\x1a\n", 8) != 0)
		return false;

	const uint32_t pngOffset = AppendBinary(&model->binary, png.data(), png.size());
	const uint32_t pngView = AddView(model, pngOffset, (uint32_t)png.size(), 0);
	const uint32_t imageIndex = (uint32_t)model->imagesJson.size();
	const uint32_t textureIndex = (uint32_t)model->texturesJson.size();
	const uint32_t material = (uint32_t)model->materialsJson.size();

	std::ostringstream image;
	image << "{\"bufferView\":" << pngView << ",\"mimeType\":\"image/png\",\"name\":\""
		<< JsonEscape(texturePath.GetFilename()) << "\"}";
	model->imagesJson.push_back(image.str());

	std::ostringstream texture;
	texture << "{\"source\":" << imageIndex << "}";
	model->texturesJson.push_back(texture.str());

	std::ostringstream mat;
	mat << "{\"name\":\"" << JsonEscape(texturePath.GetFilename())
		<< "\",\"doubleSided\":true,\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":"
		<< textureIndex << "},\"metallicFactor\":0,\"roughnessFactor\":1}}";
	model->materialsJson.push_back(mat.str());
	model->materialByPath[key] = material;
	*materialIndex = material;
	return true;
}

static std::string JsonEscape(const std::string &value) {
	std::string result;
	result.reserve(value.size() + 8);
	for (unsigned char c : value) {
		switch (c) {
		case '"': result += "\\\""; break;
		case '\\': result += "\\\\"; break;
		case '\b': result += "\\b"; break;
		case '\f': result += "\\f"; break;
		case '\n': result += "\\n"; break;
		case '\r': result += "\\r"; break;
		case '\t': result += "\\t"; break;
		default:
			if (c < 0x20) {
				char hex[7];
				std::snprintf(hex, sizeof(hex), "\\u%04x", c);
				result += hex;
			} else {
				result += (char)c;
			}
		}
	}
	return result;
}

static bool BuildPrimitive(const PendingDraw &pending, uint32_t primOp, const std::map<std::string, Path> &textures, ModelOutput *model) {
	const GPUStateSnapshot &state = pending.state;
	const uint32_t vertexType = state.vertexType;
	if ((vertexType & (1u << 23)) != 0)
		return false;  // Through-mode vertices are 2D/UI.
	VertexLayout layout;
	if (!MakeVertexLayout(vertexType, &layout))
		return false;

	const int weightType = layout.weightFormat;
	const int indexType = (vertexType >> 11) & 3;
	const uint32_t vertexCount = (uint32_t)(pending.vertices.size() / (size_t)layout.stride);
	if (vertexCount < 3)
		return false;

	uint32_t textureAddress = 0;
	Path texturePath;
	for (int level = 0; level < 8; ++level) {
		const uint32_t candidate = TextureAddress(state, level);
		char prefix[9];
		std::snprintf(prefix, sizeof(prefix), "%08x", candidate);
		auto it = textures.find(prefix);
		if (candidate != 0 && it != textures.end()) {
			textureAddress = candidate;
			texturePath = it->second;
			break;
		}
	}
	if (textureAddress == 0 || texturePath.empty())
		return false;

	const uint32_t primitiveType = (primOp >> 16) & 7;
	const uint32_t requestedCount = primOp & 0xFFFF;
	if (primitiveType != 3 && primitiveType != 4 && primitiveType != 5)
		return false;

	const uint32_t count = indexType == 0 ? std::min(requestedCount, vertexCount) :
		std::min(requestedCount, (uint32_t)(pending.indices.size() / (indexType == 1 ? 1 : indexType == 2 ? 2 : 4)));
	if (count < 3)
		return false;

	std::vector<float> positions;
	std::vector<float> normals;
	std::vector<float> uvs;
	positions.reserve((size_t)vertexCount * 3);
	normals.reserve((size_t)vertexCount * 3);
	uvs.reserve((size_t)vertexCount * 2);

	for (uint32_t i = 0; i < vertexCount; ++i) {
		const uint8_t *v = pending.vertices.data() + (size_t)i * layout.stride;
		float p[3] = { 0, 0, 0 };
		float n[3] = { 0, 0, 1 };
		float uv[2] = { 0, 0 };

		if (layout.uvFormat == 1) {
			uv[0] = DecodeUnsignedComponent(v + layout.uvOffset, 1);
			uv[1] = DecodeUnsignedComponent(v + layout.uvOffset + 1, 1);
		} else if (layout.uvFormat == 2) {
			uv[0] = DecodeUnsignedComponent(v + layout.uvOffset, 2);
			uv[1] = DecodeUnsignedComponent(v + layout.uvOffset + 2, 2);
		} else {
			uv[0] = DecodeUnsignedComponent(v + layout.uvOffset, 3);
			uv[1] = DecodeUnsignedComponent(v + layout.uvOffset + 4, 3);
		}

		if (layout.normalFormat == 1) {
			for (int j = 0; j < 3; ++j) n[j] = DecodeSignedComponent(v + layout.normalOffset + j, 1);
		} else if (layout.normalFormat == 2) {
			for (int j = 0; j < 3; ++j) n[j] = DecodeSignedComponent(v + layout.normalOffset + j * 2, 2);
		} else {
			for (int j = 0; j < 3; ++j) n[j] = DecodeSignedComponent(v + layout.normalOffset + j * 4, 3);
		}

		if (layout.positionFormat == 1) {
			for (int j = 0; j < 3; ++j) p[j] = DecodeSignedComponent(v + layout.positionOffset + j, 1);
		} else if (layout.positionFormat == 2) {
			for (int j = 0; j < 3; ++j) p[j] = DecodeSignedComponent(v + layout.positionOffset + j * 2, 2);
		} else {
			for (int j = 0; j < 3; ++j) p[j] = DecodeSignedComponent(v + layout.positionOffset + j * 4, 3);
		}

		float skinnedPosition[3] = { 0, 0, 0 };
		float skinnedNormal[3] = { 0, 0, 0 };
		if (weightType != 0) {
			float weights[8]{};
			for (int j = 0; j < layout.weightCount; ++j) {
				const uint8_t *w = v + layout.weightOffset + j * (weightType == 1 ? 1 : weightType == 2 ? 2 : 4);
				weights[j] = DecodeUnsignedComponent(w, weightType);
			}
			for (int bone = 0; bone < layout.weightCount; ++bone) {
				const float weight = weights[bone];
				if (std::fabs(weight) < 0.000001f)
					continue;
				float identity[12];
				const float *matrix = nullptr;
				if (state.boneValid[bone]) {
					matrix = &state.boneMatrices[bone * 12];
				} else {
					for (int k = 0; k < 12; ++k) identity[k] = IdentityValue(k);
					matrix = identity;
				}
				float transformedPosition[3], transformedNormal[3];
				TransformPoint(matrix, p, transformedPosition);
				TransformDirection(matrix, n, transformedNormal);
				for (int k = 0; k < 3; ++k) {
					skinnedPosition[k] += transformedPosition[k] * weight;
					skinnedNormal[k] += transformedNormal[k] * weight;
				}
			}
		} else {
			std::copy(p, p + 3, skinnedPosition);
			std::copy(n, n + 3, skinnedNormal);
		}

		float world[12];
		const float *worldMatrix = state.worldValid ? state.worldMatrix : nullptr;
		if (worldMatrix) {
			TransformPoint(worldMatrix, skinnedPosition, p);
			TransformDirection(worldMatrix, skinnedNormal, n);
		} else {
			std::copy(skinnedPosition, skinnedPosition + 3, p);
			std::copy(skinnedNormal, skinnedNormal + 3, n);
		}
		Normalize(n);
		if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) ||
			!std::isfinite(uv[0]) || !std::isfinite(uv[1]))
			return false;

		positions.insert(positions.end(), p, p + 3);
		normals.insert(normals.end(), n, n + 3);
		uvs.push_back(uv[0]);
		uvs.push_back(1.0f - uv[1]);
	}

	std::vector<uint32_t> sourceIndices;
	sourceIndices.reserve(count);
	for (uint32_t i = 0; i < count; ++i) {
		uint32_t index = 0;
		if (!ReadIndex(pending.indices, indexType, i, &index) || index >= vertexCount)
			return false;
		sourceIndices.push_back(index);
	}

	std::vector<uint32_t> triangleIndices;
	if (primitiveType == 3) {
		for (uint32_t i = 0; i + 2 < count; i += 3) {
			triangleIndices.push_back(sourceIndices[i]);
			triangleIndices.push_back(sourceIndices[i + 1]);
			triangleIndices.push_back(sourceIndices[i + 2]);
		}
	} else if (primitiveType == 4) {
		for (uint32_t i = 0; i + 2 < count; ++i) {
			if ((i & 1) == 0) {
				triangleIndices.push_back(sourceIndices[i]);
				triangleIndices.push_back(sourceIndices[i + 1]);
			} else {
				triangleIndices.push_back(sourceIndices[i + 1]);
				triangleIndices.push_back(sourceIndices[i]);
			}
			triangleIndices.push_back(sourceIndices[i + 2]);
		}
	} else {
		for (uint32_t i = 1; i + 1 < count; ++i) {
			triangleIndices.push_back(sourceIndices[0]);
			triangleIndices.push_back(sourceIndices[i]);
			triangleIndices.push_back(sourceIndices[i + 1]);
		}
	}
	if (triangleIndices.size() < 3)
		return false;

	uint32_t material = 0;
	if (!AddTextureMaterial(model, texturePath, &material))
		return false;

	PrimitiveInfo primitive;
	primitive.positionAccessor = AddFloatAccessor(model, positions, 3, "VEC3", true);
	primitive.normalAccessor = AddFloatAccessor(model, normals, 3, "VEC3", false);
	primitive.uvAccessor = AddFloatAccessor(model, uvs, 2, "VEC2", false);
	primitive.indexAccessor = AddIndexAccessor(model, triangleIndices);
	primitive.material = material;
	model->primitives.push_back(primitive);
	return true;
}

static void UpdateStateFromRegister(GPUStateSnapshot *state, uint32_t op, int *boneWord, int *worldWord) {
	const uint8_t command = (uint8_t)(op >> 24);
	const uint32_t data = op & 0x00FFFFFF;
	if (command == 0x12) {
		state->vertexType = data;
	} else if (command >= 0xA0 && command <= 0xA7) {
		state->texAddress[command - 0xA0] = data;
	} else if (command >= 0xA8 && command <= 0xAF) {
		state->texBufferWidth[command - 0xA8] = data;
	} else if (command == 0x2A) {
		*boneWord = data & 0x7F;
	} else if (command == 0x2B) {
		if (*boneWord >= 0 && *boneWord < 96) {
			state->boneMatrices[*boneWord] = DecodeFloat24(data);
			const int bone = *boneWord / 12;
			if ((*boneWord % 12) == 11)
				state->boneValid[bone] = true;
		}
		++*boneWord;
	} else if (command == 0x3A) {
		*worldWord = data & 0xF;
	} else if (command == 0x3B) {
		if (*worldWord >= 0 && *worldWord < 12) {
			state->worldMatrix[*worldWord] = DecodeFloat24(data);
			if (*worldWord == 11)
				state->worldValid = true;
		}
		++*worldWord;
	}
}

static std::string BuildJson(const ModelOutput &model) {
	std::ostringstream json;
	json << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"PPSSPP GLB Manager\"},";
	json << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"name\":\"Player\",\"mesh\":0}],";
	json << "\"meshes\":[{\"name\":\"Player\",\"primitives\":[";
	for (size_t i = 0; i < model.primitives.size(); ++i) {
		const PrimitiveInfo &p = model.primitives[i];
		if (i) json << ",";
		json << "{\"attributes\":{\"POSITION\":" << p.positionAccessor
			<< ",\"NORMAL\":" << p.normalAccessor
			<< ",\"TEXCOORD_0\":" << p.uvAccessor
			<< "},\"indices\":" << p.indexAccessor
			<< ",\"material\":" << p.material << ",\"mode\":4}";
	}
	json << "]}],";

	json << "\"materials\":[";
	for (size_t i = 0; i < model.materialsJson.size(); ++i) {
		if (i) json << ",";
		json << model.materialsJson[i];
	}
	json << "],\"textures\":[";
	for (size_t i = 0; i < model.texturesJson.size(); ++i) {
		if (i) json << ",";
		json << model.texturesJson[i];
	}
	json << "],\"images\":[";
	for (size_t i = 0; i < model.imagesJson.size(); ++i) {
		if (i) json << ",";
		json << model.imagesJson[i];
	}
	json << "],\"bufferViews\":[";
	for (size_t i = 0; i < model.views.size(); ++i) {
		const BufferViewInfo &v = model.views[i];
		if (i) json << ",";
		json << "{\"buffer\":0,\"byteOffset\":" << v.offset << ",\"byteLength\":" << v.length;
		if (v.target)
			json << ",\"target\":" << v.target;
		json << "}";
	}
	json << "],\"accessors\":[";
	for (size_t i = 0; i < model.accessors.size(); ++i) {
		const AccessorInfo &a = model.accessors[i];
		if (i) json << ",";
		json << "{\"bufferView\":" << a.view << ",\"componentType\":" << a.componentType
			<< ",\"count\":" << a.count << ",\"type\":\"" << a.type << "\"";
		if (a.hasBounds) {
			json << ",\"min\":[" << a.min[0] << "," << a.min[1] << "," << a.min[2] << "]";
			json << ",\"max\":[" << a.max[0] << "," << a.max[1] << "," << a.max[2] << "]";
		}
		json << "}";
	}
	json << "],\"buffers\":[{\"byteLength\":" << model.binary.size() << "}]}";
	return json.str();
}

static bool WriteGLB(const Path &path, ModelOutput *model) {
	std::string json = BuildJson(*model);
	while (json.size() & 3)
		json.push_back(' ');
	while (model->binary.size() & 3)
		model->binary.push_back(0);

	const uint32_t jsonLength = (uint32_t)json.size();
	const uint32_t binaryLength = (uint32_t)model->binary.size();
	const uint32_t totalLength = 12 + 8 + jsonLength + 8 + binaryLength;

	FILE *fp = File::OpenCFile(path, "wb");
	if (!fp)
		return false;

	auto writeU32 = [&](uint32_t value) {
		uint8_t b[4] = {
			(uint8_t)(value & 0xFF), (uint8_t)((value >> 8) & 0xFF),
			(uint8_t)((value >> 16) & 0xFF), (uint8_t)((value >> 24) & 0xFF),
		};
		return std::fwrite(b, 1, sizeof(b), fp) == sizeof(b);
	};

	bool ok = writeU32(0x46546C67) && writeU32(2) && writeU32(totalLength) &&
		writeU32(jsonLength) && writeU32(0x4E4F534A) &&
		std::fwrite(json.data(), 1, json.size(), fp) == json.size() &&
		writeU32(binaryLength) && writeU32(0x004E4942) &&
		(binaryLength == 0 || std::fwrite(model->binary.data(), 1, binaryLength, fp) == binaryLength);
	if (std::fclose(fp) != 0)
		ok = false;
	return ok;
}

}  // namespace

bool ExportFrameDumpToGLB(const Path &dumpPath, Path *outputPath, std::string *error) {
	std::vector<ParsedCommand> commands;
	std::vector<uint8_t> pushBuffer;
	if (!ReadDump(dumpPath, &commands, &pushBuffer, error))
		return false;

	// Version 6 stores the game ID in the PPDMP header.
	std::vector<uint8_t> headerBytes;
	if (!ReadFileBytes(dumpPath, &headerBytes) || headerBytes.size() < PPDMP_HEADER_SIZE) {
		*error = "Could not read the PPDMP game ID.";
		return false;
	}
	std::string gameID((const char *)headerBytes.data() + 12, 9);
	const size_t nul = gameID.find('\0');
	if (nul != std::string::npos)
		gameID.resize(nul);
	if (gameID.empty() || gameID.find('/') != std::string::npos || gameID.find('\\') != std::string::npos) {
		*error = "The PPDMP game ID is invalid.";
		return false;
	}

	const std::map<std::string, Path> savedTextures = FindSavedTextures(gameID);
	if (savedTextures.empty()) {
		*error = "No saved PNGs were found. Enable Save new textures and capture the player again.";
		return false;
	}

	GPUStateSnapshot state;
	for (int i = 0; i < 12; ++i)
		state.worldMatrix[i] = IdentityValue(i);
	int boneWord = 0;
	int worldWord = 0;
	PendingDraw pending;
	ModelOutput model;

	for (const ParsedCommand &cmd : commands) {
		if (cmd.type == (uint8_t)CommandType::INIT) {
			// GEState::Save stores its command-compatible state block after the first 17 words.
			// It includes the initial texture registers and all initial bone/world matrices.
			const uint32_t start = cmd.ptr + 17 * 4;
			const uint32_t end = cmd.ptr + cmd.size;
			for (uint32_t offset = start; offset + 4 <= end; offset += 4) {
				const uint32_t op = ReadU32(pushBuffer.data() + offset);
				if ((op >> 24) == 0x0C)  // GE_CMD_END
					break;
				UpdateStateFromRegister(&state, op, &boneWord, &worldWord);
			}
		} else if (cmd.type == (uint8_t)CommandType::REGISTERS) {
			const uint32_t count = cmd.size / 4;
			for (uint32_t i = 0; i < count; ++i) {
				const uint32_t op = ReadU32(pushBuffer.data() + cmd.ptr + (size_t)i * 4);
				UpdateStateFromRegister(&state, op, &boneWord, &worldWord);
				if ((op >> 24) == 0x04 && pending.haveVertices) {
					BuildPrimitive(pending, op, savedTextures, &model);
					pending = PendingDraw{};
				}
			}
		} else if (cmd.type == (uint8_t)CommandType::INDICES) {
			pending.indices.assign(pushBuffer.begin() + cmd.ptr, pushBuffer.begin() + cmd.ptr + cmd.size);
		} else if (cmd.type == (uint8_t)CommandType::VERTICES) {
			pending.vertices.assign(pushBuffer.begin() + cmd.ptr, pushBuffer.begin() + cmd.ptr + cmd.size);
			pending.state = state;
			pending.haveVertices = true;
		}
	}

	if (model.primitives.empty()) {
		*error = "No textured 3D player geometry was found in this frame. Capture the Edit Player model while it is visible.";
		return false;
	}

	*outputPath = dumpPath.WithReplacedExtension(".glb");
	if (!WriteGLB(*outputPath, &model)) {
		*error = "Could not write the GLB file.";
		return false;
	}
	return true;
}

}  // namespace GPURecord
