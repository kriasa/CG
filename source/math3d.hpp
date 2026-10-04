#pragma once

#include <cmath>

namespace math3d {

constexpr float pi = 3.14159265358979323846f;

constexpr float radians(float degrees) {
	return degrees * (pi / 180.0f);
}

struct Vec3 {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
};

struct Mat4 {
	float m[16];
};

inline Mat4 identity() {
	Mat4 r{};
	r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
	return r;
}

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r{};
	for (int c = 0; c < 4; ++c) {
		for (int row = 0; row < 4; ++row) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.m[k * 4 + row] * b.m[c * 4 + k];
			}
			r.m[c * 4 + row] = sum;
		}
	}
	return r;
}

inline Mat4 translation(const Vec3& t) {
	Mat4 r = identity();
	r.m[12] = t.x;
	r.m[13] = t.y;
	r.m[14] = t.z;
	return r;
}

inline Mat4 scaling(const Vec3& s) {
	Mat4 r = identity();
	r.m[0] = s.x;
	r.m[5] = s.y;
	r.m[10] = s.z;
	return r;
}

inline Mat4 rotationX(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[5] = c;
	r.m[6] = s;
	r.m[9] = -s;
	r.m[10] = c;
	return r;
}

inline Mat4 rotationY(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[0] = c;
	r.m[2] = -s;
	r.m[8] = s;
	r.m[10] = c;
	return r;
}

inline Mat4 rotationZ(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[0] = c;
	r.m[1] = s;
	r.m[4] = -s;
	r.m[5] = c;
	return r;
}

inline Mat4 rotationXYZ(const Vec3& angles) {
	return rotationZ(angles.z) * rotationY(angles.y) * rotationX(angles.x);
}

inline Mat4 perspective(float fov_y, float aspect, float near_plane, float far_plane) {
	const float f = 1.0f / std::tan(fov_y * 0.5f);
	Mat4 r{};
	r.m[0] = f / aspect;
	r.m[5] = -f;
	r.m[10] = far_plane / (near_plane - far_plane);
	r.m[11] = -1.0f;
	r.m[14] = (near_plane * far_plane) / (near_plane - far_plane);
	return r;
}

inline Mat4 orthographic(float width, float height, float near_plane, float far_plane) {
	Mat4 r{};
	r.m[0] = 2.0f / width;
	r.m[5] = -2.0f / height;
	r.m[10] = 1.0f / (near_plane - far_plane);
	r.m[14] = near_plane / (near_plane - far_plane);
	r.m[15] = 1.0f;
	return r;
}

} // namespace math3d