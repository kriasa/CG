#pragma once

#include <cmath>

namespace math3d {

constexpr float pi = 3.14159265358979323846f;

constexpr float radians(float degrees) {
	return degrees * (pi / 180.0f);
}

constexpr float degrees(float radians_value) {
	return radians_value * (180.0f / pi);
}

struct Vec3 {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
};

struct Mat4 {
	float m[16];
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline Vec3 operator*(const Vec3& a, float s) {
	return { a.x * s, a.y * s, a.z * s };
}

inline float dot(const Vec3& a, const Vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

inline float length(const Vec3& a) {
	return std::sqrt(dot(a, a));
}

inline Vec3 normalize(const Vec3& a) {
	const float len = length(a);
	return len > 1e-8f ? a * (1.0f / len) : Vec3{};
}

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

// Matrix whose columns are the given orthonormal basis vectors
// (maps local +X/+Y/+Z to right/up/forward).
inline Mat4 basis(const Vec3& right, const Vec3& up, const Vec3& forward) {
	Mat4 r = identity();
	r.m[0] = right.x;   r.m[1] = right.y;   r.m[2] = right.z;
	r.m[4] = up.x;      r.m[5] = up.y;      r.m[6] = up.z;
	r.m[8] = forward.x; r.m[9] = forward.y; r.m[10] = forward.z;
	return r;
}

// View matrix for a camera at `eye` looking along `forward` (right-handed, camera looks down -Z).
inline Mat4 lookAlong(const Vec3& eye, const Vec3& forward, const Vec3& world_up) {
	const Vec3 f = normalize(forward);
	const Vec3 r = normalize(cross(f, world_up));
	const Vec3 u = cross(r, f);

	Mat4 v = identity();
	v.m[0] = r.x;  v.m[4] = r.y;  v.m[8] = r.z;   v.m[12] = -dot(r, eye);
	v.m[1] = u.x;  v.m[5] = u.y;  v.m[9] = u.z;   v.m[13] = -dot(u, eye);
	v.m[2] = -f.x; v.m[6] = -f.y; v.m[10] = -f.z; v.m[14] = dot(f, eye);
	return v;
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