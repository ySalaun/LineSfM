/*********************************************************************/
// Copyright (c) 2012, 2013 Lionel MOISAN.
// Copyright (c) 2012, 2013 Pascal MONASSE.
// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/multiview/solver_essential_kernel.hpp"
#include "openMVG/multiview/essential.hpp"

using namespace openMVG;

/// logarithm (base 10) of binomial coefficient
static double logcombi(size_t k, size_t n){
  if (k>=n || k<=0) return(0.0);
  if (n-k<k) k=n-k;
  double r = 0.0;
  for (size_t i = 1; i <= k; i++)
    r += log10((double)(n-i+1))-log10((double)i);

  return r;
}

/// tabulate logcombi(.,n)
template<typename Type>
static void makelogcombi_n(size_t n, std::vector<Type> & l){
  l.resize(n+1);
  for (size_t k = 0; k <= n; k++)
    l[k] = static_cast<Type>( logcombi(k,n) );
}

/// tabulate logcombi(k,.)
template<typename Type>
static void makelogcombi_k(size_t k, size_t nmax, std::vector<Type> & l){
  l.resize(nmax+1);
  for (size_t n = 0; n <= nmax; n++)
    l[n] = static_cast<Type>(logcombi(k,n));
}

/// Compute P = K[R|t]
void P_From_KRt(
  const Mat3 &K,  const Mat3 &R,  const Vec3 &t, Mat34 *P) {
  *P = K * HStack(R,t);
}

void HomogeneousToEuclidean(const Vec4 &H, Vec3 *X) {
  double w = H(3);
  *X << H(0) / w, H(1) / w, H(2) / w;
}

double Depth(const Mat3 &R, const Vec3 &t, const Vec3 &X) {
  return (R*X)[2] + t[2];
}

// HZ 12.2 pag.312
void TriangulateDLT(const Mat34 &P1, const Vec2 &x1,
                    const Mat34 &P2, const Vec2 &x2,
                    Vec4 *X_homogeneous) {
  Mat4 design;
  for (int i = 0; i < 4; ++i) {
    design(0,i) = x1[0] * P1(2,i) - P1(0,i);
    design(1,i) = x1[1] * P1(2,i) - P1(1,i);
    design(2,i) = x2[0] * P2(2,i) - P2(0,i);
    design(3,i) = x2[1] * P2(2,i) - P2(1,i);
  }
  Nullspace(&design, X_homogeneous);
}

void TriangulateDLT(const Mat34 &P1, const Vec2 &x1,
                    const Mat34 &P2, const Vec2 &x2,
                    Vec3 *X_euclidean) {
  Vec4 X_homogeneous;
  TriangulateDLT(P1, x1, P2, x2, &X_homogeneous);
  HomogeneousToEuclidean(X_homogeneous, X_euclidean);
}

bool estimate_Rt_fromE(const Mat & x1, const Mat & x2,
		      const Mat3 & E,
		      Mat3 * R, Vec3 * t)
 {
 // Accumulator to find the best solution
  std::vector<size_t> f(4, 0);

  std::vector<Mat3> Es; // Essential,
  std::vector<Mat3> Rs;  // Rotation matrix.
  std::vector<Vec3> ts;  // Translation matrix.

  Es.push_back(E);
  // Recover best rotation and translation from E.
  openMVG::MotionFromEssential(E, &Rs, &ts);

  //-> Test the 4 solutions will all the point
  assert(Rs.size() == 4);
  assert(ts.size() == 4);

  Mat34 P1, P2;
  Mat3 R1 = Mat3::Identity();
  Vec3 t1 = Vec3::Zero();

  P_From_KRt(Mat3::Identity(), R1, t1, &P1);

  for (unsigned int i = 0; i < 4; ++i)
  {
    const Mat3 &R2 = Rs[i];
    const Vec3 &t2 = ts[i];
    P_From_KRt(Mat3::Identity(), R2, t2, &P2);
    Vec3 X;

    for (size_t k = 0; k < x1.cols(); ++k)
    {
      const Vec2 & x1_ = x1.col(k),
        &x2_ = x2.col(k);
      TriangulateDLT(P1, x1_, P2, x2_, &X);
      // Test if point is front to the two cameras.
      if (Depth(R1, t1, X) > 0 && Depth(R2, t2, X) > 0)
      {
        ++f[i];
      }
    }
  }
  // Check the solution:
  const std::vector<size_t>::iterator iter = max_element(f.begin(), f.end());
  if (*iter == 0)
  {
    return false;
  }
  const size_t index = std::distance(f.begin(), iter);
  (*R) = Rs[index];
  (*t) = ts[index];

  return true;
}