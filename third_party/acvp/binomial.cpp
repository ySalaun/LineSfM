#include "binomial.h"
#include <math.h>
#include <assert.h>
#include <float.h>
#include <iostream>
//#include "pow.h"

static const double THRESHOLD_HOEFFDING = DBL_EPSILON*1000.0; 
static const double MAX_TMP_PROBA = 1024.0;
static const double THRESHOLD_TAILDECREASE = DBL_EPSILON*0.1; 
static const double THRESHOLD_GEO_RATIO = 0.9;

std::vector<double> Binomial::invIntegers;
/// Computes the vector of inverse of integers up to \a iMax
/// if not previously computed
void Binomial::buildInvIntegers(int iMax)
{
  for(int k=(int) invIntegers.size(); k<=iMax; k++) 
    invIntegers.push_back(1.0/k);
}

std::vector<double> Binomial::logFactorials;
/// Computes the vector of log of factorial of integers up to \a iMax
/// if not previously computed
void Binomial::buildLogFactorials(unsigned int iMax)
{ 
  if(logFactorials.empty())
    logFactorials.push_back(0.0);
  for(unsigned int k=(unsigned int) logFactorials.size(); k<=iMax; k++) 
    logFactorials.push_back(log10((double) k) + logFactorials.back());

}

/// Constructor
Binomial::Binomial(const double proba)
: p(proba), q(1.0-proba), table(1, std::vector<double>(1, 1.0))
{
    assert(p >= 0 && p <= 1.0);
}

/// Access to line \a l of table
const std::vector<double>& Binomial::operator()(unsigned int l) const
{
    if(l >= table.size())
	const_cast<Binomial*>(this)->extend(l);
    return table[l];
}

/// Access operator
double Binomial::operator()(unsigned int l, unsigned int k) const
{
    assert(k <= l);
    if(l >= table.size())
	const_cast<Binomial*>(this)->extend(l);
    return table[l][k];
}

/// Fill table up to line \a l
void Binomial::extend(unsigned int l)
{
    for(unsigned int i = (unsigned int) table.size(); i <= l; i++) {
	const std::vector<double>& w = table.back();
	std::vector<double> v;
	v.reserve(i+1);
	v.push_back(1.0);
	for(unsigned int k = 1; k < i; k++)
	    v.push_back(p*w[k-1] + q*w[k]);
	v.push_back(p * w.back());
	table.push_back(v);
    }
}

/// Returns the density value \f$b(n,k,p)\f$ of the binomial law.
/// Computations are made in log, then the exponential of this value
/// is returned.
double Binomial::density(unsigned int n, unsigned int k, double p)
{
  assert( n > 0 && p >= 0 && p <= 1.0);
  if (k > n) return 0.0;

  double out = 0.0; 

  double logp = (p > 0.) ? log10(p) : -DBL_MAX;
  if(logp != -DBL_MAX)
    out += k*logp;
  else 
    if(k > 0) return 0.0; 

  double log_1_p = (p < 1.) ? log10(1-p) : -DBL_MAX;
  if(log_1_p != -DBL_MAX)
    out += (n-k)*log_1_p;
  else 
    if(k < n) return 0.0;

  buildLogFactorials(n);
  out += logFactorials[n] - logFactorials[k] - logFactorials[n-k];  
  return (out > DBL_MIN_10_EXP) ? pow(10.0, out) : 0.0;
}

/// Tail of binomial distribution, without memory (static function)
double Binomial::tailPascalTriangle(unsigned int n, unsigned int k, double p)
{
    if(k > n)
        return 0;
    double q = 1.0 - p;
    double* buffer = new double[k+1];
    buffer[0] = 1.0;
    for(unsigned int i = 1; i <= k; i++)
        buffer[i] = 0;
    for(unsigned int i = 1; i <= n; i++) {
        unsigned int jMin = (k>n-i)? k-(n-i)-1: 0;
        for(unsigned int j = (i<=k)? i: k; j != jMin; j--)
            buffer[j] = p*buffer[j-1] + q*buffer[j];
    }
    q = buffer[k];
    delete [] buffer;
    return q;
}

/// Hoeffding upper bound of the binomial tail \f$ B(n, k, p) \f$
double Binomial::hoeffding(unsigned int n, unsigned int k, double p)
{
    p *= n;
    if(k <= p) return 1;
    return pow(p/k, k) * ( (k==n)? 1: pow((n-p)/(n-k), n-k) );
}

/// log of Hoeffding upper bound of the binomial tail \f$ B(n, k, p) \f$
double Binomial::logHoeffding(unsigned int n, unsigned int k, double p)
{
    p *= n;
    if(k <= p) return 0;
    return k*log10(p/k) + ( (k==n)? 0: (n-k)*log10((n-p)/(n-k)) );
}

/// Head of binomial distribution, without memory (static function)
double Binomial::head(unsigned int n, unsigned int k, double p)
{
  assert( n > 0 && p >= 0 && p <= 1.0);
  if(k >= n || p==0.0) return 1.0;
 
  buildInvIntegers(k);
  
  unsigned int m = n;
  unsigned int n1 = n+1;
  double q = 1.0 - p;
  double pdivq = p/q;    
  double* buffer = new double[k+1];
  buffer[0] = 1.0;    
  for(unsigned int j = 1; j <= k; j++)
    buffer[j] = buffer[j-1]*pdivq*(n1-j)*invIntegers[j];
  
  for(int j = (int)k-1; j >= 0; j--) 
    buffer[j] += buffer[j+1];
  double out = buffer[0];
  out *= pow(q,m);
  
  delete[] buffer;
  
  return out;
}

/// Fast computation of the tail of binomial distribution.
/// The key idea is to start summing up the density starting from its larger
/// values, and stop summing when the rest of the tail does not change the total sum
/// (up to machine precision). This bound is derived from the geometric decrease property
/// of the binomial distribution.
double Binomial::tailFastDecrease(unsigned int n, unsigned int k, double p)
{  
  double pDivq = p/(1-p);
  double qDivp = (1-p)/p;
  
  // minimum i s.t. decreasing ratio r1 = b(i+1)/b(i) is below THRESHOLD_GEO_RATIO
  int iMin = (int)std::max((n - THRESHOLD_GEO_RATIO*qDivp)/(1.0 + THRESHOLD_GEO_RATIO*qDivp),0.0);

  buildInvIntegers(n);
  
  double bi = density(n,k,p);
  double sum = bi;
  int n1 = n+1;

  // go right, summing up to reaching iMin
  //b(i) = b(i-1)*pDivq*(n-i+1)/i      
  for(int i = (int)k + 1; i <= iMin; i++){
    bi *= pDivq*(n1-i)*Binomial::invIntegers[i];
    sum += bi;
  }

  // go right, summing up to reaching machine precision
  //b(i) = b(i-1)*pDivq*(n-i+1)/i     
  double sum1 = bi;
  for(int i = std::max((int)k,iMin)+1; i <= (int)n; i++){
    bi *= pDivq*(n1-i)*Binomial::invIntegers[i];
    sum += bi;
    sum1 += bi;
    if(bi <= sum1*THRESHOLD_TAILDECREASE){ 
      //std::cout << "iMax = " << i << std::endl; 
      break;
    }
  }
  return sum;
}

/// Checks that parameters \a n, \a k and \a p are consistent,
/// then computes the tail using the fast tail computation method
/// \b tailFastDecrease
double Binomial::tail(unsigned int n, unsigned int k, double p)
{
  assert(p >= 0 && p <= 1.0);
  assert(n > 0 || p > 0);
  if(k > n) return 0;
  if(k==0) return 1.0;  
  if(p == 0) return 0.0;
  if(p == 1.) return 1.0;
  return tailFastDecrease(n,k,p);
}


