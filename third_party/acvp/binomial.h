#ifndef BINOMIAL_H
#define BINOMIAL_H
#include <vector> // STL
//#include "stdafx.h"

#define LIBPROBA_IMEXPORT

/// Binomial tail distribution
class LIBPROBA_IMEXPORT Binomial {
public:
    Binomial(double proba);
    ~Binomial() {}
    const std::vector<double>& operator()(unsigned int n) const;
    double operator()(unsigned int n, unsigned int k) const;
    static double density(unsigned int n, unsigned int k, double p);
    static double tailPascalTriangle(unsigned int n, unsigned int k, double proba);
    static double tail(unsigned int n, unsigned int k, double proba);
    static double head(unsigned int n, unsigned int k, double proba);
    static double hoeffding(unsigned int n, unsigned int k, double proba);
    static double logHoeffding(unsigned int n, unsigned int k, double proba);
private:
    friend class Trinomial;
    static std::vector<double> invIntegers;///< Vector of inverse of integers, up to a needed integer    
    static std::vector<double> logFactorials;///< Vector of log of factorials, up to a needed integer
    static void buildInvIntegers(int iMax);
    static void buildLogFactorials(unsigned int iMax);
    const double p, q; ///< binomial parameters (usually \f$ q = 1 -p \f$)
    std::vector<std::vector<double> > table; ///< Table containing \f$ binom(n,k,p) \f$ for all \f$k = 0,...,n\f$
    void extend(unsigned int n);
    static double tailFastDecrease(unsigned int n, unsigned int k, double p);
};
#endif
