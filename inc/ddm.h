#include <cmath>
#include <Eigen/Dense>
#include <params.h>
#include <iostream>

#ifndef EPS
 #define EPS 1e-10
#endif

namespace DDM {
    // Bernoulli function
    template <typename T>
    T B(T x)
    {
        if (std::abs(x) < EPS)
            return 1.0;
        else
            return x / (std::exp(x) - 1.0);
    }

    // Bernoulli function derivative
    template <typename T>
    T dB_dx(T x)
    {
        if (std::abs(x) < EPS)
            return 0.0;
        else {
            T exp_x = std::exp(x);
            return -(exp_x * (x - 1) + 1) / ((exp_x - 1) * (exp_x - 1));
        }
    }

    // Shockley-Read-Hall recombination
    template <typename T>
    T srh(T n, T p, T nie, T tn, T tp)
    {
        T numerator = n * p - nie * nie;
        T denominator = tn * (n + nie) + tp * (p + nie);
        return numerator / denominator;
    }

    // Derivative of SRH with respect to n
    template <typename T>
    T dsrh_dn(T n, T p, T nie, T tn, T tp)
    {
        T C = (tp * (n + nie) + tn * (p + nie)); 
        T g = p / C - tn * (n * p - nie * nie) / C / C; 
        return g; 
    }

    // Derivative of SRH with respect to p
    template <typename T>
    T dsrh_dp(T n, T p, T nie, T tn, T tp)
    {
        T C = (tp * (n + nie) + tn * (p + nie)); 
        T g = n / C - tp * (n * p - nie * nie) / C / C; 
        return g; 
    }

    // calculate electron current density
    template <typename T>
    T fun_Jn(T n0, T n1, T psi0, T psi1, T dx, T Vt, T q, T mu_n)
    {
        T Bp = B((psi1 - psi0) / Vt);
        T Bm = B((psi0 - psi1) / Vt);
        return q * mu_n * Vt / dx * (n1 * Bp - n0 * Bm);
    }

    // calculate hole current density
    template <typename T>
    T fun_Jp(T p0, T p1, T psi0, T psi1, T dx, T Vt, T q, T mu_p)
    {
        T Bp = B((psi0 - psi1) / Vt);
        T Bm = B((psi1 - psi0) / Vt);
        return q * mu_p * Vt / dx * (p0 * Bm - p1 * Bp);
    }

    // calculate the quasi-Fermi level of electrons
    template <typename T>
    T fun_Efn(T n, T nie, T psi, T Vt)
    {
        return psi - Vt * std::log(n / nie);
    }

    // continuity equation for electrons at node i
    class JnContEq_at_i{
        private: 
            int i;                      // index of the node
            double dx;                   // grid spacing
            double n0;                   // electron density at node 0
            double n2;                   // electron density at node -1
            Eigen::VectorXd psi;        // electrostatic potential at nodes
            Eigen::VectorXd p;          // hole density at nodes
            Eigen::VectorXd An;         // alpha n
            Eigen::VectorXd Ap;         // alpha p
            int N;                      // number of nodes

            Params<double>* params;

        public:
            // default constructor
            JnContEq_at_i() : i(0), dx(0.0), n0(0.0), n2(0.0), N(0), params(NULL) {}

            JnContEq_at_i(int i_, double dx_, double n0_, double n2_, double* psi_, double* p_, double* An_, double* Ap_, int N_, Params<double>* params_)
                : i(i_), dx(dx_), n0(n0_), n2(n2_), N(N_), params(params_) {
                    psi = Eigen::VectorXd::Map(psi_, N);
                    p = Eigen::VectorXd::Map(p_, N);
                    An = Eigen::VectorXd::Map(An_, N - 1);
                    Ap = Eigen::VectorXd::Map(Ap_, N - 1);
                }

            JnContEq_at_i(int i_, double dx_, double n0_, double n2_, Eigen::VectorXd psi_, Eigen::VectorXd p_, Eigen::VectorXd An_, Eigen::VectorXd Ap_, int N_, Params<double>* params_)
                : i(i_), dx(dx_), n0(n0_), n2(n2_), psi(psi_), p(p_), An(An_), Ap(Ap_), N(N_), params(params_) {}

            // evaluate the continuity equation 
            double operator() (Eigen::VectorXd x, double dx);

            // evalute the gradient with respect to n
            void gradient_n(Eigen::VectorXd x, double dx, Eigen::VectorXd& grad);
    }; 

    // continuity equation for electrons
    class JnContEq {
        private: 

            int N;                     // number of nodes
            double n0;                  // electron density at node 0
            double n2;                 // electron density at node -1
            Eigen::VectorXd psi;        // electrostatic potential at nodes
            Eigen::VectorXd p;          // hole density at nodes
            Params<double>* params;
            JnContEq_at_i* JnEqs; // array of continuity equations at each internal node

        public: 
            // default constructor
            JnContEq() : N(0), n0(0.0), n2(0.0), params(NULL), JnEqs(NULL) {}

            JnContEq(double dx_, double n0_, double n2_, double* psi_, double* p_, double* An_, double* Ap_, int N_, Params<double>* params_)
             : N(N_), n0(n0_), n2(n2_), params(params_)
            {
                psi = Eigen::VectorXd::Map(psi_, N);
                p = Eigen::VectorXd::Map(p_, N);
                JnEqs = (JnContEq_at_i*) malloc((N_ - 2) * sizeof(JnContEq_at_i));
                for (int i = 1; i < N_ - 1; i++) {
                    JnEqs[i - 1] = JnContEq_at_i(i, dx_, n0_, n2_, psi_, p_, An_, Ap_, N_, params_);
                }
            }

            JnContEq(double dx_, double n0_, double n2_, Eigen::VectorXd psi_, Eigen::VectorXd p_, Eigen::VectorXd An_, Eigen::VectorXd Ap_, int N_, Params<double>* params_)
             : N(N_), n0(n0_), n2(n2_), psi(psi_), p(p_), params(params_)
            {
                JnEqs = (JnContEq_at_i*) malloc((N_ - 2) * sizeof(JnContEq_at_i));
                for (int i = 1; i < N_ - 1; i++) {
                    JnEqs[i - 1] = JnContEq_at_i(i, dx_, n0_, n2_, psi_, p_, An_, Ap_, N_, params_);
                }
            }

            // evaluate the continuity equations at all internal nodes
            Eigen::VectorXd fun(Eigen::VectorXd x, double dx);

            // evalute the gradient with respect to n
            Eigen::MatrixXd jacobian(Eigen::VectorXd x, double dx);

            // evalute the current density at all edges
            Eigen::VectorXd Jn(Eigen::VectorXd x, double dx); 
    }; 
}
