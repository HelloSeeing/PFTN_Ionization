#include <Eigen/Dense>
#include <iostream>

Eigen::VectorXd NewtonIteration(
    std::function<Eigen::VectorXd(Eigen::VectorXd, double)> func,
    std::function<Eigen::MatrixXd(Eigen::VectorXd, double)> jacobian,
    Eigen::VectorXd x0,
    double dx, 
    double ftol = 1e-6,
    double xtol = 1e-6,
    int max_iter = 100, 
    bool verbose = false)
{
    Eigen::VectorXd x(x0); 
    int iter = 0; 
    while (iter < max_iter) {
        Eigen::VectorXd f_val = func(x, dx);
        Eigen::MatrixXd J = jacobian(x, dx);

        // Solve J * delta = -f_val
        Eigen::VectorXd delta = J.fullPivLu().solve(-f_val);

        if (verbose) {
            std::cout << "Iteration " << iter << ": x = " << x.transpose() << std::endl;
            std::cout << "Iteration " << iter << ": f(x) = " << f_val.transpose() << std::endl;
            std::cout << "Iteration " << iter << ": dx = " << delta.transpose() << std::endl;
        }

        x += delta;
        iter++; 

        if (delta.norm() < xtol and f_val.norm() < ftol) {
            break; // Converged
        }
    }

    if (verbose) {
        std::cout << "Newton's method finished in " << iter << " iterations." << std::endl;
        std::cout << "Final residual norm: " << func(x, dx).norm() << std::endl;
    }
    
    return x;
}
