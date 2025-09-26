#include <Eigen/Dense>
#include <iostream>
#include <ddm.h>
#include <newton_iteration.h>
#include <stdlib.h>

int main() {

    Params<double> params; 

    // grid spacing
    int N = 10; 
    double dx = 2.0e-4 * params.cm / (N - 1); 

    // electrostatic potential at nodes
    double psi[10] = {4.1473229734337885e-01, 4.1473169015170430e-01, 4.1455994456864870e-01, 3.6364739031166103e-01, -6.2669795630223607e+00, -2.0533297801167279e+01, -2.0533785264383035e+01, -2.0533785280882615e+01, -2.0533785280883794e+01, -2.0533785280887205e+01};

    // electron density at nodes
    double elec[10] = {1.0000010784770064e-01, 9.999775923208490e-02, 9.9335635134857866e-02, 1.3861578969428820e-02, 3.11794662615377718e-13, 4.3013824161211626e-20, 7.7716418834887609e-18, 1.0366090179219692e-17, 1.1253455053568247e-17, 1.1631151615318097e-17}; 

    // hole density at nodes
    double hole[10] = {1.1631139072656444e-15, 7.0493976537457729e-16, 2.2826494926497466e-16, 3.4934994841812490e-18, 4.3635166759573616e-13, 9.8132013478913045e+00, 9.9999936177116542e+00, 9.999999997839808e+00, 1.00000000000000000e+01, 1.00000000001078478e+01}; 

    // Electron avalanche ionization rate at edges
    double An_max[9] = {0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II}; 

    // Hole avalanche ionization rate at edges
    double Ap_max[9] = {0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II, 0.0f * params.II};
    
    //// Drift-diffusion model object

    // construct continuity equations
    DDM::JnContEq JnEqs(dx, elec[0], elec[N - 1], psi, hole, An_max, Ap_max, N, &params);

    // Eigen::VectorXd elec_in = Eigen::VectorXd::Map(elec + 1, N - 2);
    Eigen::VectorXd elec_in = Eigen::VectorXd::Zero(N - 2);

    // solve the continuity equations using Newton's method
    std::function<Eigen::VectorXd(Eigen::VectorXd, double)> func = std::bind(&DDM::JnContEq::fun, &JnEqs, std::placeholders::_1, std::placeholders::_2);
    std::function<Eigen::MatrixXd(Eigen::VectorXd, double)> jacobian = std::bind(&DDM::JnContEq::jacobian, &JnEqs, std::placeholders::_1, std::placeholders::_2);
    Eigen::VectorXd elec_out = NewtonIteration(func, jacobian, elec_in, dx, 1e-10, 1e-10, 100, true); 
    std::cout << "Solved electron density at internal nodes:" << std::endl;
    std::cout << elec_out.transpose() << std::endl;

    // evaluate the current density at all edges
    Eigen::VectorXd Jn = JnEqs.Jn(elec_out, dx);
    std::cout << "Electron current density at all edges:" << std::endl;
    std::cout << Jn.transpose() << std::endl;

    return 0;
}
