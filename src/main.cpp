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
    double psi[10] = {4.1473229734337885e-01 * params.V, -8.1260032366087298e-01 * params.V, -2.1247444867659375e+00 * params.V, -4.0099059730555187e+00 * params.V, -8.7236608939140492e+00 * params.V, -2.0386848167857419e+01 * params.V, -2.0424128529627694e+01 * params.V, -2.0460679577412183e+01 * params.V, -2.0497232309763017e+01 * params.V, -2.0533785280887376e+01 * params.V};

    // electron density at nodes
    double elec[10] = {1.0000010784770062e+17 * params.conc, 9.8889537646814176e+16 * params.conc, 9.2497728836036832e+16 * params.conc, 6.4381801724163072e+16 * params.conc, 2.5277480776435376e+16 * params.conc, 5.0044867184310290e+15 * params.conc, 6.4856809631662212e+14 * params.conc, 8.5643343514044922e+13 * params.conc, 1.0596665565144477e+13 * params.conc, 1.1631151615318096e+01 * params.conc}; 

    // hole density at nodes
    double hole[10] = {1.1631139072656442e+03 * params.conc, 1.0036703946522678e+09 * params.conc, 4.1920669266315485e+11 * params.conc, 1.4174372370654452e+15 * params.conc, 1.6268515580149880e+16 * params.conc, 9.8527829459186688e+18 * params.conc, 1.0000639018966129e+19 * params.conc, 1.0000085665400054e+19 * params.conc, 1.0000010599791899e+19 * params.conc, 1.0000000001078477e+19 * params.conc}; 

    // Electron avalanche ionization rate at edges
    double An_max[9] = {5.9235947676736080e-05 * params.II, 2.6530822983179386e-04 * params.II, 1.9410922567445107e-01 * params.II, 1.6744239464827231e+03 * params.II, 6.1209107718072737e+04 * params.II, 0.0000000000000000e+00 * params.II, 0.0000000000000000e+00 * params.II, 0.0000000000000000e+00 * params.II, 0.0000000000000000e+00 * params.II}; 

    // Hole avalanche ionization rate at edges
    double Ap_max[9] = {3.3245408765973377e-11 * params.II, 3.9693499009593061e-10 * params.II, 2.1681476294898329e-05 * params.II, 7.0095981859056053e+01 * params.II, 2.6959873036004476e+04 * params.II, 0.0000000000000000e+00 * params.II, 0.0000000000000000e+00 * params.II, 0.0000000000000000e+00 * params.II, 0.0000000000000000e+00 * params.II};
    
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
