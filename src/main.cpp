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

    //// Ionization case
    // electrostatic potential at nodes
    double psi[10] = {4.1473229734337885e-01, -8.1260032366100166e-01, -2.1247444867659522e+00, -4.0099059730554076e+00, -8.7236608939135802e+00, -2.0386848167856964e+01, -2.0424128520627289e+01, -2.0460879577411845e+01, -2.049732309762790e+01, -2.0533785288087376e+01};

    // electron density at nodes
    double elec[10] = {1.0000010784770064e-01, 9.8889537646814185e-02, 9.249772885603638e-02, 6.4381801724163085e-02, 2.5277480776435379e-02, 5.004486148310315e-03, 6.4856808611659476e-04, 8.5643343514035176e-05, 1.0590665565141462e-05, 1.1631151615318097e-17}; 

    // hole density at nodes
    double hole[10] = {1.1631139072556444e-15, 1.0036783946520538e-09, 4.1920669266312947e-07, 1.4174372370654288e-03, 1.6268515580149870e-02, 9.8573739459186691e+00, 1.0000639018966130e+01, 1.0000085665408054e+01, 1.0000010599791899e+01, 1.0000000000078478e+01}; 

    // Electron avalanche ionization rate at edges
    double An_max[9] = {5.923594767688028e-11, 2.6530822983129435e-10, 1.9410922567425453e-07, 1.6744239464819542e-03, 6.120910718072933e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00}; 

    // Hole avalanche ionization rate at edges
    double Ap_max[9] = {3.3245408766107318e-17, 3.9693499099469527e-16, 2.1681476294862051e-11, 7.0095981859002765e-05, 2.6959873036004618e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00};
            
    double* Efn0 = (double*) malloc(N * sizeof(double));
    for (int ii = 0; ii < N; ii++) {
        Efn0[ii] = DDM::fun_Efn(elec[ii], params.nie, psi[ii], params.Vt);
    }
    std::cout << "Electron quasi-Fermi level at all nodes:" << std::endl;
    for (int ii = 0; ii < N; ii++) {
        std::cout << Efn0[ii] << ", ";
    }
    std::cout << std::endl;
    free(Efn0);

    // //// No ionization case
    // // electrostatic potential at nodes
    // double psi[10] = {4.1473229734337885e-01, 4.1473169015170430e-01, 4.1455994456864870e-01, 3.6364739031166103e-01, -6.2669795630223607e+00, -2.0533297801167279e+01, -2.0533785264383035e+01, -2.0533785280882615e+01, -2.0533785280883794e+01, -2.0533785280887205e+01};

    // // electron density at nodes
    // double elec[10] = {1.0000010784770064e-01, 9.999775923208490e-02, 9.9335635134857866e-02, 1.3861578969428820e-02, 3.11794662615377718e-13, 4.3013824161211626e-20, 7.7716418834887609e-18, 1.0366090179219692e-17, 1.1253455053568247e-17, 1.1631151615318097e-17}; 

    // // hole density at nodes
    // double hole[10] = {1.1631139072656444e-15, 7.0493976537457729e-16, 2.2826494926497466e-16, 3.4934994841812490e-18, 4.3635166759573616e-13, 9.8132013478913045e+00, 9.9999936177116542e+00, 9.999999997839808e+00, 1.00000000000000000e+01, 1.00000000001078478e+01}; 

    // // Electron avalanche ionization rate at edges
    // double An_max[9] = {0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II}; 

    // // Hole avalanche ionization rate at edges
    // double Ap_max[9] = {0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II, 0.0 * params.II};

    int maxit = 100;
    double ftol = 1e-10, xtol = 1e-10;

    int M2 = 1;
    int M3 = 1;
    int M4 = 11;
    double lb2 = 1.0; 
    double ub2 = 1.0; 
    double lb3 = 1.0; 
    double ub3 = 1.0; 
    double lb4 = 1e-5; 
    double ub4 = 1e-4; 

    double min_Jn = -1000; 

    for (int it = 0; it < M2 * M3 * M4; it++)
    {
        int i = it / (M3 * M4); 
        int j = (it / M4) % M3;
        int k = it % M4;

        double c2 = M2 > 1 ? (ub2 - lb2) * i / (M2 - 1) + lb2 : 1.0;
        double c3 = M3 > 1 ? (ub3 - lb3) * j / (M3 - 1) + lb3 : 1.0;
        double c4 = M4 > 1 ? (ub4 - lb4) * k / (M4 - 1) + lb4 : 1.0;

        double An[9]; 
        double Ap[9];

        memcpy(An, An_max, 9 * sizeof(double));
        memcpy(Ap, Ap_max, 9 * sizeof(double));
        An[2] *= c2;
        An[3] *= c3;
        An[4] *= c4;
        Ap[2] *= c2;
        Ap[3] *= c3;
        Ap[4] *= c4;

        DDM::JnContEq JnEqs_it(dx, elec[0], elec[N - 1], psi, hole, An, Ap, N, &params);
        Eigen::VectorXd elec_in = Eigen::VectorXd::Map(elec + 1, N - 2);

        // solve the continuity equations using Newton's method
        std::function<Eigen::VectorXd(Eigen::VectorXd, double)> func = std::bind(&DDM::JnContEq::fun, &JnEqs_it, std::placeholders::_1, std::placeholders::_2);
        std::function<Eigen::MatrixXd(Eigen::VectorXd, double)> jacobian = std::bind(&DDM::JnContEq::jacobian, &JnEqs_it, std::placeholders::_1, std::placeholders::_2);
        Eigen::VectorXd elec_out = NewtonIteration(func, jacobian, elec_in, dx, ftol, xtol, maxit, false); 
        // std::cout << "Solved electron density at internal nodes:" << std::endl;
        // std::cout << elec_out.transpose() << std::endl;

        // evaluate the current density at all edges
        Eigen::VectorXd Jn = JnEqs_it.Jn(elec_out, dx);
        // std::cout << "Electron current density at all edges:" << std::endl;
        // std::cout << Jn.transpose() << std::endl;

        if (Jn.minCoeff() > 0.0)
        {
            std::cout << "An scaling factors: " << c2 << ", " << c3 << ", " << c4 << std::endl;
            std::cout << "Ap scaling factors: " << c2 << ", " << c3 << ", " << c4 << std::endl;
            std::cout << "Electron current density at all edges:" << std::endl;
            std::cout << Jn.transpose() << std::endl;

            Jn = JnEqs_it.Jn(Eigen::VectorXd::Map(elec + 1, N - 2), dx);
            std::cout << "Initial electron current density at all edges:" << std::endl;
            std::cout << Jn.transpose() << std::endl;

            double* elec_full = (double*) malloc(N * sizeof(double));
            elec_full[0] = elec[0];
            for (int ii = 1; ii < N - 1; ii++) {
                elec_full[ii] = elec_out[ii - 1];
            }
            elec_full[N - 1] = elec[N - 1];
            
            double* Efn = (double*) malloc(N * sizeof(double));
            for (int ii = 0; ii < N; ii++) {
                Efn[ii] = DDM::fun_Efn(elec_full[ii], params.nie, psi[ii], params.Vt);
            }
            std::cout << "Electron quasi-Fermi level at all nodes:" << std::endl;
            for (int ii = 0; ii < N; ii++) {
                std::cout << Efn[ii] << ", ";
            }
            std::cout << std::endl;
            free(elec_full);
            free(Efn);
        }

        min_Jn = std::max(min_Jn, Jn.minCoeff());
    }

    std::cout << "Maximum of minimum electron current density at all edges: " << min_Jn << std::endl;
    
    // // Drift-diffusion model object

    // // construct continuity equations
    // memset(An_max, 0, 9 * sizeof(double));
    // memset(Ap_max, 0, 9 * sizeof(double));
    // DDM::JnContEq JnEqs(dx, elec[0], elec[N - 1], psi, hole, An_max, Ap_max, N, &params);

    // Eigen::VectorXd elec_in = Eigen::VectorXd::Map(elec + 1, N - 2);
    // // Eigen::VectorXd elec_in = Eigen::VectorXd::Zero(N - 2);

    // // solve the continuity equations using Newton's method
    // std::function<Eigen::VectorXd(Eigen::VectorXd, double)> func = std::bind(&DDM::JnContEq::fun, &JnEqs, std::placeholders::_1, std::placeholders::_2);
    // std::function<Eigen::MatrixXd(Eigen::VectorXd, double)> jacobian = std::bind(&DDM::JnContEq::jacobian, &JnEqs, std::placeholders::_1, std::placeholders::_2);
    // Eigen::VectorXd elec_out = NewtonIteration(func, jacobian, elec_in, dx, ftol, xtol, maxit, true); 
    // std::cout << "Solved electron density at internal nodes:" << std::endl;
    // std::cout << elec_out.transpose() << std::endl;

    // // evaluate the current density at all edges
    // Eigen::VectorXd Jn = JnEqs.Jn(elec_out, dx);
    // std::cout << "Electron current density at all edges:" << std::endl;
    // std::cout << Jn.transpose() << std::endl;

    return 0;
}
