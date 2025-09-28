#include <ddm.h>
#include <cmath>

double DDM::JnContEq_at_i::operator()(Eigen::VectorXd x, double dx)
{
    double n1 = x[i - 1];
    double n0 = (i == 1) ? this->n0 : x[i - 2];
    double n2 = (i == N - 2) ? this->n2 : x[i + 0];

    double p0 = p[i - 1]; 
    double p1 = p[i];
    double p2 = p[i + 1];

    double psi0 = psi[i - 1];
    double psi1 = psi[i];
    double psi2 = psi[i + 1];

    double Jn0 = DDM::fun_Jn(n0, n1, psi0, psi1, dx, params->Vt, params->q, params->mun);
    double Jn1 = DDM::fun_Jn(n1, n2, psi1, psi2, dx, params->Vt, params->q, params->mun);

    double Jp0 = DDM::fun_Jp(p0, p1, psi0, psi1, dx, params->Vt, params->q, params->mup);
    double Jp1 = DDM::fun_Jp(p1, p2, psi1, psi2, dx, params->Vt, params->q, params->mup);

    double ef0 = (psi0 - psi1) / dx; // electric field at left edge
    double ef1 = (psi1 - psi2) / dx; // electric field at right edge

    double a0 = An[i - 1]; 
    double a1 = An[i];
    double b0 = Ap[i - 1];
    double b1 = Ap[i];

    double R = DDM::srh(n1, p1, params->nie, params->tn, params->tp); 

    double G_left = 0.0; 
    if (ef0 * (Jn0 + Jp0) > 0)
    {
        G_left = a0 * std::abs(Jn0 / params->q) + b0 * std::abs(Jp0 / params->q);
    }

    double G_right = 0.0;
    if (ef1 * (Jn1 + Jp1) > 0)
    {
        G_right = a1 * std::abs(Jn1 / params->q) + b1 * std::abs(Jp1 / params->q);
    }

    double G = 0.5 * (G_left + G_right);

    double residual = Jn1 - Jn0 + dx * params->q * (G - R); 

    return residual; 
}

void DDM::JnContEq_at_i::gradient_n(Eigen::VectorXd x, double dx, Eigen::VectorXd& grad)
{
    // zero the gradient
    for (int j = 0; j < N - 2; j++) {
        grad[j] = 0.0;
    }

    double n1 = x[i - 1];
    double n0 = (i == 1) ? this->n0 : x[i - 2];
    double n2 = (i == N - 2) ? this->n2 : x[i + 0];

    double p0 = p[i - 1];
    double p1 = p[i];
    double p2 = p[i + 1];

    double psi0 = psi[i - 1];
    double psi1 = psi[i];
    double psi2 = psi[i + 1];

    double Jn0 = DDM::fun_Jn(n0, n1, psi0, psi1, dx, params->Vt, params->q, params->mun);
    double Jn1 = DDM::fun_Jn(n1, n2, psi1, psi2, dx, params->Vt, params->q, params->mun);

    double Jp0 = DDM::fun_Jp(p0, p1, psi0, psi1, dx, params->Vt, params->q, params->mup);
    double Jp1 = DDM::fun_Jp(p1, p2, psi1, psi2, dx, params->Vt, params->q, params->mup);

    double dpsi0 = (psi1 - psi0) / params->Vt; 
    double dpsi1 = (psi2 - psi1) / params->Vt;

    Eigen::VectorXd d_Jn0 = Eigen::VectorXd::Zero(N - 2);
    Eigen::VectorXd d_Jn1 = Eigen::VectorXd::Zero(N - 2);

    d_Jn0[i - 1] = params->q * params->mun * params->Vt / dx * DDM::B(dpsi0);
    if (i > 1) {
        d_Jn0[i - 2] = -params->q * params->mun * params->Vt / dx * DDM::B(-dpsi0);
    }
    d_Jn1[i - 1] = -params->q * params->mun * params->Vt / dx * DDM::B(-dpsi1);
    if (i < N - 2) {
        d_Jn1[i] = params->q * params->mun * params->Vt / dx * DDM::B(dpsi1);
    }

    // add the gradient from the current difference
    grad += d_Jn1 - d_Jn0;

    // add the gradient from srh combination term
    double dR_dn = DDM::dsrh_dn(n1, p1, params->nie, params->tn, params->tp);
    grad[i - 1] -= dx * params->q * dR_dn;

    double ef0 = (psi0 - psi1) / dx; // electric field at left edge
    double ef1 = (psi1 - psi2) / dx; // electric field at right edge

    double a0 = An[i - 1]; 
    double a1 = An[i];
    // double b0 = Ap[i - 1];
    // double b1 = Ap[i];

    Eigen::VectorXd d_G_left = Eigen::VectorXd::Zero(N - 2);
    if (ef0 * (Jn0 + Jp0) > 0)
    {
        if (Jn0 > 0)
        {
            d_G_left += a0 * d_Jn0 / params->q;
        }
        else
        {
            d_G_left -= a0 * d_Jn0 / params->q;
        }
    }

    Eigen::VectorXd d_G_right = Eigen::VectorXd::Zero(N - 2);
    if (ef1 * (Jn1 + Jp1) > 0)
    {
        if (Jn1 > 0)
        {
            d_G_right += a1 * d_Jn1 / params->q;
        }
        else
        {
            d_G_right -= a1 * d_Jn1 / params->q;
        }
    }

    Eigen::VectorXd dG = 0.5 * (d_G_left + d_G_right);

    // add the gradient from generation term
    grad += dx * params->q * dG;
}

Eigen::VectorXd DDM::JnContEq::fun(Eigen::VectorXd x, double dx)
{
    Eigen::VectorXd res = Eigen::VectorXd::Zero(N - 2); 
    for (int i = 1; i < N - 1; i++) {
        res[i - 1] = JnEqs[i - 1](x, dx);
    }
    return res; 
}

Eigen::MatrixXd DDM::JnContEq::jacobian(Eigen::VectorXd x, double dx)
{
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(N - 2, N - 2); 
    Eigen::VectorXd g = Eigen::VectorXd::Zero(N - 2); 
    for (int i = 1; i < N - 1; i++) {
        JnEqs[i - 1].gradient_n(x, dx, g);
        grad.row(i - 1) = g.transpose(); 
        g.setZero(); 
    }
    return grad; 
}

Eigen::VectorXd DDM::JnContEq::Jn(Eigen::VectorXd x, double dx)
{
    Eigen::VectorXd Jn_all = Eigen::VectorXd::Zero(N - 1); 
    for (int i = 0; i < N - 1; i++) {
        double n0 = (i == 0) ? this->n0 : x[i - 1];
        double n1 = (i == N - 2) ? this->n2 : x[i + 0];

        // double p0 = p[i];
        // double p1 = p[i + 1];

        double psi0 = psi[i];
        double psi1 = psi[i + 1];

        Jn_all[i] = DDM::fun_Jn(n0, n1, psi0, psi1, dx, params->Vt, params->q, params->mun);
    }
    return Jn_all; 
}
