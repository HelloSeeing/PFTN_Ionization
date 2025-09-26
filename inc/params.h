#include <string>

template<typename T>
class Params{
    public:
        //// Basis units
        // length
        const T cm = 1e6; 

        // 
        const T s = 1e12; 

        // voltage
        const T V = 1.0;

        // charge
        const T C = 1.0 / 1.602176462e-19;

        // temperature
        const T K = 1.0 / 300; 


        //// Derived units
        // meter
        const T m = 1e2 * cm;

        // micrometer
        const T um = 1e-4 * cm;

        // Ampere
        const T A = C / s;

        // II
        const T II = 1.0 / cm;

        // temperature
        const T Temp = 300 * K;

        // Boltzmann constant
        const T kb = 1.3806503e-23 * C * V / K;

        // electronic charge
        const T e = 1.602176462e-19 * C;

        // electron voltage
        const T eV = e * V;

        // permittivity of free space
        const T eps0 = 8.854187818e-12 * C * V / m; 

        // concentration
        const T conc = std::pow(cm, -3);

        // intrinsic carrier density of silicon at 300K
        const T nie = 10784781693.00264 * conc; 

        // q
        const T q = 1.6021766208e-19 * C; 

        // thermal voltage at 300K
        const T Vt = kb * Temp / e;

        // mobility of electron
        const T mun = 1417 * cm * cm / V / s; 

        // mobility of hole
        const T mup = 470.5 * cm * cm / V / s; 

        // SRH lifetime of electron
        const T tn = 909.09090909090912 / 1e12 * s; 

        // SRH lifetime of hole
        const T tp = 909.09090909090912 / 1e12 * s;

        Params() {};

        ~Params() {};
}; 
