#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <glog/logging.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

using namespace std;

//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//
//   4 residuals:
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2

// method 1 analytic manual jacobian
class PowellAnalytic : public ceres::SizedCostFunction<4, 4> {
public:
    virtual ~PowellAnalytic() {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        const double x1 = parameters[0][0];
        const double x2 = parameters[0][1];
        const double x3 = parameters[0][2];
        const double x4 = parameters[0][3];
        residuals[0] = x1 + 10 * x2;
        residuals[1] = sqrt(5) * (x3 - x4);
        residuals[2] = (x2 - 2 * x3) * (x2 - 2 * x3);
        residuals[3] = sqrt(10) * (x1 - x4) * (x1 - x4);

        //   4 residuals:
        //   f1 = x1 + 10*x2;
        //   f2 = sqrt(5) * (x3 - x4)
        //   f3 = (x2 - 2*x3)^2
        //   f4 = sqrt(10) * (x1 - x4)^2
        //   Jacobian calculation, J =
        //    [d f1 / d x1         d f1 / d x2         d f1 / d x3         d f1 / d x4]
        //    [d f2 / d x1         d f2 / d x2         d f2 / d x3         d f2 / d x4]
        //    [d f3 / d x1         d f3 / d x2         d f3 / d x3         d f3 / d x4]
        //    [d f4 / d x1         d f4 / d x2         d f4 / d x3         d f4 / d x4]
        //    =
        //    [1                   10                  0                   0                  ]
        //    [0                   0                   sqrt(5)             -sqrt(5)           ]
        //    [0                   2*(x2-2*x3)         -4*(x2-2*x3)        0                  ]
        //    [2*sqrt(10)*(x1-x4)  0                   0                   -2*sqrt(10)*(x1-x4)]

        if (!jacobians) {
            return true;
        }

        /* using Eigen::Map<Matrix> to jacobian which is just slightly slower but maybe more convenient */
        if (jacobians[0] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> jacobian(jacobians[0]);
            jacobian.setZero();

            jacobian(0, 0) = 1;
            jacobian(0, 1) = 10;
            jacobian(0, 2) = 0;
            jacobian(0, 3) = 0;

            jacobian(1, 0) = 0;
            jacobian(1, 1) = 0;
            jacobian(1, 2) = sqrt(5);
            jacobian(1, 3) = -sqrt(5);

            jacobian(2, 0) = 0;
            jacobian(2, 1) = 2 * (x2 - 2 * x3);
            jacobian(2, 2) = -4 * (x2 - 2 * x3);
            jacobian(2, 3) = 0;

            jacobian(3, 0) = 2 * sqrt(10) * (x1 - x4);
            jacobian(3, 1) = 0;
            jacobian(3, 2) = 0;
            jacobian(3, 3) = -2 * sqrt(10) * (x1 - x4);
        }

        /* slightly faster by manually setting the array values */
        // if (jacobians != NULL && jacobians[0] != NULL) {
        //     double *jacobian = jacobians[0];
        //     jacobian[0] = 1;
        //     jacobian[1] = 10;
        //     jacobian[2] = 0;
        //     jacobian[3] = 0;

        //     jacobian[4] = 0;
        //     jacobian[5] = 0;
        //     jacobian[6] = sqrt(5);
        //     jacobian[7] = -sqrt(5);

        //     jacobian[8] = 0;
        //     jacobian[9] = 2 * (x2 - 2 * x3);
        //     jacobian[10] = -4 * (x2 - 2 * x3);
        //     jacobian[11] = 0;

        //     jacobian[12] = 2 * sqrt(10) * (x1 - x4);
        //     jacobian[13] = 0;
        //     jacobian[14] = 0;
        //     jacobian[15] = -2 * sqrt(10) * (x1 - x4);
        // }

        return true;
    }
};

// method 2 automatic derivatives one cost function
struct CostFunctor {
    template <typename T> bool operator()(const T *const x, T *residual) const {
        residual[0] = x[0] + 10.0 * x[1];
        residual[1] = sqrt(5) * (x[2] - x[3]);
        residual[2] = pow((x[1] - 2.0 * x[2]), 2);
        residual[3] = sqrt(10) * pow(x[0] - x[3], 2);

        return true;
    }
};

// method 3 (original) automatic derivatives 4 cost functions
// https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/powell.cc
struct F1 {
    template <typename T> bool operator()(const T *const x1, const T *const x2, T *residual) const {
        // f1 = x1 + 10 * x2;
        residual[0] = x1[0] + 10.0 * x2[0];
        return true;
    }
};
struct F2 {
    template <typename T> bool operator()(const T *const x3, const T *const x4, T *residual) const {
        // f2 = sqrt(5) (x3 - x4)
        residual[0] = sqrt(5.0) * (x3[0] - x4[0]);
        return true;
    }
};
struct F3 {
    template <typename T> bool operator()(const T *const x2, const T *const x3, T *residual) const {
        // f3 = (x2 - 2 x3)^2
        residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]);
        return true;
    }
};
struct F4 {
    template <typename T> bool operator()(const T *const x1, const T *const x4, T *residual) const {
        // f4 = sqrt(10) (x1 - x4)^2
        residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
        return true;
    }
};

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    int n = 1e4;
    // initial values
    double x[4];

    double t_analytic = 0;
    for (size_t i = 0; i < n; i++) {
        Problem problemAnalytic;
        CostFunction *costFunctionAnalytic = new PowellAnalytic;
        x[0] = 3.0;
        x[1] = -1.0;
        x[2] = 0.0;
        x[3] = 1.0;
        problemAnalytic.AddResidualBlock(costFunctionAnalytic, NULL, x);

        Solver::Options optionsAnalytic;
        optionsAnalytic.linear_solver_type = ceres::DENSE_QR;
        // options2.minimizer_progress_to_stdout = true;
        Solver::Summary summaryAnalytic;
        Solve(optionsAnalytic, &problemAnalytic, &summaryAnalytic);

        // std::cout << summary2.BriefReport() << std::endl;
        t_analytic += summaryAnalytic.total_time_in_seconds;
        for (size_t i = 0; i < 4; i++) {
            std::cout << "x" << i + 1 << " -> " << x[i] << std::endl;
        }
        cout << "iter analytic " << i << endl;
    }

    double t_autoOneCostFunction = 0;
    for (size_t i = 0; i < n; i++) {
        x[0] = 3.0;
        x[1] = -1.0;
        x[2] = 0.0;
        x[3] = 1.0;

        Problem problem;
        CostFunction *costFunction = new AutoDiffCostFunction<CostFunctor, 4, 4>(new CostFunctor);
        problem.AddResidualBlock(costFunction, NULL, x);

        Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;

        Solver::Summary summary;
        Solve(options, &problem, &summary);

        t_autoOneCostFunction += summary.total_time_in_seconds;
        // std::cout << summary.BriefReport() << std::endl;
        for (size_t i = 0; i < 4; i++) {
            std::cout << "x" << i + 1 << " -> " << x[i] << std::endl;
        }
        cout << "iter one cost function " << i << endl;
    }

    double t_autoFourCostFunctions = 0;
    for (size_t i = 0; i < n; i++) {
        double x1 = 3.0;
        double x2 = -1.0;
        double x3 = 0.0;
        double x4 = 1.0;

        Problem problemOriginal;
        problemOriginal.AddResidualBlock(new AutoDiffCostFunction<F1, 1, 1, 1>(new F1), NULL, &x1, &x2);
        problemOriginal.AddResidualBlock(new AutoDiffCostFunction<F2, 1, 1, 1>(new F2), NULL, &x3, &x4);
        problemOriginal.AddResidualBlock(new AutoDiffCostFunction<F3, 1, 1, 1>(new F3), NULL, &x2, &x3);
        problemOriginal.AddResidualBlock(new AutoDiffCostFunction<F4, 1, 1, 1>(new F4), NULL, &x1, &x4);
        Solver::Options options;

        options.linear_solver_type = ceres::DENSE_QR;
        // options.minimizer_progress_to_stdout = true;

        Solver::Summary summaryOriginal;
        Solve(options, &problemOriginal, &summaryOriginal);
        // std::cout << summary.FullReport() << "\n";
        double x[4] = {x1, x2, x3, x4};
        for (size_t i = 0; i < 4; i++) {
            std::cout << "x" << i + 1 << " -> " << x[i] << std::endl;
        }

        t_autoFourCostFunctions += summaryOriginal.total_time_in_seconds;
        cout << "iter original " << i << endl;
    }

    cout << "times given in seconds" << endl;
    cout << "total time analytic, manual jacobi matrix     " << t_analytic << " mean " << t_analytic / n << endl;
    cout << "total time auto, one cost function            " << t_autoOneCostFunction << " mean " << t_autoOneCostFunction / n << endl;
    cout << "total time original, 4 single cost functions  " << t_autoFourCostFunctions << " mean " << t_autoFourCostFunctions / n << endl;
    cout << "analytic faster than auto one cost function       by " << t_autoOneCostFunction / t_analytic * 100.0 << "%" << endl;
    cout << "analytic faster than original four cost functions by " << t_autoFourCostFunctions / t_analytic * 100.0 << "%" << endl;

    return 0;
}
