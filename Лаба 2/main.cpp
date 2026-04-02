#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cblas.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace chrono;

// Данные об авторе
const string AUTHOR = "Булаева Алиса Ростиславовна";
const string GROUP = "РПИа-025";

// Генерация случайной матрицы размера n x n (значения от -1.0 до 1.0)
vector<float> generate_matrix(int n) {
    vector<float> mat(n * n);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n * n; ++i)
        mat[i] = dist(gen);
    return mat;
}

// ------------------------------------------------------------
// 1-й вариант: классическое умножение (i, j, k)
// ------------------------------------------------------------
void multiply_classic(const vector<float>& A, const vector<float>& B, vector<float>& C, int n) {
    fill(C.begin(), C.end(), 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// ------------------------------------------------------------
// 3-й вариант: блочное умножение с оптимизацией (i,k,j) и OpenMP
// ------------------------------------------------------------
void multiply_blocked(const vector<float>& A, const vector<float>& B, vector<float>& C,
                      int n, int block_size) {
    fill(C.begin(), C.end(), 0.0f);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i0 = 0; i0 < n; i0 += block_size) {
        for (int j0 = 0; j0 < n; j0 += block_size) {
            for (int k0 = 0; k0 < n; k0 += block_size) {
                int imax = min(i0 + block_size, n);
                int jmax = min(j0 + block_size, n);
                int kmax = min(k0 + block_size, n);
                for (int i = i0; i < imax; ++i) {
                    for (int k = k0; k < kmax; ++k) {
                        float aik = A[i * n + k];
                        for (int j = j0; j < jmax; ++j) {
                            C[i * n + j] += aik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------
// Функция замера времени и расчёта MFlops
// ------------------------------------------------------------
void benchmark(const string& name, const vector<float>& A, const vector<float>& B,
               vector<float>& C, int n, void (*func)(const vector<float>&, const vector<float>&,
                                                      vector<float>&, int, int),
               int extra_param) {
    auto start = high_resolution_clock::now();
    func(A, B, C, n, extra_param);
    auto end = high_resolution_clock::now();
    double t = duration<double>(end - start).count();
    double c = 2.0 * n * n * n;               // 2n^3 операций с плавающей точкой
    double mflops = c / t * 1e-6;
    cout << name << ": время = " << t << " с, MFlops = " << mflops << endl;
}

void benchmark_classic(const string& name, const vector<float>& A, const vector<float>& B,
                       vector<float>& C, int n, void (*func)(const vector<float>&,
                                                              const vector<float>&,
                                                              vector<float>&, int)) {
    auto start = high_resolution_clock::now();
    func(A, B, C, n);
    auto end = high_resolution_clock::now();
    double t = duration<double>(end - start).count();
    double c = 2.0 * n * n * n;
    double mflops = c / t * 1e-6;
    cout << name << ": время = " << t << " с, MFlops = " << mflops << endl;
}

// ------------------------------------------------------------
// Главная функция
// ------------------------------------------------------------
int main() {
    const int n = 2048;
    cout << "Автор: " << AUTHOR << "\nГруппа: " << GROUP << "\n";
    cout << "Размер матриц: " << n << " x " << n << "\n\n";

    // Генерация исходных матриц
    cout << "Генерация матриц A и B...\n";
    vector<float> A = generate_matrix(n);
    vector<float> B = generate_matrix(n);
    vector<float> C(n * n, 0.0f);

    // --------------------------------------------------------
    // 1-й вариант: классический
    // --------------------------------------------------------
    cout << "\n--- 1-й вариант: классическое умножение (i,j,k) ---\n";
    benchmark_classic("Классический", A, B, C, n, multiply_classic);

    // --------------------------------------------------------
    // 2-й вариант: BLAS (cblas_sgemm)
    // --------------------------------------------------------
    cout << "\n--- 2-й вариант: BLAS (cblas_sgemm из Intel MKL / OpenBLAS) ---\n";
    auto start = high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
    auto end = high_resolution_clock::now();
    double t_blas = duration<double>(end - start).count();
    double c = 2.0 * n * n * n;
    double mflops_blas = c / t_blas * 1e-6;
    cout << "BLAS: время = " << t_blas << " с, MFlops = " << mflops_blas << "\n";

    // --------------------------------------------------------
    // 3-й вариант: оптимизированный (блочный + OpenMP)
    // --------------------------------------------------------
    cout << "\n--- 3-й вариант: оптимизированный (блочный, i,k,j, OpenMP) ---\n";
    int block_size = 128;   // подобран экспериментально
    fill(C.begin(), C.end(), 0.0f);
    start = high_resolution_clock::now();
    multiply_blocked(A, B, C, n, block_size);
    end = high_resolution_clock::now();
    double t_opt = duration<double>(end - start).count();
    double mflops_opt = c / t_opt * 1e-6;
    cout << "Оптимизированный: время = " << t_opt << " с, MFlops = " << mflops_opt << "\n";

    // Сравнение с BLAS
    double ratio = mflops_opt / mflops_blas * 100.0;
    cout << "\nПроизводительность оптимизированного варианта: " << ratio << "% от BLAS.\n";
    if (ratio >= 30.0)
        cout << "Условие (>=30%) выполнено.\n";
    else
        cout << "Условие не выполнено. Требуется дополнительная оптимизация.\n";

    return 0;
}