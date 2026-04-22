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

const string AUTHOR = "Булаева Алиса Ростиславовна";
const string GROUP = "РПИа-025";

// Генерация матрицы

vector<float> generate_matrix(int n) {
    vector<float> mat(n * n);
    mt19937 gen(42); // фиксированное зерно
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < n * n; ++i)
        mat[i] = dist(gen);

    return mat;
}

// 1. Классический алгоритм

void multiply_classic(const vector<float>& A, const vector<float>& B, vector<float>& C, int n) {
    fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float aik = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}


// 3. Блочный + OpenMP

void multiply_blocked(const vector<float>& A, const vector<float>& B,
                      vector<float>& C, int n, int block_size) {

    fill(C.begin(), C.end(), 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < n; i0 += block_size) {
        for (int k0 = 0; k0 < n; k0 += block_size) {
            for (int j0 = 0; j0 < n; j0 += block_size) {

                int imax = min(i0 + block_size, n);
                int kmax = min(k0 + block_size, n);
                int jmax = min(j0 + block_size, n);

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

// Проверка корректности

float max_error(const vector<float>& A, const vector<float>& B) {
    float err = 0.0f;
    for (size_t i = 0; i < A.size(); ++i) {
        err = max(err, fabs(A[i] - B[i]));
    }
    return err;
}


// Benchmark

double benchmark_classic(const vector<float>& A, const vector<float>& B,
                         vector<float>& C, int n) {
    auto start = high_resolution_clock::now();
    multiply_classic(A, B, C, n);
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

double benchmark_blocked(const vector<float>& A, const vector<float>& B,
                         vector<float>& C, int n, int block) {
    auto start = high_resolution_clock::now();
    multiply_blocked(A, B, C, n, block);
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}


int main() {
    const int n = 2048;
    const int block_size = 128;

    cout << "Автор: " << AUTHOR << "\nГруппа: " << GROUP << "\n";
    cout << "Размер: " << n << "x" << n << "\n\n";

    vector<float> A = generate_matrix(n);
    vector<float> B = generate_matrix(n);

    vector<float> C1(n*n), C2(n*n), C3(n*n);

    double c = 2.0 * n * n * n;

    // 1. Classic
    double t1 = benchmark_classic(A, B, C1, n);
    double p1 = c / t1 * 1e-6;
    cout << "Classic: " << t1 << " s, " << p1 << " MFLOPS\n";

    // 2. BLAS
    auto start = high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0f,
                A.data(), n,
                B.data(), n,
                0.0f,
                C2.data(), n);
    auto end = high_resolution_clock::now();

    double t2 = duration<double>(end - start).count();
    double p2 = c / t2 * 1e-6;
    cout << "BLAS: " << t2 << " s, " << p2 << " MFLOPS\n";

    // 3. Optimized
    double t3 = benchmark_blocked(A, B, C3, n, block_size);
    double p3 = c / t3 * 1e-6;
    cout << "Optimized: " << t3 << " s, " << p3 << " MFLOPS\n";

    // Проверка
    cout << "\nОшибка Classic vs BLAS: " << max_error(C1, C2) << endl;
    cout << "Ошибка Optimized vs BLAS: " << max_error(C3, C2) << endl;

    // Сравнение
    cout << "\nOptimized = " << (p3 / p2 * 100.0) << "% от BLAS\n";

    return 0;
}
