#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>

void fill_random(std::vector<float>& M, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : M) {
        x = dist(rng);
    }
}

CompareResult compare_matrices(const float* ref, const float* test, int N,
                               float atol, float rtol) {
    CompareResult result{true, 0, -1, 0.0f};
    const std::size_t count = static_cast<std::size_t>(N) * N;

    for (std::size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(test[i] - ref[i]);
        if (diff > result.max_abs_error) {
            result.max_abs_error = diff;
        }
        // Written as !(diff <= limit) so that NaN in the test output counts as a failure.
        if (!(diff <= atol + rtol * std::fabs(ref[i]))) {
            if (result.first_bad_index < 0) {
                result.first_bad_index = static_cast<int>(i);
            }
            ++result.mismatches;
            result.passed = false;
        }
    }
    return result;
}

TimingStats summarize_timings(std::vector<double> timings_ms) {
    if (timings_ms.empty()) {
        return {0.0, 0.0, 0.0, 0};
    }

    std::sort(timings_ms.begin(), timings_ms.end());
    const std::size_t count = timings_ms.size();
    const double median = (count % 2 == 0)
                              ? (timings_ms[count / 2 - 1] + timings_ms[count / 2]) / 2.0
                              : timings_ms[count / 2];
    const double mean =
        std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) / count;

    return {timings_ms.front(), median, mean, static_cast<int>(count)};
}
