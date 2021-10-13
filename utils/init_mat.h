#include <random>

auto rng_init_matrix(float *buf, int len, int seed, float min = 0.1, float max = 2.5)
    -> void {
  auto rng  = std::mt19937_64(seed);
  auto dist = std::uniform_real_distribution<float>(min, max);
  for (auto i = 0; i < len; ++i) {
    buf[i] = dist(rng);
    // buf[i] = rand() % 10 + 1;;
  }
}